"""
This module handles the Telegram client, message listeners, and routing signals
and commands to their respective handlers.
"""

import sys
import os
import json
from threading import Thread
from typing import List, Dict, Optional
from loguru import logger
from pyrogram import filters, errors
from pyrogram.sync import idle
from pyrogram.client import Client
from pyrogram.types import Message
from dotenv import load_dotenv

from .text_parser import parse_message_for_signal, ai_parse_message_for_signal
from .order_handler import place_order, updater_thread_worker
from .command_handler import handle_command

# Load environment variables
load_dotenv()

# --- Configuration from Environment Variables ---
try:
    API_ID = int(os.getenv("API_ID"))
    API_HASH = os.getenv("API_HASH")

    if not API_ID or not API_HASH:
        raise ValueError("API_ID and API_HASH must be set in environment variables")

except (ValueError, TypeError) as e:
    logger.critical(f"Configuration error: {e}. Please check your .env file.")
    sys.exit(1)

app = Client("my_account", api_id=API_ID, api_hash=API_HASH)
IS_SIMULATION = False
CHANNELS_CONFIG: Dict = {}


def _load_channel_ids() -> Optional[List[int]]:
    import sqlite3

    db_path = os.getenv("DB_PATH", "total.db")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT channel_id FROM channels ORDER BY channel_id")
            rows = cursor.fetchall()
            CHAT_IDS = [
                int(
                    str(
                        row["channel_id"]
                        if str(row["channel_id"])[0] == "-"
                        else "-100" + str(row["channel_id"])
                    )
                )
                for row in rows
            ]
            logger.info(f"Loaded {len(CHAT_IDS)} channels from database.")
            return CHAT_IDS
    except Exception as e:
        logger.error(f"Failed to load channels from database: {e}")
        logger.info("Falling back to JSON configuration...")
        return None


CHAT_IDS = _load_channel_ids()
if CHAT_IDS is None:
    logger.critical("Failed to load channel IDs.")
    sys.exit(1)
logger.info(f"Loaded {len(CHAT_IDS)} channels from configuration.")


# --- Message Handlers ---
@app.on_message(filters.chat(CHAT_IDS))  # type: ignore
async def message_handler(client: Client, message: Message):
    """
    Primary message handler that listens to configured channels and private messages.
    """
    try:
        chat_id = message.chat.id
        text = message.text or message.caption
        if not text:
            return  # Ignore messages with no text content

        # --- Command Handling (only from 'me' chat) ---
        if message.from_user and message.from_user.is_self and text.startswith("."):
            command = text.lower().strip()
            await handle_command(command, client, message, CHAT_IDS, IS_SIMULATION)
            return

        # --- Signal Processing ---
        # TODO: do regex as alternative to AI parsing
        # signal = parse_message_for_signal(text, chat_id)
        signal = ai_parse_message_for_signal(text)
        if not signal:
            return

        # TODO: Prevent placing duplicate orders

        # Place the order
        success = place_order(chat_id, signal, IS_SIMULATION)
        if success:
            logger.info(
                f"Successfully processed order for {signal.get('pair')} {signal.get('direction')}."
            )
        else:
            logger.error(
                f"Failed to process order for {signal.get('pair')} {signal.get('direction')}."
            )

    except Exception as e:
        logger.error(f"Error in message handler: {type(e).__name__}: {e}")
        logger.debug(f"Full traceback: {e}", exc_info=True)


# --- Application Startup ---
async def main_telegram_loop():
    """Starts the Telegram client and keeps it running."""
    global CHAT_IDS
    try:
        await app.start()
        me = await app.get_me()
        logger.success(
            f"Telegram client started successfully for user: {me.first_name}"
        )

        CHAT_IDS = _load_channel_ids()
        if not CHAT_IDS:
            logger.critical("No channels configured to listen to. Exiting.")
            sys.exit(1)

        # Verify that the bot can access the configured channels
        valid_chats = []
        logger.info(f"Validating access to {len(CHAT_IDS)} configured channels...")
        for chat_id in CHAT_IDS:
            try:
                chat = await app.get_chat(chat_id)
                valid_chats.append(chat_id)
                logger.success(f"✓ Chat {chat_id} ({chat.title}) is accessible")
            except errors.ChannelInvalid:
                logger.error(f"✗ Chat {chat_id} is invalid or inaccessible")
            except errors.PeerIdInvalid:
                logger.error(f"✗ Chat {chat_id} has invalid peer ID")
            except Exception as e:
                logger.error(
                    f"✗ Could not access chat {chat_id}: {type(e).__name__}: {e}"
                )
                valid_chats.append(chat_id)
        CHAT_IDS = valid_chats

        if not CHAT_IDS:
            logger.critical(
                "No valid, accessible chats found. The bot has nothing to listen to. Exiting."
            )
            sys.exit(1)

        logger.info(f"Listening for signals on {len(CHAT_IDS)} valid channels.")
        await idle()

    except Exception as e:
        logger.critical(f"A critical error occurred in the main Telegram loop: {e}")
    finally:
        await app.stop()
        logger.warning("Telegram client stopped.")


def start_telegram_parser(is_simulation: bool):
    """
    Initializes and starts all components of the bot.
    """
    global IS_SIMULATION
    IS_SIMULATION = is_simulation
    mode = "Simulation" if is_simulation else "Live Trading"
    logger.info(f"Starting bot in {mode} mode.")

    # Start the background thread for updating orders and GUI data
    Thread(target=updater_thread_worker, daemon=True).start()
    logger.info("Started background updater thread.")

    # Run the main async Telegram client loop
    app.run(main_telegram_loop())
