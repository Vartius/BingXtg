"""
This module handles the Telegram client, message listeners, and routing signals
and commands to their respective handlers.
"""

import sys
import os
from threading import Thread
from typing import List, Dict, Optional
from loguru import logger
from telethon import TelegramClient, events
from telethon.errors import ChannelInvalidError, PeerIdInvalidError
from dotenv import load_dotenv

from .text_parser import ai_parse_message_for_signal
from .order_handler import place_order, updater_thread_worker
from .command_handler import handle_command

# Load environment variables
load_dotenv()

# --- Configuration from Environment Variables ---
try:
    API_ID = int(os.getenv("API_ID") or 0)
    API_HASH = os.getenv("API_HASH")

    if not API_ID or not API_HASH:
        raise ValueError("API_ID and API_HASH must be set in environment variables")

except (ValueError, TypeError) as e:
    logger.critical(f"Configuration error: {e}. Please check your .env file.")
    sys.exit(1)

app = TelegramClient("my_account", api_id=API_ID, api_hash=API_HASH)
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
@app.on(events.NewMessage())
async def message_handler(event):
    """
    Primary message handler that listens to configured channels and private messages.
    """
    try:
        chat_id = event.chat_id
        text = event.message.text
        if not text:
            return  # Ignore messages with no text content

        # Only process messages from configured chats or private messages
        if chat_id not in CHAT_IDS and not event.is_private:
            return

        # --- Command Handling (only from 'me' chat) ---
        if event.is_private and text.startswith("."):
            command = text.lower().strip()
            if CHAT_IDS is None:
                logger.error(
                    "No channels configured to listen to. Cannot process commands."
                )
                return
            await handle_command(command, app, event, CHAT_IDS, IS_SIMULATION)
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
def main_telegram_loop():
    """Starts the Telegram client and keeps it running."""
    global CHAT_IDS
    try:
        app.start()
        me = app.get_me()
        logger.success(
            f"Telegram client started successfully for user: {getattr(me, 'first_name', 'Unknown')}"
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
                entity = app.get_entity(chat_id)
                valid_chats.append(chat_id)
                logger.success(
                    f"✓ Chat {chat_id} ({getattr(entity, 'title', chat_id)}) is accessible"
                )
            except ChannelInvalidError:
                logger.error(f"✗ Chat {chat_id} is invalid or inaccessible")
            except PeerIdInvalidError:
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
        app.run_until_disconnected()

    except Exception as e:
        logger.critical(f"A critical error occurred in the main Telegram loop: {e}")
    finally:
        app.disconnect()
        logger.warning("Telegram client stopped.")


def start_telegram_parser(is_simulation: bool):
    """
    Initializes and starts all components of the bot.
    """
    import sqlite3

    global IS_SIMULATION
    IS_SIMULATION = is_simulation
    mode = "Simulation" if is_simulation else "Live Trading"
    logger.info(f"Starting bot in {mode} mode.")

    # Store simulation mode state in database for dashboard access
    db_path = os.getenv("DB_PATH", "total.db")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Create app_state table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Store simulation mode state
            simulation_value = "true" if is_simulation else "false"
            conn.execute(
                """
                INSERT OR REPLACE INTO app_state (key, value) 
                VALUES ('simulation_mode', ?)
            """,
                (simulation_value,),
            )

            logger.info(f"Set simulation_mode in database to: {simulation_value}")
    except Exception as e:
        logger.error(f"Failed to store simulation mode in database: {e}")

    # Start the background thread for updating orders and GUI data
    Thread(target=updater_thread_worker, daemon=True).start()
    logger.info("Started background updater thread.")

    # Run the main Telegram client loop
    try:
        main_telegram_loop()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down gracefully.")
