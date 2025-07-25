"""
This module handles the Telegram client, message listeners, and routing signals
and commands to their respective handlers.
"""

import os
import sys
import json
from threading import Thread
from typing import List, Dict
from loguru import logger
from pyrogram import filters, errors
from pyrogram.sync import idle
from pyrogram.client import Client
from pyrogram.types import Message

from src.text_parser import parse_message_for_signal
from src.order_handler import place_order, updater_thread_worker
from src.command_handler import handle_command

# Import GUI starter only if not in a container
if os.getenv("CONTAINER") != "YES":
    from src.tableviewer import start_gui

# --- Configuration and Client Setup ---
try:
    from config import API_ID, API_HASH
except ImportError:
    logger.critical(
        "Could not import `config.py`. Please rename `config.example.py` to `config.py` and fill it out."
    )
    sys.exit(1)

if not API_ID or not API_HASH:
    logger.critical("API_ID and API_HASH are not configured in `config.py`.")
    sys.exit(1)

app = Client("my_account", api_id=API_ID, api_hash=API_HASH)
IS_SIMULATION = False
CHANNELS_CONFIG: Dict = {}


def _load_channel_ids() -> List[int]:
    """Loads channel IDs from the JSON config file."""
    global CHANNELS_CONFIG
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            CHANNELS_CONFIG = json.load(f)
        return [int(channel_id) for channel_id in CHANNELS_CONFIG]
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.critical(
            f"Could not load or parse `channels.json`: {e}. Please ensure it is valid."
        )
        sys.exit(1)


CHAT_IDS = _load_channel_ids()
logger.info(f"Loaded {len(CHAT_IDS)} channels from configuration.")


# --- Message Handlers ---
@app.on_message(filters.chat(CHAT_IDS))  # type: ignore
async def message_handler(client: Client, message: Message):
    """
    Primary message handler that listens to configured channels and private messages.
    """
    chat_id_str = str(message.chat.id)
    text = message.text or message.caption
    if not text:
        return  # Ignore messages with no text content

    # --- Command Handling (only from 'me' chat) ---
    if message.from_user and message.from_user.is_self and text.startswith("."):
        command = text.lower().strip()
        await handle_command(command, client, message, CHAT_IDS, IS_SIMULATION)
        return

    # --- Signal Processing ---
    signal = parse_message_for_signal(text, chat_id_str, CHANNELS_CONFIG)
    if not signal:
        return

    coin, side, log_message = signal
    logger.success(log_message)

    # Prevent placing duplicate orders
    try:
        with open("data/state.json", "r") as f:
            state = json.load(f)
        if coin in state.get("orders", {}).get(chat_id_str, {}):
            logger.warning(
                f"Order for {coin} from {chat_id_str} already exists. Skipping."
            )
            return
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # File may not exist yet, proceed

    # Place the order
    success = place_order(chat_id_str, coin, side, IS_SIMULATION)
    if success:
        logger.info(f"Successfully processed order for {coin} {side}.")
    else:
        logger.error(f"Failed to process order for {coin} {side}.")


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

        # Verify that the bot can access the configured channels
        valid_chats = []
        print(CHAT_IDS)
        for chat_id in CHAT_IDS:
            try:
                await app.get_chat(chat_id)
                valid_chats.append(chat_id)
            except errors.ChannelInvalid:
                logger.error(f"Chat {chat_id} is invalid.")
            except Exception as e:
                logger.error(f"Could not access chat {chat_id}: {e}")
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

    # Start the GUI in a separate thread if not in a container
    if os.getenv("CONTAINER") != "YES":
        Thread(target=start_gui, daemon=True).start()  # type: ignore
        logger.info("Starting GUI in a separate thread.")
    else:
        logger.warning("GUI is disabled in container mode.")

    # Run the main async Telegram client loop
    app.run(main_telegram_loop())
