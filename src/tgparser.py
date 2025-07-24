import os
import json

from typing import List
from threading import Thread as th

from loguru import logger
from pyrogram.client import Client
from pyrogram import filters, errors
from pyrogram.sync import idle
from pyrogram.types import Message
from src.text import textHandler
from src.order_handler import set_order, updater
from src.command_handler import handle_command


if os.getenv("CONTAINER") != "YES":
    # Importing tableviewer only if not in a container environment
    from src.tableviewer import startTable

try:
    from config import api_id, api_hash
except ImportError:
    logger.error("API_ID and API_HASH are not set in config.py. Please define them.")
    exit(1)

if not api_id or not api_hash:
    logger.error("API_ID and API_HASH are not set in config.py. Exiting.")
    exit()

app = Client("my_account", api_id=api_id, api_hash=api_hash)

sim: bool = False


def get_chats() -> List[int | str]:
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
        return [int(channel) for channel in channels]
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Could not load or parse chats from channels.json: {e}")
        exit(1)


chats = get_chats()
if chats:
    logger.success(f"Imported {len(chats)} chats")
else:
    logger.warning("No chats were loaded. The bot will not listen to any channels.")


@app.on_message(filters.chat(chats) or filters.me)
async def channel_parser(client: Client, message: Message):
    chat_id_str = str(message.chat.id)
    chat_title = message.chat.title or "Unknown Chat"
    logger.info(f"{chat_title}({chat_id_str}) Got message")

    text = message.text or message.caption
    if not text:
        logger.warning(f"{chat_title}({chat_id_str}) No text in message.")
        return

    # --- Command Handling ---
    if text.startswith("."):
        command = text.lower()
        await handle_command(command, client, message, chats, sim)
        return

    # --- Signal Processing ---
    res = textHandler(text, chat_id_str)
    if not res:
        return

    coin, method, log_msg = res
    logger.success(f"{chat_title}({chat_id_str}) {log_msg}")

    try:
        with open("data/curdata.json", "r+", encoding="utf-8") as f:
            data = json.load(f)
            if coin not in data.get("orders", {}).get(chat_id_str, {}):
                new_data = set_order(chat_id_str, coin, method, sim)
                if new_data:
                    f.seek(0)
                    json.dump(new_data, f, indent=4)
                    f.truncate()
                    logger.success(
                        f"({chat_id_str}) Successfully set new order for {coin}."
                    )
            else:
                logger.warning(f"({chat_id_str}) Coin {coin} already exists in orders.")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.error(f"Error processing order for {coin}: {e}")


async def check_chats():
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not read channels.json: {e}")
        return

    for chat_id in chats.copy():
        if str(chat_id) not in channels:
            logger.warning(
                f"Chat ID {chat_id} not found in channels.json. Removing from list."
            )
            chats.remove(chat_id)
            continue

        try:
            await app.get_chat(chat_id)
        except errors.ChannelInvalid as e:
            logger.error(f"Chat ID {chat_id} is invalid: {e}")
            chats.remove(chat_id)
        except Exception as e:
            logger.error(f"Error checking chat {chat_id}: {e}")


async def app_suc():
    try:
        await app.start()
        me = await app.get_me()
        logger.success(f"Telegram parser started as {me.first_name}")
        await check_chats()
        logger.success(
            f"Checked {len(chats)} chats for validity, {len(chats)} valid chats found."
        )
        if not chats:
            logger.error("No valid chats found. Exiting.")
            exit(1)
        else:
            # list valid chats
            logger.info(f"Valid chats: {', '.join(str(chat) for chat in chats)}")
        await idle()
    except Exception as e:
        logger.critical(f"A critical error occurred in the main Telegram loop: {e}")
    finally:
        if app.is_connected:
            await app.stop()


def start_parsing(is_simulating):
    global sim
    sim = bool(is_simulating)
    th(target=updater, daemon=True).start()
    # stop the table viewer because of docker issue with qt6
    if os.getenv("CONTAINER") != "YES":
        # Start the table viewer only if not in a container environment
        logger.success("Starting Table View")
        th(target=startTable, daemon=True).start()  # type: ignore
    else:
        logger.warning("Table View is disabled in container mode due to qt6 issues.")
    app.run(app_suc())
