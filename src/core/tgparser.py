import os
import json
import random

from typing import List
from threading import Thread as th

import dataframe_image as dfi
import pandas as pd

from loguru import logger
from pyrogram.client import Client
from pyrogram import filters, errors
from pyrogram.sync import idle
from pyrogram.types import Message
from src.core.text import textHandler
from src.core.bingx import set_order, updater
from src.core.tableviewer import startTable

from config import api_id, api_hash

if not api_id or not api_hash:
    logger.error("API_ID and API_HASH are not set in config.py. Exiting.")
    exit()

app = Client("my_account", api_id=api_id, api_hash=api_hash)

def highlight_x(s):
    return [
        "background-color: #1b1b1b; color: #ffffff;"
        if k % 2 == 0
        else "background-color: #353535; color: #ffffff;"
        for k in range(len(s))
    ]


def highlight_headers(s):
    return ["background-color: #474747; color: #ffffff;" for k in range(len(s))]


def hide_na(val):
    return "" if pd.isna(val) else val


def short_long(val):
    if val == "long":
        return "background-color: #00FF7F; color: #ffffff;"
    elif val == "short":
        return "background-color: #DC143C; color: #ffffff;"
    return ""

def get_chats() -> List[int | str]:
    try:
        with open("src/data/channels.json", "r", encoding="utf-8") as f:
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

@app.on_message(filters.chat(chats))
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
        if command == ".chatscheck":
            titles = []
            for chat_id in chats:
                try:
                    chat = await client.get_chat(chat_id)
                    titles.append(chat.title or f"Chat {chat_id}")
                except errors as e:
                    logger.error(f"Could not get info for chat {chat_id}: {e}")
                    titles.append(f"Error fetching chat {chat_id}")
            await message.reply("\n".join(titles))
            logger.success(f"({chat_id_str}) Chats check completed.")
        
        elif command == ".chats":
            try:
                with open("src/data/channels.json", "r", encoding="utf-8") as f:
                    channels = json.load(f)
                response = "\n".join(f"{ch_data.get('name', 'N/A')} {ch_id}" for ch_id, ch_data in channels.items())
                await message.reply(response)
                logger.success(f"({chat_id_str}) .chats command executed.")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                await message.reply("Error reading channels file.")
                logger.error(f"({chat_id_str}) Failed to execute .chats: {e}")

        elif command == ".stop":
            logger.info(f"({chat_id_str}) stop command received. Shutting down.")
            os._exit(0)

        elif command == ".addtestorders":
            for coin in ["CRV", "UNI", "BTC", "ETH", "XRP", "STORJ", "AAVE"]:
                data = set_order(chat_id_str, coin, random.choice(["long", "short"]), sim)
                if data:
                    try:
                        with open("src/data/curdata.json", "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4)
                    except IOError as e:
                        logger.error(f"Failed to write test order data: {e}")
            logger.success(f"({chat_id_str}) Added test orders.")
            
        elif command == ".getdata":
            logger.info(f"({chat_id_str}) Sending all data.")
            try:
                with open("src/data/table.json", "r", encoding="utf-8") as f:
                    table = json.load(f)
                
                headers = [ "Channel", "Coin", "Diraction", "Deposit*L", "Order Price", "Current Price", "Profit", "Procent" ]
                df = pd.DataFrame(table.get("data", []), columns=headers)

                # Styling
                df_styled = df.style.set_table_styles(
            [
                {
                    "selector": ".blank",
                    "props": [("background-color", "#353535"), ("color", "white")],
                }
            ]
        )
                df_styled = df_styled.apply(highlight_x, axis=0)
                df_styled = df_styled.apply_index(highlight_x, axis=0)
                df_styled = df_styled.apply_index(highlight_headers, axis=1)
                df_styled = df_styled.background_gradient(
                    subset=["Profit", "Procent"], cmap="RdYlGn"
                )
                df_styled = df_styled.format(hide_na)
                df_styled = df_styled.map(short_long)

                image_path = "table.png"
                dfi.export(df_styled, image_path, max_rows=-1) # type: ignore
                await client.send_photo(message.chat.id, image_path)
                os.remove(image_path)
            except FileNotFoundError:
                await message.reply("`table.json` not found. No data to display.")
            except Exception as e:
                logger.error(f"Failed to generate or send data image: {e}")
                await message.reply("An error occurred while generating the data image.")
        return

    # --- Signal Processing ---
    res = textHandler(text, chat_id_str)
    if not res:
        return

    coin, method, log_msg = res
    logger.success(f"{chat_title}({chat_id_str}) {log_msg}")

    try:
        with open("src/data/curdata.json", "r+", encoding="utf-8") as f:
            data = json.load(f)
            if coin not in data.get("orders", {}).get(chat_id_str, {}):
                new_data = set_order(chat_id_str, coin, method)
                if new_data:
                    f.seek(0)
                    json.dump(new_data, f, indent=4)
                    f.truncate()
                    logger.success(f"({chat_id_str}) Successfully set new order for {coin}.")
            else:
                logger.warning(f"({chat_id_str}) Coin {coin} already exists in orders.")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.error(f"Error processing order for {coin}: {e}")



async def app_suc():
    try:
        await app.start()
        me = await app.get_me()
        logger.success(f"Telegram parser started as {me.first_name}")
        await idle()
    except Exception as e:
        logger.critical(f"A critical error occurred in the main Telegram loop: {e}")
    finally:
        if app.is_connected:
            await app.stop()


def start_parsing(is_simulating):
    global sim
    sim = is_simulating
    th(target=updater, daemon=True).start()
    th(target=startTable, daemon=True).start()
    app.run(app_suc())
