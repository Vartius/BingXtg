import os
import json
import random

import dataframe_image as dfi
import pandas as pd
from loguru import logger
from pyrogram.client import Client
from pyrogram import errors
from pyrogram.types import Message

from src.order_handler import set_order


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


async def handle_chats_check(client: Client, message: Message, chats: list):
    chat_id_str = str(message.chat.id)
    titles = []
    for chat_id in chats:
        try:
            chat = await client.get_chat(chat_id)
            titles.append(chat.title or f"Chat {chat_id}")
        except errors.exceptions as e:
            logger.error(f"Could not get info for chat {chat_id}: {e}")
            titles.append(f"Error fetching chat {chat_id}")
    await message.reply("\n".join(titles))
    logger.success(f"({chat_id_str}) Chats check completed.")


async def handle_chats(message: Message):
    chat_id_str = str(message.chat.id)
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
        response = "\n".join(
            f"{ch_data.get('name', 'N/A')} {ch_id}"
            for ch_id, ch_data in channels.items()
        )
        await message.reply(response)
        logger.success(f"({chat_id_str}) .chats command executed.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        await message.reply("Error reading channels file.")
        logger.error(f"({chat_id_str}) Failed to execute .chats: {e}")


async def handle_stop(message: Message):
    chat_id_str = str(message.chat.id)
    logger.info(f"({chat_id_str}) stop command received. Shutting down.")
    os._exit(0)


async def handle_add_test_orders(message: Message, sim: bool):
    chat_id_str = str(message.chat.id)
    for coin in ["CRV", "UNI", "BTC", "ETH", "XRP", "STORJ", "AAVE"]:
        data = set_order(chat_id_str, coin, random.choice(["long", "short"]), sim)
        if data:
            try:
                with open("data/curdata.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            except IOError as e:
                logger.error(f"Failed to write test order data: {e}")
    logger.success(f"({chat_id_str}) Added test orders.")


async def handle_get_data(client: Client, message: Message):
    chat_id_str = str(message.chat.id)
    logger.info(f"({chat_id_str}) Sending all data.")
    try:
        with open("data/table.json", "r", encoding="utf-8") as f:
            table = json.load(f)

        headers = [
            "Channel",
            "Coin",
            "Diraction",
            "Deposit*L",
            "Order Price",
            "Current Price",
            "Profit",
            "Procent",
        ]
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
        dfi.export(df_styled, image_path, max_rows=-1)  # type: ignore
        await client.send_photo(message.chat.id, image_path)
        os.remove(image_path)
    except FileNotFoundError:
        await message.reply("`table.json` not found. No data to display.")
    except Exception as e:
        logger.error(f"Failed to generate or send data image: {e}")
        await message.reply("An error occurred while generating the data image.")


async def handle_command(
    command: str, client: Client, message: Message, chats: list, sim: bool
):
    if command == ".chatscheck":
        await handle_chats_check(client, message, chats)
    elif command == ".chats":
        await handle_chats(message)
    elif command == ".stop":
        await handle_stop(message)
    elif command == ".addtestorders":
        await handle_add_test_orders(message, sim)
    elif command == ".getdata":
        await handle_get_data(client, message)
