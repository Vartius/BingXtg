import os
import json
import time
import random

from typing import List
from threading import Thread as th

import dataframe_image as dfi
import pandas as pd
import numpy as np

from loguru import logger
from pyrogram.client import Client
from pyrogram import filters
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
        return []


chats = get_chats()
logger.success(f"Imported {len(chats)} chats")


@app.on_message(filters.chat(chats))
async def channel_parser(user: Client, message: Message):
    logger.info(f"{message.chat.title}({message.chat.id}) Got message")
    method = ""
    coin = ""
    text = message.caption
    if text is None:
        text = message.text
    if text is None:
        logger.warning(f"{message.chat.title}({message.chat.id}) No text")
        return

    if text == ".chatsCheck":
        titles = []
        for i in chats:
            try:
                title = await app.get_chat(int(i))
                titles.append(title.title)
            except Exception as e:
                logger.error(e)
        await message.reply("\n".join(i for i in titles))
        logger.success(
            f"{message.chat.title}({message.chat.id}) chats checking (slow mode) success"
        )
        return
    elif text == ".chats":
        with open("src/data/channels.json", encoding="utf-8") as f:
            channels = json.load(f)
        await message.reply("\n".join(f"{channels[i]['name']} {i}" for i in channels))
        await message.reply(" ".join(str(i) for i in chats))
        logger.success(
            f"{message.chat.title}({message.chat.id}) chats checking success"
        )
        return
    elif text == ".stop":
        logger.info(f"{message.chat.title}({message.chat.id}) stop")
        os._exit(0)
    elif text == ".addTestOrders":
        for coin in ["CRV", "UNI", "BTC", "ETH", "XRP", "STORJ", "AAVE"]:
            data = set_order(
                str(message.chat.id), coin, random.choice(["long", "short"]), sim
            )
            with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        return
    elif text == ".getData":
        logger.info(f"{message.chat.title}({message.chat.id}) sending all data")
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
        with open("src/data/table.json", encoding="utf-8") as f:
            table = json.load(f)
        new_data = [i for i in table["data"]]
        table_data = {}
        for i in headers:
            table_data[i] = []
        for row in new_data:
            t = list(row)
            while len(t) != 8:
                t.append(np.nan)
            for k, i in enumerate(headers):
                table_data[i].append(t[k])
        df = pd.DataFrame(table_data)
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
        dfi.export(df_styled, "table.png", max_rows=-1, max_cols=-1) # type: ignore
        await app.send_photo(message.chat.id, "table.png")
        return

    res = textHandler(text, str(message.chat.id))
    if res is not None:
        coin = res[0]
        method = res[1]
        logger.success(f"{message.chat.title}({message.chat.id}) {res[2]}")
    else:
        return
    if method != "" and coin != "":
        with open("src/data/curdata.json", encoding="utf-8") as f:
            coins = json.load(f)["orders"][str(message.chat.id)].keys()
        if coin not in coins:
            data = set_order(str(message.chat.id), coin, method)
            with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            time.sleep(0.6)
            with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.success(
                f"{message.chat.title}({message.chat.id}) Setting new order successfully"
            )
        else:
            logger.warning(
                f"{message.chat.title}({message.chat.id}) this coin already exist in data"
            )


async def app_suc():
    await app.start()
    while 1:
        try:
            await app.get_me()
            break
        except Exception as e:
            logger.error(e)
            logger.info("Sleeping for 5 sec")
            time.sleep(5)
            continue
    logger.success("Telegram parser started")
    await idle()
    await app.stop()


def start_parsing(is_simulating):
    global sim
    sim = is_simulating
    th(target=updater).start()
    th(target=startTable).start()
    app.run(app_suc())
