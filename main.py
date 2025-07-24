import json
import os

from loguru import logger

from src.tgparser import start_parsing
from src.data_handler import (
    get_curdata,
    save_curdata,
    get_winrate,
    save_winrate,
    get_channels,
)

try:
    from config import START_BALANCE
except ImportError:
    logger.error("START_BALANCE is not defined in config.py. Please set it.")
    exit(1)


def renew_data(data):
    data = {
        "balance": START_BALANCE,
        "available_balance": START_BALANCE,
        "winrate": 0,
        "orders": {},
    }
    return data


def continue_data(data):
    data["available_balance"] = data["balance"]
    data["orders"] = {}
    with open("config.py", "r", encoding="utf-8") as f:
        conf = f.read().replace(str(START_BALANCE), str(data["balance"]))
    with open("config.py", "w+", encoding="utf-8") as f:
        f.write(conf)
    return data


def simulate(sim):
    data = get_curdata()
    if data is None:
        logger.error("curdata.json not found or invalid, creating a new one")
        with open("data/curdata.json", "w+", encoding="utf-8") as f:
            json.dump({}, f, indent=4)
        data = {}

    channels = get_channels()
    if channels is None:
        logger.critical(
            "channels.json not found or invalid, please create it, example in data/channels.json.example"
        )
        assert False, "channels.json not found or invalid"

    winrate = get_winrate()
    if winrate is None:
        logger.error("winrate.json not found or invalid, creating a new one")
        with open("data/winrate.json", "w+", encoding="utf-8") as f:
            json.dump({}, f, indent=4)
        winrate = {}

    if "balance" in data:
        CHOICE = input("Found previous data, do you want to renew it? Y/N: ")
    else:
        CHOICE = "y"

    if CHOICE.lower() == "y":
        data = renew_data(data)
    else:
        data = continue_data(data)

    for i in channels:
        if i not in winrate:
            winrate[i] = {"name": channels[i]["name"], "win": 0, "lose": 0}
        if "orders" not in data:
            data["orders"] = {}
        data["orders"][i] = {}

    save_curdata(data)
    save_winrate(winrate)

    logger.info("files checked")
    start_parsing(sim)


logger.add("logs.log")


print("Menu\n\n1: Start trading\n2: Start simulating\n0/(any key): Exit")
CHOICE = input("Input: ")
if CHOICE == "2":
    simulate(1)
elif CHOICE == "1":
    simulate(0)
else:
    os._exit(0)
