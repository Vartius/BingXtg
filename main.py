import json
import os

from loguru import logger

from src.core.tgparser import start_parsing
from config import START_BALANCE


def simulate(sim):
    with open("src/data/curdata.json", encoding="utf-8") as f:
        data = json.load(f)

    with open("src/data/channels.json", encoding="utf-8") as f:
        channels = json.load(f)

    with open("src/data/winrate.json", encoding="utf-8") as f:
        winrate = json.load(f)

    if "balance" in data:
        CHOICE = input("Found previous data, do you want to renew it? Y/N: ")
    else:
        CHOICE = "y"

    if CHOICE == "y" or CHOICE == "Y":
        data = {
            "balance": START_BALANCE,
            "available_balance": START_BALANCE,
            "winrate": 0,
            "orders": {},
        }
    else:
        data = {
            "balance": data["balance"],
            "available_balance": data["balance"],
            "winrate": data["winrate"],
            "orders": {},
        }
        with open("config.py", "r", encoding="utf-8") as f:
            conf = f.read().replace(str(START_BALANCE), str(data["balance"]))
        with open("config.py", "w+", encoding="utf-8") as f:
            f.write(conf)

    for i in channels:
        if i not in winrate.keys():
            winrate[i] = {"name": channels[i]["name"], "win": 0, "lose": 0}
        data["orders"][i] = {}

    with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    with open("src/data/winrate.json", "w+", encoding="utf-8") as f:
        json.dump(winrate, f, indent=4)

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
