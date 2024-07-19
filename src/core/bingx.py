import json
import time
import requests

from loguru import logger
from bingx.api import BingxAPI

from config import (
    MIN_ORDERS_TO_HIGH,
    MAX_PERCENT,
    LEVERAGE,
    SECRETKEY,
    APIKEY,
    TP,
    SL,
    APIURL,
)


def set_order(chan_id, coin, method, sim=True):
    with open("src/data/curdata.json", encoding="utf-8") as f:
        data = json.load(f)
    with open("src/data/winrate.json", encoding="utf-8") as f:
        winrate = json.load(f)
    logger.success(f"{chan_id} got winrate and curdata")

    if (winrate[chan_id]["lose"] + winrate[chan_id]["win"]) <= MIN_ORDERS_TO_HIGH:
        k = 0
    else:
        k = winrate[chan_id]["win"] / (
            winrate[chan_id]["lose"] + winrate[chan_id]["win"]
        )
        logger.log(f"IT SHOULD BE TRADING {k} {not sim}")
    money = data["available_balance"] * (0.01 + MAX_PERCENT * k)

    if k != 0 and not sim:
        logger.log(f"IT SHOULD BE TRADING OKAY {k} {not sim}")
        set_order_bingx(coin, method, 0.01 + MAX_PERCENT * k)

    data["available_balance"] -= money
    price = get_price(coin)
    if price is None:
        logger.warning(f"{coin} price = None")
        data["available_balance"] += money
        return data
    data["orders"][chan_id][coin] = {
        "method": method,
        "money": money * LEVERAGE,
        "order_price": price,
        "cur_price": price,
        "profit": 0,
        "profitPerc": 0,
    }
    return data


def get_price(coin):
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        return float(bingx.get_latest_price(f"{coin}-USDT"))
    except Exception as e:
        logger.error(e)
        return None


def get_balance():
    bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")

    res = bingx.get_perpetual_balance()["data"]["balance"]
    print(res)
    balance = res["balance"]
    available_balance = res["availableMargin"]
    print("Total balance:", balance)
    print("Available balance:", available_balance)

    return balance, available_balance


def set_order_bingx(coin, diraction, percent):
    bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
    bingx.set_margin_mode(f"{coin}-USDT", "ISOLATED")
    bingx.set_levarage(f"{coin}-USDT", diraction, LEVERAGE)
    price = float(bingx.get_latest_price(f"{coin}-USDT"))
    balance, available_balance = get_balance()
    print(LEVERAGE)
    q = float(available_balance) * percent / price * LEVERAGE
    if diraction == "LONG":
        take = "{:.8f}".format(price * (TP / LEVERAGE + 1))
        stop = "{:.8f}".format(price * (SL / LEVERAGE + 1))
    else:
        stop = "{:.8f}".format(price * (TP / LEVERAGE + 1))
        take = "{:.8f}".format(price * (SL / LEVERAGE + 1))

    try:
        order_data = bingx.open_market_order(
            f"{coin}-USDT", diraction, q, tp=take, sl=stop
        )
        logger.success(
            f"ORDER: {order_data['symbol']} {order_data['positionSide']} {order_data['orderId']}"
        )
    except Exception as e:
        logger.error(e)


def update_orders():
    try:
        with open("src/data/curdata.json", encoding="utf-8") as f:
            data = json.load(f)
        for chan_id in data["orders"]:
            for coin in data["orders"][chan_id].keys():
                price = get_price(coin)
                if price is None:
                    time.sleep(5)
                    return
                with open("src/data/curdata.json", encoding="utf-8") as f:
                    data = json.load(f)
                data["orders"][chan_id][coin]["cur_price"] = price
                if data["orders"][chan_id][coin]["method"] == "long":
                    profit_perc = (
                        price / data["orders"][chan_id][coin]["order_price"] - 1
                    ) * LEVERAGE
                else:
                    profit_perc = (
                        data["orders"][chan_id][coin]["order_price"] / price - 1
                    ) * LEVERAGE

                profit = data["orders"][chan_id][coin]["money"] / LEVERAGE * profit_perc

                data["orders"][chan_id][coin]["profitPerc"] = profit_perc * 100
                data["orders"][chan_id][coin]["profit"] = profit

                if profit_perc >= TP:
                    with open("src/data/winrate.json", encoding="utf-8") as f:
                        winrate = json.load(f)
                    winrate[chan_id]["win"] += 1
                    with open("src/data/winrate.json", "w+", encoding="utf-8") as f:
                        json.dump(winrate, f, ensure_ascii=False, indent=4)
                    data["available_balance"] += (
                        profit + data["orders"][chan_id][coin]["money"] / LEVERAGE
                    )
                    logger.success(
                        f'{data["orders"][chan_id][coin]} was closed with {profit}$'
                    )
                    del data["orders"][chan_id][coin]
                elif profit_perc <= SL:
                    with open("src/data/winrate.json", encoding="utf-8") as f:
                        winrate = json.load(f)
                    winrate[chan_id]["lose"] += 1
                    with open("src/data/winrate.json", "w+", encoding="utf-8") as f:
                        json.dump(winrate, f, ensure_ascii=False, indent=4)
                    data["available_balance"] += (
                        profit + data["orders"][chan_id][coin]["money"] / LEVERAGE
                    )
                    logger.success(
                        f'{data["orders"][chan_id][coin]} was closed with {profit}$'
                    )
                    del data["orders"][chan_id][coin]
                with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

        with open("src/data/curdata.json", encoding="utf-8") as f:
            data = json.load(f)
        balance = data["available_balance"]
        for chan_id in data["orders"]:
            for coin in data["orders"][chan_id].keys():
                balance += profit + data["orders"][chan_id][coin]["money"] / LEVERAGE
        data["balance"] = balance
        with open("src/data/curdata.json", "w+", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"UPDATER: {e}")


def updater():
    while 1:
        with open("src/data/channels.json", encoding="utf-8") as f:
            channels = json.load(f)
        update_orders()
        with open("src/data/curdata.json", encoding="utf-8") as f:
            data = json.load(f)
        table_file = {"data": []}
        for chan_id in data["orders"]:
            for coin in data["orders"][chan_id]:
                method = data["orders"][chan_id][coin]["method"]
                money = round(data["orders"][chan_id][coin]["money"], 3)
                order_price = data["orders"][chan_id][coin]["order_price"]
                cur_price = data["orders"][chan_id][coin]["cur_price"]
                profit = data["orders"][chan_id][coin]["profit"]
                profit_perc = data["orders"][chan_id][coin]["profitPerc"]
                table_file["data"].append(
                    [
                        channels[chan_id]["name"],
                        coin,
                        method,
                        money,
                        order_price,
                        cur_price,
                        round(profit, 3),
                        round(profit_perc, 3),
                    ]
                )
        with open("src/data/winrate.json", encoding="utf-8") as f:
            winrate = json.load(f)
        wins = 0
        loses = 0
        for i in winrate:
            wins += winrate[i]["win"]
            loses += winrate[i]["lose"]
        if wins + loses == 0:
            winrate_global = 0
        else:
            winrate_global = round(wins / (wins + loses) * 100, 2)
        table_file["data"].append([])
        table_file["data"].append(
            ["Available balance:", round(data["available_balance"], 2)]
        )
        table_file["data"].append(["Balance:", round(data["balance"], 2)])
        table_file["data"].append(["Winrate:", round(winrate_global, 2)])
        with open("src/data/table.json", "w+", encoding="utf-8") as f:
            json.dump(table_file, f, indent=4)
        time.sleep(1)
