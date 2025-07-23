
import json
import time

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
)


def set_order(chan_id, coin, method, sim=True):
    try:
        with open("src/data/curdata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        with open("src/data/winrate.json", "r", encoding="utf-8") as f:
            winrate = json.load(f)
        logger.success(f"{chan_id} got winrate and curdata")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading data files in set_order: {e}")
        return None

    chan_winrate = winrate.get(chan_id, {"win": 0, "lose": 0})

    if (chan_winrate["lose"] + chan_winrate["win"]) <= MIN_ORDERS_TO_HIGH:
        k = 0
    else:
        k = chan_winrate["win"] / (
            chan_winrate["lose"] + chan_winrate["win"]
        )
        logger.info(f"IT SHOULD BE TRADING {k} {not sim}")

    money = data.get("available_balance", 0) * (0.01 + MAX_PERCENT * k)

    if k != 0 and not sim:
        logger.info(f"IT SHOULD BE TRADING OKAY {k} {not sim}")
        set_order_bingx(coin, method, 0.01 + MAX_PERCENT * k)

    data["available_balance"] -= money
    price = get_price(coin)
    if price is None:
        logger.warning(f"{coin} price = None")
        data["available_balance"] += money
        return data
        
    if chan_id not in data["orders"]:
        data["orders"][chan_id] = {}

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
        return float(bingx.get_latest_price(f"{coin}-USDT")['data']['price'])
    except Exception as e:
        logger.error(f"Error getting price for {coin}: {e}")
        return None


def get_balance():
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        res = bingx.get_perpetual_balance()
        if res and res.get('code') == 0:
            balance_data = res['data']['balance']
            balance = balance_data["balance"]
            available_balance = balance_data["availableMargin"]
            print("Total balance:", balance)
            print("Available balance:", available_balance)
            return balance, available_balance
        else:
            logger.error(f"Error fetching balance: {res.get('msg')}")
            return None, None
    except Exception as e:
        logger.error(f"Exception in get_balance: {e}")
        return None, None

def set_order_bingx(coin, diraction, percent):
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        bingx.set_margin_mode(f"{coin}-USDT", "ISOLATED")
        bingx.set_levarage(f"{coin}-USDT", diraction, LEVERAGE)
        
        price_response = bingx.get_latest_price(f"{coin}-USDT")
        if price_response.get('code') != 0:
            logger.error(f"Could not get price for {coin} to set order.")
            return

        price = float(price_response['data']['price'])
        balance, available_balance = get_balance()

        if balance is None or available_balance is None:
            logger.error("Could not retrieve balance to set order.")
            return

        q = float(available_balance) * percent / price * LEVERAGE
        
        if diraction.upper() == "LONG":
            take = "{:.8f}".format(price * (1 + TP / LEVERAGE))
            stop = "{:.8f}".format(price * (1 - SL / LEVERAGE))
        else:
            take = "{:.8f}".format(price * (1 - TP / LEVERAGE))
            stop = "{:.8f}".format(price * (1 + SL / LEVERAGE))

        order_data = bingx.open_market_order(
            f"{coin}-USDT", diraction.upper(), q, tp=take, sl=stop
        )

        if order_data.get('code') == 0:
            logger.success(
                f"ORDER: {order_data['data']['order']['symbol']} {order_data['data']['order']['positionSide']} {order_data['data']['order']['orderId']}"
            )
        else:
            logger.error(f"Failed to place order: {order_data.get('msg')}")

    except Exception as e:
        logger.error(f"Exception in set_order_bingx: {e}")



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
                order = data["orders"][chan_id][coin]
                profit = order["profit"]
                balance += profit + order["money"] / LEVERAGE
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
