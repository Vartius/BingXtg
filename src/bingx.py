
import json
import time

from loguru import logger
from bingx.api import BingxAPI

try:
    from config import (
        MIN_ORDERS_TO_HIGH,
        MAX_PERCENT,
        LEVERAGE,
        SECRETKEY,
        APIKEY,
        TP,
        SL,
    )
except ImportError:
    logger.error("Configuration parameters are not set in config.py. Please define them.")
    exit(1)


def set_order(chan_id, coin, method, sim=True):
    try:
        with open("data/curdata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        with open("data/winrate.json", "r", encoding="utf-8") as f:
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
        with open("data/curdata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"UPDATER: Could not read curdata.json: {e}")
        return

    orders_to_delete = []

    for chan_id in data.get("orders", {}):
        for coin, order in data["orders"][chan_id].items():
            price = get_price(coin)
            if price is None:
                time.sleep(5)
                continue

            order["cur_price"] = price
            order_price = order.get("order_price", 0)
            leverage = LEVERAGE
            money = order.get("money", 0)

            if order_price == 0: 
                continue

            if order.get("method") == "long":
                profit_perc = (price / order_price - 1) * leverage
            else:
                profit_perc = (order_price / price - 1) * leverage

            profit = (money / leverage) * profit_perc
            order["profitPerc"] = profit_perc * 100
            order["profit"] = profit

            if profit_perc >= TP or profit_perc <= SL:
                try:
                    with open("data/winrate.json", "r", encoding="utf-8") as f:
                        winrate = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    winrate = {}
                
                if chan_id not in winrate:
                    winrate[chan_id] = {"win": 0, "lose": 0}

                if profit_perc >= TP:
                    winrate[chan_id]["win"] += 1
                else:
                    winrate[chan_id]["lose"] += 1

                with open("data/winrate.json", "w", encoding="utf-8") as f:
                    json.dump(winrate, f, ensure_ascii=False, indent=4)
                
                data["available_balance"] += (money / leverage) + profit
                logger.success(f'{order} was closed with {profit}$')
                orders_to_delete.append((chan_id, coin))

    if orders_to_delete:
        for chan_id, coin in orders_to_delete:
            if chan_id in data["orders"] and coin in data["orders"][chan_id]:
                del data["orders"][chan_id][coin]

    balance = data.get("available_balance", 0)
    for chan_id in data.get("orders", {}):
        for coin, order in data["orders"][chan_id].items():
            profit = order.get("profit", 0)
            balance += profit + order.get("money", 0) / LEVERAGE
    data["balance"] = balance

    try:
        with open("data/curdata.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logger.error(f"UPDATER: Could not write to curdata.json: {e}")


def updater():
    while True:
        try:
            update_orders()
            
            with open("data/curdata.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            with open("data/channels.json", "r", encoding="utf-8") as f:
                channels = json.load(f)
            with open("data/winrate.json", "r", encoding="utf-8") as f:
                winrate = json.load(f)

            table_file = {"data": []}
            for chan_id, order_data in data.get("orders", {}).items():
                for coin, order in order_data.items():
                    table_file["data"].append([
                        channels.get(chan_id, {}).get("name", "Unknown"),
                        coin,
                        order.get("method"),
                        round(order.get("money", 0), 3),
                        order.get("order_price"),
                        order.get("cur_price"),
                        round(order.get("profit", 0), 3),
                        round(order.get("profitPerc", 0), 3),
                    ])

            wins = sum(w.get("win", 0) for w in winrate.values())
            loses = sum(w.get("lose", 0) for w in winrate.values())
            winrate_global = round(wins / (wins + loses) * 100, 2) if (wins + loses) > 0 else 0

            table_file["data"].append([])
            table_file["data"].append(["Available balance:", round(data.get("available_balance", 0), 2)])
            table_file["data"].append(["Balance:", round(data.get("balance", 0), 2)])
            table_file["data"].append(["Winrate:", winrate_global])
            
            with open("data/table.json", "w", encoding="utf-8") as f:
                json.dump(table_file, f, indent=4)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error in updater loop: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in updater: {e}")
        
        time.sleep(1)

