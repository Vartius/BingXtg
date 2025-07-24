"""
This module handles order management.
"""

import time
from loguru import logger

from src.bingx_api import get_price, set_order_bingx
from src.data_handler import (
    get_curdata,
    save_curdata,
    get_winrate,
    save_winrate,
    get_channels,
    save_table,
)

try:
    from config import (
        MIN_ORDERS_TO_HIGH,
        MAX_PERCENT,
        LEVERAGE,
        TP,
        SL,
    )
except ImportError:
    logger.error(
        "Configuration parameters are not set in config.py. Please define them."
    )
    exit(1)


def set_order(chan_id, coin, method, sim=True):
    """Places a new order."""
    data = get_curdata()
    winrate = get_winrate()
    if data is None or winrate is None:
        logger.error(f"Could not get data for setting order for {chan_id}")
        return None

    logger.success(f"{chan_id} got winrate and curdata")

    chan_winrate = winrate.get(chan_id, {"win": 0, "lose": 0})

    if (chan_winrate["lose"] + chan_winrate["win"]) <= MIN_ORDERS_TO_HIGH:
        k = 0
    else:
        k = chan_winrate["win"] / (chan_winrate["lose"] + chan_winrate["win"])
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
    save_curdata(data)
    return data


def update_orders():
    """Updates existing orders."""
    data = get_curdata()
    if data is None:
        logger.error("UPDATER: Could not read curdata.json")
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
            money = order.get("money", 0)

            if order_price == 0:
                continue

            if order.get("method") == "long":
                profit_perc = (price / order_price - 1) * LEVERAGE
            else:
                profit_perc = (order_price / price - 1) * LEVERAGE

            profit = (money / LEVERAGE) * profit_perc
            order["profitPerc"] = profit_perc * 100
            order["profit"] = profit

            if profit_perc >= TP or profit_perc <= SL:
                winrate = get_winrate()
                if winrate is None:
                    winrate = {}

                if chan_id not in winrate:
                    winrate[chan_id] = {"win": 0, "lose": 0}

                if profit_perc >= TP:
                    winrate[chan_id]["win"] += 1
                else:
                    winrate[chan_id]["lose"] += 1

                save_winrate(winrate)

                data["available_balance"] += (money / LEVERAGE) + profit
                logger.success(f"{order} was closed with {profit}$")
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

    save_curdata(data)


def updater():
    """Main updater loop."""
    while True:
        try:
            update_orders()

            data = get_curdata()
            channels = get_channels()
            winrate = get_winrate()

            if not all([data, channels, winrate]):
                logger.error("Updater loop missing data, skipping iteration.")
                time.sleep(1)
                continue

            orders = []
            if not data:
                logger.error("No data found, skipping order processing.")
                time.sleep(1)
                continue
            for chan_id, order_data in data.get("orders", {}).items():
                for coin, order in order_data.items():
                    if not channels:
                        logger.error("No channels found, skipping order processing.")
                        continue
                    orders.append(
                        [
                            channels.get(chan_id, {}).get("name", "Unknown"),
                            coin,
                            order.get("method"),
                            round(order.get("money", 0), 3),
                            order.get("order_price"),
                            order.get("cur_price"),
                            round(order.get("profit", 0), 3),
                            round(order.get("profitPerc", 0), 3),
                        ]
                    )

            if not winrate:
                logger.error("No winrate data found, skipping order processing.")
                continue
            wins = sum(w.get("win", 0) for w in winrate.values())
            loses = sum(w.get("lose", 0) for w in winrate.values())
            winrate_global = (
                round(wins / (wins + loses) * 100, 2) if (wins + loses) > 0 else 0
            )

            table_data = {
                "orders": orders,
                "available_balance": round(data.get("available_balance", 0), 2),
                "balance": round(data.get("balance", 0), 2),
                "winrate": winrate_global,
            }

            save_table(table_data)

        except Exception as e:
            logger.error(f"An unexpected error occurred in updater: {e}", exc_info=True)

        time.sleep(1)
