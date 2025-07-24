"""
This module handles all interactions with the Bingx API.
"""

from loguru import logger
from bingx.api import BingxAPI

try:
    from config import (
        LEVERAGE,
        SECRETKEY,
        APIKEY,
        TP,
        SL,
    )
except ImportError:
    logger.error(
        "Configuration parameters are not set in config.py. Please define them."
    )
    exit(1)


def get_price(coin):
    """Gets the latest price of a coin from BingX"""
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        return float(bingx.get_latest_price(f"{coin}-USDT"))
    except Exception as e:
        logger.error(f"Error getting price for {coin}: {e}")
        return None


def get_balance():
    """Gets the balance from BingX"""
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        res = bingx.get_perpetual_balance()
        if res and res.get("code") == 0:
            balance_data = res["data"]["balance"]
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
    """Sets an order on BingX"""
    try:
        bingx = BingxAPI(APIKEY, SECRETKEY, timestamp="local")
        bingx.set_margin_mode(f"{coin}-USDT", "ISOLATED")
        bingx.set_levarage(f"{coin}-USDT", diraction, LEVERAGE)

        price_response = bingx.get_latest_price(f"{coin}-USDT")
        if price_response.get("code") != 0:
            logger.error(f"Could not get price for {coin} to set order.")
            return

        price = float(price_response["data"]["price"])
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

        if order_data.get("code") == 0:
            logger.success(
                f"ORDER: {order_data['data']['order']['symbol']} {order_data['data']['order']['positionSide']} {order_data['data']['order']['orderId']}"
            )
        else:
            logger.error(f"Failed to place order: {order_data.get('msg')}")

    except Exception as e:
        logger.error(f"Exception in set_order_bingx: {e}")
