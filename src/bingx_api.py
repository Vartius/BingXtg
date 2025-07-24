import sys
from functools import lru_cache
from loguru import logger
from bingx.api import BingxAPI

# --- Configuration Import ---
try:
    from config import LEVERAGE, SECRETKEY, APIKEY, TP, SL
except ImportError:
    logger.critical(
        "`config.py` is missing or incomplete. Please configure it before running."
    )
    sys.exit(1)


@lru_cache(maxsize=1)
def get_bingx_client() -> BingxAPI:
    """
    Initializes and returns a singleton BingxAPI client instance.
    The client is cached to avoid re-creation on every call.
    """
    if not APIKEY or not SECRETKEY:
        logger.critical("BingX APIKEY or SECRETKEY is not set in `config.py`.")
        raise ValueError("API credentials are not configured.")
    return BingxAPI(APIKEY, SECRETKEY, timestamp="local")


def get_price(coin: str) -> float | None:
    """
    Gets the latest price of a specified coin ticker from BingX.

    Args:
        coin: The coin ticker (e.g., "BTC").

    Returns:
        The latest price as a float, or None if an error occurs.
    """
    try:
        bingx = get_bingx_client()
        price_response = bingx.get_latest_price(f"{coin}-USDT")
        if price_response.get("code") == 0:
            return float(price_response["data"]["price"])
        else:
            logger.error(f"Error getting price for {coin}: {price_response.get('msg')}")
            return None
    except Exception as e:
        logger.error(f"Exception while getting price for {coin}: {e}")
        return None


def get_balance() -> tuple[float, float] | tuple[None, None]:
    """
    Gets the total and available balance from the BingX perpetual account.

    Returns:
        A tuple of (total_balance, available_balance), or (None, None) on failure.
    """
    try:
        bingx = get_bingx_client()
        res = bingx.get_perpetual_balance()
        if res and res.get("code") == 0:
            balance_data = res["data"]["balance"]
            balance = float(balance_data["balance"])
            available_balance = float(balance_data["availableMargin"])
            return balance, available_balance
        else:
            logger.error(f"Error fetching balance: {res.get('msg')}")
            return None, None
    except Exception as e:
        logger.error(f"Exception in get_balance: {e}")
        return None, None


def set_order_bingx(coin: str, direction: str, percent: float) -> bool:
    """
    Places a market order on BingX with pre-configured leverage, TP, and SL.

    Args:
        coin: The coin ticker (e.g., "BTC").
        direction: The trade direction ("LONG" or "SHORT").
        percent: The percentage of the available balance to use for the margin.

    Returns:
        True if the order was placed successfully, False otherwise.
    """
    symbol = f"{coin}-USDT"
    try:
        bingx = get_bingx_client()

        # Set margin mode and leverage
        bingx.set_margin_mode(symbol, "ISOLATED")
        bingx.set_levarage(symbol, direction.upper(), LEVERAGE)

        # Get current price and balance
        price = get_price(coin)
        _, available_balance = get_balance()

        if price is None or available_balance is None:
            logger.error(
                f"Could not retrieve price or balance for {symbol} to place order."
            )
            return False

        # Calculate quantity and TP/SL levels
        quantity = (available_balance * percent / price) * LEVERAGE

        if direction.upper() == "LONG":
            take_profit = price * (1 + TP / LEVERAGE)
            stop_loss = price * (1 - SL / LEVERAGE)
        else:  # SHORT
            take_profit = price * (1 - TP / LEVERAGE)
            stop_loss = price * (1 + SL / LEVERAGE)

        # Place the market order
        order_data = bingx.open_market_order(
            symbol,
            direction.upper(),
            quantity,
            tp=f"{take_profit:.8f}",
            sl=f"{stop_loss:.8f}",
        )

        if order_data.get("code") == 0:
            order_info = order_data["data"]["order"]
            logger.success(
                f"ORDER PLACED: {order_info['symbol']} {order_info['positionSide']} | OrderID: {order_info['orderId']}"
            )
            return True
        else:
            logger.error(f"Failed to place order for {symbol}: {order_data.get('msg')}")
            return False

    except Exception as e:
        logger.error(f"Exception in set_order_bingx for {symbol}: {e}")
        return False
