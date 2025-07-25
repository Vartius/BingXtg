"""
This module contains the core logic for order management, including placing new orders,
updating their status, and handling the main update loop.
"""

import time
import sys
from loguru import logger
from src.bingx_api import get_price, set_order_bingx
from src.data_handler import (
    get_state,
    save_state,
    get_winrate,
    save_winrate,
    get_channels,
    save_table,
)

# --- Configuration Import ---
try:
    from config import MIN_ORDERS_TO_HIGH, MAX_PERCENT, LEVERAGE, TP, SL
except ImportError:
    logger.critical(
        "`config.py` is missing or incomplete. Please configure it before running."
    )
    sys.exit(1)


def place_order(channel_id: str, coin: str, side: str, is_simulation: bool) -> bool:
    """
    Places a new order based on a signal, either in simulation or live mode.

    Args:
        channel_id: The ID of the channel that sent the signal.
        coin: The coin ticker (e.g., "BTC").
        side: The order side ("long" or "short").
        is_simulation: If True, simulates the trade; otherwise, places a real order.

    Returns:
        True if the order was successfully placed/simulated, False otherwise.
    """
    state = get_state()
    winrate_data = get_winrate()
    if not state or not winrate_data:
        logger.error(
            f"Could not get state or winrate data to place order for {channel_id}."
        )
        return False

    # Calculate winrate factor for dynamic position sizing
    channel_winrate = winrate_data.get(channel_id, {"win": 0, "lose": 0})
    total_trades = channel_winrate["win"] + channel_winrate["lose"]

    winrate_factor = 0
    if total_trades > MIN_ORDERS_TO_HIGH:
        winrate_factor = channel_winrate["win"] / total_trades

    # Base investment is 1% of balance, with a bonus up to MAX_PERCENT based on winrate
    investment_percent = 0.01 + (MAX_PERCENT - 0.01) * winrate_factor
    margin = state.get("available_balance", 0) * investment_percent

    # In live mode, place the order via the API
    if not is_simulation:
        if not set_order_bingx(coin, side, investment_percent):
            logger.error(f"Failed to place live order for {coin} {side}.")
            return False

    # For both live and simulation, update the state file
    price = get_price(coin)
    if price is None:
        logger.warning(f"Could not fetch price for {coin}. Order placement aborted.")
        return False

    state["available_balance"] -= margin
    order_details = {
        "side": side,
        "margin": margin,
        "entry_price": price,
        "current_price": price,
        "pnl": 0.0,
        "pnl_percent": 0.0,
    }

    if channel_id not in state["orders"]:
        state["orders"][channel_id] = {}
    state["orders"][channel_id][coin] = order_details

    return save_state(state)


def _update_open_orders(state: dict, winrate_data: dict) -> tuple[dict, dict]:
    """Helper function to process and update all open orders."""
    orders_to_close = []
    for channel_id, orders in state.get("orders", {}).items():
        for coin, order in orders.items():
            price = get_price(coin)
            if price is None:
                continue  # Skip update if price is unavailable

            order["current_price"] = price
            entry_price = order.get("entry_price", 0)
            if entry_price == 0:
                continue

            # Calculate PnL
            if order.get("side") == "long":
                pnl_percent = (price / entry_price - 1) * LEVERAGE
            else:  # Short
                pnl_percent = (entry_price / price - 1) * LEVERAGE

            order["pnl"] = order["margin"] * pnl_percent
            order["pnl_percent"] = pnl_percent * 100

            # Check for TP/SL hit
            if pnl_percent >= TP or pnl_percent <= SL:
                if pnl_percent >= TP:
                    winrate_data[channel_id]["win"] += 1
                else:
                    winrate_data[channel_id]["lose"] += 1

                state["available_balance"] += order["margin"] + order["pnl"]
                logger.success(f"Order for {coin} closed with PnL: ${order['pnl']:.2f}")
                orders_to_close.append((channel_id, coin))

    # Remove closed orders from the state
    for channel_id, coin in orders_to_close:
        if coin in state["orders"].get(channel_id, {}):
            del state["orders"][channel_id][coin]

    return state, winrate_data


def _update_display_data(state: dict, channels: dict, winrate_data: dict):
    """Prepares and saves the data required for the GUI table."""
    table_orders = []
    total_pnl = 0
    total_margin = 0

    for channel_id, orders in state.get("orders", {}).items():
        for coin, order in orders.items():
            channel_name = channels.get(channel_id, {}).get("name", "Unknown")
            table_orders.append(
                [
                    channel_name,
                    coin,
                    order.get("side"),
                    f"${order.get('margin', 0):.2f}",
                    order.get("entry_price"),
                    order.get("current_price"),
                    f"${order.get('pnl', 0):.2f}",
                    f"{order.get('pnl_percent', 0):.2f}%",
                ]
            )
            total_pnl += order.get("pnl", 0)
            total_margin += order.get("margin", 0)

    # Calculate global winrate
    total_wins = sum(w.get("win", 0) for w in winrate_data.values())
    total_loses = sum(w.get("lose", 0) for w in winrate_data.values())
    global_winrate = (
        (total_wins / (total_wins + total_loses) * 100)
        if (total_wins + total_loses) > 0
        else 0
    )

    # Calculate total equity
    state["balance"] = state.get("available_balance", 0) + total_margin + total_pnl

    table_data = {
        "orders": table_orders,
        "available_balance": round(state.get("available_balance", 0), 2),
        "balance": round(state.get("balance", 0), 2),
        "winrate": round(global_winrate, 2),
    }
    save_table(table_data)
    save_state(state)  # Save final state after balance calculation
    save_winrate(winrate_data)


def updater_thread_worker():
    """
    The main worker loop that runs in a separate thread. It periodically updates
    order statuses, calculates PnL, and prepares data for the GUI.
    """
    while True:
        try:
            state = get_state()
            winrate_data = get_winrate()
            channels = get_channels()

            if not all([state, winrate_data, channels]):
                logger.warning(
                    "Updater: Missing state, winrate, or channels data. Retrying..."
                )
                time.sleep(5)
                continue

            # Update order PnL and close positions that hit TP/SL
            state, winrate_data = _update_open_orders(state, winrate_data)

            # Prepare and save data for the GUI and save updated state
            _update_display_data(state, channels, winrate_data)

        except Exception as e:
            logger.error(
                f"An unexpected error occurred in the updater thread: {e}",
                exc_info=True,
            )

        time.sleep(2)  # Update interval
