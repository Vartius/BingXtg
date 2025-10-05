"""
This module contains the core logic for order management, including placing new orders,
updating their status, and handling the main update loop.
"""

import sqlite3
import time
import sys
import os
from loguru import logger
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from dotenv import load_dotenv
from .bingx_api import get_price, set_order_bingx

# Global variable for debouncing websocket updates
_last_websocket_update = 0
_WEBSOCKET_UPDATE_INTERVAL = 1.0  # Minimum 1 second between updates

# Load environment variables
load_dotenv()

# --- Configuration from Environment Variables ---
try:
    MIN_ORDERS_TO_HIGH = int(os.getenv("MIN_ORDERS_TO_HIGH", "20"))
    MAX_PERCENT = float(os.getenv("MAX_PERCENT", "30"))
    LEVERAGE = int(os.getenv("LEVERAGE", "20"))
    TP = float(os.getenv("TP", "20"))
    SL = float(os.getenv("SL", "-50"))
    # Used for ROI calculation in trading_stats
    START_BALANCE = float(os.getenv("START_BALANCE", "0"))
except (ValueError, TypeError) as e:
    logger.critical(f"Configuration error: {e}. Please check your .env file.")
    sys.exit(1)


def _normalize_channel_id(channel_id: int) -> int:
    """
    Normalizes channel ID to handle both positive and negative formats.
    Telegram channels can have IDs like -1002595715996 (negative)
    but may be stored in database as 2595715996 (positive).

    Args:
        channel_id: The raw channel ID from Telegram

    Returns:
        The normalized channel ID that should exist in the database
    """
    if channel_id < 0:
        # Convert negative format (-1002595715996) to positive (2595715996)
        str_id = str(abs(channel_id))
        if str_id.startswith("100"):
            return int(str_id[3:])  # Remove the "100" prefix
    return abs(channel_id)


def place_order(channel_id: int, data: dict, is_simulation: bool) -> bool:
    """
    Places a new order based on a signal, using the SQLite DB (trades table)

    Args:
        channel_id: The ID of the channel that sent the signal.
        data: A dictionary containing the signal data (all data).
        is_simulation: If True, simulates the trade; otherwise, places a real order.

    Returns:
        True if the order was successfully placed/simulated, False otherwise.
    """
    coin = "???"
    try:
        # Import here to avoid touching module-level imports
        from .bingx_api import get_balance

        # Normalize channel ID to handle both positive and negative formats
        normalized_channel_id = _normalize_channel_id(channel_id)
        logger.debug(f"Channel ID {channel_id} normalized to {normalized_channel_id}")

        # Handle direction conversion (AI models return numeric: 0=long, 1=short)
        direction_value = data.get("direction")
        if isinstance(direction_value, (int, float)):
            direction = "long" if int(direction_value) == 0 else "short"
        else:
            direction = str(direction_value)

        coin = str(data.get("pair"))
        # Fetch current price first; required for both live and simulation
        price = get_price(coin)
        if price is None:
            return False

        # Resolve DB path and channel id
        db_path = os.getenv("DB_PATH", "total.db")
        # Compute dynamic position sizing based on per-channel historical winrate from SQL
        wins = losses = 0
        if normalized_channel_id is not None:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

                # Ensure trades table exists
                _ensure_trades_table(conn)

                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT 
                        SUM(CASE WHEN status='closed' AND pnl > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN status='closed' AND pnl < 0 THEN 1 ELSE 0 END) AS losses
                    FROM trades
                    WHERE channel_id = ?
                    """,
                    (normalized_channel_id,),
                )
                row = cur.fetchone() or (0, 0)
                wins = int(row[0] or 0)
                losses = int(row[1] or 0)

        total_trades = wins + losses
        winrate_factor = (
            (wins / total_trades) if total_trades > MIN_ORDERS_TO_HIGH else 0.0
        )

        # Base investment is 1% of balance, with a bonus up to MAX_PERCENT based on winrate
        max_percent_decimal = MAX_PERCENT / 100  # Convert percentage to decimal
        investment_percent = 0.01 + (max_percent_decimal - 0.01) * winrate_factor

        # Determine margin from BingX available balance (fallback to START_BALANCE if unavailable or in simulation)
        if is_simulation:
            # In simulation mode, always use START_BALANCE
            available_balance = START_BALANCE
        else:
            # In live mode, use actual BingX balance with fallback
            _, available_balance = get_balance()
            if available_balance is None:
                available_balance = START_BALANCE
        margin = float(available_balance) * float(investment_percent)

        # Log margin calculation details for debugging
        logger.debug(
            f"Margin calculation - Channel: {normalized_channel_id}, Wins: {wins}, Losses: {losses}, "
            f"Winrate factor: {winrate_factor:.4f}, Max percent: {MAX_PERCENT}% ({max_percent_decimal:.4f}), "
            f"Investment %: {investment_percent:.4f}, Available balance: {available_balance}, "
            f"Calculated margin: {margin:.2f}"
        )
        if margin <= 0:
            logger.error("Calculated margin <= 0; cannot place order.")
            return False

        # Place live order via API when not simulating
        if not is_simulation:
            order_data = {
                "coin": coin,
                "direction": direction,
                "pair": data.get("pair", f"{coin}-USDT"),
                "entry": data.get("entry"),
                "stop_loss": data.get("stop_loss"),
                "targets": data.get("targets", []),
                "leverage": data.get("leverage", LEVERAGE),
                "percent": investment_percent,
            }
            if not set_order_bingx(order_data):
                logger.error(f"Failed to place live order for {coin} {direction}.")
                return False

        # Persist the trade in SQL (status = 'open')
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Ensure trades table exists
            _ensure_trades_table(conn)

            # Ensure channel exists in channels table
            conn.execute(
                "INSERT OR IGNORE INTO channels (channel_id) VALUES (?)",
                (normalized_channel_id,),
            )

            # Extract additional fields from data
            if data.get("leverage"):
                leverage = float(data.get("leverage", LEVERAGE))
            else:
                leverage = LEVERAGE

            if data.get("targets"):
                raw_targets = data.get("targets")
            else:
                raw_targets = [
                    price * (1 + TP / 100 / leverage)
                    if direction.upper() == "LONG"
                    else price * (1 - TP / 100 / leverage)
                ]
            if data.get("stop_loss"):
                sl = data.get("stop_loss")
            else:
                sl = (
                    price * (1 + SL / 100 / leverage)
                    if direction.upper() == "LONG"
                    else price * (1 - SL / 100 / leverage)
                )

            # Determine entry price and status
            entry_price = data.get("entry")
            if entry_price is not None:
                entry_price = float(entry_price)
                # Check if we need to wait for entry or can enter immediately
                entry_tolerance = 0.001  # 0.1% tolerance for immediate entry

                if direction.upper() == "LONG":
                    # For LONG: enter immediately if current price is at or below entry price (+ tolerance)
                    can_enter_immediately = price <= entry_price * (1 + entry_tolerance)
                else:  # SHORT
                    # For SHORT: enter immediately if current price is at or above entry price (- tolerance)
                    can_enter_immediately = price >= entry_price * (1 - entry_tolerance)

                status = "open" if can_enter_immediately else "waiting"
            else:
                # No specific entry price, enter at current market price
                entry_price = price
                status = "open"

            # Filter targets based on direction and entry price
            if raw_targets:
                if direction.upper() == "LONG":
                    # For LONG: only keep targets above entry price
                    valid_targets = [t for t in raw_targets if float(t) > entry_price]
                else:  # SHORT
                    # For SHORT: only keep targets below entry price
                    valid_targets = [t for t in raw_targets if float(t) < entry_price]

                targets = str(valid_targets) if valid_targets else str([])
                if not valid_targets:
                    logger.warning(
                        f"No valid targets for {direction} trade at entry {entry_price}. Original targets: {raw_targets}"
                    )
                else:
                    logger.info(
                        f"Valid targets for {direction} trade at entry {entry_price}: {valid_targets}"
                    )
            else:
                targets = str([])
                logger.warning(
                    f"No targets provided for {direction} trade at entry {entry_price}"
                )

            if status == "open":
                conn.execute(
                    """
                    INSERT INTO trades 
                        (channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent, status, activated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        normalized_channel_id,
                        coin.upper(),
                        direction,
                        targets,
                        leverage,
                        sl,
                        margin,
                        entry_price,
                        price,
                        status,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO trades 
                        (channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?)
                    """,
                    (
                        normalized_channel_id,
                        coin.upper(),
                        direction,
                        targets,
                        leverage,
                        sl,
                        margin,
                        entry_price,
                        price,
                        status,
                    ),
                )
            conn.commit()

        logger.success(
            f"Order placed: channel={channel_id} (normalized={normalized_channel_id}) coin={coin.upper()} side={direction} margin={margin:.2f} entry={entry_price} status={status}"
        )

        # Send websocket update after placing order
        _send_dashboard_update()

        return True

    except Exception as e:
        logger.error(f"Error placing order for {coin}: {e}")
        return False


def updater_thread_worker():
    """
    The main worker loop that runs in a separate thread. It periodically updates
    order statuses, calculates PnL, and prepares data for the GUI.
    """
    while True:
        try:
            db_path = os.getenv("DB_PATH", "total.db")
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

                # Ensure trading_stats table exists and has singleton row
                _ensure_trading_stats_singleton(conn)

                # Ensure trades table exists
                _ensure_trades_table(conn)

                # Get all trades from trades table with open or waiting status
                cur = conn.cursor()

                # First, check waiting trades for entry conditions
                cur.execute(
                    "SELECT trade_id, channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price FROM trades WHERE status = 'waiting'"
                )
                waiting_trades = cur.fetchall()
                for trade in waiting_trades:
                    (
                        trade_id,
                        channel_id,
                        coin,
                        direction,
                        targets,
                        leverage,
                        sl,
                        margin,
                        entry_price,
                        current_price,
                    ) = trade

                    price = get_price(coin)
                    if price is None:
                        continue

                    # Check if entry conditions are met
                    entry_hit = False
                    if direction.upper() == "LONG":
                        # For LONG: price must go down to or below entry price
                        entry_hit = price <= entry_price
                    else:  # SHORT
                        # For SHORT: price must go up to or above entry price
                        entry_hit = price >= entry_price

                    if entry_hit:
                        # Activate the trade
                        cur.execute(
                            "UPDATE trades SET status = 'open', activated_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                            (trade_id,),
                        )
                        logger.success(
                            f"Trade {trade_id} activated - Entry hit for {coin} at {price} (target entry: {entry_price})"
                        )
                    else:
                        # Update current price for waiting trades
                        cur.execute(
                            "UPDATE trades SET current_price = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                            (price, trade_id),
                        )

                # Now process open trades for PnL and target/SL checks
                cur.execute(
                    "SELECT trade_id, channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent FROM trades WHERE status = 'open'"
                )
                open_trades = cur.fetchall()
                for trade in open_trades:
                    (
                        trade_id,
                        channel_id,
                        coin,
                        direction,
                        targets,
                        leverage,
                        sl,
                        margin,
                        entry_price,
                        current_price,
                        pnl,
                        pnl_percent,
                    ) = trade
                    price = get_price(coin)
                    if price is None:
                        continue

                    # Calculate new PnL using trade-specific leverage if available
                    trade_leverage = leverage if leverage is not None else LEVERAGE
                    if direction.upper() == "LONG":
                        new_pnl = (
                            (price - entry_price)
                            * (margin * trade_leverage)
                            / entry_price
                        )
                        new_pnl_percent = (
                            ((price / entry_price) - 1) * trade_leverage * 100
                        )
                    else:  # SHORT
                        new_pnl = (
                            (entry_price - price)
                            * (margin * trade_leverage)
                            / entry_price
                        )
                        new_pnl_percent = (
                            ((entry_price / price) - 1) * trade_leverage * 100
                        )

                    # Update trade PnL
                    cur.execute(
                        "UPDATE trades SET pnl = ?, pnl_percent = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                        (new_pnl, new_pnl_percent, price, trade_id),
                    )

                    # Check for stop loss hit
                    sl_hit = False
                    if sl is not None:
                        if direction.upper() == "LONG" and price <= sl:
                            sl_hit = True
                        elif direction.upper() == "SHORT" and price >= sl:
                            sl_hit = True

                    if sl_hit:
                        # Close trade as loss
                        cur.execute(
                            "UPDATE trades SET status = 'closed', close_price = ?, closed_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                            (price, trade_id),
                        )
                        logger.info(
                            f"Trade {trade_id} closed due to stop loss hit. Coin: {coin}, Price: {price}, SL: {sl}"
                        )
                        continue

                    # Check for target hits (take profit)
                    if targets:
                        try:
                            # Parse targets string to list
                            import ast

                            targets_list = (
                                ast.literal_eval(targets)
                                if isinstance(targets, str)
                                else targets
                            )
                            if isinstance(targets_list, list) and targets_list:
                                # Filter and sort targets based on direction
                                if direction.upper() == "LONG":
                                    # For LONG: only keep targets above entry price, sort ascending (closest first)
                                    targets_list = [
                                        t
                                        for t in targets_list
                                        if float(t) > entry_price
                                    ]
                                    targets_list = sorted(targets_list, key=float)
                                else:  # SHORT
                                    # For SHORT: only keep targets below entry price, sort descending (closest first)
                                    targets_list = [
                                        t
                                        for t in targets_list
                                        if float(t) < entry_price
                                    ]
                                    targets_list = sorted(
                                        targets_list, key=float, reverse=True
                                    )

                                # Find the nearest target based on direction
                                target_hit = None
                                if direction.upper() == "LONG":
                                    # For LONG, check if price reached any target (price >= target)
                                    for target in targets_list:
                                        if price >= float(target):
                                            target_hit = float(target)
                                            break
                                else:  # SHORT
                                    # For SHORT, check if price reached any target (price <= target)
                                    for target in targets_list:
                                        if price <= float(target):
                                            target_hit = float(target)
                                            break

                                if target_hit is not None:
                                    # Get original targets to calculate portion size
                                    original_targets = (
                                        ast.literal_eval(targets)
                                        if isinstance(targets, str)
                                        else targets
                                    )
                                    original_targets_count = len(original_targets)

                                    # Calculate partial margin (equal distribution between all targets)
                                    partial_margin = margin / original_targets_count

                                    # Remove the hit target from the list
                                    targets_list.remove(target_hit)

                                    # Calculate partial profit for this target
                                    if direction.upper() == "LONG":
                                        partial_pnl = (
                                            (target_hit - entry_price)
                                            * (partial_margin * trade_leverage)
                                            / entry_price
                                        )
                                        partial_pnl_percent = (
                                            ((target_hit / entry_price) - 1)
                                            * trade_leverage
                                            * 100
                                        )
                                    else:  # SHORT
                                        partial_pnl = (
                                            (entry_price - target_hit)
                                            * (partial_margin * trade_leverage)
                                            / entry_price
                                        )
                                        partial_pnl_percent = (
                                            ((entry_price / target_hit) - 1)
                                            * trade_leverage
                                            * 100
                                        )

                                    # Clone the trade as closed with partial profit
                                    cur.execute(
                                        """
                                        INSERT INTO trades 
                                            (channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent, status, close_price, closed_at)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed', ?, CURRENT_TIMESTAMP)
                                        """,
                                        (
                                            channel_id,
                                            coin,
                                            direction,
                                            f"[{target_hit}]",  # Record which target was hit
                                            trade_leverage,
                                            sl,
                                            partial_margin,  # Use partial margin for the closed portion
                                            entry_price,
                                            target_hit,
                                            partial_pnl,
                                            partial_pnl_percent,
                                            target_hit,
                                        ),
                                    )

                                    # Update remaining margin in the original trade
                                    remaining_margin = margin - partial_margin

                                    # Move stop loss to breakeven after first target hit
                                    new_sl = sl
                                    if (
                                        len(targets_list) == original_targets_count - 1
                                    ):  # First target hit
                                        new_sl = entry_price  # Move to breakeven
                                        logger.info(
                                            f"Moving stop loss to breakeven for trade {trade_id}. New SL: {new_sl}"
                                        )

                                    logger.success(
                                        f"Target hit for trade {trade_id}. Coin: {coin}, Target: {target_hit}, Partial PnL: {partial_pnl:.2f}, Partial margin: {partial_margin:.2f}"
                                    )

                                    # If no more targets, close the original trade
                                    if not targets_list:
                                        cur.execute(
                                            "UPDATE trades SET status = 'closed', close_price = ?, closed_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                                            (target_hit, trade_id),
                                        )
                                        logger.info(
                                            f"All targets hit for trade {trade_id}. Trade fully closed."
                                        )
                                    else:
                                        # Update the original trade with remaining targets, margin and new SL
                                        cur.execute(
                                            "UPDATE trades SET targets = ?, margin = ?, sl = ? WHERE trade_id = ?",
                                            (
                                                str(targets_list),
                                                remaining_margin,
                                                new_sl,
                                                trade_id,
                                            ),
                                        )
                                        logger.info(
                                            f"Trade {trade_id} updated - remaining targets: {targets_list}, margin: {remaining_margin:.2f}, SL: {new_sl}"
                                        )

                        except (ValueError, SyntaxError, TypeError) as e:
                            logger.warning(
                                f"Could not parse targets for trade {trade_id}: {targets}. Error: {e}"
                            )

                # Update trading_stats singleton with aggregated data
                _update_trading_stats_singleton(conn)

                conn.commit()

                # Send websocket update after database changes
                _send_dashboard_update()

        except Exception as e:
            logger.error(
                f"An unexpected error occurred in the updater thread: {e}",
                exc_info=True,
            )

        time.sleep(2)  # Update interval


def _ensure_trading_stats_singleton(conn: sqlite3.Connection) -> None:
    """
    Ensure the trading_stats table exists and has exactly one row with id=1.
    This implements the singleton pattern for trading statistics.
    """
    cur = conn.cursor()

    # Ensure singleton row exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trading_stats (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            profit REAL DEFAULT 0.0,
            roi REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("INSERT OR IGNORE INTO trading_stats (id) VALUES (1)")


def _ensure_trades_table(conn: sqlite3.Connection) -> None:
    """
    Ensure the trades table exists with all required columns.
    """
    cur = conn.cursor()

    # Create channels table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create trades table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER,
            coin TEXT NOT NULL,
            direction TEXT NOT NULL,
            targets TEXT,
            leverage REAL,
            sl REAL,
            margin REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            pnl REAL DEFAULT 0.0,
            pnl_percent REAL DEFAULT 0.0,
            status TEXT DEFAULT 'open',
            close_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            activated_at TIMESTAMP,
            FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
        )
    """)

    # Add activated_at column if it doesn't exist (for backward compatibility)
    try:
        cur.execute("ALTER TABLE trades ADD COLUMN activated_at TIMESTAMP")
    except sqlite3.OperationalError:
        pass  # Column already exists


def _update_trading_stats_singleton(conn: sqlite3.Connection) -> None:
    """
    Update the singleton trading_stats row with current aggregated data.
    """
    cur = conn.cursor()

    # Aggregate stats from trades (exclude waiting trades from total count)
    cur.execute("SELECT COUNT(*) FROM trades WHERE status != 'waiting'")
    total_trades = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND pnl > 0")
    wins = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND pnl < 0")
    losses = cur.fetchone()[0] or 0

    cur.execute("SELECT IFNULL(SUM(pnl), 0) FROM trades WHERE status != 'waiting'")
    profit = cur.fetchone()[0] or 0.0

    denom = wins + losses
    win_rate = (wins / denom * 100.0) if denom > 0 else 0.0

    # ROI relative to START_BALANCE (percentage). If START_BALANCE <= 0, ROI = 0
    roi = (profit / START_BALANCE * 100.0) if START_BALANCE > 0 else 0.0

    # Update singleton row
    cur.execute(
        """
        UPDATE trading_stats SET
            total_trades = ?,
            wins = ?,
            losses = ?,
            win_rate = ?,
            profit = ?,
            roi = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
        """,
        (total_trades, wins, losses, win_rate, profit, roi),
    )


def _send_dashboard_update() -> None:
    """
    Send dashboard update via WebSocket to all connected clients.
    Includes debouncing to avoid too frequent updates.
    """
    global _last_websocket_update

    current_time = time.time()
    if current_time - _last_websocket_update < _WEBSOCKET_UPDATE_INTERVAL:
        return  # Skip update if too soon

    try:
        from .views import _get_dashboard_data
        import asyncio

        # Get current dashboard data
        dashboard_data = _get_dashboard_data()

        # Send via channel layer - handle both sync and async contexts
        channel_layer = get_channel_layer()
        if channel_layer:
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, schedule the task
                    asyncio.create_task(
                        channel_layer.group_send(
                            "dashboard",
                            {
                                "type": "dashboard_update",
                                "message": dashboard_data,
                            },
                        )
                    )
                else:
                    # No running loop, use async_to_sync
                    async_to_sync(channel_layer.group_send)(
                        "dashboard",
                        {
                            "type": "dashboard_update",
                            "message": dashboard_data,
                        },
                    )
            except RuntimeError:
                # No event loop, use async_to_sync
                async_to_sync(channel_layer.group_send)(
                    "dashboard",
                    {
                        "type": "dashboard_update",
                        "message": dashboard_data,
                    },
                )

            _last_websocket_update = current_time
            logger.debug("Dashboard update sent via WebSocket")
    except Exception as e:
        logger.error(f"Error sending dashboard update via WebSocket: {e}")
