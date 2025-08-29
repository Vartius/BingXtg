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
    MAX_PERCENT = float(os.getenv("MAX_PERCENT", "0.3"))
    LEVERAGE = int(os.getenv("LEVERAGE", "20"))
    TP = float(os.getenv("TP", "0.2"))
    SL = float(os.getenv("SL", "-0.5"))
    # Used for ROI calculation in trading_stats
    START_BALANCE = float(os.getenv("START_BALANCE", "0"))
except (ValueError, TypeError) as e:
    logger.critical(f"Configuration error: {e}. Please check your .env file.")
    sys.exit(1)


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

        direction = str(data.get("direction"))
        coin = str(data.get("pair"))
        # Fetch current price first; required for both live and simulation
        price = get_price(coin)
        if price is None:
            return False

        # Resolve DB path and channel id
        db_path = os.getenv("DB_PATH", "total.db")
        # Compute dynamic position sizing based on per-channel historical winrate from SQL
        wins = losses = 0
        if channel_id is not None:
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
                    (channel_id,),
                )
                row = cur.fetchone() or (0, 0)
                wins = int(row[0] or 0)
                losses = int(row[1] or 0)

        total_trades = wins + losses
        winrate_factor = (
            (wins / total_trades) if total_trades > MIN_ORDERS_TO_HIGH else 0.0
        )

        # Base investment is 1% of balance, with a bonus up to MAX_PERCENT based on winrate
        investment_percent = 0.01 + (MAX_PERCENT - 0.01) * winrate_factor

        # Determine margin from BingX available balance (fallback to START_BALANCE if unavailable)
        _, available_balance = get_balance()
        if available_balance is None:
            available_balance = START_BALANCE
        margin = float(available_balance) * float(investment_percent)
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
                "INSERT OR IGNORE INTO channels (channel_id) VALUES (?)", (channel_id,)
            )

            conn.execute(
                """
                INSERT INTO trades 
                    (channel_id, coin, direction, margin, entry_price, current_price, pnl, pnl_percent, status)
                VALUES (?, ?, ?, ?, ?, ?, 0.0, 0.0, 'open')
                """,
                (channel_id, coin.upper(), direction, margin, price, price),
            )
            conn.commit()

        logger.success(
            f"Order placed: channel={channel_id} coin={coin.upper()} side={direction} margin={margin:.2f} entry={price}"
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

                # Get all trades from trades table with open status
                cur = conn.cursor()
                cur.execute(
                    "SELECT trade_id, channel_id, coin, direction, margin, entry_price, current_price, pnl, pnl_percent FROM trades WHERE status = 'open'"
                )
                open_trades = cur.fetchall()
                for trade in open_trades:
                    (
                        trade_id,
                        channel_id,
                        coin,
                        direction,
                        margin,
                        entry_price,
                        current_price,
                        pnl,
                        pnl_percent,
                    ) = trade
                    price = get_price(coin)
                    if price is None:
                        continue

                    # Calculate new PnL
                    if direction == "LONG":
                        new_pnl = (
                            (price - entry_price) * (margin * LEVERAGE) / entry_price
                        )
                        new_pnl_percent = (price / entry_price - 1) * LEVERAGE
                    else:  # SHORT
                        new_pnl = (
                            (entry_price - price) * (margin * LEVERAGE) / entry_price
                        )
                        new_pnl_percent = (entry_price / price - 1) * LEVERAGE

                    # Update trade PnL
                    cur.execute(
                        "UPDATE trades SET pnl = ?, pnl_percent = ?, current_price = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                        (new_pnl, new_pnl_percent, price, trade_id),
                    )

                    # check tp/sl hit
                    if new_pnl_percent >= TP or new_pnl_percent <= SL:
                        # close trade
                        logger.info(
                            f"Auto-closing trade {trade_id} for {coin} at price {price} due to TP/SL hit."
                        )
                        cur.execute(
                            "UPDATE trades SET status = 'closed', close_price = ?, closed_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                            (price, trade_id),
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
            FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
        )
    """)


def _update_trading_stats_singleton(conn: sqlite3.Connection) -> None:
    """
    Update the singleton trading_stats row with current aggregated data.
    """
    cur = conn.cursor()

    # Aggregate stats from trades
    cur.execute("SELECT COUNT(*) FROM trades")
    total_trades = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND pnl > 0")
    wins = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed' AND pnl < 0")
    losses = cur.fetchone()[0] or 0

    cur.execute("SELECT IFNULL(SUM(pnl), 0) FROM trades")
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

        # Get current dashboard data
        dashboard_data = _get_dashboard_data()

        # Send via channel layer
        channel_layer = get_channel_layer()
        if channel_layer:
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
