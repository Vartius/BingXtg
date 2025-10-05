import json
import sqlite3
import os
from django.shortcuts import render
from django.http import JsonResponse
from loguru import logger
from .bingx_api import get_balance


def dashboard(request):
    """Renders the main dashboard page with initial data."""
    initial_data = _get_dashboard_data()

    return render(
        request,
        "index.html",
        {
            "initial_data": json.dumps(initial_data)  # Pass as a JSON string
        },
    )


def dashboard_data(request):
    """
    REST API endpoint that returns the current dashboard data.
    Used as a fallback when WebSocket is not available.
    """
    try:
        data = _get_dashboard_data()
        return JsonResponse(data)
    except Exception as e:
        logger.error(f"Unexpected error in dashboard_data: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)


def _get_dashboard_data():
    """
    Helper function to fetch dashboard data from the SQLite database.

    Returns:
        Dictionary containing dashboard data with the following structure:
        {
            "orders": [...],  # List of open trades
            "balance": 0.0,   # Total balance (BingX API or simulated)
            "available_balance": 0.0,  # Available balance (BingX API or simulated)
            "winrate": 0.0,   # Win rate from trading_stats
            "total_trades": 0,  # Total trades count
            "wins": 0,        # Number of wins
            "losses": 0,      # Number of losses
            "profit": 0.0,    # Total profit/loss
            "roi": 0.0        # Return on investment
        }
    """
    try:
        db_path = os.getenv("DB_PATH", "total.db")
        start_balance = float(os.getenv("START_BALANCE", "0"))

        # Check if we're in simulation mode by looking at app_state
        is_simulation = False
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Create app_state table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM app_state WHERE key = ?", ("simulation_mode",)
            )
            row = cur.fetchone()
            is_simulation = row and row[0] == "true"

        # Get balance based on mode
        if is_simulation:
            # Simulation mode: calculate balance from START_BALANCE + closed trades P&L
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()

                # Get total P&L from closed trades
                cur.execute(
                    "SELECT IFNULL(SUM(pnl), 0) FROM trades WHERE status = 'closed'"
                )
                total_pnl = cur.fetchone()[0] or 0.0

                # Calculate current balance (START_BALANCE + total P&L)
                balance = start_balance + total_pnl

                # Calculate used margin from open orders
                cur.execute(
                    "SELECT IFNULL(SUM(margin), 0) FROM trades WHERE status IN ('open', 'waiting')"
                )
                used_margin = cur.fetchone()[0] or 0.0

                # Available balance = current balance - used margin
                available_balance = balance - used_margin

                # Ensure available balance doesn't go negative
                available_balance = max(0.0, available_balance)
        else:
            # Live mode: get balance from BingX API
            balance, available_balance = get_balance()
            if balance is None:
                balance = 0.0
            if available_balance is None:
                available_balance = 0.0

        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Ensure trading_stats singleton exists
            conn.execute("""
                INSERT OR IGNORE INTO trading_stats (id) VALUES (1)
            """)

            cur = conn.cursor()

            # Get open and waiting trades for orders list
            cur.execute("""
                SELECT 
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
                    status,
                    updated_at
                FROM trades 
                WHERE status IN ('open', 'waiting')
                ORDER BY 
                    CASE status WHEN 'waiting' THEN 0 WHEN 'open' THEN 1 END,
                    updated_at DESC
            """)

            orders = []
            for row in cur.fetchall():
                # Convert to array format expected by JavaScript
                # [trade_id, channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent, status]
                status = row[12]  # status column

                orders.append(
                    [
                        row[0],  # trade_id
                        row[1],  # channel_id
                        row[2],  # coin
                        row[3],  # direction
                        row[4] if row[4] else "-",  # targets
                        f"{row[5]:.0f}x"
                        if row[5]
                        else "-",  # leverage (formatted as integer with 'x')
                        f"{row[6]:.4f}" if row[6] else "-",  # sl (formatted)
                        f"{row[7]:.2f}",  # margin (formatted)
                        f"{row[8]:.4f}",  # entry_price (formatted)
                        f"{row[9]:.4f}",  # current_price (formatted)
                        f"{row[10]:.2f}"
                        if status == "open"
                        else "0.00",  # pnl (formatted, 0 for waiting)
                        f"{row[11]:.2f}"
                        if status == "open"
                        else "0.00",  # pnl_percent (formatted, 0 for waiting)
                        status.upper(),  # status (uppercase for display)
                    ]
                )

            # Get trading stats from singleton table
            cur.execute("""
                SELECT 
                    total_trades,
                    wins,
                    losses,
                    win_rate,
                    profit,
                    roi
                FROM trading_stats 
                WHERE id = 1
            """)

            stats_row = cur.fetchone()
            if stats_row:
                total_trades, wins, losses, win_rate, profit, roi = stats_row
            else:
                # Fallback if stats table is empty
                total_trades = wins = losses = 0
                win_rate = profit = roi = 0.0

            return {
                "orders": orders,
                "balance": float(balance),
                "available_balance": float(available_balance),
                "winrate": float(win_rate),
                "total_trades": int(total_trades),
                "wins": int(wins),
                "losses": int(losses),
                "profit": float(profit),
                "roi": float(roi),
                "is_simulation": is_simulation,
            }

    except Exception as e:
        logger.error(f"Error fetching dashboard data from database: {e}")
        # Return default structure on error
        return {
            "orders": [],
            "balance": 0.0,
            "available_balance": 0.0,
            "winrate": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "roi": 0.0,
            "is_simulation": False,
        }
