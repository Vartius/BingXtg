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
        "trading_dashboard.html",
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
            "balance": 0.0,   # Total balance from BingX
            "available_balance": 0.0,  # Available balance from BingX
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

        # Get balance from BingX API
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

            # Get open trades for orders list
            cur.execute("""
                SELECT 
                    trade_id, 
                    channel_id, 
                    coin, 
                    direction, 
                    margin, 
                    entry_price, 
                    current_price, 
                    pnl, 
                    pnl_percent,
                    updated_at
                FROM trades 
                WHERE status = 'open'
                ORDER BY updated_at DESC
            """)

            orders = []
            for row in cur.fetchall():
                # Convert to array format expected by JavaScript
                # [trade_id, channel_id, coin, direction, margin, entry_price, current_price, pnl, pnl_percent]
                orders.append(
                    [
                        row[0],  # trade_id
                        row[1],  # channel_id
                        row[2],  # coin
                        row[3],  # direction
                        f"{row[4]:.2f}",  # margin (formatted)
                        f"{row[5]:.4f}",  # entry_price (formatted)
                        f"{row[6]:.4f}",  # current_price (formatted)
                        f"{row[7]:.2f}",  # pnl (formatted)
                        f"{row[8]:.2f}",  # pnl_percent (formatted)
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
        }
