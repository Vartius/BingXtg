import json
from django.shortcuts import render
from django.http import JsonResponse
from loguru import logger


def dashboard(request):
    """Renders the main dashboard page with initial data."""
    initial_data = {}
    try:
        with open("data/table.json", "r", encoding="utf-8") as f:
            initial_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Provide a default structure if the file doesn't exist or is invalid
        initial_data = {
            "orders": [],
            "balance": 0.0,
            "available_balance": 0.0,
            "winrate": 0.0,
        }
        logger.warning(
            "table.json not found or invalid. Loading dashboard with empty data."
        )

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
        with open("data/table.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return JsonResponse(data)
    except FileNotFoundError:
        # Return default structure if file doesn't exist
        default_data = {
            "orders": [],
            "balance": 0.0,
            "available_balance": 0.0,
            "winrate": 0.0,
        }
        logger.warning("table.json not found. Returning default data.")
        return JsonResponse(default_data)
    except json.JSONDecodeError:
        logger.error("table.json contains invalid JSON. Returning default data.")
        default_data = {
            "orders": [],
            "balance": 0.0,
            "available_balance": 0.0,
            "winrate": 0.0,
        }
        return JsonResponse(default_data)
    except Exception as e:
        logger.error(f"Unexpected error in dashboard_data: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)
