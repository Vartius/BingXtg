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
        "bot/dashboard.html",
        {
            "initial_data": json.dumps(initial_data)  # Pass as a JSON string
        },
    )


def dashboard_data(request):
    """
    This view is no longer used by the WebSocket frontend, but we can keep it
    for debugging or for a potential REST API in the future.
    """
    try:
        with open("data/table.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return JsonResponse(data)
    except FileNotFoundError:
        return JsonResponse({"error": "Data file not found."}, status=404)
