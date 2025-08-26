#!/usr/bin/env python3
"""
Debug script to test the update display data function.
"""

import os
import sys
import django
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bingxtg_project.settings")
django.setup()

from loguru import logger
from apps.trading_bot.data_handler import get_state, get_winrate, get_channels
from apps.trading_bot.order_handler import _update_display_data

if __name__ == "__main__":
    logger.info("Testing _update_display_data function...")

    state = get_state()
    winrate_data = get_winrate()
    channels = get_channels()

    # Convert to dict if they are lists (same logic as in updater_thread_worker)
    if isinstance(state, list):
        if len(state) > 0 and isinstance(state[0], dict):
            state = state[0]
        else:
            state = {}
    if isinstance(winrate_data, list):
        winrate_data = {
            item.get("channel_id"): item
            for item in winrate_data
            if isinstance(item, dict) and "channel_id" in item
        }
    if isinstance(channels, list):
        channels = {
            item.get("channel_id"): item
            for item in channels
            if isinstance(item, dict) and "channel_id" in item
        }

    # Ensure we have valid dictionaries
    if state is None:
        state = {}
    if winrate_data is None:
        winrate_data = {}
    if channels is None:
        channels = {}

    logger.info("Calling _update_display_data...")
    _update_display_data(state, channels, winrate_data)
    logger.info("_update_display_data call complete.")

    # Check if table.json was updated
    with open("data/table.json", "r") as f:
        table_data = json.load(f)
        logger.info(f"Updated table.json: {table_data}")
