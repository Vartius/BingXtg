#!/usr/bin/env python3
"""
Debug script to check data loading.
"""

import os
import sys
import django
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bingxtg_project.settings")
django.setup()

from loguru import logger
from apps.trading_bot.data_handler import get_state, get_winrate, get_channels

if __name__ == "__main__":
    logger.info("Testing data loading...")

    state = get_state()
    logger.info(f"State type: {type(state)}, Content preview: {str(state)[:200]}...")

    winrate = get_winrate()
    logger.info(
        f"Winrate type: {type(winrate)}, Content preview: {str(winrate)[:200]}..."
    )

    channels = get_channels()
    logger.info(
        f"Channels type: {type(channels)}, Content preview: {str(channels)[:200]}..."
    )

    logger.info("Data loading test complete.")
