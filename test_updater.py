#!/usr/bin/env python3
"""
Test script to run just the updater thread to verify dashboard updates.
"""

import os
import sys
import django
import time
from threading import Thread
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bingxtg_project.settings")
django.setup()

from loguru import logger
from apps.trading_bot.order_handler import updater_thread_worker

if __name__ == "__main__":
    logger.info("Starting updater thread test...")

    # Start the updater thread
    updater_thread = Thread(target=updater_thread_worker, daemon=False)
    updater_thread.start()

    logger.info("Updater thread started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping updater thread test...")
        sys.exit(0)
