import sys
import os
from django.core.management.base import BaseCommand
from loguru import logger
from dotenv import load_dotenv
from apps.trading_bot.data_handler import (
    get_state,
    save_state,
    get_channels,
    get_winrate,
    save_winrate,
)
from apps.trading_bot.tg_parser import start_telegram_parser

# Load environment variables
load_dotenv()

# Safely import configuration
try:
    START_BALANCE = float(os.getenv("START_BALANCE", "10.0"))
except (ValueError, TypeError) as e:
    logger.critical(f"Configuration error: {e}. Please check your .env file.")
    sys.exit(1)


# !CHECK AI GENERATED BULLSHIT
def initialize_new_state():
    """Initializes and returns a fresh state dictionary for a new session."""
    logger.info("Initializing a new trading session.")
    return {
        "balance": START_BALANCE,
        "available_balance": START_BALANCE,
        "winrate": 0,
        "orders": {},
    }


# !CHECK AI GENERATED BULLSHIT
def continue_existing_state(state):
    """Prepares an existing state for a continued session."""
    logger.info(f"Continuing existing session with balance: {state['balance']:.2f}")
    state["available_balance"] = state["balance"]
    state["orders"] = {}  # Clear stale orders from previous run
    return state


# !CHECK AI GENERATED BULLSHIT
def setup_session(is_simulation: bool):
    """
    Sets up the user session by either loading existing data or creating a new session.
    It also ensures all necessary data files are present and valid.
    """
    state = get_state()
    channels = get_channels()
    winrate = get_winrate()

    # Validate required channels configuration
    if channels is None:
        logger.critical(
            "`channels.json` not found or invalid. Please create it. See README for an example."
        )
        sys.exit(1)

    # Initialize winrate file if it doesn't exist or is not a dict
    if winrate is None or not isinstance(winrate, dict):
        logger.warning("`winrate.json` not found or invalid, creating a new one.")
        winrate = {}

    # Decide whether to start a new session or continue
    if state and "balance" in state:
        choice = input(
            "Found previous session data. Do you want to start a new session? (Y/N): "
        ).lower()
        if choice == "y":
            state = initialize_new_state()
        else:
            state = continue_existing_state(state)
    else:
        state = initialize_new_state()

    # Clean up winrate and state data to only include currently configured channels
    # Remove any channel IDs from winrate that are not in current channels config
    channels_to_remove = [ch_id for ch_id in winrate.keys() if ch_id not in channels]
    for channel_id in channels_to_remove:
        logger.info(f"Removing obsolete channel {channel_id} from winrate data")
        del winrate[channel_id]

    # Ensure winrate and order structures are initialized for each channel
    for channel_id in channels:
        if channel_id not in winrate:
            winrate[channel_id] = {
                "name": channels[channel_id].get("name", "Unknown"),
                "win": 0,
                "lose": 0,
            }
        if not isinstance(state, dict):
            logger.critical("State is not a dictionary. Please check your state file.")
            sys.exit(1)
        if "orders" not in state or not isinstance(state["orders"], dict):
            state["orders"] = {}
        state["orders"][channel_id] = {}

    # Clean up state orders to only include currently configured channels
    if (
        isinstance(state, dict)
        and "orders" in state
        and isinstance(state["orders"], dict)
    ):
        orders_to_remove = [
            ch_id for ch_id in state["orders"].keys() if ch_id not in channels
        ]
        for channel_id in orders_to_remove:
            logger.info(f"Removing obsolete channel {channel_id} from state orders")
            del state["orders"][channel_id]

    if not isinstance(state, dict):
        logger.critical("State is not a dictionary. Please check your state file.")
        sys.exit(1)
    save_state(state)
    save_winrate(winrate)

    logger.info("Session setup complete. Starting bot...")
    start_telegram_parser(is_simulation)


class Command(BaseCommand):
    help = (
        "Starts the trading bot, including the Telegram client and background updater."
    )

    def handle(self, *args, **options):
        logger.add("logs.log", rotation="10 MB", compression="zip")

        print("\n--- Trading Bot Menu ---\n")
        print("  1: Start Live Trading")
        print("  2: Start Simulation")
        print("  0: Exit\n")

        choice = input("Select an option: ")

        if choice == "1":
            setup_session(is_simulation=False)
        elif choice == "2":
            setup_session(is_simulation=True)
        else:
            logger.info("Exiting application.")
            sys.exit(0)
