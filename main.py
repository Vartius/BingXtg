import sys
from loguru import logger
from src.tg_parser import start_telegram_parser
from src.data_handler import (
    get_state,
    save_state,
    get_channels,
    get_winrate,
    save_winrate,
)

# Safely import configuration
try:
    from config import START_BALANCE
except ImportError:
    logger.critical(
        "Could not import `config.py`. Please rename `config.example.py` to `config.py` and fill it out."
    )
    sys.exit(1)


def initialize_new_state():
    """Initializes and returns a fresh state dictionary for a new session."""
    logger.info("Initializing a new trading session.")
    return {
        "balance": START_BALANCE,
        "available_balance": START_BALANCE,
        "winrate": 0,
        "orders": {},
    }


def continue_existing_state(state):
    """Prepares an existing state for a continued session."""
    logger.info(f"Continuing existing session with balance: {state['balance']:.2f}")
    state["available_balance"] = state["balance"]
    state["orders"] = {}  # Clear stale orders from previous run
    return state


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

    if not isinstance(state, dict):
        logger.critical("State is not a dictionary. Please check your state file.")
        sys.exit(1)
    save_state(state)
    save_winrate(winrate)

    logger.info("Session setup complete. Starting bot...")
    start_telegram_parser(is_simulation)


def main():
    """Main entry point for the application."""
    logger.add("logs.log", rotation="10 MB", compression="zip")

    print("\n--- Trading Bot Menu ---\n")
    print("  1: Start Live Trading")
    print("  2: Start Simulation")
    print("  0: Exit\n")

    choice = input("Select an option: ")

    if choice == "1":
        simulate = False
        setup_session(is_simulation=simulate)
    elif choice == "2":
        simulate = True
        setup_session(is_simulation=simulate)
    else:
        logger.info("Exiting application.")
        sys.exit(0)


if __name__ == "__main__":
    main()
