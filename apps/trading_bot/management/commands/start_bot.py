import sys
from django.core.management.base import BaseCommand
from loguru import logger
from apps.trading_bot.tg_parser import start_telegram_parser
from core.config.validation import check_all


def setup_session(is_simulation: bool):
    """
    Sets up the user session by either loading existing data or creating a new session.
    It also ensures all necessary data files are present and valid.
    """

    check_all()
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
