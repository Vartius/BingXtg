#
# --- DEPRECATED: Configuration moved to .env file ---
#
# This file is deprecated. Configuration has been moved to environment variables.
#
# 1. COPY `.env.example` to `.env`
# 2. EDIT `.env` and fill in your API credentials and preferred settings.
# 3. DO NOT share your `.env` file or commit it to version control.
#
# The application now uses python-dotenv to load configuration from .env file.
# See .env.example for all available configuration options.
#

# This file is kept for reference only and is no longer used by the application.

# --- BingX API Credentials (now in .env as APIKEY, SECRETKEY) ---
# Create these at https://bingx.com/en-us/account/api/
APIURL = "https://open-api.bingx.com"  # now APIURL in .env
APIKEY = "YOUR_BINGX_API_KEY"  # now APIKEY in .env
SECRETKEY = "YOUR_BINGX_SECRET_KEY"  # now SECRETKEY in .env

# --- Telegram API Credentials (now in .env as API_ID, API_HASH) ---
# Get these from my.telegram.org
API_ID = 12345678  # now API_ID in .env
API_HASH = "YOUR_TELEGRAM_API_HASH"  # now API_HASH in .env

# --- Trading Parameters ---
# The leverage to be used for all trades.
LEVERAGE = 20
# Stop Loss as a negative percentage (e.g., -0.5 is a 0.5% loss from entry).
SL = -0.5
# Take Profit as a positive percentage (e.g., 0.2 is a 0.2% profit from entry).
TP = 0.2

# --- Simulation Parameters ---
# The starting balance for simulation mode.
START_BALANCE = 1000

# --- Strategy Parameters ---
# The minimum number of trades a channel needs before its win rate is used for dynamic position sizing.
MIN_ORDERS_TO_HIGH = 20
# The maximum percentage of available balance to risk on a single trade from a high-win-rate channel.
MAX_PERCENT = 0.3
