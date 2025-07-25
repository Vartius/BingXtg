#
# --- Configuration for the Trading Bot ---
#
# 1. RENAME this file from `config.example.py` to `config.py`.
# 2. FILL IN your API credentials and preferred settings.
# 3. DO NOT share your `config.py` file or commit it to version control.
#

# --- BingX API Credentials ---
# Create these at https://bingx.com/en-us/account/api/
APIURL = "https://open-api.bingx.com"
APIKEY = "YOUR_BINGX_API_KEY"
SECRETKEY = "YOUR_BINGX_SECRET_KEY"

# --- Telegram API Credentials ---
# Get these from my.telegram.org
API_ID = 12345678
API_HASH = "YOUR_TELEGRAM_API_HASH"

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
