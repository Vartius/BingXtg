# Trading Bot

This is a trading bot designed to automate trading strategies using Telegram channels as signal sources. It parses messages from specified Telegram channels, sets trading orders, and updates a GUI table with the current status of the trades. The bot supports both live trading and simulation modes.

## Features

- **Telegram Integration**: Parses trading signals from specified Telegram channels.
- **Trading Automation**: Automatically places and manages trading orders on BingX exchange.
- **Simulation Mode**: Allows testing strategies without real money.
- **Real-Time Data**: Updates orders and balance in real-time.
- **GUI Interface**: Displays current trades, balance, and win rate using a Tkinter-based table.

## Requirements

- Python 3.8+
- Required Python packages listed in `requirements.txt`
- BingX API credentials
- Telegram API credentials

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Vartius/BingXtg.git
    cd BingXtg
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure the bot**:
    - Create a `config.py` file with your BingX and Telegram API credentials and other configuration parameters.
    - Example `config.py`:
      ```python
      START_BALANCE = 1000
      MIN_ORDERS_TO_HIGH = 5
      MAX_PERCENT = 0.05
      LEVERAGE = 10
      SECRETKEY = "your_secret_key"
      APIKEY = "your_api_key"
      TP = 0.05
      SL = -0.03
      APIURL = "https://api.bingx.com"
      api_id = "your_telegram_api_id"
      api_hash = "your_telegram_api_hash"
      ```

4. **Prepare data files**:
    - Ensure the following JSON files are present in `src/data/` directory:
      - `curdata.json`
      - `channels.json`
      - `winrate.json`
      - `table.json`
    - Example JSON structures:
      - `channels.json`:
        ```json
        {
          "channel_id": {
            "name": "Channel Name",
            "regex": "coin_regex",
            "long": "long_trigger_word",
            "short": "short_trigger_word",
            "do": true
          }
        }
        ```
      - `curdata.json`, `winrate.json`, and `table.json` can be initially empty or set with appropriate data structures as per the bot's requirements.

## Running the Bot

1. **Start the bot**:
    ```bash
    python main.py
    ```

2. **Choose an option**:
    - `1`: Start live trading
    - `2`: Start simulation
    - `0` or any other key: Exit

## Files Overview

- **main.py**: Entry point of the bot. Initializes the simulation or live trading mode.
- **bingx.py**: Handles interaction with BingX API for placing orders and fetching prices.
- **tableviewer.py**: Manages the GUI table that displays current trades and balances.
- **text.py**: Parses text messages from Telegram channels to extract trading signals.
- **tgparser.py**: Integrates with Telegram API to receive messages from specified channels and triggers trading actions.

## Logging

- The bot uses `loguru` for logging. Logs are saved to `logs.log`.

## Usage Notes

- Ensure your `config.py` and JSON data files are properly configured before running the bot.
- The bot updates the GUI table and logs in real-time. Check `logs.log` for detailed logs of bot activity.
- Simulation mode is useful for testing strategies without risking real money.

## Contributing

- Feel free to open issues or submit pull requests to improve the bot.
- Ensure code quality and consistency with existing project structure and style.

## License

- This project is licensed under the MIT License. See the `LICENSE` file for details.