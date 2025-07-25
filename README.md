# Telegram Signal Trading Bot for BingX

This is a Python-based trading bot that automates trading on the BingX exchange by parsing signals from Telegram channels. It features a real-time GUI, live trading, and simulation modes.

## ✨ Features

-   **Telegram Integration**: Parses trading signals from specified public or private Telegram channels.
-   **Automated Trading**: Automatically places market orders on the BingX Perpetual Swap market.
-   **Live & Simulation Modes**: Test your strategies with a paper trading mode or run it live with real assets.
-   **Dynamic Investment Strategy**: Adjusts the capital allocated to a trade based on the historical win rate of the signal channel.
-   **Real-Time GUI**: A PyQt6-based interface displays current trades, PnL, account balance, and overall win rate.
-   **Command Interface**: Manage the bot via commands in your private Telegram chat (e.g., check status, view data).

## ⚠️ Disclaimer

Trading cryptocurrency involves significant risk. This bot is provided as-is, and the authors are not responsible for any financial losses. Always test thoroughly in simulation mode before trading with real money.

## 🛠️ Requirements

-   Python 3.11+
-   A Telegram account and API credentials.
-   A BingX account and API credentials.
-   [uv](https://github.com/astral-sh/uv) (recommended for fast package installation)
-   [Docker](https://www.docker.com/) (optional, for containerized deployment)

## 🚀 Setup and Configuration

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Vartius/BingXtg.git
    cd BingXtg
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    # Using uv (recommended)
    uv pip install -r requirements.txt

    # Or using pip
    pip install -r requirements.txt
    ```

4.  **Configure the Bot**:
    -   Rename `config.example.py` to `config.py`.
    -   Open `config.py` and fill in your BingX and Telegram API credentials.
    -   Adjust the trading parameters like `LEVERAGE`, `TP` (Take Profit), and `SL` (Stop Loss) to fit your strategy.

5.  **Configure Signal Channels**:
    -   Edit `src/data/channels.json`.
    -   For each channel you want to parse, add an entry using its Telegram Chat ID as the key.
    -   Define the `regex` to capture the coin ticker (e.g., `(BTC)`), and the keywords that trigger a `long` or `short` trade.
    -   Set `"do": true` to enable parsing for that channel.

    **Example `channels.json`**:
    ```json
    {
      "-1001234567890": {
        "name": "My Signal Channel",
        "regex": "^\\w* (\\w*)",
        "long": "LONG",
        "short": "SHORT",
        "do": true
      }
    }
    ```

6.  **Prepare Data Files**:
    -   The `src/data/` directory contains the bot's state and data. The required files (`state.json`, `winrate.json`, `table.json`) will be created automatically on the first run if they don't exist.

## ▶️ Running the Bot

Once configured, you can start the bot from your terminal.

1.  **Run the Main Script**:
    ```bash
    python main.py
    ```

2.  **Choose an Option**:
    -   `1`: Start live trading with your BingX account.
    -   `2`: Start a simulation using the `START_BALANCE` from your config.
    -   `0`: Exit the application.

## 🐳 Docker Usage

You can build and run the bot in a Docker container for isolated and consistent execution.

1.  **Build the Docker Image**:
    ```bash
    docker build -t trading-bot .
    ```

2.  **Run the Docker Container**:
    > **Note**: The GUI is disabled in Docker mode. You can manage the bot via Telegram commands.
    ```bash
    # Create a volume to persist data
    docker volume create trading-bot-data

    # Run the container
    docker run -it --rm \
      -v trading-bot-data:/app/src/data \
      --name my-trading-bot \
      trading-bot
    ```

## 📂 Project Structure

```
├── src/
│   ├── data/                 # Bot state and channel configs
│   │   ├── channels.json     # Channel signal definitions
│   │   ├── state.json        # Current balance and open orders
│   │   └── ...
│   ├── bingx_api.py          # Handles BingX API communication
│   ├── command_handler.py    # Logic for Telegram bot commands
│   ├── data_handler.py       # Manages reading/writing JSON data
│   ├── order_handler.py      # Core logic for placing and updating orders
│   ├── tableviewer.py        # PyQt6 GUI for real-time data
│   ├── text_parser.py        # Extracts signals from messages
│   └── tg_parser.py          # Telegram client and message routing
├── config.example.py         # Configuration template
├── main.py                   # Main entry point of the application
├── requirements.txt          # Python dependencies
└── Dockerfile                # Docker configuration```

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue to discuss a new feature or submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```