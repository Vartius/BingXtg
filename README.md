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
    -   Copy `.env.example` to `.env`.
    -   Open `.env` and fill in your BingX and Telegram API credentials.
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

## 🚀 Quick Start

1.  **Install Dependencies**:
    ```bash
    # Using uv (recommended)
    uv sync
    
    # Or using pip
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env with your API credentials
    ```

3.  **Run Database Migrations**:
    ```bash
    python manage.py migrate
    ```

4.  **Start the Django Development Server**:
    ```bash
    python manage.py runserver
    ```

5.  **Access the Web Interface**:
    - Trading Dashboard: http://localhost:8000/
    - Admin Panel: http://localhost:8000/admin/

6.  **Start the Trading Bot** (in a separate terminal):
    ```bash
    python manage.py start_bot
    ```

## 🧠 AI Model Training

All AI model training is now done exclusively in the `ai.ipynb` Jupyter notebook. The Django application uses the trained models for inference only.

### Training Process:
1. Open `ai.ipynb` in Jupyter Lab or VS Code
2. Follow the notebook cells to train both classifier and NER models
3. Models are automatically saved to the `ai_model/` directory
4. The Django application will automatically use the trained models

## 🐳 Docker Usage

You can build and run the bot in a Docker container for isolated and consistent execution.

1.  **Build the Docker Image**:
    ```bash
    docker build -t bingxtg .
    ```

2.  **Run the Docker Container**:
    ```bash
    # Create a volume to persist data
    docker volume create bingxtg-data

    # Run the container
    docker run -it --rm \
      -v bingxtg-data:/app/data \
      -p 8000:8000 \
      --name bingxtg \
      bingxtg
    ```

## 📂 Project Structure

The project follows Django best practices with a modular app structure:

```
BingXtg/
├── apps/                     # Django applications
│   ├── telegram_client/      # Telegram integration
│   └── trading_bot/          # Main trading bot functionality
├── bingxtg_project/          # Main Django project configuration
├── utils/                    # Shared utilities and business logic
├── static/                   # Static files (CSS, JS, images)
├── templates/                # Django templates
├── data/                     # Configuration and data files
├── ai.ipynb                  # AI model training notebook
├── ai_model/                 # Trained AI models (classifier & NER)
├── manage.py                 # Django management script
└── requirements.txt          # Python dependencies
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.

## 🎮 Web Interface

The project includes a modern web interface built with Django:

### Trading Dashboard
- Real-time trading status and PnL
- Live order monitoring
- Performance analytics
- WebSocket-powered real-time updates

### Admin Panel
- User management
- Database administration
- System configuration

## 🤖 Management Commands

The project includes several Django management commands:

```bash
# Start the trading bot
python manage.py start_bot

# Extract messages from Telegram
python manage.py extract_messages

# Train AI models
python manage.py train_models

# Database maintenance
python manage.py migrate
python manage.py collectstatic
```

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue to discuss a new feature or submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```