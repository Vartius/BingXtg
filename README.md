# Telegram Signal Trading Bot for BingX

This is a Django-based trading platform that automates trades on the BingX exchange by parsing signals from curated Telegram channels. It ships with an AI-assisted parsing pipeline, a real-time web dashboard, and both live and simulation trading modes.

## ✨ Features

-   **Telegram Ingestion**: Asynchronous Telethon clients capture channel messages, normalize them, and persist them to SQLite with channel metadata.
-   **AI Signal Parsing**: spaCy classifiers and NER pipelines detect genuine trading signals, infer direction, and extract entries, targets, and stop losses.
-   **Automated Trading**: The trading engine sizes positions using per-channel win rates and can execute orders on BingX or run in simulation.
-   **Real-Time Dashboard**: A Django + Channels web UI (Catppuccin teal theme) streams live trades, balances, and performance metrics over WebSockets.
-   **Labeling Studio**: Browser-based labeling workflow with AI suggestions and optional batch auto-labeling via management command.
-   **Maintenance Tooling**: Scripts and commands validate configuration, repair datasets, and keep models fresh.

## ⚠️ Disclaimer

Trading cryptocurrency involves significant risk. This bot is provided as-is, and the authors are not responsible for any financial losses. Always test thoroughly in simulation mode before trading with real money.

## 🛠️ Requirements

-   Python 3.11+
-   A Telegram account and API credentials
-   A BingX account and API credentials
-   [uv](https://github.com/astral-sh/uv) (primary package manager)
-   [Docker](https://www.docker.com/) (optional, for containerized deployment)
-   Redis (optional) for production Channel layers; the project falls back to an in-memory backend locally

## 🚀 Setup and Configuration

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Vartius/BingXtg.git
    cd BingXtg
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **(Optional) Spawn a Shell with Dependencies**:
    ```bash
    uv shell
    ```

4.  **Configure the Bot**:
    -   Copy `.env.example` to `.env`.
    -   Open `.env` and fill in your BingX and Telegram API credentials.
    -   Adjust the trading parameters like `LEVERAGE`, `TP` (Take Profit), and `SL` (Stop Loss) to fit your strategy.

5.  **Configure Signal Channels**:
    -   Edit `data/channels.json`.
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
    -   The `data/` directory contains the bot's state and data. The required files (`state.json`, `winrate.json`, `table.json`) will be created automatically on the first run if they don't exist.

## ▶️ Running the Bot

Once configured, you can start the bot from your terminal.

## 🚀 Quick Start

1.  **Install Dependencies** (if not already):
    ```bash
    uv sync
    ```

2.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env with your API credentials
    ```

3.  **Run Database Migrations**:
    ```bash
    uv run manage.py migrate
    ```

4.  **Start the Django Development Server**:
    ```bash
    uv run manage.py runserver
    ```

5.  **Access the Web Interface**:
    - Trading Dashboard: http://localhost:8000/
    - Admin Panel: http://localhost:8000/admin/

6.  **Start the Trading Bot** (in a separate terminal):
    ```bash
    uv run manage.py start_bot
    ```

## 🧠 AI Model Training

All AI model training is now done exclusively in the `ai.ipynb` Jupyter notebook. The Django application uses the trained models for inference through `ai/inference/ai_service.py`.

### Training Process:
1. Open `ai.ipynb` in Jupyter Lab or VS Code
2. Follow the notebook cells to train both classifier and NER models
3. Models are automatically saved to the `ai/models/` directory
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

The high-level layout mirrors the reference in `PROJECT.MD`:

```
BingXtg/
├── ai/                # Inference utilities, labeling tools, trained spaCy models
├── apps/
│   ├── labeling/      # Browser-based labeling studio + auto-labeling services
│   ├── telegram_client/  # Telegram ingestion clients
│   └── trading_bot/   # Trading engine, text parsing, websocket consumers
├── bingxtg_project/   # Django project settings, ASGI/WSGI, URLs
├── core/              # Shared config, database manager, trading helpers
├── data/              # Runtime JSON config/state for channels & dashboard
├── docs/              # Additional documentation (placeholder)
├── logs/              # Rotated log files
├── static/            # CSS/JS assets (Catppuccin teal theme)
├── templates/         # Django templates for dashboard + labeling UI
├── tests/             # Test harnesses (app-level tests live alongside apps)
├── manage.py
├── pyproject.toml     # Project metadata and dependencies (managed by uv)
└── uv.lock            # Resolved dependency lockfile
```

## 🎮 Web Interface

The project includes a modern web interface built with Django:

### Trading Dashboard
- Real-time trading status, balances, and order history
- Live order monitoring with WebSocket updates
- Simulation/live mode indicator and per-channel win rate stats

### Labeling Studio
- Web-first annotation workflow with AI suggestions
- Batch auto-labeling via management command or dashboard action
- SQLite-backed datasets with progress tracking

## 🤖 Management Commands

Common Django management commands:

```bash
# Start the trading bot (prompts for simulation/live mode)
uv run manage.py start_bot

# Run database migrations
uv run manage.py migrate

# Launch the batch auto-labeling workflow
uv run manage.py auto_label
```

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue to discuss a new feature or submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```