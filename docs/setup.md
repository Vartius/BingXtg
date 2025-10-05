# Getting Started

This guide walks through preparing a local or remote environment, configuring secrets, and launching the BingXtg stack.

## Prerequisites

- **Python 3.11+** with [`uv`](https://github.com/astral-sh/uv) installed (`pip install uv`)
- **SQLite 3.35+** (included with most OS distributions)
- 
## Project Setup

1. **Clone the repository**
   ```zsh
   git clone https://github.com/Vartius/BingXtg.git
   cd BingXtg
   ```

2. **Install dependencies**
   ```zsh
   uv sync
   ```

3. **Create the `.env` file**

   Use `.env.example` if available, or copy the template below and populate real credentials:

   ```zsh
   cp .env.example .env  # if the file exists
   ```

   ```zsh
   # Telegram API Configuration
   API_ID=00000000
   API_HASH=00000000000000000000000000000000
   FOLDER_ID=1

   # BingX API Configuration
   APIURL=https://open-api.bingx.com
   APIKEY=your_api_key
   SECRETKEY=your_secret

   # Trading Configuration
   LEVERAGE=20
   SL=-0.5
   TP=0.2
   START_BALANCE=10.0
   MIN_ORDERS_TO_HIGH=20
   MAX_PERCENT=0.3

   # Django
   DJANGO_SECRET_KEY=replace_me
   DJANGO_DEBUG=1
   DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

   # Paths (optional, defaults are fine)
   DB_PATH=total.db
   MODEL_DIR=ai_model
   IS_SIGNAL_MODEL_PATH=ai/models/is_signal_model
   DIRECTION_MODEL_PATH=ai/models/direction_model
   HF_CLASSIFIER_MODEL_PATH=ai/models/signal_classifier
   HF_NER_MODEL_PATH=ai/models/ner_extractor
   SESSION_FILE=my_account.session
   DATA_DIR=data

   # Logging
   LOG_LEVEL=INFO

   # Optional AI providers for labeling studio
   GOOGLE_API_KEY=...
   GROQ_API_KEY=...
   ANTHROPIC_API_KEY=...
   COHERE_API_KEY=...
   GITHUB_TOKEN=...
   ```

   > **Tip:** `HF_CLASSIFIER_MODEL_PATH` and `HF_NER_MODEL_PATH` take precedence over spaCy paths when defined. Point them to trained HuggingFace model folders under `ai/models/`.

4. **Collect static files (optional for development)**
   ```zsh
   uv run manage.py collectstatic --noinput
   ```

5. **Apply database migrations**
   ```zsh
   uv run manage.py migrate
   ```

## Launching the Stack

- **Django web UI**
  ```zsh
  uv run manage.py runserver
  ```

- **Telegram ingestion & trading bot**
  ```zsh
  uv run manage.py start_bot
  ```

  When prompted, choose simulation or live mode. The web dashboard (at `http://127.0.0.1:8000/`) displays current mode in the top-left banner and WebSocket connection status in the bottom-left.

- **Labeling studio (terminal UI)**
  ```zsh
  uv run python ai/labeling/main_textual.py
  ```

## Database Preparation Workflow

1. Normalize legacy labels:
   ```zsh
   uv run python scripts/postprocess_db.py --db total.db
   ```

2. Populate entity offsets for NER training:
   ```zsh
   uv run python scripts/fix_db.py --db total.db
   ```

3. Export training datasets:
   ```zsh
   uv run python ai/training/hf/export_data.py
   ```

## Development Tips

- Keep long-lived processes (bot, Channels workers) restarted after changing model paths or `.env` values because configuration is read at import time.
- Telethon session files default to `my_account.session`. Override via `SESSION_FILE` if you need separate identities per environment.
- Logs can be tailed in `logs/` (Django) and `logs.log` (root).
