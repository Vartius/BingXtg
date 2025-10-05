# System Architecture

The BingXtg platform ties together Telegram ingestion, AI decision making, and automated order execution. This document outlines the major components and how they collaborate.

## High-Level Overview

```
+----------------+      +--------------------+      +-----------------+
| Telegram APIs  | ---> | Telegram Ingestion | ---> | SQLite Databases|
+----------------+      +--------------------+      +-----------------+
                                               \           /
                                                v         v
                                         +---------------------+
                                         | AI Inference Layer  |
                                         +---------------------+
                                                |
                                                v
                                         +------------------+
                                         | Trading Engine   |
                                         +------------------+
                                                |
                                                v
                                         +------------------+
                                         | Web Dashboard    |
                                         +------------------+
```

## Core Services

### Telegram Ingestion (`apps/telegram_client`)
- Uses Telethon to iterate dialogs (`message_extractor.py`).
- Normalizes channel IDs, persists raw messages to `messages` table via `core.database.manager.DatabaseManager`.
- Maintains channel metadata with automatic `updated_at` trigger.

### AI Inference Layer (`ai/inference`)
- `AIInferenceService` loads either HuggingFace or spaCy pipelines depending on installed models.
- Classifies messages into one of four states (`NON_SIGNAL`, `SIGNAL_LONG`, `SIGNAL_SHORT`, `SIGNAL_NONE`).
- Extracts token-level entities (pairs, targets, SL) for downstream use.

### Trading Engine (`apps/trading_bot`)
- `order_handler.py` sizes positions based on BingX balance and channel win rate.
- `bingx_api.py` wraps REST calls to BingX (mock in tests to avoid hitting the exchange).
- `updater_thread_worker` monitors trades, recomputes PnL, and triggers dashboard updates.

### Operator Interfaces
- **Web dashboard** (Django + Channels) renders live metrics and order tables. Frontend assets live in `static/bot/` with Catppuccin styling.
- **Labeling studio** (`ai/labeling/main_textual.py`) is a textual UI for assisted labeling with optional third-party LLM providers.

## Persistence & Configuration

- **Database**: 
  - `total.db` - Single SQLite database containing everything: Telegram messages, trades, labels, and Django system tables
  - Configurable via `DB_PATH` environment variable (defaults to `total.db`)
  - Always access through `DatabaseManager` to ensure WAL mode and foreign keys are configured
- **Runtime JSON**: Configuration and cached stats in `data/` directory (created at runtime if not present): `channels.json`, `winrate.json`, `state.json`, `table.json`
- **Environment Variables**: Paths and API keys are read at import-time by most modules; restart long-lived processes after changes.

## AI Model Layout

```
ai/
├── inference/
│   └── ai_service.py
├── models/
│   ├── signal_classifier/      # HuggingFace classifier artefacts
│   └── ner_extractor/          # HuggingFace token classification artefacts
└── training/
    ├── hf/                     # HuggingFace training scripts & notebooks
    └── (legacy spaCy scripts)
```

- When both HuggingFace and spaCy models are present, HuggingFace takes precedence.
- Exported datasets live in `data_exports/` (`classification_data.csv`, `ner_data.jsonl`).

## Django Project Structure

- `bingxtg_project/` contains Django settings, ASGI/WSGI entry points, and Celery config if enabled.
- Pluggable apps live under `apps/`, each with `management`, `services`, and `tests` packages.
- Real-time updates use Django Channels (`apps/trading_bot/consumers.py`) broadcasting to the `dashboard` group.

## Deployment Considerations

- Use `uv run manage.py migrate` to initialize the SQLite schema; production deployments may swap SQLite for PostgreSQL with minimal changes.
- Long-running bots should use process managers (systemd, supervisord) and dedicated `.env` per environment.
- Secure secrets (`.env`) via vault/backups; rotate API keys routinely.
