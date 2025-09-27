# BingXtg Trading Bot – Project Documentation

## 1. General Description
The BingXtg project is an AI-assisted cryptocurrency trading platform built on Django. It ingests trading signals from curated Telegram channels, enriches them with natural language processing, and converts qualified opportunities into actionable trades on the BingX exchange. Alongside live execution, the system offers simulation mode, real-time dashboards, and extensive tooling for dataset labeling, model training, and database hygiene.

## 2. Details
### Functional Highlights
- **Signal ingestion:** Asynchronous Telegram clients collect channel traffic, persist it to SQLite, and maintain channel metadata.
- **AI inference:** spaCy-based classifiers and NER pipelines distinguish genuine signals, infer trade direction, and extract structured order details.
- **Order lifecycle:** A trading engine sizes positions based on channel win rates, places/simulates BingX orders, and continuously recalculates PnL with live pricing.
- **Operations console:** A Django front end (styled with a Catppuccin teal theme) surfaces balances, open trades, statistics, websocket-driven updates, and provides one-click access to the Labeling Studio.
- **Labeling workflow:** A Django "Labeling Studio" surfaces the next unlabeled message in the browser, pre-fills AI suggestions, and persists annotations to SQLite. An integrated auto-labeling service can batch label messages directly from the dashboard or via the `auto_label` management command (the legacy Textual TUI remains available for power users).
- **Maintenance scripts:** Rich CLI utilities validate environment configuration, repair misaligned labels, and standardize datasets to keep training corpora healthy.

### Data & Control Flow
1. **Collection** – `MessageExtractor` retrieves Telegram messages and records them in `total.db` via `DatabaseManager`.
2. **Enrichment** – The AI inference service loads trained spaCy pipelines from `ai/models` and normalizes text for consistent predictions.
3. **Decisioning** – `tg_parser` routes new messages through AI parsing; qualifying signals trigger `order_handler.place_order` to size and open trades.
4. **Execution** – Depending on mode, orders are simulated locally or sent through `bingx_api` to the exchange; outcomes update the `trades` table.
5. **Monitoring** – Websocket broadcasts update the dashboard, while REST fallbacks expose the same data for non-WS clients.
6. **Feedback Loop** – Labeling Studio (web) and legacy CLI tools iterate on datasets, producing improved models consumed by the inference service.

### Operational Considerations
- Environment variables (stored in `.env`) control API keys, leverage defaults, database locations, and deployment switches.
- Redis is used opportunistically for channel layers; the project falls back to an in-memory backend when Redis is unavailable.
- SQLite WAL mode keeps concurrent readers responsive while updater threads adjust open trades.
- Logging routes through `loguru` and Django’s logging framework, capturing diagnostics in `logs.log` and `logs/`.
- `LABELING_DB_PATH` lets the web labeling studio target a specific SQLite file (defaults to the primary database when unset).

## 3. Technology Stack
| Layer | Technologies | Purpose |
| --- | --- | --- |
| Language & Runtime | Python 3.11+, uv package manager | Core implementation and dependency management |
| Backend Framework | Django 5.2, Django Channels, Daphne | Web UI, REST endpoints, websocket delivery |
| Messaging & Data | Telethon, SQLite (total.db) | Telegram ingestion and persistent storage |
| Trading Integration | `bingx` & `py-bingx` SDKs | Communication with the BingX exchange |
| AI / ML | spaCy, scikit-learn, transformers, Django labeling studio, legacy Textual CLI, LLM SDKs (OpenAI, Anthropic, Groq, Cohere, Gemini) | Training, inference, and annotation workflows |
| Frontend | Django templates, vanilla JS, PicoCSS, Catppuccin teal theme CSS | Trading dashboard presentation |
| Tooling & Ops | loguru, python-dotenv, Redis (optional), Docker | Logging, configuration, optional deployment |

## 4. Structure Overview
```
BingXtg/
├── .dockerignore
├── .env
├── .env.example
├── .git/
├── .github/
│   ├── copilot-instructions.md
│   └── prompts/
│       ├── plan.prompt.md
│       ├── specify.prompt.md
│       └── tasks.prompt.md
├── .gitignore
├── .python-version
├── .venv/
├── Dockerfile
├── README.md
├── ai/
│   ├── inference/
│   ├── labeling/
│   ├── models/
│   └── training/
├── apps/
│   ├── labeling/
│   ├── telegram_client/
│   └── trading_bot/
├── bingxtg_project/
├── core/
│   ├── config/
│   ├── database/
│   └── trading/
├── data/
├── docs/
├── logs/
├── logs.log
├── manage.py
├── my_account.session
├── pyproject.toml
├── pyrightconfig.json
├── scripts/
├── static/
├── templates/
├── tests/
├── total.db
├── total.db-shm
├── total.db-wal
└── uv.lock
```

## 5. Folder & File Reference (Detailed)
### Root Items
| Path | Description |
| --- | --- |
| `.dockerignore` | Docker build ignore list tuned for Python, datasets, and venv artifacts. |
| `.env` | Runtime environment configuration (API keys, DB paths, trading parameters); should stay local and secret. |
| `.env.example` | Template for required environment variables with placeholders for credentials. |
| `.git/` | Git metadata for version control (no project logic inside). |
| `.github/` | Repository automation assets; see section below for details. |
| `.gitignore` | Ignore rules covering virtual environments, compiled assets, logs, and databases. |
| `.python-version` | Pin for pyenv/uv indicating the project expects Python 3.11. |
| `.venv/` | Local virtual environment directory (developer-specific, not part of application logic). |
| `Dockerfile` | Container recipe using Python 3.11-slim and uv for dependency syncing; defaults to `uv run main.py` as the entrypoint. |
| `README.md` | High-level project introduction, setup instructions, and feature highlights. |
| `logs.log` | Primary aggregated log file written by loguru and Django logging configuration. |
| `manage.py` | Django management entry point, enabling migrations, server start, and custom commands. |
| `my_account.session` | Telethon session file for authenticating the Telegram client. |
| `pyproject.toml` | Project metadata and dependency list managed by uv (broad ML and Django stack). |
| `pyrightconfig.json` | Pyright type checker configuration pointing to the local virtual environment. |
| `total.db` | Primary SQLite database holding labeled signals, trades, and trading stats. |
| `total.db-shm` / `total.db-wal` | SQLite write-ahead logging sidecars enabling concurrent reads. |
| `uv.lock` | Resolved dependency lockfile ensuring reproducible uv installs. |
| `ai/` | AI tooling, models, and training scripts (detailed below). |
| `apps/` | Django application packages (`telegram_client`, `trading_bot`). |
| `bingxtg_project/` | Django project configuration (settings, ASGI/WSGI, URLs). |
| `core/` | Shared configuration, database utilties, and trading helpers. |
| `data/` | JSON configuration/state files for channels, dashboard, and win-rate snapshots. |
| `docs/` | Placeholder documentation directory (currently empty). |
| `logs/` | Rotated log outputs from Django and Telethon subsystems. |
| `scripts/` | Standalone maintenance scripts for databases and environment setup. |
| `static/` | Frontend static assets (CSS/JS) for the trading dashboard. |
| `templates/` | Django HTML templates (base layout and dashboard). |
| `tests/` | Placeholder for project-level automated tests (currently empty). |

### `.github/`
- **Folder Purpose:** CI/editor automation aides and Copilot instructions.
  - `copilot-instructions.md` – Describes coding standards, architecture, and workflows for AI assistants.
  - `prompts/` – Prompt templates used for structured Copilot interactions.
    - `plan.prompt.md` – Template for planning responses.
    - `specify.prompt.md` – Template for requirement clarification.
    - `tasks.prompt.md` – Template guiding step-by-step task execution.

### `ai/`
- **Folder Purpose:** Tooling for AI inference, labeling, training, and persisted models.
  - `inference/` – Runtime inference utilities.
    - `ai_service.py` – `AIInferenceService` class loading spaCy models, normalizing text, and exposing helpers such as `parse_signal` and `parse_message_with_ai`.
    - `text_preprocessing.py` – Placeholder for additional preprocessing routines (currently empty). 
  - `labeling/` – Legacy Textual tooling retained for power users who prefer a terminal workflow.
  - `main_textual.py` – 1.7k-line TUI application orchestrating database connections, batching unlabeled messages, cycling through multiple LLM providers, and persisting results (now superseded by the Django auto-labeling service but kept for archival use).
    - `main_textual.css` – Styling for the Textual interface to mirror the Catppuccin teal theme.
  - `models/` – Saved spaCy pipelines used in production inference.
    - `direction_model/` – Text classification pipeline artifacts; includes `config.cfg`, `meta.json`, and generated directories (`tokenizer/`, `textcat/`, `vocab/`) containing binary model weights and vocabulary tables.
    - `is_signal_model/` – Binary classifier artifacts with the same layout as above for detecting signal vs. noise.
    - `ner_model/` – Named Entity Recognition pipeline artifacts (`config.cfg`, `meta.json`, `tokenizer/`, `vocab/`, component data). Generated binary files reside within subdirectories and are consumed directly by spaCy.
  - `training/` – Scripts for preparing and training AI models.
    - `__init__.py` – Marks the training package and documents its purpose.
    - `classification_train.py` – Trains spaCy textcat models for signal detection and direction classification using data from `total.db`.
  - `classification_test.py` – Smoke tests trained classifiers with sample messages.
  - `ner_training.py` – Builds multilingual NER pipeline, handles alignment, evaluates progress, and saves the resulting model.
  - `ner_test.py` – Sanity-check script that loads the saved NER model and prints extracted entities from sample texts.
    - `total_ai_test.py` – Comprehensive evaluation script that exercises classification and NER components across datasets.
    - `utils.py` – Shared helpers for normalization, database access, batch creation, and evaluation metrics.

### `apps/`
- **Folder Purpose:** Django applications providing Telegram ingestion and trading functionality.
  - `telegram_client/`
    - `__init__.py` – Exports `MessageExtractor` for easy import elsewhere.
    - `apps.py` – Registers the Telegram client app with Django.
    - `message_extractor.py` – Async helper class handling channel discovery, message scraping, metadata enrichment, and database persistence.
    - `urls.py` – Placeholder URL module (currently empty list) for future expansion.
    - `management/` – Management commands namespace (currently only `__pycache__` placeholder; no commands defined yet).
    - `migrations/` – Reserved for future Django migrations (currently empty).
  - `labeling/`
    - `apps.py` – Registers the labeling interface as a first-class Django app.
    - `forms.py` – Defines `LabelingForm`, handling validation, AI-assisted defaults, and serialization to the database manager payload.
    - `views.py` – Class-based view serving the browser-based Labeling Studio with AI suggestions and a POST endpoint that triggers the batch auto-labeling service.
    - `services/auto_labeler.py` – Batch processing helper that calls the spaCy inference pipelines and persists predictions.
    - `management/commands/auto_label.py` – Management command that exposes the batch processor from the CLI.
    - `urls.py` – Routes `/labeling/` to the labeling dashboard.
    - `tests/` – Integration tests ensuring the labeling UI renders, saves decisions, and handles empty queues.
  - `trading_bot/`
    - `__init__.py` – Package marker.
    - `apps.py` – Registers the trading bot application.
    - `bingx_api.py` – BingX REST client wrapper with cached client instantiation, price retrieval, balance checks, and market order placement.
    - `command_handler.py` – Telegram command routing helpers for bot management (e.g., `.chatscheck`, `.stop`).
    - `consumers.py` – Django Channels websocket consumer broadcasting dashboard updates.
    - `order_handler.py` – 770-line core trading engine managing trade creation, sizing logic, PnL updates, and websocket notifications.
    - `routing.py` – Websocket routing definitions for Channels.
    - `text_parser.py` – Regex and AI-assisted message parsing for trade signals.
    - `tg_parser.py` – Telegram client bootstrapper, message handlers, and orchestrator linking parsing to order placement.
    - `urls.py` – View routing for dashboard endpoints and favicon shim.
    - `views.py` – Dashboard views providing initial data payloads and REST fallbacks.
    - `management/`
      - `__init__.py` – Package marker.
      - `commands/`
        - `__init__.py` – Command package marker.
        - `start_bot.py` – Interactive CLI for launching the trading bot in live or simulation mode; validates config via `check_all` before startup.
    - `migrations/` – Reserved for database migrations (currently empty).
    - `tests/` – Placeholder test package (currently empty).

### `bingxtg_project/`
- **Folder Purpose:** Django project configuration and entry points.
  - `__init__.py` – Project package marker.
  - `asgi.py` – Configures ASGI application with HTTP and websocket routing via `ProtocolTypeRouter`.
  - `settings.py` – Central settings file loading `.env`, configuring installed apps, static paths, channel layers (Redis fallback), logging, and database connection.
    - Defines `LABELING_DB_PATH`, allowing the web labeling studio to target `total.db` or another SQLite file via environment override.
  - `urls.py` – Root URL router delegating to Django admin, trading dashboard, and telegram app; optionally serves static/media in DEBUG.
  - `wsgi.py` – WSGI entry point for traditional deployments.

### `core/`
- **Folder Purpose:** Shared infrastructure and supporting assets.
  - `config/`
    - `settings.py` – Loads environment variables, resolves project paths, and configures logging.
    - `validation.py` – Environment schema validation and database schema bootstrapper ensuring tables/indexes exist.
  - `database/`
    - `manager.py` – Comprehensive SQLite manager handling initialization, CRUD operations, app state storage, and schema self-healing.
  - `trading/`
    - `order_manager.py` – Placeholder file reserved for future trading orchestration (currently empty).
    - `signal_parser.py` – Placeholder for shared parsing utilities (currently empty).
  - `my_account.session` – Alternate Telethon session file kept under `core/` for modular access.
  - `total.db` – Duplicate SQLite database snapshot located within the `core` namespace (likely backup/reference copy).

### `data/`
- **Folder Purpose:** Runtime configuration snapshots for channel behavior and dashboard state.
  - `channels.json` – Channel descriptors with regex patterns, direction keywords, and enable flags.
  - `state.json` – Cached dashboard state including sample orders and balances.
  - `table.json` – Simplified table snapshot used for UI prototyping/testing.
  - `winrate.json` – Historical win/loss counts per channel powering adaptive sizing logic.

### `docs/`
- **Folder Purpose:** Reserved for future documentation (currently empty).

### `logs/`
- **Folder Purpose:** Rotated and subsystem-specific logs.
  - `django.log` – Django application log output.
  - `telethon_test.log` – Telethon client diagnostic log.

### `scripts/`
- **Folder Purpose:** One-off maintenance utilities.
  - `fix_db.py` – Interactive script correcting misaligned labels, adding positional metadata, and sanitizing numbers in `labeled` entries.
  - `postprocess_db.py` – Rich CLI workflow that standardizes direction/pair fields, cleans non-signals, and optionally removes incomplete records.
  - `setup_environment.py` – Placeholder for bootstrap automation (currently empty).

### `static/`
- **Folder Purpose:** Frontend assets for the Django dashboard.
  - `bot/css/`
    - `catppuccin-teal-theme.css` – Extensive theming file defining Catppuccin Macchiato + teal palette and UI components.
    - `style.css` – Additional layout, table, and animation styling layered on PicoCSS.
  - `bot/js/`
    - `dashboard.js` – Client-side controller that hydrates initial data, manages websocket connections, updates UI state, and handles reconnection logic.

### `templates/`
- **Folder Purpose:** Django templates powering the public UI.
  - `base.html` – Layout template importing PicoCSS, Catppuccin theme, global scripts, and navigation.
  - `index.html` – Landing dashboard that wraps trading metrics, exposes quick access to the Labeling Studio, and now includes an "Auto Label" action that kicks off the batch processor.
  - `trading_dashboard.html` – Dashboard page presenting stat cards, orders table, websocket status indicator, and simulation banner.
  - `labeling/index.html` – Browser-based labeling studio showing the next unlabeled message, AI hints, and annotation form.

### `tests/`
- **Folder Purpose:** Placeholder for future project-wide suites (app-specific coverage currently lives under `apps/labeling/tests/`).

### Other Datastores & Artifacts
- `total.db`, `core/total.db` – SQLite databases holding raw messages, labeled data, and trading history.
- `my_account.session`, `core/my_account.session` – Telethon session files enabling authenticated Telegram connections.
- `logs.log`, `logs/*.log` – Persistent log outputs for debugging and auditing.

## 6. Summary
This documentation consolidates architectural context, stack choices, and a path-by-path reference for the BingXtg trading bot. Use it as the authoritative index when onboarding new contributors, auditing system behavior, or tracing dependencies between Telegram ingestion, AI inference, and trading execution.
