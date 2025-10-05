# Data & Persistence

This document covers the SQLite schemas, JSON runtime files, and data export utilities that power the BingXtg workflows.

## SQLite Database

### `total.db`
The single SQLite database file containing all application data: Telegram messages, Django system tables, labeling artefacts, and trading state. Configurable via `DB_PATH` environment variable. Key tables include:

| Table | Purpose |
| --- | --- |
| `messages` | Raw Telegram messages with source channel metadata. |
| `labeled` | Labeled training data enriched with AI predictions and entity offsets. |
| `trades` | Live and historical trades, including status transitions and PnL metrics. |
| `trading_stats` | Aggregated trading performance (total trades, wins, losses, ROI). |
| `channels` | Telegram channel metadata with automatic `updated_at` trigger. |
| `app_state` | Generic key-value store for runtime state. |
| `labeling_*` | Tables tracking labeling sessions, logs, and processed messages. |

> **Note:** Django system tables (auth, contenttypes, sessions, migrations) also reside here.

#### Inspecting the schema
```zsh
sqlite3 total.db ".schema"
```

### Database Access Patterns

- Always read/write through `core.database.manager.DatabaseManager` to ensure WAL mode and foreign keys are enabled.
- When adding new tables, mirror the pragma configuration in `DatabaseManager` or reuse existing `init_*` helpers.

## Runtime JSON State (`data/`)

The `data/` directory is created automatically at runtime and contains JSON files for configuration and state management:

- `channels.json`: Channel metadata cache used during signal parsing (created on first bot run)
- `state.json`: Tracks live application mode (simulation vs live) and other toggles
- `winrate.json`: Per-channel win rate cache that feeds position sizing
- `table.json`: Snapshot of orders rendered in the dashboard

Use helpers from `core/` modules and related services to ensure consistent schemas when updating these files.

## Data Preparation Scripts

| Script | Description |
| --- | --- |
| `scripts/postprocess_db.py` | Cleans legacy labels (e.g., converts numeric directions to `short`/`long`). |
| `scripts/fix_db.py` | Backfills entity start/end indices required for NER training. |
| `ai/training/hf/export_data.py` | Generates `classification_data.csv` and `ner_data.jsonl` in `data_exports/`. |

Run the scripts in the order shown to prepare a high-quality dataset for model training.

## Data Exports

- `classification_data.csv`: Row-based dataset for signal classification, contains label columns and metadata.
- `ner_data.jsonl`: Token-level annotations for entity extraction (pairs, targets, leverage, entry, etc.).

Regenerate exports whenever new labels are added:
```zsh
uv run python ai/training/hf/export_data.py
```

## Backups

- Rolling backups of `total.db` live under `bkp/` (`total_2.db`, `total_3.db`, ...). Rotate or archive older versions periodically.
- Ensure application downtime or WAL checkpointing is handled before copying live databases to avoid corruption.
