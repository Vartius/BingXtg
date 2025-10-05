# BingXtg Documentation

Welcome to the documentation hub for the **BingXtg Trading Bot**. This project orchestrates Telegram signal ingestion, AI-assisted trade classification, and automated trade execution against the BingX exchange, all wrapped in a Django-powered control panel.

Use the sections below as a guided map:

- [Getting Started](./setup.md)
- [System Architecture](./architecture.md)
- [Data & Persistence](./data.md)
- [AI Workflows](./ai.md)
- [Operations Handbook](./operations.md)
- [Contributing](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)

## Key Capabilities

- **Signal ingestion:** Streams Telegram messages, de-duplicates them, and stores raw content for later processing.
- **AI-driven labeling:** Classifies messages into actionable trade signals using spaCy or HuggingFace pipelines.
- **Order execution:** Places, monitors, and reconciles trades on BingX with live/simulated modes.
- **Operator tooling:** Provides a web UI, labeling studio, and maintenance scripts to keep models and datasets healthy.

## Repositories & Conventions

- Python tooling is managed with [`uv`](https://github.com/astral-sh/uv); always invoke scripts via `uv run` unless documented otherwise.
- `loguru` is the preferred logger for runtime modules; Django views stick with the standard `logging` API.
- SQLite databases (`total.db`, `messages.db`) reside at the project root unless `DB_PATH` overrides them.

For detailed setup, data, and operational playbooks, continue with the linked sections above.
