# Operations Handbook

This guide focuses on day-to-day operations: running services, monitoring, troubleshooting, and recommended workflows.

## Running Services

### 1. Web Dashboard
```zsh
uv run manage.py runserver
```
- Serves `templates/index.html` and `templates/trading_dashboard.html`.
- WebSocket status is shown on the bottom-left; simulation/live mode banner is on the top-left.

### 2. Trading & Telegram Bot
```zsh
uv run manage.py start_bot
```
- Prompts for simulation vs live mode.
- Normalizes channel IDs and streams messages into `total.db`.
- Places or simulates orders via `apps/trading_bot/order_handler.py`.

### 3. Labeling Studio
```zsh
uv run python ai/labeling/main_textual.py
```
- Textual UI for human-in-the-loop labeling.
- Supports AI-assisted suggestions when provider API keys are configured.

### 4. Batch Auto-Labeling
```zsh
uv run manage.py auto_label --limit 50
```
- Runs automated labeling on unlabeled messages using AI models.
- Limits processing to 50 messages per run to avoid overwhelming the system.

## Dashboard Metrics

- **Balance / Available**: Current BingX or simulated balance, and balance minus allocated margins.
- **Global Winrate**: Computed from historical trades; relies on `winrate.json` and `trading_stats` table.
- **Total Trades / Wins / Losses**: Aggregated counters from `trading_stats`.
- **Total Profit / ROI**: Realized PnL and return on investment.

### Order Table Columns

| Column | Description |
| --- | --- |
| `ID` | Internal trade identifier. |
| `Channel` | Telegram channel ID that produced the signal. |
| `Coin` | Trading pair (e.g., BTCUSDT). |
| `Side` | `Long` or `Short`. |
| `Targets` | Comma-separated take profit levels. |
| `Leverage` | Multiplier applied to the position. |
| `SL` | Stop loss price. |
| `Margin` | Capital allocated to the trade. |
| `Entry` | Target entry price. |
| `Current` | Latest parse`d price. |
| `PNL ($ / %)` | Profit metrics updated by the updater thread. |
| `Status` | `Waiting` (pending entry) or `Open` (active). |

## Automation & Background Tasks

- `updater_thread_worker` recalculates PnL and pushes updates via `DashboardConsumer`.
- JSON caches in `data/` (created at runtime) are kept in sync by bot processes; avoid editing them manually while services run.

## Maintenance Playbook

1. **Refresh `.env` secrets** when rotating API keys; restart services afterward.
2. **Clean labels** using `scripts/postprocess_db.py` followed by `scripts/fix_db.py` whenever new data is imported.
3. **Archive databases** from the `bkp/` directory regularly; ensure WAL files (`total.db-wal`, `total.db-shm`) are checkpointed before copying.
4. **Monitor logs**
   - Django: `tail -f logs/django.log`
   - Telegram: `tail -f logs/telethon_test.log`
   - Root: `tail -f logs.log`

## Development Patterns

- **Logging**: Use `loguru.logger` in Telegram, AI, and trading modules; `logging` in Django views
- **Database Access**: Always use `DatabaseManager` for connections to ensure WAL mode and foreign keys
- **Channel IDs**: Telegram channels use negative IDs (-1002595715996) but stored as positive (2595715996); use `_normalize_channel_id()` for consistency
- **Model Paths**: `HF_*` environment variables for HuggingFace models, bare names for spaCy models
- **Testing**: Mock BingX API calls in tests to avoid hitting the exchange

## Troubleshooting

| Symptom | Possible Cause | Fix |
| --- | --- | --- |
| Dashboard shows "WebSocket disconnected" | Channels worker not running or server restarted | Refresh page; restart `runserver` if needed. |
| Trades stuck in `Waiting` | Entry price not reached; check market feed | Confirm signal text and ensure updater thread is running. |
| AI predictions missing | Model paths misconfigured | Verify `.env` `HF_*` or spaCy paths and restart bot. |
| Labeling studio crashes on launch | Missing API keys for selected provider | Set the required keys or switch to manual labeling mode. |
| Database locked errors | SQLite WAL not configured | Access DB via `DatabaseManager`; avoid parallel writes outside the app. |

## Deployment Checklist

- Set `DJANGO_DEBUG=0` and configure `DJANGO_ALLOWED_HOSTS` properly.
- Use HTTPS termination (nginx, Caddy) in front of Django ASGI workers (uvicorn/daphne).
- Externalize SQLite databases to persistent storage; consider migration to PostgreSQL for multi-user setups.
- Run periodic backups of models, JSON data (data/ directory), databases, and Telethon session files.
