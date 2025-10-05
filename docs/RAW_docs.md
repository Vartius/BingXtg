## DATA
all the data positioned in file total.db
```sh
sqlite3 total.db ".schema"
```

```sql
CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
         ## Database Preparation

**Step 1: Normalize legacy labels**
```zsh
uv run python scripts/postprocess_db.py
```
Removes incorrectly formatted labels (e.g., converts numeric directions `1`/`0` to `long`/`short`).

**Step 2: Populate entity positions**
```zsh
uv run python scripts/fix_db.py
```
Finds entity text and adds start/end character positions required for NER training.

**Step 3: Export training data**
```zsh
uv run python ai/training/hf/export_data.py
```
Generates formatted datasets in `data_exports/`:
- `classification_data.csv` - Signal classification training data
- `ner_data.jsonl` - Named entity recognition training data_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                is_signal BOOLEAN NOT NULL DEFAULT 0,
                regex TEXT
            );
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE labeled (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    is_signal BOOLEAN NOT NULL,
                    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, direction INTEGER, pair TEXT, stop_loss REAL, take_profit REAL, leverage REAL, targets TEXT, entry REAL, ai_is_signal INTEGER, ai_confidence REAL, ai_direction TEXT, ai_pair TEXT, ai_stop_loss REAL, ai_leverage REAL, ai_targets TEXT, ai_entry REAL, ai_processed_at TIMESTAMP, pair_start INTEGER, pair_end INTEGER, stop_loss_start INTEGER, stop_loss_end INTEGER, leverage_start INTEGER, leverage_end INTEGER, target_4_start INTEGER, target_4_end INTEGER, target_1_start INTEGER, target_1_end INTEGER, target_2_start INTEGER, target_2_end INTEGER, target_3_start INTEGER, target_3_end INTEGER, target_5_start INTEGER, target_5_end INTEGER, target_6_start INTEGER, target_6_end INTEGER, target_7_start INTEGER, target_7_end INTEGER, target_8_start INTEGER, target_8_end INTEGER, target_9_start INTEGER, target_9_end INTEGER, target_10_start INTEGER, target_10_end INTEGER, target_11_start INTEGER, target_11_end INTEGER, target_12_start INTEGER, target_12_end INTEGER, target_13_start INTEGER, target_13_end INTEGER, target_14_start INTEGER, target_14_end INTEGER, target_15_start INTEGER, target_15_end INTEGER, target_16_start INTEGER, target_16_end INTEGER, target_17_start INTEGER, target_17_end INTEGER, entry_start INTEGER, entry_end INTEGER,
                    FOREIGN KEY (message_id) REFERENCES messages (id)
                );
CREATE TABLE IF NOT EXISTS "django_migrations" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "app" varchar(255) NOT NULL, "name" varchar(255) NOT NULL, "applied" datetime NOT NULL);
CREATE TABLE IF NOT EXISTS "auth_group_permissions" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "group_id" integer NOT NULL REFERENCES "auth_group" ("id") DEFERRABLE INITIALLY DEFERRED, "permission_id" integer NOT NULL REFERENCES "auth_permission" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "auth_user_groups" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "user_id" integer NOT NULL REFERENCES "auth_user" ("id") DEFERRABLE INITIALLY DEFERRED, "group_id" integer NOT NULL REFERENCES "auth_group" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "auth_user_user_permissions" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "user_id" integer NOT NULL REFERENCES "auth_user" ("id") DEFERRABLE INITIALLY DEFERRED, "permission_id" integer NOT NULL REFERENCES "auth_permission" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE UNIQUE INDEX "auth_group_permissions_group_id_permission_id_0cd325b0_uniq" ON "auth_group_permissions" ("group_id", "permission_id");
CREATE INDEX "auth_group_permissions_group_id_b120cbf9" ON "auth_group_permissions" ("group_id");
CREATE INDEX "auth_group_permissions_permission_id_84c5c92e" ON "auth_group_permissions" ("permission_id");
CREATE UNIQUE INDEX "auth_user_groups_user_id_group_id_94350c0c_uniq" ON "auth_user_groups" ("user_id", "group_id");
CREATE INDEX "auth_user_groups_user_id_6a12ed8b" ON "auth_user_groups" ("user_id");
CREATE INDEX "auth_user_groups_group_id_97559544" ON "auth_user_groups" ("group_id");
CREATE UNIQUE INDEX "auth_user_user_permissions_user_id_permission_id_14a6b632_uniq" ON "auth_user_user_permissions" ("user_id", "permission_id");
CREATE INDEX "auth_user_user_permissions_user_id_a95ead1b" ON "auth_user_user_permissions" ("user_id");
CREATE INDEX "auth_user_user_permissions_permission_id_1fbb5f2c" ON "auth_user_user_permissions" ("permission_id");
CREATE TABLE IF NOT EXISTS "django_admin_log" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "object_id" text NULL, "object_repr" varchar(200) NOT NULL, "action_flag" smallint unsigned NOT NULL CHECK ("action_flag" >= 0), "change_message" text NOT NULL, "content_type_id" integer NULL REFERENCES "django_content_type" ("id") DEFERRABLE INITIALLY DEFERRED, "user_id" integer NOT NULL REFERENCES "auth_user" ("id") DEFERRABLE INITIALLY DEFERRED, "action_time" datetime NOT NULL);
CREATE INDEX "django_admin_log_content_type_id_c4bce8eb" ON "django_admin_log" ("content_type_id");
CREATE INDEX "django_admin_log_user_id_c564eba6" ON "django_admin_log" ("user_id");
CREATE TABLE IF NOT EXISTS "django_content_type" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "app_label" varchar(100) NOT NULL, "model" varchar(100) NOT NULL);
CREATE UNIQUE INDEX "django_content_type_app_label_model_76bd3d3b_uniq" ON "django_content_type" ("app_label", "model");
CREATE TABLE IF NOT EXISTS "auth_permission" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "content_type_id" integer NOT NULL REFERENCES "django_content_type" ("id") DEFERRABLE INITIALLY DEFERRED, "codename" varchar(100) NOT NULL, "name" varchar(255) NOT NULL);
CREATE UNIQUE INDEX "auth_permission_content_type_id_codename_01ab375a_uniq" ON "auth_permission" ("content_type_id", "codename");
CREATE INDEX "auth_permission_content_type_id_2f476e4b" ON "auth_permission" ("content_type_id");
CREATE TABLE IF NOT EXISTS "auth_group" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "name" varchar(150) NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS "auth_user" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "password" varchar(128) NOT NULL, "last_login" datetime NULL, "is_superuser" bool NOT NULL, "username" varchar(150) NOT NULL UNIQUE, "last_name" varchar(150) NOT NULL, "email" varchar(254) NOT NULL, "is_staff" bool NOT NULL, "is_active" bool NOT NULL, "date_joined" datetime NOT NULL, "first_name" varchar(150) NOT NULL);
CREATE TABLE IF NOT EXISTS "django_session" ("session_key" varchar(40) NOT NULL PRIMARY KEY, "session_data" text NOT NULL, "expire_date" datetime NOT NULL);
CREATE INDEX "django_session_expire_date_a5c62663" ON "django_session" ("expire_date");
CREATE TABLE channels (
                channel_id INTEGER PRIMARY KEY,
                title TEXT,
                username TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
CREATE INDEX idx_messages_channel_id ON messages(channel_id);
CREATE INDEX idx_labeled_message_id ON labeled(message_id);
CREATE INDEX idx_labeled_channel_id ON labeled(channel_id);
CREATE TRIGGER trg_channels_updated_at
        AFTER UPDATE ON channels
        BEGIN
          UPDATE channels SET updated_at = CURRENT_TIMESTAMP WHERE channel_id = NEW.channel_id;
        END;
CREATE TABLE app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
CREATE TABLE trades (
              trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
              channel_id INTEGER,
              coin TEXT,
              direction TEXT,
              targets TEXT,
              leverage REAL,
              sl REAL,
              margin REAL,
              entry_price REAL,
              current_price REAL,
              pnl REAL,
              pnl_percent REAL,
              status TEXT,
              close_price REAL,
              closed_at TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, activated_at TIMESTAMP,
              FOREIGN KEY(channel_id) REFERENCES channels(channel_id) ON DELETE SET NULL
            );
CREATE TABLE trading_stats (
              id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
              total_trades INTEGER DEFAULT 0,
              wins INTEGER DEFAULT 0,
              losses INTEGER DEFAULT 0,
              win_rate REAL DEFAULT 0.0,
              profit REAL DEFAULT 0.0,
              roi REAL DEFAULT 0.0,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
CREATE INDEX idx_trades_channel_id ON trades(channel_id);
CREATE UNIQUE INDEX idx_labeled_message_id_unique ON labeled(message_id);
CREATE TABLE IF NOT EXISTS "labeling_labelingsession" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "session_id" varchar(64) NOT NULL UNIQUE, "status" varchar(20) NOT NULL, "current_model" varchar(100) NOT NULL, "total_messages" integer NOT NULL, "processed_count" integer NOT NULL, "signals_found" integer NOT NULL, "errors" integer NOT NULL, "batch_count" integer NOT NULL, "started_at" datetime NULL, "completed_at" datetime NULL, "created_at" datetime NOT NULL, "updated_at" datetime NOT NULL);
CREATE TABLE IF NOT EXISTS "labeling_labelinglog" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "log_type" varchar(10) NOT NULL, "message" text NOT NULL, "timestamp" datetime NOT NULL, "session_id" bigint NOT NULL REFERENCES "labeling_labelingsession" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS "labeling_processedmessage" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "message_id" varchar(100) NOT NULL, "channel_id" varchar(100) NULL, "message_text" text NOT NULL, "is_signal" bool NOT NULL, "pair" varchar(20) NULL, "direction" varchar(10) NULL, "entry" real NULL, "targets_json" text NOT NULL, "stop_loss" real NULL, "leverage" integer NULL, "model_used" varchar(100) NOT NULL, "processing_time" real NOT NULL, "processed_at" datetime NOT NULL, "session_id" bigint NOT NULL REFERENCES "labeling_labelingsession" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX "labeling_labelinglog_session_id_c9346fee" ON "labeling_labelinglog" ("session_id");
CREATE UNIQUE INDEX "labeling_processedmessage_session_id_message_id_31e09add_uniq" ON "labeling_processedmessage" ("session_id", "message_id");
CREATE INDEX "labeling_processedmessage_session_id_db56511a" ON "labeling_processedmessage" ("session_id");
```


## Environment Variables
All necessary environment variables are inside `.env`. See `.env.example` for the full template:

**Key Variables:**
- `API_ID`, `API_HASH`, `FOLDER_ID` - Telegram API credentials
- `APIURL`, `APIKEY`, `SECRETKEY` - BingX API configuration
- `LEVERAGE`, `SL`, `TP`, `START_BALANCE`, `MIN_ORDERS_TO_HIGH`, `MAX_PERCENT` - Trading parameters
- `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS` - Django settings
- `DB_PATH` - Main SQLite database file containing all data (default: `total.db`)
- `MODEL_DIR` - Base directory for AI models (default: `ai_model`)
- `HF_CLASSIFIER_MODEL_PATH`, `HF_NER_MODEL_PATH` - HuggingFace model paths (take precedence)
- `IS_SIGNAL_MODEL_PATH`, `DIRECTION_MODEL_PATH`, `NER_MODEL_PATH` - Legacy spaCy model paths
- `SESSION_FILE` - Telethon session file path (default: `my_account.session`)
- `DATA_DIR` - JSON configuration directory (default: `data`, created at runtime)
- `LOG_LEVEL` - Logging verbosity
- `GOOGLE_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `GITHUB_TOKEN` - Optional AI labeling providers

## Website
To start the web dashboard:
```zsh
uv run manage.py runserver
```

**index.html / trading_dashboard.html**

This is the overview of trading. You should start the Telegram bot in another terminal for this to work:
```zsh
uv run manage.py start_bot
```

- Top-left: Purple popup indicates simulation mode
- Bottom-left: WebSocket connection status indicator

**Account Overview:**
- **Balance**: Your actual balance (Simulation or exchange, depends on mode)
- **Available**: Balance - Sum of margins from all orders
- **Global Winrate**: Overall win rate across all channels
- **Total trades**: Count of all orders
- **Wins**: Count of profitable orders
- **Losses**: Count of losing orders
- **Total profit**: Total PNL
- **ROI**: Return on Investment percentage

**Active and Waiting Orders:**
- **ID**: Order identifier
- **Channel**: Telegram channel ID where signal originated
- **Coin**: The trading pair
- **Side**: Direction (Short/Long)
- **Targets**: List of take-profit levels
- **Leverage**: Position multiplier
- **SL**: Stop loss price
- **Margin**: Capital invested in the trade
- **Entry**: Target entry price
- **Current**: Current parsed market price
- **PNL($)**: Profit in dollars
- **PNL(%)**: Profit percentage
- **Status**: `Waiting` (awaiting entry) or `Open` (position active)


## Labeling Studio
Textual-based TUI for message labeling with AI assistance.

**Launch the labeling studio:**
```zsh
uv run python ai/labeling/main_textual.py
```

**Features:**
- AI-powered labeling suggestions
- Support for multiple AI providers (Gemini, Groq, Anthropic, Cohere, GitHub Copilot)
- Configure API keys in `.env` for different providers
- Gemini typically provides the most labels when configured

**UI Components:**
- Message to label: Raw message text from Telegram
- Label this message: AI-suggested labeling for review/editing

**Styling:** All CSS uses Catppuccin Macchiato theme with teal accent

**Batch Auto-Labeling:**
For automated batch processing via management command:
```zsh
uv run manage.py auto_label --limit 50
```


How to prepare db:
firstly use postprocess_db.py
it help you to remove all wrongly formatted labels like 1/0 for direction when should be short/long
then fix_db.py
it will find necessary data and add start and end position of found text for ner training

data_exports:
here is formatted data from total.db to train models, should be two files:
classification_data.csv
ner_data.jsonl
can be generated using 
```zsh
uv run ai/training/hf/export_data.py
```

projcet arch:
core/
    config/
        settings.py # information
        validation.py # information
    database/
        manager.py # information

bingxtg_project/
    ...

apps/
    ...

ai/
    inference/
        ai_service.py
        predict.py
    labeling/
        main_textual.css
        main_textual.py
    models/
        ner_extractor/
        signal_classifier/
    training/
        hf/
            ...
        ...

## AI Training Methods

### 1. HuggingFace Fine-tuning (Recommended)
Location: `ai/training/hf/`

**Models:** Fine-tunes `xlm-roberta-base` for signal classification and NER

**Training:**
- Use `train_classifier.ipynb` and `train_ner.ipynb` notebooks
- Best run on Google Colab with T4 GPU or better
- Alternatively, use Python scripts: `train_classifier.py`, `train_ner.py`

**Testing:**
- `test_classifier.py` - Test classifier model
- `test_ner.py` - Test NER model
- `total_hf_ai_test.py` - Comprehensive test suite (can run on local CPU)

**Deployment:**
- Models saved to `ai/models/signal_classifier/` and `ai/models/ner_extractor/`
- Configure paths via `HF_CLASSIFIER_MODEL_PATH` and `HF_NER_MODEL_PATH` in `.env`

### 2. Legacy spaCy Training (Deprecated)
Location: `ai/training/` (not hf subdirectory)

**Features:**
- Smaller models, CPU-friendly
- Can train and run entirely on local PC
- Lower accuracy than HuggingFace models
- Maintained for backward compatibility only

**Note:** HuggingFace models take precedence when both are present. New projects should use HuggingFace approach.

