import os
import sqlite3
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Required environment variables and their expected types
REQUIRED_ENV_SCHEMA = {
    # Telegram API
    "API_ID": int,
    "API_HASH": str,
    # BingX API
    "APIKEY": str,
    "SECRETKEY": str,
    # Trading Config
    "LEVERAGE": int,
    "SL": float,
    "TP": float,
    "START_BALANCE": float,
    "MIN_ORDERS_TO_HIGH": int,
    "MAX_PERCENT": float,
    # Django
    "DJANGO_SECRET_KEY": str,
    "DJANGO_DEBUG": bool,
    "DJANGO_ALLOWED_HOSTS": str,
}

# Optional environment variables with their expected types and defaults
OPTIONAL_ENV_SCHEMA = {
    # Telegram (optional)
    "FOLDER_ID": (int, 1),
    # BingX (optional)
    "APIURL": (str, "https://open-api.bingx.com"),
    # Paths (optional defaults)
    "DB_PATH": (str, "total.db"),
    "MODEL_DIR": (str, "ai_model"),
    "SESSION_FILE": (str, "my_account.session"),
    "DATA_DIR": (str, "data"),
    # Logging
    "LOG_LEVEL": (str, "INFO"),
    # Redis (optional - uses in-memory channel layer if not provided)
    "REDIS_HOST": (str, "127.0.0.1"),
    "REDIS_PORT": (int, 6379),
    # Email (optional - for notifications)
    "EMAIL_HOST": (str, ""),
    "EMAIL_PORT": (int, 587),
    "EMAIL_USE_TLS": (bool, True),
    "EMAIL_HOST_USER": (str, ""),
    "EMAIL_HOST_PASSWORD": (str, ""),
}

SCHEMAS = {
    "channels": {
        "create": """
            CREATE TABLE IF NOT EXISTS channels (
              channel_id INTEGER PRIMARY KEY,
              title TEXT,
              username TEXT,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "columns": {
            "channel_id": "INTEGER",
            "title": "TEXT",
            "username": "TEXT",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "indexes": [],
    },
    "messages": {
        "create": """
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              channel_id INTEGER NOT NULL,
              message TEXT,
              is_signal INTEGER,
              FOREIGN KEY(channel_id) REFERENCES channels(channel_id) ON DELETE CASCADE
            )
        """,
        "columns": {
            "id": "INTEGER",
            "channel_id": "INTEGER",
            "message": "TEXT",
            "is_signal": "INTEGER",
        },
        "indexes": [
            (
                "idx_messages_channel_id",
                "CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id)",
            )
        ],
    },
    "trades": {
        "create": """
            CREATE TABLE IF NOT EXISTS trades (
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
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(channel_id) REFERENCES channels(channel_id) ON DELETE SET NULL
            )
        """,
        "columns": {
            "trade_id": "INTEGER",
            "channel_id": "INTEGER",
            "coin": "TEXT",
            "direction": "TEXT",
            "targets": "TEXT",
            "leverage": "REAL",
            "sl": "REAL",
            "margin": "REAL",
            "entry_price": "REAL",
            "current_price": "REAL",
            "pnl": "REAL",
            "pnl_percent": "REAL",
            "status": "TEXT",
            "close_price": "REAL",
            "closed_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "indexes": [
            (
                "idx_trades_channel_id",
                "CREATE INDEX IF NOT EXISTS idx_trades_channel_id ON trades(channel_id)",
            )
        ],
    },
    "labeled": {
        "create": """
            CREATE TABLE IF NOT EXISTS labeled (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              message_id INTEGER,
              channel_id INTEGER,
              message TEXT,
              is_signal INTEGER,
              labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              direction TEXT,
              pair TEXT,
              stop_loss REAL,
              leverage REAL,
              targets TEXT,
              entry REAL,
              FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
              FOREIGN KEY(channel_id) REFERENCES channels(channel_id) ON DELETE CASCADE
            )
        """,
        "columns": {
            "id": "INTEGER",
            "message_id": "INTEGER",
            "channel_id": "INTEGER",
            "message": "TEXT",
            "is_signal": "INTEGER",
            "labeled_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "direction": "TEXT",
            "pair": "TEXT",
            "stop_loss": "REAL",
            "leverage": "REAL",
            "targets": "TEXT",
            "entry": "REAL",
        },
        "indexes": [
            (
                "idx_labeled_message_id",
                "CREATE INDEX IF NOT EXISTS idx_labeled_message_id ON labeled(message_id)",
            ),
            (
                "idx_labeled_channel_id",
                "CREATE INDEX IF NOT EXISTS idx_labeled_channel_id ON labeled(channel_id)",
            ),
        ],
    },
    "trading_stats": {
        "create": """
            CREATE TABLE IF NOT EXISTS trading_stats (
              id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
              total_trades INTEGER DEFAULT 0,
              wins INTEGER DEFAULT 0,
              losses INTEGER DEFAULT 0,
              win_rate REAL DEFAULT 0.0,
              profit REAL DEFAULT 0.0,
              roi REAL DEFAULT 0.0,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "columns": {
            "id": "INTEGER",
            "total_trades": "INTEGER DEFAULT 0",
            "wins": "INTEGER DEFAULT 0",
            "losses": "INTEGER DEFAULT 0",
            "win_rate": "REAL DEFAULT 0.0",
            "profit": "REAL DEFAULT 0.0",
            "roi": "REAL DEFAULT 0.0",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "indexes": [],
    },
}

TRIGGERS = {
    "trg_channels_updated_at": """
        CREATE TRIGGER IF NOT EXISTS trg_channels_updated_at
        AFTER UPDATE ON channels
        BEGIN
          UPDATE channels SET updated_at = CURRENT_TIMESTAMP WHERE channel_id = NEW.channel_id;
        END;
    """
}


def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Create tables first
    for spec in SCHEMAS.values():
        cur.execute(spec["create"])

    # Add missing columns
    for table, spec in SCHEMAS.items():
        cur.execute(f"PRAGMA table_info({table});")
        existing_cols = {row[1] for row in cur.fetchall()}
        for col, decl in spec["columns"].items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # Create indexes
    for spec in SCHEMAS.values():
        for _, ddl in spec.get("indexes", []):
            cur.execute(ddl)

    # Create triggers
    for ddl in TRIGGERS.values():
        cur.execute(ddl)


def _validate_env():
    """Validate required environment variables and warn about optional ones."""
    missing = []
    wrong_types = []

    # Validate required environment variables
    for key, expected_type in REQUIRED_ENV_SCHEMA.items():
        val = os.getenv(key)
        if val is None or val.strip() == "":
            missing.append(key)
            continue

        try:
            if expected_type is bool:
                if val.lower() not in ("true", "false", "1", "0"):
                    raise ValueError
            elif expected_type is int:
                int(val)
            elif expected_type is float:
                float(val)
        except ValueError:
            wrong_types.append((key, val, expected_type.__name__))

    # Validate optional environment variables (only if provided)
    optional_warnings = []
    for key, (expected_type, default) in OPTIONAL_ENV_SCHEMA.items():
        val = os.getenv(key)
        if val is None or val.strip() == "":
            # Optional variables don't need to be present
            continue

        try:
            if expected_type is bool:
                if val.lower() not in ("true", "false", "1", "0"):
                    raise ValueError
            elif expected_type is int:
                int(val)
            elif expected_type is float:
                float(val)
        except ValueError:
            optional_warnings.append((key, val, expected_type.__name__))

    # Report errors for required variables
    if missing:
        logger.error(f"Missing REQUIRED environment variables: {', '.join(missing)}")

    if wrong_types:
        for key, val, typ in wrong_types:
            logger.error(f"Invalid type for REQUIRED {key}: '{val}' (expected {typ})")

    # Only raise error if required variables are missing/invalid
    if missing or wrong_types:
        raise EnvironmentError("Required environment configuration is invalid!")

    # Report warnings for optional variables (not fatal)
    if optional_warnings:
        for key, val, typ in optional_warnings:
            logger.warning(
                f"Invalid type for optional {key}: '{val}' (expected {typ}), using default"
            )

    logger.info("All required environment variables are valid.")


def check_all():
    # Check environment variables
    _validate_env()

    # Check DB schema
    db_path = os.getenv("DB_PATH", "total.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _ensure_schema(conn)
        logger.info("Database schema is up to date.")


if __name__ == "__main__":
    check_all()
