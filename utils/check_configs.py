import os
import sqlite3
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Required environment variables and their expected types
ENV_SCHEMA = {
    # Telegram API
    "API_ID": int,
    "API_HASH": str,
    "FOLDER_ID": int,
    # BingX API
    "APIURL": str,
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
    # Paths (optional defaults)
    "DB_PATH": str,
    "MODEL_DIR": str,
    "SESSION_FILE": str,
    "DATA_DIR": str,
    # Logging
    "LOG_LEVEL": str,
    # Redis
    "REDIS_HOST": str,
    "REDIS_PORT": int,
    # Email (optional)
    "EMAIL_HOST": str,
    "EMAIL_PORT": int,
    "EMAIL_USE_TLS": bool,
    "EMAIL_HOST_USER": str,
    "EMAIL_HOST_PASSWORD": str,
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
    missing = []
    wrong_types = []

    for key, expected_type in ENV_SCHEMA.items():
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

    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")

    if wrong_types:
        for key, val, typ in wrong_types:
            logger.error(f"Invalid type for {key}: '{val}' (expected {typ})")

    if missing or wrong_types:
        raise EnvironmentError("Environment configuration is invalid!")

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
