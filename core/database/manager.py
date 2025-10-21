"""
Database Manager Module

This module provides a centralized interface for all database operations
used throughout the RegexGenAI project.
"""

import os
import sqlite3
import logging
from typing import Optional, List, Dict, Any, Any as _Any
from pathlib import Path

# Set up logging for this module
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Centralized database manager for all SQL operations in the project.

    Handles database initialization, connection management, and all CRUD
    operations for messages and labels.
    """

    def __init__(self, db_path: os.PathLike[str] | str = "total.db"):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = str(db_path)
        self._ensure_database_exists()
        # Auto-initialize tables if file was just created or tables missing
        try:
            self.init_database()
        except Exception:
            # In case of race conditions or permissions, log and continue
            logger.exception("Auto initialization of database failed.")

    def _ensure_database_exists(self) -> None:
        """Ensure the database file and its parent directory exist."""
        try:
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row_factory for dict-like access.

        Returns:
            A new sqlite3.Connection object.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute_query(
        self, query: str, params: tuple = (), fetch_type: str = "none"
    ) -> _Any:
        """
        Execute a database query with proper context management and error handling.

        Args:
            query: The SQL query string to execute.
            params: A tuple of parameters to substitute into the query.
            fetch_type: Type of fetch operation ("one", "all", or "none").

        Returns:
            The result of the query based on fetch_type.

        Raises:
            sqlite3.Error: If a database-related error occurs.
            ValueError: If an invalid fetch_type is provided.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                if fetch_type == "one":
                    return cursor.fetchone()
                elif fetch_type == "all":
                    return cursor.fetchall()
                elif fetch_type == "none":
                    conn.commit()
                    return cursor.rowcount
                else:
                    raise ValueError(f"Invalid fetch_type specified: {fetch_type}")

        except sqlite3.OperationalError as e:
            # Attempt to self-heal if tables are missing
            if "no such table" in str(e).lower():
                logger.warning(
                    "Database missing tables; initializing now and retrying query."
                )
                self.init_database()
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    if fetch_type == "one":
                        return cursor.fetchone()
                    elif fetch_type == "all":
                        return cursor.fetchall()
                    elif fetch_type == "none":
                        conn.commit()
                        return cursor.rowcount
            logger.error(
                f"Database error {e} executing query: {query[:100]}...", exc_info=True
            )
            raise
        except sqlite3.Error as e:
            logger.error(
                f"Database error {e} executing query: {query[:100]}...", exc_info=True
            )
            raise

    def _get_table_columns(self, table_name: str) -> set[str]:
        """Return a set of column names for the given table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                rows = cursor.fetchall()
        except sqlite3.Error:
            logger.exception(f"Failed to read table info for '{table_name}'")
            return set()

        cols: set[str] = set()
        for row in rows:
            try:
                name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
            except Exception:
                name = row[1] if len(row) > 1 else None
            if name:
                cols.add(str(name))
        return cols

    # ==================== DATABASE INITIALIZATION ====================
    def init_database(self) -> None:
        """Initialize the database and create all required tables."""
        logger.info(f"Initializing database at {self.db_path}...")

        # Enable WAL mode and foreign keys
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")

        # Create all tables
        self.init_channels_table()
        self.init_messages_table()
        self.init_labeled_table()
        self.init_app_state_table()
        self.init_trades_table()
        self.init_trading_stats_table()

        # Create indexes and triggers
        self.init_indexes()
        self.init_triggers()

        logger.info("Database initialized successfully.")

    def init_messages_table(self) -> None:
        """Create the 'messages' table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                is_signal BOOLEAN NOT NULL DEFAULT 0,
                regex TEXT
            )
        """
        self._execute_query(query)
        logger.debug("Table 'messages' is ready.")

    def init_labeled_table(self) -> None:
        """Create the 'labeled' table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS labeled (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    is_signal BOOLEAN NOT NULL,
                    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, direction INTEGER, pair TEXT, stop_loss REAL, take_profit REAL, leverage REAL, targets TEXT, entry REAL, ai_is_signal INTEGER, ai_confidence REAL, ai_direction TEXT, ai_pair TEXT, ai_stop_loss REAL, ai_leverage REAL, ai_targets TEXT, ai_entry REAL, ai_processed_at TIMESTAMP, pair_start INTEGER, pair_end INTEGER, stop_loss_start INTEGER, stop_loss_end INTEGER, leverage_start INTEGER, leverage_end INTEGER, target_4_start INTEGER, target_4_end INTEGER, target_1_start INTEGER, target_1_end INTEGER, target_2_start INTEGER, target_2_end INTEGER, target_3_start INTEGER, target_3_end INTEGER, target_5_start INTEGER, target_5_end INTEGER, target_6_start INTEGER, target_6_end INTEGER, target_7_start INTEGER, target_7_end INTEGER, target_8_start INTEGER, target_8_end INTEGER, target_9_start INTEGER, target_9_end INTEGER, target_10_start INTEGER, target_10_end INTEGER, target_11_start INTEGER, target_11_end INTEGER, target_12_start INTEGER, target_12_end INTEGER, target_13_start INTEGER, target_13_end INTEGER, target_14_start INTEGER, target_14_end INTEGER, target_15_start INTEGER, target_15_end INTEGER, target_16_start INTEGER, target_16_end INTEGER, target_17_start INTEGER, target_17_end INTEGER, entry_start INTEGER, entry_end INTEGER,
                    FOREIGN KEY (message_id) REFERENCES messages (id)
                )
        """
        self._execute_query(query)
        logger.debug("Table 'labeled' is ready.")

    def init_channels_table(self) -> None:
        """Create the 'channels' table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS channels (
                channel_id INTEGER PRIMARY KEY,
                title TEXT,
                username TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self._execute_query(query)
        logger.debug("Table 'channels' is ready.")

    def init_indexes(self) -> None:
        """Create all required indexes for the database."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_labeled_message_id ON labeled(message_id)",
            "CREATE INDEX IF NOT EXISTS idx_labeled_channel_id ON labeled(channel_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_labeled_message_id_unique ON labeled(message_id)",
        ]
        for index_query in indexes:
            try:
                self._execute_query(index_query)
            except Exception as e:
                logger.warning(f"Error creating index: {e}")
        logger.debug("Database indexes created.")

    def init_triggers(self) -> None:
        """Create all required triggers for the database."""
        # Trigger to auto-update updated_at on channels table
        trigger_query = """
            CREATE TRIGGER IF NOT EXISTS trg_channels_updated_at
        AFTER UPDATE ON channels
        BEGIN
          UPDATE channels SET updated_at = CURRENT_TIMESTAMP WHERE channel_id = NEW.channel_id;
        END
        """
        try:
            self._execute_query(trigger_query)
            logger.debug("Database triggers created.")
        except Exception as e:
            logger.warning(f"Error creating triggers: {e}")

    # ==================== APP STATE (PERSISTENT) ====================
    def init_app_state_table(self) -> None:
        """Create a simple key-value table for app-level state (counters, pointers)."""
        query = """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        self._execute_query(query)
        logger.debug("Table 'app_state' is ready.")

    def init_trades_table(self) -> None:
        """Create the 'trades' table if it doesn't exist."""
        query = """
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
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, activated_at TIMESTAMP,
              FOREIGN KEY(channel_id) REFERENCES channels(channel_id) ON DELETE SET NULL
            )
        """
        self._execute_query(query)

        # Create index for trades
        try:
            self._execute_query(
                "CREATE INDEX IF NOT EXISTS idx_trades_channel_id ON trades(channel_id)"
            )
        except Exception as e:
            logger.warning(f"Error creating trades index: {e}")

        logger.debug("Table 'trades' is ready.")

    def init_trading_stats_table(self) -> None:
        """Create the 'trading_stats' table if it doesn't exist."""
        query = """
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
        """
        self._execute_query(query)
        logger.debug("Table 'trading_stats' is ready.")

    def get_app_state(self, key: str) -> Optional[str]:
        """Get a value from app_state by key."""
        row = self._execute_query(
            "SELECT value FROM app_state WHERE key = ?", (key,), "one"
        )
        return row["value"] if row else None

    def set_app_state(self, key: str, value: str) -> None:
        """Upsert a key-value pair into app_state."""
        self._execute_query(
            "INSERT INTO app_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
            "none",
        )

    # ==================== MESSAGE OPERATIONS ====================
    def save_message(self, channel_id: int, message_text: str) -> int:
        """
        Save a message to the messages table, ignoring duplicates.

        Args:
            channel_id: Telegram channel ID.
            message_text: Message content.

        Returns:
            The ID of the inserted message.
        """
        query = "INSERT OR IGNORE INTO messages (channel_id, message) VALUES (?, ?)"
        self._execute_query(query, (channel_id, message_text))

        with self._get_connection() as conn:
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def upsert_channel(
        self,
        channel_id: int,
        title: Optional[str] = None,
        username: Optional[str] = None,
    ) -> None:
        """Insert or update a channel's metadata."""
        query = """
            INSERT INTO channels (channel_id, title, username)
            VALUES (?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                title=excluded.title,
                username=excluded.username,
                updated_at=CURRENT_TIMESTAMP
        """
        self._execute_query(query, (channel_id, title, username), "none")

    def get_unlabeled_messages(self, limit: int = 100) -> List[sqlite3.Row]:
        """
        Get messages that haven't been labeled yet.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            A list of unlabeled message rows.
        """
        query = """
            SELECT m.*
            FROM messages m
            LEFT JOIN labeled l ON m.id = l.message_id
            WHERE l.message_id IS NULL
            ORDER BY RANDOM()
            LIMIT ?
        """
        return self._execute_query(query, (limit,), "all")

    def get_message_by_id(self, message_id: int) -> Optional[sqlite3.Row]:
        """Retrieve a single message by its identifier."""
        query = "SELECT * FROM messages WHERE id = ?"
        return self._execute_query(query, (message_id,), "one")

    def get_random_unlabeled_message_from_channel(
        self, channel_id: int
    ) -> Optional[sqlite3.Row]:
        """
        Get a random unlabeled message from a specific channel.

        Args:
            channel_id: The ID of the channel to query.

        Returns:
            A single random unlabeled message row, or None if none are available.
        """
        query = """
            SELECT m.*
            FROM messages m
            LEFT JOIN labeled l ON m.id = l.message_id
            WHERE l.message_id IS NULL AND m.channel_id = ?
            ORDER BY RANDOM()
            LIMIT 1
        """
        return self._execute_query(query, (channel_id,), "one")

    # ==================== LABEL OPERATIONS ====================

    def get_label_by_message_id(self, message_id: int) -> Optional[sqlite3.Row]:
        """Retrieve a labeled row corresponding to a message identifier."""
        query = "SELECT * FROM labeled WHERE message_id = ?"
        return self._execute_query(query, (message_id,), "one")

    def save_label(
        self,
        message_id: int,
        channel_id: int,
        message: str,
        is_signal: bool,
        direction: Optional[int] = None,
        pair: Optional[str] = None,
        stop_loss: Optional[float] = None,
        leverage: Optional[float] = None,
        targets: Optional[str] = None,
        entry: Optional[float] = None,
    ) -> int:
        """
        Save a label for a message, replacing any existing label for that message.

        Args:
            message_id: ID of the message being labeled.
            channel_id: Channel ID of the message.
            message: The text of the message.
            is_signal: The label (True for signal, False for not signal).
            direction: Direction (0: LONG, 1: SHORT) - only if is_signal=True
            pair: Trading pair (e.g., "btc/usdt") - only if is_signal=True
            stop_loss: Stop loss value - only if is_signal=True
            leverage: Leverage value - only if is_signal=True
            targets: JSON string of target array - only if is_signal=True
            entry: Entry price - only if is_signal=True

        Returns:
            The ID of the inserted label.
        """
        # If not a signal, force all other fields to None
        if not is_signal:
            direction = pair = stop_loss = leverage = targets = entry = None

        query = """
            INSERT OR REPLACE INTO labeled 
            (message_id, channel_id, message, is_signal, direction, pair, stop_loss, 
             leverage, targets, entry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query(
            query,
            (
                message_id,
                channel_id,
                message,
                is_signal,
                direction,
                pair,
                stop_loss,
                leverage,
                targets,
                entry,
            ),
        )

        with self._get_connection() as conn:
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def update_label_by_id(
        self,
        labeled_id: int,
        is_signal: bool,
        direction: Optional[int] = None,
        pair: Optional[str] = None,
        stop_loss: Optional[float] = None,
        leverage: Optional[float] = None,
        targets: Optional[str] = None,
        entry: Optional[float] = None,
    ) -> int:
        """
        Update an existing labeled row by its primary key id. When is_signal is False,
        extended fields are nulled. Updates labeled_at to CURRENT_TIMESTAMP.

        Args:
            labeled_id: Primary key id in labeled table
            is_signal: Whether the message is a signal
            direction, pair, stop_loss, leverage, targets, entry: Extended fields
        Returns:
            Number of rows updated (0 or 1)
        """
        if not is_signal:
            direction = pair = stop_loss = leverage = targets = entry = None
        query = """
            UPDATE labeled
               SET is_signal = ?,
                   direction = ?,
                   pair = ?,
                   stop_loss = ?,
                   leverage = ?,
                   targets = ?,
                   entry = ?,
                   labeled_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """
        return int(
            self._execute_query(
                query,
                (
                    is_signal,
                    direction,
                    pair,
                    stop_loss,
                    leverage,
                    targets,
                    entry,
                    labeled_id,
                ),
                "none",
            )
        )

    def get_labeled_data(self) -> List[sqlite3.Row]:
        """
        Get all labeled data, ordered by when they were labeled.

        Returns:
            A list of all labeled data rows.
        """
        query = "SELECT message, is_signal FROM labeled ORDER BY labeled_at"
        return self._execute_query(query, (), "all")

    def get_extended_labeled_data(self) -> List[sqlite3.Row]:
        """
        Get all labeled data with extended fields, ordered by when they were labeled.

        Returns:
            A list of all labeled data rows with all fields.
        """
        query = """
            SELECT message, is_signal, direction, pair, stop_loss, 
                   leverage, targets, entry, labeled_at 
            FROM labeled 
            ORDER BY labeled_at
        """
        return self._execute_query(query, (), "all")

    def get_labeled_message_by_id(self, message_id: int) -> Optional[sqlite3.Row]:
        """
        Get a specific labeled message by its message_id.

        Args:
            message_id: ID of the message to retrieve.

        Returns:
            The labeled message row or None if not found.
        """
        query = """
            SELECT * FROM labeled WHERE message_id = ?
        """
        return self._execute_query(query, (message_id,), "one")

    # ==================== STATISTICS AND ANALYTICS ====================
    def get_available_channels(self) -> List[int]:
        """
        Get a list of distinct channel IDs that still have unlabeled messages.

        Returns:
            A list of integer channel IDs.
        """
        query = """
            SELECT DISTINCT m.channel_id
            FROM messages m
            LEFT JOIN labeled l ON m.id = l.message_id
            WHERE l.message_id IS NULL
            ORDER BY m.channel_id
        """
        rows = self._execute_query(query, (), "all")
        return [row["channel_id"] for row in rows]

    def get_labeling_stats(self) -> Dict[str, int]:
        """
        Get comprehensive statistics about the state of labeling.

        Returns:
            A dictionary containing total, labeled, and signal message counts.
        """
        query = """
            SELECT
                (SELECT COUNT(*) FROM messages) as total,
                (SELECT COUNT(*) FROM labeled) as labeled,
                (SELECT COUNT(*) FROM labeled WHERE is_signal = 1) as signals
        """
        stats = self._execute_query(query, (), "one")
        return dict(stats) if stats else {"total": 0, "labeled": 0, "signals": 0}

    def get_channel_stats(self) -> List[Dict[str, Any]]:
        """
        Get per-channel statistics on total and labeled messages, with channel metadata.

        Returns:
            A list of dictionaries, each representing a channel's stats.
        """
        query = """
            SELECT 
                m.channel_id,
                c.title AS title,
                c.username AS username,
                COUNT(m.id) AS total_messages,
                COUNT(l.id) AS labeled_messages
            FROM messages m
            LEFT JOIN labeled l ON m.id = l.message_id
            LEFT JOIN channels c ON c.channel_id = m.channel_id
            GROUP BY m.channel_id
            ORDER BY m.channel_id
        """
        rows = self._execute_query(query, (), "all")
        return [dict(row) for row in rows]

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get statistics relevant for AI model training readiness.

        Returns:
            A dictionary with detailed stats and recommendations.
        """
        stats = self.get_labeling_stats()
        labeled_count = stats.get("labeled", 0)
        signal_count = stats.get("signals", 0)
        non_signal_count = labeled_count - signal_count

        min_samples = 20
        min_per_class = 5

        ready = (
            labeled_count >= min_samples
            and signal_count >= min_per_class
            and non_signal_count >= min_per_class
        )

        action = "Ready for training."
        if not ready:
            reasons = []
            if labeled_count < min_samples:
                reasons.append(
                    f"need at least {min_samples} total labeled messages (have {labeled_count})"
                )
            if signal_count < min_per_class:
                reasons.append(
                    f"need at least {min_per_class} signals (have {signal_count})"
                )
            if non_signal_count < min_per_class:
                reasons.append(
                    f"need at least {min_per_class} non-signals (have {non_signal_count})"
                )
            action = f"Continue labeling: {', '.join(reasons)}."

        return {
            "total_labeled": labeled_count,
            "signal_count": signal_count,
            "non_signal_count": non_signal_count,
            "ready_for_training": ready,
            "recommended_action": action,
        }

    # ==================== CHANNEL METADATA UTILITIES ====================
    def get_channels_missing_metadata(self) -> List[int]:
        """
        Return channel IDs that are missing metadata (title/username) or not present
        in the channels table at all.

        This helps trigger a Telegram backfill to populate names.
        """
        query = """
            SELECT DISTINCT channel_id FROM (
                -- Channels referenced by messages but missing in channels table
                SELECT m.channel_id AS channel_id
                FROM messages m
                LEFT JOIN channels c ON c.channel_id = m.channel_id
                WHERE c.channel_id IS NULL
                UNION
                -- Channels present but missing title and username
                SELECT c.channel_id AS channel_id
                FROM channels c
                WHERE COALESCE(c.title, '') = '' AND COALESCE(c.username, '') = ''
            )
            ORDER BY channel_id
        """
        rows = self._execute_query(query, (), "all")
        return [row["channel_id"] for row in rows]
