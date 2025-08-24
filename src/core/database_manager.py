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

    def __init__(self, db_path: os.PathLike[str] | str = "messages.db"):
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

    # Helper: list columns of a table to gate ALTER TABLE operations
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
        self.init_channels_table()
        self.init_messages_table()
        self.init_labeled_table()
        # Initialize app_state for persistent app-level settings/counters
        self.init_app_state_table()
        logger.info("Database initialized successfully.")

    def init_messages_table(self) -> None:
        """Create the 'messages' table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                is_signal BOOLEAN NOT NULL DEFAULT 0,
                regex TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self._execute_query(query)
        logger.debug("Table 'messages' is ready.")

    def init_labeled_table(self) -> None:
        """Create the 'labeled' table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS labeled (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL UNIQUE,
                channel_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                is_signal BOOLEAN NOT NULL,
                direction INTEGER,
                pair TEXT,
                stop_loss REAL,
                leverage REAL,
                targets TEXT,
                entry REAL,
                labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages (id)
            )
        """
        self._execute_query(query)

        # Only add missing columns by inspecting the existing schema first
        try:
            existing = self._get_table_columns("labeled")
            columns_to_add = [
                ("direction", "INTEGER"),
                ("pair", "TEXT"),
                ("stop_loss", "REAL"),
                ("leverage", "REAL"),
                ("targets", "TEXT"),
                ("entry", "REAL"),
            ]

            for column_name, column_type in columns_to_add:
                if column_name not in existing:
                    self._execute_query(
                        f"ALTER TABLE labeled ADD COLUMN {column_name} {column_type}"
                    )
                    logger.debug(f"Added column '{column_name}' to labeled table.")
        except Exception as e:
            logger.warning(f"Error checking/adding columns for labeled table: {e}")

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

    def get_available_channels_for_extended_labeling(self) -> List[int]:
        """
        Get channels that have either incomplete signals or unlabeled messages.

        Returns:
            A list of integer channel IDs.
        """
        query = """
            SELECT DISTINCT channel_id FROM (
                -- Channels with incomplete signals (is_signal=True but missing extended fields)
                SELECT DISTINCT channel_id 
                FROM labeled 
                WHERE is_signal = 1 
                AND (direction IS NULL OR pair IS NULL OR stop_loss IS NULL 
                     OR leverage IS NULL OR targets IS NULL OR entry IS NULL)
                UNION
                -- Channels with unlabeled messages
                SELECT DISTINCT m.channel_id
                FROM messages m
                LEFT JOIN labeled l ON m.id = l.message_id
                WHERE l.message_id IS NULL
            )
            ORDER BY channel_id
        """
        rows = self._execute_query(query, (), "all")
        return [row["channel_id"] for row in rows]

    def get_random_incomplete_signal_from_channel(
        self, channel_id: int
    ) -> Optional[sqlite3.Row]:
        """
        Get a random incomplete signal from a specific channel.
        An incomplete signal is one where is_signal=True but any extended field is NULL.

        Args:
            channel_id: The ID of the channel to query.

        Returns:
            A single random incomplete signal row, or None if none are available.
        """
        query = """
            SELECT * FROM labeled 
            WHERE channel_id = ? AND is_signal = 1 
            AND (direction IS NULL OR pair IS NULL OR stop_loss IS NULL 
                 OR leverage IS NULL OR targets IS NULL OR entry IS NULL)
            ORDER BY RANDOM()
            LIMIT 1
        """
        return self._execute_query(query, (channel_id,), "one")

    def get_available_channels_for_empty_extended_labeling(self) -> List[int]:
        """
        Get channels that have signals with all extended fields missing (all NULL).

        Returns:
            A list of integer channel IDs.
        """
        query = """
            SELECT DISTINCT channel_id
            FROM labeled
            WHERE is_signal = 1
              AND direction IS NULL AND pair IS NULL AND stop_loss IS NULL
              AND leverage IS NULL AND targets IS NULL AND entry IS NULL
            ORDER BY channel_id
        """
        rows = self._execute_query(query, (), "all")
        return [row["channel_id"] for row in rows]

    def get_random_empty_extended_signal_from_channel(
        self, channel_id: int
    ) -> Optional[sqlite3.Row]:
        """
        Get a random signal from a specific channel where all extended fields are NULL.

        Args:
            channel_id: The ID of the channel to query.

        Returns:
            A single random labeled row with all extended fields NULL, or None if none are available.
        """
        query = """
            SELECT * FROM labeled
            WHERE channel_id = ? AND is_signal = 1
              AND direction IS NULL AND pair IS NULL AND stop_loss IS NULL
              AND leverage IS NULL AND targets IS NULL AND entry IS NULL
            ORDER BY RANDOM()
            LIMIT 1
        """
        return self._execute_query(query, (channel_id,), "one")

    # New: count incomplete extended signals (is_signal=1 with any missing extended field)
    def count_incomplete_extended_signals(self) -> int:
        row = self._execute_query(
            """
            SELECT COUNT(*) AS cnt
            FROM labeled 
            WHERE is_signal = 1 
              AND (direction IS NULL OR pair IS NULL OR stop_loss IS NULL 
                   OR leverage IS NULL OR targets IS NULL OR entry IS NULL)
            """,
            (),
            "one",
        )
        return int(row["cnt"]) if row else 0

    # ==================== SEQUENTIAL EXTENDED LABELING HELPERS ====================
    def get_next_incomplete_signal_after(
        self, last_labeled_row_id: Optional[int]
    ) -> Optional[sqlite3.Row]:
        """
        Return the next labeled row (is_signal=1) with any missing extended fields, ordered by labeled.id ascending,
        strictly after the given labeled row id. If last_labeled_row_id is None, returns the first such row.
        """
        params: tuple
        if last_labeled_row_id is None:
            query = """
                SELECT * FROM labeled
                WHERE is_signal = 1
                  AND (direction IS NULL OR pair IS NULL OR stop_loss IS NULL
                       OR leverage IS NULL OR targets IS NULL OR entry IS NULL)
                ORDER BY id ASC
                LIMIT 1
            """
            params = ()
        else:
            query = """
                SELECT * FROM labeled
                WHERE id > ? AND is_signal = 1
                  AND (direction IS NULL OR pair IS NULL OR stop_loss IS NULL
                       OR leverage IS NULL OR targets IS NULL OR entry IS NULL)
                ORDER BY id ASC
                LIMIT 1
            """
            params = (last_labeled_row_id,)
        return self._execute_query(query, params, "one")

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
