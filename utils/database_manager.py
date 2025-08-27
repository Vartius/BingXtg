import os
import sqlite3
import logging
from typing import Optional, List, Dict, Any, Any as _Any
from pathlib import Path

# Set up logging for this module
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: os.PathLike[str] | str = "messages.db"):
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

    # ==================== LABEL OPERATIONS ====================

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

    # ==================== STATISTICS AND ANALYTICS ====================

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
