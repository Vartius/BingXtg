"""
This module provides thread-safe utilities for reading and writing JSON data files,
which store the bot's configuration and operational state.
"""

import json
import threading
from loguru import logger
from typing import Any, Dict, Optional

# A lock to prevent race conditions when multiple threads access the same files.
_file_lock = threading.Lock()


def read_json(file_path: str) -> Optional[Dict | list]:
    """
    Safely reads and decodes a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The decoded JSON data as a dictionary or list, or None if an error occurs.
    """
    with _file_lock:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.debug(
                f"File not found: {file_path}. This may be expected on first run."
            )
            return None
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding JSON from {file_path}. The file might be corrupted or empty."
            )
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
            return None


def write_json(file_path: str, data: Any) -> bool:
    """
    Safely writes data to a JSON file with pretty printing.

    Args:
        file_path: The path to the JSON file.
        data: The Python object (e.g., dict, list) to write.

    Returns:
        True if the write was successful, False otherwise.
    """
    with _file_lock:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while writing to {file_path}: {e}"
            )
            return False


# --- State File (state.json) ---
def get_state() -> Optional[Dict | None | list]:
    """Reads the primary state file (state.json)."""
    data = read_json("data/state.json")
    if data is None:
        logger.warning("State file not found or invalid.")
        return None
    return data


def save_state(data: Dict) -> bool:
    """Saves data to the primary state file (state.json)."""
    return write_json("data/state.json", data)


# --- Winrate File (winrate.json) ---
def get_winrate() -> Optional[Dict | None | list]:
    """Reads the channel winrate data file (winrate.json)."""
    data = read_json("data/winrate.json")
    if data is None:
        logger.warning("Winrate file not found or invalid.")
        return None
    return data


def save_winrate(data: Dict) -> bool:
    """Saves data to the channel winrate data file (winrate.json)."""
    return write_json("data/winrate.json", data)


# --- Channels File (channels.json) ---
def get_channels() -> Optional[Dict | None | list]:
    """Reads the Telegram channel configuration file (channels.json)."""
    data = read_json("data/channels.json")
    if data is None:
        logger.warning("Channels file not found or invalid.")
        return None
    return data


# --- Table Data File (table.json) ---
def save_table(data: Dict) -> bool:
    """Saves the data used by the GUI to table.json."""
    return write_json("data/table.json", data)
