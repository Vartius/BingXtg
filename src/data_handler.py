"""
This module handles reading and writing data to JSON files.
"""

import json
from loguru import logger


def read_json(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def write_json(file_path, data):
    """Writes data to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error writing to {file_path}: {e}")


def get_curdata():
    """Reads and returns data from curdata.json."""
    return read_json("data/curdata.json")


def save_curdata(data):
    """Saves data to curdata.json."""
    write_json("data/curdata.json", data)


def get_winrate():
    """Reads and returns data from winrate.json."""
    return read_json("data/winrate.json")


def save_winrate(data):
    """Saves data to winrate.json."""
    write_json("data/winrate.json", data)


def get_channels():
    """Reads and returns data from channels.json."""
    return read_json("data/channels.json")


def save_table(data):
    """Saves data to table.json."""
    write_json("data/table.json", data)
