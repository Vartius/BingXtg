"""
Configuration utilities for the project.
Loads configuration from environment variables.
"""

# !CHECK AI GENERATED BULLSHIT
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database configuration
DB_PATH = os.getenv("DB_PATH", "messages.db")
if not Path(DB_PATH).is_absolute():
    DB_PATH = PROJECT_ROOT / DB_PATH

# Model directory configuration
MODEL_DIR = os.getenv("MODEL_DIR", "ai_model")
if not Path(MODEL_DIR).is_absolute():
    MODEL_DIR = PROJECT_ROOT / MODEL_DIR

# Session file configuration
SESSION_FILE = os.getenv("SESSION_FILE", "my_account.session")
if not Path(SESSION_FILE).is_absolute():
    SESSION_FILE = PROJECT_ROOT / SESSION_FILE

# Data directory configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
if not Path(DATA_DIR).is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR


def setup_logging():
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs.log", mode="a")],
    )

    # Set specific loggers to reduce noise
    logging.getLogger("django").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
