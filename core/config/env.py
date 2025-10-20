"""
Environment configuration helper module.

Provides centralized access to environment variables with proper defaults
and type conversion.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: Any = None, var_type: type = str) -> Any:
    """
    Get environment variable with type conversion and default value.

    Args:
        key: Environment variable name
        default: Default value if not set
        var_type: Expected type (str, int, float, bool)

    Returns:
        Environment variable value converted to the specified type
    """
    val = os.getenv(key)

    if val is None or val.strip() == "":
        return default

    try:
        if var_type is bool:
            return val.lower() in ("true", "1", "yes", "on")
        elif var_type is int:
            return int(val)
        elif var_type is float:
            return float(val)
        else:
            return val
    except (ValueError, AttributeError):
        return default


# Required environment variables
API_ID = get_env("API_ID", var_type=int)
API_HASH = get_env("API_HASH")
APIKEY = get_env("APIKEY")
SECRETKEY = get_env("SECRETKEY")
LEVERAGE = get_env("LEVERAGE", var_type=int)
SL = get_env("SL", var_type=float)
TP = get_env("TP", var_type=float)
START_BALANCE = get_env("START_BALANCE", var_type=float)
MIN_ORDERS_TO_HIGH = get_env("MIN_ORDERS_TO_HIGH", var_type=int)
MAX_PERCENT = get_env("MAX_PERCENT", var_type=float)
DJANGO_SECRET_KEY = get_env("DJANGO_SECRET_KEY")
DJANGO_DEBUG = get_env("DJANGO_DEBUG", default=True, var_type=bool)
DJANGO_ALLOWED_HOSTS = get_env("DJANGO_ALLOWED_HOSTS", default="localhost,127.0.0.1")

# Optional environment variables with defaults
FOLDER_ID = get_env("FOLDER_ID", default=1, var_type=int)
APIURL = get_env("APIURL", default="https://open-api.bingx.com")
DB_PATH = get_env("DB_PATH", default="total.db")
MODEL_DIR = get_env("MODEL_DIR", default="ai_model")
SESSION_FILE = get_env("SESSION_FILE", default="my_account.session")
DATA_DIR = get_env("DATA_DIR", default="data")
LOG_LEVEL = get_env("LOG_LEVEL", default="INFO")
REDIS_HOST = get_env("REDIS_HOST", default="127.0.0.1")
REDIS_PORT = get_env("REDIS_PORT", default=6379, var_type=int)
EMAIL_HOST = get_env("EMAIL_HOST", default="")
EMAIL_PORT = get_env("EMAIL_PORT", default=587, var_type=int)
EMAIL_USE_TLS = get_env("EMAIL_USE_TLS", default=True, var_type=bool)
EMAIL_HOST_USER = get_env("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = get_env("EMAIL_HOST_PASSWORD", default="")

# AI Model paths (optional, automatic fallback)
HF_CLASSIFIER_MODEL_PATH = get_env(
    "HF_CLASSIFIER_MODEL_PATH", default="ai/models/signal_classifier"
)
HF_NER_MODEL_PATH = get_env("HF_NER_MODEL_PATH", default="ai/models/ner_extractor")
IS_SIGNAL_MODEL_PATH = get_env(
    "IS_SIGNAL_MODEL_PATH", default="ai/models/is_signal_model"
)
DIRECTION_MODEL_PATH = get_env(
    "DIRECTION_MODEL_PATH", default="ai/models/direction_model"
)
NER_MODEL_PATH = get_env("NER_MODEL_PATH", default="ai/models/ner_model")
