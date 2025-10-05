import sqlite3
import google.generativeai as genai
import json
import os
import time
import logging
import asyncio
import random
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import groq
import cohere

# --- TEXTUAL TUI SYSTEM ---
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Log, ProgressBar
from textual.reactive import reactive
from textual.worker import Worker
from textual import on

# --- CONFIGURATION (Identical to original script) ---

# Batch processing configuration
BATCH_SIZE = 10
MAX_RETRIES = 5
BASE_DELAY = 2

# Database file name
DB_FILE = "total.db"

# Model configuration
MODEL_CONFIGS = [
    {"type": "gemini", "name": "gemini-1.5-flash"},
    {"type": "gemini", "name": "gemini-1.5-flash-8b"},
    {"type": "gemini", "name": "gemini-2.0-flash-lite"},
    {"type": "gemini", "name": "gemini-2.5-flash"},
    {"type": "github", "name": "deepseek-v3-0324"},
    {"type": "github", "name": "gpt-4.1"},
    {"type": "github", "name": "gpt-4.1-mini"},
    {"type": "github", "name": "gpt-4.1-nano"},
    {"type": "github", "name": "gpt-4o"},
    {"type": "github", "name": "gpt-4o-mini"},
    {"type": "github", "name": "gpt-5-chat"},
    {"type": "github", "name": "grok-3"},
    {"type": "github", "name": "grok-3-mini"},
    {"type": "github", "name": "llama-3.3-70b-instruct"},
    {"type": "github", "name": "llama-4-maverick-17b-128e-instruct-fp8"},
    {"type": "github", "name": "meta-llama-3.1-405b-instruct"},
    {"type": "github", "name": "mistral-large-2411"},
    {"type": "github", "name": "o1-mini"},
    {"type": "github", "name": "o1-preview"},
    {"type": "github", "name": "phi-4"},
    {"type": "github", "name": "phi-4-mini-instruct"},
    {"type": "github", "name": "phi-4-mini-reasoning"},
    {"type": "github", "name": "phi-4-reasoning"},
]

# Global variables
current_model_index = 0
AVAILABLE_MODEL_CONFIGS = []
current_model_instance = None
processing_stats = {
    "processed": 0,
    "signals_found": 0,
    "errors": 0,
    "current_model": "None",
}

# Global API clients
github_client = None
groq_client = None
together_client = None
anthropic_client = None
cohere_client = None
perplexity_client = None

# --- DATABASE AND API LOGIC (Largely unchanged from original script) ---
# Note: Functions that printed to the console have been adapted to log messages
# or post updates to the Textual app.


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Global reference to the app instance for logging
app_instance = None


def dialog_message(message: str, msg_type: str = "info") -> None:
    """Send a message to the Textual app log."""
    global app_instance
    if app_instance:
        app_instance.add_log_message(message, msg_type)
    else:
        # Fallback to logger if app not available
        if msg_type == "error":
            logger.error(message)
        elif msg_type == "warning":
            logger.warning(message)
        else:
            logger.info(message)


def show_dialog() -> None:
    """Placeholder for show_dialog - not needed in Textual."""
    pass


def dialog_signal_panel(data: dict, message_id: str) -> str:
    """Create a formatted signal panel for display."""
    pair = data.get("pair", "N/A")
    direction = data.get("direction", "N/A")
    entry = data.get("entry", "N/A")
    targets = data.get("targets", [])
    stop_loss = data.get("stop_loss", "N/A")
    leverage = data.get("leverage", "N/A")

    return f"""
🎯 SIGNAL DETECTED (ID: {message_id})
💰 Pair: {pair}
📈 Direction: {direction}
🎪 Entry: {entry}
🎯 Targets: {targets}
🛑 Stop Loss: {stop_loss}
⚡ Leverage: {leverage}
"""


def update_model_display_info(model_config: dict, status: str) -> None:
    """Update model display info - handled by Textual app."""
    pass


def exponential_backoff_delay(attempt: int, base_delay: float = BASE_DELAY) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = base_delay * (2**attempt)
    # Add jitter to avoid thundering herd problem
    jitter = delay * 0.1 * (2 * time.time() % 1 - 1)  # ±10% jitter
    return delay + jitter


def setup_database(db_file):
    """Initializes the database connection and ensures proper constraints on the 'labeled' table."""
    logger.info("🗄️  Setting up database connection...")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if labeled table has unique constraint on message_id
        cursor.execute("PRAGMA index_list(labeled)")
        indexes = cursor.fetchall()

        has_unique_constraint = False
        for index in indexes:
            cursor.execute(f"PRAGMA index_info('{index[1]}')")
            index_info = cursor.fetchall()
            if (
                len(index_info) == 1
                and index_info[0][2] == "message_id"
                and index[2] == 1
            ):  # unique index
                has_unique_constraint = True
                break

        # Add unique constraint if it doesn't exist
        if not has_unique_constraint:
            try:
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_labeled_message_id_unique ON labeled(message_id)"
                )
                logger.info("✅ Added unique constraint on labeled.message_id")
            except sqlite3.IntegrityError as e:
                logger.error(
                    f"❌ Cannot add unique constraint due to existing duplicates: {e}"
                )
                logger.info("🔧 Checking for duplicate message_ids in labeled table...")

                # Find and report duplicates
                cursor.execute("""
                    SELECT message_id, COUNT(*) as count 
                    FROM labeled 
                    GROUP BY message_id 
                    HAVING COUNT(*) > 1
                """)
                duplicates = cursor.fetchall()

                if duplicates:
                    logger.warning(
                        f"⚠️  Found {len(duplicates)} message_ids with duplicates:"
                    )
                    for msg_id, count in duplicates[:5]:  # Show first 5
                        logger.warning(f"   Message ID {msg_id}: {count} entries")
                    if len(duplicates) > 5:
                        logger.warning(f"   ... and {len(duplicates) - 5} more")

                    logger.info(
                        "💡 Consider cleaning up duplicates before running the labeling process"
                    )
                else:
                    logger.info("✅ No duplicates found, constraint should work")

        conn.commit()
        logger.info(
            f"✅ Database connection established: [bold green]{db_file}[/bold green]"
        )
        return conn, cursor
    except Exception as e:
        logger.error(f"❌ Failed to setup database: {e}")
        raise


def get_already_labeled_ids(cursor):
    """Fetches the IDs of messages that are already in the 'labeled' table."""
    logger.info("🔍 Fetching already labeled message IDs...")
    try:
        cursor.execute("SELECT message_id FROM labeled")
        labeled_ids = {row[0] for row in cursor.fetchall()}
        logger.info(
            f"📊 Found [bold yellow]{len(labeled_ids)}[/bold yellow] previously labeled messages"
        )
        return labeled_ids
    except Exception as e:
        logger.error(f"❌ Failed to fetch labeled IDs: {e}")
        return set()


def get_unlabeled_messages(cursor):
    """Retrieves messages from the 'messages' table that have not been labeled yet."""
    logger.info("📋 Fetching unlabeled messages...")
    try:
        # Use a more efficient approach with LEFT JOIN to avoid SQL variable limits
        # This avoids the "too many SQL variables" error when there are many labeled IDs
        query = """
            SELECT m.id, m.channel_id, m.message 
            FROM messages m 
            LEFT JOIN labeled l ON m.id = l.message_id 
            WHERE l.message_id IS NULL
        """
        cursor.execute(query)
        messages = cursor.fetchall()
        logger.info(
            f"📝 Found [bold cyan]{len(messages)}[/bold cyan] unlabeled messages to process"
        )
        return messages
    except Exception as e:
        logger.error(f"❌ Failed to fetch unlabeled messages: {e}")
        return []


def save_to_database(conn, cursor, message_id, channel_id, message_text, data):
    """Saves the extracted signal data into the 'labeled' table."""
    try:
        # First check if this message_id already exists in labeled table
        cursor.execute(
            "SELECT COUNT(*) FROM labeled WHERE message_id = ?", (message_id,)
        )
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            return True  # Consider it successful since it's already processed

        is_signal = data.get("is_signal", False)

        # We even save non-signals to avoid re-processing them in the future
        if is_signal:
            cursor.execute(
                """
            INSERT INTO labeled (message_id, channel_id, message, is_signal, pair, direction, entry, targets, stop_loss, leverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    channel_id,
                    message_text,
                    True,
                    data.get("pair"),
                    data.get("direction"),
                    data.get("entry"),
                    # Convert list of targets to a string representation
                    json.dumps(data.get("targets", [])),
                    data.get("stop_loss"),
                    data.get("leverage"),
                ),
            )

            # Show signal in dialog
            pair = data.get("pair", "Unknown")
            direction = data.get("direction", "Unknown")
            dialog_message(
                f"🎯 SIGNAL FOUND: {pair} {direction} (ID: {message_id})", "signal"
            )
            processing_stats["signals_found"] += 1

            # Log signal details
            logger.info(f"Signal found: {dialog_signal_panel(data, message_id)}")

            # Optionally show raw message (for debugging)
            # dialog_show_raw_message(message_id, message_text, data)

        else:
            # If it's not a signal, we still mark it as processed.
            cursor.execute(
                """
            INSERT INTO labeled (message_id, channel_id, message, is_signal)
            VALUES (?, ?, ?, ?)
            """,
                (message_id, channel_id, message_text, False),
            )

        conn.commit()
        return True

    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            return True  # Consider it successful since it's already processed
        else:
            dialog_message(
                f"Integrity error saving data for message {message_id}: {str(e)[:100]}",
                "error",
            )
            return False
    except Exception as e:
        dialog_message(
            f"Failed to save data for message {message_id}: {str(e)[:100]}", "error"
        )
        return False


def configure_apis():
    """Configures all AI APIs with the provided keys."""
    global \
        github_client, \
        groq_client, \
        together_client, \
        anthropic_client, \
        cohere_client, \
        perplexity_client
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Configure Gemini API
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            dialog_message("GOOGLE_API_KEY environment variable not set", "error")
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. Please create a .env file with GOOGLE_API_KEY=your_api_key_here"
            )

        genai.configure(api_key=gemini_api_key)  # type: ignore
        dialog_message("Gemini API configured successfully", "success")

        # Configure GitHub Copilot API
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            dialog_message("GITHUB_TOKEN environment variable not set", "error")
            raise ValueError(
                "GITHUB_TOKEN environment variable not set. Please add GITHUB_TOKEN=your_github_token_here to your .env file"
            )

        # Initialize GitHub Copilot client
        github_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=github_token,
        )
        dialog_message("GitHub Copilot API configured successfully", "success")

        # Configure Groq API (optional - free tier)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            groq_client = groq.Groq(api_key=groq_api_key)
            dialog_message("Groq API configured successfully", "success")
        else:
            dialog_message(
                "GROQ_API_KEY not set - Groq models will be skipped", "warning"
            )

        # Configure Together AI API (optional - free tier)
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if together_api_key:
            together_client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=together_api_key,
            )
            dialog_message("Together AI API configured successfully", "success")
        else:
            dialog_message(
                "TOGETHER_API_KEY not set - Together AI models will be skipped",
                "warning",
            )

        # Configure Anthropic API (optional - free tier)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            dialog_message("Anthropic API configured successfully", "success")
        else:
            dialog_message(
                "ANTHROPIC_API_KEY not set - Anthropic models will be skipped",
                "warning",
            )

        # Configure Cohere API (optional - free tier)
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            cohere_client = cohere.Client(api_key=cohere_api_key)
            dialog_message("Cohere API configured successfully", "success")
        else:
            dialog_message(
                "COHERE_API_KEY not set - Cohere models will be skipped", "warning"
            )

        # Configure Perplexity AI API (optional - free tier)
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_api_key:
            perplexity_client = OpenAI(
                base_url="https://api.perplexity.ai",
                api_key=perplexity_api_key,
            )
            dialog_message("Perplexity AI API configured successfully", "success")
        else:
            dialog_message(
                "PERPLEXITY_API_KEY not set - Perplexity AI models will be skipped",
                "warning",
            )

        return True
    except Exception as e:
        dialog_message(f"Failed to configure APIs: {e}", "error")
        logger.error(f"Failed to configure APIs: {e}")
        raise


def get_current_model():
    """Get the current model configuration based on the current model index."""
    global current_model_index, AVAILABLE_MODEL_CONFIGS
    if not AVAILABLE_MODEL_CONFIGS:
        raise ValueError(
            "AVAILABLE_MODEL_CONFIGS is empty. Make sure configure_apis() and get_available_models() are called first."
        )
    if current_model_index >= len(AVAILABLE_MODEL_CONFIGS):
        current_model_index = 0  # Reset to first model if we've exhausted all
    return AVAILABLE_MODEL_CONFIGS[current_model_index]


def switch_to_next_model():
    """Switch to the next model in the list."""
    global current_model_index, AVAILABLE_MODEL_CONFIGS, current_model_instance
    old_model_config = get_current_model()

    current_model_index += 1
    if current_model_index >= len(AVAILABLE_MODEL_CONFIGS):
        current_model_index = 0  # Reset to first model
        dialog_message("Exhausted all models, cycling back to first model", "warning")
        return False  # Indicate we've cycled through all models

    new_model_config = get_current_model()

    # Try to create new model instance
    try:
        current_model_instance = create_model_instance(new_model_config)
    except Exception as e:
        dialog_message(
            f"Failed to create instance for {new_model_config['name']}: {str(e)[:100]}",
            "error",
        )
        current_model_instance = None

    update_model_display_info(new_model_config, "Active")

    # Show model switch in dialog
    dialog_message(
        f"Model switch: {old_model_config['name']} → {new_model_config['name']} (Rate limit)",
        "model",
    )
    processing_stats["current_model"] = new_model_config["name"]

    return True


def create_model_instance(model_config):
    """Create a model instance based on the model configuration."""
    global \
        groq_client, \
        together_client, \
        anthropic_client, \
        cohere_client, \
        perplexity_client

    if model_config["type"] == "gemini":
        return genai.GenerativeModel(model_config["name"])  # type: ignore
    elif model_config["type"] == "github":
        return github_client
    elif model_config["type"] == "groq":
        if groq_client is None:
            logger.warning(
                f"⚠️  Groq client not configured, skipping model: {model_config['name']}"
            )
            raise ValueError("Groq client not configured")
        return groq_client
    elif model_config["type"] == "together":
        if together_client is None:
            logger.warning(
                f"⚠️  Together AI client not configured, skipping model: {model_config['name']}"
            )
            raise ValueError("Together AI client not configured")
        return together_client
    elif model_config["type"] == "anthropic":
        if anthropic_client is None:
            logger.warning(
                f"⚠️  Anthropic client not configured, skipping model: {model_config['name']}"
            )
            raise ValueError("Anthropic client not configured")
        return anthropic_client
    elif model_config["type"] == "cohere":
        if cohere_client is None:
            logger.warning(
                f"⚠️  Cohere client not configured, skipping model: {model_config['name']}"
            )
            raise ValueError("Cohere client not configured")
        return cohere_client
    elif model_config["type"] == "perplexity":
        if perplexity_client is None:
            logger.warning(
                f"⚠️  Perplexity AI client not configured, skipping model: {model_config['name']}"
            )
            raise ValueError("Perplexity AI client not configured")
        return perplexity_client
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")


def get_api_response(model_config, model_instance, prompt):
    """Get response from any of the supported AI APIs."""
    system_message = "You are a helpful assistant that analyzes cryptocurrency trading signals. Always respond with valid JSON only, no additional text."

    try:
        # Add debugging info
        logger.debug(
            f"API call: {model_config['type']} model {model_config['name']}, instance: {type(model_instance)}"
        )

        if model_config["type"] == "gemini":
            if model_instance is None:
                raise ValueError("Gemini model instance is None")
            response = model_instance.generate_content(prompt)
            return (
                response.text.strip().replace("```json", "").replace("```", "").strip()
            )

        elif model_config["type"] == "github":
            if model_instance is None:
                raise ValueError("GitHub model instance is None")
            response = model_instance.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            return (
                response.choices[0]
                .message.content.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif model_config["type"] == "groq":
            if model_instance is None:
                raise ValueError("Groq model instance is None")
            response = model_instance.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            return (
                response.choices[0]
                .message.content.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif model_config["type"] == "together":
            if model_instance is None:
                raise ValueError("Together model instance is None")
            response = model_instance.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            return (
                response.choices[0]
                .message.content.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif model_config["type"] == "anthropic":
            if model_instance is None:
                raise ValueError("Anthropic model instance is None")
            response = model_instance.messages.create(
                model=model_config["name"],
                max_tokens=4000,
                temperature=0.1,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
            )
            return (
                response.content[0]
                .text.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        elif model_config["type"] == "cohere":
            if model_instance is None:
                raise ValueError("Cohere model instance is None")
            response = model_instance.chat(
                model=model_config["name"],
                message=prompt,
                temperature=0.1,
                max_tokens=4000,
                preamble=system_message,
            )
            return (
                response.text.strip().replace("```json", "").replace("```", "").strip()
            )

        elif model_config["type"] == "perplexity":
            if model_instance is None:
                raise ValueError("Perplexity model instance is None")
            response = model_instance.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            return (
                response.choices[0]
                .message.content.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")

    except Exception as e:
        # Log to file, but don't show in dialog as this is handled by retry logic
        logger.error(
            f"API call failed for {model_config['type']} model {model_config['name']}: {e}"
        )
        raise


def create_batch_prompt(messages_batch):
    """Creates a batch prompt for processing multiple messages at once."""
    prompt_header = """
Analyze the following messages to determine if each is a cryptocurrency trading signal. Signal should be only CRYPTOCURRENCY, not forex or stocks. Signals that already completed label as non-signal. Mark signals only if you abosolutely sure that they are signals, otherwise mark as non-signal.
If message has mistake, then skip mistake place and extract the rest. For example if targets are 0.0010.002 0.003, 0.004 then extract [0.003, 0.004].
Your response MUST be a valid JSON array containing one object for each message, with no text before or after it.

Each JSON object should have the following structure:
{
  "message_id": "the provided message ID",
  "is_signal": boolean (0 or 1),
  "pair": "string (e.g., 'BTC', 'ETH', not a pair like 'BTC/USDT') or null",
  "direction": "'LONG' or 'SHORT' or null",
  "entry": "float (entry price if there are some of them, then only the first one) or null",
  "targets": "list of floats (take-profit prices) or []",
  "stop_loss": "float (stop-loss price, if there are some of them, then only the first one) or null",
  "leverage": "integer (e.g., 5, 10, 20), if there are some of them, then only the first one, or null"
}

Here are some examples:

Message: "Signal: #BTC LONG, Entry: 60000, Targets: [61000, 62000, 63000], SL: 59000, Leverage: 10x"
Response:
{
  "message_id": "example_id",
  "is_signal": true,
  "pair": "BTC",
  "direction": "LONG",
  "entry": 60000.0,
  "targets": [61000.0, 62000.0, 63000.0],
  "stop_loss": 59000.0,
  "leverage": 10
}

---
Now, analyze these messages:

"""

    messages_text = ""
    for msg_id, channel_id, message_text in messages_batch:
        messages_text += f'Message ID: {msg_id}\nMessage: "{message_text}"\n\n'

    prompt_footer = f"""
Response (JSON array with {len(messages_batch)} objects):
"""

    return prompt_header + messages_text + prompt_footer


def create_prompt(message_text):
    """Creates a detailed prompt for the Gemini model to extract structured data."""
    prompt = f"""
Analyze the following message to determine if it's a cryptocurrency trading signal. Signal should be only CRYPTOCURRENCY, not forex or stocks. Signals that already completed label as non-signal. Mark signals only if you abosolutely sure that they are signals, otherwise mark as non-signal.
Your response MUST be a valid JSON object, with no text before or after it.

The JSON object should have the following structure:
{{
  "is_signal": boolean,
  "pair": "string (e.g., 'BTC', 'ETH', not a pair like 'BTC/USDT') or null",
  "direction": "'LONG' or 'SHORT' or null",
  "entry": "float (entry price if there are some of them, then only the first one) or null",
  "targets": "list of floats (take-profit prices) or []",
  "stop_loss": "float (stop-loss price) or null",
  "leverage": "integer (e.g., 5, 10, 20) or null"
}}

Here are some examples:

Message: "Signal: #BTC LONG, Entry: 60000, Targets: [61000, 62000, 63000], SL: 59000, Leverage: 10x"
Response:
{{
  "is_signal": true,
  "pair": "BTC",
  "direction": "LONG",
  "entry": 60000.0,
  "targets": [61000.0, 62000.0, 63000.0],
  "stop_loss": 59000.0,
  "leverage": 10
}}

Message: "The market is looking very bullish for Ethereum today. I think it could hit 4k soon."
Response:
{{
  "is_signal": false,
  "pair": null,
  "direction": null,
  "entry": null,
  "targets": [],
  "stop_loss": null,
  "leverage": null
}}

Message: "Thinking of shorting SOL around 150. SL at 155. TP 145, 142."
Response:
{{
  "is_signal": true,
  "pair": "SOL",
  "direction": "SHORT",
  "entry": 150.0,
  "targets": [145.0, 142.0],
  "stop_loss": 155.0,
  "leverage": null
}}

---
Now, analyze this message:

Message: "{message_text}"
Response:
    """
    return prompt


# --- MODIFIED FUNCTION ---
def get_response_with_retry(model_config, model_instance, prompt, is_batch=False):
    """Calls the API with automatic retry, exponential backoff, and model switching on rate limits."""
    global current_model_index, current_model_instance
    current_model = model_instance
    current_config = model_config

    for attempt in range(MAX_RETRIES):
        try:
            # Check if current_model is None and recreate if needed
            if current_model is None:
                dialog_message(
                    f"Model instance is None, recreating {current_config['name']}",
                    "warning",
                )
                current_model = create_model_instance(current_config)
                current_model_instance = current_model

            dialog_message(
                f"Sending request to {current_config['name']} (attempt {attempt + 1}/{MAX_RETRIES})",
                "info",
            )
            response_text = get_api_response(current_config, current_model, prompt)
            dialog_message(
                f"Received response from {current_config['name']}", "success"
            )
            return response_text

        except Exception as e:
            error_str = str(e)

            if (
                "429" in error_str
                or "quota" in error_str.lower()
                or "rate limit" in error_str.lower()
            ):
                dialog_message(
                    f"Rate limit hit with {current_config['name']}", "warning"
                )

                if switch_to_next_model():
                    new_model_config = get_current_model()
                    try:
                        current_model = create_model_instance(new_model_config)
                        current_model_instance = current_model
                        current_config = new_model_config

                        dialog_message(
                            f"Switched to model: {new_model_config['name']} ({new_model_config['type']})",
                            "model",
                        )
                        processing_stats["current_model"] = new_model_config["name"]

                        response_text = get_api_response(
                            current_config, current_model, prompt
                        )
                        dialog_message(
                            f"Success with new model: {new_model_config['name']}",
                            "success",
                        )
                        return response_text

                    except Exception as model_switch_error:
                        dialog_message(
                            f"Error with new model {new_model_config['name']}: {str(model_switch_error)[:100]}",
                            "error",
                        )
                        if attempt < MAX_RETRIES - 1:
                            delay = exponential_backoff_delay(attempt)
                            dialog_message(
                                f"Waiting {delay:.1f}s before retry {attempt + 1}/{MAX_RETRIES}",
                                "warning",
                            )
                            time.sleep(delay)  # Use synchronous sleep
                            continue
                else:
                    if attempt < MAX_RETRIES - 1:
                        delay = exponential_backoff_delay(attempt)
                        dialog_message(
                            f"All models exhausted. Waiting {delay:.1f}s before retry",
                            "warning",
                        )
                        time.sleep(delay)  # Use synchronous sleep
                        continue
                    else:
                        dialog_message("Rate limit exceeded after all retries", "error")
                        return None
            else:
                dialog_message(f"API error: {str(e)[:100]}", "error")
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff_delay(attempt)
                    dialog_message(f"Waiting {delay:.1f}s before retry", "warning")
                    time.sleep(delay)  # Use synchronous sleep
                    continue
                else:
                    return None
    return None


def get_api_response_wrapper(model_config, model_instance, prompt):
    """Wrapper function for API calls that returns both response and updated model info."""
    global current_model_index, current_model_instance

    response = get_response_with_retry(
        model_config, model_instance, prompt, is_batch=False
    )

    # Return updated model info after potential model switches
    updated_model_config = get_current_model()
    return response, updated_model_config, current_model_instance


def process_batch(model_config, model_instance, messages_batch):
    """Process a batch of messages using the current API."""
    global current_model_index, current_model_instance
    dialog_message(
        f"Processing batch of {len(messages_batch)} messages with {model_config['name']}",
        "info",
    )
    batch_prompt = create_batch_prompt(messages_batch)
    json_response = get_response_with_retry(
        model_config, model_instance, batch_prompt, is_batch=True
    )
    updated_model_config = get_current_model()

    if not json_response:
        dialog_message("Failed to get response for batch", "error")
        return [], updated_model_config, current_model_instance
    try:
        batch_results = json.loads(json_response)
        results_map = {
            str(result["message_id"]): result
            for result in batch_results
            if isinstance(result, dict) and "message_id" in result
        }
        matched_results = []
        for msg_id, channel_id, message_text in messages_batch:
            result = results_map.get(str(msg_id))
            if result:
                result.pop("message_id", None)
            matched_results.append((msg_id, channel_id, message_text, result))
        return matched_results, updated_model_config, current_model_instance
    except Exception as e:
        dialog_message(f"Error processing batch: {str(e)[:100]}", "error")
        return [], updated_model_config, current_model_instance


def cleanup_duplicate_labeled_entries(conn, cursor):
    """Remove duplicate entries in the labeled table, keeping the most recent one."""
    logger.info("🧹 Checking for duplicate entries in labeled table...")
    try:
        # Find duplicates
        cursor.execute("""
            SELECT message_id, COUNT(*) as count 
            FROM labeled 
            GROUP BY message_id 
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()

        if not duplicates:
            logger.info("✅ No duplicate entries found in labeled table")
            return True

        logger.warning(f"⚠️  Found {len(duplicates)} message_ids with duplicate entries")

        # For each duplicate, keep only the row with the highest ID (most recent)
        total_removed = 0
        for message_id, count in duplicates:
            cursor.execute(
                """
                DELETE FROM labeled 
                WHERE message_id = ? AND id NOT IN (
                    SELECT MAX(id) FROM labeled WHERE message_id = ?
                )
            """,
                (message_id, message_id),
            )
            removed = cursor.rowcount
            total_removed += removed
            logger.debug(
                f"📝 Removed {removed} duplicate entries for message_id {message_id}"
            )

        conn.commit()
        logger.info(
            f"✅ Cleaned up {total_removed} duplicate entries from labeled table"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to cleanup duplicates: {e}")
        return False


def get_available_models():
    """Filter MODEL_CONFIGS to only include models with configured API keys."""
    global \
        groq_client, \
        together_client, \
        anthropic_client, \
        cohere_client, \
        perplexity_client

    available_models = []

    for model_config in MODEL_CONFIGS:
        model_type = model_config["type"]

        # Always include Gemini and GitHub models (they're required)
        if model_type in ["gemini", "github"]:
            available_models.append(model_config)
        # Include optional models only if their clients are configured
        elif model_type == "groq" and groq_client is not None:
            available_models.append(model_config)
        elif model_type == "together" and together_client is not None:
            available_models.append(model_config)
        elif model_type == "anthropic" and anthropic_client is not None:
            available_models.append(model_config)
        elif model_type == "cohere" and cohere_client is not None:
            available_models.append(model_config)
        elif model_type == "perplexity" and perplexity_client is not None:
            available_models.append(model_config)

    logger.info(
        f"✅ Available models: {len(available_models)} out of {len(MODEL_CONFIGS)} total"
    )
    for model in available_models:
        logger.debug(f"   - {model['name']} ({model['type']})")

    return available_models


def should_process_message(message_text):
    """
    Pre-filter messages to determine if they should be sent to API.
    Returns True if the message might be a trading signal, False otherwise.
    NOW MUCH MORE PERMISSIVE - Let the AI decide instead of auto-filtering!
    """
    if not message_text or not message_text.strip():
        return False

    message_lower = message_text.lower().strip()

    # Skip very short messages (likely not signals)
    if len(message_lower) < 5:
        return False

    # Skip very long messages (likely not signals)
    if len(message_lower) > 2000:
        return False

    # Only skip messages that are CLEARLY not trading signals
    definitely_not_signals = [
        # Very obvious non-trading patterns
        "http://",
        "https://",
        "www.",
        ".com",
        ".org",
        ".net",
        # Emojis only messages
        "😀",
        "😊",
        "❤️",
        "👍",
        "🎉",
        "💕",
        "😍",
        "🔥",
        # Pure social messages
        "good morning everyone",
        "good night everyone",
        "how is everyone",
        "thank you so much",
        "thanks everyone",
        "welcome to our",
        # Admin messages
        "message deleted",
        "user banned",
        "group rules",
        "pinned message",
    ]

    # Only filter out if it's definitely not a signal
    for pattern in definitely_not_signals:
        if pattern in message_lower and len(message_lower) < 50:
            return False

    # Let the AI handle everything else - be very permissive!
    return True


# --- TEXTUAL APP ---

# --- (Keep all the code above the SignalLabelerApp class the same) ---

# --- TEXTUAL APP ---


class SignalLabelerApp(App):
    """A Textual TUI for labeling cryptocurrency trading signals."""

    CSS_PATH = "main_textual.css"
    BINDINGS = [("d", "toggle_dark", "Toggle Dark Mode"), ("q", "quit", "Quit")]

    # --- Reactive properties for real-time UI updates ---
    processed_count = reactive(0)
    signals_found = reactive(0)
    errors = reactive(0)
    current_model_name = reactive("None")
    batch_count = reactive(0)
    progress = reactive(0.0)
    total_messages = reactive(0)
    current_message_text = reactive("[dim]Waiting to start...[/dim]")
    current_message_id = reactive("")
    extracted_labels_content = reactive("[dim]No analysis results yet...[/dim]")

    # --- App State ---
    unlabeled_messages = []
    conn = None
    cursor = None
    current_model_config = None
    current_model_instance = None
    AVAILABLE_MODEL_CONFIGS = []
    processed_messages_history = []  # Store all processed messages for browsing
    selected_message = None  # Currently selected message for viewing

    # --- CONSTANTS ---
    MAX_HISTORY_ITEMS = 50

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Container(id="app-grid"):
            with Vertical(id="left-column"):
                yield Static("📈 Statistics", classes="panel-title")
                yield Static(id="stats-panel")
                yield Static("📋 Activity Log", classes="panel-title")
                yield Log(id="log", highlight=True, max_lines=1000)
                yield ProgressBar(id="progress-bar", total=100)

            with Vertical(id="middle-column"):
                with Container(id="main-content"):
                    yield Static("📝 Current Message", classes="panel-title")
                    yield ScrollableContainer(id="current-message-container")
                    yield Static("🏷️ Extracted Labels", classes="panel-title")
                    yield ScrollableContainer(id="extracted-labels-container")
                with Horizontal(id="button-container"):
                    yield Button("Start Processing", variant="success", id="start")
                    yield Button(
                        "Clear Selection", variant="primary", id="clear-selection"
                    )
                    yield Button("Quit", variant="error", id="quit")

            with Vertical(id="right-column"):
                yield Static("📜 Message History", classes="panel-title")
                yield ScrollableContainer(id="history-container")

        yield Footer()

    # --- Watch methods for reactive properties ---

    def watch_processed_count(self, new_count: int) -> None:
        self.update_stats_panel()
        if self.total_messages > 0:
            self.query_one(ProgressBar).progress = (
                new_count / self.total_messages
            ) * 100

    def watch_signals_found(self, new_count: int) -> None:
        self.update_stats_panel()

    def watch_errors(self, new_count: int) -> None:
        self.update_stats_panel()

    def watch_batch_count(self, new_count: int) -> None:
        self.update_stats_panel()

    def watch_current_model_name(self, new_name: str) -> None:
        self.update_stats_panel()

    def watch_current_message_text(self, new_text: str) -> None:
        if not self.selected_message:
            container = self.query_one("#current-message-container")
            content = (
                f"[bold cyan]ID:[/bold cyan] {self.current_message_id}\n\n{new_text}"
            )
            container.remove_children()
            container.mount(Static(content))

    def watch_extracted_labels_content(self, new_content: str) -> None:
        if not self.selected_message:
            container = self.query_one("#extracted-labels-container")
            container.remove_children()
            container.mount(Static(new_content))

    # --- Update methods ---

    def update_stats_panel(self) -> None:
        """Update the statistics panel with current data."""
        content = (
            f"🔧 [cyan]Model[/cyan]: [yellow]{self.current_model_name}[/yellow]\n"
            f"📊 [cyan]Processed[/cyan]: [green]{self.processed_count}[/green] / {self.total_messages}\n"
            f"🎯 [cyan]Signals Found[/cyan]: [magenta]{self.signals_found}[/magenta]\n"
            f"❌ [cyan]Errors[/cyan]: [red]{self.errors}[/red]\n"
            f"📦 [cyan]Batches[/cyan]: [blue]{self.batch_count}[/blue]"
        )
        self.query_one("#stats-panel", Static).update(content)

    def add_log_message(self, message: str, msg_type: str = "info") -> None:
        """Add a styled message to the log widget."""
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "signal": "🚀",
            "model": "🔄",
        }
        icon = icons.get(msg_type, "📝")
        self.query_one(Log).write_line(f"{icon} {message}")

    def update_extracted_labels(self, data: dict) -> None:
        """Update the extracted labels panel based on analysis data."""
        if not data:
            self.extracted_labels_content = "[dim]No analysis results yet...[/dim]"
            return

        if data.get("is_signal", False):
            pair = data.get("pair", "N/A")
            direction = data.get("direction", "N/A")
            direction_color = (
                "green"
                if direction == "LONG"
                else "red"
                if direction == "SHORT"
                else "white"
            )

            content = (
                f"🎯 [cyan]Signal[/cyan]: [bold green]YES[/bold green]\n"
                f"💰 [cyan]Pair[/cyan]: [yellow]{pair}[/yellow]\n"
                f"📈 [cyan]Direction[/cyan]: [{direction_color}]{direction}[/{direction_color}]\n"
                f"🎪 [cyan]Entry[/cyan]: {data.get('entry', 'N/A')}\n"
                f"🎯 [cyan]Targets[/cyan]: [green]{data.get('targets', [])}[/green]\n"
                f"🛑 [cyan]Stop Loss[/cyan]: [red]{data.get('stop_loss', 'N/A')}[/red]\n"
                f"⚡ [cyan]Leverage[/cyan]: [blue]{data.get('leverage', 'N/A')}[/blue]"
            )
            self.extracted_labels_content = content
        else:
            self.extracted_labels_content = "[red]❌ Not a trading signal.[/red]"

    # --- MODIFIED: Efficiently manages the data list and calls the UI update ---
    def add_message_to_history(
        self, message_id: str, message_text: str, extracted_data: dict | None = None
    ) -> None:
        """Add a processed message to the history data list and trigger a UI update."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        extracted_data = extracted_data or {}
        message_entry = {
            "timestamp": timestamp,
            "message_id": message_id,
            "message_text": message_text,
            "extracted_data": extracted_data,
            "is_signal": extracted_data.get("is_signal", False),
        }

        # Add to the beginning of the list
        self.processed_messages_history.insert(0, message_entry)

        # Prune the data list if it's too long
        if len(self.processed_messages_history) > self.MAX_HISTORY_ITEMS:
            self.processed_messages_history.pop()

        # Call the efficient UI update function
        self.add_message_to_history_panel(message_entry)

    # --- NEW: Efficiently adds a single new message to the history panel ---
    def add_message_to_history_panel(self, message_entry: dict) -> None:
        """Adds a new message button to the top of the history panel and removes the oldest if over limit."""
        container = self.query_one("#history-container", ScrollableContainer)

        message_id = message_entry["message_id"]
        timestamp = message_entry["timestamp"]
        is_signal = message_entry["is_signal"]

        # Create a short preview of the message
        text_preview = message_entry["message_text"].replace("\n", " ").strip()
        text_preview = (
            text_preview[:40] + "..." if len(text_preview) > 40 else text_preview
        )

        button_variant = "success" if is_signal else "default"
        signal_indicator = "🎯" if is_signal else "📝"

        button_text = f"{signal_indicator} [{timestamp}] {text_preview}"

        # Create a button and give it the message_id for later retrieval
        button = Button(
            button_text,
            variant=button_variant,
            id=f"msg_{message_id}",
            classes="message-button",
        )

        # Mount the new button at the top of the list
        container.mount(button, before=0)

        # Remove the oldest button from the bottom if the panel has too many children
        if len(container.children) > self.MAX_HISTORY_ITEMS:
            container.children[-1].remove()

    # --- Event Handlers and Workers ---

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        self.add_log_message("App initialized. Press 'Start Processing' to begin.")
        self.update_stats_panel()
        # Set initial empty state for history
        self.query_one("#history-container").mount(
            Static("[dim]History will appear here.[/dim]", id="history-placeholder")
        )

    @on(Button.Pressed, "#start")
    def on_start_button_pressed(self) -> None:
        """Handle the start button press by launching a background worker."""
        # Remove placeholder text when starting
        placeholder = self.query_one("#history-placeholder", Static)
        if placeholder.is_mounted:
            placeholder.remove()

        self.query_one("#start", Button).disabled = True
        self.query_one("#quit", Button).disabled = True
        self.add_log_message("Starting processing worker...", "info")
        self.run_worker(self.process_messages_worker, name="message_processor")

    @on(Button.Pressed, "#quit")
    def on_quit_button_pressed(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#clear-selection")
    def on_clear_selection_pressed(self) -> None:
        """Clear the selected message and return to current processing view."""
        if not self.selected_message:
            return

        # De-select the button in the UI
        try:
            selected_button = self.query_one(
                f"#msg_{self.selected_message['message_id']}", Button
            )
            is_signal = self.selected_message.get("is_signal", False)
            selected_button.variant = "success" if is_signal else "default"
        except Exception:
            pass  # Button might not exist anymore, that's fine

        self.selected_message = None
        self.add_log_message(
            "Cleared message selection, returning to live processing view", "info"
        )

        # Trigger reactive properties to update the view to the latest live data
        self.watch_current_message_text(self.current_message_text)
        self.watch_extracted_labels_content(self.extracted_labels_content)

    @on(Button.Pressed, ".message-button")
    def on_message_button_pressed(self, event: Button.Pressed) -> None:
        """Handle clicking on a message in the history."""
        button_id = event.button.id
        if not button_id or not button_id.startswith("msg_"):
            return

        try:
            # Clear previous selection highlight
            if self.selected_message:
                prev_button = self.query_one(
                    f"#msg_{self.selected_message['message_id']}", Button
                )
                is_signal = self.selected_message.get("is_signal", False)
                prev_button.variant = "success" if is_signal else "default"
        except Exception:
            pass

        message_id = button_id.split("_", 1)[1]

        # Find the message in our data list
        selected_msg_data = next(
            (
                msg
                for msg in self.processed_messages_history
                if msg["message_id"] == message_id
            ),
            None,
        )

        if selected_msg_data:
            self.selected_message = selected_msg_data
            self.show_selected_message()
            # Highlight the newly selected button
            event.button.variant = "warning"
            self.add_log_message(
                f"Viewing message ID {message_id} from history", "info"
            )
        else:
            self.add_log_message(
                f"Could not find data for message ID {message_id}", "error"
            )

    def show_selected_message(self) -> None:
        """Display the selected message in the main panels."""
        if not self.selected_message:
            return

        msg_text = self.selected_message["message_text"]
        data = self.selected_message["extracted_data"]

        # Update current message panel
        msg_container = self.query_one("#current-message-container")
        content = f"[bold magenta]📋 VIEWING FROM HISTORY[/bold magenta]\n"
        content += f"[bold cyan]ID:[/bold cyan] {self.selected_message['message_id']}\n"
        content += f"[bold cyan]Time:[/bold cyan] {self.selected_message['timestamp']}\n\n{msg_text}"
        msg_container.remove_children()
        msg_container.mount(Static(content))

        # Update extracted labels panel
        labels_container = self.query_one("#extracted-labels-container")
        labels_container.remove_children()
        labels_container.mount(Static(self.format_extracted_labels(data)))

    def format_extracted_labels(self, data: dict) -> str:
        """Helper to format extracted data into a string for display."""
        if not data:
            return "[dim]No analysis data available...[/dim]"
        if data.get("is_signal", False):
            pair = data.get("pair", "N/A")
            direction = data.get("direction", "N/A")
            direction_color = (
                "green"
                if direction == "LONG"
                else "red"
                if direction == "SHORT"
                else "white"
            )
            return (
                f"🎯 [cyan]Signal[/cyan]: [bold green]YES[/bold green]\n"
                f"💰 [cyan]Pair[/cyan]: [yellow]{pair}[/yellow]\n"
                f"📈 [cyan]Direction[/cyan]: [{direction_color}]{direction}[/{direction_color}]\n"
                f"🎪 [cyan]Entry[/cyan]: {data.get('entry', 'N/A')}\n"
                f"🎯 [cyan]Targets[/cyan]: [green]{data.get('targets', [])}[/green]\n"
                f"🛑 [cyan]Stop Loss[/cyan]: [red]{data.get('stop_loss', 'N/A')}[/red]\n"
                f"⚡ [cyan]Leverage[/cyan]: [blue]{data.get('leverage', 'N/A')}[/blue]"
            )
        else:
            return "[red]❌ Not a trading signal.[/red]"

    async def process_messages_worker(self) -> None:
        """
        Async worker that launches the blocking database/API work in a separate thread
        and handles UI updates via a thread-safe callback.
        """
        global app_instance
        app_instance = self

        def ui_update_callback(event_type: str, data: dict):
            """A thread-safe function to send updates from the worker thread to the UI."""
            try:
                # This is the correct way to send updates from the background thread.
                self.call_from_thread(self.handle_worker_update, event_type, data)
            except RuntimeError:
                # This error can happen if the app is shutting down while the worker
                # thread is still sending a message. It's safe to ignore in this case.
                pass

        try:
            # Launch the entire synchronous process in a single background thread.
            # This ensures the database connection is created and used in the same thread.
            await asyncio.to_thread(
                self._blocking_database_and_api_work, ui_update_callback
            )
        except Exception as e:
            # This 'except' block runs on the main thread, so we call UI methods directly.
            self.add_log_message(f"Critical worker failure: {e}", "error")
            logger.exception("Critical error launching the blocking worker")
        finally:
            # This 'finally' block also runs on the main thread.
            # We call the method directly to re-enable buttons.
            self._re_enable_buttons()

    def handle_worker_update(self, event_type: str, data: dict):
        """This method is always called on the main thread to safely update the UI."""
        if event_type == "log":
            self.add_log_message(data["message"], data["type"])
        elif event_type == "model_update":
            self.current_model_name = data["name"]
        elif event_type == "totals":
            self.total_messages = data["total"]
        elif event_type == "progress":
            # Update live view
            self.current_message_id = data["message_id"]
            self.current_message_text = data["message_text"]

            # Format and set extracted labels
            if data["extracted_data"]:
                self.extracted_labels_content = self.format_extracted_labels(
                    data["extracted_data"]
                )
                if data["extracted_data"].get("is_signal"):
                    self.signals_found += 1
            else:
                self.errors += 1
                self.extracted_labels_content = "[red]Error processing message.[/red]"

            # Update history and counts
            self.add_message_to_history(
                data["message_id"], data["message_text"], data["extracted_data"]
            )
            self.processed_count += 1
        elif event_type == "batch_count":
            self.batch_count += 1

    def _blocking_database_and_api_work(self, callback):
        """
        This function runs entirely in a single background thread.
        It handles all database and API calls synchronously.
        """
        conn = None
        try:
            global AVAILABLE_MODEL_CONFIGS, current_model_index, current_model_instance
            current_model_index = 0

            callback("log", {"message": "Configuring APIs...", "type": "info"})
            configure_apis()

            AVAILABLE_MODEL_CONFIGS = get_available_models()
            if not AVAILABLE_MODEL_CONFIGS:
                callback(
                    "log",
                    {
                        "message": "No AI models available! Check .env file.",
                        "type": "error",
                    },
                )
                return

            model_config = get_current_model()
            model_instance = create_model_instance(model_config)
            current_model_instance = model_instance  # Initialize global instance
            callback("model_update", {"name": model_config["name"]})

            callback("log", {"message": "Setting up database...", "type": "info"})
            conn, cursor = setup_database(DB_FILE)
            cleanup_duplicate_labeled_entries(conn, cursor)

            callback(
                "log", {"message": "Fetching unlabeled messages...", "type": "info"}
            )
            all_messages = get_unlabeled_messages(cursor)
            if not all_messages:
                callback(
                    "log", {"message": "No new messages to label.", "type": "success"}
                )
                return

            # Randomize message processing order
            all_messages = list(all_messages)  # Convert to list if it's not already
            random.shuffle(all_messages)
            callback(
                "log",
                {
                    "message": f"Shuffled {len(all_messages)} messages for random processing order",
                    "type": "info",
                },
            )

            valid_messages = []
            skipped_count = 0
            for msg_id, channel_id, msg_text in all_messages:
                if (
                    not msg_text
                    or not msg_text.strip()
                    or not should_process_message(msg_text)
                ):
                    non_signal_data = {"is_signal": False}
                    save_to_database(
                        conn, cursor, msg_id, channel_id, msg_text, non_signal_data
                    )
                    # For skipped messages, we also send a progress update
                    callback(
                        "progress",
                        {
                            "message_id": str(msg_id),
                            "message_text": msg_text,
                            "extracted_data": non_signal_data,
                        },
                    )
                    skipped_count += 1
                else:
                    valid_messages.append((msg_id, channel_id, msg_text))

            callback("totals", {"total": len(all_messages)})
            callback(
                "log",
                {
                    "message": f"Skipped {skipped_count} non-signals. Processing {len(valid_messages)} via API.",
                    "type": "info",
                },
            )

            # Shuffle valid messages again for maximum randomness
            if valid_messages:
                random.shuffle(valid_messages)
                callback(
                    "log",
                    {
                        "message": f"Shuffled {len(valid_messages)} valid messages for batch processing",
                        "type": "info",
                    },
                )

            total_batches = (len(valid_messages) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(valid_messages), BATCH_SIZE):
                batch = valid_messages[i : i + BATCH_SIZE]
                callback("batch_count", {})
                callback(
                    "log",
                    {
                        "message": f"Processing batch {i // BATCH_SIZE + 1}/{total_batches}...",
                        "type": "info",
                    },
                )

                batch_results, new_config, new_instance = process_batch(
                    model_config, model_instance, batch
                )
                model_config = new_config
                model_instance = new_instance
                current_model_instance = new_instance  # Update global instance
                callback("model_update", {"name": model_config["name"]})

                for msg_id, channel_id, message_text, data in batch_results:
                    save_to_database(
                        conn, cursor, msg_id, channel_id, message_text, data
                    )
                    callback(
                        "progress",
                        {
                            "message_id": str(msg_id),
                            "message_text": message_text,
                            "extracted_data": data,
                        },
                    )

            callback("log", {"message": "Processing complete!", "type": "success"})

        except Exception as e:
            logger.exception("Critical error in blocking worker thread")
            callback(
                "log",
                {
                    "message": f"A critical error occurred in worker: {e}",
                    "type": "error",
                },
            )
        finally:
            if conn:
                conn.close()
                callback(
                    "log", {"message": "Database connection closed.", "type": "info"}
                )

    def _re_enable_buttons(self) -> None:
        """Re-enable buttons after processing."""
        self.query_one("#start", Button).disabled = False
        self.query_one("#quit", Button).disabled = False


# --- (Keep the if __name__ == "__main__": block the same) ---
if __name__ == "__main__":
    app = SignalLabelerApp()
    app.run()
