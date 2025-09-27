"""
This module is responsible for parsing incoming text from Telegram messages
to identify potential trading signals based on pre-configured rules.
"""

import re
from loguru import logger
from typing import Optional, List, Dict
from ai.inference.ai_service import get_ai_service

# Initialize AI service instance
_ai_service = None


def _get_ai_service():
    """Get or create AI service instance with lazy loading."""
    global _ai_service
    if _ai_service is None:
        _ai_service = get_ai_service()
        if _ai_service and not _ai_service.is_available():
            logger.warning(
                "AI models not found or failed to load. AI parsing will return None."
            )
    return _ai_service


def parse_message_for_signal(
    text: str, channel_id: int, channels_config: Dict
) -> Optional[List[str]]:
    """
    Parses a message text to find a trading signal (coin and side).

    Args:
        text: The text content of the message.
        channel_id: The ID of the channel where the message originated.
        channels_config: The dictionary of all configured channels.

    Returns:
        A list containing [coin, side, log_message] if a signal is found,
        otherwise None.
    """
    try:
        channel_info = channels_config[channel_id]
        channel_name = channel_info.get("name", "Unknown Channel")

        if not channel_info.get("do", False):
            logger.trace(
                f"Parsing skipped for {channel_name} ({channel_id}) as 'do' is False."
            )
            return None

        # Find the coin ticker using the specified regex
        coin_match = re.search(channel_info["regex"], text, re.IGNORECASE)
        if not coin_match:
            logger.trace(f"No coin pattern found in message from {channel_name}.")
            return None

        coin = coin_match.group(1).upper()
        lower_text = text.lower()

        # Check for long/short keywords
        if channel_info["long"].lower() in lower_text:
            side = "long"
        elif channel_info["short"].lower() in lower_text:
            side = "short"
        else:
            logger.trace(
                f"No long/short keyword found for {coin} in message from {channel_name}."
            )
            return None

        log_message = f"Signal Found: {channel_name} | {coin} | {side.upper()}"
        return [coin, side, log_message]

    except KeyError:
        logger.error(f"Channel ID '{channel_id}' not found in channels.json.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in text_parser: {e}")
        return None


def ai_parse_message_for_signal(text: str) -> Optional[dict]:
    """
    Uses AI to parse a message text to find a trading signal (all data).

    Args:
        text: The text content of the message.

    Returns:
        A dictionary containing parsed signal data if found, otherwise None.
    """
    try:
        # Get AI service instance
        ai_service = _get_ai_service()

        # If AI service is not available, return None
        if ai_service is None or not ai_service.is_available():
            logger.debug("AI service not available for parsing")
            return None

        # Parse the message using the AI service
        result = ai_service.parse_signal(text)

        if not result:
            logger.debug("AI parsing returned no result")
            return None

        # Ensure a standard confidence field exists for downstream checks
        signal_confidence = result.get("signal_confidence", 0.0)
        result.setdefault("confidence", signal_confidence)

        # Only return result if it's detected as a signal with sufficient confidence
        if result.get("is_signal", False) and signal_confidence >= 0.5:
            direction_value = result.get("direction")

            # Normalize direction to string format
            if isinstance(direction_value, (int, float)):
                result["direction"] = "long" if int(direction_value) == 0 else "short"
            elif isinstance(direction_value, str):
                result["direction"] = direction_value.lower()

            logger.info(f"AI detected signal: {result}")
            return result

        logger.debug(
            "AI did not detect signal or confidence too low: {confidence}",
            confidence=signal_confidence,
        )
        return None

    except Exception as e:
        logger.error(f"Error in AI parsing: {e}")
        return None
