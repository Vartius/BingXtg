"""
This module is responsible for parsing incoming text from Telegram messages
to identify potential trading signals based on pre-configured rules.
"""

import re
from loguru import logger
from typing import Optional, List, Dict


def parse_message_for_signal(
    text: str, channel_id: str, channels_config: Dict
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
