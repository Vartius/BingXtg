import json
import re
from loguru import logger


def textHandler(text, chanId):
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
        
        channel_info = channels[chanId]
        name = channel_info["name"]
        regex = channel_info["regex"]
        long_kw = channel_info["long"]
        short_kw = channel_info["short"]
        do = channel_info.get("do", False) # Use .get for safer access

        if not do:
            logger.warning(f"{name} ({chanId}): 'do' is set to False, skipping.")
            return None

        coin_match = re.search(regex, text)
        if coin_match:
            coin = coin_match.group(1).upper()
            lower_text = text.lower()
            
            if long_kw in lower_text:
                return [coin, "long", f"{name}: {coin} long"]
            elif short_kw in lower_text:
                return [coin, "short", f"{name}: {coin} short"]
            else:
                logger.warning(f"{name} ({chanId}): long/short keyword not found in message.")
                return None
        else:
            logger.warning(f"{name} ({chanId}): coin pattern not found in message.")
            return None
            
    except FileNotFoundError:
        logger.error("channels.json not found.")
        return None
    except json.JSONDecodeError:
        logger.error("Error decoding channels.json.")
        return None
    except KeyError:
        logger.error(f"Channel ID '{chanId}' not found in channels.json.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in textHandler: {e}")
        return None