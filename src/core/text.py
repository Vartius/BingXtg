import json
import re
from loguru import logger


def textHandler(text, chanId):
    with open("src/data/channels.json", "r", encoding="utf-8") as f:
        channels = json.load(f)
    channels = json.load(open("src/data/channels.json", encoding="utf-8"))
    name = channels[chanId]["name"]
    regex = channels[chanId]["regex"]
    long = channels[chanId]["long"]
    short = channels[chanId]["short"]
    do = channels[chanId]["do"]
    if do is False:
        logger.warning(f"{chanId}: do=False")
        return None
    coin = re.search(regex, text)
    if coin is not None:
        t = text.lower()
        coin = coin[1].upper()
        if long in t:
            return [coin, "long", f"{name}: {coin} long"]
        elif short in t:
            return [coin, "short", f"{name}: {coin} short"]
        else:
            logger.warning(f"{chanId}: long/short not found")
            return None
    else:
        logger.warning(f"{chanId}: coin not found")
        return None
