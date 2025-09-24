"""
This module defines handlers for user commands sent to the bot via Telegram.
"""

import sys
from loguru import logger
from telethon import TelegramClient
from telethon.events import NewMessage


# --- Command Handlers ---


async def handle_chats_check(
    client: TelegramClient, event: NewMessage.Event, chat_ids: list
):
    """Verifies the bot can access the configured channels."""
    titles = []
    for chat_id in chat_ids:
        try:
            entity = await client.get_entity(chat_id)
            titles.append(f"✅ {getattr(entity, 'title', f'Chat {chat_id}')}")
        except Exception as e:
            logger.error(f"Could not get info for chat {chat_id}: {e}")
            titles.append(f"❌ Error fetching chat {chat_id}")
    await event.reply(
        "<b>Channel Accessibility Check:</b>\n\n" + "\n".join(titles), parse_mode="html"
    )


async def handle_list_channels(event: NewMessage.Event):
    """Lists the channels configured in channels.json."""
    # TODO: Load from database


async def handle_stop(event: NewMessage.Event):
    """Stops the bot gracefully."""
    logger.warning(f"Stop command received from {event.chat_id}. Shutting down.")
    await event.reply("Bot is shutting down...")
    sys.exit(0)  # Exits the entire application


async def handle_add_test_orders(is_simulation: bool):
    """Adds a batch of random test orders for demonstration."""
    logger.info("Adding a batch of test orders.")
    test_coins = ["CRV", "UNI", "BTC", "ETH", "XRP", "STORJ", "AAVE", "SOL"]
    for coin in test_coins:
        # Use a dummy channel ID for test orders
        # TODO: make data for placing test orders
        pass
    logger.success("Finished adding test orders.")


# TODO
async def handle_get_data(client: TelegramClient, event: NewMessage.Event):
    """Generates and sends an image of the current trading data table."""
    logger.info(f"Generating data table image for {event.chat_id}.")


async def handle_command(
    command: str,
    client: TelegramClient,
    event: NewMessage.Event,
    chat_ids: list,
    is_simulation: bool,
):
    """Routes commands to their respective handlers."""
    if command == ".chatscheck":
        await handle_chats_check(client, event, chat_ids)
    elif command == ".chats":
        await handle_list_channels(event)
    elif command == ".stop":
        await handle_stop(event)
    elif command == ".addtestorders":
        await handle_add_test_orders(is_simulation)
    elif command == ".getdata":
        await handle_get_data(client, event)
    elif command == ".help":
        help_text = (
            "<b>Available Commands:</b>\n"
            "• <code>.chatscheck</code> - Verify access to configured channels.\n"
            "• <code>.chats</code> - List configured channels.\n"
            "• <code>.stop</code> - Stop the bot gracefully.\n"
            "• <code>.addtestorders</code> - Add random test orders (simulation mode only).\n"
            "• <code>.getdata</code> - Get an image of current trading data.\n"
            "• <code>.help</code> - Show this help message."
        )
        await event.reply(help_text, parse_mode="html")
    else:
        await event.reply(f"Unknown command: `{command}`")
