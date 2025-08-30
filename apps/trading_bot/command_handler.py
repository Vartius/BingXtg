"""
This module defines handlers for user commands sent to the bot via Telegram.
"""

import random
import sys
from loguru import logger
from pyrogram.client import Client
from pyrogram.types import Message
from apps.trading_bot.order_handler import place_order


# --- Command Handlers ---


async def handle_chats_check(client: Client, message: Message, chat_ids: list):
    """Verifies the bot can access the configured channels."""
    titles = []
    for chat_id in chat_ids:
        try:
            chat = await client.get_chat(chat_id)
            titles.append(f"✅ {chat.title or f'Chat {chat_id}'}")
        except Exception as e:
            logger.error(f"Could not get info for chat {chat_id}: {e}")
            titles.append(f"❌ Error fetching chat {chat_id}")
    await message.reply_text(
        "<b>Channel Accessibility Check:</b>\n\n" + "\n".join(titles)
    )


async def handle_list_channels(message: Message):
    """Lists the channels configured in channels.json."""
    # TODO: Load from database


async def handle_stop(message: Message):
    """Stops the bot gracefully."""
    logger.warning(f"Stop command received from {message.chat.id}. Shutting down.")
    await message.reply_text("Bot is shutting down...")
    sys.exit(0)  # Exits the entire application


async def handle_add_test_orders(is_simulation: bool):
    """Adds a batch of random test orders for demonstration."""
    logger.info("Adding a batch of test orders.")
    test_coins = ["CRV", "UNI", "BTC", "ETH", "XRP", "STORJ", "AAVE", "SOL"]
    for coin in test_coins:
        # Use a dummy channel ID for test orders
        # TODO: make data for placing test orders
        data = {}
    logger.success("Finished adding test orders.")


# TODO
async def handle_get_data(client: Client, message: Message):
    """Generates and sends an image of the current trading data table."""
    logger.info(f"Generating data table image for {message.chat.id}.")


async def handle_command(
    command: str, client: Client, message: Message, chat_ids: list, is_simulation: bool
):
    """Routes commands to their respective handlers."""
    if command == ".chatscheck":
        await handle_chats_check(client, message, chat_ids)
    elif command == ".chats":
        await handle_list_channels(message)
    elif command == ".stop":
        await handle_stop(message)
    elif command == ".addtestorders":
        await handle_add_test_orders(is_simulation)
    elif command == ".getdata":
        await handle_get_data(client, message)
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
        await message.reply_text(help_text)
    else:
        await message.reply_text(f"Unknown command: `{command}`")
