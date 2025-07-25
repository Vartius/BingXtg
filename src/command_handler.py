"""
This module defines handlers for user commands sent to the bot via Telegram.
"""

import os
import json
import random
import sys
import pandas as pd
import dataframe_image as dfi
from loguru import logger
from pyrogram.client import Client
from pyrogram.types import Message
from src.order_handler import place_order


# --- DataFrame Styling Functions ---


def _style_table(df: pd.DataFrame):
    """Applies a consistent style to the data table for image export."""
    styles = [
        {
            "selector": "th",
            "props": [("background-color", "#474747"), ("color", "white")],
        },
        {"selector": "td", "props": [("color", "white")]},
        {"selector": "tr:nth-child(even)", "props": [("background-color", "#353535")]},
        {"selector": "tr:nth-child(odd)", "props": [("background-color", "#1b1b1b")]},
    ]
    return (
        df.style.set_table_styles(styles)  # type: ignore
        .background_gradient(subset=["PnL ($)", "PnL (%)"], cmap="RdYlGn")
        .apply(
            lambda x: [
                "background-color: #00FF7F"
                if v == "long"
                else "background-color: #DC143C"
                if v == "short"
                else ""
                for v in x
            ],
            subset=["Side"],
        )
        .format({"PnL ($)": "${:.2f}", "PnL (%)": "{:.2f}%"})
    )


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
    try:
        with open("data/channels.json", "r", encoding="utf-8") as f:
            channels = json.load(f)
        response = "<b>Configured Channels:</b>\n\n" + "\n".join(
            f"• {ch_data.get('name', 'N/A')} (<code>{ch_id}</code>)"
            for ch_id, ch_data in channels.items()
        )
        await message.reply_text(response)
    except FileNotFoundError:
        await message.reply_text("`channels.json` not found.")
    except json.JSONDecodeError:
        await message.reply_text("Error reading `channels.json`.")


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
        side = random.choice(["long", "short"])
        # Use a dummy channel ID for test orders
        place_order(
            channel_id="-1000000000000",
            coin=coin,
            side=side,
            is_simulation=is_simulation,
        )
    logger.success("Finished adding test orders.")


async def handle_get_data(client: Client, message: Message):
    """Generates and sends an image of the current trading data table."""
    logger.info(f"Generating data table image for {message.chat.id}.")
    try:
        with open("data/table.json", "r", encoding="utf-8") as f:
            table_data = json.load(f)

        headers = [
            "Channel",
            "Coin",
            "Side",
            "Margin ($)",
            "Entry Price",
            "Current Price",
            "PnL ($)",
            "PnL (%)",
        ]
        df = pd.DataFrame(table_data.get("orders", []), columns=headers)

        if df.empty:
            await message.reply_text("No open orders to display.")
            return

        df_styled = _style_table(df)
        image_path = "table_export.png"
        dfi.export(df_styled, image_path, table_conversion="matplotlib")  # type: ignore

        caption = (
            f"<b>Balance:</b> ${table_data.get('balance', 'N/A'):.2f}\n"
            f"<b>Available:</b> ${table_data.get('available_balance', 'N/A'):.2f}\n"
            f"<b>Global Winrate:</b> {table_data.get('winrate', 'N/A')}%"
        )
        await client.send_photo(message.chat.id, image_path, caption=caption)
        os.remove(image_path)
    except FileNotFoundError:
        await message.reply_text("`table.json` not found. No data to display.")
    except Exception as e:
        logger.error(f"Failed to generate or send data image: {e}")
        await message.reply_text("An error occurred while generating the data image.")


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
    else:
        await message.reply_text(f"Unknown command: `{command}`")
