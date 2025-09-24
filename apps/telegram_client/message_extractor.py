import os
import logging
from typing import Optional, List, Union, TypedDict
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.custom.dialog import Dialog
from telethon.tl.custom.message import Message
from core.database.manager import DatabaseManager
from pathlib import Path

from core.config.settings import SESSION_FILE

load_dotenv()
logger = logging.getLogger(__name__)

# Type aliases
EntityType = Union[int, str]


class ChannelInfo(TypedDict):
    id: int
    title: str
    username: Optional[str]


# Absolute project root (…/regexGenAI)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MessageExtractor:
    """Handles Telegram message extraction and storage."""

    def __init__(self, db_path: str = "messages.db", folder_id: Optional[int] = None):
        self.db_path = db_path
        self.folder_id = folder_id or int(os.getenv("FOLDER_ID", 1))
        self.db_manager = DatabaseManager(db_path)

    async def get_crypto_channels(self, client: TelegramClient) -> List[ChannelInfo]:
        """
        Gets all channel IDs from the configured folder in Telegram.

        Args:
            client: An active Telegram client instance.

        Returns:
            A list of dictionaries containing channel information.
        """
        logger.info(f"Searching for channels in folder ID: {self.folder_id}")
        crypto_channels: List[ChannelInfo] = []
        try:
            async for dialog in client.iter_dialogs():
                dialog: Dialog = dialog
                if dialog.folder_id == self.folder_id and dialog.is_channel:
                    channel_info: ChannelInfo = {
                        "id": dialog.id,
                        "title": dialog.title,
                        "username": dialog.entity.username
                        if hasattr(dialog.entity, "username")
                        else None,
                    }
                    crypto_channels.append(channel_info)
                    try:
                        self.db_manager.upsert_channel(
                            channel_info["id"],
                            channel_info["title"],
                            channel_info["username"],
                        )
                    except Exception:
                        logger.exception("Failed to upsert channel metadata")
                    logger.info(
                        f"Found channel '{dialog.title}' (ID: {dialog.id}) in target folder."
                    )
            return crypto_channels
        except Exception:
            logger.exception(
                "Error occurred while getting crypto channels from folder."
            )
            return []

    async def parse_channel_messages(
        self,
        client: TelegramClient,
        channel_entity: EntityType,
        limit: Optional[int] = None,
    ) -> None:
        """
        Parses and saves messages from a specific Telegram channel.

        Args:
            client: An active Telegram client instance.
            channel_entity: The channel's ID or username.
            limit: The maximum number of messages to fetch.
        """
        try:
            entity = await client.get_entity(channel_entity)
            # Handle case where get_entity might return a list
            if isinstance(entity, list):
                if not entity:
                    logger.error(f"No entity found for '{channel_entity}'")
                    return
                channel = entity[0]  # Take the first entity
            else:
                channel = entity

            # Upsert channel metadata
            try:
                title = getattr(channel, "title", getattr(channel, "first_name", None))
                username = getattr(channel, "username", None)
                ch_id = int(getattr(channel, "id"))  # ensure int
                self.db_manager.upsert_channel(ch_id, title, username)
            except Exception:
                logger.exception(
                    "Failed to upsert channel metadata from parse_channel_messages"
                )

            logger.info(
                f"Begin parsing messages from channel: '{getattr(channel, 'title', 'N/A')}' (ID: {channel.id})."
            )

            message_count = 0
            # Handle the limit parameter properly - use None or pass the limit directly
            if limit is None:
                async for message in client.iter_messages(channel):
                    message: Message = message
                    if message.text:
                        self.db_manager.save_message(channel.id, message.text)
                        message_count += 1
                        if message_count % 100 == 0:
                            logger.info(
                                f"Processed {message_count} messages from '{getattr(channel, 'title', 'N/A')}'..."
                            )
            else:
                async for message in client.iter_messages(channel, limit=limit):
                    message: Message = message
                    if message.text:
                        self.db_manager.save_message(channel.id, message.text)
                        message_count += 1
                        if message_count % 100 == 0:
                            logger.info(
                                f"Processed {message_count} messages from '{getattr(channel, 'title', 'N/A')}'..."
                            )

            logger.info(
                f"Finished parsing. Total messages processed from '{getattr(channel, 'title', 'N/A')}': {message_count}."
            )

        except ValueError:
            logger.error(f"Channel '{channel_entity}' not found or invalid.")
        except Exception:
            logger.exception(
                f"An unexpected error occurred while parsing messages from channel: {channel_entity}."
            )

    async def extract_messages_from_folder(
        self, api_id: int, api_hash: str, limit: Optional[int] = None
    ) -> None:
        """
        Main method to connect to Telegram and extract messages from all channels in the configured folder.

        Args:
            api_id: Your Telegram API ID.
            api_hash: Your Telegram API Hash.
            limit: The maximum number of messages to fetch per channel.
        """
        self.db_manager.init_database()

        async with TelegramClient(SESSION_FILE, api_id, api_hash) as client:
            logger.info("Telegram client started.")

            crypto_channels = await self.get_crypto_channels(client)

            if not crypto_channels:
                logger.warning(
                    "No channels found in the specified folder. Nothing to extract."
                )
                return

            logger.info(f"Found {len(crypto_channels)} channels to process.")
            for channel in crypto_channels:
                await self.parse_channel_messages(client, channel["id"], limit=limit)

        logger.info("Message extraction process finished.")

    async def extract_messages_from_channel(
        self,
        api_id: int,
        api_hash: str,
        channel_entity: EntityType,
        limit: Optional[int] = None,
    ) -> None:
        """
        Extract messages from a specific Telegram channel, group, or chat.

        Args:
            api_id: Your Telegram API ID.
            api_hash: Your Telegram API Hash.
            channel_entity: The entity's ID, username, or invite link.
                          Examples:
                          - Username: "channel_name" (without @)
                          - Entity ID: 1234567890
                          - Invite link: "https://t.me/joinchat/XXXXX"
            limit: The maximum number of messages to fetch from the entity.
        """
        self.db_manager.init_database()

        async with TelegramClient(SESSION_FILE, api_id, api_hash) as client:
            logger.info("Telegram client started.")

            try:
                # Get the entity (channel, group, or chat)
                entity = await client.get_entity(channel_entity)

                # Handle case where get_entity might return a list
                if isinstance(entity, list):
                    if not entity:
                        logger.error(f"No entity found for '{channel_entity}'")
                        return
                    channel = entity[0]  # Take the first entity
                else:
                    channel = entity

                # Log entity type for debugging
                entity_type = (
                    "channel"
                    if hasattr(channel, "broadcast") and channel.broadcast
                    else "group/chat"
                )
                logger.info(f"Entity type detected: {entity_type}")

                logger.info(
                    f"Starting message extraction from {entity_type}: '{getattr(channel, 'title', getattr(channel, 'first_name', 'N/A'))}' (ID: {channel.id})"
                )

                # Extract messages from the specific entity
                await self.parse_channel_messages(client, channel.id, limit=limit)

                logger.info(
                    f"Message extraction completed for {entity_type}: '{getattr(channel, 'title', getattr(channel, 'first_name', 'N/A'))}'"
                )

            except ValueError as e:
                logger.error(f"Entity '{channel_entity}' not found or invalid: {e}")
                raise ValueError(
                    f"Entity '{channel_entity}' not found. Please check the username, ID, or invite link."
                )
            except PermissionError as e:
                logger.error(f"Access denied to entity '{channel_entity}': {e}")
                raise PermissionError(
                    f"Access denied to entity '{channel_entity}'. You may need to join the group/channel first."
                )
            except Exception:
                logger.exception(
                    f"An unexpected error occurred while extracting messages from entity: {channel_entity}"
                )
                raise

        logger.info("Single entity extraction process finished.")

    async def get_channel_info(
        self, api_id: int, api_hash: str, channel_entity: EntityType
    ) -> Optional[ChannelInfo]:
        """
        Get basic information about a specific channel, group, or chat without extracting messages.

        Args:
            api_id: Your Telegram API ID.
            api_hash: Your Telegram API Hash.
            channel_entity: The entity's ID, username, or invite link.

        Returns:
            ChannelInfo dictionary with entity details, or None if entity not found.
        """
        async with TelegramClient(SESSION_FILE, api_id, api_hash) as client:
            client: TelegramClient = client
            try:
                logging.info(f"Fetching entity info for: {channel_entity}")
                entity = await client.get_entity(channel_entity)

                # Handle case where get_entity might return a list
                if isinstance(entity, list):
                    if not entity:
                        return None
                    channel = entity[0]
                else:
                    channel = entity

                channel_info: ChannelInfo = {
                    "id": channel.id,
                    "title": getattr(
                        channel, "title", getattr(channel, "first_name", "N/A")
                    ),
                    "username": getattr(channel, "username", None),
                }

                return channel_info

            except ValueError:
                logger.error(f"Entity '{channel_entity}' not found or invalid.")
                return None
            except Exception:
                logger.exception(f"Error getting entity info for: {channel_entity}")
                return None

    async def backfill_channel_metadata(self, api_id: int, api_hash: str) -> int:
        """
        Find channels in the DB that lack a title/username and fetch their metadata
        from Telegram, updating the channels table. Returns number of channels updated.
        """
        missing_ids = self.db_manager.get_channels_missing_metadata()
        if not missing_ids:
            logger.info("No channels missing metadata.")
            return 0
        updated = 0
        async with TelegramClient(SESSION_FILE, api_id, api_hash) as client:
            for ch_id in missing_ids:
                try:
                    entity = await client.get_entity(int(ch_id))
                    title = getattr(
                        entity, "title", getattr(entity, "first_name", None)
                    )
                    username = getattr(entity, "username", None)
                    self.db_manager.upsert_channel(int(ch_id), title, username)
                    updated += 1
                    logger.info(
                        f"Updated channel metadata for {ch_id}: title='{title}', username='{username}'"
                    )
                except Exception:
                    logger.exception(f"Failed to fetch metadata for channel {ch_id}")
        return updated
