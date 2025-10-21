"""
Django models for Telegram client app.
"""

from django.db import models


class Channel(models.Model):
    """
    Represents a Telegram channel that the bot monitors.
    """

    channel_id = models.BigIntegerField(
        primary_key=True, help_text="Telegram channel ID (stored as positive integer)"
    )
    title = models.TextField(null=True, blank=True, help_text="Channel title")
    username = models.TextField(
        null=True, blank=True, help_text="Channel username (without @)"
    )
    updated_at = models.DateTimeField(auto_now=True, help_text="Last update timestamp")

    class Meta:
        db_table = "channels"
        ordering = ["channel_id"]

    def __str__(self):
        return f"{self.title or self.username or self.channel_id}"


class Message(models.Model):
    """
    Represents a message received from Telegram channels.
    """

    id = models.AutoField(primary_key=True)
    channel_id = models.BigIntegerField(help_text="Telegram channel ID")
    message = models.TextField(help_text="Message content")
    is_signal = models.BooleanField(
        default=False, help_text="Whether this message is a trading signal"
    )
    regex = models.TextField(
        null=True, blank=True, help_text="Regex pattern used to extract signal"
    )

    class Meta:
        db_table = "messages"
        indexes = [
            models.Index(fields=["channel_id"], name="idx_messages_channel_id"),
        ]

    def __str__(self):
        return f"Message {self.id} from {self.channel_id}"
