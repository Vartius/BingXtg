"""
Django models for labeling app.
"""

from django.db import models


class LabeledMessage(models.Model):
    """
    Represents a labeled trading signal message.
    """

    id = models.AutoField(primary_key=True)
    message_id = models.IntegerField(unique=True, help_text="Original message ID")
    channel_id = models.BigIntegerField(help_text="Telegram channel ID")
    message = models.TextField(help_text="Message content")
    is_signal = models.BooleanField(help_text="Whether this is a trading signal")
    labeled_at = models.DateTimeField(
        auto_now_add=True, help_text="When this message was labeled"
    )

    # Trading signal fields
    direction = models.IntegerField(
        null=True, blank=True, help_text="Trade direction (1=long, -1=short)"
    )
    pair = models.TextField(
        null=True, blank=True, help_text="Trading pair (e.g., BTC, ETH)"
    )
    stop_loss = models.FloatField(null=True, blank=True, help_text="Stop loss price")
    take_profit = models.FloatField(
        null=True, blank=True, help_text="Take profit price"
    )
    leverage = models.FloatField(null=True, blank=True, help_text="Leverage multiplier")
    targets = models.TextField(
        null=True, blank=True, help_text="JSON array of target prices"
    )
    entry = models.FloatField(null=True, blank=True, help_text="Entry price")

    # AI prediction fields
    ai_is_signal = models.IntegerField(
        null=True, blank=True, help_text="AI prediction: is signal"
    )
    ai_confidence = models.FloatField(
        null=True, blank=True, help_text="AI confidence score"
    )
    ai_direction = models.TextField(
        null=True, blank=True, help_text="AI predicted direction"
    )
    ai_pair = models.TextField(null=True, blank=True, help_text="AI extracted pair")
    ai_stop_loss = models.FloatField(
        null=True, blank=True, help_text="AI extracted stop loss"
    )
    ai_take_profit = models.FloatField(
        null=True, blank=True, help_text="AI extracted take profit"
    )
    ai_leverage = models.FloatField(
        null=True, blank=True, help_text="AI extracted leverage"
    )
    ai_targets = models.TextField(
        null=True, blank=True, help_text="AI extracted targets"
    )
    ai_entry = models.FloatField(null=True, blank=True, help_text="AI extracted entry")

    class Meta:
        db_table = "labeled"
        indexes = [
            models.Index(fields=["message_id"], name="idx_labeled_message_id"),
            models.Index(fields=["channel_id"], name="idx_labeled_channel_id"),
        ]

    def __str__(self):
        return f"Labeled {self.message_id} ({'signal' if self.is_signal else 'not signal'})"
