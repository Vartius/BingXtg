"""
Django models for trading bot app.
"""

from django.db import models


class AppState(models.Model):
    """
    Stores application state key-value pairs.
    """

    key = models.TextField(primary_key=True, help_text="State key")
    value = models.TextField(null=True, blank=True, help_text="State value")

    class Meta:
        db_table = "app_state"

    def __str__(self):
        return f"{self.key}: {self.value}"


class Trade(models.Model):
    """
    Represents an active or completed trade.
    """

    trade_id = models.TextField(primary_key=True, help_text="Unique trade identifier")
    channel_id = models.BigIntegerField(help_text="Source channel ID")
    coin = models.TextField(help_text="Trading pair/coin")
    direction = models.TextField(help_text="Trade direction (LONG/SHORT)")
    targets = models.TextField(help_text="JSON array of target prices")
    leverage = models.FloatField(help_text="Leverage multiplier")
    stop_loss = models.FloatField(null=True, blank=True, help_text="Stop loss price")
    margin = models.FloatField(help_text="Margin amount")
    entry_price = models.FloatField(help_text="Entry price")
    current_price = models.FloatField(null=True, blank=True, help_text="Current price")
    pnl = models.FloatField(null=True, blank=True, help_text="Profit/Loss amount")
    pnl_percent = models.FloatField(
        null=True, blank=True, help_text="Profit/Loss percentage"
    )
    status = models.TextField(default="PENDING", help_text="Trade status")
    created_at = models.DateTimeField(
        auto_now_add=True, help_text="Trade creation time"
    )
    updated_at = models.DateTimeField(auto_now=True, help_text="Last update time")
    closed_at = models.DateTimeField(
        null=True, blank=True, help_text="Trade close time"
    )
    close_reason = models.TextField(
        null=True, blank=True, help_text="Reason for closing"
    )

    class Meta:
        db_table = "trades"

    def __str__(self):
        return f"{self.trade_id} - {self.coin} {self.direction}"


class TradingStats(models.Model):
    """
    Stores trading statistics and metrics.
    """

    id = models.AutoField(primary_key=True)
    date = models.DateField(help_text="Date for these stats")
    total_trades = models.IntegerField(default=0, help_text="Total number of trades")
    winning_trades = models.IntegerField(
        default=0, help_text="Number of winning trades"
    )
    losing_trades = models.IntegerField(default=0, help_text="Number of losing trades")
    total_pnl = models.FloatField(default=0.0, help_text="Total profit/loss")
    win_rate = models.FloatField(default=0.0, help_text="Win rate percentage")

    class Meta:
        db_table = "trading_stats"
        indexes = [
            models.Index(fields=["date"], name="idx_trading_stats_date"),
        ]

    def __str__(self):
        return f"Stats for {self.date}"
