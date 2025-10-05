from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from django import forms


class LabelingForm(forms.Form):
    """Form used to capture labeling decisions for a Telegram message."""

    SIGNAL_CHOICES: Tuple[Tuple[str, str], ...] = (
        ("true", "Signal"),
        ("false", "Not a signal"),
    )
    DIRECTION_CHOICES: Tuple[Tuple[str, str], ...] = (
        ("", "--"),
        ("0", "Long"),
        ("1", "Short"),
    )

    message_id = forms.IntegerField(widget=forms.HiddenInput())
    channel_id = forms.IntegerField(widget=forms.HiddenInput())
    is_signal = forms.ChoiceField(choices=SIGNAL_CHOICES, widget=forms.RadioSelect)
    direction = forms.ChoiceField(
        choices=DIRECTION_CHOICES,
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    pair = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "BTC/USDT"}),
    )
    entry = forms.DecimalField(
        required=False,
        decimal_places=6,
        max_digits=18,
        widget=forms.NumberInput(attrs={"step": "0.000001"}),
    )
    stop_loss = forms.DecimalField(
        required=False,
        decimal_places=6,
        max_digits=18,
        widget=forms.NumberInput(attrs={"step": "0.000001"}),
    )
    leverage = forms.DecimalField(
        required=False,
        decimal_places=2,
        max_digits=10,
        widget=forms.NumberInput(attrs={"step": "0.1"}),
    )
    targets = forms.CharField(
        required=False,
        help_text="Comma-separated take-profit targets (e.g., 0.0012, 0.0015).",
        widget=forms.TextInput(),
    )

    def __init__(
        self,
        *args: Any,
        suggested_values: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.suggested_values = suggested_values or {}
        if suggested_values:
            self._populate_initial_from_suggestions(suggested_values)

    def _populate_initial_from_suggestions(self, suggestions: Dict[str, Any]) -> None:
        direction = suggestions.get("direction")
        if direction in {"LONG", "SHORT"}:
            self.initial.setdefault("direction", "0" if direction == "LONG" else "1")
        if suggestions.get("pair"):
            self.initial.setdefault("pair", suggestions["pair"])
        if suggestions.get("entry") is not None:
            self.initial.setdefault("entry", suggestions["entry"])
        if suggestions.get("stop_loss") is not None:
            self.initial.setdefault("stop_loss", suggestions["stop_loss"])
        if suggestions.get("leverage") is not None:
            self.initial.setdefault("leverage", suggestions["leverage"])
        targets = suggestions.get("targets")
        if isinstance(targets, list) and targets:
            formatted_targets = ", ".join(str(t) for t in targets)
            self.initial.setdefault("targets", formatted_targets)
        if suggestions.get("is_signal") is True:
            self.initial.setdefault("is_signal", "true")

    def clean_is_signal(self) -> bool:
        value = self.cleaned_data.get("is_signal", "false")
        return value == "true"

    def clean_direction(self) -> Optional[int]:
        direction_value: Optional[str] = self.cleaned_data.get("direction")
        if not direction_value:
            return None
        return int(direction_value)

    def clean_targets(self) -> Optional[str]:
        raw_targets = self.cleaned_data.get("targets")
        if not raw_targets:
            return None

        try:
            targets_list = [
                self._convert_to_decimal(value)
                for value in raw_targets.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise forms.ValidationError(
                "Targets must be numbers separated by commas."
            ) from exc

        return json.dumps([float(value) for value in targets_list])

    def clean(self) -> Dict[str, Any]:
        cleaned = super().clean()
        is_signal = cleaned.get("is_signal", False)

        if is_signal:
            if cleaned.get("direction") is None:
                self.add_error("direction", "Direction is required for signals.")
            if not cleaned.get("pair"):
                self.add_error("pair", "Pair is required for signals.")

        return cleaned

    @staticmethod
    def _convert_to_decimal(value: str) -> Decimal:
        normalized = value.replace(" ", "")
        return Decimal(normalized)

    def to_database_payload(self, message_text: str) -> Dict[str, Any]:
        """Transform cleaned data into a payload consumable by the DatabaseManager."""
        if (
            not self.is_valid()
        ):  # pragma: no cover - guard, form should be validated earlier
            raise ValueError("Form must be valid before constructing payload")

        cleaned = self.cleaned_data
        return {
            "message_id": cleaned["message_id"],
            "channel_id": cleaned["channel_id"],
            "message": message_text,
            "is_signal": cleaned["is_signal"],
            "direction": cleaned.get("direction"),
            "pair": cleaned.get("pair"),
            "entry": float(cleaned["entry"])
            if cleaned.get("entry") is not None
            else None,
            "stop_loss": float(cleaned["stop_loss"])
            if cleaned.get("stop_loss") is not None
            else None,
            "leverage": float(cleaned["leverage"])
            if cleaned.get("leverage") is not None
            else None,
            "targets": cleaned.get("targets"),
        }
