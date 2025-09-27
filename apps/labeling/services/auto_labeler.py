from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.inference.ai_service import AIInferenceService, get_ai_service
from core.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class AutoLabelSummary:
    """Aggregate result of an auto-labeling run."""

    scanned: int = 0
    labeled_signals: int = 0
    labeled_non_signals: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def as_message(self) -> str:
        """Human-readable summary for UI flash messages."""
        parts = [f"Scanned {self.scanned} msgs"]
        if self.labeled_signals:
            parts.append(f"{self.labeled_signals} signals saved")
        if self.labeled_non_signals:
            parts.append(f"{self.labeled_non_signals} non-signals saved")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "labeled_signals": self.labeled_signals,
            "labeled_non_signals": self.labeled_non_signals,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class AutoLabelingService:
    """Batch labeler that relies on the spaCy inference pipelines."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        min_signal_confidence: float = 0.6,
        label_non_signals: bool = True,
    ) -> None:
        self.db_manager = DatabaseManager(db_path) if db_path else DatabaseManager()
        self.ai_service: Optional[AIInferenceService] = get_ai_service()
        self.min_signal_confidence = min_signal_confidence
        self.label_non_signals = label_non_signals

    def label_next_batch(self, limit: int = 10) -> AutoLabelSummary:
        """Label the next batch of unlabeled messages.

        Args:
            limit: Maximum number of messages to process.

        Returns:
            Summary with counts for processed messages.
        """
        summary = AutoLabelSummary()

        if self.ai_service is None or not self.ai_service.is_available():
            message = (
                "AI inference service is not available; ensure models are trained."
            )
            logger.warning(message)
            summary.errors.append(message)
            return summary

        rows = self.db_manager.get_unlabeled_messages(limit=limit)
        summary.scanned = len(rows)

        for row in rows:
            message_id = int(row["id"])
            channel_id = int(row["channel_id"])
            message_text = row["message"]

            try:
                parsed = self.ai_service.parse_signal(message_text)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to parse message %s", message_id, exc_info=exc)
                summary.errors.append(f"{message_id}: {exc}")
                summary.skipped += 1
                continue

            if not parsed:
                if self.label_non_signals:
                    self._store_non_signal(message_id, channel_id, message_text)
                    summary.labeled_non_signals += 1
                else:
                    summary.skipped += 1
                continue

            signal_confidence = float(parsed.get("signal_confidence", 0.0))
            if signal_confidence < self.min_signal_confidence:
                logger.debug(
                    "Skipping message %s due to confidence %.3f < %.3f",
                    message_id,
                    signal_confidence,
                    self.min_signal_confidence,
                )
                summary.skipped += 1
                continue

            payload = self._build_payload(message_id, channel_id, message_text, parsed)
            try:
                self.db_manager.save_label(**payload)
            except Exception as exc:  # pragma: no cover - database guard
                logger.exception("Failed to save label for message %s", message_id)
                summary.errors.append(f"{message_id}: {exc}")
                summary.skipped += 1
                continue

            summary.labeled_signals += 1

        return summary

    def _store_non_signal(
        self, message_id: int, channel_id: int, message_text: str
    ) -> None:
        """Persist a non-signal decision to the database."""
        try:
            self.db_manager.save_label(
                message_id=message_id,
                channel_id=channel_id,
                message=message_text,
                is_signal=False,
            )
        except Exception:  # pragma: no cover - database guard
            logger.exception(
                "Failed to save non-signal label for message %s", message_id
            )
            raise

    @staticmethod
    def _build_payload(
        message_id: int, channel_id: int, message_text: str, parsed: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "message_id": message_id,
            "channel_id": channel_id,
            "message": message_text,
            "is_signal": True,
            "direction": AutoLabelingService._map_direction(parsed.get("direction")),
            "pair": AutoLabelingService._normalize_pair(parsed.get("pair")),
            "entry": AutoLabelingService._to_float(parsed.get("entry")),
            "stop_loss": AutoLabelingService._to_float(parsed.get("stop_loss")),
            "leverage": AutoLabelingService._to_float(parsed.get("leverage")),
            "targets": AutoLabelingService._serialize_targets(parsed.get("targets")),
        }
        return payload

    @staticmethod
    def _map_direction(direction: Optional[str]) -> Optional[int]:
        if not direction:
            return None
        direction_upper = direction.upper()
        if direction_upper.startswith("LONG"):
            return 0
        if direction_upper.startswith("SHORT"):
            return 1
        return None

    @staticmethod
    def _normalize_pair(pair: Optional[str]) -> Optional[str]:
        if not pair:
            return None
        return pair.strip().upper().replace(" ", "")

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_targets(targets: Any) -> Optional[str]:
        if not targets:
            return None
        if isinstance(targets, list):
            cleaned = [AutoLabelingService._to_float(item) for item in targets]
            cleaned_values = [value for value in cleaned if value is not None]
            if not cleaned_values:
                return None
            return json.dumps(cleaned_values)
        if isinstance(targets, str):
            try:
                parsed = json.loads(targets)
                if isinstance(parsed, list):
                    return AutoLabelingService._serialize_targets(parsed)
            except json.JSONDecodeError:
                return None
        return None
