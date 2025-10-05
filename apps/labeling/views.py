from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.functional import cached_property
from django.views import View

from ai.inference.ai_service import AIInferenceService, get_ai_service
from core.database.manager import DatabaseManager

from .forms import LabelingForm
from .services.auto_labeler import AutoLabelingService
from .utils import resolve_labeling_database_path

logger = logging.getLogger(__name__)


class LabelingDashboardView(View):
    """Django-based labeling workflow replacing the Textual interface."""

    template_name = "labeling/index.html"

    @cached_property
    def db_manager(self) -> DatabaseManager:
        return DatabaseManager(resolve_labeling_database_path())

    @cached_property
    def ai_service(self) -> Optional[AIInferenceService]:
        return get_ai_service()

    def get(self, request: HttpRequest) -> HttpResponse:
        message_row = self._fetch_next_message()
        if message_row is None:
            context = {"message": None, "form": None, "ai_suggestion": None}
            return render(request, self.template_name, context)

        ai_suggestion = self._build_ai_suggestion(message_row["message"])
        initial = {
            "message_id": message_row["id"],
            "channel_id": message_row["channel_id"],
            "is_signal": "true" if (ai_suggestion or {}).get("is_signal") else "false",
        }
        form = LabelingForm(initial=initial, suggested_values=ai_suggestion)

        context = {
            "message": message_row,
            "form": form,
            "ai_suggestion": ai_suggestion,
        }
        return render(request, self.template_name, context)

    def post(self, request: HttpRequest) -> HttpResponse:
        message_id = request.POST.get("message_id")
        if message_id is None:
            messages.error(request, "Missing message identifier in request.")
            return redirect(reverse("labeling:index"))

        try:
            message_id_int = int(message_id)
        except (TypeError, ValueError):
            messages.error(request, "Invalid message identifier supplied.")
            return redirect(reverse("labeling:index"))

        message_row = self.db_manager.get_message_by_id(message_id_int)
        if message_row is None:
            messages.error(
                request,
                "That message was not found. It may have been labeled by another session.",
            )
            return redirect(reverse("labeling:index"))

        ai_suggestion = self._build_ai_suggestion(message_row["message"])
        form = LabelingForm(request.POST, suggested_values=ai_suggestion)

        if form.is_valid():
            payload = form.to_database_payload(message_row["message"])
            try:
                self.db_manager.save_label(**payload)
            except (
                Exception
            ) as exception:  # pragma: no cover - database failures logged
                logger.exception("Failed to save label", exc_info=exception)
                messages.error(
                    request,
                    "Unable to persist label. Check logs for details.",
                )
                context = {
                    "message": message_row,
                    "form": form,
                    "ai_suggestion": ai_suggestion,
                }
                return render(request, self.template_name, context)

            messages.success(request, "Label saved. Loading the next message…")
            return redirect(reverse("labeling:index"))

        context = {
            "message": message_row,
            "form": form,
            "ai_suggestion": ai_suggestion,
        }
        return render(request, self.template_name, context, status=400)

    def _fetch_next_message(self) -> Optional[Dict[str, Any]]:
        rows = self.db_manager.get_unlabeled_messages(limit=1)
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row["id"],
            "channel_id": row["channel_id"],
            "message": row["message"],
        }

    def _build_ai_suggestion(self, message_text: str) -> Optional[Dict[str, Any]]:
        if not self.ai_service:
            return None

        try:
            parsed = self.ai_service.parse_signal(message_text)
        except Exception:  # pragma: no cover - AI failures logged for observability
            logger.exception("AI inference failed while suggesting labels")
            return None

        if not parsed:
            return None

        suggestion: Dict[str, Any] = {
            "is_signal": parsed.get("is_signal", False),
            "direction": parsed.get("direction"),
        }

        if parsed.get("pair"):
            suggestion["pair"] = parsed["pair"]
        else:
            entity_groups = parsed.get("entities") or {}
            pair_entities = entity_groups.get("pairs") or entity_groups.get("coins")
            if pair_entities:
                suggestion["pair"] = pair_entities[0].get("text")

        if parsed.get("entry") is not None:
            suggestion["entry"] = parsed["entry"]

        if parsed.get("stop_loss") is not None:
            suggestion["stop_loss"] = parsed["stop_loss"]

        if parsed.get("leverage") is not None:
            suggestion["leverage"] = parsed["leverage"]

        targets = parsed.get("targets")
        if isinstance(targets, list) and targets:
            suggestion["targets"] = targets

        return suggestion


class AutoLabelingTriggerView(View):
    """Kick off a batch auto-labeling run using the inference service."""

    http_method_names = ["post"]

    def post(self, request: HttpRequest) -> HttpResponse:
        limit = self._parse_positive_int(request.POST.get("batch_size"), default=10)
        min_confidence = self._parse_float(
            request.POST.get("min_confidence"), default=0.6
        )
        label_non_signals = request.POST.get("label_non_signals", "true") != "false"

        service = AutoLabelingService(
            resolve_labeling_database_path(),
            min_signal_confidence=min_confidence,
            label_non_signals=label_non_signals,
        )
        summary = service.label_next_batch(limit=limit)

        if summary.scanned == 0:
            messages.info(request, "No unlabeled messages were available to process.")
            return redirect(reverse("labeling:index"))

        message_text = summary.as_message()
        if summary.errors:
            messages.warning(
                request,
                f"Auto labeling completed with issues: {message_text}. Check logs for details.",
            )
        else:
            messages.success(
                request,
                f"Auto labeling completed successfully: {message_text}.",
            )

        return redirect(reverse("labeling:index"))

    @staticmethod
    def _parse_positive_int(value: Optional[str], default: int) -> int:
        try:
            parsed = int(value) if value is not None else default
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(value: Optional[str], default: float) -> float:
        try:
            parsed = float(value) if value is not None else default
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default
