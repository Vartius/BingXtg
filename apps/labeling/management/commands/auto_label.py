from __future__ import annotations

import json
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from apps.labeling.services.auto_labeler import AutoLabelingService
from apps.labeling.utils import resolve_labeling_database_path


class Command(BaseCommand):
    help = "Automatically label a batch of messages using the AI inference pipeline."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--limit",
            type=int,
            default=25,
            help="Maximum number of unlabeled messages to process (default: 25).",
        )
        parser.add_argument(
            "--min-confidence",
            dest="min_confidence",
            type=float,
            default=0.6,
            help="Minimum probability required to accept an AI signal prediction (default: 0.6).",
        )
        parser.add_argument(
            "--skip-non-signals",
            action="store_true",
            dest="skip_non_signals",
            help="Skip saving non-signals so that they remain available for manual review.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="as_json",
            help="Output the summary as JSON for scripting integrations.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        limit = max(1, int(options["limit"]))
        min_confidence = float(options["min_confidence"])
        label_non_signals = not bool(options["skip_non_signals"])
        as_json = bool(options["as_json"])

        service = AutoLabelingService(
            resolve_labeling_database_path(),
            min_signal_confidence=min_confidence,
            label_non_signals=label_non_signals,
        )
        summary = service.label_next_batch(limit=limit)

        if as_json:
            self.stdout.write(json.dumps(summary.to_dict(), indent=2))
            return

        message = summary.as_message()
        if summary.errors:
            self.stdout.write(self.style.WARNING(f"Completed with issues: {message}"))
            for error in summary.errors:
                self.stderr.write(f"  - {error}")
        else:
            self.stdout.write(self.style.SUCCESS(f"Completed successfully: {message}"))
