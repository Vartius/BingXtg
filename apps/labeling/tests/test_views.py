from __future__ import annotations

import os
import tempfile

from django.test import TestCase
from django.urls import reverse

from core.database.manager import DatabaseManager


class LabelingViewTests(TestCase):
    """Integration tests for the labeling dashboard view."""

    def setUp(self) -> None:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = db_path
        self.addCleanup(self._cleanup_temp_db)
        self.db_manager = DatabaseManager(self.db_path)
        self.settings_override = self.settings(LABELING_DB_PATH=self.db_path)
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def _cleanup_temp_db(self) -> None:
        try:
            os.remove(self.db_path)
        except FileNotFoundError:  # pragma: no cover - best effort cleanup
            pass

    def test_get_displays_unlabeled_message(self) -> None:
        """The view should render the next unlabeled message."""
        self.db_manager.save_message(12345, "Test signal message")

        response = self.client.get(reverse("labeling:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test signal message")
        self.assertContains(response, "Message ID")

    def test_post_persists_label_and_redirects(self) -> None:
        """Submitting the form should save the label and refresh the queue."""
        self.db_manager.save_message(67890, "Not a signal")
        message_rows = self.db_manager.get_unlabeled_messages(limit=1)
        self.assertTrue(message_rows)
        message_row = message_rows[0]
        message_id = message_row["id"]
        channel_id = message_row["channel_id"]

        payload = {
            "message_id": str(message_id),
            "channel_id": str(channel_id),
            "is_signal": "false",
            "direction": "",
            "pair": "",
            "entry": "",
            "stop_loss": "",
            "leverage": "",
            "targets": "",
        }

        response = self.client.post(reverse("labeling:index"), payload)

        self.assertEqual(response.status_code, 302)
        label_row = self.db_manager.get_label_by_message_id(message_id)
        self.assertIsNotNone(label_row)
        assert label_row is not None
        self.assertFalse(bool(label_row["is_signal"]))

    def test_empty_queue_renders_empty_state(self) -> None:
        """When no messages remain, the UI should present an empty state copy."""
        response = self.client.get(reverse("labeling:index"))

        self.assertContains(response, "Everything is labeled")
        self.assertEqual(response.status_code, 200)
