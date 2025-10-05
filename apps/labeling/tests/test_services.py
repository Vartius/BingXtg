from __future__ import annotations

from django.test import SimpleTestCase

from ai.training.utils import normalize_text
from apps.labeling.services.auto_labeler import AutoLabelingService


class AutoLabelingServiceTests(SimpleTestCase):
    """Unit tests for helper utilities in AutoLabelingService."""

    def test_to_float_handles_spaced_decimal(self) -> None:
        value = AutoLabelingService._to_float("27 000.5")
        self.assertEqual(value, 27000.5)

    def test_to_float_handles_non_breaking_space_and_comma(self) -> None:
        spaced = "27\u00a0000,5"
        value = AutoLabelingService._to_float(spaced)
        self.assertEqual(value, 27000.5)

    def test_to_float_returns_none_for_non_numeric(self) -> None:
        value = AutoLabelingService._to_float("not a number")
        self.assertIsNone(value)

    def test_normalize_text_preserves_digits_by_default(self) -> None:
        text = "Entry 27 000"
        normalized = normalize_text(text)
        self.assertEqual(normalized, "Entry 27 000")

    def test_normalize_text_collapses_digit_spaces_when_requested(self) -> None:
        text = "Target 27 000"
        normalized = normalize_text(text, collapse_digit_spaces=True)
        self.assertEqual(normalized, "Target 27000")
