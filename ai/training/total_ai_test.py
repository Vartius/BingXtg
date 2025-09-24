#!/usr/bin/env python3
"""
Comprehensive AI model testing script that evaluates all models using labeled data.
Calculates accuracy based on a point system where each field has specific point values.

Point System:
1) is_signal = false: 4p
2) is_signal = true: 2p
3) direction: 2p
4) pair: 2p
5) stop_loss: 1p
6) targets: 1p for each one
7) entry: 1p
8) leverage: 1p
"""

import sys
import os
import spacy
import json
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_training.utils import (
    normalize_text,
    load_database_connection,
    classify_signal_and_direction,
)


class TotalAITester:
    """Comprehensive AI model tester with point-based accuracy calculation."""

    def __init__(self, db_path: str = "total.db"):
        """
        Initialize the tester.

        Args:
            db_path: Path to the SQLite database with labeled data
        """
        self.db_path = db_path
        self.nlp_is_signal = None
        self.nlp_direction = None
        self.nlp_ner = None

        # Point values for each field
        self.point_values = {
            "is_signal_false": 4,  # Non-signal messages
            "is_signal_true": 2,  # Signal messages
            "direction": 2,
            "pair": 2,
            "stop_loss": 1,
            "entry": 1,
            "leverage": 1,
            "target": 1,  # 1 point per target
        }

    def load_models(self) -> bool:
        """
        Load all trained AI models.

        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            # Load classification models
            self.nlp_is_signal = spacy.load("is_signal_model")
            self.nlp_direction = spacy.load("direction_model")
            logger.info("Classification models loaded successfully")

            # Try to load NER model
            try:
                self.nlp_ner = spacy.load("./ner_model")
                logger.info("NER model loaded successfully")
            except OSError:
                logger.warning("NER model not found, will skip NER evaluation")

            return True

        except OSError as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Please run the training scripts first")
            return False

    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from the database.

        Returns:
            List of test examples with ground truth labels
        """
        conn = load_database_connection(self.db_path)
        cursor = conn.cursor()

        # Get all labeled data with ground truth and AI predictions
        query = """
        SELECT 
            message, is_signal, direction, pair, stop_loss, leverage, targets, entry,
            ai_is_signal, ai_direction, ai_pair, ai_stop_loss, ai_leverage, ai_targets, ai_entry
        FROM labeled 
        WHERE message IS NOT NULL
        ORDER BY id
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        test_data = []
        for row in rows:
            # Convert row to dictionary
            data = {
                "message": row["message"],
                "is_signal": row["is_signal"],
                "direction": row["direction"],
                "pair": row["pair"],
                "stop_loss": row["stop_loss"],
                "leverage": row["leverage"],
                "targets": row["targets"],
                "entry": row["entry"],
                "ai_is_signal": row["ai_is_signal"],
                "ai_direction": row["ai_direction"],
                "ai_pair": row["ai_pair"],
                "ai_stop_loss": row["ai_stop_loss"],
                "ai_leverage": row["ai_leverage"],
                "ai_targets": row["ai_targets"],
                "ai_entry": row["ai_entry"],
            }
            test_data.append(data)

        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data

    def normalize_direction(self, direction: Any) -> Optional[str]:
        """
        Normalize direction value to standard format.

        Args:
            direction: Direction value (could be int, str, etc.)

        Returns:
            Normalized direction string or None
        """
        if direction is None:
            return None

        if isinstance(direction, int):
            # Map integer values to strings
            direction_map = {0: "NONE", 1: "LONG", 2: "SHORT"}
            return direction_map.get(direction, "NONE")

        if isinstance(direction, str):
            direction = direction.upper().strip()
            if direction in ["LONG", "SHORT", "NONE"]:
                return direction

        return "NONE"

    def parse_targets(self, targets_str: Optional[str]) -> List[str]:
        """
        Parse targets string into list of target values.

        Args:
            targets_str: Comma-separated or JSON string of targets

        Returns:
            List of target strings
        """
        if not targets_str:
            return []

        # Try to parse as JSON first
        try:
            targets = json.loads(targets_str)
            if isinstance(targets, list):
                return [str(t) for t in targets]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to comma-separated parsing
        if isinstance(targets_str, str):
            return [t.strip() for t in targets_str.split(",") if t.strip()]

        return []

    def extract_coin_name(self, pair: str) -> Optional[str]:
        """
        Extract just the coin name from a trading pair.

        Args:
            pair: Trading pair like "BTC/USDT" or "btc/usdt"

        Returns:
            Coin name like "BTC" or None if invalid
        """
        if not pair:
            return None

        # Common patterns for trading pairs
        pair = pair.upper().strip()

        # Handle slash separator (most common)
        if "/" in pair:
            return pair.split("/")[0].strip()

        # Handle common suffixes
        common_suffixes = ["USDT", "USDC", "BTC", "ETH", "USD", "EUR"]
        for suffix in common_suffixes:
            if pair.endswith(suffix) and len(pair) > len(suffix):
                return pair[: -len(suffix)].strip()

        # If no pattern matches, return as is (might be just coin name)
        return pair if pair else None

    def normalize_float_value(self, value: Any) -> Optional[float]:
        """
        Normalize and fix common float extraction issues.

        Args:
            value: Value to normalize (could be string, float, etc.)

        Returns:
            Normalized float value or None if invalid
        """
        if value is None:
            return None

        try:
            # Convert to string first for processing
            str_value = str(value).strip()

            # Remove common non-numeric characters but keep dots and commas
            # Handle cases like "25X" -> "25"
            import re

            # Remove letters and special chars except dots, commas, and spaces
            cleaned = re.sub(r"[^\d.,\s]", "", str_value)

            # Handle space-separated numbers like "68 000"
            if " " in cleaned:
                cleaned = cleaned.replace(" ", "")

            # Handle comma as thousands separator vs decimal separator
            comma_count = cleaned.count(",")
            dot_count = cleaned.count(".")

            if comma_count == 1 and dot_count == 0:
                # Could be decimal comma (European style) or thousands separator
                parts = cleaned.split(",")
                if len(parts[1]) <= 2:  # Likely decimal comma
                    cleaned = cleaned.replace(",", ".")
                else:  # Likely thousands separator
                    cleaned = cleaned.replace(",", "")
            elif comma_count == 1 and dot_count > 0:
                # European format like "1.234,56" (dots as thousands, comma as decimal)
                parts = cleaned.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Last comma looks like decimal separator
                    cleaned = parts[0].replace(".", "") + "." + parts[1]
                else:
                    # Treat commas as thousands separators
                    cleaned = cleaned.replace(",", "")
            elif comma_count > 0:
                # Multiple commas, likely thousands separators
                # Keep only the last comma if it looks like decimal
                parts = cleaned.split(",")
                if len(parts[-1]) <= 2:
                    # Last part looks like decimal
                    cleaned = "".join(parts[:-1]) + "." + parts[-1]
                else:
                    # Remove all commas
                    cleaned = cleaned.replace(",", "")

            # Handle leading dot like ".928" -> "0.928"
            if cleaned.startswith("."):
                cleaned = "0" + cleaned

            # Handle trailing dot
            if cleaned.endswith("."):
                cleaned = cleaned[:-1]

            # Convert to float
            result = float(cleaned)

            # Sanity check for reasonable values
            if result < 0:
                return None
            if result > 1000000:  # Very large numbers might be errors
                return None

            return result

        except (ValueError, TypeError, AttributeError):
            return None

    def safe_float_compare(self, val1: Any, val2: Any, tolerance: float = 0.01) -> bool:
        """
        Safely compare two values as floats with tolerance.
        Enhanced to handle various formats.

        Args:
            val1: First value to compare
            val2: Second value to compare
            tolerance: Tolerance for comparison

        Returns:
            True if values are approximately equal, False otherwise
        """
        f1 = self.normalize_float_value(val1)
        f2 = self.normalize_float_value(val2)

        if f1 is None or f2 is None:
            return False

        return abs(f1 - f2) < tolerance

    def predict_with_models(self, message: str) -> Dict[str, Any]:
        """
        Make predictions using all loaded models.

        Args:
            message: Input message to analyze

        Returns:
            Dictionary with all predictions
        """
        predictions = {}

        # Classification predictions
        if self.nlp_is_signal and self.nlp_direction:
            result = classify_signal_and_direction(
                message, self.nlp_is_signal, self.nlp_direction
            )
            predictions.update(result)

        # NER predictions
        if self.nlp_ner:
            normalized_text = normalize_text(message)
            doc = self.nlp_ner(normalized_text)

            # Extract entities
            entities = {
                "pair": None,
                "stop_loss": None,
                "leverage": None,
                "entry": None,
                "targets": [],
            }

            for ent in doc.ents:
                entity_text = ent.text.strip()
                if ent.label_ == "PAIR":
                    # Extract only coin name from pair
                    entities["pair"] = self.extract_coin_name(entity_text)
                elif ent.label_ == "STOP_LOSS":
                    # Normalize float value
                    normalized_value = self.normalize_float_value(entity_text)
                    if normalized_value is not None:
                        entities["stop_loss"] = normalized_value
                elif ent.label_ == "LEVERAGE":
                    # Normalize leverage (remove X, etc.)
                    normalized_value = self.normalize_float_value(entity_text)
                    if normalized_value is not None:
                        entities["leverage"] = normalized_value
                elif ent.label_ == "ENTRY":
                    # Normalize entry price
                    normalized_value = self.normalize_float_value(entity_text)
                    if normalized_value is not None:
                        entities["entry"] = normalized_value
                elif "TARGET" in ent.label_:
                    # Normalize target price
                    normalized_value = self.normalize_float_value(entity_text)
                    if normalized_value is not None:
                        entities["targets"].append(str(normalized_value))

            predictions["ner_entities"] = entities

        return predictions

    def calculate_points_for_example(
        self, ground_truth: Dict, predictions: Dict
    ) -> Tuple[int, int]:
        """
        Calculate points for a single example.

        Args:
            ground_truth: Ground truth labels
            predictions: Model predictions

        Returns:
            Tuple of (earned_points, total_possible_points)
        """
        earned_points = 0
        total_points = 0

        # 1. is_signal classification
        gt_is_signal = ground_truth["is_signal"]
        if gt_is_signal == 0:
            # Non-signal: 4 points possible
            total_points += self.point_values["is_signal_false"]
            pred_is_signal = predictions.get("is_signal", None)
            if pred_is_signal == 0:
                earned_points += self.point_values["is_signal_false"]
        else:
            # Signal: 2 points possible
            total_points += self.point_values["is_signal_true"]
            pred_is_signal = predictions.get("is_signal", None)
            if pred_is_signal == 1:
                earned_points += self.point_values["is_signal_true"]

        # Only evaluate other fields for signals
        if gt_is_signal == 1:
            # 2. Pair: 2 points
            if ground_truth["pair"] is not None:
                total_points += self.point_values["pair"]
                gt_pair = self.extract_coin_name(ground_truth["pair"])
                pred_pair = predictions.get("ai_pair", None) or predictions.get(
                    "ner_entities", {}
                ).get("pair", None)
                if pred_pair and gt_pair:
                    if pred_pair.upper() == gt_pair.upper():
                        earned_points += self.point_values["pair"]

            # 3. Direction: 2 points
            if ground_truth["direction"] is not None:
                total_points += self.point_values["direction"]
                gt_direction = self.normalize_direction(ground_truth["direction"])
                pred_direction = self.normalize_direction(
                    predictions.get("direction", None)
                )
                if pred_direction == gt_direction:
                    earned_points += self.point_values["direction"]

            # 4. Stop loss: 1 point
            # 4. Stop loss: 1 point
            if ground_truth["stop_loss"] is not None:
                total_points += self.point_values["stop_loss"]
                # Check both direct prediction and NER prediction
                pred_stop_loss = predictions.get(
                    "ai_stop_loss", None
                ) or predictions.get("ner_entities", {}).get("stop_loss", None)
                if self.safe_float_compare(pred_stop_loss, ground_truth["stop_loss"]):
                    earned_points += self.point_values["stop_loss"]

            # 5. Entry: 1 point
            if ground_truth["entry"] is not None:
                total_points += self.point_values["entry"]
                pred_entry = predictions.get("ai_entry", None) or predictions.get(
                    "ner_entities", {}
                ).get("entry", None)
                if self.safe_float_compare(pred_entry, ground_truth["entry"]):
                    earned_points += self.point_values["entry"]

            # 6. Leverage: 1 point
            if ground_truth["leverage"] is not None:
                total_points += self.point_values["leverage"]
                pred_leverage = predictions.get("ai_leverage", None) or predictions.get(
                    "ner_entities", {}
                ).get("leverage", None)
                if self.safe_float_compare(pred_leverage, ground_truth["leverage"]):
                    earned_points += self.point_values["leverage"]

            # 7. Targets: 1 point per target
            gt_targets = self.parse_targets(ground_truth["targets"])
            if gt_targets:
                total_points += len(gt_targets) * self.point_values["target"]
                pred_targets = self.parse_targets(
                    predictions.get("ai_targets", None)
                ) or predictions.get("ner_entities", {}).get("targets", [])

                # Count matching targets
                for gt_target in gt_targets:
                    for pred_target in pred_targets:
                        if self.safe_float_compare(gt_target, pred_target):
                            earned_points += self.point_values["target"]
                            break

        return earned_points, total_points

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test on all labeled data.

        Returns:
            Dictionary with test results and statistics
        """
        logger.info("Starting comprehensive AI model test...")

        # Load models
        if not self.load_models():
            return {"error": "Failed to load models"}

        # Load test data
        test_data = self.load_test_data()
        if not test_data:
            return {"error": "No test data found"}

        total_earned_points = 0
        total_possible_points = 0

        # Track statistics
        stats = {
            "total_examples": len(test_data),
            "signal_examples": 0,
            "non_signal_examples": 0,
            "is_signal_correct": 0,
            "direction_correct": 0,
            "direction_total": 0,
            "field_stats": {
                "pair": {"correct": 0, "total": 0},
                "stop_loss": {"correct": 0, "total": 0},
                "entry": {"correct": 0, "total": 0},
                "leverage": {"correct": 0, "total": 0},
                "targets": {"correct": 0, "total": 0},
            },
        }

        logger.info(f"Testing {len(test_data)} examples...")

        for i, example in enumerate(test_data):
            if i % 1000 == 0:
                logger.info(f"Processing example {i}/{len(test_data)}")

            # Make predictions
            predictions = self.predict_with_models(example["message"])

            # Calculate points
            earned, total = self.calculate_points_for_example(example, predictions)
            total_earned_points += earned
            total_possible_points += total

            # Update statistics
            if example["is_signal"] == 1:
                stats["signal_examples"] += 1
            else:
                stats["non_signal_examples"] += 1

            # is_signal accuracy
            pred_is_signal = predictions.get("is_signal", None)
            if pred_is_signal == example["is_signal"]:
                stats["is_signal_correct"] += 1

            # Direction accuracy (only for signals)
            if example["is_signal"] == 1 and example["direction"] is not None:
                stats["direction_total"] += 1
                gt_direction = self.normalize_direction(example["direction"])
                pred_direction = self.normalize_direction(
                    predictions.get("direction", None)
                )
                if pred_direction == gt_direction:
                    stats["direction_correct"] += 1

            # Field-specific statistics
            if example["is_signal"] == 1:
                # Pair evaluation
                if example["pair"] is not None:
                    stats["field_stats"]["pair"]["total"] += 1
                    gt_pair = self.extract_coin_name(example["pair"])
                    pred_pair = predictions.get("ai_pair", None) or predictions.get(
                        "ner_entities", {}
                    ).get("pair", None)
                    if pred_pair and gt_pair:
                        if pred_pair.upper() == gt_pair.upper():
                            stats["field_stats"]["pair"]["correct"] += 1

                # Numeric fields evaluation
                for field in ["stop_loss", "entry", "leverage"]:
                    if example[field] is not None:
                        stats["field_stats"][field]["total"] += 1
                        # Check if predicted correctly
                        pred_value = predictions.get(
                            f"ai_{field}", None
                        ) or predictions.get("ner_entities", {}).get(field, None)
                        if self.safe_float_compare(pred_value, example[field]):
                            stats["field_stats"][field]["correct"] += 1

                # Targets
                gt_targets = self.parse_targets(example["targets"])
                if gt_targets:
                    stats["field_stats"]["targets"]["total"] += len(gt_targets)
                    pred_targets = self.parse_targets(
                        predictions.get("ai_targets", None)
                    ) or predictions.get("ner_entities", {}).get("targets", [])
                    for gt_target in gt_targets:
                        for pred_target in pred_targets:
                            if self.safe_float_compare(gt_target, pred_target):
                                stats["field_stats"]["targets"]["correct"] += 1
                                break

        # Calculate final accuracy
        overall_accuracy = (
            total_earned_points / total_possible_points
            if total_possible_points > 0
            else 0.0
        )

        results = {
            "overall_accuracy": overall_accuracy,
            "total_earned_points": total_earned_points,
            "total_possible_points": total_possible_points,
            "statistics": stats,
        }

        # Calculate individual accuracies
        results["is_signal_accuracy"] = (
            stats["is_signal_correct"] / stats["total_examples"]
        )
        results["direction_accuracy"] = (
            stats["direction_correct"] / stats["direction_total"]
            if stats["direction_total"] > 0
            else 0.0
        )

        # Field accuracies
        results["field_accuracies"] = {}
        for field, field_stats in stats["field_stats"].items():
            if field_stats["total"] > 0:
                results["field_accuracies"][field] = (
                    field_stats["correct"] / field_stats["total"]
                )
            else:
                results["field_accuracies"][field] = 0.0

        return results

    def print_results(self, results: Dict[str, Any]):
        """
        Print formatted test results.

        Args:
            results: Test results dictionary
        """
        if "error" in results:
            logger.error(f"Test failed: {results['error']}")
            return

        print("\n" + "=" * 60)
        print("COMPREHENSIVE AI MODEL TEST RESULTS")
        print("=" * 60)

        print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Points Earned: {results['total_earned_points']}")
        print(f"Points Possible: {results['total_possible_points']}")

        stats = results["statistics"]
        print("\nDataset Statistics:")
        print(f"Total Examples: {stats['total_examples']}")
        print(f"Signal Examples: {stats['signal_examples']}")
        print(f"Non-Signal Examples: {stats['non_signal_examples']}")

        print("\nModel Performance:")
        print(f"is_signal Accuracy: {results['is_signal_accuracy']:.2%}")
        print(
            f"Direction Accuracy: {results['direction_accuracy']:.2%} "
            f"({stats['direction_correct']}/{stats['direction_total']})"
        )

        print("\nField-Specific Accuracies:")
        for field, accuracy in results["field_accuracies"].items():
            field_stats = stats["field_stats"][field]
            print(
                f"{field.capitalize()}: {accuracy:.2%} "
                f"({field_stats['correct']}/{field_stats['total']})"
            )

        print("\n" + "=" * 60)


def main():
    """Main function to run the comprehensive test."""
    tester = TotalAITester()
    results = tester.run_comprehensive_test()
    tester.print_results(results)


if __name__ == "__main__":
    main()
