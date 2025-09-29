#!/usr/bin/env python3
"""
Simplified AI model testing script for HuggingFace models with custom metrics.
Evaluates models using labeled data with a custom scoring system.

Custom Metrics:
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Signal Detection Rate: Correctly identified signals / Total signals
- Confidence Score: Average prediction confidence for correct predictions
"""

import os
import torch
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
import sys
import logging
import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils import (
    load_database_connection,
)

# HuggingFace imports
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

from dotenv import load_dotenv

load_dotenv()

# Model paths
CLASSIFIER_MODEL_PATH = os.getenv(
    "CLASSIFIER_MODEL_PATH", "ai/models/signal_classifier"
)
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "ai/models/ner_extractor")


class HFAITester:
    """HuggingFace AI model tester with custom metrics."""

    def __init__(self, db_path: str = "total.db", batch_size: int = 32):
        """
        Initialize the tester.

        Args:
            db_path: Path to the SQLite database with labeled data
            batch_size: Batch size for processing multiple examples at once
        """
        self.db_path = db_path
        self.batch_size = batch_size

        # Model components
        self.classifier_model = None
        self.classifier_tokenizer = None
        self.ner_model = None
        self.ner_tokenizer = None
        self.ner_pipeline = None  # Cache the NER pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Custom metric weights
        self.metric_weights = {
            "signal_detection": 0.4,  # 40% weight for detecting signals correctly
            "direction_accuracy": 0.3,  # 30% weight for direction prediction
            "entity_extraction": 0.2,  # 20% weight for entity extraction
            "confidence_score": 0.1,  # 10% weight for prediction confidence
        }

        # Label mappings
        self.classifier_labels = [
            "NON_SIGNAL",
            "SIGNAL_LONG",
            "SIGNAL_SHORT",
            "SIGNAL_NONE",
        ]
        self.label2id = {label: idx for idx, label in enumerate(self.classifier_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def load_models(self) -> bool:
        """
        Load HuggingFace models for testing.

        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading models on device: {self.device}")

            # Load classifier model
            logger.info(f"Loading classifier from: {CLASSIFIER_MODEL_PATH}")
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH
            )
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                CLASSIFIER_MODEL_PATH
            )
            self.classifier_model.to(self.device)
            logger.info("✓ Classifier model loaded successfully")

            # Load NER model
            logger.info(f"Loading NER model from: {NER_MODEL_PATH}")
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                NER_MODEL_PATH
            )
            self.ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
            self.ner_model.to(self.device)

            # Create and cache NER pipeline once
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.ner_tokenizer,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("✓ NER model and pipeline loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict_signal_and_direction(self, text: str) -> Dict[str, Any]:
        """
        Predict signal and direction using classifier model.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with predictions
        """
        if self.classifier_model is None or self.classifier_tokenizer is None:
            return {"is_signal": None, "direction": None, "confidence": 0.0}

        try:
            # Tokenize input
            inputs = self.classifier_tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()

            # Map to our format
            predicted_label = self.classifier_labels[predicted_class]

            if predicted_label == "NON_SIGNAL":
                is_signal = 0
                direction = "NONE"
            elif predicted_label == "SIGNAL_LONG":
                is_signal = 1
                direction = "LONG"
            elif predicted_label == "SIGNAL_SHORT":
                is_signal = 1
                direction = "SHORT"
            else:  # SIGNAL_NONE
                is_signal = 1
                direction = "NONE"

            return {
                "is_signal": is_signal,
                "direction": direction,
                "confidence": confidence,
                "raw_prediction": predicted_label,
            }

        except Exception as e:
            logger.error(f"Error in classification prediction: {e}")
            return {"is_signal": None, "direction": None, "confidence": 0.0}

    def predict_entities(self, text: str) -> Dict[str, Any]:
        """
        Predict entities using NER model.

        Args:
            text: Input text to extract entities from

        Returns:
            Dictionary with extracted entities
        """
        if self.ner_model is None or self.ner_tokenizer is None:
            return {
                "pair": None,
                "stop_loss": None,
                "entry": None,
                "leverage": None,
                "targets": [],
            }

        try:
            # Use cached pipeline for NER prediction
            if self.ner_pipeline is None:
                return {
                    "pair": None,
                    "stop_loss": None,
                    "entry": None,
                    "leverage": None,
                    "targets": [],
                }

            # Get NER predictions
            entities = self.ner_pipeline(text)

            # Parse entities
            extracted = {
                "pair": None,
                "stop_loss": None,
                "entry": None,
                "leverage": None,
                "targets": [],
            }

            for entity in entities:
                entity_type = entity["entity_group"].upper()
                entity_text = entity["word"].strip()

                if "PAIR" in entity_type:
                    # Extract coin from pair (e.g., "BTC/USDT" -> "BTC")
                    if "/" in entity_text:
                        extracted["pair"] = entity_text.split("/")[0]
                    else:
                        extracted["pair"] = entity_text
                elif "STOP_LOSS" in entity_type:
                    try:
                        extracted["stop_loss"] = float(entity_text.replace(",", ""))
                    except ValueError:
                        pass
                elif "ENTRY" in entity_type:
                    try:
                        extracted["entry"] = float(entity_text.replace(",", ""))
                    except ValueError:
                        pass
                elif "LEVERAGE" in entity_type:
                    try:
                        # Remove 'x' suffix if present
                        leverage_text = entity_text.lower().replace("x", "")
                        extracted["leverage"] = float(leverage_text)
                    except ValueError:
                        pass
                elif "TARGET" in entity_type:
                    try:
                        target_value = float(entity_text.replace(",", ""))
                        extracted["targets"].append(str(target_value))
                    except ValueError:
                        pass

            return extracted

        except Exception as e:
            logger.error(f"Error in NER prediction: {e}")
            return {
                "pair": None,
                "stop_loss": None,
                "entry": None,
                "leverage": None,
                "targets": [],
            }

    def predict_with_models(self, text: str) -> Dict[str, Any]:
        """
        Make predictions using all loaded models.

        Args:
            text: Input message to analyze

        Returns:
            Dictionary with all predictions
        """
        predictions = {}

        # Get classification predictions
        class_results = self.predict_signal_and_direction(text)
        predictions.update(class_results)

        # Get NER predictions
        ner_results = self.predict_entities(text)
        predictions["ner_entities"] = ner_results

        return predictions

    def predict_signal_and_direction_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict signal and direction for a batch of texts using classifier model.

        Args:
            texts: List of input texts to classify

        Returns:
            List of dictionaries with predictions
        """
        if self.classifier_model is None or self.classifier_tokenizer is None:
            return [
                {"is_signal": None, "direction": None, "confidence": 0.0} for _ in texts
            ]

        try:
            # Tokenize all inputs at once
            inputs = self.classifier_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions for all texts
            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=-1)
                confidences = torch.max(predictions, dim=-1).values

            results = []
            for i in range(len(texts)):
                predicted_class = predicted_classes[i].item()
                confidence = confidences[i].item()
                predicted_label = self.classifier_labels[predicted_class]

                if predicted_label == "NON_SIGNAL":
                    is_signal = 0
                    direction = "NONE"
                elif predicted_label == "SIGNAL_LONG":
                    is_signal = 1
                    direction = "LONG"
                elif predicted_label == "SIGNAL_SHORT":
                    is_signal = 1
                    direction = "SHORT"
                else:  # SIGNAL_NONE
                    is_signal = 1
                    direction = "NONE"

                results.append(
                    {
                        "is_signal": is_signal,
                        "direction": direction,
                        "confidence": confidence,
                        "raw_prediction": predicted_label,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in batch classification prediction: {e}")
            return [
                {"is_signal": None, "direction": None, "confidence": 0.0} for _ in texts
            ]

    def predict_entities_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict entities for a batch of texts using cached NER pipeline.

        Args:
            texts: List of input texts to extract entities from

        Returns:
            List of dictionaries with extracted entities
        """
        if self.ner_pipeline is None:
            return [
                {
                    "pair": None,
                    "stop_loss": None,
                    "entry": None,
                    "leverage": None,
                    "targets": [],
                }
                for _ in texts
            ]

        try:
            # Use cached pipeline for batch prediction
            batch_entities = self.ner_pipeline(texts)

            results = []
            for entities in batch_entities:
                extracted = {
                    "pair": None,
                    "stop_loss": None,
                    "entry": None,
                    "leverage": None,
                    "targets": [],
                }

                for entity in entities:
                    entity_type = entity["entity_group"].upper()
                    entity_text = entity["word"].strip()

                    if "PAIR" in entity_type:
                        if "/" in entity_text:
                            extracted["pair"] = entity_text.split("/")[0]
                        else:
                            extracted["pair"] = entity_text
                    elif "STOP_LOSS" in entity_type:
                        try:
                            extracted["stop_loss"] = float(entity_text.replace(",", ""))
                        except ValueError:
                            pass
                    elif "ENTRY" in entity_type:
                        try:
                            extracted["entry"] = float(entity_text.replace(",", ""))
                        except ValueError:
                            pass
                    elif "LEVERAGE" in entity_type:
                        try:
                            leverage_text = entity_text.lower().replace("x", "")
                            extracted["leverage"] = float(leverage_text)
                        except ValueError:
                            pass
                    elif "TARGET" in entity_type:
                        try:
                            target_value = float(entity_text.replace(",", ""))
                            extracted["targets"].append(str(target_value))
                        except ValueError:
                            pass

                results.append(extracted)

            return results

        except Exception as e:
            logger.error(f"Error in batch NER prediction: {e}")
            return [
                {
                    "pair": None,
                    "stop_loss": None,
                    "entry": None,
                    "leverage": None,
                    "targets": [],
                }
                for _ in texts
            ]

    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from the database.

        Returns:
            List of test examples with ground truth labels
        """
        conn = load_database_connection(self.db_path)
        cursor = conn.cursor()

        # Get all labeled data
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
            direction_map = {0: "NONE", 1: "LONG", 2: "SHORT"}
            return direction_map.get(direction, "NONE")

        if isinstance(direction, str):
            direction = direction.upper().strip()
            if direction in ["LONG", "SHORT", "NONE"]:
                return direction

        return "NONE"

    def calculate_precision_recall(
        self, tp: int, fp: int, fn: int
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1_score

    def calculate_confidence_score(
        self, predictions: List[Dict], correct_predictions: List[bool]
    ) -> float:
        """
        Calculate average confidence score for correct predictions.

        Args:
            predictions: List of prediction dictionaries
            correct_predictions: List of boolean indicating correct predictions

        Returns:
            Average confidence score for correct predictions
        """
        confidence_scores = []

        for pred, is_correct in zip(predictions, correct_predictions):
            if is_correct and "confidence" in pred:
                confidence_scores.append(pred["confidence"])

        return (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

    def evaluate_signal_detection(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate signal detection performance using live model predictions.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary with signal detection metrics
        """
        tp = fp = fn = tn = 0

        logger.info("Evaluating signal detection...")

        # Process in batches for speed
        for batch_start in range(0, len(test_data), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(test_data))
            batch = test_data[batch_start:batch_end]

            if batch_start % (self.batch_size * 50) == 0 and batch_start > 0:
                logger.info(
                    f"Signal detection - Processed {batch_start}/{len(test_data)} examples"
                )

            # Get batch predictions
            texts = [example["message"] for example in batch]
            batch_predictions = self.predict_signal_and_direction_batch(texts)

            # Process batch results
            for example, prediction in zip(batch, batch_predictions):
                gt_signal = example["is_signal"]
                pred_signal = prediction.get("is_signal", None)

                if pred_signal is not None:
                    if gt_signal == 1 and pred_signal == 1:
                        tp += 1
                    elif gt_signal == 0 and pred_signal == 1:
                        fp += 1
                    elif gt_signal == 1 and pred_signal == 0:
                        fn += 1
                    elif gt_signal == 0 and pred_signal == 0:
                        tn += 1

        precision, recall, f1_score = self.calculate_precision_recall(tp, fp, fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

    def evaluate_direction_prediction(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate direction prediction performance using live model predictions.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary with direction prediction metrics
        """
        correct = 0
        total = 0
        direction_counts = {
            "LONG": {"correct": 0, "total": 0},
            "SHORT": {"correct": 0, "total": 0},
            "NONE": {"correct": 0, "total": 0},
        }

        logger.info("Evaluating direction prediction...")

        # Filter signal examples for direction evaluation
        signal_examples = [
            ex
            for ex in test_data
            if ex["is_signal"] == 1 and ex["direction"] is not None
        ]

        # Process in batches for speed
        for batch_start in range(0, len(signal_examples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(signal_examples))
            batch = signal_examples[batch_start:batch_end]

            if batch_start % (self.batch_size * 50) == 0 and batch_start > 0:
                logger.info(
                    f"Direction prediction - Processed {batch_start}/{len(signal_examples)} signal examples"
                )

            # Get batch predictions
            texts = [example["message"] for example in batch]
            batch_predictions = self.predict_signal_and_direction_batch(texts)

            # Process batch results
            for example, prediction in zip(batch, batch_predictions):
                total += 1
                gt_direction = self.normalize_direction(example["direction"])
                pred_direction = self.normalize_direction(
                    prediction.get("direction", None)
                )

                if gt_direction in direction_counts:
                    direction_counts[gt_direction]["total"] += 1

                if gt_direction == pred_direction:
                    correct += 1
                    if gt_direction in direction_counts:
                        direction_counts[gt_direction]["correct"] += 1

        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-direction accuracies
        direction_accuracies = {}
        for direction, counts in direction_counts.items():
            if counts["total"] > 0:
                direction_accuracies[direction.lower()] = (
                    counts["correct"] / counts["total"]
                )
            else:
                direction_accuracies[direction.lower()] = 0.0

        return {
            "overall_accuracy": accuracy,
            "correct_predictions": correct,
            "total_predictions": total,
            "per_direction": direction_accuracies,
        }

    def evaluate_entity_extraction(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate entity extraction performance using live model predictions.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary with entity extraction metrics
        """
        entity_stats = {
            "pair": {"correct": 0, "total": 0},
            "stop_loss": {"correct": 0, "total": 0},
            "entry": {"correct": 0, "total": 0},
            "leverage": {"correct": 0, "total": 0},
            "targets": {"correct": 0, "total": 0},
        }

        logger.info("Evaluating entity extraction...")

        # Filter signal examples for entity evaluation
        signal_examples = [ex for ex in test_data if ex["is_signal"] == 1]

        # Process in batches for speed
        for batch_start in range(0, len(signal_examples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(signal_examples))
            batch = signal_examples[batch_start:batch_end]

            if batch_start % (self.batch_size * 50) == 0 and batch_start > 0:
                logger.info(
                    f"Entity extraction - Processed {batch_start}/{len(signal_examples)} signal examples"
                )

            # Get batch predictions
            texts = [example["message"] for example in batch]
            batch_ner_predictions = self.predict_entities_batch(texts)

            # Process batch results
            for example, ner_entities in zip(batch, batch_ner_predictions):
                # Check pair extraction
                if example["pair"] is not None:
                    entity_stats["pair"]["total"] += 1
                    gt_pair = str(example["pair"]).upper()
                    if "/" in gt_pair:
                        gt_pair = gt_pair.split("/")[0]
                    pred_pair = ner_entities.get("pair")
                    if pred_pair and str(pred_pair).upper() == gt_pair:
                        entity_stats["pair"]["correct"] += 1

                # Check numeric fields
                for field in ["stop_loss", "entry", "leverage"]:
                    if example[field] is not None:
                        entity_stats[field]["total"] += 1
                        pred_value = ner_entities.get(field)
                        try:
                            if (
                                pred_value is not None
                                and abs(float(pred_value) - float(example[field]))
                                < 0.01
                            ):
                                entity_stats[field]["correct"] += 1
                        except (ValueError, TypeError):
                            pass

                # Check targets (simplified)
                if example["targets"] is not None:
                    entity_stats["targets"]["total"] += 1
                    pred_targets = ner_entities.get("targets", [])
                    # For now, just check if any targets were extracted
                    if pred_targets:
                        entity_stats["targets"]["correct"] += 1

        # Calculate accuracies
        accuracies = {}
        overall_correct = 0
        overall_total = 0

        for entity, stats in entity_stats.items():
            if stats["total"] > 0:
                accuracies[entity] = stats["correct"] / stats["total"]
                overall_correct += stats["correct"]
                overall_total += stats["total"]
            else:
                accuracies[entity] = 0.0

        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "per_entity": accuracies,
            "entity_stats": entity_stats,
        }

    def calculate_custom_score(self, metrics: Dict[str, Dict]) -> float:
        """
        Calculate custom weighted score.

        Args:
            metrics: Dictionary containing all metric results

        Returns:
            Weighted custom score
        """
        signal_score = metrics["signal_detection"]["f1_score"]
        direction_score = metrics["direction_prediction"]["overall_accuracy"]
        entity_score = metrics["entity_extraction"]["overall_accuracy"]
        confidence_score = (
            0.8  # Placeholder as we don't have confidence scores in current data
        )

        custom_score = (
            signal_score * self.metric_weights["signal_detection"]
            + direction_score * self.metric_weights["direction_accuracy"]
            + entity_score * self.metric_weights["entity_extraction"]
            + confidence_score * self.metric_weights["confidence_score"]
        )

        return custom_score

    def run_comprehensive_test(
        self, max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive test on all labeled data.

        Args:
            max_examples: Maximum number of examples to test. If None, tests all data.

        Returns:
            Dictionary with test results and custom metrics
        """
        logger.info("Starting HuggingFace AI model test...")

        # Load models first
        if not self.load_models():
            return {"error": "Failed to load models"}

        # Load test data
        test_data = self.load_test_data()
        if not test_data:
            return {"error": "No test data found"}

        # Use a subset if requested
        if max_examples is not None and len(test_data) > max_examples:
            logger.info(
                f"Using first {max_examples} examples out of {len(test_data)} for testing"
            )
            test_data = test_data[:max_examples]
        else:
            logger.info(f"Testing all {len(test_data)} examples...")

        logger.info(f"Starting evaluation on {len(test_data)} examples...")

        # Evaluate different aspects
        signal_metrics = self.evaluate_signal_detection(test_data)
        direction_metrics = self.evaluate_direction_prediction(test_data)
        entity_metrics = self.evaluate_entity_extraction(test_data)

        # Combine all metrics
        results = {
            "signal_detection": signal_metrics,
            "direction_prediction": direction_metrics,
            "entity_extraction": entity_metrics,
            "dataset_stats": {
                "total_examples": len(test_data),
                "signal_examples": sum(1 for x in test_data if x["is_signal"] == 1),
                "non_signal_examples": sum(1 for x in test_data if x["is_signal"] == 0),
            },
        }

        # Calculate custom score
        results["custom_score"] = self.calculate_custom_score(results)

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

        print("\n" + "=" * 70)
        print("HUGGINGFACE AI MODEL TEST RESULTS - CUSTOM METRICS")
        print("=" * 70)

        print(f"\nCustom Weighted Score: {results['custom_score']:.3f}")

        # Dataset statistics
        stats = results["dataset_stats"]
        print("\nDataset Statistics:")
        print(f"Total Examples: {stats['total_examples']}")
        print(f"Signal Examples: {stats['signal_examples']}")
        print(f"Non-Signal Examples: {stats['non_signal_examples']}")

        # Signal detection metrics
        signal = results["signal_detection"]
        print("\nSignal Detection Performance:")
        print(f"Precision: {signal['precision']:.3f}")
        print(f"Recall: {signal['recall']:.3f}")
        print(f"F1-Score: {signal['f1_score']:.3f}")
        print(f"Accuracy: {signal['accuracy']:.3f}")
        print(
            f"TP: {signal['true_positives']}, FP: {signal['false_positives']}, "
            f"FN: {signal['false_negatives']}, TN: {signal['true_negatives']}"
        )

        # Direction prediction metrics
        direction = results["direction_prediction"]
        print("\nDirection Prediction Performance:")
        print(f"Overall Accuracy: {direction['overall_accuracy']:.3f}")
        print(
            f"Correct: {direction['correct_predictions']}/{direction['total_predictions']}"
        )
        print("Per-direction accuracies:")
        for dir_name, acc in direction["per_direction"].items():
            print(f"  {dir_name.upper()}: {acc:.3f}")

        # Entity extraction metrics
        entity = results["entity_extraction"]
        print("\nEntity Extraction Performance:")
        print(f"Overall Accuracy: {entity['overall_accuracy']:.3f}")
        print("Per-entity accuracies:")
        for entity_name, acc in entity["per_entity"].items():
            stats = entity["entity_stats"][entity_name]
            print(
                f"  {entity_name.capitalize()}: {acc:.3f} ({stats['correct']}/{stats['total']})"
            )

        # Metric weights
        print("\nMetric Weights Used:")
        for metric, weight in self.metric_weights.items():
            print(f"  {metric.replace('_', ' ').title()}: {weight:.1%}")

        print("\n" + "=" * 70)


def main():
    """Main function to run the HuggingFace test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HuggingFace AI model comprehensive test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to test",
    )
    args = parser.parse_args()

    # Enable memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    tester = HFAITester(batch_size=args.batch_size)
    results = tester.run_comprehensive_test(max_examples=args.max_examples)
    tester.print_results(results)


if __name__ == "__main__":
    main()
