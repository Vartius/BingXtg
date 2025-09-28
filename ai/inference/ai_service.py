"""
This module uses already trained AI models to parse Telegram messages.

This service provides AI-powered signal detection and information extraction
using trained spaCy models for classification and Named Entity Recognition.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from dotenv import load_dotenv
from loguru import logger

from ai.training.utils import normalize_text as training_normalize_text
from ai.training.utils import relabel_numeric_entities

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    pipeline = None  # type: ignore

load_dotenv()  # Load environment variables from .env file
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "ai/models/ner_model")
IS_SIGNAL_MODEL_PATH = os.getenv("IS_SIGNAL_MODEL_PATH", "ai/models/is_signal_model")
DIRECTION_MODEL_PATH = os.getenv("DIRECTION_MODEL_PATH", "ai/models/direction_model")
HF_CLASSIFIER_PATH = Path(
    os.getenv("HF_CLASSIFIER_MODEL_PATH", "ai/models/signal_classifier")
)
HF_NER_PATH = Path(os.getenv("HF_NER_MODEL_PATH", "ai/models/ner_extractor"))


class AIInferenceService:
    """
    Service for AI-powered parsing of Telegram messages using trained models.

    Uses three trained spaCy models:
    - is_signal_model: Binary classification to detect if message contains trading signal
    - direction_model: Classify trading direction (LONG/SHORT)
    - ner_model: Named Entity Recognition to extract coins, targets, stop losses
    """

    def __init__(self):
        """Initialize the AI inference service."""
        self.nlp_is_signal = None
        self.nlp_direction = None
        self.nlp_ner = None
        self.hf_classifier = None
        self.hf_ner = None
        self._models_loaded = False

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by replacing common Cyrillic lookalikes with Latin equivalents.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text with Cyrillic characters replaced
        """
        return training_normalize_text(text, collapse_digit_spaces=True)

    def _try_load_hf_classifier(self) -> bool:
        if pipeline is None:
            logger.debug("transformers pipeline not available; skipping HF classifier")
            return False
        if not HF_CLASSIFIER_PATH.exists():
            logger.debug("HF classifier path %s does not exist", HF_CLASSIFIER_PATH)
            return False

        try:
            self.hf_classifier = pipeline(
                "text-classification",
                model=str(HF_CLASSIFIER_PATH),
                tokenizer=str(HF_CLASSIFIER_PATH),
                top_k=None,
            )
            logger.info("Hugging Face classifier loaded from %s", HF_CLASSIFIER_PATH)
            return True
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error(f"Failed to load Hugging Face classifier: {exc}")
            self.hf_classifier = None
            return False

    def _try_load_hf_ner(self) -> bool:
        if pipeline is None:
            return False
        if not HF_NER_PATH.exists():
            return False

        try:
            self.hf_ner = pipeline(
                "token-classification",
                model=str(HF_NER_PATH),
                tokenizer=str(HF_NER_PATH),
                aggregation_strategy="simple",
            )
            logger.info("Hugging Face NER loaded from %s", HF_NER_PATH)
            return True
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error(f"Failed to load Hugging Face NER pipeline: {exc}")
            self.hf_ner = None
            return False

    def _try_load_spacy_signal(self) -> bool:
        try:
            self.nlp_is_signal = spacy.load(str(IS_SIGNAL_MODEL_PATH))
            logger.info("spaCy signal model loaded successfully")
            return True
        except OSError as exc:
            logger.warning(f"spaCy signal model unavailable: {exc}")
            self.nlp_is_signal = None
            return False

    def _try_load_spacy_direction(self) -> bool:
        try:
            self.nlp_direction = spacy.load(str(DIRECTION_MODEL_PATH))
            logger.info("spaCy direction model loaded successfully")
            return True
        except OSError as exc:
            logger.warning(f"spaCy direction model unavailable: {exc}")
            self.nlp_direction = None
            return False

    def _try_load_spacy_ner(self) -> bool:
        try:
            self.nlp_ner = spacy.load(str(NER_MODEL_PATH))
            logger.info("spaCy NER model loaded successfully")
            return True
        except OSError as exc:
            logger.warning(f"spaCy NER model unavailable: {exc}")
            self.nlp_ner = None
            return False

    def load_models(self) -> bool:
        """
        Load all trained AI models.

        Returns:
            True if all models loaded successfully, False otherwise
        """
        classifier_loaded = self._try_load_hf_classifier()
        ner_loaded = self._try_load_hf_ner()

        direction_loaded = classifier_loaded  # HF classifier carries direction labels

        if not classifier_loaded:
            classifier_loaded = self._try_load_spacy_signal()
            direction_loaded = (
                self._try_load_spacy_direction() if classifier_loaded else False
            )

        if not ner_loaded:
            ner_loaded = self._try_load_spacy_ner()

        self._models_loaded = classifier_loaded and direction_loaded

        if not classifier_loaded:
            logger.error("No classification model could be loaded; AI service disabled")
        if not ner_loaded:
            logger.warning("NER model not available; entity extraction will be empty")

        return self._models_loaded

    def _hf_classify(self, normalized_text: str) -> Dict[str, Any]:
        if self.hf_classifier is None:
            raise RuntimeError("Hugging Face classifier not loaded")

        results = self.hf_classifier(normalized_text, return_all_scores=True)[0]
        best = max(
            results,
            key=lambda item: float(item["score"]),  # type: ignore[index]
        )
        label = str(best["label"])  # type: ignore[index]
        score = float(best["score"])  # type: ignore[index]
        return {"label": label, "score": score, "all_scores": results}

    @staticmethod
    def _direction_from_label(label: str) -> str:
        mapping = {
            "SIGNAL_LONG": "LONG",
            "SIGNAL_SHORT": "SHORT",
            "SIGNAL_NONE": "NONE",
        }
        return mapping.get(label, "NONE")

    def is_available(self) -> bool:
        """
        Check if the AI service is available and models are loaded.

        Returns:
            True if models are loaded and service is ready
        """
        if not self._models_loaded:
            return False
        if self.hf_classifier is not None:
            return True
        return self.nlp_is_signal is not None and self.nlp_direction is not None

    def is_signal(self, message: str) -> Tuple[bool, float]:
        """
        Check if message contains a trading signal.

        Args:
            message: Input message text

        Returns:
            Tuple of (is_signal: bool, confidence: float)
        """
        if not self.is_available():
            logger.warning("AI service not available for signal detection")
            return False, 0.0

        try:
            normalized_text = self.normalize_text(message)
            if self.hf_classifier is not None:
                result = self._hf_classify(normalized_text)
                is_signal = result["label"] != "NON_SIGNAL"
                logger.debug(
                    "HF signal detection: %s (label=%s, confidence=%.3f)",
                    is_signal,
                    result["label"],
                    result["score"],
                )
                return is_signal, float(result["score"])

            if self.nlp_is_signal is None:
                logger.warning("spaCy signal model not loaded")
                return False, 0.0

            doc = self.nlp_is_signal(normalized_text)
            signal_prob = doc.cats.get("signal", 0.0)
            is_signal = signal_prob > 0.5

            logger.debug(
                f"Signal detection: {is_signal} (confidence: {signal_prob:.3f})"
            )
            return is_signal, signal_prob

        except Exception as e:
            logger.error(f"Error in signal detection: {e}")
            return False, 0.0

    def get_direction(self, message: str) -> Tuple[str, float]:
        """
        Extract trading direction from message.

        Args:
            message: Input message text

        Returns:
            Tuple of (direction: str, confidence: float)
        """
        if not self.is_available():
            logger.warning("AI service not available for direction classification")
            return "NONE", 0.0

        try:
            normalized_text = self.normalize_text(message)
            if self.hf_classifier is not None:
                result = self._hf_classify(normalized_text)
                direction = self._direction_from_label(result["label"])
                confidence = float(result["score"])

                logger.debug(
                    "HF direction classification: %s (confidence: %.3f)",
                    direction,
                    confidence,
                )
                return direction, confidence

            if self.nlp_direction is None:
                logger.warning("spaCy direction model not loaded")
                return "NONE", 0.0

            doc = self.nlp_direction(normalized_text)

            # Find direction with highest confidence
            direction_cats = doc.cats
            if direction_cats:
                direction = max(direction_cats.keys(), key=lambda k: direction_cats[k])
                confidence = direction_cats[direction]

                logger.debug(
                    f"Direction classification: {direction} (confidence: {confidence:.3f})"
                )
                return direction, confidence
            else:
                return "NONE", 0.0

        except Exception as e:
            logger.error(f"Error in direction classification: {e}")
            return "NONE", 0.0

    def extract_entities(self, message: str) -> List[Dict[str, Any]]:
        """
        Extract named entities (coins, targets, stop losses) from message.

        Args:
            message: Input message text

        Returns:
            List of extracted entities with labels and positions
        """
        if self.hf_ner is None and self.nlp_ner is None:
            logger.debug("NER model not available")
            return []

        try:
            normalized_text = self.normalize_text(message)
            if self.hf_ner is not None:
                hf_entities = self.hf_ner(normalized_text)
                raw_entities = [
                    {
                        "text": ent["word"],
                        "label": ent["entity_group"],
                        "start": ent["start"],
                        "end": ent["end"],
                        "confidence": float(ent["score"]),
                    }
                    for ent in hf_entities
                ]
            else:
                if self.nlp_ner is None:
                    logger.debug("spaCy NER model not loaded")
                    return []

                doc = self.nlp_ner(normalized_text)
                raw_entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": getattr(ent, "score", 1.0),
                    }
                    for ent in doc.ents
                ]

            entities = relabel_numeric_entities(normalized_text, raw_entities)

            logger.debug(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []

    @staticmethod
    def _extract_numeric_value(raw_text: str) -> Optional[float]:
        """Extract the first floating-point number from the provided text."""
        if not raw_text:
            return None

        cleaned = raw_text.replace(",", ".")
        cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
        cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", cleaned)
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not match:
            return None

        try:
            return float(match.group(0))
        except ValueError:
            return None

    def parse_signal(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Complete signal parsing pipeline.

        Args:
            message: Input message text

        Returns:
            Dictionary with parsed signal information or None if not a signal
        """
        if not self.is_available():
            logger.warning("AI service not available for signal parsing")
            return None

        normalized_text = self.normalize_text(message)

        if self.hf_classifier is not None:
            classification = self._hf_classify(normalized_text)
            if classification["label"] == "NON_SIGNAL":
                logger.debug("Message classified as non-signal (HF classifier)")
                return None

            signal_confidence = float(classification["score"])
            direction = self._direction_from_label(classification["label"])
            direction_confidence = signal_confidence
        else:
            # Step 1: Check if message is a signal using spaCy models
            is_signal, signal_confidence = self.is_signal(message)

            if not is_signal:
                logger.debug("Message classified as non-signal")
                return None

            # Step 2: Extract direction via spaCy model
            direction, direction_confidence = self.get_direction(message)
            classification = {
                "label": "SIGNAL_LONG"
                if direction == "LONG"
                else "SIGNAL_SHORT"
                if direction == "SHORT"
                else "SIGNAL_NONE",
                "score": signal_confidence,
                "all_scores": [],
            }

        # Step 3: Extract entities (coins, targets, etc.)
        entities = self.extract_entities(message)

        # Organize entities by type
        pairs = [e for e in entities if e["label"] in {"PAIR", "COIN", "SYMBOL"}]
        entries = [e for e in entities if e["label"] in {"ENTRY", "ENTRY_PRICE"}]
        targets = [e for e in entities if e["label"] in {"TARGET", "TP"}]
        stop_losses = [
            e for e in entities if e["label"] in {"SL", "STOP_LOSS", "STOPLOSS"}
        ]
        leverages = [e for e in entities if e["label"] == "LEVERAGE"]

        primary_pair = pairs[0]["text"].strip() if pairs else None
        primary_entry = (
            self._extract_numeric_value(entries[0]["text"]) if entries else None
        )
        primary_stop = (
            self._extract_numeric_value(stop_losses[0]["text"]) if stop_losses else None
        )
        leverage_value = (
            self._extract_numeric_value(leverages[0]["text"]) if leverages else None
        )
        target_values = [
            value
            for value in (
                self._extract_numeric_value(target["text"]) for target in targets
            )
            if value is not None
        ]

        result = {
            "is_signal": True,
            "signal_confidence": signal_confidence,
            "direction": direction,
            "direction_confidence": direction_confidence,
            "entities": {
                "coins": pairs,
                "pairs": pairs,
                "entries": entries,
                "targets": targets,
                "stop_losses": stop_losses,
                "leverages": leverages,
                "all": entities,
            },
            "pair": primary_pair,
            "entry": primary_entry,
            "stop_loss": primary_stop,
            "leverage": leverage_value,
            "targets": target_values,
            "targets_numeric": target_values,
            "normalized_text": normalized_text,
            "classification": classification,
        }

        logger.info(
            f"Signal parsed: {direction} signal with {len(pairs)} pair candidates and {len(targets)} targets"
        )
        return result


# Global service instance
_ai_service_instance: Optional[AIInferenceService] = None


def get_ai_service() -> Optional[AIInferenceService]:
    """
    Get or create the global AI inference service instance.

    Returns:
        AIInferenceService instance if models can be loaded, None otherwise
    """
    global _ai_service_instance

    if _ai_service_instance is None:
        _ai_service_instance = AIInferenceService()

        if not _ai_service_instance.load_models():
            logger.error("Failed to load AI models, service will not be available")
            _ai_service_instance = None

    return _ai_service_instance


def parse_message_with_ai(message: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to parse a message using the AI service.

    Args:
        message: Input message text

    Returns:
        Parsed signal information or None
    """
    service = get_ai_service()
    if service is None:
        return None

    return service.parse_signal(message)
