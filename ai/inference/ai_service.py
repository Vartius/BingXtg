"""
This module uses already trained AI models to parse Telegram messages.

This service provides AI-powered signal detection and information extraction
using trained spaCy models for classification and Named Entity Recognition.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from dotenv import load_dotenv
from loguru import logger

from ai.training.utils import normalize_text as training_normalize_text
from ai.training.utils import relabel_numeric_entities

load_dotenv()  # Load environment variables from .env file
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "ai/models/ner_model")
IS_SIGNAL_MODEL_PATH = os.getenv("IS_SIGNAL_MODEL_PATH", "ai/models/is_signal_model")
DIRECTION_MODEL_PATH = os.getenv("DIRECTION_MODEL_PATH", "ai/models/direction_model")


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

    def load_models(self) -> bool:
        """
        Load all trained AI models.

        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            # Load signal detection model
            self.nlp_is_signal = spacy.load(IS_SIGNAL_MODEL_PATH)
            logger.info("Signal detection model loaded successfully")

            # Load direction classification model
            self.nlp_direction = spacy.load(str(DIRECTION_MODEL_PATH))
            logger.info("Direction classification model loaded successfully")

            self.nlp_ner = spacy.load(str(NER_MODEL_PATH))
            logger.info("NER model loaded successfully")

            self._models_loaded = True
            return True

        except OSError as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Please ensure all models are trained and available")
            return False

    def is_available(self) -> bool:
        """
        Check if the AI service is available and models are loaded.

        Returns:
            True if models are loaded and service is ready
        """
        return (
            self._models_loaded
            and self.nlp_is_signal is not None
            and self.nlp_direction is not None
        )

    def is_signal(self, message: str) -> Tuple[bool, float]:
        """
        Check if message contains a trading signal.

        Args:
            message: Input message text

        Returns:
            Tuple of (is_signal: bool, confidence: float)
        """
        if not self.is_available() or self.nlp_is_signal is None:
            logger.warning("AI service not available for signal detection")
            return False, 0.0

        try:
            normalized_text = self.normalize_text(message)
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
        if not self.is_available() or self.nlp_direction is None:
            logger.warning("AI service not available for direction classification")
            return "NONE", 0.0

        try:
            normalized_text = self.normalize_text(message)
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
        if self.nlp_ner is None:
            logger.debug("NER model not available")
            return []

        try:
            normalized_text = self.normalize_text(message)
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

        # Step 1: Check if message is a signal
        is_signal, signal_confidence = self.is_signal(message)

        if not is_signal:
            logger.debug("Message classified as non-signal")
            return None

        # Step 2: Extract direction
        direction, direction_confidence = self.get_direction(message)

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
            "normalized_text": self.normalize_text(message),
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
