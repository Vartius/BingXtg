"""
This module uses already trained AI models to parse Telegram messages.

This service provides AI-powered signal detection and information extraction
using trained spaCy models for classification and Named Entity Recognition.
"""

import spacy
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
from pathlib import Path


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

        # Define model paths relative to project root
        self.project_root = Path(__file__).parent.parent
        self.is_signal_model_path = self.project_root / "is_signal_model"
        self.direction_model_path = self.project_root / "direction_model"
        self.ner_model_path = self.project_root / "ner_model"

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by replacing common Cyrillic lookalikes with Latin equivalents.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text with Cyrillic characters replaced
        """
        if text is None:
            return None

        # Cyrillic to Latin mapping for common lookalike characters
        cyrillic_to_latin = {
            # Uppercase letters
            "А": "A",
            "В": "B",
            "Е": "E",
            "К": "K",
            "М": "M",
            "Н": "H",
            "О": "O",
            "Р": "P",
            "С": "C",
            "Т": "T",
            "Х": "X",
            # Lowercase letters
            "а": "a",
            "в": "b",
            "е": "e",
            "к": "k",
            "м": "m",
            "н": "h",
            "о": "o",
            "р": "p",
            "с": "c",
            "т": "t",
            "х": "x",
        }

        # Replace Cyrillic characters
        for cyrillic, latin in cyrillic_to_latin.items():
            text = text.replace(cyrillic, latin)

        # Replace comma with dot for decimal numbers
        text = text.replace(",", ".")

        return text

    def load_models(self) -> bool:
        """
        Load all trained AI models.

        Returns:
            True if all models loaded successfully, False otherwise
        """
        try:
            # Load signal detection model
            if self.is_signal_model_path.exists():
                self.nlp_is_signal = spacy.load(str(self.is_signal_model_path))
                logger.info("Signal detection model loaded successfully")
            else:
                logger.warning(f"Signal model not found at {self.is_signal_model_path}")
                return False

            # Load direction classification model
            if self.direction_model_path.exists():
                self.nlp_direction = spacy.load(str(self.direction_model_path))
                logger.info("Direction classification model loaded successfully")
            else:
                logger.warning(
                    f"Direction model not found at {self.direction_model_path}"
                )
                return False

            # Load NER model (optional - may not always be available)
            if self.ner_model_path.exists():
                try:
                    self.nlp_ner = spacy.load(str(self.ner_model_path))
                    logger.info("NER model loaded successfully")
                except OSError as e:
                    logger.warning(f"NER model failed to load: {e}")
                    # NER is optional, continue without it

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

            entities = []
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": getattr(
                            ent, "score", 1.0
                        ),  # spaCy doesn't always provide scores
                    }
                )

            logger.debug(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []

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
        coins = [e for e in entities if e["label"] in ["COIN", "SYMBOL"]]
        targets = [e for e in entities if e["label"] in ["TARGET", "TP"]]
        stop_losses = [e for e in entities if e["label"] in ["SL", "STOP_LOSS"]]

        result = {
            "is_signal": True,
            "signal_confidence": signal_confidence,
            "direction": direction,
            "direction_confidence": direction_confidence,
            "entities": {
                "coins": coins,
                "targets": targets,
                "stop_losses": stop_losses,
                "all": entities,
            },
            "normalized_text": self.normalize_text(message),
        }

        logger.info(
            f"Signal parsed: {direction} signal with {len(coins)} coins and {len(targets)} targets"
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
