#!/usr/bin/env python3
"""Standalone inference script for Hugging Face models."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from transformers import pipeline

from ai.training.utils import normalize_text

CLASSIFIER_DIR = Path(
    os.getenv("HF_CLASSIFIER_MODEL_PATH", "ai/models/signal_classifier")
)
NER_DIR = Path(os.getenv("HF_NER_MODEL_PATH", "ai/models/ner_extractor"))


@lru_cache(maxsize=1)
def _load_classifier():
    """Load the HF classification pipeline with error handling."""
    if not CLASSIFIER_DIR.exists():
        raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_DIR}")

    return pipeline(
        "text-classification",
        model=str(CLASSIFIER_DIR),
        tokenizer=str(CLASSIFIER_DIR),
        top_k=None,
    )


@lru_cache(maxsize=1)
def _load_ner():
    """Load the HF NER pipeline with error handling."""
    if not NER_DIR.exists():
        raise FileNotFoundError(f"NER model not found at {NER_DIR}")

    return pipeline(
        "token-classification",
        model=str(NER_DIR),
        tokenizer=str(NER_DIR),
        aggregation_strategy="simple",
    )


def classify_message(message: str) -> Dict[str, Any]:
    """Classify a message using the HF classifier pipeline."""
    classifier = _load_classifier()
    normalized = normalize_text(message, collapse_digit_spaces=True)
    results = classifier(normalized, return_all_scores=True)[0]

    best = max(results, key=lambda item: float(item["score"]))  # type: ignore[index,operator]
    return {
        "label": str(best["label"]),  # type: ignore[index]
        "score": float(best["score"]),  # type: ignore[index]
        "all_scores": results,
    }


def extract_entities(message: str) -> Dict[str, Any]:
    """Extract named entities using the HF NER pipeline."""
    ner = _load_ner()
    normalized = normalize_text(message, collapse_digit_spaces=True)
    raw_entities = ner(normalized)

    entities: Dict[str, Any] = {}
    for ent in raw_entities:
        label = str(ent["entity_group"])  # type: ignore[index]
        payload = {
            "text": str(ent["word"]),  # type: ignore[index]
            "start": int(ent["start"]),  # type: ignore[index]
            "end": int(ent["end"]),  # type: ignore[index]
            "confidence": float(ent["score"]),  # type: ignore[index]
        }
        entities.setdefault(label, []).append(payload)
    return entities


def predict(message: str) -> Dict[str, Any]:
    """Complete prediction pipeline: classification + entity extraction."""
    classification = classify_message(message)
    output = {"classification": classification, "entities": {}}

    if classification["label"] != "NON_SIGNAL":
        output["entities"] = extract_entities(message)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a message")
    parser.add_argument("message", help="Message text to analyze")
    args = parser.parse_args()

    result = predict(args.message)
    print(result)


if __name__ == "__main__":
    main()
