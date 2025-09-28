#!/usr/bin/env python3
"""Standalone inference script for Hugging Face models."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Any, Dict

from transformers import pipeline

from ai.training.utils import normalize_text

CLASSIFIER_DIR = os.getenv("HF_CLASSIFIER_MODEL_PATH", "ai/models/signal_classifier")
NER_DIR = os.getenv("HF_NER_MODEL_PATH", "ai/models/ner_extractor")


@lru_cache(maxsize=1)
def _load_classifier():
    return pipeline(
        "text-classification",
        model=CLASSIFIER_DIR,
        tokenizer=CLASSIFIER_DIR,
        top_k=None,
    )


@lru_cache(maxsize=1)
def _load_ner():
    return pipeline(
        "token-classification",
        model=NER_DIR,
        tokenizer=NER_DIR,
        aggregation_strategy="simple",
    )


def classify_message(message: str) -> Dict[str, Any]:
    classifier = _load_classifier()
    normalized = normalize_text(message, collapse_digit_spaces=True)
    results = classifier(normalized, return_all_scores=True)[0]

    best = max(results, key=lambda item: item["score"])
    return {
        "label": best["label"],
        "score": best["score"],
        "all_scores": results,
    }


def extract_entities(message: str) -> Dict[str, Any]:
    ner = _load_ner()
    normalized = normalize_text(message, collapse_digit_spaces=True)
    raw_entities = ner(normalized)

    entities: Dict[str, Any] = {}
    for ent in raw_entities:
        label = ent["entity_group"]
        payload = {
            "text": ent["word"],
            "start": ent["start"],
            "end": ent["end"],
            "confidence": ent["score"],
        }
        entities.setdefault(label, []).append(payload)
    return entities


def predict(message: str) -> Dict[str, Any]:
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
