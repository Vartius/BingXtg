#!/usr/bin/env python3
"""
Common utilities for AI training modules.
Contains shared functions for text normalization, data loading, model evaluation, etc.
"""

import sqlite3
import spacy
from spacy.training.iob_utils import offsets_to_biluo_tags
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize text by replacing common Cyrillic lookalikes with Latin equivalents.
    This is the comprehensive version that handles both classification and NER cases.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with Cyrillic characters replaced and commas converted to dots
    """
    if text is None:
        return None

    # Comprehensive Cyrillic to Latin mapping (combining both versions from files)
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
        "у": "y",
        # Additional Ukrainian/other Cyrillic characters
        "і": "i",
        "ї": "i",
        "є": "e",
    }

    normalized = text
    for cyrillic, latin in cyrillic_to_latin.items():
        normalized = normalized.replace(cyrillic, latin)

    # Replace comma with dot (for decimal numbers)
    normalized = normalized.replace(",", ".")
    return normalized


def load_database_connection(db_path: str = "total.db") -> sqlite3.Connection:
    """
    Create a database connection with row factory for easier column access.

    Args:
        db_path: Path to the SQLite database

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def load_classification_data(
    db_path: str = "total.db",
) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """
    Load data from SQLite for classification tasks.
    Returns train/dev splits for both is_signal and direction classification.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Tuple of ((is_signal_train, is_signal_dev), (direction_train, direction_dev))
    """
    conn = load_database_connection(db_path)
    cursor = conn.cursor()

    # Query: all data (signals and non-signals)
    cursor.execute("SELECT message, is_signal, direction FROM labeled")
    rows = cursor.fetchall()
    conn.close()

    # Prepare training data
    TRAIN_IS_SIGNAL = []
    TRAIN_DIRECTION = []  # Only for is_signal=1

    for row in rows:
        message, is_signal, direction = row
        text = normalize_text(message)
        if not text:
            continue

        # For is_signal: binary classification
        cats = {"signal": float(is_signal), "non_signal": float(1 - is_signal)}
        TRAIN_IS_SIGNAL.append((text, {"cats": cats}))

        # For direction: Only if is_signal=1
        if is_signal == 1:
            dir_cats = {
                "LONG": 1.0 if direction == "LONG" else 0.0,
                "SHORT": 1.0 if direction == "SHORT" else 0.0,
                "NONE": 1.0 if direction == "NONE" else 0.0,
            }
            TRAIN_DIRECTION.append((text, {"cats": dir_cats}))

    # Split 80/20
    train_is, dev_is = train_test_split(TRAIN_IS_SIGNAL, test_size=0.2, random_state=42)
    train_dir, dev_dir = train_test_split(
        TRAIN_DIRECTION, test_size=0.2, random_state=42
    )

    print(
        f"is_signal: {len(TRAIN_IS_SIGNAL)} examples. Train: {len(train_is)}, Dev: {len(dev_is)}"
    )
    print(
        f"direction: {len(TRAIN_DIRECTION)} examples. Train: {len(train_dir)}, Dev: {len(dev_dir)}"
    )

    return (train_is, dev_is), (train_dir, dev_dir)


def align_entities_with_tokens(
    text: str, entities: List[Tuple[int, int, str]], nlp
) -> List[Tuple[int, int, str]]:
    """
    Align entity positions with spaCy token boundaries to avoid alignment issues.

    Args:
        text: The input text
        entities: List of (start, end, label) tuples
        nlp: spaCy nlp object for tokenization

    Returns:
        List of aligned (start, end, label) tuples
    """
    doc = nlp.make_doc(text)
    aligned_entities = []

    for start, end, label in entities:
        # Find tokens that overlap with the entity
        overlapping_tokens = []
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)

            # Check if token overlaps with entity
            if token_start < end and token_end > start:
                overlapping_tokens.append(token)

        if not overlapping_tokens:
            continue

        # Use the span from first to last overlapping token
        first_token = overlapping_tokens[0]
        last_token = overlapping_tokens[-1]

        new_start = first_token.idx
        new_end = last_token.idx + len(last_token.text)

        # Verify the aligned entity contains the original entity text
        original_text = text[start:end]
        aligned_text = text[new_start:new_end]

        if original_text.lower() in aligned_text.lower():
            aligned_entities.append((new_start, new_end, label))

    return aligned_entities


def validate_entity_alignment(
    text: str, entities: List[Tuple[int, int, str]], nlp
) -> bool:
    """
    Validate that entities align with token boundaries using BILOU tags.

    Args:
        text: The input text
        entities: List of (start, end, label) tuples
        nlp: spaCy nlp object for tokenization

    Returns:
        True if entities are properly aligned, False otherwise
    """
    try:
        doc = nlp.make_doc(text)
        tags = offsets_to_biluo_tags(doc, entities)

        # Check if any tags are '-' (misaligned)
        misaligned = [i for i, tag in enumerate(tags) if tag == "-"]
        return len(misaligned) == 0
    except Exception:
        return False


def load_ner_training_data(
    db_path: str = "total.db",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load and prepare NER training data with proper entity alignment.
    Dynamically handles all available target entities.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of (text, annotations) tuples ready for training
    """
    nlp = spacy.blank("xx")  # Multilingual blank model
    conn = load_database_connection(db_path)
    cursor = conn.cursor()

    # Get all labeled signal records
    cursor.execute("""
        SELECT * FROM labeled 
        WHERE is_signal = 1 
        AND (pair_start IS NOT NULL OR leverage_start IS NOT NULL 
             OR entry_start IS NOT NULL OR target_1_start IS NOT NULL 
             OR stop_loss_start IS NOT NULL)
        ORDER BY id
    """)

    rows = cursor.fetchall()
    conn.close()

    train_data = []
    processed = 0
    skipped = 0

    print(f"Processing {len(rows)} labeled records...")

    # Dynamically find the maximum number of targets
    max_targets = 0
    if rows:
        sample_row = rows[0]
        for col_name in sample_row.keys():
            if col_name.startswith("target_") and col_name.endswith("_start"):
                try:
                    num = int(col_name.split("_")[1])
                    if num > max_targets:
                        max_targets = num
                except (ValueError, IndexError):
                    continue

    print(f"Detected up to {max_targets} target columns to process.")

    for row in rows:
        normalized_text = normalize_text(row["message"])
        entities = []

        # Helper function to add entities
        def add_entity(label: str, start_col: str, end_col: str):
            if row[start_col] is not None and row[end_col] is not None:
                entities.append((row[start_col], row[end_col], label))

        # Add all standard entities
        add_entity("PAIR", "pair_start", "pair_end")
        add_entity("LEVERAGE", "leverage_start", "leverage_end")
        add_entity("ENTRY", "entry_start", "entry_end")
        add_entity("STOP_LOSS", "stop_loss_start", "stop_loss_end")

        # Dynamically add all available TARGET entities
        for i in range(1, max_targets + 1):
            add_entity("TARGET", f"target_{i}_start", f"target_{i}_end")

        if not entities:
            skipped += 1
            continue

        aligned_entities = align_entities_with_tokens(normalized_text, entities, nlp)

        if not aligned_entities:
            skipped += 1
            continue

        if not validate_entity_alignment(normalized_text, aligned_entities, nlp):
            skipped += 1
            continue

        train_data.append((normalized_text, {"entities": aligned_entities}))
        processed += 1

        if processed % 1000 == 0:
            print(f"Processed {processed} records...")

    print(f"Successfully processed {processed} records, skipped {skipped}")
    return train_data


def evaluate_textcat_model(
    nlp, dev_data: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Evaluate text classification model performance.

    Args:
        nlp: Trained spaCy model with textcat component
        dev_data: Development/test data as list of (text, annotations) tuples

    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    label_counts = {}
    label_correct = {}

    for text, ann in dev_data:
        doc = nlp(text)
        predicted_cats = doc.cats
        true_cats = ann["cats"]

        # Get the label with highest predicted score
        pred_label = max(predicted_cats.keys(), key=lambda k: predicted_cats[k])
        # Get the true label (should have score 1.0)
        true_label = max(true_cats.keys(), key=lambda k: true_cats[k])

        # Track per-label statistics
        if true_label not in label_counts:
            label_counts[true_label] = 0
            label_correct[true_label] = 0

        label_counts[true_label] += 1

        if pred_label == true_label:
            correct += 1
            label_correct[true_label] += 1
        total += 1

    overall_accuracy = correct / total if total > 0 else 0.0

    # Calculate per-label accuracy
    cats_score = {}
    for label in label_counts:
        accuracy = (
            label_correct[label] / label_counts[label]
            if label_counts[label] > 0
            else 0.0
        )
        cats_score[label] = {"accuracy": accuracy}

    return {"cats": cats_score, "overall_accuracy": overall_accuracy}


def create_training_batches(
    train_data: List[Tuple[str, Dict[str, Any]]], batch_size: int = 32
) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """
    Create training batches from training data.

    Args:
        train_data: List of training examples
        batch_size: Size of each batch

    Returns:
        List of batches, where each batch is a list of training examples
    """
    return [
        train_data[x : x + batch_size] for x in range(0, len(train_data), batch_size)
    ]


def initialize_textcat_model(model_name: str = "en", labels=None):
    """
    Initialize a spaCy model with text classification component.

    Args:
        model_name: spaCy model name (e.g., "en", "xx")
        labels: List of classification labels to add

    Returns:
        Initialized spaCy model with textcat component
    """
    nlp = spacy.blank(model_name)
    textcat = nlp.add_pipe("textcat")

    if labels:
        for label in labels:
            textcat.add_label(label)  # type: ignore

    return nlp


def initialize_ner_model(model_name: str = "xx", labels=None):
    """
    Initialize a spaCy model with NER component.

    Args:
        model_name: spaCy model name (e.g., "en", "xx")
        labels: List of NER labels to add

    Returns:
        Initialized spaCy model with NER component
    """
    nlp = spacy.blank(model_name)
    nlp.add_pipe("tok2vec")
    nlp.add_pipe("ner")

    ner = nlp.get_pipe("ner")
    if labels:
        for label in labels:
            ner.add_label(label)  # type: ignore

    return nlp


def print_training_sample(
    train_data: List[Tuple[str, Dict[str, Any]]], num_samples: int = 3
):
    """
    Print sample training examples for inspection.

    Args:
        train_data: List of training examples
        num_samples: Number of samples to print
    """
    print("\nSample training examples:")
    for i in range(min(num_samples, len(train_data))):
        text, annotations = train_data[i]
        print(f"\nExample {i + 1}:")
        print(f"Text: {text[:150]}...")

        if "entities" in annotations:
            print(f"Entities: {annotations['entities']}")
            # Show the actual entity text
            for start, end, label in annotations["entities"]:
                entity_text = text[start:end]
                print(f"  '{entity_text}' -> {label}")
        elif "cats" in annotations:
            print(f"Categories: {annotations['cats']}")


def classify_signal_and_direction(
    message: str, nlp_is_signal=None, nlp_direction=None
) -> Dict[str, Any]:
    """
    Classify message for signal detection and direction prediction.
    This is the main inference function used by the classification test module.

    Args:
        message: Input message to classify
        nlp_is_signal: Trained is_signal model (will load if None)
        nlp_direction: Trained direction model (will load if None)

    Returns:
        Dictionary with classification results
    """
    # Load models if not provided
    if nlp_is_signal is None or nlp_direction is None:
        try:
            nlp_is_signal = spacy.load("is_signal_model")
            nlp_direction = spacy.load("direction_model")
        except OSError as e:
            raise RuntimeError(
                f"Could not load models: {e}. Please run classification_train.py first."
            )

    text = normalize_text(message)
    doc = nlp_is_signal(text)

    # is_signal classification
    cats = doc.cats
    is_signal_prob = cats["signal"]
    is_signal = 1 if is_signal_prob > 0.5 else 0  # Threshold 0.5

    direction = "NONE"  # Default
    direction_prob = 0.0

    # Direction classification (only if signal detected)
    if is_signal == 1:
        doc_dir = nlp_direction(text)
        dir_cats = doc_dir.cats
        direction = max(dir_cats.keys(), key=lambda k: dir_cats[k])  # Max confidence
        direction_prob = dir_cats[direction]

    return {
        "is_signal": is_signal,
        "is_signal_prob": is_signal_prob,
        "direction": direction,
        "direction_prob": direction_prob,
    }
