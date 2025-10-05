#!/usr/bin/env python3
"""Test/evaluate xlm-roberta-base fine-tuned model for NER (Named Entity Recognition) on trading entities.

Note: This file may show type checking warnings due to the dynamic nature of the transformers
and datasets libraries. These are expected and do not affect functionality.

Example usage:
    # Test with default settings
    python test_ner.py

    # Test with custom model and data
    python test_ner.py --model-path ai/models/ner_extractor --test-data-file data_exports/ner_data.jsonl

    # Test using full dataset and output predictions
    python test_ner.py --use-full-dataset --output-predictions

    # Test with larger batch size
    python te        # Show sample predictions
        num_examples = min(args.show_examples, len(predictions))
        for i in range(num_examples):
            print(f"\\nSample {i + 1}:")
            try:
                if original_test is not None and i < len(original_test):
                    text_sample = str(original_test[i]['text'])
                    print(f"Text: {text_sample[:100]}{'...' if len(text_sample) > 100 else ''}")
                else:
                    print("Text: [Text not available]")
            except (IndexError, KeyError, TypeError):
                print("Text: [Text access error]")

            # Extract entities from predictions for this sample
            pred_entities = []
            true_entities = []

            pred_seq = predictions[i]
            label_seq = labels[i] if labels is not None else []batch-size 64

    # Test single text
    python test_ner.py --text "⭐ #BTC/USDT #LONG\nLeverage: 25x\n✅ Entry: 50000\n🎯 Target: 52000\n❌ Stop Loss: 48000"
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

ENTITY_LABELS = [
    "O",
    "B-PAIR",
    "I-PAIR",
    "B-LEVERAGE",
    "I-LEVERAGE",
    "B-ENTRY",
    "I-ENTRY",
    "B-STOP_LOSS",
    "I-STOP_LOSS",
    "B-TARGET",
    "I-TARGET",
]
LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
ENTITY_TO_PREFIX = {
    "PAIR": "PAIR",
    "LEVERAGE": "LEVERAGE",
    "ENTRY": "ENTRY",
    "STOP_LOSS": "STOP_LOSS",
    "TARGET": "TARGET",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="ai/models/ner_extractor",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--test-data-file",
        default="data_exports/ner_data.jsonl",
        help="JSONL file with keys 'text' and 'entities' for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (if splitting the data)",
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use the full dataset for testing instead of splitting",
    )
    parser.add_argument(
        "--output-predictions",
        action="store_true",
        help="Output detailed predictions for each sample",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to extract entities from (instead of using dataset)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing single text to extract entities from (instead of using dataset)",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of prediction examples to show when using --output-predictions",
    )
    return parser.parse_args()


def tokenize_and_align_labels(
    tokenizer: AutoTokenizer, examples: Dict[str, List]
) -> Dict[str, List]:
    """Tokenize inputs and align NER labels with tokenized inputs."""
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
        is_split_into_words=False,
    )

    labels = []
    for i, entities in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the B- or O label
                char_start = (
                    tokenized_inputs["offset_mapping"][i][len(label_ids)][0]
                    if "offset_mapping" in tokenized_inputs
                    else None
                )
                char_end = (
                    tokenized_inputs["offset_mapping"][i][len(label_ids)][1]
                    if "offset_mapping" in tokenized_inputs
                    else None
                )

                # Find matching entity
                found_label = "O"
                if char_start is not None and char_end is not None:
                    for entity in entities:
                        entity_start = entity["start"]
                        entity_end = entity["end"]
                        entity_label = entity["label"]

                        # Check if token overlaps with entity
                        if char_start < entity_end and char_end > entity_start:
                            if char_start == entity_start:
                                found_label = f"B-{entity_label}"
                            else:
                                found_label = f"I-{entity_label}"
                            break

                label_ids.append(LABEL2ID[found_label])
            else:
                # Subsequent tokens of the same word get I- label
                if label_ids and label_ids[-1] != -100:
                    prev_label = ID2LABEL[label_ids[-1]]
                    if prev_label.startswith("B-"):
                        label_ids.append(LABEL2ID[prev_label.replace("B-", "I-")])
                    elif prev_label.startswith("I-"):
                        label_ids.append(label_ids[-1])
                    else:
                        label_ids.append(LABEL2ID["O"])
                else:
                    label_ids.append(LABEL2ID["O"])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_prepare_test_dataset(
    data_file: str,
    tokenizer: AutoTokenizer,
    test_split: float,
    seed: int,
    use_full_dataset: bool,
):
    """Load and prepare the test dataset for NER evaluation."""
    # Load the JSONL file as a dataset
    dataset_dict = load_dataset("json", data_files=data_file)
    dataset = dataset_dict["train"]  # Get the train split from the loaded dataset

    if use_full_dataset:
        # Use the entire dataset for testing
        test_dataset = dataset
    else:
        # Split the dataset and use only the test portion
        dataset_splits = dataset.train_test_split(test_size=test_split, seed=seed)
        test_dataset = dataset_splits["test"]

    # Add offset mapping for proper alignment
    def add_offset_mapping(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
            return_offsets_mapping=True,
        )
        batch.update(tokenized)
        return batch

    # First add offset mappings
    test_dataset = test_dataset.map(add_offset_mapping, batched=True)

    # Then tokenize and align labels
    tokenize_fn = partial(tokenize_and_align_labels, tokenizer)
    tokenized = test_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text", "entities"]
    )

    return tokenized


def predict_single_text(text: str, model, tokenizer, device):
    """Extract entities from a single text input."""
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
        return_offsets_mapping=True,
    )

    # Move to device
    offset_mapping = inputs.pop(
        "offset_mapping"
    )  # Remove offset mapping before sending to model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Convert predictions to entities
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predictions = predictions[0].cpu().numpy()
    offset_mapping = offset_mapping[0].numpy()

    entities = []
    current_entity = None

    for i, (token, pred_id, (start, end)) in enumerate(
        zip(tokens, predictions, offset_mapping)
    ):
        if pred_id == -100:  # Skip special tokens
            continue

        pred_label = ID2LABEL[pred_id]

        if pred_label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif pred_label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = pred_label[2:]
            current_entity = {
                "label": entity_type,
                "start": start,
                "end": end,
                "text": text[start:end],
                "confidence": float(torch.softmax(logits[0][i], dim=-1).max()),
            }
        elif pred_label.startswith("I-") and current_entity:
            entity_type = pred_label[2:]
            if entity_type == current_entity["label"]:
                current_entity["end"] = end
                current_entity["text"] = text[current_entity["start"] : end]

    if current_entity:
        entities.append(current_entity)

    return entities


def compute_metrics(eval_pred):
    """Compute NER evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_predictions: List[List[str]] = []
    true_labels: List[List[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        filtered_preds: List[str] = []
        filtered_labels: List[str] = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            filtered_preds.append(ID2LABEL[pred_id])
            filtered_labels.append(ID2LABEL[label_id])

        true_predictions.append(filtered_preds)
        true_labels.append(filtered_labels)

    metric = evaluate.load("seqeval")
    scores = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": scores.get("overall_precision", 0.0) if scores else 0.0,
        "recall": scores.get("overall_recall", 0.0) if scores else 0.0,
        "f1": scores.get("overall_f1", 0.0) if scores else 0.0,
        "accuracy": scores.get("overall_accuracy", 0.0) if scores else 0.0,
    }


def print_detailed_results(predictions, labels, texts=None):
    """Print detailed NER results including per-entity metrics."""
    print("\n" + "=" * 60)
    print("DETAILED NER EVALUATION RESULTS")
    print("=" * 60)

    # Convert predictions and labels to entity format for seqeval
    true_predictions: List[List[str]] = []
    true_labels: List[List[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        filtered_preds: List[str] = []
        filtered_labels: List[str] = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            filtered_preds.append(ID2LABEL[pred_id])
            filtered_labels.append(ID2LABEL[label_id])

        true_predictions.append(filtered_preds)
        true_labels.append(filtered_labels)

    # Calculate per-entity metrics
    metric = evaluate.load("seqeval")
    scores = metric.compute(predictions=true_predictions, references=true_labels)

    print("Overall Metrics:")
    if scores:
        print(f"  Precision: {scores.get('overall_precision', 0.0):.4f}")
        print(f"  Recall:    {scores.get('overall_recall', 0.0):.4f}")
        print(f"  F1-Score:  {scores.get('overall_f1', 0.0):.4f}")
        print(f"  Accuracy:  {scores.get('overall_accuracy', 0.0):.4f}")

        if "PAIR" in scores:
            print("\nPer-Entity Metrics:")
            for entity_type in ["PAIR", "LEVERAGE", "ENTRY", "STOP_LOSS", "TARGET"]:
                if entity_type in scores:
                    entity_scores = scores[entity_type]
                    print(
                        f"  {entity_type:10}: P={entity_scores.get('precision', 0.0):.3f} R={entity_scores.get('recall', 0.0):.3f} F1={entity_scores.get('f1-score', 0.0):.3f} Support={entity_scores.get('number', 0)}"
                    )
    else:
        print("  No metrics available")

    # Flatten for token-level classification report
    flat_preds = [label for seq in true_predictions for label in seq]
    flat_labels = [label for seq in true_labels for label in seq]

    print("\nToken-Level Classification Report:")
    unique_labels = sorted(list(set(flat_labels + flat_preds)))
    print(
        classification_report(
            flat_labels,
            flat_preds,
            labels=unique_labels,
            target_names=unique_labels,
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading NER model from: {args.model_path}")

    # Load the fine-tuned model and tokenizer
    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("✓ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model path is correct and the model has been trained.")
        return

    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device = "cpu"
    if use_cuda:
        try:
            torch.cuda.set_device(0)
            test_tensor = torch.randn(2, 2).cuda()
            del test_tensor
            device = "cuda"
            print("✓ CUDA is available and working - using GPU")
        except Exception as e:
            print(f"⚠️  CUDA test failed: {e}")
            print("Falling back to CPU")
            use_cuda = False
    else:
        print("CUDA not available - using CPU")

    # Move model to device
    model.to(device)

    # Handle single text prediction
    if args.text or args.text_file:
        if args.text_file:
            with open(args.text_file, "r", encoding="utf-8") as f:
                text_to_analyze = f.read().strip()
            print(f"\nExtracting entities from file: {args.text_file}")
        else:
            text_to_analyze = args.text
            print("\nExtracting entities from single text:")

        print(f"Text: {text_to_analyze}")
        print("-" * 50)

        entities = predict_single_text(text_to_analyze, model, tokenizer, device)

        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(
                    f"  {entity['label']:10}: '{entity['text']}' (confidence: {entity['confidence']:.4f})"
                )
        else:
            print("No entities found.")
        return

    # Load and prepare test dataset
    print(f"Loading test data from: {args.test_data_file}")
    test_dataset = load_and_prepare_test_dataset(
        args.test_data_file,
        tokenizer,
        args.test_split,
        args.seed,
        args.use_full_dataset,
    )
    print(f"✓ Test dataset prepared with {len(test_dataset)} samples")

    # Set up evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=False,
        no_cuda=not use_cuda,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nRunning NER evaluation...")

    # Get predictions
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    # Print basic metrics
    metrics = compute_metrics((predictions_output.predictions, labels))
    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:15}: {metric_value:.4f}")

    # Print detailed results
    print_detailed_results(predictions, labels)

    # Output predictions if requested
    if args.output_predictions:
        print("\n" + "=" * 50)
        print("SAMPLE PREDICTIONS")
        print("=" * 50)

        # Load original dataset to get text samples
        try:
            original_dataset_dict = load_dataset("json", data_files=args.test_data_file)
            original_dataset = original_dataset_dict["train"]

            if not args.use_full_dataset:
                original_splits = original_dataset.train_test_split(
                    test_size=args.test_split, seed=args.seed
                )
                original_test = original_splits["test"]
            else:
                original_test = original_dataset
        except Exception as e:
            print(f"Warning: Could not load original dataset for text display: {e}")
            original_test = None

        # Show sample predictions
        num_examples = min(args.show_examples, len(predictions))
        for i in range(num_examples):
            print(f"\nSample {i + 1}:")
            try:
                if original_test is not None and i < len(original_test):
                    text_sample = str(original_test[i]["text"])
                    print(
                        f"Text: {text_sample[:100]}{'...' if len(text_sample) > 100 else ''}"
                    )
                else:
                    print("Text: [Text not available]")
            except (IndexError, KeyError, TypeError):
                print("Text: [Text access error]")

            # Extract entities from predictions for this sample
            pred_entities = []
            true_entities = []

            pred_seq = predictions[i]
            label_seq = labels[i] if labels is not None else []

            # Convert to entity format
            current_pred_entity = None
            current_true_entity = None

            for j, (pred_id, true_id) in enumerate(zip(pred_seq, label_seq)):
                if true_id == -100:
                    continue

                pred_label = ID2LABEL[pred_id]
                true_label = ID2LABEL[true_id]

                # Process predicted entities
                if pred_label.startswith("B-"):
                    if current_pred_entity:
                        pred_entities.append(current_pred_entity)
                    current_pred_entity = {"type": pred_label[2:], "start": j}
                elif pred_label == "O" and current_pred_entity:
                    current_pred_entity["end"] = j
                    pred_entities.append(current_pred_entity)
                    current_pred_entity = None

                # Process true entities
                if true_label.startswith("B-"):
                    if current_true_entity:
                        true_entities.append(current_true_entity)
                    current_true_entity = {"type": true_label[2:], "start": j}
                elif true_label == "O" and current_true_entity:
                    current_true_entity["end"] = j
                    true_entities.append(current_true_entity)
                    current_true_entity = None

            # Close any remaining entities
            if current_pred_entity:
                current_pred_entity["end"] = len(pred_seq)
                pred_entities.append(current_pred_entity)
            if current_true_entity:
                current_true_entity["end"] = len(label_seq)
                true_entities.append(current_true_entity)

            print(
                f"  True entities: {len(true_entities)} - {[e['type'] for e in true_entities]}"
            )
            print(
                f"  Pred entities: {len(pred_entities)} - {[e['type'] for e in pred_entities]}"
            )

            # Calculate accuracy for this sample
            correct = sum(
                1 for p, t in zip(pred_seq, label_seq) if t != -100 and p == t
            )
            total = sum(1 for t in label_seq if t != -100)
            accuracy = correct / total if total > 0 else 0
            print(f"  Token accuracy: {accuracy:.4f} ({correct}/{total})")

    print(f"\n✓ NER evaluation completed on {len(test_dataset)} samples")


if __name__ == "__main__":
    main()
