#!/usr/bin/env python3
"""Test/evaluate xlm-roberta-base fine-tuned model for 4-way trading signal classification.

Example usage:
    # Test with default settings
    python test_classifier.py

    # Test with custom model and data
    python test_classifier.py --model-path ai/models/signal_classifier --test-data-file data_exports/classification_data.csv

    # Test using full dataset and output predictions
    python test_classifier.py --use-full-dataset --output-predictions

    # Test with larger batch size
    python test_classifier.py --batch-size 64
"""

from __future__ import annotations

import argparse
from typing import Dict

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_LIST = ["NON_SIGNAL", "SIGNAL_LONG", "SIGNAL_SHORT", "SIGNAL_NONE"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="ai/models/signal_classifier",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--test-data-file",
        default="data_exports/classification_data.csv",
        help="CSV file with columns 'text' and 'label' for testing",
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
        help="Single text to classify (instead of using dataset)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing single text to classify (instead of using dataset)",
    )
    return parser.parse_args()


def load_and_prepare_test_dataset(
    data_file: str,
    tokenizer: AutoTokenizer,
    test_split: float,
    seed: int,
    use_full_dataset: bool,
):
    # Load the CSV file as a dataset
    dataset_dict = load_dataset("csv", data_files=data_file)
    dataset = dataset_dict["train"]  # Get the train split from the loaded dataset

    if use_full_dataset:
        # Use the entire dataset for testing
        test_dataset = dataset
    else:
        # Split the dataset and use only the test portion
        dataset_splits = dataset.train_test_split(test_size=test_split, seed=seed)
        test_dataset = dataset_splits["test"]

    def preprocess(batch: Dict[str, list]) -> Dict[str, list]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )
        tokenized["labels"] = [LABEL2ID[label] for label in batch["label"]]
        return tokenized

    tokenized = test_dataset.map(
        preprocess, batched=True, remove_columns=["text", "label"]
    )
    return tokenized


def predict_single_text(text: str, model, tokenizer, device):
    """Predict the class of a single text input."""
    # Tokenize the input text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class, probabilities.cpu().numpy()[0]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    accuracy_result = accuracy_metric.compute(predictions=preds, references=labels)
    f1_result = f1_metric.compute(predictions=preds, references=labels, average="macro")
    precision_result = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )
    recall_result = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )

    accuracy = accuracy_result["accuracy"] if accuracy_result else 0.0
    macro_f1 = f1_result["f1"] if f1_result else 0.0
    macro_precision = precision_result["precision"] if precision_result else 0.0
    macro_recall = recall_result["recall"] if recall_result else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def print_detailed_results(predictions, labels, label_names):
    """Print detailed classification results including per-class metrics."""
    print("\n" + "=" * 50)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 50)

    # Get unique labels present in the data
    unique_labels = sorted(list(set(labels) | set(predictions)))
    present_label_names = [
        label_names[i] for i in unique_labels if i < len(label_names)
    ]

    print(
        f"Classes found in data: {len(unique_labels)} out of {len(label_names)} total"
    )
    print(f"Present classes: {present_label_names}")

    # Classification report with only present labels
    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=unique_labels,
            target_names=present_label_names,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    print("True\\Predicted", end="")
    for label in present_label_names:
        print(f"\t{label[:8]}", end="")
    print()

    for i, true_label in enumerate(present_label_names):
        print(f"{true_label[:12]}", end="")
        for j in range(len(present_label_names)):
            print(f"\t{cm[i][j]}", end="")
        print()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading model from: {args.model_path}")

    # Load the fine-tuned model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
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
                text_to_classify = f.read().strip()
            print(f"\nPredicting text from file: {args.text_file}")
        else:
            text_to_classify = args.text
            print("\nPredicting single text:")

        print(
            f"Text: {text_to_classify[:100]}{'...' if len(text_to_classify) > 100 else ''}"
        )
        print("-" * 50)

        predicted_class, probabilities = predict_single_text(
            text_to_classify, model, tokenizer, device
        )
        predicted_label = ID2LABEL[int(predicted_class)]

        print(f"Predicted class: {predicted_label}")
        print(f"Confidence: {probabilities[predicted_class]:.4f}")
        print("\nAll class probabilities:")
        for i, (label, prob) in enumerate(zip(LABEL_LIST, probabilities)):
            marker = "→ " if i == predicted_class else "  "
            print(f"{marker}{label:15}: {prob:.4f}")
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nRunning evaluation...")

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
    print_detailed_results(predictions, labels, LABEL_LIST)

    # Output predictions if requested
    if args.output_predictions:
        print("\n" + "=" * 30)
        print("SAMPLE PREDICTIONS")
        print("=" * 30)

        # Show first 10 predictions as examples
        for i in range(min(10, len(predictions))):
            predicted_label = ID2LABEL[int(predictions[i])]
            true_label = ID2LABEL[int(labels[i])] if labels is not None else "UNKNOWN"
            confidence = (
                float(np.max(predictions_output.predictions[i]))
                if predictions_output.predictions is not None
                else 0.0
            )
            status = "✓" if labels is not None and predictions[i] == labels[i] else "❌"

            print(
                f"Sample {i + 1:2d}: {status} Pred: {predicted_label:12} | True: {true_label:12} | Conf: {confidence:.4f}"
            )

    print(f"\n✓ Evaluation completed on {len(test_dataset)} samples")


if __name__ == "__main__":
    main()
