#!/usr/bin/env python3
"""Fine-tune xlm-roberta-base for 4-way trading signal classification."""

from __future__ import annotations

import argparse
from typing import Dict

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
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
        "--data-file",
        default="data_exports/classification_data.csv",
        help="CSV file with columns 'text' and 'label'",
    )
    parser.add_argument(
        "--output-dir",
        default="ai/models/signal_classifier",
        help="Directory to store the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable FP16 training (disabled by default)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for evaluation",
    )
    return parser.parse_args()


def load_and_prepare_dataset(
    data_file: str, test_split: float, seed: int
) -> DatasetDict:
    # Load the CSV file as a dataset
    dataset_dict = load_dataset("csv", data_files=data_file)
    dataset = dataset_dict["train"]  # Get the train split from the loaded dataset
    dataset = dataset.train_test_split(test_size=test_split, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def preprocess(batch: Dict[str, list]) -> Dict[str, list]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )
        tokenized["labels"] = [LABEL2ID[label] for label in batch["label"]]
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])
    return DatasetDict({"train": tokenized["train"], "eval": tokenized["test"]})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    accuracy_result = accuracy_metric.compute(predictions=preds, references=labels)
    f1_result = f1_metric.compute(predictions=preds, references=labels, average="macro")

    accuracy = accuracy_result["accuracy"] if accuracy_result else 0.0
    macro_f1 = f1_result["f1"] if f1_result else 0.0

    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenized = load_and_prepare_dataset(args.data_file, args.test_split, args.seed)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Try to use CUDA, but fall back to CPU if there are issues
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            # Test if we can actually use CUDA
            torch.cuda.set_device(0)
            test_tensor = torch.randn(2, 2).cuda()
            del test_tensor
            print("✓ CUDA is available and working - using GPU")
        except Exception as e:
            print(f"⚠️  CUDA test failed: {e}")
            print("Falling back to CPU")
            use_cuda = False
    else:
        print("CUDA not available - using CPU")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=args.use_fp16 and use_cuda,  # Only use fp16 with CUDA
        push_to_hub=False,
        no_cuda=not use_cuda,  # Disable CUDA if test failed
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
