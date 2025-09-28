#!/usr/bin/env python3
"""Fine-tune xlm-roberta-base for token classification on trading entities."""

from __future__ import annotations

import argparse
from functools import partial
from typing import Dict, List, Tuple

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
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
        "--data-file",
        default="data_exports/ner_data.jsonl",
        help="JSONL file with keys 'text' and 'entities'.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai/models/ner_extractor",
        help="Directory to store the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate",
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
        default=0.1,
        help="Fraction of data to reserve for evaluation",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable FP16 training",
    )
    return parser.parse_args()


def tokenize_and_align_labels(
    tokenizer: AutoTokenizer, examples: Dict[str, List]
) -> Dict[str, List]:
    texts = examples["text"]
    entities = examples["entities"]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        max_length=256,
    )

    aligned_labels: List[List[int]] = []

    for offsets, sentence_entities in zip(tokenized["offset_mapping"], entities):
        labels = ["O"] * len(offsets)

        for entity in sentence_entities:
            ent_start = int(entity["start"])
            ent_end = int(entity["end"])
            ent_label = ENTITY_TO_PREFIX.get(entity["label"], None)
            if ent_label is None:
                continue

            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue  # special tokens
                if tok_end <= ent_start or tok_start >= ent_end:
                    continue
                prefix = "B" if tok_start == ent_start else "I"
                labels[idx] = f"{prefix}-{ent_label}"

        label_ids: List[int] = []
        for (tok_start, tok_end), label in zip(offsets, labels):
            if tok_start == tok_end:
                label_ids.append(-100)
            else:
                label_ids.append(LABEL2ID[label])

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    tokenized.pop("offset_mapping")
    return tokenized


def load_and_prepare_dataset(
    data_file: str, test_split: float, seed: int
) -> Tuple[DatasetDict, AutoTokenizer]:
    dataset: Dataset = load_dataset("json", data_files={"all": data_file})["all"]
    dataset = dataset.train_test_split(test_size=test_split, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenize_fn = partial(tokenize_and_align_labels, tokenizer)

    tokenized = dataset.map(tokenize_fn, batched=True)
    return DatasetDict(
        {"train": tokenized["train"], "eval": tokenized["test"]}
    ), tokenizer


def compute_metrics(eval_pred) -> Dict[str, float]:
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
        "precision": scores["overall_precision"],
        "recall": scores["overall_recall"],
        "f1": scores["overall_f1"],
        "accuracy": scores["overall_accuracy"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset, tokenizer = load_and_prepare_dataset(
        args.data_file, args.test_split, args.seed
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(ENTITY_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=args.use_fp16,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
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
