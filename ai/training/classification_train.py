#!/usr/bin/env python3

from spacy.training.example import Example
import random
import sys
import os

# Add the project root to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.training.utils import (
    load_classification_data,
    evaluate_textcat_model,
    initialize_textcat_model,
)


def train_is_signal_model(
    train_data, dev_data, output_dir="is_signal_model", n_iter=20
):
    """
    Обучает модель для классификации is_signal.
    """
    nlp = initialize_textcat_model("en", ["signal", "non_signal"])
    optimizer = nlp.begin_training()

    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = [
            train_data[x : x + 32] for x in range(0, len(train_data), 32)
        ]  # Батчи
        for batch in batches:
            examples = [
                Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch
            ]
            nlp.update(examples, sgd=optimizer, losses=losses)

        print(f"Итерация {i + 1}, потери: {losses}")

        # Валидация каждые 5
        if i % 5 == 0:
            score = evaluate_textcat_model(nlp, dev_data)
            if "cats" in score and "signal" in score["cats"]:
                print(f"Accuracy is_signal: {score['cats']['signal']['accuracy']:.3f}")
            else:
                print(f"Evaluation completed, score structure: {list(score.keys())}")

    nlp.to_disk(output_dir)
    print(f"Модель is_signal сохранена в {output_dir}")
    return nlp


def train_direction_model(
    train_data, dev_data, output_dir="direction_model", n_iter=20
):
    """
    Обучает модель для классификации direction (только на is_signal=1).
    """
    nlp = initialize_textcat_model("en", ["LONG", "SHORT", "NONE"])
    optimizer = nlp.begin_training()

    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = [train_data[x : x + 32] for x in range(0, len(train_data), 32)]
        for batch in batches:
            examples = [
                Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch
            ]
            nlp.update(examples, sgd=optimizer, losses=losses)

        print(f"Итерация {i + 1}, потери: {losses}")

        if i % 5 == 0:
            score = evaluate_textcat_model(nlp, dev_data)
            # Try to extract accuracy for direction labels
            if "cats" in score:
                accuracies = []
                for label in ["LONG", "SHORT", "NONE"]:
                    if label in score["cats"]:
                        accuracies.append(score["cats"][label].get("accuracy", 0.0))
                if accuracies:
                    best_acc = max(accuracies)
                    print(f"Лучшая accuracy direction: {best_acc:.3f}")
                else:
                    print(
                        f"Evaluation completed, score structure: {list(score.keys())}"
                    )
            else:
                print(f"Evaluation completed, score structure: {list(score.keys())}")

    nlp.to_disk(output_dir)
    print(f"Модель direction сохранена в {output_dir}")
    return nlp


if __name__ == "__main__":
    print("=== Training Classification Models ===")

    # Load data
    (is_train, is_dev), (dir_train, dir_dev) = load_classification_data("total.db")

    # Train is_signal model
    print("\n=== Training is_signal Model ===")
    nlp_is = train_is_signal_model(is_train, is_dev)

    # Train direction model
    print("\n=== Training direction Model ===")
    nlp_dir = train_direction_model(dir_train, dir_dev)

    print("\n=== Training Complete ===")
    print("Models saved to:")
    print("- is_signal_model/")
    print("- direction_model/")
