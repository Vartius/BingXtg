from spacy.training.example import Example
import random

from utils import (
    load_classification_data,
    evaluate_textcat_model,
    initialize_textcat_model,
)

import os
from dotenv import load_dotenv

load_dotenv()
IS_SIGNAL_MODEL_PATH = os.getenv("IS_SIGNAL_MODEL_PATH", "ai/models/is_signal_model")
DIRECTION_MODEL_PATH = os.getenv("DIRECTION_MODEL_PATH", "ai/models/direction_model")


def train_is_signal_model(
    train_data, dev_data, output_dir=IS_SIGNAL_MODEL_PATH, n_iter=20
):
    """
    Trains the model for is_signal classification.
    """
    nlp = initialize_textcat_model("en", ["signal", "non_signal"])
    optimizer = nlp.begin_training()

    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = [
            train_data[x : x + 32] for x in range(0, len(train_data), 32)
        ]  # Batches
        for batch in batches:
            examples = [
                Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch
            ]
            nlp.update(examples, sgd=optimizer, losses=losses)

        print(f"Iteration {i + 1}, losses: {losses}")

        # Validation every 5 iterations
        if i % 5 == 0:
            score = evaluate_textcat_model(nlp, dev_data)
            if "cats" in score and "signal" in score["cats"]:
                print(f"Accuracy is_signal: {score['cats']['signal']['accuracy']:.3f}")
            else:
                print(f"Evaluation completed, score structure: {list(score.keys())}")

    nlp.to_disk(output_dir)
    print(f"is_signal model saved to {output_dir}")
    return nlp


def train_direction_model(
    train_data, dev_data, output_dir=DIRECTION_MODEL_PATH, n_iter=20
):
    """
    Trains the model for direction classification (only on is_signal=1).
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

        print(f"Iteration {i + 1}, losses: {losses}")

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
                    print(f"Best direction accuracy: {best_acc:.3f}")
                else:
                    print(
                        f"Evaluation completed, score structure: {list(score.keys())}"
                    )
            else:
                print(f"Evaluation completed, score structure: {list(score.keys())}")

    nlp.to_disk(output_dir)
    print(f"direction model saved to {output_dir}")
    return nlp


if __name__ == "__main__":
    print("=== Training Classification Models ===")

    # Load data
    (is_train, is_dev), (dir_train, dir_dev) = load_classification_data("total.db")

    # Train is_signal model
    print("\n=== Training is_signal Model ===")
    nlp_is = train_is_signal_model(is_train, is_dev, n_iter=10)

    # Train direction model
    print("\n=== Training direction Model ===")
    nlp_dir = train_direction_model(dir_train, dir_dev, n_iter=10)

    print("\n=== Training Complete ===")
    print("Models saved to:")
    print(f"- {IS_SIGNAL_MODEL_PATH}/")
    print(f"- {DIRECTION_MODEL_PATH}/")
