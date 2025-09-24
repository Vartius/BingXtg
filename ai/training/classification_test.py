import spacy
import sys
import os

# Add the project root to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_training.utils import classify_signal_and_direction


if __name__ == "__main__":
    print("\n=== Testing Trained Classification Models ===")

    # Load the trained models
    try:
        nlp_is = spacy.load("is_signal_model")
        nlp_dir = spacy.load("direction_model")
        print("Models loaded successfully!")
    except OSError as e:
        print(f"Error loading models: {e}")
        print("Please run classification_train.py first to train the models.")
        exit(1)

    # Test with signal message
    signal_message = """📤 LONG - SCALP V 4.0\nBNB/USDT  Leverage 20x  \n\n💹 Entry: 292.3 - 287.6232\n\n🧿 Targets: 294.3461 - 296.6845 - 300.4844 - 303.992 - 306.915 - 311.2995 - 315.684 - 321.53 \n\n❌ StopLoss: 279.1465"""
    print("\n=== Testing Signal Message ===")
    print(f"Message: {signal_message[:100]}...")
    result = classify_signal_and_direction(signal_message)
    print(f"Result: {result}")

    # Test with non-signal message
    non_signal_message = "Просто приветствие без торгового контента."
    print("\n=== Testing Non-Signal Message ===")
    print(f"Message: {non_signal_message}")
    result = classify_signal_and_direction(non_signal_message)
    print(f"Result: {result}")

    print("\n=== Testing Complete ===")


# Load models globally if run as module
try:
    nlp_is = spacy.load("is_signal_model")
    nlp_dir = spacy.load("direction_model")
except OSError:
    # Models not found, will be handled in main or when functions are called
    nlp_is = None
    nlp_dir = None
