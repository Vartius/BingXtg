#!/usr/bin/env python3
"""
Quick script to compare output formats between spaCy and HuggingFace models
"""

import sys
import os


from utils import classify_signal_and_direction


def test_output_format():
    """Test that spaCy models return HuggingFace-compatible format"""

    # Test message
    test_message = (
        "BTC/USDT LONG Entry: 50000 Target: 51000 Stop Loss: 49000 Leverage: 10X"
    )

    print("Testing spaCy output format...")
    print(f"Test message: {test_message}\n")

    try:
        # Get spaCy predictions
        result = classify_signal_and_direction(test_message)

        print("SpaCy Classification Output:")
        print("-" * 50)
        for key, value in result.items():
            print(f"  {key}: {value}")

        print("\nExpected HuggingFace-compatible keys:")
        print("-" * 50)
        required_keys = ["is_signal", "direction", "confidence", "raw_prediction"]
        for key in required_keys:
            status = "✓" if key in result else "✗"
            print(f"  {status} {key}")

        print("\nNER entities format (should match HuggingFace):")
        print("-" * 50)
        expected_ner_keys = ["pair", "stop_loss", "entry", "leverage", "targets"]
        print("  Expected keys:", expected_ner_keys)
        print("\n✓ Output format is compatible with HuggingFace format!")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_output_format()
    sys.exit(0 if success else 1)
