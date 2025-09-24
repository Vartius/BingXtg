#!/usr/bin/env python3
"""
Test script to verify the normalization functions work correctly
for various edge cases.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_training.total_ai_test import TotalAITester


def test_normalization_functions():
    """Test the normalization functions with edge cases."""
    tester = TotalAITester()

    print("=== Testing Float Normalization ===")

    # Test cases for normalize_float_value
    test_cases = [
        # Format: (input, expected_output, description)
        (".928", 0.928, "Leading dot case"),
        ("928", 928.0, "Plain number"),
        ("25X", 25.0, "Number with letter suffix"),
        ("68 000", 68000.0, "Space-separated thousands"),
        ("65,000", 65000.0, "Comma-separated thousands"),
        ("0.928", 0.928, "Normal decimal"),
        ("1,234.56", 1234.56, "Thousands with decimal"),
        ("1.234,56", 1234.56, "European decimal format"),
        ("10X", 10.0, "Leverage with X"),
        ("50x", 50.0, "Leverage with lowercase x"),
        ("", None, "Empty string"),
        ("abc", None, "Non-numeric text"),
        ("$100", 100.0, "With currency symbol"),
        ("100$", 100.0, "Currency symbol at end"),
        ("123.456.789", None, "Invalid multiple dots"),
        ("68 000.50", 68000.5, "Space thousands with decimal"),
    ]

    for input_val, expected, description in test_cases:
        result = tester.normalize_float_value(input_val)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} {description}: '{input_val}' -> {result} (expected: {expected})"
        )
        if result != expected:
            print("   ⚠️  Mismatch detected!")

    print("\n=== Testing Pair Extraction ===")

    pair_test_cases = [
        ("BTC/USDT", "BTC", "Standard pair format"),
        ("btc/usdt", "BTC", "Lowercase pair"),
        ("ETHUSDT", "ETH", "No separator, USDT suffix"),
        ("ADABTC", "ADA", "BTC pair"),
        ("DOGEETH", "DOGE", "ETH pair"),
        ("BTC", "BTC", "Just coin name"),
        ("BTC/USD", "BTC", "USD pair"),
        ("", None, "Empty string"),
        ("INVALID", "INVALID", "Unknown format"),
    ]

    for input_val, expected, description in pair_test_cases:
        result = tester.extract_coin_name(input_val)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} {description}: '{input_val}' -> {result} (expected: {expected})"
        )

    print("\n=== Testing Float Comparison ===")

    comparison_test_cases = [
        # (val1, val2, expected_result, description)
        (0.928, ".928", True, "Leading dot vs normal"),
        ("25X", 25.0, True, "Leverage format vs float"),
        ("68 000", 68000, True, "Space-separated vs normal"),
        ("65,000", 65000.0, True, "Comma thousands vs float"),
        (100.0, 100.01, False, "Just outside tolerance"),
        (100.0, 100.005, True, "Within tolerance"),
        ("invalid", 100, False, "Invalid vs valid"),
    ]

    for val1, val2, expected, description in comparison_test_cases:
        result = tester.safe_float_compare(val1, val2)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} {description}: {val1} vs {val2} -> {result} (expected: {expected})"
        )


if __name__ == "__main__":
    test_normalization_functions()
