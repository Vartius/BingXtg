#!/usr/bin/env python3
"""
Test script to verify that all environment variables are loaded correctly
from the .env file.
"""

import os
from dotenv import load_dotenv


def test_env_config():
    """Test loading environment variables from .env file."""
    load_dotenv()

    # Test BingX API configuration
    print("=== BingX API Configuration ===")
    print(f"APIURL: {os.getenv('APIURL', 'NOT SET')}")
    print(f"APIKEY: {'SET' if os.getenv('APIKEY') else 'NOT SET'}")
    print(f"SECRETKEY: {'SET' if os.getenv('SECRETKEY') else 'NOT SET'}")

    # Test Trading configuration
    print("\n=== Trading Configuration ===")
    print(f"LEVERAGE: {os.getenv('LEVERAGE', 'NOT SET')}")
    print(f"SL: {os.getenv('SL', 'NOT SET')}")
    print(f"TP: {os.getenv('TP', 'NOT SET')}")
    print(f"START_BALANCE: {os.getenv('START_BALANCE', 'NOT SET')}")
    print(f"MIN_ORDERS_TO_HIGH: {os.getenv('MIN_ORDERS_TO_HIGH', 'NOT SET')}")
    print(f"MAX_PERCENT: {os.getenv('MAX_PERCENT', 'NOT SET')}")

    # Test Telegram API configuration
    print("\n=== Telegram API Configuration ===")
    print(f"API_ID: {'SET' if os.getenv('API_ID') else 'NOT SET'}")
    print(f"API_HASH: {'SET' if os.getenv('API_HASH') else 'NOT SET'}")

    # Test all variables are set
    required_vars = [
        "APIURL",
        "APIKEY",
        "SECRETKEY",
        "LEVERAGE",
        "SL",
        "TP",
        "START_BALANCE",
        "MIN_ORDERS_TO_HIGH",
        "MAX_PERCENT",
        "API_ID",
        "API_HASH",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("\n✅ All required environment variables are set!")
        return True


if __name__ == "__main__":
    test_env_config()
