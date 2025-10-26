#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working correctly
"""

import requests
import time
from datetime import datetime
from config import ENDPOINTS, USER_AGENTS, TELEGRAM_ENABLED, TELEGRAM_CHAT_ID
import random

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def test_market_breadth():
    """Test market breadth API"""
    print_header("Testing Market Breadth API")
    
    try:
        url = ENDPOINTS["market_breadth"]
        print_info(f"URL: {url}")
        
        response = requests.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=10
        )
        
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Market Breadth API is working!")
            
            if "breadth" in data:
                breadth = data["breadth"]
                print(f"   Total: {breadth.get('total', 'N/A')}")
                print(f"   Advancing: {breadth.get('advancing', 'N/A')}")
                print(f"   Declining: {breadth.get('declining', 'N/A')}")
            
            if "industry" in data:
                print(f"   Industries: {len(data['industry'])} sectors")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_sector_performance():
    """Test sector performance API"""
    print_header("Testing Sector Performance API")
    
    try:
        url = ENDPOINTS["sector_performance"]
        print_info(f"URL: {url}")
        
        response = requests.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=10
        )
        
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Sector Performance API is working!")
            
            if "data" in data:
                indices = data["data"]
                print(f"   Indices: {len(indices)}")
                if indices:
                    first = indices[0]
                    print(f"   Sample: {first.get('sector_index', 'N/A')} - Momentum: {first.get('momentum', 'N/A')}")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_support_resistance():
    """Test support/resistance API"""
    print_header("Testing Support & Resistance API")
    
    try:
        url = ENDPOINTS["support_resistance"]
        print_info(f"URL: {url}")
        
        test_symbols = ["NSE_RELIANCE", "NSE_TCS"]
        payload = {
            "time_frame": "day",
            "stocks": test_symbols,
            "user_broker_id": "ZMS"
        }
        
        print_info(f"Testing with: {test_symbols}")
        
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Support & Resistance API is working!")
            
            for symbol in test_symbols:
                if symbol in data:
                    sr = data[symbol]
                    print(f"\n   {symbol}:")
                    print(f"      Close: ₹{sr.get('close', 'N/A')}")
                    print(f"      R1: ₹{sr.get('r1', 'N/A')}")
                    print(f"      S1: ₹{sr.get('s1', 'N/A')}")
                else:
                    print_warning(f"{symbol} not in response")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_candlestick_data():
    """Test candlestick data API"""
    print_header("Testing Candlestick Data API")
    
    try:
        symbol = "NSE:RELIANCE"
        timeframe = "day"
        url = f"{ENDPOINTS['candlestick_data']}?stock={symbol}&timeFrame={timeframe}&user_id="
        
        print_info(f"URL: {url}")
        print_info(f"Testing: {symbol} on {timeframe}")
        
        response = requests.get(url, timeout=10)
        
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Candlestick Data API is working!")
            
            if isinstance(data, list) and len(data) > 0:
                print(f"   Candles: {len(data)}")
                latest = data[-1]
                print(f"   Latest candle:")
                print(f"      Time: {latest[0]}")
                print(f"      Open: ₹{latest[1]}")
                print(f"      High: ₹{latest[2]}")
                print(f"      Low: ₹{latest[3]}")
                print(f"      Close: ₹{latest[4]}")
                print(f"      Volume: {latest[5]}")
            else:
                print_warning("No candle data returned")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_telegram():
    """Test Telegram API"""
    print_header("Testing Telegram API")
    
    if not TELEGRAM_ENABLED:
        print_warning("Telegram is disabled in config")
        return True
    
    try:
        url = ENDPOINTS["telegram"]
        print_info(f"URL: {url[:50]}...")
        print_info(f"Chat ID: {TELEGRAM_CHAT_ID}")
        
        test_message = f"🧪 Test message from StockGenie Pro\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": test_message,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        
        print_info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                print_success("Telegram API is working!")
                print(f"   Message ID: {data.get('result', {}).get('message_id', 'N/A')}")
                return True
            else:
                print_error("Telegram API returned ok=false")
                print(f"   Response: {data}")
                return False
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_error(f"Exception occurred: {str(e)}")
        return False

def test_env_variables():
    """Test environment variables"""
    print_header("Testing Environment Variables")
    
    from config import (
        CLIENT_ID, PASSWORD, TOTP_SECRET,
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED
    )
    
    all_good = True
    
    # Check SmartAPI credentials
    if CLIENT_ID:
        print_success(f"CLIENT_ID: {CLIENT_ID}")
    else:
        print_error("CLIENT_ID is not set")
        all_good = False
    
    if PASSWORD:
        print_success(f"PASSWORD: {'*' * len(PASSWORD)}")
    else:
        print_error("PASSWORD is not set")
        all_good = False
    
    if TOTP_SECRET:
        print_success(f"TOTP_SECRET: {TOTP_SECRET[:4]}...{TOTP_SECRET[-4:]}")
    else:
        print_error("TOTP_SECRET is not set")
        all_good = False
    
    # Check Telegram
    if TELEGRAM_ENABLED:
        print_success("TELEGRAM_ENABLED: true")
        
        if TELEGRAM_BOT_TOKEN:
            print_success(f"TELEGRAM_BOT_TOKEN: {TELEGRAM_BOT_TOKEN[:10]}...")
        else:
            print_error("TELEGRAM_BOT_TOKEN is not set")
            all_good = False
        
        if TELEGRAM_CHAT_ID:
            print_success(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")
        else:
            print_error("TELEGRAM_CHAT_ID is not set")
            all_good = False
    else:
        print_warning("TELEGRAM_ENABLED: false")
    
    return all_good

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  🧪 StockGenie Pro - API Testing Suite")
    print("  Testing all external APIs and configurations")
    print("="*70)
    
    results = {}
    
    # Test environment variables first
    results["Environment Variables"] = test_env_variables()
    time.sleep(1)
    
    # Test external APIs
    results["Market Breadth"] = test_market_breadth()
    time.sleep(1)
    
    results["Sector Performance"] = test_sector_performance()
    time.sleep(1)
    
    results["Support & Resistance"] = test_support_resistance()
    time.sleep(1)
    
    results["Candlestick Data"] = test_candlestick_data()
    time.sleep(1)
    
    results["Telegram"] = test_telegram()
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, status in results.items():
        if status:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{'='*70}")
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"{'='*70}\n")
    
    if failed == 0:
        print("🎉 All tests passed! You can now run the application.")
        print("\nRun: ./run.sh")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the app.")
        print("\nCommon fixes:")
        print("  - Check internet connection")
        print("  - Verify .env file has correct credentials")
        print("  - Ensure APIs are not rate-limited")
        print("  - Check if market is open (for live data)")
        return 1

if __name__ == "__main__":
    exit(main())
