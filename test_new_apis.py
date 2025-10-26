#!/usr/bin/env python3
"""
Test script for new API endpoints:
- Shareholdings
- Financials
- Technical Analysis (Streak API)
- Complete Stock Data
"""

import requests
import json
from pprint import pprint

BASE_URL = "http://127.0.0.1:5000"

def test_shareholdings(symbol="JKPAPER-EQ"):
    """Test shareholdings API"""
    print(f"\n{'='*60}")
    print(f"📊 TESTING SHAREHOLDINGS API - {symbol}")
    print(f"{'='*60}\n")
    
    url = f"{BASE_URL}/api/shareholdings/{symbol}"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            shareholdings = data.get("data", {}).get("Shareholdings")
            if shareholdings:
                shareholdings_data = json.loads(shareholdings)
                print("\n📈 Latest Shareholding Pattern:")
                latest_quarter = list(shareholdings_data.keys())[0]
                latest_data = shareholdings_data[latest_quarter]
                print(f"Quarter: {latest_quarter}")
                print(f"  • Promoter: {latest_data.get('Promoter')}%")
                print(f"  • FII: {latest_data.get('FII')}%")
                print(f"  • DII: {latest_data.get('DII')}%")
                print(f"  • Retail < 2L: {latest_data.get('Retail < 2L')}%")
                print(f"  • Pledge: {latest_data.get('Pledge')}%")
                print(f"  • No. of Shareholders: {latest_data.get('No. of Shareholders'):,}")
        else:
            print(f"❌ Error: {data.get('error')}")
    else:
        print(f"❌ HTTP Error: {response.text}")

def test_financials(symbol="JKPAPER-EQ"):
    """Test financials API"""
    print(f"\n{'='*60}")
    print(f"💰 TESTING FINANCIALS API - {symbol}")
    print(f"{'='*60}\n")
    
    url = f"{BASE_URL}/api/financials/{symbol}"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            financials = data.get("data", {})
            
            # Summary
            summary = financials.get("Summary")
            if summary:
                summary_data = json.loads(summary)
                print("\n📊 Financial Summary (Last 3 Years):")
                for year, values in sorted(summary_data.items(), reverse=True)[:3]:
                    print(f"\n{year}:")
                    print(f"  • Revenue: ₹{values.get('Revenue')} Cr")
                    print(f"  • Net Profit: ₹{values.get('Net Profit')} Cr")
                    print(f"  • Debt: ₹{values.get('Debt')} Cr")
            
            # Financial Ratios
            ratios = financials.get("Financial Ratios")
            if ratios:
                ratios_data = json.loads(ratios)
                print("\n📈 Financial Ratios (Latest Year):")
                latest_year = sorted(ratios_data.keys(), reverse=True)[0]
                latest_ratios = ratios_data[latest_year]
                print(f"\n{latest_year}:")
                print(f"  • Operating Profit Margin: {latest_ratios.get('Operating Profit Margin (OPM)')}%")
                print(f"  • Net Profit Margin: {latest_ratios.get('Net Profit Margin')}%")
                print(f"  • EPS: ₹{latest_ratios.get('Earnings Per Share (EPS)')}")
                print(f"  • EV/EBITDA: {latest_ratios.get('EV/EBITDA')}")
        else:
            print(f"❌ Error: {data.get('error')}")
    else:
        print(f"❌ HTTP Error: {response.text}")

def test_technical_analysis(symbol="JKPAPER-EQ", timeframe="5min"):
    """Test technical analysis API"""
    print(f"\n{'='*60}")
    print(f"📉 TESTING TECHNICAL ANALYSIS API - {symbol} ({timeframe})")
    print(f"{'='*60}\n")
    
    url = f"{BASE_URL}/api/technical-analysis/{symbol}?timeframe={timeframe}"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            tech = data.get("data", {})
            print(f"\n📊 Current Price: ₹{tech.get('close')}")
            print(f"🎯 State: {tech.get('state')} (-1=Bearish, 0=Neutral, 1=Bullish)")
            
            print("\n📈 Technical Indicators:")
            print(f"  • RSI: {tech.get('rsi'):.2f} (Rec: {tech.get('rec_rsi')})")
            print(f"  • MACD: {tech.get('macd'):.4f} (Rec: {tech.get('rec_macd')})")
            print(f"  • ADX: {tech.get('adx'):.2f} (Rec: {tech.get('rec_adx')})")
            print(f"  • CCI: {tech.get('cci'):.2f} (Rec: {tech.get('rec_cci')})")
            print(f"  • Stochastic K: {tech.get('stochastic_k'):.2f}")
            
            print("\n🎲 Historical Performance:")
            print(f"  • Win Signals: {tech.get('win_signals')}")
            print(f"  • Loss Signals: {tech.get('loss_signals')}")
            print(f"  • Win Rate: {tech.get('win_pct', 0) * 100:.1f}%")
            print(f"  • Win Amount: ₹{tech.get('win_amt', 0):.2f}")
            print(f"  • Loss Amount: ₹{tech.get('loss_amt', 0):.2f}")
            
            print("\n📊 Moving Averages:")
            print(f"  • EMA5: {tech.get('ema5'):.2f}")
            print(f"  • EMA20: {tech.get('ema20'):.2f}")
            print(f"  • SMA50: {tech.get('sma50'):.2f}")
            print(f"  • SMA200: {tech.get('sma200'):.2f}")
        else:
            print(f"❌ Error: {data.get('error')}")
    else:
        print(f"❌ HTTP Error: {response.text}")

def test_complete_stock_data(symbol="JKPAPER-EQ", timeframe="day"):
    """Test complete stock data API"""
    print(f"\n{'='*60}")
    print(f"🎯 TESTING COMPLETE STOCK DATA API - {symbol}")
    print(f"{'='*60}\n")
    
    url = f"{BASE_URL}/api/stock-complete/{symbol}?timeframe={timeframe}"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            stock_data = data.get("data", {})
            
            print(f"📍 Symbol: {stock_data.get('symbol')}")
            print(f"⏰ Timestamp: {stock_data.get('timestamp')}")
            
            # Technical Analysis
            tech = stock_data.get("technical", {})
            if tech:
                print(f"\n📉 Technical Score: {tech.get('score')}/100")
                print(f"   Current Price: ₹{tech.get('current_price')}")
                print(f"   RSI: {tech.get('rsi')}")
                print(f"   MACD: {tech.get('macd')}")
                print(f"   State: {tech.get('state')}")
            
            # Support/Resistance
            sr = stock_data.get("support_resistance", {})
            if sr:
                print(f"\n🎯 Support & Resistance:")
                print(f"   Pivot Point: ₹{sr.get('pp')}")
                print(f"   R1: ₹{sr.get('r1')} | R2: ₹{sr.get('r2')} | R3: ₹{sr.get('r3')}")
                print(f"   S1: ₹{sr.get('s1')} | S2: ₹{sr.get('s2')} | S3: ₹{sr.get('s3')}")
            
            # Has Financials?
            if "financials" in stock_data:
                print(f"\n💰 Financials: ✅ Available")
            
            # Has Shareholdings?
            if "shareholdings" in stock_data:
                print(f"📊 Shareholdings: ✅ Available")
        else:
            print(f"❌ Error: {data.get('error')}")
    else:
        print(f"❌ HTTP Error: {response.text}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 TESTING NEW API ENDPOINTS")
    print("="*60)
    
    # Test with JKPAPER stock
    symbol = "JKPAPER-EQ"
    
    try:
        # Test individual endpoints
        test_shareholdings(symbol)
        test_financials(symbol)
        test_technical_analysis(symbol, timeframe="day")
        
        # Test complete data endpoint
        test_complete_stock_data(symbol, timeframe="day")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
