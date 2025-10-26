#!/usr/bin/env python3
import sys
import time
from datetime import datetime
from api_manager import APIManager
from market_analyzer import MarketAnalyzer
from technical_analyzer import TechnicalAnalyzer
from stock_scanner import StockScanner
from alert_manager import AlertManager

# Sample watchlist - replace with your stocks
WATCHLIST = [
    "RELIANCE-EQ",
    "TCS-EQ",
    "INFY-EQ",
    "HDFCBANK-EQ",
    "ICICIBANK-EQ",
    "SBIN-EQ",
    "BHARTIARTL-EQ",
    "ITC-EQ",
    "HINDUNILVR-EQ",
    "LT-EQ"
]

def print_banner():
    """Print application banner"""
    print("\n" + "="*60)
    print("  📈 StockGenie Pro - Smart Stock Analysis Tool")
    print("  🎯 Swing & Intraday Trading Opportunities")
    print("="*60 + "\n")

def main_menu():
    """Display main menu"""
    print("\n📋 Select Analysis Mode:")
    print("1. 📊 Market Health Check")
    print("2. 🎯 Swing Trading Scan")
    print("3. ⚡ Intraday Trading Scan")
    print("4. 📈 Full Analysis (Market + Swing + Intraday)")
    print("5. 🔍 Single Stock Analysis")
    print("6. ❌ Exit")
    
    choice = input("\n👉 Enter your choice (1-6): ").strip()
    return choice

def analyze_market(market_analyzer: MarketAnalyzer, alert_manager: AlertManager):
    """Analyze market health"""
    print("\n🔍 Analyzing market health...\n")
    
    health = market_analyzer.get_market_health()
    
    print(f"🌡️  Market Health: {health['health'].upper()}")
    print(f"📊 Overall Score: {health['score']}/100")
    print(f"\n📈 Advance/Decline:")
    print(f"   ✅ Advancing: {health['advancing']}")
    print(f"   ❌ Declining: {health['declining']}")
    print(f"   📊 Total: {health['total']}")
    print(f"   📊 A/D Ratio: {health['ad_ratio']}%")
    
    print(f"\n🏢 Sector Analysis:")
    print(f"   ✅ Positive Sectors: {health['positive_sectors']}/{health['total_sectors']}")
    
    # Get bullish sectors
    bullish_sectors = market_analyzer.get_bullish_sectors()
    if bullish_sectors:
        print(f"\n🔥 Top Bullish Sectors:")
        for sector in bullish_sectors[:5]:
            print(f"   • {sector['name']}: +{sector['change']:.2f}%")
    
    # Send alert
    alert_manager.send_market_update(health)
    
    return health

def scan_swing_trades(scanner: StockScanner, alert_manager: AlertManager):
    """Scan for swing trading opportunities"""
    print("\n🎯 Scanning for swing trading opportunities...\n")
    
    opportunities = scanner.scan_for_swing_trades(WATCHLIST)
    
    if opportunities:
        print(f"✅ Found {len(opportunities)} swing trading opportunities:\n")
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['symbol']}")
            print(f"   💰 Entry: ₹{opp['entry']:.2f}")
            print(f"   🎯 Target: ₹{opp['target']:.2f}")
            print(f"   🛑 Stop Loss: ₹{opp['stop_loss']:.2f}")
            print(f"   📊 Risk:Reward = 1:{opp['risk_reward']:.2f}")
            print(f"   ⭐ Score: {opp['score']:.0f}/100")
            print()
        
        alert_manager.send_swing_alert(opportunities)
    else:
        print("❌ No swing trading opportunities found.")
    
    return opportunities

def scan_intraday_trades(scanner: StockScanner, alert_manager: AlertManager):
    """Scan for intraday trading opportunities"""
    print("\n⚡ Scanning for intraday trading opportunities...\n")
    
    opportunities = scanner.scan_for_intraday_trades(WATCHLIST)
    
    if opportunities:
        print(f"✅ Found {len(opportunities)} intraday trading opportunities:\n")
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['symbol']}")
            print(f"   💰 Entry: ₹{opp['entry']:.2f}")
            print(f"   🎯 Target: ₹{opp['target']:.2f}")
            print(f"   🛑 SL: ₹{opp['stop_loss']:.2f}")
            print(f"   📊 R:R = 1:{opp['risk_reward']:.2f}")
            print(f"   ⭐ Score: {opp['score']:.0f}/100")
            print(f"   📦 Volume: {opp['volume_ratio']:.1f}x avg")
            print()
        
        alert_manager.send_intraday_alert(opportunities)
    else:
        print("❌ No intraday trading opportunities found.")
    
    return opportunities

def analyze_single_stock(technical_analyzer: TechnicalAnalyzer):
    """Analyze a single stock"""
    symbol = input("\n📊 Enter stock symbol (e.g., RELIANCE-EQ): ").strip().upper()
    timeframe = input("⏰ Enter timeframe (5min/15min/hour/day): ").strip().lower()
    
    if timeframe not in ["1min", "3min", "5min", "10min", "15min", "30min", "hour", "day"]:
        print("❌ Invalid timeframe!")
        return
    
    print(f"\n🔍 Analyzing {symbol} on {timeframe} timeframe...\n")
    
    analysis = technical_analyzer.analyze_stock(symbol, timeframe)
    
    if analysis:
        print(f"📊 {symbol} Analysis:")
        print(f"   💰 Price: ₹{analysis['current_price']:.2f}")
        print(f"   ⭐ Score: {analysis['score']:.0f}/100")
        
        if analysis.get('rsi'):
            print(f"   📉 RSI: {analysis['rsi']:.1f}")
        if analysis.get('adx'):
            print(f"   💪 ADX: {analysis['adx']:.1f}")
        
        print(f"   📦 Volume: {analysis['volume_ratio']:.1f}x average")
        print(f"\n🎯 Signals:")
        for key, value in analysis['signals'].items():
            print(f"   • {key.upper()}: {value}")
    else:
        print("❌ Unable to analyze stock. Please check symbol and try again.")

def main():
    """Main application loop"""
    print_banner()
    
    # Initialize components
    api_manager = APIManager()
    market_analyzer = MarketAnalyzer(api_manager)
    technical_analyzer = TechnicalAnalyzer(api_manager)
    scanner = StockScanner(api_manager)
    alert_manager = AlertManager(api_manager)
    
    while True:
        choice = main_menu()
        
        if choice == "1":
            analyze_market(market_analyzer, alert_manager)
        
        elif choice == "2":
            scan_swing_trades(scanner, alert_manager)
        
        elif choice == "3":
            scan_intraday_trades(scanner, alert_manager)
        
        elif choice == "4":
            print("\n🔄 Running full analysis...\n")
            analyze_market(market_analyzer, alert_manager)
            time.sleep(2)
            scan_swing_trades(scanner, alert_manager)
            time.sleep(2)
            scan_intraday_trades(scanner, alert_manager)
        
        elif choice == "5":
            analyze_single_stock(technical_analyzer)
        
        elif choice == "6":
            print("\n👋 Thank you for using StockGenie Pro!")
            sys.exit(0)
        
        else:
            print("\n❌ Invalid choice! Please try again.")
        
        input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting StockGenie Pro. Goodbye!")
        sys.exit(0)
