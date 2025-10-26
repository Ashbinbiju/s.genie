"""How the Trading System Works - User Guide"""

import streamlit as st

st.set_page_config(
    page_title="How It Works - Trading System Guide",
    page_icon="📖",
    layout="wide"
)

st.title("📖 How the Trading System Works")

# Introduction
st.markdown("""
This dashboard replicates a professional **Pine Script trading strategy** using **Angel One SmartAPI** 
for real-time market data and **advanced technical analysis** for signal generation.
""")

st.markdown("---")

# System Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 System Overview")
    st.markdown("""
    **Multi-Signal Strategy:**
    - ✅ 4 Buy Signals + 4 Sell Signals
    - ✅ Signal Strength (0-4 scale)
    - ✅ Quality Score (0-4 scale)
    - ✅ Risk Management (Auto SL/TP)
    
    **Technical Indicators Used:**
    - 📊 EMAs (9, 21, 50 periods)
    - 📊 RSI (14 period)
    - 📊 MACD (12, 26, 9)
    - 📊 Supertrend (10, 3.0)
    - 📊 VWAP (Volume Weighted)
    - 📊 ADX (14 period)
    - 📊 ATR (14 period)
    """)

with col2:
    st.subheader("🔄 Data Flow")
    st.markdown("""
    ```
    1. Login to Angel One API
    2. Select Stock or Wishlist
    3. Fetch Real-time Data (5/15/30 min)
    4. Calculate Technical Indicators
    5. Generate Buy/Sell Signals
    6. Display Charts & Metrics
    7. Auto-refresh (Optional)
    ```
    """)

st.markdown("---")

# Signal System
st.subheader("🚦 4-Signal Trading System")

# Buy Signals
st.markdown("### 🟢 BUY SIGNALS")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Signal 1: EMA Crossover**")
    st.caption("✅ EMA 9 crosses above EMA 21")
    st.caption("✅ RSI > 50")
    st.caption("✅ Price > VWAP")
    st.info("Strong trend confirmation")

with col2:
    st.markdown("**Signal 2: MACD Momentum**")
    st.caption("✅ MACD crosses above Signal")
    st.caption("✅ RSI < 70")
    st.caption("✅ Bullish Trend")
    st.info("Momentum confirmation")

with col3:
    st.markdown("**Signal 3: Supertrend**")
    st.caption("✅ Price crosses above Supertrend")
    st.caption("✅ RSI > 40")
    st.caption("✅ Volume Spike (1.5x)")
    st.info("Trend reversal with volume")

with col4:
    st.markdown("**Signal 4: RSI Recovery**")
    st.caption("✅ RSI was < 30 (oversold)")
    st.caption("✅ RSI crosses above 30")
    st.caption("✅ Price > EMA 21")
    st.info("Oversold bounce")

# Sell Signals
st.markdown("### 🔴 SELL SIGNALS")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Signal 1: EMA Breakdown**")
    st.caption("❌ EMA 9 crosses below EMA 21")
    st.caption("❌ RSI < 50")
    st.caption("❌ Price < VWAP")
    st.warning("Trend reversal down")

with col2:
    st.markdown("**Signal 2: MACD Weakness**")
    st.caption("❌ MACD crosses below Signal")
    st.caption("❌ RSI > 30")
    st.caption("❌ Bearish Trend")
    st.warning("Momentum loss")

with col3:
    st.markdown("**Signal 3: Supertrend Break**")
    st.caption("❌ Price crosses below Supertrend")
    st.caption("❌ RSI < 60")
    st.caption("❌ Volume Spike (1.5x)")
    st.warning("Trend breakdown with volume")

with col4:
    st.markdown("**Signal 4: RSI Exhaustion**")
    st.caption("❌ RSI was > 70 (overbought)")
    st.caption("❌ RSI crosses below 70")
    st.caption("❌ Price < EMA 21")
    st.warning("Overbought correction")

st.markdown("---")

# Quality & Strength
st.subheader("⚡ Signal Quality & Strength")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 Signal Strength (0-4)**")
    st.markdown("""
    - **4/4**: All 4 signals active - Strongest confirmation
    - **3/4**: 3 signals active - Strong setup
    - **2/4**: 2 signals active - Moderate setup
    - **1/4**: 1 signal active - Weak setup
    - **0/4**: No signals - No trade
    
    💡 *System filters trades with minimum strength requirement*
    """)

with col2:
    st.markdown("**⭐ Trade Quality (0-4)**")
    st.markdown("""
    Quality based on:
    - ✅ Trend alignment (ADX > 20)
    - ✅ EMA alignment (fast > medium)
    - ✅ Not choppy market
    - ✅ Not late entry (< 1.5 ATR from EMA)
    
    💡 *Higher quality = Higher probability*
    """)

st.markdown("---")

# Risk Management
st.subheader("🛡️ Automatic Risk Management")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Stop Loss (SL)**")
    st.markdown("""
    - 📍 **Buy**: Entry - (2.5 × ATR)
    - 📍 **Sell**: Entry + (2.5 × ATR)
    - 📍 ATR-based (adapts to volatility)
    - 📍 Protects against losses
    """)

with col2:
    st.markdown("**Take Profit (TP)**")
    st.markdown("""
    - 🎯 **Buy**: Entry + (3.0 × ATR)
    - 🎯 **Sell**: Entry - (3.0 × ATR)
    - 🎯 Risk:Reward = 1:1.2
    - 🎯 Locks in profits
    """)

with col3:
    st.markdown("**Position Sizing**")
    st.markdown("""
    - 💰 Based on risk per trade
    - 💰 ATR-adjusted quantity
    - 💰 Account balance %
    - 💰 Conservative approach
    """)

st.markdown("---")

# How to Use
st.subheader("🚀 How to Use the Dashboard")

tab1, tab2, tab3 = st.tabs(["Single Stock Mode", "Wishlist Scanner", "Advanced Features"])

with tab1:
    st.markdown("### 📊 Single Stock Analysis")
    st.markdown("""
    **Step-by-step:**
    
    1. **Login**: Click "Login to Angel One" in sidebar
    2. **Select Mode**: Choose "Single Stock" (default)
    3. **Pick Stock**: Select from dropdown (RELIANCE, TCS, etc.)
    4. **Choose Timeframe**: 5min / 15min / 30min
    5. **View Data**:
       - 📈 Interactive candlestick chart
       - 📊 6 metrics (LTP, Trend, RSI, ADX, VWAP, Volume)
       - 🎯 Active signals with entry/SL/TP prices
       - 📉 RSI and MACD subplots
    
    6. **Refresh**: Click "🔄 Refresh Now" or enable auto-refresh
    
    💡 **Best for**: Detailed analysis of one stock at a time
    """)

with tab2:
    st.markdown("### 📋 Wishlist Scanner (Multi-Stock)")
    st.markdown("""
    **Step-by-step:**
    
    1. **Switch Mode**: Select "Wishlist Scan" radio button
    2. **Add Stocks** (3 ways):
       - ⚡ **Quick Add by Sector**: 
         - Select sector (Banking, IT, Pharma, etc.)
         - Click "Add All" or individual stock buttons
       - 🔍 **Search**: 
         - Expand "Advanced Search"
         - Type stock name, click search
         - Add from results
       - ➕ **Manual**: Add from popular stocks list
    
    3. **Build Wishlist**: Add 5-50 stocks (more = longer scan time)
    4. **Scan**: Click "🔄 Scan Now"
    5. **View Results**:
       - 🟢 BUY signals listed (sorted by strength)
       - 🔴 SELL signals listed (sorted by strength)
       - Each shows: LTP, Strength, Quality, RSI, Trend
    
    6. **Take Action**: Click on stock to see detailed analysis
    
    💡 **Best for**: Finding opportunities across multiple stocks
    """)

with tab3:
    st.markdown("### ⚙️ Advanced Features")
    st.markdown("""
    **Auto Refresh:**
    - ✅ Enable in sidebar
    - ⏱️ Set interval (30-300 seconds)
    - 🔄 Dashboard updates automatically
    - 💡 Use during market hours
    
    **Debug Mode:**
    - ✅ Shows token, exchange info
    - ✅ Rate limit stats (requests/sec, requests/min)
    - ✅ Live LTP data (open, high, low, close)
    - 💡 For troubleshooting
    
    **Rate Limiting:**
    - 🚦 3 requests/sec for candle data
    - 🚦 180 requests/min limit
    - 🚦 Automatic throttling
    - 💡 Prevents API blocks
    
    **Session Management:**
    - 🔐 Secure login with TOTP
    - 🔄 Auto re-login on session expiry
    - 🚪 Clean logout
    - 💡 Handles errors gracefully
    """)

st.markdown("---")

# Tips & Best Practices
st.subheader("💡 Tips & Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**✅ DO's**")
    st.markdown("""
    - ✅ Use 30min timeframe for swing trades
    - ✅ Use 5min timeframe for scalping
    - ✅ Wait for 3-4 signal strength
    - ✅ Check quality score (≥2)
    - ✅ Respect SL and TP levels
    - ✅ Scan multiple sectors
    - ✅ Use during market hours (9:20-15:30 IST)
    - ✅ Monitor VWAP position
    - ✅ Check volume confirmation
    - ✅ Keep wishlist under 50 stocks
    """)

with col2:
    st.markdown("**❌ DON'Ts**")
    st.markdown("""
    - ❌ Don't trade with 0-1 signal strength
    - ❌ Don't ignore stop losses
    - ❌ Don't trade during first 15 minutes
    - ❌ Don't trade near market close
    - ❌ Don't add 100+ stocks to wishlist
    - ❌ Don't refresh too frequently (< 30 sec)
    - ❌ Don't ignore ADX (trend strength)
    - ❌ Don't trade choppy markets
    - ❌ Don't chase late entries
    - ❌ Don't overtrade
    """)

st.markdown("---")

# Market Hours
st.subheader("⏰ NSE Market Hours")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Pre-Open**: 9:00 - 9:15 AM")
with col2:
    st.success("**Trading**: 9:15 AM - 3:30 PM")
with col3:
    st.warning("**Post-Close**: 3:30 - 4:00 PM")

st.caption("💡 Data updates during trading hours. Historical data available anytime.")

st.markdown("---")

# Disclaimer
st.subheader("⚠️ Disclaimer")
st.warning("""
**Important Notice:**
- This system is for **educational and informational purposes only**
- Not financial advice - **Do Your Own Research (DYOR)**
- Past performance does not guarantee future results
- Trading involves risk - only trade with money you can afford to lose
- The system provides signals, but **you make the final decision**
- Always test on paper trading first
- Consult a financial advisor before real trading

**No Guarantees:** Signal accuracy varies with market conditions. Use proper risk management.
""")

st.markdown("---")

# Support
st.subheader("📞 Need Help?")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Common Issues:**
    - **No data**: Check market hours, internet connection
    - **Login failed**: Verify credentials, TOTP code
    - **Session expired**: Click logout and login again
    - **Rate limit error**: Wait 60 seconds, reduce refresh rate
    """)

with col2:
    st.markdown("""
    **Resources:**
    - 📄 README.md - Setup instructions
    - 📄 STOCK_TOKENS.md - Popular stock tokens
    - 📄 DEPLOYMENT_GUIDE.md - Cloud deployment
    - 🔧 Debug mode - Enable for diagnostics
    """)

st.success("🎉 Ready to start? Go back to the main dashboard and login to begin trading!")
