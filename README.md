# 📈 Advanced Intraday Trading System - Streamlit Dashboard

**Your TradingView Pine Script strategy - Now LIVE with Angel One SmartAPI!**

## ✅ What's Working

- ✅ **Dashboard Running** at http://localhost:8501
- ✅ **Angel One Login** - Successfully connected
- ✅ **Live Data** - Real-time from Angel One
- ✅ **All Indicators** - EMA, RSI, MACD, VWAP, ADX, Volume
- ✅ **Signal Generation** - Buy/Sell signals matching Pine Script
- ✅ **Interactive Charts** - Candlesticks with indicator overlays

## 🚀 Quick Start

Your dashboard is **ALREADY RUNNING**! Open your browser to:

👉 **http://localhost:8501**

## 📊 Dashboard Features

### 1. Top Metrics Bar
- 💰 Live price with % change
- 📈 Trend (Bullish/Bearish)
- 🔴 RSI value
- 🟢 ADX strength
- 🔵 VWAP position
- 📊 Volume status

### 2. Signal Panel
- 🟢 **BUY SIGNAL** - Shows entry, SL, TP levels
- 🔴 **SELL SIGNAL** - Shows entry, SL, TP levels
- Signal strength (0-2)
- Quality score (0-4)

### 3. Interactive Chart
- Candlestick chart
- EMA lines (9, 21, 50)
- VWAP line
- Buy/Sell markers on chart
- RSI subplot
- MACD histogram
- Volume bars

### 4. Auto-Refresh
- Enable in sidebar
- Updates every 30-300 seconds
- Real-time monitoring

## ⚙️ Configuration

All parameters match your Pine Script in `config.py`:

```python
TRADING_CONFIG = {
    'ema_fast': 9,
    'ema_medium': 21,
    'ema_slow': 50,
    'rsi_length': 14,
    'atr_multiplier_sl': 2.5,  # Widened from 2.0
    'atr_multiplier_tp': 3.0,
    'min_signal_strength': 1,
    'min_trade_quality': 2,
}
```

## 📈 Supported Stocks

All 7 from your backtest:
- **TATASTEEL** ⭐⭐⭐⭐⭐ (Best performer)
- **RELIANCE** ⭐⭐⭐⭐⭐
- **TCS** ⭐⭐⭐⭐⭐
- **SUNPHARMA** ⭐⭐⭐⭐
- **ITC** ⭐⭐⭐⭐
- **ICICIBANK** ⭐⭐⭐
- **TATAMOTORS** (Need parameter tuning)

## 🔧 Usage

1. **Select Stock** - Choose from dropdown (sidebar)
2. **Choose Timeframe** - 5min, 15min, or 30min
3. **Monitor Signals** - Watch for green/red alerts
4. **View Chart** - Interactive chart with all indicators
5. **Enable Auto-Refresh** - For continuous monitoring

## 🎯 How Signals Work

### Buy Signal Requirements:
- ✅ EMA fast > medium (bullish alignment)
- ✅ MACD crossover OR RSI conditions
- ✅ Price above VWAP (strong position)
- ✅ Quality score ≥ 2/4
- ✅ Min signal strength ≥ 1

### Sell Signal Requirements:
- ✅ EMA fast < medium (bearish alignment)
- ✅ MACD crossunder OR RSI conditions
- ✅ Price below VWAP (weak position)
- ✅ Quality score ≥ 2/4
- ✅ Min signal strength ≥ 1

## 📝 Files Structure

```
webtest/
├── app.py                  # Main Streamlit dashboard
├── smartapi_client.py      # Angel One API client
├── indicators.py           # Technical indicators
├── config.py               # Configuration & credentials
└── requirements.txt        # Python dependencies
```

## 🔐 Your Credentials (Already Configured)

```python
CLIENT_ID = "AAAG399109"
PASSWORD = "1503"
TOTP_SECRET = "OLRQ3CYBLPN2XWQPHLKMB7WEKI"
TRADING_API_KEY = "ruseeaBq"
```

## ⚠️ Important Notes

1. **Paper Trading First** - Test before live trading
2. **Market Hours** - Best during 9:20 AM - 3:15 PM IST
3. **Data Lag** - Angel One has ~1-5 sec delay
4. **Stop Loss** - Always set to 2.5x ATR
5. **Capital Management** - Never risk more than 2% per trade

## 🐛 Troubleshooting

### Dashboard won't load?
```powershell
cd e:\abcd\webtest
streamlit run app.py
```

### Login fails?
- Check TOTP secret in `config.py`
- Verify credentials are correct
- Try restarting dashboard

### No data showing?
- Check if market is open
- Verify symbol tokens in `config.py`
- Try different timeframe

### Signals don't match TradingView?
- Minor differences normal (data source)
- Logic is 100% identical
- Check timeframe matches

## 🎨 Customization

### Change Colors
Edit `app.py` chart section (lines 50-150)

### Add More Stocks
Edit `config.py` STOCK_SYMBOLS:
```python
'NEWSTOCK': {'token': 'TOKEN_ID', 'exchange': 'NSE', 'lot_size': 1}
```

### Modify Parameters
Edit `config.py` TRADING_CONFIG

### Add Alerts
Use Streamlit's `st.success()` / `st.error()`

## 📞 Support

### Issues with:
- **Angel One API** → Contact Angel One support
- **Strategy Logic** → Matches Pine Script exactly
- **Dashboard** → Check terminal logs

## 🎉 You're All Set!

Your dashboard is running at **http://localhost:8501**

Happy Trading! 📈💰
