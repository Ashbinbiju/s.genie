# Paper Trading Setup Guide

## 📄 Automated Paper Trading System

This system provides **fully automated paper trading** with **Supabase database** integration and **Telegram notifications**.

## 🚀 Features

- ✅ **Automated Trade Execution** - Signals trigger paper trades automatically
- ✅ **Supabase Database** - All trades stored in cloud database
- ✅ **Telegram Alerts** - Real-time notifications for every trade
- ✅ **Risk Management** - Position sizing based on 2% risk per trade
- ✅ **Stop Loss / Take Profit** - Automatic SL/TP monitoring
- ✅ **Performance Analytics** - Win rate, P&L, profit factor tracking
- ✅ **Trade History** - Complete audit trail with export to CSV

## 📋 Setup Instructions

### 1. Create Supabase Project

1. Go to https://supabase.com and create a free account
2. Click "New Project"
3. Name your project (e.g., "trading-system")
4. Set a database password (save it!)
5. Choose a region close to you
6. Wait 2-3 minutes for setup

### 2. Create Database Tables

1. In Supabase dashboard, click "SQL Editor"
2. Click "New Query"
3. Copy and paste the contents of `supabase_schema.sql`
4. Click "Run" to create tables and indexes

### 3. Get API Credentials

1. In Supabase dashboard, go to "Settings" → "API"
2. Copy **Project URL** (looks like: https://xxxxx.supabase.co)
3. Copy **anon/public key** (looks like: eyJhbG...)

### 4. Add to Streamlit Secrets

#### For Streamlit Cloud:
1. Go to your app dashboard on Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Add:
```toml
[supabase]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key-here"

[telegram]
TELEGRAM_ENABLED = "true"
TELEGRAM_CHAT_ID = "-1002411670969"
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
```

#### For Local Development:
1. Edit `config_secure.py`
2. Add your credentials in the except block

### 5. Enable Paper Trading

1. Go to the **"📄 Paper Trading"** page in sidebar
2. Go to "⚙️ Settings" tab
3. Check "Enable Automated Trading"
4. Set your criteria:
   - Minimum Signal Strength (recommended: 3/4)
   - Minimum Signal Quality (recommended: 2/4)
   - Risk Per Trade (recommended: 2%)

## 📊 How It Works

### Signal Detection
The system monitors all stocks in your watchlist and detects:
- **BUY Signals**: 4 different signal types (EMA, MACD, Supertrend, RSI)
- **SELL Signals**: 4 different exit signals

### Automatic Execution
When a signal meets your criteria:
1. ✅ System calculates position size (2% risk)
2. ✅ Creates paper trade in database
3. ✅ Sends Telegram alert with details
4. ✅ Monitors for Stop Loss / Take Profit

### Position Management
- **Stop Loss Hit**: Position closed automatically, P&L recorded
- **Take Profit Hit**: Position closed automatically, profit locked
- **Opposite Signal**: Exit existing position if criteria met
- **Manual Close**: Close anytime from dashboard

## 📱 Telegram Alerts

You'll receive alerts for:
- 🟢 **BUY Signal Executed** - Entry price, SL, TP, quantity, strength, quality
- 🔴 **Position CLOSED** - Exit price, P&L, P&L%, reason
- 🎯 **Take Profit Hit** - Target achieved
- 🛑 **Stop Loss Hit** - Risk limited

Example alert:
```
🟢 BUY Signal Executed (Paper Trade)

📊 Symbol: RELIANCE
💰 Entry: ₹2,450.00
🛑 Stop Loss: ₹2,400.00
🎯 Take Profit: ₹2,550.00
📦 Quantity: 8
💵 Cost: ₹19,600.00

⚡ Strength: 4/4
⭐ Quality: 3/4
📈 RSI: 55.2
🎯 Trend: Bullish

Risk:Reward = 1:2.0
```

## 📈 Dashboard Features

### Account Summary
- Initial capital tracking
- Available capital (updates with trades)
- Total P&L with percentage
- Win rate and trade statistics

### Open Positions
- Real-time view of all open trades
- Entry price, SL, TP levels
- Signal strength and quality
- Manual close option

### Trade History
- Complete trade log
- Filter by symbol
- P&L tracking
- Export to CSV

## 🎯 Trading Strategy

### Entry Criteria (BUY)
- Signal Strength ≥ 3/4
- Signal Quality ≥ 2/4
- Any of 4 buy signals active
- Sufficient capital available

### Exit Criteria
- Stop Loss hit (protects capital)
- Take Profit hit (locks profit)
- Opposite signal (quality ≥ 2)
- Manual close

### Position Sizing
```
Risk Amount = Capital × (Risk% / 100)
Price Risk = |Entry - Stop Loss|
Quantity = Risk Amount / Price Risk
```

Example:
- Capital: ₹100,000
- Risk%: 2%
- Risk Amount: ₹2,000
- Entry: ₹500, SL: ₹480
- Price Risk: ₹20
- Quantity: 100 shares

## 📊 Database Schema

```sql
paper_trades:
  - id (primary key)
  - symbol, token, exchange
  - signal_type (BUY/SELL)
  - entry_price, exit_price
  - stop_loss, take_profit
  - quantity
  - signal_strength, signal_quality
  - rsi, trend
  - status (OPEN/CLOSED)
  - profit_loss, profit_loss_percent
  - entry_time, exit_time
```

## 🔒 Security Notes

1. **Never commit credentials** - Use Streamlit secrets
2. **Supabase RLS** - Enable Row Level Security if needed
3. **API Keys** - Keep Telegram bot token secret
4. **Database backups** - Supabase auto-backs up daily

## 🐛 Troubleshooting

### Database Not Connected
- Check SUPABASE_URL and SUPABASE_KEY are correct
- Verify Supabase project is active
- Check network connectivity

### No Trades Executing
- Verify "Enable Automated Trading" is checked
- Check signal strength/quality criteria
- Ensure sufficient capital available
- Check if stock has existing open position

### Telegram Not Working
- Verify TELEGRAM_ENABLED = "true"
- Check bot token is correct
- Confirm chat ID is correct
- Test with "Test Telegram Alert" button

## 📚 Further Reading

- [Supabase Documentation](https://supabase.com/docs)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Pine Script Strategy](./README.md)

## ⚠️ Disclaimer

**This is paper trading (simulation only)**. No real money is involved. Use this to test your strategy before going live. Always practice proper risk management.

## 🆘 Support

For issues or questions:
1. Check the logs in Streamlit Cloud
2. Review Supabase database logs
3. Test Telegram connection
4. Check signal criteria settings

---

**Happy Paper Trading! 📄📈**
