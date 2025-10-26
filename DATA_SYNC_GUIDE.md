# 🔄 Data Sync Guide - Dashboard vs TradingView

## Why Data Might Differ

### 1. **Timeframe Mismatch** ⏰
- **Dashboard:** Check selected timeframe (5min/15min/30min)
- **TradingView:** Ensure same timeframe is selected
- **Fix:** Match both to 30min

### 2. **VWAP Calculation** 📊
- **Dashboard:** Daily VWAP (resets at market open 9:15 AM IST)
- **TradingView:** Check if using daily or session VWAP
- **Fix:** Both should use daily VWAP for intraday

### 3. **Data Source Delay** ⏱️
- **Angel One API:** 1-5 second delay is normal
- **TradingView:** Live data with minimal delay
- **Fix:** Refresh both to sync to latest candle

### 4. **Candle Close Time** 🕐
- **Dashboard:** Shows data at candle close
- **TradingView:** Updates during candle formation
- **Fix:** Wait for candle to close (see purple "Exit: SessionEnd" in TV)

### 5. **Exchange Differences** 🏦
- **Dashboard:** NSE data via Angel One
- **TradingView:** May use different exchange/data provider
- **Fix:** Ensure both use NSE data

## How to Sync Perfectly

### Step 1: Match Timeframes ✅
```
Dashboard: 30min ←→ TradingView: 30m
```

### Step 2: Refresh Both 🔄
```
Dashboard: Click "🔄 Refresh Now" button
TradingView: Refresh browser or wait for next candle
```

### Step 3: Compare Key Values 🔍
```
✓ Current Price (should be within ₹1-2)
✓ RSI (should be within 1-2 points)
✓ Trend direction (should match)
✓ VWAP position (Above/Below should match)
```

### Step 4: Check Timestamps ⏰
```
Dashboard: Shows "Last updated: YYYY-MM-DD HH:MM:SS IST"
TradingView: Check chart timestamp at bottom
```

## Expected Value Ranges

| Indicator | Acceptable Difference |
|-----------|----------------------|
| **Price** | ± ₹0.50 - ₹2.00 |
| **RSI** | ± 1-2 points |
| **MACD** | ± 0.01-0.05 |
| **ADX** | ± 1-3 points |
| **VWAP** | ± ₹1-5 |
| **EMAs** | ± ₹0.50 - ₹2.00 |

## Common Issues & Fixes

### Issue: Price shows ₹1696.50 vs TradingView ₹1699.70 ❌
**Cause:** Different candle (dashboard may be 1-2 candles behind)
**Fix:** 
1. Check timestamps match
2. Refresh dashboard
3. Wait for next candle close

### Issue: RSI shows 49.5 vs TradingView 52.92 ❌
**Cause:** Different candle data or period setting
**Fix:**
1. Verify RSI period = 14 on both
2. Ensure same timeframe (30min)
3. Refresh to sync candles

### Issue: Trend shows "Bear" vs TradingView "Bullish" ❌
**Cause:** Using different EMAs or different candle
**Fix:**
1. Verify EMA settings: 9, 21, 50
2. Check if looking at same candle
3. Trend changes at crossovers - may be transitioning

### Issue: No signals but TradingView shows trade ❌
**Cause:** TradingView strategy uses different logic or shows historical
**Fix:**
1. Dashboard only shows signals at candle close
2. TradingView may show open positions from earlier
3. Check "Exit: SessionEnd" label - that was a forced exit

## Debug Checklist ✓

Before reporting data mismatch:

- [ ] Same symbol (SUNPHARMA vs SUNPHARMA)
- [ ] Same timeframe (30min vs 30m)
- [ ] Same exchange (NSE)
- [ ] Refreshed both within last minute
- [ ] Checked timestamps match
- [ ] Market is open (9:20 AM - 3:30 PM IST)
- [ ] Looking at same candle (not historical vs live)

## Understanding TradingView Indicators

Your TradingView screenshot shows:
```
Trend: Bearish (red)
RSI: 52.92
MACD: 17.13 (red - bearish)
ADX: 6.33 (weak trend)
Quality: 3/4
Position: None (just exited via SessionEnd)
```

This matches a **previous position that closed**, not a new signal.

The "Exit: SessionEnd" marker shows the strategy auto-closed a short position at 3:00 PM (15 min before market close), which is the time-based exit feature.

## Real-Time Sync Test

To verify sync:

1. **Open both side-by-side**
2. **Wait for next candle close** (e.g., 2:00 PM → 2:30 PM)
3. **Immediately refresh dashboard**
4. **Compare these 5 values:**
   - Current Price
   - RSI
   - Trend (Bull/Bear/Neutral)
   - VWAP position (Above/Below)
   - Volume status (High/Normal)

If these match within acceptable ranges, your sync is perfect! ✅

## Pro Tips 💡

1. **Use Auto-Refresh** - Enable in sidebar with 60s interval
2. **Enable Debug Info** - Shows API tokens and intervals
3. **Check Data Timestamp** - Shows exact time of last update
4. **Compare Quality Score** - Should match 0-4 range
5. **Watch for Signal Bars** - Green/Red markers appear at same time

## Support

If data consistently differs by >5%:
- Check Angel One API status
- Verify symbol tokens in config.py
- Try different stock to isolate issue
- Check terminal logs for errors

---

**Remember:** Small differences (1-3%) are normal due to:
- Data source variations
- Network latency
- Exchange reporting times
- Calculation precision differences
