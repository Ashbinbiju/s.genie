import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
import itertools
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# API Keys
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
TELEGRAM_CHAT_ID = "-1002411670969"

# Tooltip explanations
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "PivotPoints": "Support/Resistance levels for intraday",
    "Breakdown": "Break below support with volume",
    "MACrossover": "Fast MA crosses slow MA",
    "VWAPRejection": "Price rejects VWAP level",
    "RSIReversal": "RSI overbought with trend reversal",
    "BearishFlag": "Bearish flag pattern breakdown",
}

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

def retry(max_retries=3, delay=1, backoff_factor=2, jitter=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {str(e)}")
                        raise
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch NSE stock list: {str(e)}. Using fallback list.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, interval="5m", period="1d"):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logger.warning(f"No data returned for {symbol} ({interval})")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} ({interval}): {str(e)}")
        return pd.DataFrame()

def optimize_rsi_window(data, windows=range(5, 10)):
    try:
        best_window, best_sharpe = 7, -float('inf')
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 5:
            return best_window
        for window in windows:
            if len(data) >= window + 1:
                rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
                signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
                strategy_returns = signals.shift(1) * returns
                sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe, best_window = sharpe, window
        return best_window
    except Exception as e:
        logger.error(f"Error optimizing RSI window: {str(e)}")
        return 7

def calculate_pivot_points(data):
    if len(data) < 2:
        return None, None, None
    try:
        prev_high = data['High'].iloc[-2]
        prev_low = data['Low'].iloc[-2]
        prev_close = data['Close'].iloc[-2]
        pp = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pp - prev_low
        s1 = 2 * pp - prev_high
        return pp, r1, s1
    except Exception as e:
        logger.error(f"Error calculating pivot points: {str(e)}")
        return None, None, None

def detect_engulfing(data):
    if len(data) < 2:
        return None
    try:
        curr_open, curr_close = data['Open'].iloc[-1], data['Close'].iloc[-1]
        prev_open, prev_close = data['Open'].iloc[-2], data['Close'].iloc[-2]
        bullish = (curr_close > curr_open and prev_close < prev_open and 
                   curr_close > prev_open and curr_open < prev_close)
        bearish = (curr_close < curr_open and prev_close > prev_open and 
                   curr_close < prev_open and curr_open > prev_close)
        return "Bullish" if bullish else "Bearish" if bearish else None
    except Exception as e:
        logger.error(f"Error detecting engulfing pattern: {str(e)}")
        return None

def detect_bearish_flag(data):
    if len(data) < 10:
        return False
    try:
        recent_data = data.tail(10)
        pole_drop = (recent_data['Close'].iloc[0] - recent_data['Close'].iloc[4]) / recent_data['Close'].iloc[0]
        if pole_drop < -0.02:  # 2%+ drop
            flag_range = recent_data['High'].iloc[5:].max() - recent_data['Low'].iloc[5:].min()
            if flag_range < (recent_data['Close'].iloc[0] * 0.01):  # Tight consolidation
                breakdown = recent_data['Close'].iloc[-1] < recent_data['Low'].iloc[5:].min()
                return breakdown
        return False
    except Exception as e:
        logger.error(f"Error detecting bearish flag: {str(e)}")
        return False

def check_data_sufficiency(data, timeframe="5m"):
    intervals = len(data)
    st.info(f"📅 {timeframe} Data: {intervals} intervals")
    if intervals < 5:
        st.warning(f"⚠️ Less than 5 intervals for {timeframe}; some indicators unavailable.")

def analyze_stock(symbol):
    data_5m = fetch_stock_data_cached(symbol, interval="5m", period="1d")
    data_15m = fetch_stock_data_cached(symbol, interval="15m", period="1d")
    data_1h = fetch_stock_data_cached(symbol, interval="1h", period="5d")
    data_daily = fetch_stock_data_cached(symbol, interval="1d", period="5d")
    
    if data_5m.empty:
        st.warning(f"⚠️ No 5-minute data for {symbol}")
        return data_5m, {}, {}
    
    check_data_sufficiency(data_5m, "5m")
    check_data_sufficiency(data_15m, "15m")
    check_data_sufficiency(data_1h, "1h")
    check_data_sufficiency(data_daily, "daily")
    
    intervals = len(data_5m)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data_5m.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing columns in 5m data for {symbol}: {', '.join(missing_cols)}")
        return data_5m, {}, {}
    
    windows = {
        'rsi': min(optimize_rsi_window(data_5m), max(5, intervals - 1)),
        'macd_slow': 13,
        'macd_fast': 6,
        'macd_sign': 5,
        'bollinger': 10,
        'atr': 7,
        'volume': 5,
        'donchian': 10,
        'ema_fast': 9,
        'ema_slow': 21,
    }

    try:
        data_5m['RSI'] = ta.momentum.RSIIndicator(data_5m['Close'], window=windows['rsi']).rsi()
        macd = ta.trend.MACD(data_5m['Close'], window_slow=windows['macd_slow'], 
                             window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
        data_5m['MACD'] = macd.macd()
        data_5m['MACD_signal'] = macd.macd_signal()
        bollinger = ta.volatility.BollingerBands(data_5m['Close'], window=windows['bollinger'], window_dev=2)
        data_5m['Middle_Band'] = bollinger.bollinger_mavg()
        data_5m['Upper_Band'] = bollinger.bollinger_hband()
        data_5m['Lower_Band'] = bollinger.bollinger_lband()
        data_5m['ATR'] = ta.volatility.AverageTrueRange(data_5m['High'], data_5m['Low'], data_5m['Close'], 
                                                        window=windows['atr']).average_true_range()
        data_5m['Cumulative_TP'] = ((data_5m['High'] + data_5m['Low'] + data_5m['Close']) / 3) * data_5m['Volume']
        data_5m['Cumulative_Volume'] = data_5m['Volume'].cumsum()
        data_5m['VWAP'] = data_5m['Cumulative_TP'].cumsum() / data_5m['Cumulative_Volume']
        data_5m['Avg_Volume'] = data_5m['Volume'].rolling(window=windows['volume']).mean()
        data_5m['Volume_Spike'] = data_5m['Volume'] > (data_5m['Avg_Volume'] * 2)
        data_5m['OBV'] = ta.volume.OnBalanceVolumeIndicator(data_5m['Close'], data_5m['Volume']).on_balance_volume()
        donchian = ta.volatility.DonchianChannel(data_5m['High'], data_5m['Low'], data_5m['Close'], 
                                                window=windows['donchian'])
        data_5m['Donchian_Upper'] = donchian.donchian_channel_hband()
        data_5m['Donchian_Lower'] = donchian.donchian_channel_lband()
        data_5m['EMA_Fast'] = ta.trend.EMAIndicator(data_5m['Close'], window=windows['ema_fast']).ema_indicator()
        data_5m['EMA_Slow'] = ta.trend.EMAIndicator(data_5m['Close'], window=windows['ema_slow']).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Error computing indicators for {symbol}: {str(e)}")
        return data_5m, {}, {}

    mtf_indicators = {}
    for timeframe, df in [("15m", data_15m), ("1h", data_1h), ("daily", data_daily)]:
        if not df.empty:
            try:
                mtf_indicators[timeframe] = {
                    'RSI': ta.momentum.RSIIndicator(df['Close'], window=14).rsi().iloc[-1] if len(df) >= 14 else None,
                    'MACD': ta.trend.MACD(df['Close']).macd().iloc[-1] if len(df) >= 26 else None,
                    'MACD_signal': ta.trend.MACD(df['Close']).macd_signal().iloc[-1] if len(df) >= 26 else None,
                    'VWAP': (df['Close'] * df['Volume']).cumsum().iloc[-1] / df['Volume'].cumsum().iloc[-1] if df['Volume'].sum() > 0 else None,
                }
            except Exception as e:
                logger.error(f"Error computing MTFA indicators for {timeframe} ({symbol}): {str(e)}")

    price_action = {
        'Engulfing_5m': detect_engulfing(data_5m),
        'Engulfing_15m': detect_engulfing(data_15m) if not data_15m.empty else None,
        'Bearish_Flag_5m': detect_bearish_flag(data_5m),
    }
    
    return data_5m, mtf_indicators, price_action

def calculate_buy_at(data):
    try:
        if data.empty or 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
            return None
        last_close = data['Close'].iloc[-1]
        last_rsi = data['RSI'].iloc[-1]
        if pd.notnull(last_close) and pd.notnull(last_rsi):
            buy_at = last_close * 0.99 if last_rsi < 30 else last_close
            return round(buy_at, 2)
        return None
    except Exception as e:
        logger.error(f"Error calculating buy_at: {str(e)}")
        return None

def calculate_stop_loss(data, atr_multiplier=1.5):
    try:
        if data.empty or 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
            return None
        last_close = data['Close'].iloc[-1]
        last_atr = data['ATR'].iloc[-1]
        if pd.notnull(last_close) and pd.notnull(last_atr):
            stop_loss = last_close - (atr_multiplier * last_atr)
            return round(stop_loss, 2)
        return None
    except Exception as e:
        logger.error(f"Error calculating stop_loss: {str(e)}")
        return None

def calculate_target(data, risk_reward_ratio=2):
    try:
        stop_loss = calculate_stop_loss(data)
        if stop_loss is None:
            return None
        last_close = data['Close'].iloc[-1]
        if pd.notnull(last_close) and pd.notnull(stop_loss):
            risk = last_close - stop_loss
            target = last_close + (risk * risk_reward_ratio)
            return round(target, 2)
        return None
    except Exception as e:
        logger.error(f"Error calculating target: {str(e)}")
        return None

def generate_recommendations(data, mtf_indicators, price_action, symbol=None):
    recommendations = {
        "Intraday": "Hold",
        "Breakdown": "Hold",
        "MACrossover": "Hold",
        "VWAPRejection": "Hold",
        "RSIReversal": "Hold",
        "BearishFlag": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        return recommendations
    
    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2) if pd.notnull(data['Close'].iloc[-1]) else None
        buy_score = sell_score = 0
        last_close = data['Close'].iloc[-1]

        # Breakdown Strategy
        if 'Donchian_Lower' in data.columns and pd.notnull(data['Donchian_Lower'].iloc[-1]):
            if last_close < data['Donchian_Lower'].iloc[-1] and data['Volume_Spike'].iloc[-1] and data['OBV'].iloc[-1] < data['OBV'].iloc[-2]:
                sell_score += 2
                recommendations["Breakdown"] = "Sell"

        # Moving Average Crossover
        if 'EMA_Fast' in data.columns and 'EMA_Slow' in data.columns and len(data) >= 2:
            if (data['EMA_Fast'].iloc[-1] > data['EMA_Slow'].iloc[-1] and 
                data['EMA_Fast'].iloc[-2] <= data['EMA_Slow'].iloc[-2]):
                buy_score += 2
                recommendations["MACrossover"] = "Buy"
            elif (data['EMA_Fast'].iloc[-1] < data['EMA_Slow'].iloc[-1] and 
                  data['EMA_Fast'].iloc[-2] >= data['EMA_Slow'].iloc[-2]):
                sell_score += 2
                recommendations["MACrossover"] = "Sell"

        # VWAP Rejection Strategy
        if 'VWAP' in data.columns and len(data) >= 3:
            vwap = data['VWAP'].iloc[-1]
            if (data['Close'].iloc[-2] > vwap and data['Close'].iloc[-1] < vwap and 
                data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]):
                sell_score += 2
                recommendations["VWAPRejection"] = "Sell"
            elif (data['Close'].iloc[-2] < vwap and data['Close'].iloc[-1] > vwap and 
                  data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]):
                buy_score += 2
                recommendations["VWAPRejection"] = "Buy"

        # RSI Overbought + Trend Reversal
        if 'RSI' in data.columns and 'MACD' in data.columns and 'MACD_signal' in data.columns:
            if (data['RSI'].iloc[-1] > 70 and data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] and 
                data['MACD'].iloc[-2] >= data['MACD_signal'].iloc[-2]):
                sell_score += 2
                recommendations["RSIReversal"] = "Sell"
            elif (data['RSI'].iloc[-1] < 30 and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] and 
                  data['MACD'].iloc[-2] <= data['MACD_signal'].iloc[-2]):
                buy_score += 2
                recommendations["RSIReversal"] = "Buy"

        # Bearish Flag and Pole
        if price_action['Bearish_Flag_5m']:
            sell_score += 2
            recommendations["BearishFlag"] = "Sell"

        # Volume Confirmation
        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1]:
            buy_score += 1 if data['OBV'].iloc[-1] > data['OBV'].iloc[-2] else 0
            sell_score += 1 if data['OBV'].iloc[-1] < data['OBV'].iloc[-2] else 0

        # Price Action Boost
        if price_action['Engulfing_5m'] == "Bullish":
            buy_score += 1
        elif price_action['Engulfing_5m'] == "Bearish":
            sell_score += 1

        # Multi-Timeframe Confirmation
        for timeframe in ['15m', '1h', 'daily']:
            if timeframe in mtf_indicators:
                tf = mtf_indicators[timeframe]
                if tf['RSI'] is not None:
                    if tf['RSI'] < 30:
                        buy_score += 0.5
                    elif tf['RSI'] > 70:
                        sell_score += 0.5
                if tf['MACD'] is not None and tf['MACD_signal'] is not None:
                    if tf['MACD'] > tf['MACD_signal']:
                        buy_score += 0.5
                    elif tf['MACD'] < tf['MACD_signal']:
                        sell_score += 0.5
                if tf['VWAP'] is not None:
                    if last_close > tf['VWAP']:
                        buy_score += 0.5
                    elif last_close < tf['VWAP']:
                        sell_score += 0.5

        # Decision Logic
        if buy_score >= 5:
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 5:
            recommendations["Intraday"] = "Strong Sell"
        elif buy_score > sell_score + 2:
            recommendations["Intraday"] = "Buy"
        elif sell_score > buy_score + 2:
            recommendations["Intraday"] = "Sell"

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations for {symbol}: {str(e)}")

    return recommendations

def analyze_batch(stock_batch):
    results = []
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            valid_futures = {}
            for symbol in stock_batch:
                try:
                    data_5m, mtf_indicators, price_action = analyze_stock(symbol)
                    if not data_5m.empty and len(data_5m) >= 5:
                        valid_futures[executor.submit(analyze_stock_parallel, symbol, data_5m, mtf_indicators, price_action)] = symbol
                    else:
                        st.warning(f"⚠️ Skipping {symbol}: insufficient or invalid data")
                except Exception as e:
                    st.warning(f"⚠️ Error submitting {symbol} for analysis: {str(e)}")
            
            for future in as_completed(valid_futures):
                symbol = valid_futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"⚠️ Error processing result for {symbol}: {str(e)}")
    except Exception as e:
        st.error(f"⚠️ Fatal error in analyze_batch: {str(e)}")
    return results

def analyze_stock_parallel(symbol, data_5m, mtf_indicators, price_action):
    try:
        recommendations = generate_recommendations(data_5m, mtf_indicators, price_action, symbol)
        return {
            "Symbol": symbol,
            "Current Price": recommendations["Current Price"],
            "Buy At": recommendations["Buy At"],
            "Stop Loss": recommendations["Stop Loss"],
            "Target": recommendations["Target"],
            "Intraday": recommendations["Intraday"],
            "Breakdown": recommendations["Breakdown"],
            "MACrossover": recommendations["MACrossover"],
            "VWAPRejection": recommendations["VWAPRejection"],
            "RSIReversal": recommendations["RSIReversal"],
            "BearishFlag": recommendations["BearishFlag"],
            "Score": recommendations.get("Score", 0),
        }
    except Exception as e:
        logger.error(f"Error in analyze_stock_parallel for {symbol}: {str(e)}")
        return None

def update_progress(progress_bar, loading_text, progress_value, loading_messages, start_time, total_items, processed_items):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    elapsed_time = time.time() - start_time
    if processed_items > 0:
        time_per_item = elapsed_time / processed_items
        remaining_items = total_items - processed_items
        eta = timedelta(seconds=int(time_per_item * remaining_items))
        loading_text.text(f"{loading_message}{dots} | Processed {processed_items}/{total_items} (ETA: {eta})")
    else:
        loading_text.text(f"{loading_message}{dots} | Processed 0/{total_items}")

def analyze_all_stocks(stock_list, batch_size=50, price_range=None, progress_callback=None):
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()

    results = []
    total_items = len(stock_list)
    start_time = time.time()
    
    for i in range(0, total_items, batch_size):
        if st.session_state.cancel_operation:
            st.warning("⚠️ Analysis canceled by user.")
            break
        
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        processed_items = min(i + len(batch), total_items)
        if progress_callback:
            progress_callback(processed_items / total_items, start_time, total_items, processed_items)
    
    if not results:
        st.warning("⚠️ No results generated")
        return pd.DataFrame()
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    for col in ['Current Price', 'Buy At', 'Stop Loss', 'Target']:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    if price_range:
        results_df = results_df[results_df['Current Price'].notnull() & 
                                (results_df['Current Price'] >= price_range[0]) & 
                                (results_df['Current Price'] <= price_range[1])]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def filter_strong_buy_stocks(results_df):
    if results_df.empty:
        return pd.DataFrame()
    strong_buy_df = results_df[results_df['Intraday'] == "Strong Buy"]
    return strong_buy_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    return recommendation

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{"7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"}/sendMessage"
    payload = {"chat_id": "-1002411670969", "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        st.success("📩 Telegram message sent successfully!")
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to send Telegram message: {str(e)}")
        return False

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro - Intraday Analysis (MTFA)")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=10000, value=(100, 1000))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generate Top Intraday Picks"):
            st.session_state.cancel_operation = False
            progress_bar = st.progress(0)
            loading_text = st.empty()
            cancel_button = st.button("❌ Cancel Intraday Analysis")
            loading_messages = itertools.cycle(["Analyzing trends...", "Fetching data...", "Crunching numbers..."])
            
            if cancel_button:
                st.session_state.cancel_operation = True
            
            try:
                results_df = analyze_all_stocks(
                    NSE_STOCKS,
                    price_range=price_range,
                    progress_callback=lambda progress, start_time, total, processed: update_progress(
                        progress_bar, loading_text, progress, loading_messages, start_time, total, processed
                    )
                )
                progress_bar.empty()
                loading_text.empty()
                if not results_df.empty and not st.session_state.cancel_operation:
                    st.subheader("🏆 Top 5 Intraday Stocks")
                    telegram_message = f"*🏆 Top 5 Intraday Stocks ({datetime.now().strftime('%d %b %Y')})*\n\n"
                    for _, row in results_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                            current_price = f"{row['Current Price']:.2f}" if pd.notnull(row['Current Price']) else "N/A"
                            buy_at = f"{row['Buy At']:.2f}" if pd.notnull(row['Buy At']) else "N/A"
                            stop_loss = f"{row['Stop Loss']:.2f}" if pd.notnull(row['Stop Loss']) else "N/A"
                            target = f"{row['Target']:.2f}" if pd.notnull(row['Target']) else "N/A"
                            st.markdown(f"""
                            {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                            Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                            Target: ₹{target}  
                            Intraday: {colored_recommendation(row['Intraday'])}  
                            Breakdown: {colored_recommendation(row['Breakdown'])}  
                            MA Crossover: {colored_recommendation(row['MACrossover'])}  
                            VWAP Rejection: {colored_recommendation(row['VWAPRejection'])}  
                            RSI Reversal: {colored_recommendation(row['RSIReversal'])}  
                            Bearish Flag: {colored_recommendation(row['BearishFlag'])}  
                            """)
                        telegram_message += (
                            f"*{row['Symbol']}* (Score: {row['Score']}/7)\n"
                            f"Price: ₹{current_price}\n"
                            f"Buy At: ₹{buy_at} | SL: ₹{stop_loss} | Target: ₹{target}\n"
                            f"Intraday: {row['Intraday']}\n\n"
                        )
                    send_telegram_message(telegram_message)
                elif not st.session_state.cancel_operation:
                    st.warning("⚠️ No top picks available due to data issues.")
            except Exception as e:
                st.error(f"⚠️ Error during batch analysis: {str(e)}")

    with col2:
        if st.button("💪 Generate Top Strong Buy Picks"):
            st.session_state.cancel_operation = False
            progress_bar = st.progress(0)
            loading_text = st.empty()
            cancel_button = st.button("❌ Cancel Strong Buy Analysis")
            loading_messages = itertools.cycle(["Scanning strong buys...", "Evaluating signals...", "Finalizing results..."])
            
            if cancel_button:
                st.session_state.cancel_operation = True
            
            try:
                results_df = analyze_all_stocks(
                    NSE_STOCKS,
                    price_range=price_range,
                    progress_callback=lambda progress, start_time, total, processed: update_progress(
                        progress_bar, loading_text, progress, loading_messages, start_time, total, processed
                    )
                )
                strong_buy_df = filter_strong_buy_stocks(results_df)
                progress_bar.empty()
                loading_text.empty()
                if not strong_buy_df.empty and not st.session_state.cancel_operation:
                    st.subheader("💪 Top Strong Buy Stocks")
                    telegram_message = f"*💪 Top Strong Buy Stocks ({datetime.now().strftime('%d %b %Y')})*\n\n"
                    for _, row in strong_buy_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                            current_price = f"{row['Current Price']:.2f}" if pd.notnull(row['Current Price']) else "N/A"
                            buy_at = f"{row['Buy At']:.2f}" if pd.notnull(row['Buy At']) else "N/A"
                            stop_loss = f"{row['Stop Loss']:.2f}" if pd.notnull(row['Stop Loss']) else "N/A"
                            target = f"{row['Target']:.2f}" if pd.notnull(row['Target']) else "N/A"
                            st.markdown(f"""
                            {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                            Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                            Target: ₹{target}  
                            Intraday: {colored_recommendation(row['Intraday'])}  
                            Breakdown: {colored_recommendation(row['Breakdown'])}  
                            MA Crossover: {colored_recommendation(row['MACrossover'])}  
                            VWAP Rejection: {colored_recommendation(row['VWAPRejection'])}  
                            RSI Reversal: {colored_recommendation(row['RSIReversal'])}  
                            Bearish Flag: {colored_recommendation(row['BearishFlag'])}  
                            """)
                        telegram_message += (
                            f"*{row['Symbol']}* (Score: {row['Score']}/7)\n"
                            f"Price: ₹{current_price}\n"
                            f"Buy At: ₹{buy_at} | SL: ₹{stop_loss} | Target: ₹{target}\n"
                            f"Intraday: {row['Intraday']}\n\n"
                        )
                    send_telegram_message(telegram_message)
                elif not st.session_state.cancel_operation:
                    st.warning("⚠️ No strong buy stocks found.")
            except Exception as e:
                st.error(f"⚠️ Error during strong buy analysis: {str(e)}")
    
    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Intraday Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = f"{recommendations['Current Price']:.2f}" if recommendations['Current Price'] is not None else "N/A"
            st.metric(tooltip("Current Price", TOOLTIPS['Stop Loss']), f"₹{current_price}")
        with col2:
            buy_at = f"{recommendations['Buy At']:.2f}" if recommendations['Buy At'] is not None else "N/A"
            st.metric("Buy At", f"₹{buy_at}")
        with col3:
            stop_loss = f"{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = f"{recommendations['Target']:.2f}" if recommendations['Target'] is not None else "N/A"
            st.metric("Target", f"₹{target}")
        st.subheader("📈 Intraday Recommendations")
        cols = st.columns(3)
        strategies = ["Intraday", "Breakdown", "MACrossover", "VWAPRejection", "RSIReversal", "BearishFlag"]
        for i, col in enumerate(cols):
            with col:
                for strategy in strategies[i::3]:
                    st.markdown(f"**{strategy.replace('_', ' ')}**")
                    st.markdown(colored_recommendation(recommendations[strategy]))
        tab1, tab2 = st.tabs(["📊 Price Action", "📉 Momentum"])
        with tab1:
            price_cols = ['Close', 'VWAP', 'Upper_Band', 'Lower_Band', 'EMA_Fast', 'EMA_Slow']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with VWAP, Bollinger & EMAs (5m)")
                st.plotly_chart(fig)
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators (5m)")
                st.plotly_chart(fig)

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False

    st.sidebar.title("🔍 Stock Search")
    NSE_STOCKS = fetch_nse_stock_list()
    
    symbol = None
    selected_option = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=[""] + NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x != "Custom" and x != "" else x
    )
    
    if selected_option == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g., RELIANCE):")
        if custom_symbol:
            symbol = f"{custom_symbol}.NS"
    elif selected_option != "":
        symbol = selected_option
    
    if symbol:
        if ".NS" not in symbol:
            symbol += ".NS"
        if symbol not in NSE_STOCKS:
            st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable")
        try:
            data_5m, mtf_indicators, price_action = analyze_stock(symbol)
            if not data_5m.empty:
                recommendations = generate_recommendations(data_5m, mtf_indicators, price_action, symbol)
                display_dashboard(symbol, data_5m, recommendations, NSE_STOCKS)
            else:
                st.error(f"❌ Failed to load data for {symbol}")
        except Exception as e:
            st.error(f"❌ Error analyzing {symbol}: {str(e)}")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()