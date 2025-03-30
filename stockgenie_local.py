import os
import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
import numpy as np
import logging
import warnings
import threading
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log
from requests.exceptions import RequestException

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

DEBUG = False  # Can be set via st.sidebar.checkbox

class RateLimitError(Exception):
    pass

class YahooFinanceClient:
    def __init__(self):
        self._lock = threading.RLock()
        self.last_call = 0
        self.delay = 0.6
        self.consecutive_failures = 0
        self.max_failures = 5

    def get_history(self, symbol, period="5y", interval="1d"):
        with self._lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()
            return self._fetch_with_circuit_breaker(symbol, period=period, interval=interval)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RequestException, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _fetch_with_circuit_breaker(self, symbol, period, interval):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                # Try with shorter period if no data
                if period == "5y":
                    logger.debug(f"No data for 5y, trying 2y for {symbol}")
                    data = ticker.history(period="2y", interval=interval)
                if data.empty and interval == "1d":
                    logger.debug(f"No daily data, trying 1m for {symbol}")
                    data = ticker.history(period="1mo", interval="1h")
            return data
        except Exception as e:
            if "429" in str(e):
                self._adjust_delay(e)
                raise RateLimitError("API quota exceeded")
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()

    def _adjust_delay(self, exception):
        with self._lock:
            if "429" in str(exception):
                self.consecutive_failures += 1
                self.delay = min(5, 0.6 * (2 ** self.consecutive_failures))
                logger.warning(f"Backoff delay increased to {self.delay}s")
            else:
                self.consecutive_failures = max(0, self.consecutive_failures - 1)

yahoo_client = YahooFinanceClient()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RequestException, RateLimitError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def fetch_stock_data_with_retry(symbol, period="5y", interval="1d"):
    if ".NS" not in symbol:
        symbol += ".NS"
    data = yahoo_client.get_history(symbol, period, interval)
    time.sleep(0.1)
    return data

@lru_cache(maxsize=2000)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    try:
        data = fetch_stock_data_with_retry(symbol, period, interval)
        if len(data) > 1000:
            return data.iloc[-1000:]
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol} after retries: {str(e)}")
        return pd.DataFrame()

def clear_cache():
    fetch_stock_data_cached.cache_clear()
    logger.info("Cache cleared.")

def calculate_volume_profile(data, bins=50):
    if len(data) < 2 or 'Volume' not in data.columns or data['Close'].isna().all():
        logger.debug("Insufficient data for volume profile")
        return pd.Series()
    
    try:
        price_min = data['Low'].min()
        price_max = data['High'].max()
        if price_min == price_max:
            return pd.Series()
            
        price_bins = np.linspace(price_min, price_max, bins + 1)
        price_categories = pd.cut(data['Close'], bins=price_bins, include_lowest=True)
        volume_profile = data.groupby(price_categories, observed=True)['Volume'].sum()
        volume_profile = volume_profile.reindex(
            pd.IntervalIndex.from_arrays(price_bins[:-1], price_bins[1:]), 
            fill_value=0
        )
        midpoints = (price_bins[:-1] + price_bins[1:]) / 2
        return pd.Series(volume_profile.values, index=midpoints, name='Volume_Profile')
    except Exception as e:
        logger.error(f"Volume profile error: {str(e)}")
        return pd.Series()

def get_volume_multiplier(data, window=14):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return 1.5
    volatility_ratio = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
    return min(2.5, max(1.5, 1.5 + (volatility_ratio * 10)))

def detect_volume_climax(data):
    if not all(col in data.columns for col in ['Close', 'High', 'Low', 'Volume_Spike', 'RSI']):
        return None
        
    last_close = data['Close'].iloc[-1]
    is_high = last_close == data['High'].rolling(5, min_periods=1).max().iloc[-1]
    is_low = last_close == data['Low'].rolling(5, min_periods=1).min().iloc[-1]
    volume_spike = data['Volume_Spike'].iloc[-1]
    
    if volume_spike and is_high and data['RSI'].iloc[-1] > 70:
        return "Bearish Climax"
    elif volume_spike and is_low and data['RSI'].iloc[-1] < 30:
        return "Bullish Climax"
    return None

def add_vwap_indicators(data):
    if 'VWAP' not in data.columns or data['VWAP'].isna().all():
        return data
        
    data['VWAP_Deviation'] = (data['Close'] - data['VWAP']) / data['VWAP'] * 100
    data['VWAP_Signal'] = np.where(data['VWAP_Deviation'] > 1, 1, 
                                 np.where(data['VWAP_Deviation'] < -1, -1, 0))
    return data

def dynamic_position_size(data, portfolio_size=100000):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]) or data['Close'].iloc[-1] == 0:
        return 0.01
        
    risk_per_share = data['ATR'].iloc[-1] * 2
    max_risk = portfolio_size * 0.01
    position_size = min(0.2, max_risk / (risk_per_share * data['Close'].iloc[-1]))
    return max(0.01, position_size)

def detect_volume_confirmed_breakout(data):
    required_cols = ['Donchian_Upper', 'Donchian_Lower', 'Volume', 'Avg_Volume', 'Close']
    if not all(col in data.columns for col in required_cols):
        return "Neutral"
        
    last_close = data['Close'].iloc[-1]
    upper_band = data['Donchian_Upper'].iloc[-1]
    lower_band = data['Donchian_Lower'].iloc[-1]
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Avg_Volume'].iloc[-1] or 1e-6
    volume_multiplier = get_volume_multiplier(data)
    
    if last_close > upper_band and volume > avg_volume * volume_multiplier:
        return "Bullish Breakout"
    elif last_close < lower_band and volume > avg_volume * volume_multiplier:
        return "Bearish Breakout"
    return "Neutral"

def add_moving_averages(data):
    for window in [5, 10, 20, 50, 200]:
        try:
            data[f'EMA_{window}'] = ta.trend.EMAIndicator(data['Close'], window=window).ema_indicator()
        except:
            data[f'EMA_{window}'] = np.nan
    return data

def add_standard_macd(data):
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
    except:
        data['MACD'] = data['MACD_Signal'] = data['MACD_Histogram'] = np.nan
    return data

def calculate_cpr(data):
    if len(data) < 2:
        return data
        
    try:
        prev_day = data.iloc[-2]
        high = prev_day['High']
        low = prev_day['Low']
        close = prev_day['Close']
        
        pivot = (high + low + close) / 3
        bc = (high + low) / 2
        tc = (pivot - bc) + pivot
        
        data['Pivot'] = pivot
        data['TC'] = tc
        data['BC'] = bc
        data['R1'] = 2 * pivot - low
        data['S1'] = 2 * pivot - high
        data['R2'] = pivot + (high - low)
        data['S2'] = pivot - (high - low)
    except:
        pass
        
    return data

def add_supertrend(data):
    try:
        supertrend = ta.trend.SuperTrend(data['High'], data['Low'], data['Close'], period=10, multiplier=3)
        data['Supertrend'] = supertrend.supertrend()
        data['Supertrend_Direction'] = supertrend.supertrend_direction()
    except:
        data['Supertrend'] = data['Supertrend_Direction'] = np.nan
    return data

def calculate_fibonacci_levels(data, lookback=20):
    if len(data) < lookback:
        return data
        
    try:
        recent_data = data.tail(lookback)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        levels = {
            'Fib_23.6': swing_high - (diff * 0.236),
            'Fib_38.2': swing_high - (diff * 0.382),
            'Fib_50.0': swing_high - (diff * 0.50),
            'Fib_61.8': swing_high - (diff * 0.618),
            'Swing_High': swing_high,
            'Swing_Low': swing_low
        }
        
        for key, value in levels.items():
            data[key] = value
    except:
        pass
        
    return data

def add_price_action(data, lookback=5):
    try:
        data['Higher_High'] = data['High'] > data['High'].shift(1).rolling(lookback).max()
        data['Lower_Low'] = data['Low'] < data['Low'].shift(1).rolling(lookback).min()
    except:
        data['Higher_High'] = data['Lower_Low'] = False
    return data

def analyze_stock(data):
    if data.empty:
        return data
        
    # Ensure required columns exist
    for col in ['Close', 'High', 'Low', 'Volume']:
        if col not in data.columns:
            data[col] = np.nan
            
    if len(data) < 10:
        logger.warning(f"Insufficient data: {len(data)} rows")
        return data
        
    try:
        # Basic indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        
        # Volume analysis
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        data = add_vwap_indicators(data)
        
        data['Avg_Volume'] = data['Volume'].rolling(window=10, min_periods=1).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * get_volume_multiplier(data))
        
        # Volume profile
        vol_profile = calculate_volume_profile(data)
        if not vol_profile.empty:
            data['Volume_Profile'] = data['Close'].map(vol_profile)
            data['POC'] = vol_profile.idxmax()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Lower_Band'] = bollinger.bollinger_lband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        
        # Additional indicators
        data = add_moving_averages(data)
        data = add_standard_macd(data)
        data = calculate_cpr(data)
        data = add_supertrend(data)
        data = calculate_fibonacci_levels(data)
        data = add_price_action(data)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
    
    return data

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Current Price": None, "Buy At": None, 
        "Stop Loss": None, "Target": None, "Score": 0, 
        "Position_Size": 0.1, "Signal": 0
    }
    
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        return recommendations
        
    last_close = data['Close'].iloc[-1]
    recommendations["Current Price"] = round(float(last_close), 2)
    buy_score = 0
    sell_score = 0

    # RSI scoring
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2

    # EMA Crossover
    if 'EMA_5' in data.columns and 'EMA_20' in data.columns:
        ema5 = data['EMA_5'].iloc[-1]
        ema20 = data['EMA_20'].iloc[-1]
        ema5_prev = data['EMA_5'].iloc[-2] if len(data) > 1 else ema5
        ema20_prev = data['EMA_20'].iloc[-2] if len(data) > 1 else ema20
        
        if ema5 > ema20 and ema5_prev <= ema20_prev:
            buy_score += 1.5
        elif ema5 < ema20 and ema5_prev >= ema20_prev:
            sell_score += 1.5

    # MACD scoring
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        macd_prev = data['MACD'].iloc[-2] if len(data) > 1 else macd
        signal_prev = data['MACD_Signal'].iloc[-2] if len(data) > 1 else macd_signal
        
        if macd > macd_signal and macd_prev <= signal_prev:
            buy_score += 1.5
        elif macd < macd_signal and macd_prev >= signal_prev:
            sell_score += 1.5
            
        if 'MACD_Histogram' in data.columns:
            hist = data['MACD_Histogram'].iloc[-1]
            if hist > 0:
                buy_score += 0.5 * abs(hist)
            elif hist < 0:
                sell_score += 0.5 * abs(hist)

    # Volume breakout scoring
    breakout_signal = detect_volume_confirmed_breakout(data)
    if breakout_signal == "Bullish Breakout":
        buy_score += 2
    elif breakout_signal == "Bearish Breakout":
        sell_score += 2
        
    climax_signal = detect_volume_climax(data)
    if climax_signal == "Bullish Climax":
        buy_score += 1.5
    elif climax_signal == "Bearish Climax":
        sell_score += 1.5

    # Volume multiplier
    volume_multiplier = get_volume_multiplier(data)
    if data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1] * volume_multiplier:
        buy_score *= 1.2
        sell_score *= 1.2

    # Determine recommendation
    base_threshold = 1.5  # Reduced threshold
    strong_threshold = 2.5
    
    if buy_score >= strong_threshold and sell_score < base_threshold:
        recommendations["Intraday"] = "🚀 Strong Buy"
        recommendations["Signal"] = 1
    elif buy_score >= base_threshold and sell_score < base_threshold:
        recommendations["Intraday"] = "📈 Buy"
        recommendations["Signal"] = 1
    elif sell_score >= strong_threshold and buy_score < base_threshold:
        recommendations["Intraday"] = "🔥 Strong Sell"
        recommendations["Signal"] = -1
    elif sell_score >= base_threshold and buy_score < base_threshold:
        recommendations["Intraday"] = "📉 Sell"
        recommendations["Signal"] = -1
    elif buy_score >= base_threshold and sell_score >= base_threshold:
        recommendations["Intraday"] = "⚠️ Hold (High Volatility)"
    else:
        recommendations["Intraday"] = "🛑 Hold"

    # Calculate score (-10 to 10 scale)
    max_possible_score = 10
    recommendations["Score"] = round(((buy_score - sell_score) / max_possible_score) * 10, 2)

    # Calculate trade parameters
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        atr = data['ATR'].iloc[-1]
        recommendations["Buy At"] = round(last_close * 0.99, 2)
        recommendations["Stop Loss"] = round(last_close - (atr * 2.5), 2)
        recommendations["Target"] = round(last_close + ((last_close - recommendations["Stop Loss"]) * 3), 2)
    
    recommendations["Position_Size"] = dynamic_position_size(data)
    
    if DEBUG:
        logger.info(f"{symbol}: Buy={buy_score:.1f}, Sell={sell_score:.1f}, Rec={recommendations['Intraday']}")
    
    return recommendations

def analyze_stock_parallel(symbol):
    try:
        data = fetch_stock_data_cached(symbol)
        if data.empty or len(data) < 10:
            return None
            
        data = analyze_stock(data)
        recommendations = generate_recommendations(data, symbol)
        
        return {
            "Symbol": symbol,
            "Current Price": recommendations["Current Price"],
            "Buy At": recommendations["Buy At"],
            "Stop Loss": recommendations["Stop Loss"],
            "Target": recommendations["Target"],
            "Intraday": recommendations["Intraday"],
            "Score": recommendations["Score"],
            "Position_Size": recommendations["Position_Size"]
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def fetch_nse_stock_list():
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception as e:
        logger.error(f"Failed to fetch NSE list: {str(e)}")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def filter_stocks_by_price(stock_list, price_range):
    filtered = []
    for symbol in stock_list[:200]:  # Limit to first 200 for performance
        price = fetch_current_price(symbol)
        if price and price_range[0] <= price <= price_range[1]:
            filtered.append(symbol)
    return filtered

@lru_cache(maxsize=2000)
def fetch_current_price(symbol):
    try:
        data = yahoo_client.get_history(symbol, period="1d", interval="1d")
        return data['Close'].iloc[-1] if not data.empty and 'Close' in data.columns else None
    except:
        return None

def analyze_all_stocks(stock_list, price_range=None, short_sell=False):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Filter by price if range specified
    if price_range:
        stock_list = filter_stocks_by_price(stock_list, price_range)
    
    # Process stocks in batches
    results = []
    batch_size = 10
    total_stocks = len(stock_list)
    
    for i in range(0, total_stocks, batch_size):
        batch = stock_list[i:i+batch_size]
        batch_results = []
        
        for symbol in batch:
            result = analyze_stock_parallel(symbol)
            if result:
                batch_results.append(result)
            
            # Update progress
            progress = min(1.0, (i + len(batch_results)) / total_stocks)
            progress_bar.progress(progress)
            progress_text.text(f"Analyzed {i + len(batch_results)}/{total_stocks} stocks")
        
        results.extend(batch_results)
    
    progress_bar.empty()
    progress_text.empty()
    
    # Convert to DataFrame and filter recommendations
    results_df = pd.DataFrame([r for r in results if r is not None])
    
    if results_df.empty:
        return pd.DataFrame()
    
    if short_sell:
        return results_df[results_df['Intraday'].str.contains("Sell")].sort_values("Score").head(5)
    else:
        return results_df[results_df['Intraday'].str.contains("Buy")].sort_values("Score", ascending=False).head(5)

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro")
    
    # Sidebar controls
    st.sidebar.button("🧹 Clear Cache", on_click=clear_cache)
    global DEBUG
    DEBUG = st.sidebar.checkbox("Debug Mode", False)
    enable_alerts = st.sidebar.checkbox("Enable Alerts", False)
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))
    
    # Main columns for actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Top Buy Picks"):
            with st.spinner("Finding best opportunities..."):
                results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range)
                if not results_df.empty:
                    st.subheader("🏆 Top 5 Buy Opportunities")
                    for _, row in results_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']:.1f}"):
                            st.write(f"**Price:** ₹{row['Current Price']:.2f}")
                            st.write(f"**Signal:** {row['Intraday']}")
                            st.write(f"**Buy At:** ₹{row['Buy At']:.2f}" if row['Buy At'] else "N/A")
                            st.write(f"**Stop Loss:** ₹{row['Stop Loss']:.2f}" if row['Stop Loss'] else "N/A")
                            st.write(f"**Target:** ₹{row['Target']:.2f}" if row['Target'] else "N/A")
                            st.write(f"**Position Size:** {row['Position_Size']*100:.1f}%")
                else:
                    st.warning("No strong buy opportunities found")
    
    with col2:
        if symbol and st.button("📈 Analyze Selected"):
            with st.spinner(f"Analyzing {symbol}..."):
                data = fetch_stock_data_cached(symbol)
                if not data.empty:
                    data = analyze_stock(data)
                    recommendations = generate_recommendations(data, symbol)
                    
                    st.subheader(f"📋 {symbol.split('.')[0]} Analysis")
                    cols = st.columns(4)
                    cols[0].metric("Price", f"₹{data['Close'].iloc[-1]:.2f}")
                    cols[1].metric("RSI", f"{data['RSI'].iloc[-1]:.1f}" if 'RSI' in data.columns else "N/A")
                    cols[2].metric("Volume", f"{data['Volume'].iloc[-1]/1000:.0f}K")
                    cols[3].metric("ATR", f"{data['ATR'].iloc[-1]:.2f}" if 'ATR' in data.columns else "N/A")
                    
                    if recommendations:
                        st.write(f"**Recommendation:** {recommendations['Intraday']}")
                        st.write(f"**Score:** {recommendations['Score']:.1f}/10")
                    
                    # Show chart
                    fig = px.line(data.tail(100), y=['Close', 'EMA_20', 'VWAP'], 
                                  title=f"{symbol.split('.')[0]} Price Action")
                    st.plotly_chart(fig)
                else:
                    st.error(f"Could not fetch data for {symbol}")
    
    with col3:
        if st.button("📉 Short Sell Picks"):
            with st.spinner("Finding short opportunities..."):
                results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range, short_sell=True)
                if not results_df.empty:
                    st.subheader("🏆 Top 5 Short Opportunities")
                    for _, row in results_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']:.1f}"):
                            st.write(f"**Price:** ₹{row['Current Price']:.2f}")
                            st.write(f"**Signal:** {row['Intraday']}")
                            st.write(f"**Stop Loss:** ₹{row['Buy At']:.2f}" if row['Buy At'] else "N/A")
                            st.write(f"**Target:** ₹{row['Stop Loss']:.2f}" if row['Stop Loss'] else "N/A")
                            st.write(f"**Position Size:** {row['Position_Size']*100:.1f}%")
                else:
                    st.warning("No strong short opportunities found")

def main():
    # Initialize session state
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    
    # Get stock list
    NSE_STOCKS = fetch_nse_stock_list()
    
    # Stock selector
    symbol = st.sidebar.selectbox("Select Stock:", [""] + NSE_STOCKS[:200])  # Limit to 200 for performance
    
    # Display dashboard
    display_dashboard(symbol=symbol if symbol else None, NSE_STOCKS=NSE_STOCKS)

if __name__ == "__main__":
    main()
