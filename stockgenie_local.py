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

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

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
                logger.debug(f"No data fetched for {symbol}")
                return pd.DataFrame()
            self.consecutive_failures = 0
            logger.debug(f"Fetched {len(data)} rows for {symbol}")
            return data
        except Exception as e:
            if "429" in str(e):
                self._adjust_delay(e)
                raise RateLimitError("API quota exceeded")
            raise

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
    if (len(data) < 2 or 'Volume' not in data.columns or 
        data['Close'].isna().all() or data['High'].equals(data['Low'])):
        logger.debug("Insufficient or invalid data for volume profile calculation")
        return pd.Series()
    
    try:
        price_min = data['Low'].min()
        price_max = data['High'].max()
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
        logger.error(f"Error in volume profile calculation: {str(e)}")
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
    denominator = risk_per_share * data['Close'].iloc[-1]
    if denominator == 0:
        return 0.01
    position_size = min(0.2, max_risk / denominator)
    return max(0.01, position_size)

def detect_rectangle_pattern(data, lookback=20):
    """Detect Bullish/Bearish Rectangle patterns based on consolidation between support and resistance."""
    try:
        if len(data) < lookback:
            return None
        
        recent_data = data.tail(lookback)
        highs = recent_data['High']
        lows = recent_data['Low']
        
        resistance = highs.max()
        support = lows.min()
        
        high_touches = sum(1 for h in highs if abs(h - resistance) / resistance < 0.01)  # Within 1% of resistance
        low_touches = sum(1 for l in lows if abs(l - support) / support < 0.01)       # Within 1% of support
        
        if high_touches >= 3 and low_touches >= 3:
            last_close = recent_data['Close'].iloc[-1]
            if last_close > resistance:
                return "Bullish Rectangle Breakout"
            elif last_close < support:
                return "Bearish Rectangle Breakdown"
        return None
    except Exception as e:
        logger.error(f"Error detecting Rectangle pattern: {str(e)}")
        return None

def detect_triangle_pattern(data, lookback=20):
    """Detect Ascending, Descending, or Symmetrical Triangle patterns."""
    try:
        if len(data) < lookback:
            return None
        
        recent_data = data.tail(lookback)
        highs = recent_data['High'].rolling(2).max()
        lows = recent_data['Low'].rolling(2).min()
        
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        last_close = recent_data['Close'].iloc[-1]
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        
        if abs(high_slope) < 0.01: high_slope = 0  # Treat small slopes as flat
        if abs(low_slope) < 0.01: low_slope = 0
        
        if high_slope < 0 and low_slope > 0:  # Symmetrical Triangle
            if last_close > resistance:
                return "Symmetrical Triangle Breakout"
            elif last_close < support:
                return "Symmetrical Triangle Breakdown"
        elif high_slope == 0 and low_slope > 0:  # Ascending Triangle
            if last_close > resistance:
                return "Ascending Triangle Breakout"
        elif high_slope < 0 and low_slope == 0:  # Descending Triangle
            if last_close < support:
                return "Descending Triangle Breakdown"
        return None
    except Exception as e:
        logger.error(f"Error detecting Triangle pattern: {str(e)}")
        return None

# Placeholder functions for other patterns (to be implemented)
def detect_double_top_bottom(data, lookback=20):
    return None  # Implement later

def detect_head_and_shoulders(data, lookback=20):
    return None  # Implement later

def detect_flag_pattern(data, lookback=20):
    return None  # Implement later

def detect_cup_and_handle(data, lookback=20):
    return None  # Implement later

def detect_retest(data, pattern_type="Rectangle", lookback=5):
    """Check for retest after breakout/breakdown."""
    try:
        if len(data) < lookback + 2:
            return False
        
        recent_data = data.tail(lookback + 2)
        last_close = recent_data['Close'].iloc[-1]
        
        if pattern_type == "Rectangle" and 'Rectangle_Pattern' in data.columns:
            pattern = recent_data['Rectangle_Pattern'].iloc[-2]
            if pattern == "Bullish Rectangle Breakout":
                resistance = recent_data['High'].iloc[:-2].max()
                if (recent_data['Close'].iloc[-2] > resistance and 
                    last_close < resistance and 
                    last_close > recent_data['Low'].iloc[:-2].min()):
                    return True
            elif pattern == "Bearish Rectangle Breakdown":
                support = recent_data['Low'].iloc[:-2].min()
                if (recent_data['Close'].iloc[-2] < support and 
                    last_close > support and 
                    last_close < recent_data['High'].iloc[:-2].max()):
                    return True
        return False
    except Exception as e:
        logger.error(f"Error detecting retest: {str(e)}")
        return False

def detect_volume_confirmed_breakout(data):
    required_cols = ['Donchian_Upper', 'Donchian_Lower', 'Volume', 'Avg_Volume', 'Close']
    if (not all(col in data.columns for col in required_cols) or 
        data[required_cols].iloc[-1].isna().any() or len(data) < 20):
        logger.debug("Insufficient data for Donchian breakout")
        return "Neutral"
    
    last_close = data['Close'].iloc[-1]
    upper_band = data['Donchian_Upper'].iloc[-1]
    lower_band = data['Donchian_Lower'].iloc[-1]
    volume = data['Volume'].iloc[-1]
    avg_volume = data['Avg_Volume'].iloc[-1] or 1e-6
    volume_multiplier = get_volume_multiplier(data)
    
    if 'Rectangle_Pattern' in data.columns and data['Rectangle_Pattern'].iloc[-1]:
        pattern = data['Rectangle_Pattern'].iloc[-1]
        if ("Bullish" in pattern and last_close > upper_band and 
            volume > avg_volume * volume_multiplier):
            return "Bullish Rectangle Breakout (Volume Confirmed)"
        elif ("Bearish" in pattern and last_close < lower_band and 
              volume > avg_volume * volume_multiplier):
            return "Bearish Rectangle Breakdown (Volume Confirmed)"
    
    if last_close > upper_band and volume > avg_volume * volume_multiplier:
        return "Bullish Breakout"
    elif last_close < lower_band and volume > avg_volume * volume_multiplier:
        return "Bearish Breakout"
    return "Neutral"

def add_moving_averages(data):
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['EMA_5'] = ta.trend.EMAIndicator(data['Close'], window=5).ema_indicator()
    data['EMA_10'] = ta.trend.EMAIndicator(data['Close'], window=10).ema_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
    return data

def add_standard_macd(data):
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    return data

def calculate_cpr(data):
    if len(data) < 2:
        return data
    
    prev_day = data.iloc[-2]
    high = prev_day['High']
    low = prev_day['Low']
    close = prev_day['Close']
    
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot if pivot > bc else (bc - pivot) + bc
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    data['Pivot'] = pivot
    data['TC'] = tc
    data['BC'] = bc
    data['R1'] = r1
    data['S1'] = s1
    data['R2'] = r2
    data['S2'] = s2
    data['R3'] = r3
    data['S3'] = s3
    return data

def add_supertrend(data):
    supertrend = ta.trend.SuperTrend(data['High'], data['Low'], data['Close'], period=10, multiplier=3)
    data['Supertrend'] = supertrend.supertrend()
    data['Supertrend_Direction'] = supertrend.supertrend_direction()
    return data

def calculate_fibonacci_levels(data, lookback=20):
    if len(data) < lookback:
        return data
    
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
    return data

def add_price_action(data, lookback=5):
    data['Higher_High'] = data['High'] > data['High'].shift(1).rolling(lookback).max()
    data['Lower_Low'] = data['Low'] < data['Low'].shift(1).rolling(lookback).min()
    return data

def analyze_stock(data):
    if data.empty or len(data) < 10:
        logger.warning(f"Insufficient data: only {len(data)} rows.")
        return data
    
    try:
        required_cols = ['Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing required columns: {set(required_cols) - set(data.columns)}")
            return data
        
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['RSI'] = rsi.reindex(data.index, fill_value=0)
        
        atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        data['ATR'] = atr.reindex(data.index, fill_value=0)
        
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        data = add_vwap_indicators(data)
        
        avg_volume = data['Volume'].rolling(window=10, min_periods=1).mean()
        data['Avg_Volume'] = avg_volume.reindex(data.index, fill_value=0)
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * get_volume_multiplier(data))
        
        vol_profile = calculate_volume_profile(data)
        data['Volume_Profile'] = data['Close'].map(vol_profile) if not vol_profile.empty else 0
        data['POC'] = vol_profile.idxmax() if not vol_profile.empty else np.nan
        data['HVN'] = vol_profile[vol_profile > np.percentile(vol_profile.dropna(), 70)].index.tolist() if not vol_profile.empty else []
        
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband().reindex(data.index, fill_value=0)
        data['Lower_Band'] = bollinger.bollinger_lband().reindex(data.index, fill_value=0)
        
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband().reindex(data.index, fill_value=0)
        data['Donchian_Lower'] = donchian.donchian_channel_lband().reindex(data.index, fill_value=0)
        
        data = add_moving_averages(data)
        data = add_standard_macd(data)
        data = calculate_cpr(data)
        data = add_supertrend(data)
        data = calculate_fibonacci_levels(data)
        data = add_price_action(data)
        
        # Add chart pattern detection
        data['Rectangle_Pattern'] = data.apply(lambda row: detect_rectangle_pattern(data.loc[:row.name]), axis=1)
        data['Triangle_Pattern'] = data.apply(lambda row: detect_triangle_pattern(data.loc[:row.name]), axis=1)
        # Add other patterns here when implemented
        
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
    
    return data

def generate_recommendations(data, symbol=None, trader_type="Hybrid"):
    recommendations = {
        "Intraday": "Hold", "Current Price": None, "Buy At": None, "Stop Loss": None, 
        "Target": None, "Score": 0, "Position_Size": 0.1, "Signal": 0
    }
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        logger.debug(f"No valid close price for {symbol}")
        return recommendations
    
    last_close = data['Close'].iloc[-1]
    recommendations["Current Price"] = round(float(last_close), 2)
    buy_score = 0
    sell_score = 0

    if trader_type in ["Hybrid", "Technical Indicators Only"]:
        # RSI
        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
            if data['RSI'].iloc[-1] < 30:
                buy_score += 2
            elif data['RSI'].iloc[-1] > 70:
                sell_score += 2

        # EMA Crossover
        if 'EMA_5' in data.columns and 'EMA_20' in data.columns:
            if data['EMA_5'].iloc[-1] > data['EMA_20'].iloc[-1] and data['EMA_5'].iloc[-2] <= data['EMA_20'].iloc[-2]:
                buy_score += 1.5
            elif data['EMA_5'].iloc[-1] < data['EMA_20'].iloc[-1] and data['EMA_5'].iloc[-2] >= data['EMA_20'].iloc[-2]:
                sell_score += 1.5

        # MACD
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
                buy_score += 1.5
            elif data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
                sell_score += 1.5
            if 'MACD_Histogram' in data.columns and data['MACD_Histogram'].iloc[-1] > 0:
                buy_score += 0.5 * abs(data['MACD_Histogram'].iloc[-1])
            elif 'MACD_Histogram' in data.columns and data['MACD_Histogram'].iloc[-1] < 0:
                sell_score += 0.5 * abs(data['MACD_Histogram'].iloc[-1])

        # CPR
        if 'Pivot' in data.columns and pd.notnull(data['Pivot'].iloc[-1]):
            if last_close > data['TC'].iloc[-1] and last_close > data['R1'].iloc[-1]:
                buy_score += 1.5
            elif last_close < data['BC'].iloc[-1] and last_close < data['S1'].iloc[-1]:
                sell_score += 1.5
            elif data['BC'].iloc[-1] <= last_close <= data['TC'].iloc[-1]:
                buy_score += 0.5
                sell_score += 0.5

        # Supertrend
        if 'Supertrend_Direction' in data.columns:
            if data['Supertrend_Direction'].iloc[-1] == 1 and data['Supertrend_Direction'].iloc[-2] == -1:
                buy_score += 1.5
            elif data['Supertrend_Direction'].iloc[-1] == -1 and data['Supertrend_Direction'].iloc[-2] == 1:
                sell_score += 1.5

        # Fibonacci
        if 'Fib_61.8' in data.columns:
            if last_close <= data['Fib_61.8'].iloc[-1] and last_close > data['Fib_50.0'].iloc[-1]:
                buy_score += 1
            elif last_close >= data['Fib_23.6'].iloc[-1] and last_close < data['Fib_38.2'].iloc[-1]:
                sell_score += 1

    if trader_type in ["Hybrid", "Price Action Only"]:
        # Price Action and Chart Patterns
        if 'Higher_High' in data.columns and data['Higher_High'].iloc[-1]:
            buy_score += 1
        if 'Lower_Low' in data.columns and data['Lower_Low'].iloc[-1]:
            sell_score += 1
        
        if 'Rectangle_Pattern' in data.columns and data['Rectangle_Pattern'].iloc[-1]:
            pattern = data['Rectangle_Pattern'].iloc[-1]
            if "Bullish" in pattern:
                buy_score += 2.0
            elif "Bearish" in pattern:
                sell_score += 2.0
        if 'Triangle_Pattern' in data.columns and data['Triangle_Pattern'].iloc[-1]:
            pattern = data['Triangle_Pattern'].iloc[-1]
            if "Breakout" in pattern:
                buy_score += 2.0
            elif "Breakdown" in pattern:
                sell_score += 2.0
        
        # Retest Boost
        if detect_retest(data, "Rectangle"):
            if "Bullish" in data['Rectangle_Pattern'].iloc[-2]:
                buy_score += 1.5
                recommendations["Buy At"] = round(data['Close'].iloc[-1], 2)
            elif "Bearish" in data['Rectangle_Pattern'].iloc[-2]:
                sell_score += 1.5
                recommendations["Sell At"] = round(data['Close'].iloc[-1], 2)

    # Volume and Breakout
    breakout_signal = detect_volume_confirmed_breakout(data)
    climax_signal = detect_volume_climax(data)
    if climax_signal == "Bullish Climax":
        buy_score += 1.5
    elif climax_signal == "Bearish Climax":
        sell_score += 1.5

    if 'VWAP_Signal' in data.columns and pd.notnull(data['VWAP_Signal'].iloc[-1]):
        if data['VWAP_Signal'].iloc[-1] == 1:
            sell_score += 1.5
        elif data['VWAP_Signal'].iloc[-1] == -1:
            buy_score += 1.5

    volume_multiplier = get_volume_multiplier(data)
    if data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1] * volume_multiplier:
        buy_score *= 1.2
        sell_score *= 1.2

    base_threshold = 2.0 + (data['ATR'].iloc[-1] / data['Close'].iloc[-1]) if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 2.0
    strong_threshold = base_threshold + 1.5

    if breakout_signal in ["Bullish Breakout", "Bullish Rectangle Breakout (Volume Confirmed)"] and buy_score > sell_score:
        recommendations["Intraday"] = "🚀 Strong Buy" if buy_score >= strong_threshold else "📈 Buy"
        recommendations["Signal"] = 1
    elif breakout_signal in ["Bearish Breakout", "Bearish Rectangle Breakdown (Volume Confirmed)"] and sell_score > buy_score:
        recommendations["Intraday"] = "🔥 Strong Sell" if sell_score >= strong_threshold else "📉 Sell"
        recommendations["Signal"] = -1
    elif buy_score >= strong_threshold and sell_score < base_threshold:
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

    max_possible_score = 12  # Adjusted for additional pattern logic
    recommendations["Score"] = round(((buy_score - sell_score) / max_possible_score) * 10, 2)

    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        recommendations["Buy At"] = round(last_close * 0.99, 2) if recommendations.get("Buy At") is None else recommendations["Buy At"]
        recommendations["Stop Loss"] = round(last_close - (data['ATR'].iloc[-1] * 2.5), 2)
        recommendations["Target"] = round(last_close + ((last_close - recommendations["Stop Loss"]) * 3), 2)
    
    recommendations["Position_Size"] = dynamic_position_size(data)
    logger.debug(f"{symbol}: Buy Score={buy_score:.2f}, Sell Score={sell_score:.2f}, Recommendation={recommendations['Intraday']}")
    return recommendations

def check_real_time_alerts(symbol):
    data = fetch_stock_data_cached(symbol, period='1d', interval='5m')
    if len(data) < 10:
        return
    data = analyze_stock(data)
    current = data.iloc[-1]
    if current['Volume'] > 2 * data['Volume'].rolling(20, min_periods=1).mean().iloc[-1]:
        send_telegram_message(f"🚨 Volume spike in {symbol}: {current['Volume']/1000:.0f}K shares")
    if current['Close'] > data['Upper_Band'].iloc[-1]:
        send_telegram_message(f"🚀 Breakout in {symbol} at ₹{current['Close']:.2f}")

def show_performance_metrics(data, recommendations):
    st.subheader("📊 Strategy Performance")
    signals = []
    for i in range(len(data)):
        temp_data = data.iloc[:i+1]
        rec = generate_recommendations(temp_data)
        signals.append(rec["Signal"])
    data['Signal'] = signals
    
    returns = data['Close'].pct_change()
    strategy_returns = returns * data['Signal'].shift(1).fillna(0)
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) == 0:
        st.write("No trades executed.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col2:
        avg_return = strategy_returns.mean() * 100 if not pd.isna(strategy_returns.mean()) else 0
        st.metric("Avg Win/Loss", f"{avg_return:.2f}%")
    with col3:
        sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

def analyze_batch(stock_batch, progress_bar, progress_text, total_stocks, processed_count):
    results = []
    failed_symbols = []
    max_workers = min(4, os.cpu_count() - 1)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stock_worker") as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures, timeout=30):
            symbol = futures[future]
            try:
                result = future.result(timeout=10)
                if result:
                    results.append(result)
                else:
                    failed_symbols.append(symbol)
            except FuturesTimeoutError:
                logger.warning(f"Timeout processing {symbol}")
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
            processed_count[0] += 1
            progress = min(1.0, processed_count[0] / total_stocks)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {processed_count[0]}/{total_stocks} stocks analyzed "
                             f"(Failed: {len(failed_symbols)}, Estimated time remaining: {int((total_stocks - processed_count[0]) * 0.5)}s)")
    
    if failed_symbols:
        logger.info(f"Retrying {len(failed_symbols)} failed symbols")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in failed_symbols}
            for future in as_completed(futures, timeout=30):
                symbol = futures[future]
                try:
                    result = future.result(timeout=10)
                    if result:
                        results.append(result)
                except FuturesTimeoutError:
                    logger.warning(f"Retry timeout for {symbol}")
                except Exception as e:
                    logger.error(f"Failed retry for {symbol}: {str(e)}")
    
    clear_cache()
    return results

def analyze_stock_parallel(symbol):
    data = fetch_stock_data_cached(symbol)
    if not data.empty and len(data) >= 10:
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
    logger.debug(f"Skipping {symbol}: insufficient data ({len(data)} rows)")
    return None

def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception:
        st.warning("⚠️ Failed to fetch NSE stock list; using fallback.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def filter_stocks_by_price(stock_list, price_range, progress_bar, progress_text, total_stocks, processed_count):
    filtered_stocks = []
    max_workers = min(4, os.cpu_count() - 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_current_price, symbol): symbol for symbol in stock_list}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                current_price = future.result()
                if current_price is not None and price_range[0] <= current_price <= price_range[1]:
                    filtered_stocks.append(symbol)
                processed_count[0] += 1
                progress = min(1.0, processed_count[0] / total_stocks)
                progress_bar.progress(progress)
                progress_text.text(f"Filtering: {processed_count[0]}/{total_stocks} stocks processed "
                                 f"(Estimated time remaining: {int((total_stocks - processed_count[0]) * 0.5)}s)")
            except Exception as e:
                logger.error(f"Error filtering {symbol}: {str(e)}")
    return filtered_stocks

@lru_cache(maxsize=2000)
def fetch_current_price(symbol):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        data = yahoo_client.get_history(symbol, period="1d", interval="1d")
        if not data.empty and 'Close' in data.columns:
            return data['Close'].iloc[-1]
        logger.debug(f"No current price data for {symbol}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching current price for {symbol}: {str(e)}")
        return None

def analyze_all_stocks(stock_list, batch_size=10, price_range=None, short_sell=False):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    processed_count = [0]
    
    if price_range:
        total_stocks = len(stock_list)
        stock_list = filter_stocks_by_price(stock_list, price_range, progress_bar, progress_text, total_stocks, processed_count)
        processed_count[0] = 0
    
    results = []
    total_stocks = len(stock_list)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        results.extend(analyze_batch(batch, progress_bar, progress_text, total_stocks, processed_count))
    
    progress_bar.empty()
    progress_text.empty()
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        logger.debug("No valid stock analysis results returned.")
        return pd.DataFrame(columns=["Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Intraday", "Score", "Position_Size"])
    
    if short_sell:
        return results_df[results_df['Intraday'].isin(["📉 Sell", "🔥 Strong Sell"])].sort_values(by="Score").head(5)
    return results_df[results_df['Intraday'].isin(["📈 Buy", "🚀 Strong Buy"])].sort_values(by="Score", ascending=False).head(5)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")

def scan_for_patterns(stock_list, pattern_type="Rectangle"):
    results = []
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            if pattern_type == "Rectangle" and data['Rectangle_Pattern'].iloc[-1]:
                results.append({"Symbol": symbol, "Pattern": data['Rectangle_Pattern'].iloc[-1]})
            elif pattern_type == "Triangle" and data['Triangle_Pattern'].iloc[-1]:
                results.append({"Symbol": symbol, "Pattern": data['Triangle_Pattern'].iloc[-1]})
    return pd.DataFrame(results)

def plot_chart_patterns(data, symbol):
    fig = px.line(data, y=['Close'], title=f"{symbol.split('.')[0]} with Chart Patterns")
    
    if 'Rectangle_Pattern' in data.columns and data['Rectangle_Pattern'].notna().any():
        breakout_idx = data.index[data['Rectangle_Pattern'].notna()][-1]
        breakout_price = data.loc[breakout_idx, 'Close']
        fig.add_shape(type="rect",
                      x0=data.index[-20], y0=data['Low'].tail(20).min(),
                      x1=breakout_idx, y1=data['High'].tail(20).max(),
                      line=dict(color="Yellow", width=2), fillcolor="rgba(255,255,0,0.2)")
        fig.add_annotation(x=breakout_idx, y=breakout_price, text="Rectangle Breakout",
                           showarrow=True, arrowhead=2)
    
    if 'Triangle_Pattern' in data.columns and data['Triangle_Pattern'].notna().any():
        breakout_idx = data.index[data['Triangle_Pattern'].notna()][-1]
        breakout_price = data.loc[breakout_idx, 'Close']
        fig.add_shape(type="rect",
                      x0=data.index[-20], y0=data['Low'].tail(20).min(),
                      x1=breakout_idx, y1=data['High'].tail(20).max(),
                      line=dict(color="Green", width=2), fillcolor="rgba(0,255,0,0.2)")
        fig.add_annotation(x=breakout_idx, y=breakout_price, text="Triangle Breakout",
                           showarrow=True, arrowhead=2)
    
    st.plotly_chart(fig)

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro")
    st.sidebar.button("🧹 Clear Cache", on_click=clear_cache, key="clear_cache")
    enable_alerts = st.sidebar.checkbox("Enable Real-time Alerts", False)
    trader_type = st.sidebar.selectbox("Trader Type", ["Price Action Only", "Technical Indicators Only", "Hybrid"])
    
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🚀 Generate Top Picks", key="top_picks"):
            with st.spinner("Analyzing stocks..."):
                results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range, short_sell=False)
                if not results_df.empty:
                    st.subheader("🏆 Top 5 Buy Picks")
                    for _, row in results_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']}"):
                            st.write(f"Current Price: ₹{row['Current Price']:.2f}")
                            st.write(f"Buy At: ₹{row['Buy At']:.2f}" if row['Buy At'] else "Buy At: N/A")
                            st.write(f"Stop Loss: ₹{row['Stop Loss']:.2f}" if row['Stop Loss'] else "Stop Loss: N/A")
                            st.write(f"Target: ₹{row['Target']:.2f}" if row['Target'] else "Target: N/A")
                            st.write(f"Intraday: {row['Intraday']}")
                            st.write(f"Position Size: {row['Position_Size']*100:.1f}%")
                else:
                    st.write("No buy recommendations found.")
    
    with col2:
        if st.button("📈 Intraday Analysis", key="intraday_analysis") and symbol:
            with st.spinner("Fetching intraday data..."):
                st.session_state.intraday_clicked = True
                period = '1d' if enable_alerts else '5y'
                interval = '5m' if enable_alerts else '1d'
                data = fetch_stock_data_cached(symbol, period=period, interval=interval)
                if not data.empty and len(data) >= 10:
                    data = analyze_stock(data)
                    recommendations = generate_recommendations(data, symbol, trader_type)
                    st.subheader(f"📋 {symbol.split('.')[0]} Intraday Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{recommendations['Current Price']:.2f}")
                    with col2:
                        st.metric("Buy At", f"₹{recommendations['Buy At']:.2f}" if recommendations['Buy At'] else "N/A")
                    with col3:
                        st.metric("Stop Loss", f"₹{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] else "N/A")
                    with col4:
                        st.metric("Target", f"₹{recommendations['Target']:.2f}" if recommendations['Target'] else "N/A")
                    st.write(f"Intraday: {recommendations['Intraday']}")
                    st.write(f"Position Size: {recommendations['Position_Size']*100:.1f}%")
                    show_performance_metrics(data, recommendations)
                    if enable_alerts:
                        check_real_time_alerts(symbol)
                    plot_chart_patterns(data, symbol)
                else:
                    st.error(f"❌ Insufficient intraday data for {symbol}.")
    
    with col3:
        if st.button("📉 Short Sell Picks", key="short_sell_picks"):
            with st.spinner("Analyzing short sell opportunities..."):
                results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range, short_sell=True)
                if not results_df.empty:
                    st.subheader("🏆 Top 5 Short Sell Picks")
                    for _, row in results_df.iterrows():
                        with st.expander(f"{row['Symbol']} - Score: {row['Score']}"):
                            st.write(f"Current Price: ₹{row['Current Price']:.2f}")
                            st.write(f"Sell At: ₹{row['Current Price']:.2f}")
                            st.write(f"Stop Loss: ₹{row['Current Price'] + (row['Current Price'] - row['Buy At']) if row['Buy At'] else row['Current Price']:.2f}")
                            st.write(f"Target: ₹{row['Buy At']:.2f}" if row['Buy At'] else "N/A")
                            st.write(f"Intraday: {row['Intraday']}")
                            st.write(f"Position Size: {row['Position_Size']*100:.1f}%")
                else:
                    st.write("No short sell recommendations found.")
    
    with col4:
        if st.button("🔍 Scan for Patterns", key="pattern_scan"):
            with st.spinner("Scanning for patterns..."):
                pattern_results = scan_for_patterns(NSE_STOCKS, "Rectangle")
                if not pattern_results.empty:
                    st.subheader("Detected Rectangle Patterns")
                    st.dataframe(pattern_results)
                pattern_results = scan_for_patterns(NSE_STOCKS, "Triangle")
                if not pattern_results.empty:
                    st.subheader("Detected Triangle Patterns")
                    st.dataframe(pattern_results)

    if symbol and data is not None and recommendations is not None:
        if 'intraday_clicked' not in st.session_state or not st.session_state.intraday_clicked:
            st.subheader(f"📋 {symbol.split('.')[0]} Long-term Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{recommendations['Current Price']:.2f}")
            with col2:
                st.metric("Buy At", f"₹{recommendations['Buy At']:.2f}" if recommendations['Buy At'] else "N/A")
            with col3:
                st.metric("Stop Loss", f"₹{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] else "N/A")
            with col4:
                st.metric("Target", f"₹{recommendations['Target']:.2f}" if recommendations['Target'] else "N/A")
            st.write(f"Intraday: {recommendations['Intraday']}")
            st.write(f"Position Size: {recommendations['Position_Size']*100:.1f}%")
            show_performance_metrics(data, recommendations)
            plot_chart_patterns(data, symbol)

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    if 'intraday_clicked' not in st.session_state:
        st.session_state.intraday_clicked = False
    
    NSE_STOCKS = fetch_nse_stock_list()
    symbol = st.sidebar.selectbox("Choose stock:", [""] + NSE_STOCKS)
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 10:
            data = analyze_stock(data)
            trader_type = st.sidebar.selectbox("Trader Type", ["Price Action Only", "Technical Indicators Only", "Hybrid"])
            recommendations = generate_recommendations(data, symbol, trader_type)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error(f"❌ Insufficient data for {symbol}.")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()