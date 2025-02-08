import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from diskcache import Cache
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
import io
import random
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tooltip explanations
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
}

# Tooltip function
def tooltip(label, explanation):
    """Returns a formatted tooltip string"""
    return f"{label} 📌 ({explanation})"

# Persistent caching with diskcache
cache = Cache('stock_data_cache')

# Retry decorator for Yahoo Finance requests with jitter
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
                        st.error(f"❌ Max retries reached for function {func.__name__}")
                        raise e
                    # Exponential backoff with jitter
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
        return wrapper
    return decorator

# Fetch NSE stock list with fallback
@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    """Fetch live NSE stock list from the official NSE website."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        st.success("✅ Fetched live NSE stock list successfully!")
        return stock_list
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch live NSE stock list. Falling back to predefined list. Error: {str(e)}")
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS",
            "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS",
            "AATMAJ.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS",
            "ABFRL.NS",
        ]

# Fetch stock data with caching
@retry(max_retries=3, delay=2)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    """Fetch data with retries and persistent caching."""
    cache_key = f"{symbol}_{period}_{interval}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        # Data validation: Ensure required columns are present
        required_columns = {'Close', 'High', 'Low', 'Volume'}
        if not required_columns.issubset(data.columns):
            missing_cols = required_columns - set(data.columns)
            st.error(f"❌ Missing required columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        cache[cache_key] = data
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch data for {symbol} after 3 attempts")
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

# Enhanced Fundamental Analysis
def enhanced_fundamental_filter(stock):
    """Advanced fundamental screening."""
    info = stock.info
    metrics = {
        'peg_ratio': info.get('pegRatio', 2),
        'current_ratio': info.get('currentRatio', 1),
        'operating_margin': info.get('operatingMargins', 0),
        'sector': info.get('sector', 'N/A')
    }
    
    # Sector-aware valuation
    sector_ratios = {
        'Technology': {'peg': 1.5, 'margin': 0.2},
        'Financial Services': {'peg': 2.0, 'margin': 0.15},
        'Healthcare': {'peg': 1.8, 'margin': 0.18},
        'Consumer Cyclical': {'peg': 1.7, 'margin': 0.12},
        'N/A': {'peg': 2.0, 'margin': 0.1}
    }
    sector = metrics['sector']
    
    return (
        metrics['peg_ratio'] < sector_ratios.get(sector, {}).get('peg', 2) and
        metrics['current_ratio'] > 1.5 and
        metrics['operating_margin'] > sector_ratios.get(sector, {}).get('margin', 0.1)
    )

# Configurable Technical Indicators
def calculate_technical_indicators(data, rsi_window=14, macd_fast=12, macd_slow=26):
    """Calculate technical indicators with configurable parameters."""
    if data.empty or len(data) < 27:
        return data
    try:
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        # MACD
        macd = ta.trend.MACD(data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
        # Moving Averages
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
        # ATR
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Error calculating indicators: {e}")
    return data

# Advanced Risk Management
def dynamic_stop_loss(data):
    """Trend-aware stop loss calculation."""
    adx = data['ADX'].iloc[-1]
    trend_direction = data['EMA_50'].iloc[-1] > data['EMA_200'].iloc[-1]
    
    base_multiplier = 2.5
    if adx > 25:
        multiplier = base_multiplier * (1.5 if trend_direction else 0.8)
    else:
        multiplier = base_multiplier * 0.6
        
    return data['Close'].iloc[-1] - (multiplier * data['ATR'].iloc[-1])

# Comprehensive Risk Metrics
def calculate_portfolio_metrics(results_df):
    """Advanced performance analytics."""
    metrics = {
        'win_rate': (results_df['Target'] > results_df['Current Price']).mean(),
        'avg_risk_reward': (results_df['Target'] - results_df['Current Price']).mean() / 
                          (results_df['Current Price'] - results_df['Stop Loss']).mean(),
        'volatility': results_df['Current Price'].pct_change().std(),
        'sharpe_ratio': None
    }
    
    if len(results_df) > 1:
        returns = results_df['Current Price'].pct_change().dropna()
        metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
    
    return metrics

# Main function
def main():
    """Main function with enhanced features."""
    st.sidebar.title("🔍 StockGenie Pro")
    NSE_STOCKS = fetch_nse_stock_list()
    
    # Configurable Technical Indicators
    st.sidebar.header("Indicator Settings")
    rsi_window = st.sidebar.slider("RSI Period", 5, 21, 14)
    macd_fast = st.sidebar.slider("MACD Fast", 5, 25, 12)
    macd_slow = st.sidebar.slider("MACD Slow", 15, 35, 26)
    
    # Symbol input with validation
    symbol = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x != "Custom" else x
    )
    if symbol == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g.: RELIANCE):")
        symbol = f"{custom_symbol}.NS" if custom_symbol else None
    if symbol:
        if ".NS" not in symbol:
            symbol += ".NS"
        if symbol not in NSE_STOCKS and not st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable"):
            return
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = calculate_technical_indicators(data, rsi_window, macd_fast, macd_slow)
            recommendations = generate_recommendations(data)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error("❌ Failed to load data for this symbol")

if __name__ == "__main__":
    main()