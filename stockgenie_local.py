import os
import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
import numpy as np
import logging
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

class YahooFinanceClient:
    def __init__(self):
        self.last_call = 0
        self.delay = 1.0
        self.consecutive_failures = 0
        self.max_failures = 5
    
    def get_history(self, symbol, period="5y", interval="1d"):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        try:
            self.last_call = time.time()
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
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    logger.warning("Rate limit exceeded, pausing for 60s")
                    time.sleep(60)
                    self.consecutive_failures = 0
                else:
                    time.sleep(10)
            else:
                self.consecutive_failures += 1
            raise

yahoo_client = YahooFinanceClient()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def fetch_stock_data_with_retry(symbol, period="5y", interval="1d"):
    if ".NS" not in symbol:
        symbol += ".NS"
    data = yahoo_client.get_history(symbol, period, interval)
    time.sleep(0.2)
    return data

@lru_cache(maxsize=500)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    try:
        data = fetch_stock_data_with_retry(symbol, period, interval)
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol} after retries: {str(e)}")
        return pd.DataFrame()

def clear_cache():
    fetch_stock_data_cached.cache_clear()
    fetch_current_price.cache_clear()
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
        volume_profile = pd.Series(volume_profile.values, index=midpoints, name='Volume_Profile')
        logger.debug(f"Volume profile calculated with {bins} bins, total volume: {volume_profile.sum()}")
        return volume_profile
    
    except Exception as e:
        logger.error(f"Error in volume profile calculation: {str(e)}")
        return pd.Series()

def get_volume_multiplier(data, window=14):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return 1.5
    volatility_ratio = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
    return min(2.5, max(1.5, 1.5 + (volatility_ratio * 10)))

def volume_weighted_macd(data, window_fast=12, window_slow=26):
    if len(data) < window_slow or 'Volume' not in data.columns:
        return pd.Series(index=data.index, data=0, dtype=float)
    vwma_fast = (data['Close'] * data['Volume']).rolling(window_fast, min_periods=1).sum() / data['Volume'].rolling(window_fast, min_periods=1).sum()
    vwma_slow = (data['Close'] * data['Volume']).rolling(window_slow, min_periods=1).sum() / data['Volume'].rolling(window_slow, min_periods=1).sum()
    return (vwma_fast - vwma_slow).reindex(data.index, fill_value=0)

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
    if last_close > upper_band and volume > avg_volume * volume_multiplier:
        return "Bullish Breakout"
    elif last_close < lower_band and volume > avg_volume * volume_multiplier:
        return "Bearish Breakout"
    return "Neutral"

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
        
        data['VW_MACD'] = volume_weighted_macd(data).reindex(data.index, fill_value=0)
        
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband().reindex(data.index, fill_value=0)
        data['Lower_Band'] = bollinger.bollinger_lband().reindex(data.index, fill_value=0)
        
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband().reindex(data.index, fill_value=0)
        data['Donchian_Lower'] = donchian.donchian_channel_lband().reindex(data.index, fill_value=0)
        
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
    
    return data

def generate_recommendations(data, symbol=None):
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

    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        if data['RSI'].iloc[-1] < 30:
            buy_score += 2
        elif data['RSI'].iloc[-1] > 70:
            sell_score += 2

    if 'VW_MACD' in data.columns and pd.notnull(data['VW_MACD'].iloc[-1]):
        if data['VW_MACD'].iloc[-1] > 0:
            buy_score += 1
        elif data['VW_MACD'].iloc[-1] < 0:
            sell_score += 1

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

    if breakout_signal == "Bullish Breakout" and buy_score > sell_score:
        recommendations["Intraday"] = "🚀 Strong Buy" if buy_score >= strong_threshold else "📈 Buy"
        recommendations["Signal"] = 1
    elif breakout_signal == "Bearish Breakout" and sell_score > buy_score:
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

    max_possible_score = 7
    recommendations["Score"] = round(((buy_score - sell_score) / max_possible_score) * 10, 2)

    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        recommendations["Buy At"] = round(last_close * 0.99, 2)
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
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_symbols.append(symbol)
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in failed_symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
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

@lru_cache(maxsize=2000)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=60))
def fetch_current_price(symbol):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        data = yahoo_client.get_history(symbol, period="1d", interval="1d")
        time.sleep(random.uniform(0.5, 1.0))  # Increased random delay
        if not data.empty and 'Close' in data.columns:
            return data['Close'].iloc[-1]
        logger.debug(f"No current price data for {symbol}")
        return None
    except Exception as e:
        if "429" in str(e):
            logger.warning(f"Rate limit hit for {symbol}")
            raise
        logger.warning(f"Error fetching current price for {symbol}: {str(e)}")
        return None

def fetch_nse_stock_list_with_prices(price_range, batch_size=50):
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        symbols = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        
        filtered_stocks = []
        progress_bar = st.progress(0)
        progress_text = st.empty()
        processed_count = [0]
        total_stocks = len(symbols)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_symbol = {executor.submit(fetch_current_price, symbol): symbol for symbol in batch}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        current_price = future.result()
                        if current_price is not None and price_range[0] <= current_price <= price_range[1]:
                            filtered_stocks.append(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {str(e)}")
                    processed_count[0] += 1
                    progress = min(1.0, processed_count[0] / total_stocks)
                    progress_bar.progress(progress)
                    progress_text.text(f"Initial Filtering: {processed_count[0]}/{total_stocks} stocks processed "
                                     f"(Filtered: {len(filtered_stocks)})")
            time.sleep(2)  # Pause between batches
        
        progress_bar.empty()
        progress_text.empty()
        return filtered_stocks
    except Exception:
        st.warning("⚠️ Failed to fetch NSE stock list; using fallback.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def analyze_all_stocks(stock_list, batch_size=10, short_sell=False):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    processed_count = [0]
    total_stocks = len(stock_list)
    
    results = []
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

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro")
    st.sidebar.button("🧹 Clear Cache", on_click=clear_cache, key="clear_cache")
    enable_alerts = st.sidebar.checkbox("Enable Real-time Alerts", False)
    
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))
    
    if 'filtered_stocks' not in st.session_state or st.session_state.price_range != price_range:
        with st.spinner("Fetching and filtering stocks by price range..."):
            st.session_state.filtered_stocks = fetch_nse_stock_list_with_prices(price_range)
            st.session_state.price_range = price_range
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Generate Top Picks", key="top_picks"):
            with st.spinner("Analyzing stocks..."):
                results_df = analyze_all_stocks(st.session_state.filtered_stocks, short_sell=False)
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
                    recommendations = generate_recommendations(data, symbol)
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
                    fig = px.line(data, y=['Close', 'VWAP'], title="Price and VWAP")
                    st.plotly_chart(fig)
                else:
                    st.error(f"❌ Insufficient intraday data for {symbol}.")
    
    with col3:
        if st.button("📉 Short Sell Picks", key="short_sell_picks"):
            with st.spinner("Analyzing short sell opportunities..."):
                results_df = analyze_all_stocks(st.session_state.filtered_stocks, short_sell=True)
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
            fig = px.line(data, y=['Close', 'VWAP'], title="Price and VWAP (Long-term)")
            st.plotly_chart(fig)

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    if 'intraday_clicked' not in st.session_state:
        st.session_state.intraday_clicked = False
    
    symbol = st.sidebar.selectbox("Choose stock:", [""] + (st.session_state.get('filtered_stocks', fetch_nse_stock_list_with_prices((100, 1000)))))
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 10:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, st.session_state.get('filtered_stocks'))
        else:
            st.error(f"❌ Insufficient data for {symbol}.")
    else:
        display_dashboard(None, None, None, st.session_state.get('filtered_stocks'))

if __name__ == "__main__":
    main()