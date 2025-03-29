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

# Retry decorator for handling 429 errors with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def fetch_stock_data_with_retry(symbol, period="5y", interval="1d"):
    if ".NS" not in symbol:
        symbol += ".NS"
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)
    if data.empty:
        logger.debug(f"No data fetched for {symbol}")
        return pd.DataFrame()
    logger.debug(f"Fetched {len(data)} rows for {symbol}")
    time.sleep(0.5)  # Small delay to avoid hitting rate limits
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
    logger.info("Cache cleared.")

def calculate_volume_profile(data, bins=50):
    if len(data) < 2 or 'Volume' not in data.columns or data['Close'].isna().all():
        return pd.Series(index=data.index, data=0, dtype=float)
    try:
        volume_profile = pd.Series(index=data.index, data=0, dtype=float)
        for idx in data.index:
            volume_profile[idx] = data['Volume'].loc[idx]
        return volume_profile
    except Exception as e:
        logger.error(f"Error in volume profile calculation: {str(e)}")
        return pd.Series(index=data.index, data=0, dtype=float)

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
    if not all(col in data.columns for col in required_cols) or data[required_cols].iloc[-1].isna().any():
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
        
        data['Volume_Profile'] = calculate_volume_profile(data)
        data['POC'] = data['Volume_Profile'].idxmax() if not data['Volume_Profile'].isna().all() else np.nan
        data['HVN'] = data['Volume_Profile'][data['Volume_Profile'] > 
                      np.percentile(data['Volume_Profile'].dropna(), 70)].index.tolist() if not data['Volume_Profile'].isna().all() else []
        
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
            buy_score += 0.5
        elif data['VW_MACD'].iloc[-1] < 0:
            sell_score += 0.5

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

    if 'VWAP_Signal' in data.columns and pd.notnull(data['VWAP_Signal'].iloc[-1]):
        if data['VWAP_Signal'].iloc[-1] == 1:
            sell_score += 1
        elif data['VWAP_Signal'].iloc[-1] == -1:
            buy_score += 1

    if buy_score >= 2.5:
        recommendations["Intraday"] = "Buy"
        recommendations["Signal"] = 1
    elif sell_score >= 2.5:
        recommendations["Intraday"] = "Sell"
        recommendations["Signal"] = -1

    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        recommendations["Buy At"] = round(last_close * 0.99, 2)
        recommendations["Stop Loss"] = round(last_close - (data['ATR'].iloc[-1] * 2.5), 2)
        recommendations["Target"] = round(last_close + ((last_close - recommendations["Stop Loss"]) * 3), 2)
    
    recommendations["Score"] = buy_score - sell_score
    recommendations["Position_Size"] = dynamic_position_size(data)
    logger.debug(f"{symbol}: Buy Score={buy_score}, Sell Score={sell_score}, Recommendation={recommendations['Intraday']}")
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
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
                processed_count[0] += 1
                progress = min(1.0, processed_count[0] / total_stocks)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {processed_count[0]}/{total_stocks} stocks analyzed "
                                 f"(Estimated time remaining: {int((total_stocks - processed_count[0]) * 0.5)}s)")
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {str(e)}")
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
    with ThreadPoolExecutor(max_workers=4) as executor:
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
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1d")
        if not data.empty and 'Close' in data.columns:
            time.sleep(0.5)  # Small delay for rate limiting
            return data['Close'].iloc[-1]
        logger.debug(f"No current price data for {symbol}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching current price for {symbol}: {str(e)}")
        return None

def analyze_all_stocks(stock_list, batch_size=10, price_range=None, short_sell=False):  # Reduced batch size
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
        return results_df[results_df['Intraday'] == "Sell"].sort_values(by="Score").head(5)
    return results_df[results_df['Intraday'] == "Buy"].sort_values(by="Score", ascending=False).head(5)

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
    
    col1, col2, col3 = st.columns(3)
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
    
    NSE_STOCKS = fetch_nse_stock_list()
    symbol = st.sidebar.selectbox("Choose stock:", [""] + NSE_STOCKS)
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 10:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error(f"❌ Insufficient data for {symbol}.")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()