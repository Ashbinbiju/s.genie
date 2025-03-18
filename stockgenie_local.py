import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from datetime import datetime, timedelta, date
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import logging
import os
import json
import tempfile
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# API Keys and Constants
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
TELEGRAM_CHAT_ID = "-1002411670969"
PERFORMANCE_FILE = "stock_performance.json"
NEWS_API_KEY = "ed58659895e84dfb8162a8bb47d8525e"  # NewsAPI key
GNEWS_API_KEY = "e4f5f1442641400694645433a8f98b94"  # GNews API key

# Tooltips and Weights (unchanged, included for completeness)
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Stoch": "Stochastic Oscillator - Momentum indicator",
    "OBV": "On-Balance Volume - Buying/selling pressure",
    "PSAR": "Parabolic SAR - Trend reversal indicator",
    "Sentiment": "Sentiment from news sources",
}

WEIGHTS = {
    "RSI": 0.2,
    "MACD": 0.3,
    "VWAP": 0.25,
    "Volume_Spike": 0.15,
    "ATR": 0.1,
    "VWMACD": 0.25,
    "VWRSI": 0.2,
    "Stoch": 0.15,
    "OBV": 0.1,
    "PSAR": 0.1,
    "Sentiment": 0GREEN1,
    "Nifty_Corr": 0.1,
}

# Existing functions (unchanged, included for context)
def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

@st.cache_data(ttl=300)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=30)
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch NSE stock list: {str(e)}. Using fallback list.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def preprocess_data(data):
    if len(data) < 10:
        return pd.DataFrame()
    data = data.interpolate(method='linear').bfill()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        data[col] = np.where(np.abs(z_scores) < 3, data[col], data[col].median())
    data = data.asfreq('5min', method='ffill')
    return data

@st.cache_data(ttl=300)
def fetch_stock_data_batch(symbols, interval="5m", period="1d"):
    try:
        data = yf.download(symbols, period=period, interval=interval, group_by='ticker', 
                          threads=False, auto_adjust=True)
        if data.empty:
            return {}
        data = data.replace([np.inf, -np.inf], np.nan)
        return {symbol: preprocess_data(data[symbol].dropna()) for symbol in symbols if symbol in data.columns and not data[symbol].empty}
    except Exception as e:
        logger.error(f"Error fetching batch data: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def fetch_market_data():
    try:
        nifty_data = yf.download("^NSEI", period="1d", interval="5m", auto_adjust=True)
        return preprocess_data(nifty_data)
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

def fetch_newsapi_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol.split('.')[0]}&apiKey={"ed58659895e84dfb81 62a8bb47d8525e"759903}&language=en&sortBy=publishedAt&pageSize=10"
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get('articles', [])
        if not articles:
            return 0
        sentiment = sum(1 if "positive" in article['title'].lower() or "up" in article['title'].lower() else 
                       -1 if "negative" in article['title'].lower() or "down" in article['title'].lower() else 0 
                       for article in articles) / len(articles)
        return sentiment
    except Exception as e:
        logger.error(f"Error fetching NewsAPI sentiment: {str(e)}")
        return 0

def fetch_gnews_sentiment(symbol):
    url = f"https://gnews.io/api/v4/search?q={symbol.split('.')[0]}&token={"e4f5f1442641400694645433a8f98b94"}&lang=en&max=10"
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get('articles', [])
        if not articles:
            return 0
        sentiment = sum(1 if "positive" in article['title'].lower() or "up" in article['title'].lower() else 
                       -1 if "negative" in article['title'].lower() or "down" in article['title'].lower() else 0 
                       for article in articles) / len(articles)
        return sentiment
    except Exception as e:
        logger.error(f"Error fetching GNews sentiment: {str(e)}")
        return 0

@lru_cache(maxsize=100)
def optimize_rsi_window(close_data):
    close = np.array(close_data)
    best_window, best_sharpe = 14, -float('inf')
    if len(close) < 15:
        return best_window
    close = np.where(close == 0, 1e-10, close)
    returns = np.diff(close) / close[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    for window in range(10, 15):
        rsi = ta.momentum.RSIIndicator(pd.Series(close), window=window).rsi().values
        signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        strategy_returns = signals[:-1] * returns
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) != 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_window = sharpe, window
    return best_window

def calculate_advanced_indicators(data):
    close_vol = data['Close'] * data['Volume']
    vmacd_fast = close_vol.ewm(span=12, adjust=False).mean() / data['Volume'].ewm(span=12, adjust=False).mean()
    vmacd_slow = close_vol.ewm(span=26, adjust=False).mean() / data['Volume'].ewm(span=26, adjust=False).mean()
    data['VWMACD'] = vmacd_fast - vmacd_slow
    data['VWMACD_signal'] = data['VWMACD'].ewm(span=9, adjust=False).mean()

    gains = data['Close'].diff().where(data['Close'].diff() > 0, 0) * data['Volume']
    losses = -data['Close'].diff().where(data['Close'].diff() < 0, 0) * data['Volume']
    avg_gain = gains.rolling(window=14, min_periods=1).mean()
    avg_loss = losses.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['VWRSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_indicators(data, symbol):
    if len(data) < 5:
        return data
    try:
        rsi_window = optimize_rsi_window(tuple(data['Close'].values))
    except Exception as e:
        logger.error(f"Error optimizing RSI: {str(e)}")
        rsi_window = 14
    
    windows = {
        'rsi': min(rsi_window, len(data) - 1),
        'macd_slow': min(26, len(data) - 1),
        'macd_fast': min(12, len(data) - 1),
        'macd_sign': min(9, len(data) - 1),
        'bollinger': min(20, len(data) - 1),
        'atr': min(7, len(data) - 1),
        'volume': min(5, len(data) - 1),
        'stoch': min(14, len(data) - 1),
    }

    data = calculate_advanced_indicators(data)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=windows['rsi']).rsi()
    macd = ta.trend.MACD(data['Close'], window_slow=windows['macd_slow'], window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
    data[['MACD', 'MACD_signal']] = pd.DataFrame({'MACD': macd.macd(), 'MACD_signal': macd.macd_signal()}, index=data.index)
    bollinger = ta.volatility.BollingerBands(data['Close'], window=windows['bollinger'])
    data[['Upper_Band', 'Lower_Band']] = pd.DataFrame({
        'Upper_Band': bollinger.bollinger_hband(),
        'Lower_Band': bollinger.bollinger_lband()
    }, index=data.index)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=windows['atr']).average_true_range()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['Avg_Volume'] = data['Volume'].rolling(window=windows['volume'], min_periods=1).mean()
    data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 2)
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=windows['stoch'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data['PSAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    market_data = fetch_market_data()
    if not market_data.empty:
        data['Nifty_Corr'] = data['Close'].pct_change().rolling(window=10).corr(market_data['Close'].pct_change())
    else:
        data['Nifty_Corr'] = 0
    return data

def calculate_risk_metrics(data):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 5:
        return 0, 0
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
    downside_returns = returns[returns < 0]
    sortino = returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
    return sharpe, sortino

def detect_patterns(data):
    patterns = {}
    if len(data) >= 2:
        curr, prev = data.iloc[-1], data.iloc[-2]
        patterns['Engulfing'] = "Bullish" if (curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and
                                              curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']) else \
                               "Bearish" if (curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and
                                             curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']) else None
    
    if len(data) >= 20:
        recent = data.tail(20)
        highs, lows = recent['High'], recent['Low']
        if highs.max() - highs.min() < 0.01 * highs.mean() and lows.diff().mean() > 0:
            patterns['Triangle'] = "Bullish"
        elif lows.max() - lows.min() < 0.01 * lows.mean() and highs.diff().mean() < 0:
            patterns['Triangle'] = "Bearish"
    return patterns

def train_ml_model(data):
    features = ['RSI', 'MACD', 'VWMACD', 'VWAP', 'Volume_Spike', 'ATR', 'Stoch_K', 'OBV', 'PSAR', 'Nifty_Corr']
    X = data[features].dropna()
    if len(X) < 10:
        return None, None
    y = (data['Close'].loc[X.index].shift(-1) > data['Close'].loc[X.index]).astype(int)
    X = X.iloc[:-1]
    y = y.iloc[1:].dropna()
    if len(X) != len(y) or len(X) < 10:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info(f"Model Accuracy: {model.score(X_test, y_test):.2f}")
    return model, scaler

def predict_with_ml(data, model, scaler):
    if model is None or scaler is None:
        return "Hold"
    features = ['RSI', 'MACD', 'VWMACD', 'VWAP', 'Volume_Spike', 'ATR', 'Stoch_K', 'OBV', 'PSAR', 'Nifty_Corr']
    X = scaler.transform(data[features].iloc[-1:].values)
    prediction = model.predict(X)[0]
    return "Buy" if prediction == 1 else "Sell"

def get_dynamic_thresholds(data):
    volatility = data['Close'].pct_change().std() * 100
    thresholds = {"RSI_low": 30, "RSI_high": 70, "MACD_threshold": 0, "VWAP_factor": 1}
    if volatility > 2:
        thresholds.update({"RSI_low": 20, "RSI_high": 80, "MACD_threshold": 0.5, "VWAP_factor": 1.5})
    elif volatility < 0.5:
        thresholds.update({"RSI_low": 40, "RSI_high": 60, "MACD_threshold": 0.1, "VWAP_factor": 0.8})
    return thresholds

# Updated generate_recommendations to ensure all numeric fields are set
def generate_recommendations(data, sharpe, sortino, price_action, model, scaler, sentiment=0):
    rec = {k: "Hold" for k in ["Intraday", "RSIReversal", "MACrossover", "VWAPRejection"]}
    last_close = data['Close'].iloc[-1] if not data.empty else 0
    rec.update({
        "Current Price": round(last_close, 2) if not data.empty else 0,
        "Buy At": round(last_close, 2),  # Default to current price
        "Stop Loss": round(last_close, 2),  # Default to current price
        "Target": round(last_close, 2),  # Default to current price
        "Score": 0
    })
    if data.empty:
        return rec
    
    buy_score, sell_score = 0, 0
    thresholds = get_dynamic_thresholds(data)

    # RSI Reversal
    if 'RSI' in data:
        rsi_sell = data['RSI'].iloc[-1] > thresholds["RSI_high"]
        rsi_buy = data['RSI'].iloc[-1] < thresholds["RSI_low"]
        rec["RSIReversal"] = "Sell" if rsi_sell else "Buy" if rsi_buy else "Hold"
        buy_score += WEIGHTS["RSI"] * 2 if rec["RSIReversal"] == "Buy" else 0
        sell_score += WEIGHTS["RSI"] * 2 if rec["RSIReversal"] == "Sell" else 0

    # MACD Crossover
    if 'MACD' in data:
        macd_buy = data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]
        macd_sell = data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]
        rec["MACrossover"] = "Buy" if macd_buy else "Sell" if macd_sell else "Hold"
        buy_score += WEIGHTS["MACD"] * 2 if rec["MACrossover"] == "Buy" else 0
        sell_score += WEIGHTS["MACD"] * 2 if rec["MACrossover"] == "Sell" else 0

    # VWAP Rejection
    if 'VWAP' in data and len(data) >= 3:
        vwap = data['VWAP'].iloc[-1]
        vwap_sell = data['Close'].iloc[-2] > vwap and data['Close'].iloc[-1] < vwap
        vwap_buy = data['Close'].iloc[-2] < vwap and data['Close'].iloc[-1] > vwap
        rec["VWAPRejection"] = "Sell" if vwap_sell else "Buy" if vwap_buy else "Hold"
        buy_score += WEIGHTS["VWAP"] * 2 if rec["VWAPRejection"] == "Buy" else 0
        sell_score += WEIGHTS["VWAP"] * 2 if rec["VWAPRejection"] == "Sell" else 0

    # Additional Indicators
    if 'Volume_Spike' in data:
        if data['Volume_Spike'].iloc[-1]:
            buy_score += WEIGHTS["Volume_Spike"] if last_close > data['Close'].iloc[-2] else 0
            sell_score += WEIGHTS["Volume_Spike"] if last_close < data['Close'].iloc[-2] else 0

    if 'Stoch_K' in data:
        stoch_buy = data['Stoch_K'].iloc[-1] < 20 and data['Stoch_K'].iloc[-1] > data['Stoch_D'].iloc[-1]
        stoch_sell = data['Stoch_K'].iloc[-1] > 80 and data['Stoch_K'].iloc[-1] < data['Stoch_D'].iloc[-1]
        buy_score += WEIGHTS["Stoch"] * 2 if stoch_buy else 0
        sell_score += WEIGHTS["Stoch"] * 2 if stoch_sell else 0

    if 'OBV' in data:
        obv_trend = data['OBV'].diff().iloc[-1] > 0
        buy_score += WEIGHTS["OBV"] if obv_trend else 0
        sell_score += WEIGHTS["OBV"] if not obv_trend else 0

    if 'PSAR' in data:
        psar_buy = data['PSAR'].iloc[-1] < data['Close'].iloc[-1]
        psar_sell = data['PSAR'].iloc[-1] > data['Close'].iloc[-1]
        buy_score += WEIGHTS["PSAR"] if psar_buy else 0
        sell_score += WEIGHTS["PSAR"] if psar_sell else 0

    if 'Nifty_Corr' in data:
        buy_score += WEIGHTS["Nifty_Corr"] if data['Nifty_Corr'].iloc[-1] > 0.5 else 0
        sell_score += WEIGHTS["Nifty_Corr"] if data['Nifty_Corr'].iloc[-1] < -0.5 else 0

    # Sentiment adjustment
    buy_score += WEIGHTS["Sentiment"] if sentiment > 0 else 0
    sell_score += WEIGHTS["Sentiment"] if sentiment < 0 else 0

    # Pattern-based scoring
    if price_action.get('Triangle') == "Bullish":
        buy_score += 1
    elif price_action.get('Triangle') == "Bearish":
        sell_score += 1

    # ML Prediction
    ml_rec = predict_with_ml(data, model, scaler)
    buy_score += 2 if ml_rec == "Buy" else 0
    sell_score += 2 if ml_rec == "Sell" else 0

    # Confirmation signals
    confirmation_count = sum([
        1 if rec["RSIReversal"] == "Buy" else -1 if rec["RSIReversal"] == "Sell" else 0,
        1 if rec["MACrossover"] == "Buy" else -1 if rec["MACrossover"] == "Sell" else 0,
        1 if rec["VWAPRejection"] == "Buy" else -1 if rec["VWAPRejection"] == "Sell" else 0
    ])

    # Final recommendation
    if confirmation_count >= 2 and buy_score > sell_score + 0.5:
        rec["Intraday"] = "Buy"
    elif confirmation_count <= -2 and sell_score > buy_score + 0.5:
        rec["Intraday"] = "Sell"
    else:
        rec["Intraday"] = "Hold"

    # Dynamic Stop Loss and Target, ensuring no None values
    atr = data['ATR'].iloc[-1] if 'ATR' in data and not pd.isna(data['ATR'].iloc[-1]) else 0
    volatility_factor = data['Close'].pct_change().std() * 100 if not data.empty else 1
    stop_loss_factor = 1.5 if volatility_factor < 2 else 2 if volatility_factor < 4 else 3

    # Always set Buy At, Stop Loss, and Target based on recommendation
    if rec["Intraday"] == "Buy":
        rec["Buy At"] = round(last_close * 0.99, 2)
        rec["Stop Loss"] = round(last_close - (stop_loss_factor * atr), 2)
        rec["Target"] = round(last_close + 2 * (last_close - rec["Stop Loss"]), 2)
    elif rec["Intraday"] == "Sell":
        rec["Buy At"] = round(last_close * 1.01, 2)  # Slightly above for sell signal
        rec["Stop Loss"] = round(last_close + (stop_loss_factor * atr), 2)
        rec["Target"] = round(last_close - 2 * (rec["Stop Loss"] - last_close), 2)
    else:  # Hold
        rec["Buy At"] = round(last_close, 2)
        rec["Stop Loss"] = round(last_close, 2)
        rec["Target"] = round(last_close, 2)

    rec["Score"] = buy_score - sell_score
    return rec

def analyze_stock(symbol, data_batch):
    data = data_batch.get(symbol, pd.DataFrame())
    if len(data) < 5:
        return None
    
    data = calculate_indicators(data, symbol)
    sharpe, sortino = calculate_risk_metrics(data)
    price_action = detect_patterns(data)
    model, scaler = train_ml_model(data)
    rec = generate_recommendations(data, sharpe, sortino, price_action, model, scaler)
    result = {
        "Symbol": symbol, "Current Price": rec["Current Price"], "Buy At": rec["Buy At"],
        "Stop Loss": rec["Stop Loss"], "Target": rec["Target"], "Intraday": rec["Intraday"],
        "Score": rec["Score"], "Sharpe": sharpe, "Sortino": sortino
    }
    
    date_str = date.today().strftime('%Y-%m-%d')
    save_performance(symbol, result, date_str)
    return result

def analyze_batch(stock_batch, data_batch):
    results = []
    for symbol in stock_batch:
        result = analyze_stock(symbol, data_batch)
        if result:
            results.append(result)
    return results

@st.cache_data
def analyze_all_stocks(stock_list, batch_size=25, price_range=None):
    if st.session_state.get('cancel_operation', False):
        return pd.DataFrame()
    
    results = []
    total_items = len(stock_list)
    progress_bar = st.progress(0)
    loading_text = st.empty()
    start_time = time.time()

    for i in range(0, total_items, batch_size):
        if st.session_state.get('cancel_operation', False):
            break
        batch = stock_list[i:i + batch_size]
        data_batch = fetch_stock_data_batch(batch)
        batch_results = analyze_batch(batch, data_batch)
        results.extend(batch_results)
        processed_items = min(i + batch_size, total_items)
        progress_bar.progress(processed_items / total_items)
        elapsed = time.time() - start_time
        eta = int((elapsed / processed_items) * (total_items - processed_items)) if processed_items > 0 else 0
        loading_text.text(f"Processing {processed_items}/{total_items} (ETA: {eta}s)")

    progress_bar.empty()
    loading_text.empty()
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    if price_range:
        results_df = results_df[results_df['Current Price'].notna() & results_df['Current Price'].between(price_range[0], price_range[1])]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5).raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")

# Updated display_dashboard with safe formatting as a precaution
def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro - Intraday Analysis")
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Top Intraday Picks"):
            st.session_state.cancel_operation = False
            cancel_button = st.button("❌ Cancel")
            if cancel_button:
                st.session_state.cancel_operation = True
            results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range)
            if not results_df.empty:
                st.subheader("🏆 Top 5 Intraday Stocks (Refined with Sentiment)")
                telegram_msg = f"*Top 5 Intraday Stocks ({datetime.now().strftime('%d %b %Y')})*\nChat ID: {TELEGRAM_CHAT_ID}\n\n"
                
                for idx, row in results_df.iterrows():
                    symbol = row['Symbol']
                    data_batch = fetch_stock_data_batch([symbol])
                    data = data_batch.get(symbol)
                    if data is not None and not data.empty:
                        data = calculate_indicators(data, symbol)
                        sharpe, sortino = calculate_risk_metrics(data)
                        price_action = detect_patterns(data)
                        model, scaler = train_ml_model(data)
                        newsapi_sentiment = fetch_newsapi_sentiment(symbol)
                        gnews_sentiment = fetch_gnews_sentiment(symbol)
                        combined_sentiment = (newsapi_sentiment + gnews_sentiment) / 2
                        rec = generate_recommendations(data, sharpe, sortino, price_action, model, scaler, combined_sentiment)
                        results_df.at[idx, 'Intraday'] = rec['Intraday']
                        results_df.at[idx, 'Score'] = rec['Score']
                        results_df.at[idx, 'Buy At'] = rec['Buy At']
                        results_df.at[idx, 'Stop Loss'] = rec['Stop Loss']
                        results_df.at[idx, 'Target'] = rec['Target']
                        results_df.at[idx, 'Sentiment'] = combined_sentiment

                    with st.expander(f"{row['Symbol']} - Score: {rec['Score']:.2f} (Sentiment: {combined_sentiment:.2f})"):
                        # Safe formatting with defaults
                        buy_at = rec['Buy At'] if rec['Buy At'] is not None else row['Current Price']
                        stop_loss = rec['Stop Loss'] if rec['Stop Loss'] is not None else row['Current Price']
                        target = rec['Target'] if rec['Target'] is not None else row['Current Price']
                        st.write(f"Price: ₹{row['Current Price']:.2f}, Buy At: ₹{buy_at:.2f}, "
                                 f"Stop Loss: ₹{stop_loss:.2f}, Target: ₹{target:.2f}, "
                                 f"Intraday: {rec['Intraday']}")
                    telegram_msg += f"*{row['Symbol']}*: ₹{row['Current Price']:.2f} - {rec['Intraday']} (Sentiment: {combined_sentiment:.2f})\n"
                send_telegram_message(telegram_msg)

    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        sentiment = (fetch_newsapi_sentiment(symbol) + fetch_gnews_sentiment(symbol)) / 2
        refined_rec = generate_recommendations(data, *calculate_risk_metrics(data), detect_patterns(data), *train_ml_model(data), sentiment)
        buy_at = refined_rec['Buy At'] if refined_rec['Buy At'] is not None else refined_rec['Current Price']
        stop_loss = refined_rec['Stop Loss'] if refined_rec['Stop Loss'] is not None else refined_rec['Current Price']
        target = refined_rec['Target'] if refined_rec['Target'] is not None else refined_rec['Current Price']
        st.write(f"Price: ₹{refined_rec['Current Price']:.2f}, Intraday: {refined_rec['Intraday']}, "
                 f"Buy At: ₹{buy_at:.2f}, Stop Loss: ₹{stop_loss:.2f}, Target: ₹{target:.2f}, Sentiment: {sentiment:.2f}")
        fig = px.line(data, y=['Close', 'VWAP', 'PSAR'], title="Price, VWAP & PSAR (5m)")
        st.plotly_chart(fig)

def save_performance(symbol, recommendation, date_str):
    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, 'r') as f:
                performance_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            performance_data = {}
    
    if date_str not in performance_data:
        performance_data[date_str] = {}
    
    performance_data[date_str][symbol] = {
        'date': date_str,
        'current_price': recommendation['Current Price'],
        'buy_at': recommendation['Buy At'],
        'stop_loss': recommendation['Stop Loss'],
        'target': recommendation['Target'],
        'recommendation': recommendation['Intraday'],
        'timestamp': datetime.now().isoformat()
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(performance_data, temp_file, indent=2)
    shutil.move(temp_file.name, PERFORMANCE_FILE)

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    NSE_STOCKS = fetch_nse_stock_list()
    symbol = st.sidebar.selectbox("Stock:", [""] + NSE_STOCKS)
    
    if symbol:
        data_batch = fetch_stock_data_batch([symbol])
        data = data_batch.get(symbol)
        if data is not None and not data.empty:
            data = calculate_indicators(data, symbol)
            sharpe, sortino = calculate_risk_metrics(data)
            price_action = detect_patterns(data)
            model, scaler = train_ml_model(data)
            recommendations = generate_recommendations(data, sharpe, sortino, price_action, model, scaler)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error(f"❌ No data for {symbol}")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()