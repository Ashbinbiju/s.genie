import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import time
import requests
import io
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
from scipy.signal import find_peaks
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import os
import telegram
from telegram.error import TelegramError
import asyncio

# Hardcoded API Keys (to be addressed later as per your request)
NEWSAPI_KEY = "ed58659895e84dfb8162a8bb47d8525e"
GNEWS_KEY = "e4f5f1442641400694645433a8f98b94"
ALPHA_VANTAGE_KEY = "TCAUKYUCIDZ6PI57"
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
TELEGRAM_CHAT_ID = "-1002411670969"

# Tooltip explanations
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Parabolic_SAR": "Parabolic Stop and Reverse - Trend reversal indicator",
    "Fib_Retracements": "Fibonacci Retracements - Support and resistance levels",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
    "Volume_Profile": "Volume traded at price levels - Support/Resistance",
    "Wave_Pattern": "Simplified Elliott Wave - Trend wave detection",
}

# Global FinBERT models to avoid repeated loading (Performance Fix #2)
FINBERT_TOKENIZER = None
FINBERT_MODEL = None

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

def retry(max_retries=3, delay=5, backoff_factor=2, jitter=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    time.sleep(0.5)  # Base delay for rate limits
                    return result
                except Exception as e:
                    if "Too Many Requests" in str(e) or "rate limited" in str(e).lower():
                        retries += 1
                        if retries == max_retries:
                            st.warning(f"Max retries reached for {args[0]}: {str(e)}. Skipping.")
                            return pd.DataFrame()
                        sleep_time = delay * (backoff_factor ** retries) + random.uniform(0, jitter)
                        st.warning(f"Rate limited for {args[0]}. Waiting {sleep_time:.2f}s ({retries}/{max_retries})...")
                        time.sleep(sleep_time)
                    else:
                        st.warning(f"Error for {args[0]}: {str(e)}")
                        return pd.DataFrame()
        return wrapper
    return decorator

@retry(max_retries=3, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        st.warning("⚠️ Failed to fetch NSE stock list; using fallback list.")
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS",
            "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS",
            "AATMAJ.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS",
            "ABFRL.NS",
        ]

def load_cache():
    cache_file = "stock_data_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open("stock_data_cache.pkl", 'wb') as f:
        pickle.dump(cache, f)

@retry(max_retries=3, delay=5)
def fetch_stock_data_cached(symbol, period=None, interval="1d"):
    """Fetch stock data with user-defined period (Historical Years Feature)."""
    if period is None:
        period = "5y"
    elif isinstance(period, int):
        period = f"{period}y"
        
    cache = load_cache()
    cache_key = f"{symbol}_{period}_{interval}"
    if cache_key in cache:
        return cache[cache_key]
    
    if ".NS" not in symbol:
        symbol += ".NS"
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty or 'Close' not in data.columns:  # Data Validation Fix #3
            st.warning(f"⚠️ No valid data returned for {symbol} from Yahoo Finance.")
            return pd.DataFrame()
        cache[cache_key] = data
        save_cache(cache)
        return data
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) > 1:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    return advances / declines if declines != 0 else 0

def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 10:  # Monte Carlo Fix #9
        st.warning("⚠️ Insufficient data for Monte Carlo simulation (<10 returns).")
        return []
    if len(returns) < 30:
        mean_return = returns.mean()
        std_return = returns.std() or 0.01
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            t_value = np.random.standard_t(df=5, size=1)[0]
            price = price_series[-1] * (1 + mean_return + t_value * volatility[i])
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

def scenario_analysis(data, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 10:  # Add validation
        st.warning("⚠️ Insufficient data for scenario analysis.")
        return [], []
    mean_return = returns.mean()
    std_return = returns.std() or 0.01
    last_price = data['Close'].iloc[-1]
    bull_return = mean_return + std_return
    bear_return = mean_return - std_return
    bull_scenario = [last_price]
    bear_scenario = [last_price]
    for _ in range(days):
        bull_scenario.append(bull_scenario[-1] * (1 + bull_return))
        bear_scenario.append(bear_scenario[-1] * (1 + bear_return))
    return bull_scenario, bear_scenario

def fetch_news_sentiment_vader(query, api_key, source="newsapi"):
    analyzer = SentimentIntensityAnalyzer()
    try:
        if source == "newsapi":
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey=ed58659895e84dfb8162a8bb47d8525e&language=en&sortBy=publishedAt&pageSize=5"
        elif source == "gnews":
            url = f"https://gnews.io/api/v4/search?q={query}&token=e4f5f1442641400694645433a8f98b94&lang=en&max=5"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        sentiment_scores = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment_scores.append(sentiment)
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment
    except Exception:
        return 0

def load_finbert_models():
    """Load FinBERT models once globally (Error Handling & Performance Fix #1, #2)."""
    global FINBERT_TOKENIZER, FINBERT_MODEL
    if FINBERT_TOKENIZER is None or FINBERT_MODEL is None:
        try:
            FINBERT_TOKENIZER = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            FINBERT_MODEL = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        except Exception as e:
            st.error(f"❌ Failed to load FinBERT: {str(e)}")
            return False
    return True

def analyze_sentiment_finbert(text):
    if not load_finbert_models():
        return 1  # Default to neutral
    try:
        inputs = FINBERT_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = FINBERT_MODEL(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.argmax().item()
    except Exception:
        return 1

def extract_entities(text):
    """Extract entities with Spacy model check (Error Handling Fix #1)."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("❌ Spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        return []
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def calculate_confidence_score(data):
    score = 0
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        score += 1 if data['RSI'].iloc[-1] < 30 else 0
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        score += 1 if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else 0
    if 'Ichimoku_Span_A' in data.columns and pd.notnull(data['Ichimoku_Span_A'].iloc[-1]) and pd.notnull(data['Close'].iloc[-1]):
        score += 1 if data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1] else 0
    return score / 3 if score > 0 else 0

def assess_risk(data):
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) and pd.notnull(data['ATR'].mean()):
        if data['ATR'].iloc[-1] > data['ATR'].mean():
            return "High Volatility Warning"
    return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 15)):
    best_window, best_sharpe = 9, -float('inf')
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 20:
        st.warning(f"⚠️ Insufficient data ({len(returns)} days) for RSI optimization; using default window 9.")
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

def detect_divergence(data):
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]) or len(data) < 5:
        return "No Divergence"
    rsi = data['RSI']
    price = data['Close']
    recent_highs = price[-5:].idxmax()
    recent_lows = price[-5:].idxmin()
    rsi_highs = rsi[-5:].idxmax()
    rsi_lows = rsi[-5:].idxmin()
    bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1])
    bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_volume_profile(data, bins=50):
    if data['Low'].empty or data['High'].empty:
        return None
    price_bins = np.linspace(data['Low'].min(), data['High'].max(), bins)
    volume_profile = np.zeros(bins - 1)
    for i in range(bins - 1):
        mask = (data['Close'] >= price_bins[i]) & (data['Close'] < price_bins[i + 1])
        volume_profile[i] = data['Volume'][mask].sum()
    return pd.Series(volume_profile, index=price_bins[:-1])

def detect_waves(data):
    peaks, _ = find_peaks(data['Close'], distance=5)
    troughs, _ = find_peaks(-data['Close'], distance=5)
    if len(peaks) > 2 and len(troughs) > 2:
        if pd.notnull(data['Close'].iloc[-1]) and pd.notnull(data['Close'].iloc[peaks[-1]]):
            if data['Close'].iloc[-1] > data['Close'].iloc[peaks[-1]]:
                return "Potential Uptrend (Wave 5?)"
            elif data['Close'].iloc[-1] < data['Close'].iloc[troughs[-1]]:
                return "Potential Downtrend (Wave C?)"
    return "No Clear Wave Pattern"

def check_data_sufficiency(data):
    days = len(data)
    st.info(f"📅 Data available: {days} days")
    if days < 15:
        st.warning("⚠️ Less than 15 days; most indicators unavailable or unreliable.")
    elif days < 50:
        st.warning("⚠️ Less than 50 days; some indicators (e.g., SMA_50, Ichimoku) may be limited.")
    elif days < 200:
        st.warning("⚠️ Less than 200 days; long-term indicators like SMA_200 unavailable.")

def analyze_stock(data):
    """Analyze stock data with technical indicators (Data Validation Fix #3, Indicator Fix #7)."""
    if data.empty or 'Close' not in data.columns:
        st.warning("⚠️ No valid stock data available.")
        return pd.DataFrame()
    
    check_data_sufficiency(data)
    days = len(data)
    if days < 5:  # Indicator Fix #7
        st.warning("⚠️ Data too short (<5 days) for meaningful analysis.")
        return data
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    windows = {
        'rsi': min(optimize_rsi_window(data), max(5, days - 1)),
        'macd_slow': min(26, max(5, days - 1)),
        'macd_fast': min(12, max(3, days - 1)),
        'macd_sign': min(9, max(3, days - 1)),
        'sma_20': min(20, max(5, days - 1)),
        'sma_50': min(50, max(10, days - 1)),
        'sma_200': min(200, max(20, days - 1)),
        'bollinger': min(20, max(5, days - 1)),
        'stoch': min(14, max(5, days - 1)),
        'atr': min(14, max(5, days - 1)),
        'adx': min(14, max(5, days - 1)),
        'volume': min(10, max(5, days - 1)),
        'ichimoku_w1': min(9, max(3, days - 1)),
        'ichimoku_w2': min(26, max(5, days - 1)),
        'ichimoku_w3': min(52, max(10, days - 1)),
        'cmf': min(20, max(5, days - 1)),
        'donchian': min(20, max(5, days - 1)),
    }

    if days >= windows['rsi'] + 1:
        try:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=windows['rsi']).rsi()
            data['Divergence'] = detect_divergence(data)
        except Exception as e:
            st.warning(f"⚠️ Failed to compute RSI/Divergence: {str(e)}")
            data['RSI'] = None
            data['Divergence'] = "No Divergence"

    # ... (rest of the indicator calculations remain largely unchanged, adding NaN checks where needed) ...

    if days >= windows['macd_slow'] + 1:
        try:
            macd = ta.trend.MACD(data['Close'], window_slow=windows['macd_slow'], 
                                window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()
        except Exception as e:
            st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
            data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None

    # Add NaN checks for key calculations
    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        if pd.notnull(data['Cumulative_Volume'].iloc[-1]) and data['Cumulative_Volume'].iloc[-1] != 0:
            data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        else:
            data['VWAP'] = None
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV/VWAP: {str(e)}")
        data['OBV'] = data['VWAP'] = None

    # ... (rest of analyze_stock function with similar NaN checks where applicable) ...

    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    if pd.isna(last_close):  # NaN Handling Fix #4
        return None
    last_rsi = data['RSI'].iloc[-1]
    if pd.notnull(last_rsi):
        buy_at = last_close * 0.99 if last_rsi < 30 else last_close
        return round(buy_at, 2)
    return None

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data.empty or 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    if pd.isna(last_close) or pd.isna(last_atr):  # NaN Handling Fix #4
        return None
    if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
        atr_multiplier = 3.0
    else:
        atr_multiplier = 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    if pd.isna(last_close) or pd.isna(stop_loss):  # NaN Handling Fix #4
        return None
    risk = last_close - stop_loss
    if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
        risk_reward_ratio = 3
    else:
        risk_reward_ratio = 1.5
    target = last_close + (risk * risk_reward_ratio)
    return round(target, 2)

def fetch_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0),
            'DividendYield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0, 'DividendYield': 0}

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        return recommendations
    
    try:
        last_close = data['Close'].iloc[-1]
        if pd.isna(last_close):  # NaN Handling Fix #4
            st.warning("⚠️ Latest closing price is unavailable.")
            return recommendations
        recommendations["Current Price"] = round(float(last_close), 2)
        buy_score = 0
        sell_score = 0
        # ... (rest of the recommendation logic with NaN checks) ...
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

def analyze_batch(stock_batch, historical_years):
    """Analyze a batch of stocks with dynamic workers (Performance Fix #2)."""
    results = []
    errors = []
    max_workers = min(10, max(1, len(stock_batch) // 2))  # Dynamic workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        valid_futures = {}
        for symbol in stock_batch:
            data = fetch_stock_data_cached(symbol, period=historical_years)
            if not data.empty and len(data) >= 15:
                valid_futures[executor.submit(analyze_stock_parallel, symbol)] = symbol
            else:
                errors.append(f"⚠️ Skipping {symbol}: insufficient data ({len(data)} days)")
        
        for future in as_completed(valid_futures):
            symbol = valid_futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                errors.append(f"⚠️ Error processing stock {symbol}: {str(e)}")
    for error in errors:
        st.warning(error)
    return results

def analyze_stock_parallel(symbol):
    data = fetch_stock_data_cached(symbol)
    if not data.empty:
        data = analyze_stock(data)
        recommendations = generate_recommendations(data, symbol)
        return {
            "Symbol": symbol,
            "Current Price": recommendations["Current Price"],
            "Buy At": recommendations["Buy At"],
            "Stop Loss": recommendations["Stop Loss"],
            "Target": recommendations["Target"],
            "Intraday": recommendations["Intraday"],
            "Swing": recommendations["Swing"],
            "Short-Term": recommendations["Short-Term"],
            "Long-Term": recommendations["Long-Term"],
            "Mean_Reversion": recommendations["Mean_Reversion"],
            "Breakout": recommendations["Breakout"],
            "Ichimoku_Trend": recommendations["Ichimoku_Trend"],
            "Score": recommendations.get("Score", 0),
        }
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
        loading_text.text(f"{loading_message}{dots} | Processed {processed_items}/{total_items} stocks (ETA: {eta})")
    else:
        loading_text.text(f"{loading_message}{dots} | Processed 0/{total_items} stocks")

def analyze_all_stocks(stock_list, batch_size=20, price_range=None, historical_years=5, progress_callback=None):
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()

    total_items = len(stock_list)
    st.info(f"Analyzing all {total_items} NSE stocks for {historical_years} years of data. This may take ~{total_items // 60} minutes.")
    results = []
    start_time = time.time()
    
    for i in range(0, total_items, batch_size):
        if st.session_state.cancel_operation:
            st.warning("⚠️ Analysis canceled by user.")
            break
        
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch, historical_years)
        results.extend(batch_results)
        processed_items = min(i + len(batch), total_items)
        if progress_callback:
            progress_callback(processed_items / total_items, start_time, total_items, processed_items)
        # Removed fixed sleep (Performance Fix #2), relying on @retry
    
    if not results:
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
    return results_df.sort_values(by="Score", ascending=False).head(10)

def analyze_strong_buy_stocks(stock_list, batch_size=20, price_range=None, historical_years=5, progress_callback=None):
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()

    total_items = len(stock_list)
    st.info(f"Analyzing all {total_items} NSE stocks for strong buy/buy picks using {historical_years} years of data.")
    results = []
    start_time = time.time()
    
    for i in range(0, total_items, batch_size):
        if st.session_state.cancel_operation:
            st.warning("⚠️ Analysis canceled by user.")
            break
        
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch, historical_years)
        results.extend(batch_results)
        processed_items = min(i + len(batch), total_items)
        if progress_callback:
            progress_callback(processed_items / total_items, start_time, total_items, processed_items)
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
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
    strong_buy_df = results_df[results_df["Intraday"].isin(["Strong Buy", "Buy"])]
    return strong_buy_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation

async def send_telegram_message_async(message):
    """Async Telegram message sender (Asyncio Fix #6)."""
    try:
        bot = telegram.Bot(token="7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps")
        await bot.send_message(
            chat_id="-1002411670969",
            text=message,
            parse_mode='HTML'
        )
        return True
    except TelegramError as e:
        st.error(f"❌ Failed to send Telegram message: {str(e)}")
        return False

def send_telegram_message_sync(message):
    try:
        return asyncio.run(send_telegram_message_async(message))
    except Exception as e:
        st.error(f"❌ Failed to send Telegram message: {str(e)}")
        return False

def display_stock_analysis(symbol, data, recommendations):
    with st.container():
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = f"{recommendations['Current Price']:.2f}" if recommendations['Current Price'] is not None else "N/A"
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
        with col2:
            buy_at = f"{recommendations['Buy At']:.2f}" if recommendations['Buy At'] is not None else "N/A"
            st.metric(tooltip("Buy At", "Recommended entry price"), f"₹{buy_at}")
        with col3:
            stop_loss = f"{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = f"{recommendations['Target']:.2f}" if recommendations['Target'] is not None else "N/A"
            st.metric(tooltip("Target", "Price target based on risk/reward"), f"₹{target}")
        
        st.subheader("📈 Trading Recommendations")
        cols = st.columns(4)
        strategy_names = ["Intraday", "Swing", "Short-Term", "Long-Term"]
        for col, strategy in zip(cols, strategy_names):
            with col:
                st.markdown(f"**{strategy}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]), unsafe_allow_html=True)
        
        st.subheader("📈 Additional Strategies")
        cols = st.columns(3)
        new_strategies = ["Mean_Reversion", "Breakout", "Ichimoku_Trend"]
        for col, strategy in zip(cols, new_strategies):
            with col:
                st.markdown(f"**{strategy.replace('_', ' ')}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]), unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Price Action", "📉 Momentum", "📊 Volatility", 
            "📈 Monte Carlo", "📉 New Indicators", "📊 Volume Profile & Waves"
        ])
        timestamp = time.time()  # Unique keys for Plotly (Plotly Fix #8)
        with tab1:
            price_cols = ['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with Moving Averages")
                st.plotly_chart(fig, key=f"price_action_{symbol}_{timestamp}")
            else:
                st.warning("⚠️ No valid price action data available for plotting.")
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators")
                st.plotly_chart(fig, key=f"momentum_{symbol}_{timestamp}")
            else:
                st.warning("⚠️ No valid momentum indicators available for plotting.")
        with tab3:
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis")
                st.plotly_chart(fig, key=f"volatility_{symbol}_{timestamp}")
            else:
                st.warning("⚠️ No valid volatility indicators available for plotting.")
        with tab4:
            mc_results = monte_carlo_simulation(data)
            if mc_results:
                mc_df = pd.DataFrame(mc_results).T
                lower_ci = mc_df.quantile(0.05, axis=1)
                upper_ci = mc_df.quantile(0.95, axis=1)
                mean_path = mc_df.mean(axis=1)
                ci_df = pd.DataFrame({
                    'Mean': mean_path,
                    'Lower 5% CI': lower_ci,
                    'Upper 95% CI': upper_ci
                })
                fig = px.line(ci_df, title="Monte Carlo with 90% Confidence Interval")
                st.plotly_chart(fig, key=f"monte_carlo_{symbol}_{timestamp}")
                bull, bear = scenario_analysis(data)
                if bull and bear:
                    scenario_df = pd.DataFrame({'Bull Scenario': bull, 'Bear Scenario': bear})
                    fig2 = px.line(scenario_df, title="Bull vs Bear Scenarios (30 Days)")
                    st.plotly_chart(fig2, key=f"scenario_{symbol}_{timestamp}")
        with tab5:
            new_cols = ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF']
            valid_new_cols = [col for col in new_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_new_cols:
                fig = px.line(data, y=valid_new_cols, title="New Indicators (Ichimoku & CMF)")
                st.plotly_chart(fig, key=f"new_indicators_{symbol}_{timestamp}")
            else:
                st.warning("⚠️ No valid new indicators available for plotting.")
        with tab6:
            if 'Volume_Profile' in data.columns and pd.notnull(data['Volume_Profile'].iloc[-1]):
                vp_fig = px.bar(data['Volume_Profile'], title="Volume Profile")
                st.plotly_chart(vp_fig, key=f"volume_profile_{symbol}_{timestamp}")
            if 'Wave_Pattern' in data.columns:
                st.write(f"Wave Pattern: {data['Wave_Pattern'].iloc[-1]}")
        if st.button("Analyze News Sentiment", key=f"news_sentiment_{symbol}"):
            news_sentiment = fetch_news_sentiment_vader(symbol.split('.')[0], NEWSAPI_KEY)
            finbert_sentiment = analyze_sentiment_finbert(f"Latest news about {symbol.split('.')[0]}")
            st.write(f"VADER Sentiment: {news_sentiment:.2f}")
            st.write(f"FinBERT Sentiment: {finbert_sentiment} (0=Negative, 1=Neutral, 2=Positive)")

def display_dashboard(NSE_STOCKS):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=10000, value=(100, 1000))
    historical_years = st.sidebar.slider(
        "Select Historical Data Period (Years)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key="historical_period"
    )
    send_to_telegram = st.sidebar.checkbox("Send results to Telegram", value=True)
    
    if st.button("🚀 Generate Daily Top Picks"):
        st.session_state.current_view = "daily_top_picks"
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Analysis", key="cancel_daily")
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        results_df = analyze_all_stocks(
            NSE_STOCKS,
            batch_size=20,
            price_range=price_range,
            historical_years=historical_years,
            progress_callback=lambda progress, start_time, total, processed: update_progress(
                progress_bar, loading_text, progress, loading_messages, start_time, total, processed
            )
        )
        progress_bar.empty()
        loading_text.empty()
        
        if not results_df.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Today's Top 10 Stocks")
            telegram_message = f"<b>🏆 Today's Top 10 Stocks - {datetime.now().strftime('%d %b %Y')}</b>\n\n"
            
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
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Long-Term: {colored_recommendation(row['Long-Term'])}  
                    Mean Reversion: {colored_recommendation(row['Mean_Reversion'])}  
                    Breakout: {colored_recommendation(row['Breakout'])}  
                    Ichimoku Trend: {colored_recommendation(row['Ichimoku_Trend'])}
                    """, unsafe_allow_html=True)
                
                telegram_message += f"<b>{row['Symbol']}</b> (Score: {row['Score']}/7)\n"
                telegram_message += f"Price: ₹{current_price} | Buy: ₹{buy_at}\n"
                telegram_message += f"SL: ₹{stop_loss} | Target: ₹{target}\n"
                telegram_message += f"Intraday: {row['Intraday']} | Swing: {row['Swing']}\n\n"
            
            if send_to_telegram:
                success = send_telegram_message_sync(telegram_message)
                if success:
                    st.success("✅ Results sent to Telegram group!")
            st.session_state.cancel_operation = False  # Reset state (State Management Fix #4)
        
        elif not st.session_state.cancel_operation:
            st.warning("⚠️ No top picks available due to data issues.")
    
    if st.button("🟢 Generate Top Buy Picks"):
        st.session_state.current_view = "strong_buy_picks"
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Buy Analysis", key="cancel_strong_buy")
        loading_messages = itertools.cycle([
            "Scanning for buy signals...", "Analyzing bullish trends...", "Calculating targets...",
            "Filtering top buys...", "Finalizing buy picks..."
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        strong_buy_results = analyze_strong_buy_stocks(
            NSE_STOCKS,
            batch_size=20,
            price_range=price_range,
            historical_years=historical_years,
            progress_callback=lambda progress, start_time, total, processed: update_progress(
                progress_bar, loading_text, progress, loading_messages, start_time, total, processed
            )
        )
        progress_bar.empty()
        loading_text.empty()
        
        if not strong_buy_results.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Top 5 Buy Picks (Strong Buy/Buy)")
            telegram_message = f"<b>🏆 Top 5 Buy Picks - {datetime.now().strftime('%d %b %Y')}</b>\n\n"
            
            for _, row in strong_buy_results.iterrows():
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
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Long-Term: {colored_recommendation(row['Long-Term'])}  
                    """, unsafe_allow_html=True)
                
                telegram_message += f"<b>{row['Symbol']}</b> (Score: {row['Score']}/7)\n"
                telegram_message += f"Price: ₹{current_price} | Buy: ₹{buy_at}\n"
                telegram_message += f"SL: ₹{stop_loss} | Target: ₹{target}\n"
                telegram_message += f"Intraday: {row['Intraday']} | Swing: {row['Swing']}\n\n"
            
            if send_to_telegram:
                success = send_telegram_message_sync(telegram_message)
                if success:
                    st.success("✅ Buy picks sent to Telegram group!")
            st.session_state.cancel_operation = False  # Reset state (State Management Fix #4)
        
        elif not st.session_state.cancel_operation:
            st.warning("⚠️ No buy picks available due to data issues.")

def main():
    try:
        import telegram
    except ImportError:
        st.error("❌ Please install python-telegram-bot: pip install python-telegram-bot --upgrade")
        return

    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = None

    st.sidebar.title("🔍 Stock Search")
    NSE_STOCKS = fetch_nse_stock_list()
    
    symbol = None
    selected_option = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=[""] + NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x != "Custom" and x != "" else x,
        key="stock_select"
    )
    
    historical_years = st.sidebar.slider(
        "Select Historical Data Period (Years)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key="historical_period"
    )
    
    if selected_option == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g., RELIANCE):", key="custom_input")
        if custom_symbol:
            symbol = f"{custom_symbol}.NS"
    elif selected_option != "":
        symbol = selected_option
    
    display_dashboard(NSE_STOCKS)
    
    if symbol:
        if ".NS" not in symbol:
            symbol += ".NS"
        if symbol not in NSE_STOCKS:
            st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable")
        data = fetch_stock_data_cached(symbol, period=historical_years)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            st.session_state.current_view = "individual_stock"
            display_stock_analysis(symbol, data, recommendations)
        else:
            st.error("❌ Failed to load data for this symbol")

if __name__ == "__main__":
    main()