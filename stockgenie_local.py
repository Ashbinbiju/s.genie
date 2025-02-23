import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
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

# Global cache for precomputed indicators
indicator_cache = {}

# Global FinBERT model initialization
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# API Keys (Consider moving to environment variables)
NEWSAPI_KEY = "ed58659895e84dfb8162a8bb47d8525e"
GNEWS_KEY = "e4f5f1442641400694645433a8f98b94"
ALPHA_VANTAGE_KEY = "TCAUKYUCIDZ6PI57"

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
                        raise e
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
    except Exception:
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS",
            "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS",
            "AATMAJ.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS",
            "ABFRL.NS",
        ]

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None  # Return None instead of empty DataFrame
        return data
    except Exception:
        return None

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if data is not None and not data.empty:
            if pd.notnull(data['Close'].iloc[-1]) and pd.notnull(data['Close'].iloc[-2]):
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    advances += 1
                else:
                    declines += 1
    return advances / declines if declines != 0 else 0

def adjust_window(data, default_window, min_window=5):
    return min(default_window, max(min_window, len(data) - 1))

def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 30:
        st.warning(f"⚠️ Limited data ({len(returns)} days); using basic simulation.")
        last_price = data['Close'].iloc[-1]
        mean_return = returns.mean()
        std_return = returns.std() or 0.01
        # Use NumPy for vectorized simulation
        price_paths = np.zeros((simulations, days + 1))
        price_paths[:, 0] = last_price
        random_returns = np.random.normal(mean_return, std_return, (simulations, days))
        price_paths[:, 1:] = last_price * np.cumprod(1 + random_returns, axis=1)
        return price_paths.T.tolist()  # Convert to list for compatibility
    
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

def analyze_sentiment_finbert(text):
    try:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.argmax().item()
    except Exception:
        return 1

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
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

def precompute_indicators(data):
    if data is None or data.empty:
        return None
    
    indicators = {}
    try:
        rsi_window = adjust_window(data, optimize_rsi_window(data), min_window=5)
        indicators['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        indicators['RSI'] = None
    
    try:
        macd_window_slow = adjust_window(data, 17, min_window=5)
        macd_window_fast = adjust_window(data, 8, min_window=3)
        macd_window_sign = adjust_window(data, 9, min_window=3)
        macd = ta.trend.MACD(data['Close'], window_slow=macd_window_slow, window_fast=macd_window_fast, window_sign=macd_window_sign)
        indicators['MACD'] = macd.macd()
        indicators['MACD_signal'] = macd.macd_signal()
        indicators['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
        indicators['MACD'] = None
        indicators['MACD_signal'] = None
        indicators['MACD_hist'] = None
    
    try:
        sma_50_window = adjust_window(data, 50, min_window=10)
        sma_200_window = adjust_window(data, 200, min_window=20)
        indicators['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=sma_50_window).sma_indicator()
        if len(data) >= 200:
            indicators['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=sma_200_window).sma_indicator()
        else:
            indicators['SMA_200'] = None
        indicators['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=adjust_window(data, 20, min_window=5)).ema_indicator()
        indicators['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=sma_50_window).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        indicators['SMA_50'] = None
        indicators['SMA_200'] = None
        indicators['EMA_20'] = None
        indicators['EMA_50'] = None
    
    try:
        bollinger_window = adjust_window(data, 20, min_window=5)
        bollinger = ta.volatility.BollingerBands(data['Close'], window=bollinger_window, window_dev=2)
        indicators['Upper_Band'] = bollinger.bollinger_hband()
        indicators['Middle_Band'] = bollinger.bollinger_mavg()
        indicators['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Bollinger Bands: {str(e)}")
        indicators['Upper_Band'] = None
        indicators['Middle_Band'] = None
        indicators['Lower_Band'] = None
    
    try:
        stoch_window = adjust_window(data, 14, min_window=5)
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=stoch_window, smooth_window=3)
        indicators['SlowK'] = stoch.stoch()
        indicators['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Stochastic: {str(e)}")
        indicators['SlowK'] = None
        indicators['SlowD'] = None
    
    try:
        atr_window = adjust_window(data, 14, min_window=5)
        indicators['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=atr_window).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        indicators['ATR'] = None
    
    try:
        adx_window = adjust_window(data, 14, min_window=5)
        if len(data) >= adx_window + 1:
            indicators['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=adx_window).adx()
        else:
            indicators['ADX'] = None
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ADX: {str(e)}")
        indicators['ADX'] = None
    
    try:
        indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV: {str(e)}")
        indicators['OBV'] = None
    
    try:
        indicators['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        indicators['Cumulative_Volume'] = data['Volume'].cumsum()
        indicators['VWAP'] = indicators['Cumulative_TP'].cumsum() / indicators['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute VWAP: {str(e)}")
        indicators['VWAP'] = None
    
    try:
        volume_window = adjust_window(data, 10, min_window=5)
        indicators['Avg_Volume'] = data['Volume'].rolling(window=volume_window).mean()
        indicators['Volume_Spike'] = data['Volume'] > (indicators['Avg_Volume'] * 1.5)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Spike: {str(e)}")
        indicators['Volume_Spike'] = None
    
    try:
        indicators['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Parabolic SAR: {str(e)}")
        indicators['Parabolic_SAR'] = None
    
    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        indicators['Fib_23.6'] = high - diff * 0.236
        indicators['Fib_38.2'] = high - diff * 0.382
        indicators['Fib_50.0'] = high - diff * 0.5
        indicators['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Fibonacci: {str(e)}")
        indicators['Fib_23.6'] = None
        indicators['Fib_38.2'] = None
        indicators['Fib_50.0'] = None
        indicators['Fib_61.8'] = None
    
    try:
        indicators['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Divergence: {str(e)}")
        indicators['Divergence'] = "No Divergence"
    
    try:
        ichimoku_w1 = adjust_window(data, 9, min_window=3)
        ichimoku_w2 = adjust_window(data, 26, min_window=5)
        ichimoku_w3 = adjust_window(data, 52, min_window=10)
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=ichimoku_w1, window2=ichimoku_w2, window3=ichimoku_w3)
        indicators['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        indicators['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        indicators['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        indicators['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        indicators['Ichimoku_Chikou'] = data['Close'].shift(-min(26, len(data) - 1))
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        indicators['Ichimoku_Tenkan'] = None
        indicators['Ichimoku_Kijun'] = None
        indicators['Ichimoku_Span_A'] = None
        indicators['Ichimoku_Span_B'] = None
        indicators['Ichimoku_Chikou'] = None
    
    try:
        cmf_window = adjust_window(data, 20, min_window=5)
        indicators['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=cmf_window).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        indicators['CMF'] = None
    
    try:
        donchian_window = adjust_window(data, 20, min_window=5)
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=donchian_window)
        indicators['Donchian_Upper'] = donchian.donchian_channel_hband()
        indicators['Donchian_Lower'] = donchian.donchian_channel_lband()
        indicators['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        indicators['Donchian_Upper'] = None
        indicators['Donchian_Lower'] = None
        indicators['Donchian_Middle'] = None
    
    try:
        indicators['Volume_Profile'] = calculate_volume_profile(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Profile: {str(e)}")
        indicators['Volume_Profile'] = None
    
    try:
        indicators['Wave_Pattern'] = detect_waves(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Wave Pattern: {str(e)}")
        indicators['Wave_Pattern'] = "No Clear Wave Pattern"
    
    return indicators

def analyze_stock(data):
    if data is None or data.empty:
        st.warning("⚠️ No data available to analyze.")
        return None
    
    global indicator_cache
    if data.to_json() in indicator_cache:
        return pd.concat([data, pd.DataFrame(indicator_cache[data.to_json()])], axis=1)
    
    indicators = precompute_indicators(data)
    if indicators is None:
        return data
    
    result = pd.concat([data, pd.DataFrame(indicators)], axis=1)
    indicator_cache[data.to_json()] = indicators
    return result

def calculate_buy_at(data):
    if data is None or data.empty or 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_rsi):
        buy_at = last_close * 0.99 if last_rsi < 30 else last_close
        return round(buy_at, 2)
    return None

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data is None or data.empty or 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_atr):
        if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
            atr_multiplier = 3.0
        else:
            atr_multiplier = 1.5
        stop_loss = last_close - (atr_multiplier * last_atr)
        return round(stop_loss, 2)
    return None

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(stop_loss):
        risk = last_close - stop_loss
        if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
            risk_reward_ratio = 3
        else:
            risk_reward_ratio = 1.5
        target = last_close + (risk * risk_reward_ratio)
        return round(target, 2)
    return None

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
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }
    if data is None or data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        st.warning("⚠️ No valid data available for recommendations.")
        return recommendations
    
    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2) if pd.notnull(data['Close'].iloc[-1]) else None
        buy_score = 0
        sell_score = 0
        
        # Existing checks remain, but add Ichimoku, Volume Profile, and Wave Pattern
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and pd.notnull(data['Ichimoku_Span_A'].iloc[-1]) and pd.notnull(data['Ichimoku_Span_B'].iloc[-1]) and pd.notnull(data['Close'].iloc[-1]):
            if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                buy_score += 1
                recommendations["Ichimoku_Trend"] = "Strong Buy"
            elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                sell_score += 1
                recommendations["Ichimoku_Trend"] = "Strong Sell"

        if 'Volume_Profile' in data.columns and pd.notnull(data['Volume_Profile'].iloc[-1]):
            vp = data['Volume_Profile']
            last_close = data['Close'].iloc[-1]
            if pd.notnull(last_close):
                max_vp_price = vp.idxmax()
                if pd.notnull(max_vp_price) and abs(last_close - max_vp_price) < last_close * 0.02:
                    buy_score += 0.5  # Support/resistance confirmation

        if 'Wave_Pattern' in data.columns:
            if data['Wave_Pattern'].iloc[-1] == "Potential Uptrend (Wave 5?)":
                buy_score += 0.5
            elif data['Wave_Pattern'].iloc[-1] == "Potential Downtrend (Wave C?)":
                sell_score += 0.5

        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
            if data['RSI'].iloc[-1] < 30:
                buy_score += 2
            elif data['RSI'].iloc[-1] > 70:
                sell_score += 2
        else:
            if 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]) and pd.notnull(data['Close'].iloc[0]):
                if data['Close'].iloc[-1] > data['Close'].iloc[0]:
                    buy_score += 1
                else:
                    sell_score += 1

        if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                buy_score += 1
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                sell_score += 1
        
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            if pd.notnull(data['Close'].iloc[-1]) and pd.notnull(data['Lower_Band'].iloc[-1]) and pd.notnull(data['Upper_Band'].iloc[-1]):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1
        
        if 'VWAP' in data.columns and pd.notnull(data['VWAP'].iloc[-1]) and pd.notnull(data['Close'].iloc[-1]):
            if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                sell_score += 1
        
        if 'Volume' in data.columns and pd.notnull(data['Volume'].iloc[-1]):
            avg_volume = data['Volume'].rolling(window=min(10, len(data) - 1)).mean().iloc[-1]
            if pd.notnull(avg_volume):
                if data['Volume'].iloc[-1] > avg_volume * 1.5:
                    buy_score += 1
                elif data['Volume'].iloc[-1] < avg_volume * 0.5:
                    sell_score += 1
        
        if 'Divergence' in data.columns:
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1
        
        if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
            if data['CMF'].iloc[-1] > 0:
                buy_score += 1
            elif data['CMF'].iloc[-1] < 0:
                sell_score += 1
        
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
            if (pd.notnull(data['Donchian_Upper'].iloc[-1]) and 
                pd.notnull(data['Donchian_Lower'].iloc[-1]) and 
                pd.notnull(data['Close'].iloc[-1])):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"

        if 'RSI' in data.columns and 'Lower_Band' in data.columns:
            if (pd.notnull(data['RSI'].iloc[-1]) and 
                pd.notnull(data['Lower_Band'].iloc[-1]) and 
                pd.notnull(data['Close'].iloc[-1])):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"
        
        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns:
            if (pd.notnull(data['Ichimoku_Tenkan'].iloc[-1]) and 
                pd.notnull(data['Ichimoku_Kijun'].iloc[-1]) and 
                pd.notnull(data['Close'].iloc[-1]) and 
                pd.notnull(data['Ichimoku_Span_A'].iloc[-1])):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"

        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if pd.notnull(fundamentals['P/E']) and fundamentals['P/E'] < 15 and pd.notnull(fundamentals['EPS']) and fundamentals['EPS'] > 0:
                buy_score += 1
            if pd.notnull(fundamentals['RevenueGrowth']) and fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 0.5
            if pd.notnull(fundamentals['DividendYield']) and fundamentals['DividendYield'] > 2:
                buy_score += 0.5
                recommendations["Long-Term"] = "Buy" if buy_score > sell_score else "Hold"
        
        if buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
        
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

def analyze_batch(stock_batch):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
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
    if data is None:
        return None
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
        estimated_remaining_seconds = time_per_item * remaining_items
        eta = timedelta(seconds=int(estimated_remaining_seconds))
        loading_text.text(f"{loading_message}{dots} | Processed {processed_items}/{total_items} stocks (ETA: {eta})")
    else:
        loading_text.text(f"{loading_message}{dots} | Processed 0/{total_items} stocks")

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

def analyze_intraday_stocks(stock_list, batch_size=50, price_range=None, progress_callback=None):
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
    intraday_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    return intraday_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=10000, value=(100, 1000))
    
    if st.button("🚀 Generate Daily Top Picks"):
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Analysis")
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
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
            st.subheader("🏆 Today's Top 10 Stocks")
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
        elif not st.session_state.cancel_operation:
            st.warning("⚠️ No top picks available due to data issues.")
    
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Intraday Analysis")
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        intraday_results = analyze_intraday_stocks(
            NSE_STOCKS,
            price_range=price_range,
            progress_callback=lambda progress, start_time, total, processed: update_progress(
                progress_bar, loading_text, progress, loading_messages, start_time, total, processed
            )
        )
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
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
                    """, unsafe_allow_html=True)
        elif not st.session_state.cancel_operation:
            st.warning("⚠️ No intraday picks available due to data issues.")
    
    if symbol and data is not None and recommendations is not None:
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
        with tab1:
            price_cols = ['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with Moving Averages")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid price action data available for plotting.")
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid momentum indicators available for plotting.")
        with tab3:
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid volatility indicators available for plotting.")
        with tab4:
            mc_results = monte_carlo_simulation(data)
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
            st.plotly_chart(fig)
            bull, bear = scenario_analysis(data)
            scenario_df = pd.DataFrame({'Bull Scenario': bull, 'Bear Scenario': bear})
            fig2 = px.line(scenario_df, title="Bull vs Bear Scenarios (30 Days)")
            st.plotly_chart(fig2)
        with tab5:
            new_cols = ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF']
            valid_new_cols = [col for col in new_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_new_cols:
                fig = px.line(data, y=valid_new_cols, title="New Indicators (Ichimoku & CMF)")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid new indicators available for plotting.")
        with tab6:
            if 'Volume_Profile' in data.columns and pd.notnull(data['Volume_Profile'].iloc[-1]):
                vp_fig = px.bar(data['Volume_Profile'], title="Volume Profile")
                st.plotly_chart(vp_fig)
            if 'Wave_Pattern' in data.columns:
                st.write(f"Wave Pattern: {data['Wave_Pattern'].iloc[-1]}")
        if st.button("Analyze News Sentiment"):
            news_sentiment = fetch_news_sentiment_vader(symbol.split('.')[0], NEWSAPI_KEY)
            finbert_sentiment = analyze_sentiment_finbert(f"Latest news about {symbol.split('.')[0]}")
            st.write(f"VADER Sentiment: {news_sentiment:.2f}")
            st.write(f"FinBERT Sentiment: {finbert_sentiment} (0=Negative, 1=Neutral, 2=Positive)")
    elif symbol:
        st.warning("⚠️ No data available for the selected stock.")

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
        data = fetch_stock_data_cached(symbol)
        if data is not None and not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error("❌ Failed to load data for this symbol")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()