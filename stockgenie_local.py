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
    "Elliott_Wave": "Simplified wave pattern detection",
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
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception:
        return pd.DataFrame()

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    return advances / declines if declines != 0 else 0

def adjust_window(data, default_window, min_window=5):
    return min(default_window, max(min_window, len(data) - 1))

def monte_carlo_scenarios(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 30:
        st.warning(f"⚠️ Limited data ({len(returns)} days) for GARCH; using basic simulation.")
        mean_return = returns.mean()
        std_return = returns.std() if returns.std() != 0 else 0.01
        scenarios = {"Bull": mean_return * 1.5, "Bear": mean_return * -1.5, "Base": mean_return}
        results = {}
        for scenario, adj_mean in scenarios.items():
            sim_results = []
            for _ in range(simulations):
                prices = [data['Close'].iloc[-1]]
                for _ in range(days):
                    prices.append(prices[-1] * (1 + np.random.normal(adj_mean, std_return)))
                sim_results.append(prices)
            results[scenario] = sim_results
        return results, None
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal')
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    scenarios = {"Bull": mean_return * 1.5, "Bear": mean_return * -1.5, "Base": mean_return}
    results = {}
    for scenario, adj_mean in scenarios.items():
        sim_results = []
        for _ in range(simulations):
            prices = [data['Close'].iloc[-1]]
            for i in range(days):
                prices.append(prices[-1] * (1 + np.random.normal(adj_mean, volatility[i])))
            sim_results.append(prices)
        results[scenario] = sim_results
    # Calculate confidence intervals for the Base scenario
    base_df = pd.DataFrame(results["Base"]).T
    confidence_intervals = base_df.quantile([0.05, 0.95], axis=1).T
    return results, confidence_intervals

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

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def check_data_sufficiency(data):
    days = len(data)
    st.info(f"📅 Data available: {days} days")
    if days < 20:
        st.warning("⚠️ Less than 20 days; most indicators unavailable or unreliable.")
    elif days < 50:
        st.warning("⚠️ Less than 50 days; some indicators (e.g., SMA_50, Ichimoku) may be limited.")
    elif days < 200:
        st.warning("⚠️ Less than 200 days; long-term indicators like SMA_200 unavailable.")

def calculate_volume_profile(data, bins=20):
    price_bins = pd.cut(data['Close'], bins=bins)
    volume_profile = data.groupby(price_bins)['Volume'].sum()
    return volume_profile

def detect_waves(data):
    peaks, _ = find_peaks(data['Close'], distance=5)
    troughs, _ = find_peaks(-data['Close'], distance=5)
    if len(peaks) >= 2 and len(troughs) >= 1:
        last_peak = peaks[-1]
        last_trough = troughs[-1] if troughs[-1] < peaks[-1] else troughs[-2] if len(troughs) > 1 else None
        if last_trough and data['Close'].iloc[-1] > data['Close'].iloc[last_trough]:
            return "Uptrend (Potential Wave 3)"
    return "No Clear Wave Pattern"

def analyze_stock(data):
    if data.empty:
        st.warning("⚠️ No data available to analyze.")
        return data
    
    check_data_sufficiency(data)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    try:
        rsi_window = adjust_window(data, 14, min_window=5)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = None
    
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None
    
    try:
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        if len(data) >= 200:
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        else:
            data['SMA_200'] = None
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
    
    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None
    
    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        data['ATR'] = None
    
    try:
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ADX: {str(e)}")
        data['ADX'] = None
    
    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV: {str(e)}")
        data['OBV'] = None
    
    try:
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None
    
    try:
        data['Volume_Profile'] = calculate_volume_profile(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Profile: {str(e)}")
        data['Volume_Profile'] = None
    
    try:
        data['Elliott_Wave'] = detect_waves(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Elliott Wave: {str(e)}")
        data['Elliott_Wave'] = "No Clear Wave Pattern"
    
    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None
    
    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - diff * 0.236
        data['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = None
        data['Fib_61.8'] = None
    
    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Span_A'] = None
        data['Ichimoku_Span_B'] = None
    
    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        data['CMF'] = None
    
    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
    
    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, strategy="ATR", atr_multiplier=2.5, percent=2.0):
    if data.empty or 'Close' not in data.columns:
        return None
    last_close = data['Close'].iloc[-1]
    if strategy == "ATR" and 'ATR' in data.columns and data['ATR'].iloc[-1] is not None:
        multiplier = 3.0 if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25 else atr_multiplier
        return round(last_close - (multiplier * data['ATR'].iloc[-1]), 2)
    elif strategy == "Percentage":
        return round(last_close * (1 - percent / 100), 2)
    elif strategy == "Support" and 'Lower_Band' in data.columns and data['Lower_Band'].iloc[-1] is not None:
        return round(data['Lower_Band'].iloc[-1], 2)
    return None

def calculate_target(data, risk_reward_ratio=3.0):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    target = last_close + (risk * risk_reward_ratio)
    return round(target, 2)

def calculate_risk_score(data):
    if data.empty or 'Close' not in data.columns:
        return 0
    atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1] if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None else 0
    drawdown = (data['Close'].max() - data['Close'].min()) / data['Close'].max()
    risk_score = (atr_volatility * 50 + drawdown * 50)
    return min(round(risk_score * 100, 2), 100)

def fetch_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0),
            'DividendYield': info.get('dividendYield', 0),
            'DebtToEquity': info.get('debtToEquity', 0)
        }
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0, 'DividendYield': 0, 'DebtToEquity': 0}

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None,
        "Score": 0, "Risk Score": 0
    }
    if data.empty or 'Close' not in data.columns:
        return recommendations
    
    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        buy_score = 0
        sell_score = 0
        
        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
            if data['RSI'].iloc[-1] < 30:
                buy_score += 2
            elif data['RSI'].iloc[-1] > 70:
                sell_score += 2
        
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                buy_score += 1
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                sell_score += 1
        
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                sell_score += 1
        
        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None:
            if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                sell_score += 1
        
        if 'Volume_Profile' in data.columns and data['Volume_Profile'] is not None:
            vp = data['Volume_Profile']
            if isinstance(vp, pd.Series) and not vp.empty:
                max_vp_price = vp.idxmax().mid if not pd.isna(vp.idxmax()) else data['Close'].iloc[-1]
                if abs(data['Close'].iloc[-1] - max_vp_price) < data['ATR'].iloc[-1]:
                    buy_score += 1  # Near high volume node = support
        
        if 'Elliott_Wave' in data.columns:
            if data['Elliott_Wave'] == "Uptrend (Potential Wave 3)":
                buy_score += 1
        
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
            if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                buy_score += 1
                recommendations["Ichimoku_Trend"] = "Buy"
            elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                sell_score += 1
                recommendations["Ichimoku_Trend"] = "Sell"
        
        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
            if data['CMF'].iloc[-1] > 0:
                buy_score += 1
            elif data['CMF'].iloc[-1] < 0:
                sell_score += 1
        
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
            if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                buy_score += 1
                recommendations["Breakout"] = "Buy"
            elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                sell_score += 1
                recommendations["Breakout"] = "Sell"
        
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 0.5
            if fundamentals['DividendYield'] > 0.03:
                buy_score += 0.5
            if fundamentals['DebtToEquity'] < 1:
                buy_score += 0.5
        
        if buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
        
        stop_loss_strategy = st.session_state.get('stop_loss_strategy', 'ATR')
        risk_reward_ratio = st.session_state.get('risk_reward_ratio', 3.0)
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data, strategy=stop_loss_strategy)
        recommendations["Target"] = calculate_target(data, risk_reward_ratio=risk_reward_ratio)
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
        recommendations["Risk Score"] = calculate_risk_score(data)
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

def analyze_batch(stock_batch):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Error processing stock {symbol}: {str(e)}")
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
            "Risk Score": recommendations.get("Risk Score", 0),
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
        loading_text.text(f"{loading_message}{dots} (Processed {processed_items}/{total_items} stocks, ETA: {eta})")
    else:
        loading_text.text(f"{loading_message}{dots} (Processed 0/{total_items} stocks)")

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
    
    st.sidebar.header("⚙️ Risk Management Settings")
    st.session_state['risk_reward_ratio'] = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 3.0)
    st.session_state['stop_loss_strategy'] = st.sidebar.selectbox("Stop Loss Strategy", ["ATR", "Percentage", "Support"])
    
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
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7 (Risk: {row['Risk Score']}%)"):
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
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7 (Risk: {row['Risk Score']}%)"):
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
        st.subheader(f"📊 Risk Score: {recommendations['Risk Score']}%")
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Action", "📉 Momentum", "📊 Volatility", "📈 Monte Carlo"])
        with tab1:
            price_cols = ['Close', 'SMA_50', 'SMA_200', 'EMA_20']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with Moving Averages")
                st.plotly_chart(fig)
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators")
                st.plotly_chart(fig)
        with tab3:
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis")
                st.plotly_chart(fig)
        with tab4:
            mc_results, ci = monte_carlo_scenarios(data)
            for scenario, sim_results in mc_results.items():
                mc_df = pd.DataFrame(sim_results).T
                fig = px.line(mc_df.mean(axis=1), title=f"Monte Carlo - {scenario}")
                if scenario == "Base" and ci is not None:
                    fig.add_scatter(y=ci[0.05], mode='lines', name='5% CI', line=dict(dash='dash'))
                    fig.add_scatter(y=ci[0.95], mode='lines', name='95% CI', line=dict(dash='dash'))
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
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error("❌ Failed to load data for this symbol")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()