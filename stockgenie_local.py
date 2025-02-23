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
from diskcache import Cache  # Advanced caching
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setup diskcache
cache = Cache("stock_cache", expire=86400)  # Cache directory, expires after 24 hours

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
}

def tooltip(label, explanation):
    """Add tooltip with explanation to label."""
    return f"{label} 📌 ({explanation})"

def retry(max_retries=5, delay=2, backoff_factor=1.5, jitter=0.5):
    """Retry decorator for handling transient failures with more robustness."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ConnectionError, ValueError, TimeoutError) as e:
                    retries += 1
                    if retries == max_retries:
                        st.error(f"❌ Maximum retries reached for {func.__name__}: {str(e)}")
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    st.warning(f"⚠️ Retrying {func.__name__} (Attempt {retries}/{max_retries}) in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

def check_internet_connection():
    """Check if there’s an active internet connection."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except (requests.exceptions.RequestException, TimeoutError):
        return False

@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    """Fetch list of NSE stock symbols from the official CSV, cached to disk."""
    cache_key = "nse_stock_list"
    if cache_key in cache:
        return cache[cache_key]
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        cache[cache_key] = stock_list
        return stock_list
    except Exception as e:
        st.error(f"❌ Failed to fetch NSE stock list: {str(e)}")
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS",
            "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS",
            "AATMAJ.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS",
            "ABFRL.NS",
        ]

def validate_stock_data(data, symbol):
    """Validate stock data for completeness and accuracy."""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"❌ Data for {symbol} missing columns: {', '.join(missing_cols)}")
        return False
    if data[required_cols].isnull().any().any():
        st.warning(f"⚠️ Data for {symbol} contains NaN values.")
        return False
    if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
        st.error(f"❌ Data for {symbol} contains negative prices.")
        return False
    if data.index.duplicated().any():
        st.warning(f"⚠️ Data for {symbol} contains duplicate dates.")
        return False
    return True

@retry(max_retries=5, delay=2)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    """Fetch historical stock data from Yahoo Finance with disk caching and connection check."""
    if ".NS" not in symbol:
        symbol += ".NS"
    cache_key = f"{symbol}_{period}_{interval}"
    if cache_key in cache:
        data = cache[cache_key]
        if validate_stock_data(data, symbol):
            return data
    
    # Check internet connection before proceeding
    if not check_internet_connection():
        st.error("❌ No internet connection detected. Please check your network and try again.")
        return pd.DataFrame()
    
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        if validate_stock_data(data, symbol):
            cache[cache_key] = data
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to fetch data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_advance_decline_ratio(stock_list):
    """Calculate the advance/decline ratio for a list of stocks."""
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 2:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    ratio = advances / declines if declines != 0 else 0
    assert ratio >= 0, "Advance/decline ratio should be non-negative"
    return ratio

def adjust_window(data, default_window, min_window=5):
    """Adjust indicator window size based on available data."""
    return min(default_window, max(min_window, len(data) - 1))

def monte_carlo_simulation(data, simulations=1000, days=30):
    """Simulate future stock prices using Monte Carlo with GARCH or basic model."""
    returns = data['Close'].pct_change().dropna()
    returns = returns[(returns > returns.quantile(0.05)) & (returns < returns.quantile(0.95))]
    if len(returns) < 30:
        st.warning(f"⚠️ Limited data ({len(returns)} days) for GARCH; using basic simulation.")
        mean_return = returns.mean()
        std_return = returns.std() if returns.std() != 0 else 0.01
        if std_return > 0.1:
            st.warning("⚠️ High volatility detected; simulation may be unreliable.")
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal')
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    if volatility.mean() > 0.1:
        st.warning("⚠️ High volatility in GARCH model; results may be unreliable.")
    mean_return = returns.mean()
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            price = price_series[-1] * (1 + np.random.normal(mean_return, volatility[i]))
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

def fetch_news_sentiment_vader(query, api_key, source="newsapi"):
    """Fetch and analyze news sentiment using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    try:
        if source == "newsapi":
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey=ed58659895e84dfb8162a8bb47d8525e&language=en&sortBy=publishedAt&pageSize=5"
        elif source == "gnews":
            url = f"https://gnews.io/api/v4/search?q={query}&token=e4f5f1442641400694645433a8f98b94&lang=en&max=5"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            st.warning(f"⚠️ No news articles found for {query}; assuming neutral sentiment.")
            return 0
        sentiment_scores = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment_scores.append(sentiment)
        return sum(sentiment_scores) / len(sentiment_scores)
    except Exception as e:
        st.error(f"❌ Failed to fetch news sentiment for {query}: {str(e)}")
        return 0

def analyze_sentiment_finbert(text):
    """Analyze sentiment using FinBERT."""
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    try:
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.argmax().item()
    except Exception as e:
        st.error(f"❌ Failed to analyze sentiment with FinBERT: {str(e)}")
        return 1

def extract_entities(text):
    """Extract organization entities from text."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def get_trending_stocks():
    """Get trending stocks from Google Trends."""
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def create_sentiment_heatmap(sentiment_data):
    """Create a heatmap of sentiment data."""
    fig = px.imshow(sentiment_data, labels=dict(x="Stocks", y="Sentiment", color="Sentiment Score"),
                    x=sentiment_data.columns, y=sentiment_data.index)
    st.plotly_chart(fig)

def calculate_confidence_score(data):
    """Calculate confidence score based on indicators."""
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    return score / 3

def assess_risk(data):
    """Assess risk based on ATR."""
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 15)):
    """Optimize RSI window based on Sharpe ratio."""
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
    """Detect RSI-price divergence."""
    if 'RSI' not in data.columns or data['RSI'].iloc[-1] is None or len(data) < 5:
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

def check_data_sufficiency(data):
    """Check if data is sufficient and regular for analysis."""
    days = len(data)
    st.info(f"📅 Data available: {days} days")
    if days < 20:
        st.warning("⚠️ Less than 20 days; most indicators unavailable or unreliable.")
    elif days < 50:
        st.warning("⚠️ Less than 50 days; some indicators (e.g., SMA_50, Ichimoku) limited.")
    elif days < 200:
        st.warning("⚠️ Less than 200 days; long-term indicators like SMA_200 unavailable.")
    if not data.empty and data.index.to_series().diff().max() > pd.Timedelta(days=2):
        st.warning("⚠️ Irregular data intervals detected; results may be unreliable.")

def prepare_ml_data(data):
    """Prepare data for machine learning by extracting features and labels."""
    if data.empty or len(data) < 2:
        return None, None
    
    # Define features (technical indicators)
    features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'SlowK', 'SlowD', 'OBV', 'VWAP', 'CMF']
    X = data[features].dropna()
    
    # Create labels based on price movement (1 for increase, 0 for decrease/stable)
    price_changes = data['Close'].pct_change().shift(-1).dropna()
    y = (price_changes > 0).astype(int)  # 1 if price increases, 0 if decreases or stays stable
    
    # Align features and labels
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    return X, y

def train_ml_model(X, y):
    """Train a Random Forest Classifier for stock price prediction."""
    if X is None or y is None or X.empty or y.empty:
        st.warning("⚠️ Insufficient data for machine learning training.")
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model (optional, for debugging)
    accuracy = model.score(X_test_scaled, y_test)
    st.info(f"Machine Learning Model Accuracy: {accuracy:.2f}")
    
    return model, scaler

def predict_stock_trend(data, model, scaler):
    """Predict stock trend using the trained machine learning model."""
    if data.empty or model is None or scaler is None:
        return "Hold"
    
    # Prepare features for prediction
    features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'SlowK', 'SlowD', 'OBV', 'VWAP', 'CMF']
    X = data[features].dropna().tail(1)
    
    if X.empty:
        return "Hold"
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict (1 = Buy/Increase, 0 = Sell/Decrease)
    prediction = model.predict(X_scaled)[0]
    return "Buy" if prediction == 1 else "Sell"

def analyze_stock(data):
    """Analyze stock data with technical indicators and prepare for machine learning."""
    if data.empty:
        st.warning("⚠️ No data available to analyze.")
        return data
    
    check_data_sufficiency(data)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    try:
        rsi_window = adjust_window(data, optimize_rsi_window(data), min_window=5)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.error(f"❌ Failed to compute RSI: {str(e)}")
        data['RSI'] = None
    
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=adjust_window(data, 17), window_fast=adjust_window(data, 8), window_sign=adjust_window(data, 9))
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.error(f"❌ Failed to compute MACD: {str(e)}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None
    
    try:
        sma_50_window = adjust_window(data, 50, min_window=10)
        sma_200_window = adjust_window(data, 200, min_window=20)
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=sma_50_window).sma_indicator()
        if len(data) >= 200:
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=sma_200_window).sma_indicator()
        else:
            data['SMA_200'] = None
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=adjust_window(data, 20)).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=sma_50_window).ema_indicator()
    except Exception as e:
        st.error(f"❌ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None
    
    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=adjust_window(data, 20), window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.error(f"❌ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None
    
    try:
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=adjust_window(data, 14), smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.error(f"❌ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = None
        data['SlowD'] = None
    
    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=adjust_window(data, 14)).average_true_range()
    except Exception as e:
        st.error(f"❌ Failed to compute ATR: {str(e)}")
        data['ATR'] = None
    
    try:
        if len(data) >= adjust_window(data, 14) + 1:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=adjust_window(data, 14)).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        st.error(f"❌ Failed to compute ADX: {str(e)}")
        data['ADX'] = None
    
    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.error(f"❌ Failed to compute OBV: {str(e)}")
        data['OBV'] = None
    
    try:
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.error(f"❌ Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None
    
    try:
        data['Avg_Volume'] = data['Volume'].rolling(window=adjust_window(data, 10)).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.error(f"❌ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = None
    
    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.error(f"❌ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None
    
    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - diff * 0.236
        data['Fib_38.2'] = high - diff * 0.382
        data['Fib_50.0'] = high - diff * 0.5
        data['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.error(f"❌ Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = None
        data['Fib_38.2'] = None
        data['Fib_50.0'] = None
        data['Fib_61.8'] = None
    
    try:
        data['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.error(f"❌ Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"
    
    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=adjust_window(data, 9), window2=adjust_window(data, 26), window3=adjust_window(data, 52))
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Chikou'] = data['Close'].shift(-min(26, len(data) - 1))
    except Exception as e:
        st.error(f"❌ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = None
        data['Ichimoku_Kijun'] = None
        data['Ichimoku_Span_A'] = None
        data['Ichimoku_Span_B'] = None
        data['Ichimoku_Chikou'] = None
    
    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=adjust_window(data, 20)).chaikin_money_flow()
    except Exception as e:
        st.error(f"❌ Failed to compute CMF: {str(e)}")
        data['CMF'] = None
    
    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=adjust_window(data, 20))
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.error(f"❌ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
        data['Donchian_Middle'] = None
    
    # Prepare data for machine learning and train the model
    X, y = prepare_ml_data(data)
    if X is not None and y is not None and not X.empty and not y.empty:
        model, scaler = train_ml_model(X, y)
        data['ML_Prediction'] = predict_stock_trend(data, model, scaler)
    else:
        data['ML_Prediction'] = "Hold"
    
    return data

def calculate_buy_at(data):
    """Calculate recommended buy price."""
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    """Calculate stop loss based on ATR."""
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    """Calculate target price based on risk-reward ratio."""
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    risk_reward_ratio = 3 if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    target = last_close + (risk * risk_reward_ratio)
    return round(target, 2)

def fetch_fundamentals(symbol):
    """Fetch fundamental data for a stock."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0)
        }
    except Exception as e:
        st.error(f"❌ Failed to fetch fundamentals for {symbol}: {str(e)}")
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

def evaluate_indicator(data, indicator, condition):
    """Evaluate an indicator against a condition."""
    if indicator in data.columns and data[indicator].iloc[-1] is not None:
        return condition(data[indicator].iloc[-1])
    return False

def generate_recommendations(data, symbol=None):
    """Generate trading recommendations based on indicators and machine learning."""
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "ML_Prediction": "Hold",  # New field for machine learning prediction
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        st.warning("⚠️ No valid data available for recommendations.")
        return recommendations
    
    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        buy_score = 0
        sell_score = 0
        
        if 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
            if data['Close'].iloc[-1] > data['Close'].iloc[0]:
                buy_score += 1
            else:
                sell_score += 1
        
        if evaluate_indicator(data, 'RSI', lambda x: x < 30):
            buy_score += 2
        elif evaluate_indicator(data, 'RSI', lambda x: x > 70):
            sell_score += 2
        
        if evaluate_indicator(data, 'MACD', lambda x: x > data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns else False):
            buy_score += 1
        elif evaluate_indicator(data, 'MACD', lambda x: x < data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns else False):
            sell_score += 1
        
        if evaluate_indicator(data, 'Close', lambda x: x < data['Lower_Band'].iloc[-1] if 'Lower_Band' in data.columns else False):
            buy_score += 1
        elif evaluate_indicator(data, 'Close', lambda x: x > data['Upper_Band'].iloc[-1] if 'Upper_Band' in data.columns else False):
            sell_score += 1
        
        if evaluate_indicator(data, 'Close', lambda x: x > data['VWAP'].iloc[-1] if 'VWAP' in data.columns else False):
            buy_score += 1
        elif evaluate_indicator(data, 'Close', lambda x: x < data['VWAP'].iloc[-1] if 'VWAP' in data.columns else False):
            sell_score += 1
        
        if evaluate_indicator(data, 'Volume', lambda x: x > data['Avg_Volume'].iloc[-1] * 1.5 if 'Avg_Volume' in data.columns else False):
            buy_score += 1
        elif evaluate_indicator(data, 'Volume', lambda x: x < data['Avg_Volume'].iloc[-1] * 0.5 if 'Avg_Volume' in data.columns else False):
            sell_score += 1
        
        if 'Divergence' in data.columns and data['Divergence'].iloc[-1] == "Bullish Divergence":
            buy_score += 1
        elif 'Divergence' in data.columns and data['Divergence'].iloc[-1] == "Bearish Divergence":
            sell_score += 1
        
        if evaluate_indicator(data, 'Close', lambda x: x > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns else False):
            buy_score += 1
            recommendations["Ichimoku_Trend"] = "Buy"
        elif evaluate_indicator(data, 'Close', lambda x: x < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns else False):
            sell_score += 1
            recommendations["Ichimoku_Trend"] = "Sell"
        
        if evaluate_indicator(data, 'CMF', lambda x: x > 0):
            buy_score += 1
        elif evaluate_indicator(data, 'CMF', lambda x: x < 0):
            sell_score += 1
        
        if evaluate_indicator(data, 'Close', lambda x: x > data['Donchian_Upper'].iloc[-1] if 'Donchian_Upper' in data.columns else False):
            buy_score += 1
            recommendations["Breakout"] = "Buy"
        elif evaluate_indicator(data, 'Close', lambda x: x < data['Donchian_Lower'].iloc[-1] if 'Donchian_Lower' in data.columns else False):
            sell_score += 1
            recommendations["Breakout"] = "Sell"

        if evaluate_indicator(data, 'RSI', lambda x: x < 30 and data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1] if 'Lower_Band' in data.columns else False):
            buy_score += 2
            recommendations["Mean_Reversion"] = "Buy"
        elif evaluate_indicator(data, 'RSI', lambda x: x > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1] if 'Upper_Band' in data.columns else False):
            sell_score += 2
            recommendations["Mean_Reversion"] = "Sell"
        
        if evaluate_indicator(data, 'Ichimoku_Tenkan', lambda x: x > data['Ichimoku_Kijun'].iloc[-1] and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1] if 'Ichimoku_Kijun' in data.columns and 'Ichimoku_Span_A' in data.columns else False):
            buy_score += 1
            recommendations["Ichimoku_Trend"] = "Strong Buy"
        elif evaluate_indicator(data, 'Ichimoku_Tenkan', lambda x: x < data['Ichimoku_Kijun'].iloc[-1] and data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1] if 'Ichimoku_Kijun' in data.columns and 'Ichimoku_Span_B' in data.columns else False):
            sell_score += 1
            recommendations["Ichimoku_Trend"] = "Strong Sell"

        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 0.5
        
        if buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
        
        # Add machine learning prediction to recommendations
        recommendations["ML_Prediction"] = data['ML_Prediction'].iloc[-1] if 'ML_Prediction' in data.columns else "Hold"
        
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
    return recommendations

def analyze_batch(stock_batch, progress_callback=None):
    """Analyze a batch of stocks concurrently with detailed progress updates."""
    results = []
    total_stocks = len(stock_batch)
    processed_stocks = 0
    
    with ThreadPoolExecutor(max_workers=min(10, len(stock_batch))) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            processed_stocks += 1
            if progress_callback:
                progress_callback(processed_stocks / total_stocks, f"Processing {symbol} ({processed_stocks}/{total_stocks})")
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing stock {symbol}: {str(e)}")
    return results

def analyze_stock_parallel(symbol):
    """Analyze a single stock in parallel."""
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
            "ML_Prediction": recommendations["ML_Prediction"],  # New field for ML prediction
            "Score": recommendations.get("Score", 0),
        }
    return None

def update_progress(progress_bar, loading_text, progress_value, loading_messages, start_time, total_items, processed_items, current_task=""):
    """Update progress bar with detailed task information without showing warnings."""
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    elapsed_time = time.time() - start_time
    
    # Remove the warning for progress taking longer than expected, but keep the logic
    if processed_items > 0:
        time_per_item = elapsed_time / processed_items
        remaining_items = total_items - processed_items
        eta = timedelta(seconds=int(time_per_item * remaining_items))
        # Optional: Add a neutral progress message instead of a warning
        if elapsed_time > 120:
            loading_text.text(f"{loading_message}{dots} - {current_task} (Processing may take a moment, ETA: {eta}, Processed: {processed_items}/{total_items})")
        else:
            loading_text.text(f"{loading_message}{dots} - {current_task} (ETA: {eta}, Processed: {processed_items}/{total_items})")
    else:
        loading_text.text(f"{loading_message}{dots} - {current_task}")

def analyze_all_stocks(stock_list, batch_size=50, price_range=None, progress_callback=None):
    """Analyze all stocks and return top 10 with detailed progress and dynamic batch sizing, without showing slow processing warnings."""
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()

    results = []
    total_items = len(stock_list)
    start_time = time.time()
    processed_items = 0
    dynamic_batch_size = batch_size
    
    for i in range(0, total_items, dynamic_batch_size):
        if st.session_state.cancel_operation:
            st.warning("⚠️ Analysis canceled by user.")
            break
        
        batch = stock_list[i:i + dynamic_batch_size]
        batch_start_time = time.time()
        
        batch_results = analyze_batch(batch, lambda p, t: progress_callback(
            (processed_items + p * len(batch)) / total_items, t, start_time, processed_items + int(p * len(batch))
        ) if progress_callback else None)
        
        results.extend(batch_results)
        processed_items += len(batch)
        batch_elapsed = time.time() - batch_start_time
        
        # Adjust batch size if processing a batch takes too long (e.g., > 30 seconds), without showing warning
        if batch_elapsed > 30 and dynamic_batch_size > 10:
            dynamic_batch_size = max(10, dynamic_batch_size // 2)
        
        if progress_callback:
            progress_callback(processed_items / total_items, f"Completed batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}", start_time, processed_items)
    
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
    """Analyze stocks for intraday trading and return top 5 with detailed progress and dynamic batch sizing, without showing slow processing warnings."""
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()

    results = []
    total_items = len(stock_list)
    start_time = time.time()
    processed_items = 0
    dynamic_batch_size = batch_size
    
    for i in range(0, total_items, dynamic_batch_size):
        if st.session_state.cancel_operation:
            st.warning("⚠️ Analysis canceled by user.")
            break
        
        batch = stock_list[i:i + dynamic_batch_size]
        batch_start_time = time.time()
        
        batch_results = analyze_batch(batch, lambda p, t: progress_callback(
            (processed_items + p * len(batch)) / total_items, t, start_time, processed_items + int(p * len(batch))
        ) if progress_callback else None)
        
        results.extend(batch_results)
        processed_items += len(batch)
        batch_elapsed = time.time() - batch_start_time
        
        # Adjust batch size if processing a batch takes too long (e.g., > 30 seconds), without showing warning
        if batch_elapsed > 30 and dynamic_batch_size > 10:
            dynamic_batch_size = max(10, dynamic_batch_size // 2)
        
        if progress_callback:
            progress_callback(processed_items / total_items, f"Completed batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}", start_time, processed_items)
    
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
    """Color-code recommendations for display."""
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    return recommendation

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    """Display the stock analysis dashboard with machine learning predictions."""
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=10000, value=(100, 1000))
    
    if st.button("🚀 Generate Daily Top Picks"):
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Analysis")
        loading_messages = itertools.cycle([
            "Fetching stock data", "Calculating indicators", "Generating recommendations",
            "Ranking stocks", "Finalizing results"
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
            st.warning("⚠️ Analysis canceled by user.")
            st.session_state.cancel_operation = False
        
        results_df = analyze_all_stocks(
            NSE_STOCKS,
            price_range=price_range,
            progress_callback=lambda progress, task, start_time, processed: update_progress(
                progress_bar, loading_text, progress, loading_messages, start_time, len(NSE_STOCKS), processed, task
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
                    ML Prediction: {colored_recommendation(row['ML_Prediction'])}  <!-- New ML prediction display -->
                    """, unsafe_allow_html=True)
        elif not st.session_state.cancel_operation:
            st.warning("⚠️ No top picks available due to data issues.")
    
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel Intraday Analysis")
        loading_messages = itertools.cycle([
            "Fetching intraday data", "Analyzing buy signals", "Computing stop-loss levels",
            "Setting targets", "Selecting top picks"
        ])
        
        if cancel_button:
            st.session_state.cancel_operation = True
            st.warning("⚠️ Analysis canceled by user.")
            st.session_state.cancel_operation = False
        
        intraday_results = analyze_intraday_stocks(
            NSE_STOCKS,
            price_range=price_range,
            progress_callback=lambda progress, task, start_time, processed: update_progress(
                progress_bar, loading_text, progress, loading_messages, start_time, len(NSE_STOCKS), processed, task
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
                    ML Prediction: {colored_recommendation(row['ML_Prediction'])}  <!-- New ML prediction display -->
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
        new_strategies = ["Mean_Reversion", "Breakout", "Ichimoku_Trend", "ML_Prediction"]
        for col, strategy in zip(cols, new_strategies):
            with col:
                st.markdown(f"**{strategy.replace('_', ' ')}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]), unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Action", "📉 Momentum", "📊 Volatility", "📈 Monte Carlo", "📉 New Indicators"])
        with tab1:
            price_cols = ['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with Moving Averages",
                              labels={'value': 'Price (₹)', 'variable': 'Indicator'}, template="plotly_white")
                fig.update_layout(legend_title_text='Indicator')
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid price action data available for plotting.")
        
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators",
                              labels={'value': 'Value', 'variable': 'Indicator'}, template="plotly_white")
                fig.update_layout(legend_title_text='Indicator')
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid momentum indicators available for plotting.")
        
        with tab3:
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis",
                              labels={'value': 'Value', 'variable': 'Indicator'}, template="plotly_white")
                fig.update_layout(legend_title_text='Indicator')
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid volatility indicators available for plotting.")
        
        with tab4:
            mc_results = monte_carlo_simulation(data)
            mc_df = pd.DataFrame(mc_results).T
            mc_df.columns = [f"Sim {i+1}" for i in range(len(mc_results))]
            fig = px.line(mc_df, title="Monte Carlo Price Simulations (30 Days)",
                          labels={'value': 'Price (₹)', 'variable': 'Simulation'}, template="plotly_white")
            fig.update_layout(legend_title_text='Simulation')
            st.plotly_chart(fig)
        
        with tab5:
            new_cols = ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF', 'ML_Prediction']
            valid_new_cols = [col for col in new_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col]) or col == 'ML_Prediction']
            if valid_new_cols:
                if 'ML_Prediction' in valid_new_cols:
                    valid_new_cols.remove('ML_Prediction')  # Remove non-numeric ML prediction for plotting
                if valid_new_cols:
                    fig = px.line(data, y=valid_new_cols, title="New Indicators (Ichimoku & CMF)",
                                  labels={'value': 'Value', 'variable': 'Indicator'}, template="plotly_white")
                    fig.update_layout(legend_title_text='Indicator')
                    st.plotly_chart(fig)
                st.write(f"ML Prediction: {data['ML_Prediction'].iloc[-1]}")
            else:
                st.warning("⚠️ No valid new indicators available for plotting.")
        
        if st.button("Analyze News Sentiment"):
            news_sentiment = fetch_news_sentiment_vader(symbol.split('.')[0], NEWSAPI_KEY)
            finbert_sentiment = analyze_sentiment_finbert(f"Latest news about {symbol.split('.')[0]}")
            st.write(f"VADER Sentiment: {news_sentiment:.2f}")
            st.write(f"FinBERT Sentiment: {finbert_sentiment} (0=Negative, 1=Neutral, 2=Positive)")
    elif symbol:
        st.warning("⚠️ No data available for the selected stock.")

def main():
    """Main function to run the StockGenie Pro application."""
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