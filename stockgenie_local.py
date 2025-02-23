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
from diskcache import Cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import sqlite3

# Setup diskcache
cache = Cache("stock_cache", expire=86400)

# API Keys
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
    return f"{label} 📌 ({explanation})"

def retry(max_retries=3, delay=1, backoff_factor=1.5, jitter=0.5):
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
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except (requests.exceptions.RequestException, TimeoutError):
        return False

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('stock_data.db', check_same_thread=False)  # Allow multi-thread access
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT, 
                    date TEXT, 
                    open REAL, 
                    high REAL, 
                    low REAL, 
                    close REAL, 
                    volume INTEGER, 
                    last_updated TEXT,
                    PRIMARY KEY (symbol, date)
                 )''')
    conn.commit()
    return conn

# Global SQLite connection
DB_CONN = None

@retry(max_retries=3, delay=1)
def fetch_nse_stock_list():
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
    
    for col in ['Open', 'High', 'Low', 'Close']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        if not outliers.empty:
            st.warning(f"⚠️ Potential outliers detected in {symbol} {col}: {len(outliers)} values")
            data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = np.nan
    
    return True

@retry(max_retries=3, delay=1)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = init_db()  # Ensure DB is initialized if not already
    
    if ".NS" not in symbol:
        symbol += ".NS"
    
    c = DB_CONN.cursor()
    
    # Check if data exists in SQLite
    try:
        c.execute("SELECT * FROM stock_data WHERE symbol = ?", (symbol,))
        rows = c.fetchall()
        if rows:
            data = pd.DataFrame(rows, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'last_updated'])
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            if validate_stock_data(data.drop(columns=['symbol', 'last_updated']), symbol):
                last_updated = pd.to_datetime(data['last_updated'].iloc[0])
                if (datetime.now() - last_updated).days < 1:  # Cache valid for 1 day
                    return data.drop(columns=['symbol', 'last_updated'])
    except sqlite3.OperationalError as e:
        st.error(f"❌ SQLite error for {symbol}: {str(e)}")
        # Table might not exist yet; proceed to fetch and create
    
    # Fetch from Yahoo Finance if not in cache or outdated
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        if validate_stock_data(data, symbol):
            data_reset = data.reset_index()
            data_reset['symbol'] = symbol
            data_reset['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data_reset.to_sql('stock_data', DB_CONN, if_exists='replace', index=False)
            DB_CONN.commit()
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to fetch data for {symbol}: {str(e)}. Falling back to cache.")
        return pd.DataFrame()

def calculate_advance_decline_ratio(stock_list):
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
    return min(default_window, max(min_window, len(data) - 1))

def monte_carlo_simulation(data, simulations=1000, days=30, distribution="normal"):
    returns = data['Close'].pct_change().dropna()
    returns = returns[(returns > returns.quantile(0.05)) & (returns < returns.quantile(0.95))]
    
    if len(returns) < 30:
        st.warning(f"⚠️ Limited data ({len(returns)} days) for GARCH; using basic simulation.")
        mean_return = returns.mean()
        std_return = returns.std() if returns.std() != 0 else 0.01
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                if distribution == "t":
                    shock = np.random.standard_t(df=5) * std_return
                else:
                    shock = np.random.normal(0, std_return)
                price = price_series[-1] * (1 + mean_return + shock)
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    best_p, best_q = 1, 1
    best_aic = float('inf')
    for p, q in [(1, 1), (1, 2), (2, 1)]:
        try:
            model = arch_model(returns, vol='GARCH', p=p, q=q, dist=distribution)
            fit = model.fit(disp='off')
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_p, best_q = p, q
        except:
            continue
    
    model = arch_model(returns, vol='GARCH', p=best_p, q=best_q, dist=distribution)
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            if distribution == "t":
                shock = np.random.standard_t(df=5) * volatility[i]
            else:
                shock = np.random.normal(0, volatility[i])
            price = price_series[-1] * (1 + mean_return + shock)
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

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
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def create_sentiment_heatmap(sentiment_data):
    fig = px.imshow(sentiment_data, labels=dict(x="Stocks", y="Sentiment", color="Sentiment Score"),
                    x=sentiment_data.columns, y=sentiment_data.index)
    st.plotly_chart(fig)

def calculate_confidence_score(data):
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    return score / 3

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
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
    if data.empty or len(data) < 2:
        return None, None
    
    base_features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'SlowK', 'SlowD', 'OBV', 'VWAP', 'CMF',
                     'Keltner_Upper', 'Keltner_Lower', 'Force_Index', 'Z_Score']
    
    returns = data['Close'].pct_change()
    data['Lag1_Return'] = returns.shift(1)
    data['Lag2_Return'] = returns.shift(2)
    data['Volatility_5'] = returns.rolling(window=5).std()
    data['Volatility_20'] = returns.rolling(window=20).std()
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    
    all_features = base_features + ['Lag1_Return', 'Lag2_Return', 'Volatility_5', 'Volatility_20', 'Day_of_Week', 'Month']
    X = data[all_features].dropna()
    
    price_changes = returns.shift(-1).dropna()
    y = (price_changes > 0).astype(int)
    
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].dropna()
    y = y.loc[common_index]
    
    return X, y

def train_ml_model(X, y):
    if X is None or y is None or X.empty or y.empty:
        st.warning("⚠️ Insufficient data for machine learning training.")
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=10)
    X_rfe = rfe.fit_transform(X_scaled, y)
    selected_features = X.columns[rfe.support_].tolist()
    st.info(f"Selected Features by RFE: {', '.join(selected_features)}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_rfe, y, cv=tscv, scoring='accuracy')
    model.fit(X_rfe, y)
    
    st.info(f"Time-Series CV Accuracy with RFE: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
    return model, scaler

def predict_stock_trend(data, model, scaler):
    if data.empty or model is None or scaler is None:
        return "Hold"
    
    features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'SlowK', 'SlowD', 'OBV', 'VWAP', 'CMF',
                'Keltner_Upper', 'Keltner_Lower', 'Force_Index', 'Z_Score',
                'Lag1_Return', 'Lag2_Return', 'Volatility_5', 'Volatility_20', 'Day_of_Week', 'Month']
    X = data[features].dropna().tail(1)
    
    if X.empty:
        return "Hold"
    
    X_scaled = scaler.transform(X)
    X_rfe = X_scaled[:, :10]
    prediction = model.predict(X_rfe)[0]
    return "Buy" if prediction == 1 else "Sell"

def calculate_momentum_indicators(data):
    try:
        rsi_window = adjust_window(data, optimize_rsi_window(data), min_window=5)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.error(f"❌ Failed to compute RSI: {str(e)}")
        data['RSI'] = None
    
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.error(f"❌ Failed to compute MACD: {str(e)}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None
    
    try:
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.error(f"❌ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = None
        data['SlowD'] = None
    
    return data

def calculate_volatility_indicators(data):
    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.error(f"❌ Failed to compute ATR: {str(e)}")
        data['ATR'] = None
    
    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.error(f"❌ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None
    
    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.error(f"❌ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
        data['Donchian_Middle'] = None
    
    try:
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
        data['Keltner_Upper'] = keltner.keltner_channel_hband()
        data['Keltner_Middle'] = keltner.keltner_channel_mband()
        data['Keltner_Lower'] = keltner.keltner_channel_lband()
    except Exception as e:
        st.error(f"❌ Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = None
        data['Keltner_Middle'] = None
        data['Keltner_Lower'] = None
    
    return data

def calculate_trend_indicators(data):
    try:
        sma_50_window = adjust_window(data, 50, min_window=10)
        sma_200_window = adjust_window(data, 200, min_window=20)
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=sma_50_window).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=sma_200_window).sma_indicator() if len(data) >= 200 else None
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=sma_50_window).ema_indicator()
    except Exception as e:
        st.error(f"❌ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None
    
    try:
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx() if len(data) >= 15 else None
    except Exception as e:
        st.error(f"❌ Failed to compute ADX: {str(e)}")
        data['ADX'] = None
    
    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.error(f"❌ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None
    
    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
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
    
    return data

def calculate_volume_indicators(data):
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
        data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.error(f"❌ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = None
    
    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.error(f"❌ Failed to compute CMF: {str(e)}")
        data['CMF'] = None
    
    try:
        data['Force_Index'] = ta.volume.ForceIndexIndicator(data['Close'], data['Volume'], window=13).force_index()
    except Exception as e:
        st.error(f"❌ Failed to compute Force Index: {str(e)}")
        data['Force_Index'] = None
    
    return data

def analyze_stock(data):
    if data.empty:
        st.warning("⚠️ No data available to analyze.")
        return data
    
    check_data_sufficiency(data)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    data[required_columns] = data[required_columns].interpolate(method='linear', limit_direction='both')
    if data[required_columns].isnull().any().any():
        st.warning("⚠️ Some NaNs remain after interpolation; filling with forward-fill.")
        data[required_columns] = data[required_columns].fillna(method='ffill').fillna(method='bfill')
    
    data = calculate_momentum_indicators(data)
    data = calculate_volatility_indicators(data)
    data = calculate_trend_indicators(data)
    data = calculate_volume_indicators(data)
    
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
        data['Z_Score'] = (data['Close'] - data['Close'].rolling(window=20).mean()) / data['Close'].rolling(window=20).std()
    except Exception as e:
        st.error(f"❌ Failed to compute Z-Score: {str(e)}")
        data['Z_Score'] = None
    
    X, y = prepare_ml_data(data)
    if X is not None and y is not None and not X.empty and not y.empty:
        model, scaler = train_ml_model(X, y)
        data['ML_Prediction'] = predict_stock_trend(data, model, scaler)
    else:
        data['ML_Prediction'] = "Hold"
    
    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
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
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0),
            'DebtToEquity': info.get('debtToEquity', None),
            'DividendYield': info.get('dividendYield', 0)
        }
    except Exception as e:
        st.error(f"❌ Failed to fetch fundamentals for {symbol}: {str(e)}")
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0, 'DebtToEquity': None, 'DividendYield': 0}

def evaluate_indicator(data, indicator, condition):
    if indicator in data.columns and data[indicator].iloc[-1] is not None:
        return condition(data[indicator].iloc[-1])
    return False

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "ML_Prediction": "Hold",
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
            if fundamentals['DebtToEquity'] is not None and fundamentals['DebtToEquity'] < 1.0:
                buy_score += 0.5
            elif fundamentals['DebtToEquity'] is not None and fundamentals['DebtToEquity'] > 2.0:
                sell_score += 0.5
            if fundamentals['DividendYield'] > 0.02:
                buy_score += 0.5
        
        if buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
        
        recommendations["ML_Prediction"] = data['ML_Prediction'].iloc[-1] if 'ML_Prediction' in data.columns else "Hold"
        
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
    return recommendations

def analyze_batch(stock_batch, progress_callback=None):
    results = []
    total_stocks = len(stock_batch)
    processed_stocks = 0
    
    with ThreadPoolExecutor(max_workers=min(50, total_stocks)) as executor:
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
            "ML_Prediction": recommendations["ML_Prediction"],
            "Score": recommendations.get("Score", 0),
        }
    return None

def update_progress(progress_bar, loading_text, progress_value, loading_messages, start_time, total_items, processed_items, current_task=""):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    elapsed_time = time.time() - start_time
    
    if processed_items > 0:
        time_per_item = elapsed_time / processed_items
        remaining_items = total_items - processed_items
        eta = timedelta(seconds=int(time_per_item * remaining_items))
        if elapsed_time > 120:
            loading_text.text(f"{loading_message}{dots} - {current_task} (Processing may take a moment, ETA: {eta}, Processed: {processed_items}/{total_items})")
        else:
            loading_text.text(f"{loading_message}{dots} - {current_task} (ETA: {eta}, Processed: {processed_items}/{total_items})")
    else:
        loading_text.text(f"{loading_message}{dots} - {current_task}")

def analyze_all_stocks(stock_list, batch_size=50, price_range=None, progress_callback=None):
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled by user.")
        return pd.DataFrame()
    
    results = []
    total_items = len(stock_list)
    start_time = time.time()
    processed_items = 0
    
    with ThreadPoolExecutor(max_workers=min(50, total_items)) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_list}
        for future in as_completed(futures):
            if st.session_state.cancel_operation:
                st.warning("⚠️ Analysis canceled by user.")
                break
            symbol = futures[future]
            processed_items += 1
            if progress_callback:
                progress_callback(processed_items / total_items, f"Processing {symbol} ({processed_items}/{total_items})", start_time, processed_items)
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing stock {symbol}: {str(e)}")
    
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
    processed_items = 0
    
    with ThreadPoolExecutor(max_workers=min(50, total_items)) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_list}
        for future in as_completed(futures):
            if st.session_state.cancel_operation:
                st.warning("⚠️ Analysis canceled by user.")
                break
            symbol = futures[future]
            processed_items += 1
            if progress_callback:
                progress_callback(processed_items / total_items, f"Processing {symbol} ({processed_items}/{total_items})", start_time, processed_items)
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing stock {symbol}: {str(e)}")
    
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
                    ML Prediction: {colored_recommendation(row['ML_Prediction'])}
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
                    ML Prediction: {colored_recommendation(row['ML_Prediction'])}
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
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower', 'Keltner_Upper', 'Keltner_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis",
                              labels={'value': 'Value', 'variable': 'Indicator'}, template="plotly_white")
                fig.update_layout(legend_title_text='Indicator')
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid volatility indicators available for plotting.")
        
        with tab4:
            dist_option = st.selectbox("Distribution for Monte Carlo", ["normal", "t"])
            mc_results = monte_carlo_simulation(data, distribution=dist_option)
            mc_df = pd.DataFrame(mc_results).T
            mc_df.columns = [f"Sim {i+1}" for i in range(len(mc_results))]
            fig = px.line(mc_df, title="Monte Carlo Price Simulations (30 Days)",
                          labels={'value': 'Price (₹)', 'variable': 'Simulation'}, template="plotly_white")
            fig.update_layout(legend_title_text='Simulation')
            st.plotly_chart(fig)
        
        with tab5:
            new_cols = ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF', 'Force_Index', 'Z_Score']
            valid_new_cols = [col for col in new_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_new_cols:
                fig = px.line(data, y=valid_new_cols, title="New Indicators (Ichimoku, CMF, Force, Z-Score)",
                              labels={'value': 'Value', 'variable': 'Indicator'}, template="plotly_white")
                fig.update_layout(legend_title_text='Indicator')
                st.plotly_chart(fig)
            st.write(f"ML Prediction: {data['ML_Prediction'].iloc[-1]}")
        
        if st.button("Analyze News Sentiment"):
            news_sentiment = fetch_news_sentiment_vader(symbol.split('.')[0], NEWSAPI_KEY)
            finbert_sentiment = analyze_sentiment_finbert(f"Latest news about {symbol.split('.')[0]}")
            st.write(f"VADER Sentiment: {news_sentiment:.2f}")
            st.write(f"FinBERT Sentiment: {finbert_sentiment} (0=Negative, 1=Neutral, 2=Positive)")
    elif symbol:
        st.warning("⚠️ No data available for the selected stock.")

def main():
    global DB_CONN
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    
    # Initialize SQLite database at startup
    DB_CONN = init_db()
    
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
    
    # Close the database connection when done
    if DB_CONN is not None:
        DB_CONN.close()

if __name__ == "__main__":
    main()