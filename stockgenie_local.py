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
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Comprehensive list of User Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 11; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/92.0.902.67 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
]

# API Keys (Consider moving to environment variables)
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
    "Keltner": "Keltner Channels - Volatility bands based on EMA and ATR",
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
}

# Define sectors and their stocks
SECTORS = {
    "Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS", 
        "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "BANDHANBNK.NS", "INDIANB.NS", 
        "BANKINDIA.NS", "KARURVYSYA.NS", "CUB.NS", "J&KBANK.NS", "LAKSHVILAS.NS", 
        "DCBBANK.NS", "SYNDIBANK.NS", "ALBK.NS", "ANDHRABANK.NS", "CORPBANK.NS", 
        "ORIENTBANK.NS", "UNITEDBNK.NS", "AUBANK.NS"
    ],
    "Software & IT Services": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Finance": ["BAJFINANCE.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS"],
    "Automobile & Ancillaries": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "Oil&Gas": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "OIL.NS", "PETRONET.NS", "MRPL.NS", "CHENNPETRO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Power": ["NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "ADANIENSOL.NS", "JSWENERGY.NS", "INDIANREN.NS", "NLCINDIA.NS", "CESC.NS", "RPOWER.NS", "IEX.NS", "NAVA.NS", "INDIGRID.NS", "ACMESOLAR.NS", "RELINFRA.NS", "GMRP&UI.NS", "SWSOLAR.NS", "PTC.NS", "GIPCL.NS", "BFUTILITIE.NS", "RAVINDRA.NS", "DANISH.NS", "APSINDIA.NS", "SUNGARNER.NS"],
    "Capital Goods": ["LT.NS", "BHEL.NS", "SIEMENS.NS"],
    "Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"],
    "Infrastructure": ["ADANIPORTS.NS", "GMRINFRA.NS"],
    "Insurance": ["ICICIGI.NS", "NIACL.NS"],
    "Diversified": ["RODIUM.BO", "FRANKLIN.BO", "ANSALBU.NS", "SHERVANI.BO"],
    "Construction Materials": ["ULTRACEMCO.NS", "ACC.NS", "AMBUJACEM.NS"],
    "Real Estate": ["DLF.NS", "GODREJPROP.NS"],
    "Aviation": ["INDIGO.NS", "SPICEJET.NS"],
    "Retailing": ["DMART.NS", "TRENT.NS"],
    "Miscellaneous": ["ADANIENT.NS", "ADANIGREEN.NS"],
    "Diamond & Jewellery": ["TITAN.NS", "PCJEWELLER.NS"],
    "Consumer Durables": ["HAVELLS.NS", "CROMPTON.NS"],
    "Trading": ["ADANIPOWER.NS"],
    "Hospitality": ["INDHOTEL.NS", "EIHOTEL.NS"],
    "Agri": ["UPL.NS", "PIIND.NS"],
    "Textiles": ["PAGEIND.NS", "RAYMOND.NS"],
    "Industrial Gases & Fuels": ["LINDEINDIA.NS"],
    "Electricals": ["POLYCAB.NS", "KEI.NS"],
    "Alcohol": ["SDBL.NS", "GLOBUSSPR.NS", "TI.NS", "PICCADIL.NS", "ABDL.NS", "RADICO.NS", "UBL.NS", "UNITDSPR.NS"],
    "Logistics": ["CONCOR.NS", "BLUEDART.NS"],
    "Plastic Products": ["SUPREMEIND.NS"],
    "Ship Building": ["ABSMARINE.NS", "GRSE.NS", "COCHINSHIP.NS"],
    "Media & Entertainment": ["ZEEL.NS", "SUNTV.NS"],
    "ETF": ["NIFTYBEES.NS"],
    "Footwear": ["BATAINDIA.NS", "RELAXO.NS"],
    "Manufacturing": ["ASIANPAINT.NS", "BERGEPAINT.NS"],
    "Containers & Packaging": ["AGI.NS", "UFLEX.NS", "JINDALPOLY.NS", "COSMOFIRST.NS", "HUHTAMAKI.NS", "ESTER.NS", "TIRUPATI.NS", "PYRAMID.NS", "BBTCL.NS", "RAJESHCAN.NS", "IDEALTECHO.NS", "PERFECTPAC.NS", "GUJCON.NS", "HCP.NS", "SHETRON.NS"],
    "Paper": ["JKPAPER.NS", "WSTCSTPAPR.NS", "SESHAPAPER.NS", "PDMJEPAPER.NS", "NRAGRINDQ.NS", "RUCHIRA.NS", "SANGALPAPR.NS", "SVJENTERPR.NS", "METROGLOBL.NS", "SHREYANIND.NS", "SUBAMPAPER.NS", "STARPAPER.NS", "PAKKA.NS", "TNPL.NS", "KUANTUM.NS"],
    "Photographic Products": ["JINDALPHOT.NS"]
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
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

def fetch_stock_data_with_auth(symbol, period="5y", interval="1d"):
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        stock = yf.Ticker(symbol, session=session)
        time.sleep(random.uniform(1, 3))
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

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

def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 30:
        mean_return = returns.mean()
        std_return = returns.std()
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            price = price_series[-1] * (1 + np.random.normal(mean_return, volatility[i]))
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

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
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] is not None and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    # Adjust for volatility
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]  # Normalized ATR
        if atr_volatility < 0.02:  # Low volatility increases confidence
            score += 0.5
        elif atr_volatility > 0.05:  # High volatility reduces confidence
            score -= 0.5
    return min(max(score / 3.5, 0), 1)  # Normalize to 0-1 scale

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    else:
        return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 15)):
    best_window, best_sharpe = 9, -float('inf')
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 50:
        return best_window
    for window in windows:
        rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
        signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
        strategy_returns = signals.shift(1) * returns
        sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_window = sharpe, window
    return best_window

def detect_divergence(data):
    rsi = data['RSI']
    price = data['Close']
    recent_highs = price[-5:].idxmax()
    recent_lows = price[-5:].idxmin()
    rsi_highs = rsi[-5:].idxmax()
    rsi_lows = rsi[-5:].idxmin()
    bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1])
    bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_cmo(close, window=14):
    try:
        diff = close.diff()
        up_sum = diff.where(diff > 0, 0).rolling(window=window).sum()
        down_sum = abs(diff.where(diff < 0, 0)).rolling(window=window).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo
    except Exception as e:
        st.warning(f"⚠️ Failed to compute custom CMO: {str(e)}")
        return None

def analyze_stock(data):
    if data.empty or len(data) < 27:
        st.warning("⚠️ Insufficient data to compute indicators.")
        return data
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        return data
    
    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = None
    try:
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
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
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None
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
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = None
        data['SlowD'] = None
    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        data['ATR'] = None
    try:
        if len(data) >= 27:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
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
        data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = None
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
        data['Fib_38.2'] = high - diff * 0.382
        data['Fib_50.0'] = high - diff * 0.5
        data['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = None
        data['Fib_38.2'] = None
        data['Fib_50.0'] = None
        data['Fib_61.8'] = None
    try:
        data['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"
    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Chikou'] = data['Close'].shift(-26)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = None
        data['Ichimoku_Kijun'] = None
        data['Ichimoku_Span_A'] = None
        data['Ichimoku_Span_B'] = None
        data['Ichimoku_Chikou'] = None
    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        data['CMF'] = None
    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
        data['Donchian_Middle'] = None
    try:
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
        data['Keltner_Upper'] = keltner.keltner_channel_hband()
        data['Keltner_Middle'] = keltner.keltner_channel_mband()
        data['Keltner_Lower'] = keltner.keltner_channel_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = None
        data['Keltner_Middle'] = None
        data['Keltner_Lower'] = None
    try:
        data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute TRIX: {str(e)}")
        data['TRIX'] = None
    try:
        data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
            data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
        ).ultimate_oscillator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = None
    try:
        data['CMO'] = calculate_cmo(data['Close'], window=14)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = None
    try:
        data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = None
    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    if stop_loss < last_close * 0.9:
        stop_loss = last_close * 0.9
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = min(risk_reward_ratio, 5) if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else min(risk_reward_ratio, 3)
    target = last_close + (risk * adjusted_ratio)
    if target > last_close * 1.2:
        target = last_close * 1.2
    return round(target, 2)

def fetch_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'P/E': info.get('trailingPE', float('inf')),
            'EPS': info.get('trailingEps', 0),
            'RevenueGrowth': info.get('revenueGrowth', 0)
        }
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        st.warning("⚠️ No valid data available for recommendations.")
        return recommendations
    
    try:
        recommendations["Current Price"] = float(data['Close'].iloc[-1])
        buy_score = 0
        sell_score = 0
        print(f"Initial buy_score: {buy_score}, sell_score: {sell_score}")
        
        # RSI (Boosted for RSI <= 20)
        print("Checking RSI...")
        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
            print(f"RSI value: {data['RSI'].iloc[-1]}, type: {type(data['RSI'].iloc[-1])}")
            if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['RSI'].iloc[-1] <= 20:  # Extreme oversold
                    buy_score += 4
                    print(f"RSI <= 20, buy_score: {buy_score}")
                elif data['RSI'].iloc[-1] < 30:
                    buy_score += 2
                    print(f"RSI < 30, buy_score: {buy_score}")
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2
                    print(f"RSI > 70, sell_score: {sell_score}")
        
        # MACD
        print("Checking MACD...")
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None:
            print(f"MACD: {data['MACD'].iloc[-1]}, Signal: {data['MACD_signal'].iloc[-1]}")
            if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    buy_score += 1
                    print(f"MACD > Signal, buy_score: {buy_score}")
                elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                    sell_score += 1
                    print(f"MACD < Signal, sell_score: {sell_score}")
        
        # Bollinger Bands
        print("Checking Bollinger Bands...")
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
            print(f"Close: {data['Close'].iloc[-1]}, Lower: {data['Lower_Band'].iloc[-1]}, Upper: {data['Upper_Band'].iloc[-1]}")
            if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                    print(f"Close < Lower_Band, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1
                    print(f"Close > Upper_Band, sell_score: {sell_score}")
        
        # VWAP
        print("Checking VWAP...")
        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
            print(f"Close: {data['Close'].iloc[-1]}, VWAP: {data['VWAP'].iloc[-1]}")
            if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1
                    print(f"Close > VWAP, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1
                    print(f"Close < VWAP, sell_score: {sell_score}")
        
        # Volume Analysis
        print("Checking Volume Analysis...")
        if ('Volume' in data.columns and data['Volume'].iloc[-1] is not None and 
            'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None):
            volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
            print(f"Volume Ratio: {volume_ratio}, Close: {data['Close'].iloc[-1]}, Prev Close: {data['Close'].iloc[-2]}")
            if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 2
                    print(f"Volume Ratio > 1.5 & Price Up, buy_score: {buy_score}")
                elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    sell_score += 2
                    print(f"Volume Ratio > 1.5 & Price Down, sell_score: {sell_score}")
                elif volume_ratio < 0.5:
                    sell_score += 1
                    print(f"Volume Ratio < 0.5, sell_score: {sell_score}")
        
        # Volume Spikes
        print("Checking Volume Spikes...")
        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None:
            print(f"Volume_Spike: {data['Volume_Spike'].iloc[-1]}, Close: {data['Close'].iloc[-1]}, Prev Close: {data['Close'].iloc[-2]}")
            if data['Volume_Spike'].iloc[-1] and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 1
                    print(f"Volume Spike & Price Up, buy_score: {buy_score}")
                else:
                    sell_score += 1
                    print(f"Volume Spike & Price Down, sell_score: {sell_score}")
        
        # Divergence
        print("Checking Divergence...")
        if 'Divergence' in data.columns:
            print(f"Divergence: {data['Divergence'].iloc[-1]}")
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1
                print(f"Bullish Divergence, buy_score: {buy_score}")
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1
                print(f"Bearish Divergence, sell_score: {sell_score}")
        
        # Ichimoku Cloud
        print("Checking Ichimoku Cloud...")
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None:
            print(f"Close: {data['Close'].iloc[-1]}, Span_A: {data['Ichimoku_Span_A'].iloc[-1]}, Span_B: {data['Ichimoku_Span_B'].iloc[-1]}")
            if (isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Buy"
                    print(f"Close > Ichimoku Cloud, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    print(f"Close < Ichimoku Cloud, sell_score: {sell_score}")
        
        # Chaikin Money Flow
        print("Checking Chaikin Money Flow...")
        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
            print(f"CMF: {data['CMF'].iloc[-1]}")
            if isinstance(data['CMF'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMF'].iloc[-1] > 0:
                    buy_score += 1
                    print(f"CMF > 0, buy_score: {buy_score}")
                elif data['CMF'].iloc[-1] < 0:
                    sell_score += 1
                    print(f"CMF < 0, sell_score: {sell_score}")
        
        # Donchian Channels
        print("Checking Donchian Channels...")
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Close'].iloc[-1] is not None:
            print(f"Close: {data['Close'].iloc[-1]}, Upper: {data['Donchian_Upper'].iloc[-1]}, Lower: {data['Donchian_Lower'].iloc[-1]}")
            if (isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Donchian_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                    print(f"Close > Donchian Upper, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"
                    print(f"Close < Donchian Lower, sell_score: {sell_score}")
        
        # Mean Reversion
        print("Checking Mean Reversion...")
        if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
            print(f"RSI: {data['RSI'].iloc[-1]}, Close: {data['Close'].iloc[-1]}, Lower_Band: {data['Lower_Band'].iloc[-1]}")
            if (isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] >= data['Lower_Band'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                    print(f"Mean Reversion Buy, buy_score: {buy_score}")
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"
                    print(f"Mean Reversion Sell, sell_score: {sell_score}")
        
        # Ichimoku Trend
        print("Checking Ichimoku Trend...")
        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and data['Close'].iloc[-1] is not None:
            print(f"Tenkan: {data['Ichimoku_Tenkan'].iloc[-1]}, Kijun: {data['Ichimoku_Kijun'].iloc[-1]}, Close: {data['Close'].iloc[-1]}")
            if (isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating))):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                    print(f"Ichimoku Strong Buy, buy_score: {buy_score}")
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"
                    print(f"Ichimoku Strong Sell, sell_score: {sell_score}")
        
        # Keltner Channels
        print("Checking Keltner Channels...")
        if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
            data['Close'].iloc[-1] is not None):
            print(f"Close: {data['Close'].iloc[-1]}, Keltner Upper: {data['Keltner_Upper'].iloc[-1]}, Keltner Lower: {data['Keltner_Lower'].iloc[-1]}")
            if (isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 1
                    print(f"Close < Keltner Lower, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 1
                    print(f"Close > Keltner Upper, sell_score: {sell_score}")
        
        # TRIX
        print("Checking TRIX...")
        if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None:
            print(f"TRIX: {data['TRIX'].iloc[-1]}, Prev TRIX: {data['TRIX'].iloc[-2]}")
            if isinstance(data['TRIX'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['TRIX'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                    buy_score += 1
                    print(f"TRIX > 0 & Rising, buy_score: {buy_score}")
                elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                    sell_score += 1
                    print(f"TRIX < 0 & Falling, sell_score: {sell_score}")
        
        # Ultimate Oscillator
        print("Checking Ultimate Oscillator...")
        if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None:
            print(f"Ultimate Oscillator: {data['Ultimate_Osc'].iloc[-1]}")
            if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Ultimate_Osc'].iloc[-1] < 30:
                    buy_score += 1
                    print(f"Ultimate Oscillator < 30, buy_score: {buy_score}")
                elif data['Ultimate_Osc'].iloc[-1] > 70:
                    sell_score += 1
                    print(f"Ultimate Oscillator > 70, sell_score: {sell_score}")
        
        # Chande Momentum Oscillator
        print("Checking Chande Momentum Oscillator...")
        if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None:
            print(f"CMO: {data['CMO'].iloc[-1]}")
            if isinstance(data['CMO'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMO'].iloc[-1] < -50:
                    buy_score += 1
                    print(f"CMO < -50, buy_score: {buy_score}")
                elif data['CMO'].iloc[-1] > 50:
                    sell_score += 1
                    print(f"CMO > 50, sell_score: {sell_score}")
        
        # Volume Price Trend
        print("Checking Volume Price Trend...")
        if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None:
            print(f"VPT: {data['VPT'].iloc[-1]}, Prev VPT: {data['VPT'].iloc[-2]}")
            if isinstance(data['VPT'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['VPT'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                    buy_score += 1
                    print(f"VPT Rising, buy_score: {buy_score}")
                elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                    sell_score += 1
                    print(f"VPT Falling, sell_score: {sell_score}")
        
        # Fibonacci Retracements
        print("Checking Fibonacci Retracements...")
        if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
            data['Close'].iloc[-1] is not None):
            current_price = data['Close'].iloc[-1]
            fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                          data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
            print(f"Close: {current_price}, Fib Levels: {fib_levels}")
            for level in fib_levels:
                if isinstance(level, (int, float, np.integer, np.floating)) and abs(current_price - level) / current_price < 0.01:
                    if current_price > level:
                        buy_score += 1
                        print(f"Close near Fib level {level} (support), buy_score: {buy_score}")
                    else:
                        sell_score += 1
                        print(f"Close near Fib level {level} (resistance), sell_score: {sell_score}")
        
        # Parabolic SAR
        print("Checking Parabolic SAR...")
        if ('Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and 
            data['Close'].iloc[-1] is not None):
            print(f"Close: {data['Close'].iloc[-1]}, Parabolic SAR: {data['Parabolic_SAR'].iloc[-1]}")
            if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                    buy_score += 1
                    print(f"Close > Parabolic SAR, buy_score: {buy_score}")
                elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                    sell_score += 1
                    print(f"Close < Parabolic SAR, sell_score: {sell_score}")
        
        # OBV
        print("Checking OBV...")
        if ('OBV' in data.columns and data['OBV'].iloc[-1] is not None and 
            data['OBV'].iloc[-2] is not None):
            print(f"OBV: {data['OBV'].iloc[-1]}, Prev OBV: {data['OBV'].iloc[-2]}")
            if isinstance(data['OBV'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['OBV'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['OBV'].iloc[-1] > data['OBV'].iloc[-2]:
                    buy_score += 1
                    print(f"OBV Rising, buy_score: {buy_score}")
                elif data['OBV'].iloc[-1] < data['OBV'].iloc[-2]:
                    sell_score += 1
                    print(f"OBV Falling, sell_score: {sell_score}")
        
        # Fundamentals
        print("Checking Fundamentals...")
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            print(f"Fundamentals: P/E={fundamentals['P/E']}, EPS={fundamentals['EPS']}, RevenueGrowth={fundamentals['RevenueGrowth']}")
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 2
                print(f"P/E < 15 & EPS > 0, buy_score: {buy_score}")
            elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
                sell_score += 1
                print(f"P/E > 30 or EPS < 0, sell_score: {sell_score}")
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 1
                print(f"Revenue Growth > 10%, buy_score: {buy_score}")
            elif fundamentals['RevenueGrowth'] < 0:
                sell_score += 0.5
                print(f"Revenue Growth < 0%, sell_score: {sell_score}")
        
        print(f"Before final recommendations: buy_score: {buy_score}, sell_score: {sell_score}")
        
        # Set recommendations based on scores
        net_score = buy_score - sell_score
        if buy_score > sell_score and buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
            recommendations["Swing"] = "Buy" if buy_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Buy" if buy_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Buy" if buy_score >= 1 else "Hold"
        elif sell_score > buy_score and sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
            recommendations["Swing"] = "Sell" if sell_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Sell" if sell_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Sell" if sell_score >= 1 else "Hold"
        elif net_score > 0:
            recommendations["Intraday"] = "Buy" if net_score >= 3 else "Hold"
            recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
            recommendations["Short-Term"] = "Buy" if net_score >= 1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        elif net_score < 0:
            recommendations["Intraday"] = "Sell" if net_score <= -3 else "Hold"
            recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
            recommendations["Short-Term"] = "Sell" if net_score <= -1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        
        recommendations["Score"] = min(max(buy_score - sell_score, -7), 7)
        print(f"Final buy_score: {buy_score}, sell_score: {sell_score}, Score: {recommendations['Score']}")
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

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    return results_df.sort_values(by="Score", ascending=False).head(10)

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        results_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        progress_bar.empty()
        loading_text.empty()
        if not results_df.empty:
            st.subheader("🏆 Today's Top 10 Stocks")
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
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
        else:
            st.warning("⚠️ No top picks available due to data issues.")
    
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No intraday picks available due to data issues.")
    
    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = recommendations['Current Price'] if recommendations['Current Price'] is not None else "N/A"
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
        with col2:
            buy_at = recommendations['Buy At'] if recommendations['Buy At'] is not None else "N/A"
            st.metric(tooltip("Buy At", "Recommended entry price"), f"₹{buy_at}")
        with col3:
            stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Action", "📉 Momentum", "📊 Volatility", "📈 Monte Carlo", "📉 New Indicators"])
        with tab1:
            price_cols = ['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50']
            valid_price_cols = [col for col in price_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_price_cols:
                fig = px.line(data, y=valid_price_cols, title="Price with Moving Averages")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid price action data available for plotting.")
        with tab2:
            momentum_cols = ['RSI', 'MACD', 'MACD_signal', 'TRIX', 'Ultimate_Osc', 'CMO']
            valid_momentum_cols = [col for col in momentum_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_momentum_cols:
                fig = px.line(data, y=valid_momentum_cols, title="Momentum Indicators")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid momentum indicators available for plotting.")
        with tab3:
            volatility_cols = ['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower', 'Keltner_Upper', 'Keltner_Lower']
            valid_volatility_cols = [col for col in volatility_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_volatility_cols:
                fig = px.line(data, y=valid_volatility_cols, title="Volatility Analysis")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid volatility indicators available for plotting.")
        with tab4:
            mc_results = monte_carlo_simulation(data)
            mc_df = pd.DataFrame(mc_results).T
            mc_df.columns = [f"Sim {i+1}" for i in range(len(mc_results))]
            fig = px.line(mc_df, title="Monte Carlo Price Simulations (30 Days)")
            st.plotly_chart(fig)
        with tab5:
            new_cols = ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF', 'VPT']
            valid_new_cols = [col for col in new_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if valid_new_cols:
                fig = px.line(data, y=valid_new_cols, title="New Indicators (Ichimoku, CMF, VPT)")
                st.plotly_chart(fig)
            else:
                st.warning("⚠️ No valid new indicators available for plotting.")
    elif symbol:
        st.warning("⚠️ No data available for the selected stock.")

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")

def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    intraday_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    return intraday_df.sort_values(by="Score", ascending=False).head(5)

def main():
    st.sidebar.title("🔍 Stock Search & Sector Selection")
    NSE_STOCKS = fetch_nse_stock_list()
    
    st.sidebar.subheader("Select Sectors")
    all_sectors = list(SECTORS.keys())
    selected_sectors = []
    for sector in all_sectors:
        if st.sidebar.checkbox(sector, value=True):
            selected_sectors.append(sector)
    
    selected_stocks = list(set([stock for sector in selected_sectors for stock in SECTORS[sector] if stock in NSE_STOCKS]))
    
    symbol = None
    selected_option = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=[""] + selected_stocks + ["Custom"],
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
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error("❌ Failed to load data for this symbol")
    else:
        display_dashboard(None, None, None, selected_stocks)

if __name__ == "__main__":
    main()
