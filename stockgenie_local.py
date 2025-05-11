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
import sqlite3

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.2210.91 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
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

def retry(max_retries=3, delay=2, backoff_factor=2, jitter=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        st.warning(f"Rate limit hit. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
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
        time.sleep(random.uniform(3, 5))
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
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
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_volatility < 0.02:
            score += 0.5
        elif atr_volatility > 0.05:
            score -= 0.5
    return min(max(score / 3.5, 0), 1)

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

def calculate_buy_at_row(row):
    if pd.notnull(row['RSI']) and row['RSI'] < 30:
        return round(row['Close'] * 0.99, 2)
    return round(row['Close'], 2)

def calculate_stop_loss_row(row, atr_multiplier=2.5):
    if pd.notnull(row['ATR']):
        atr_multiplier = 3.0 if pd.notnull(row['ADX']) and row['ADX'] > 25 else 1.5
        stop_loss = row['Close'] - (atr_multiplier * row['ATR'])
        if stop_loss < row['Close'] * 0.9:
            stop_loss = row['Close'] * 0.9
        return round(stop_loss, 2)
    return None

def calculate_target_row(row, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss_row(row)
    if stop_loss is not None:
        risk = row['Close'] - stop_loss
        adjusted_ratio = min(risk_reward_ratio, 5) if pd.notnull(row['ADX']) and row['ADX'] > 25 else min(risk_reward_ratio, 3)
        target = row['Close'] + (risk * adjusted_ratio)
        if target > row['Close'] * 1.2:
            target = row['Close'] * 1.2
        return round(target, 2)
    return None

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
        
        # RSI (Boosted for RSI <= 20)
        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
            if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['RSI'].iloc[-1] <= 20:
                    buy_score += 4
                elif data['RSI'].iloc[-1] < 30:
                    buy_score += 2
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2
        
        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None:
            if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    buy_score += 1
                elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                    sell_score += 1
        
        # Bollinger Bands
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
            if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1
        
        # VWAP
        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
            if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1
        
        # Volume Analysis
        if ('Volume' in data.columns and data['Volume'].iloc[-1] is not None and 
            'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None):
            volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
            if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 2
                elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    sell_score += 2
                elif volume_ratio < 0.5:
                    sell_score += 1
        
        # Volume Spikes
        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None:
            if data['Volume_Spike'].iloc[-1] and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 1
                else:
                    sell_score += 1
        
        # Divergence
        if 'Divergence' in data.columns:
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1
        
        # Ichimoku Cloud
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None:
            if (isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
        
        # Chaikin Money Flow
        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
            if isinstance(data['CMF'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMF'].iloc[-1] > 0:
                    buy_score += 1
                elif data['CMF'].iloc[-1] < 0:
                    sell_score += 1
        
        # Donchian Channels
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Close'].iloc[-1] is not None:
            if (isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Donchian_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"
        
        # Mean Reversion
        if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
            if (isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] >= data['Lower_Band'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"
        
        # Ichimoku Trend
        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and data['Close'].iloc[-1] is not None:
            if (isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating))):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"
        
        # Keltner Channels
        if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
            data['Close'].iloc[-1] is not None):
            if (isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 1
        
        # TRIX
        if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None:
            if isinstance(data['TRIX'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['TRIX'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                    buy_score += 1
                elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                    sell_score += 1
        
        # Ultimate Oscillator
        if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None:
            if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Ultimate_Osc'].iloc[-1] < 30:
                    buy_score += 1
                elif data['Ultimate_Osc'].iloc[-1] > 70:
                    sell_score += 1
        
        # Chande Momentum Oscillator
        if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None:
            if isinstance(data['CMO'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMO'].iloc[-1] < -50:
                    buy_score += 1
                elif data['CMO'].iloc[-1] > 50:
                    sell_score += 1
        
        # Volume Price Trend
        if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None:
            if isinstance(data['VPT'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['VPT'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                    buy_score += 1
                elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                    sell_score += 1
        
        # Fibonacci Retracements
        if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
            data['Close'].iloc[-1] is not None):
            current_price = data['Close'].iloc[-1]
            fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                          data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
            for level in fib_levels:
                if isinstance(level, (int, float, np.integer, np.floating)) and abs(current_price - level) / current_price < 0.01:
                    if current_price > level:
                        buy_score += 1
                    else:
                        sell_score += 1
        
        # Parabolic SAR
        if ('Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and 
            data['Close'].iloc[-1] is not None):
            if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                    sell_score += 1
        
        # OBV
        if ('OBV' in data.columns and data['OBV'].iloc[-1] is not None and 
            data['OBV'].iloc[-2] is not None):
            if isinstance(data['OBV'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['OBV'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['OBV'].iloc[-1] > data['OBV'].iloc[-2]:
                    buy_score += 1
                elif data['OBV'].iloc[-1] < data['OBV'].iloc[-2]:
                    sell_score += 1
        
        # Fundamentals
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 2
            elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
                sell_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 1
            elif fundamentals['RevenueGrowth'] < 0:
                sell_score += 0.5
        
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
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

def generate_recommendations_row(row, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": row['Close'], "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }
    try:
        buy_score = 0
        sell_score = 0
        
        # RSI
        if pd.notnull(row['RSI']):
            if row['RSI'] <= 20:
                buy_score += 4
            elif row['RSI'] < 30:
                buy_score += 2
            elif row['RSI'] > 70:
                sell_score += 2
        
        # MACD
        if pd.notnull(row['MACD']) and pd.notnull(row['MACD_signal']):
            if row['MACD'] > row['MACD_signal']:
                buy_score += 1
            elif row['MACD'] < row['MACD_signal']:
                sell_score += 1
        
        # Bollinger Bands
        if pd.notnull(row['Close']) and pd.notnull(row['Lower_Band']) and pd.notnull(row['Upper_Band']):
            if row['Close'] < row['Lower_Band']:
                buy_score += 1
            elif row['Close'] > row['Upper_Band']:
                sell_score += 1
        
        # VWAP
        if pd.notnull(row['VWAP']) and pd.notnull(row['Close']):
            if row['Close'] > row['VWAP']:
                buy_score += 1
            elif row['Close'] < row['VWAP']:
                sell_score += 1
        
        # Volume Analysis
        if pd.notnull(row['Volume']) and pd.notnull(row['Avg_Volume']):
            volume_ratio = row['Volume'] / row['Avg_Volume']
            if volume_ratio > 1.5 and pd.notnull(row['Close']) and pd.notnull(row['Close_prev']):
                if row['Close'] > row['Close_prev']:
                    buy_score += 2
                elif row['Close'] < row['Close_prev']:
                    sell_score += 2
            elif volume_ratio < 0.5:
                sell_score += 1
        
        # Volume Spikes
        if pd.notnull(row['Volume_Spike']) and row['Volume_Spike']:
            if pd.notnull(row['Close']) and pd.notnull(row['Close_prev']):
                if row['Close'] > row['Close_prev']:
                    buy_score += 1
                else:
                    sell_score += 1
        
        # Divergence
        if pd.notnull(row['Divergence']):
            if row['Divergence'] == "Bullish Divergence":
                buy_score += 1
            elif row['Divergence'] == "Bearish Divergence":
                sell_score += 1
        
        # Ichimoku Cloud
        if pd.notnull(row['Ichimoku_Span_A']) and pd.notnull(row['Ichimoku_Span_B']) and pd.notnull(row['Close']):
            if row['Close'] > max(row['Ichimoku_Span_A'], row['Ichimoku_Span_B']):
                buy_score += 1
                recommendations["Ichimoku_Trend"] = "Buy"
            elif row['Close'] < min(row['Ichimoku_Span_A'], row['Ichimoku_Span_B']):
                sell_score += 1
        
        # Chaikin Money Flow
        if pd.notnull(row['CMF']):
            if row['CMF'] > 0:
                buy_score += 1
            elif row['CMF'] < 0:
                sell_score += 1
        
        # Donchian Channels
        if pd.notnull(row['Donchian_Upper']) and pd.notnull(row['Donchian_Lower']) and pd.notnull(row['Close']):
            if row['Close'] > row['Donchian_Upper']:
                buy_score += 1
                recommendations["Breakout"] = "Buy"
            elif row['Close'] < row['Donchian_Lower']:
                sell_score += 1
                recommendations["Breakout"] = "Sell"
        
        # Mean Reversion
        if pd.notnull(row['RSI']) and pd.notnull(row['Lower_Band']) and pd.notnull(row['Upper_Band']) and pd.notnull(row['Close']):
            if row['RSI'] < 30 and row['Close'] >= row['Lower_Band']:
                buy_score += 2
                recommendations["Mean_Reversion"] = "Buy"
            elif row['RSI'] > 70 and row['Close'] >= row['Upper_Band']:
                sell_score += 2
                recommendations["Mean_Reversion"] = "Sell"
        
        # Ichimoku Trend
        if pd.notnull(row['Ichimoku_Tenkan']) and pd.notnull(row['Ichimoku_Kijun']) and pd.notnull(row['Close']) and pd.notnull(row['Ichimoku_Span_A']):
            if row['Ichimoku_Tenkan'] > row['Ichimoku_Kijun'] and row['Close'] > row['Ichimoku_Span_A']:
                buy_score += 1
                recommendations["Ichimoku_Trend"] = "Strong Buy"
            elif row['Ichimoku_Tenkan'] < row['Ichimoku_Kijun'] and row['Close'] < row['Ichimoku_Span_B']:
                sell_score += 1
                recommendations["Ichimoku_Trend"] = "Strong Sell"
        
        # Keltner Channels
        if pd.notnull(row['Keltner_Upper']) and pd.notnull(row['Keltner_Lower']) and pd.notnull(row['Close']):
            if row['Close'] < row['Keltner_Lower']:
                buy_score += 1
            elif row['Close'] > row['Keltner_Upper']:
                sell_score += 1
        
        # TRIX
        if pd.notnull(row['TRIX']) and pd.notnull(row['TRIX_prev']):
            if row['TRIX'] > 0 and row['TRIX'] > row['TRIX_prev']:
                buy_score += 1
            elif row['TRIX'] < 0 and row['TRIX'] < row['TRIX_prev']:
                sell_score += 1
        
        # Ultimate Oscillator
        if pd.notnull(row['Ultimate_Osc']):
            if row['Ultimate_Osc'] < 30:
                buy_score += 1
            elif row['Ultimate_Osc'] > 70:
                sell_score += 1
        
        # Chande Momentum Oscillator
        if pd.notnull(row['CMO']):
            if row['CMO'] < -50:
                buy_score += 1
            elif row['CMO'] > 50:
                sell_score += 1
        
        # Volume Price Trend
        if pd.notnull(row['VPT']) and pd.notnull(row['VPT_prev']):
            if row['VPT'] > row['VPT_prev']:
                buy_score += 1
            elif row['VPT'] < row['VPT_prev']:
                sell_score += 1
        
        # Fibonacci Retracements
        if pd.notnull(row['Fib_23.6']) and pd.notnull(row['Fib_38.2']) and pd.notnull(row['Close']):
            fib_levels = [row['Fib_23.6'], row['Fib_38.2'], row['Fib_50.0'], row['Fib_61.8']]
            for level in fib_levels:
                if pd.notnull(level) and abs(row['Close'] - level) / row['Close'] < 0.01:
                    if row['Close'] > level:
                        buy_score += 1
                    else:
                        sell_score += 1
        
        # Parabolic SAR
        if pd.notnull(row['Parabolic_SAR']) and pd.notnull(row['Close']):
            if row['Close'] > row['Parabolic_SAR']:
                buy_score += 1
            elif row['Close'] < row['Parabolic_SAR']:
                sell_score += 1
        
        # OBV
        if pd.notnull(row['OBV']) and pd.notnull(row['OBV_prev']):
            if row['OBV'] > row['OBV_prev']:
                buy_score += 1
            elif row['OBV'] < row['OBV_prev']:
                sell_score += 1
        
        # Fundamentals
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 2
            elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
                sell_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 1
            elif fundamentals['RevenueGrowth'] < 0:
                sell_score += 0.5
        
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
            recommendations["Intraday"] = "Buy" if net_score >= 

3 else "Hold"
            recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
            recommendations["Short-Term"] = "Buy" if net_score >= 1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        elif net_score < 0:
            recommendations["Intraday"] = "Sell" if net_score <= -3 else "Hold"
            recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
            recommendations["Short-Term"] = "Sell" if net_score <= -1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        
        recommendations["Buy At"] = calculate_buy_at_row(row)
        recommendations["Stop Loss"] = calculate_stop_loss_row(row)
        recommendations["Target"] = calculate_target_row(row)
        recommendations["Score"] = min(max(buy_score - sell_score, -7), 7)
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

def backtest_stock(data, strategy="Swing"):
    data = data.copy()
    data['Close_prev'] = data['Close'].shift(1)
    data['TRIX_prev'] = data['TRIX'].shift(1)
    data['VPT_prev'] = data['VPT'].shift(1)
    data['OBV_prev'] = data['OBV'].shift(1)
    
    key_cols = ['Open', 'Close', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX']
    data = data.dropna(subset=key_cols)
    if len(data) < 2:
        return None
    
    initial_capital = 100000
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        recommendations = generate_recommendations_row(row)
        
        if position == 0 and recommendations[strategy] in ["Buy", "Strong Buy"]:
            entry_price = next_row['Open']
            shares = capital // entry_price
            if shares > 0:
                position = shares
                capital -= shares * entry_price
                trades.append({"entry_date": next_row.name, "entry_price": entry_price})
        
        elif position > 0 and recommendations[strategy] in ["Sell", "Strong Sell"]:
            exit_price = next_row['Open']
            capital += position * exit_price
            trades[-1]["exit_date"] = next_row.name
            trades[-1]["exit_price"] = exit_price
            position = 0
    
    if position > 0:
        exit_price = data['Close'].iloc[-1]
        capital += position * exit_price
        trades[-1]["exit_date"] = data.index[-1]
        trades[-1]["exit_price"] = exit_price
    
    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    wins = sum(1 for trade in trades if trade["exit_price"] > trade["entry_price"])
    win_rate = (wins / len(trades) * 100) if trades else 0
    
    return {
        "total_return": total_return,
        "trades": len(trades),
        "win_rate": win_rate,
        "trade_details": trades
    }

def init_database():
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_picks (
            date TEXT,
            symbol TEXT,
            score REAL,
            current_price REAL,
            buy_at REAL,
            stop_loss REAL,
            target REAL,
            intraday TEXT,
            swing TEXT,
            short_term TEXT,
            long_term TEXT,
            mean_reversion TEXT,
            breakout TEXT,
            ichimoku_trend TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.close()

def insert_top_picks(results_df):
    conn = sqlite3.connect('stock_picks.db')
    for _, row in results_df.head(5).iterrows():
        conn.execute('''
            INSERT OR IGNORE INTO daily_picks (
                date, symbol, score, current_price, buy_at, stop_loss, target,
                intraday, swing, short_term, long_term, mean_reversion, breakout, ichimoku_trend
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'), row['Symbol'], row['Score'], row['Current Price'],
            row['Buy At'], row['Stop Loss'], row['Target'], row['Intraday'], row['Swing'],
            row['Short-Term'], row['Long-Term'], row['Mean_Reversion'], row['Breakout'], row['Ichimoku_Trend']
        ))
    conn.commit()
    conn.close()

def analyze_batch(stock_batch):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
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
        time.sleep(5)
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    return results_df.sort_values(by="Score", ascending=False).head(5)

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
    intraday_df = results_df[results_df Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal."Intraday"].str.contains("Buy", na=False)]
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

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")

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
        insert_top_picks(results_df)
        progress_bar.empty()
        loading_text.empty()
        if not results_df.empty:
            st.subheader("🏆 Today's Top 5 Stocks")
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
    
    if st.button("📜 View Historical Picks"):
        conn = sqlite3.connect('stock_picks.db')
        history_df = pd.read_sql_query("SELECT * FROM daily_picks ORDER BY date DESC", conn)
        conn.close()
        if not history_df.empty:
            st.subheader("📜 Historical Top Picks")
            date_filter = st.selectbox("Filter by Date", ["All"] + sorted(history_df['date'].unique(), reverse=True))
            if date_filter != "All":
                history_df = history_df[history_df['date'] == date_filter]
            st.dataframe(history_df)
        else:
            st.warning("⚠️ No historical data available.")
    
    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = recommendations['Current Price'] if recommendations['Current Price'] is not None else "N/A"
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
        with col2:
            buy_at = recommendations['Buy At'] if recommendations['Buy At'] is not None else "N/A"
            st.metric("Buy At", f"₹{buy_at}")
        with col3:
            stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
            st.metric("Target", f"₹{target}")
        
        st.subheader("📈 Trading Recommendations")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Intraday**: {colored_recommendation(recommendations['Intraday'])}")
            st.write(f"**Swing**: {colored_recommendation(recommendations['Swing'])}")
        with col2:
            st.write(f"**Short-Term**: {colored_recommendation(recommendations['Short-Term'])}")
            st.write(f"**Long-Term**: {colored_recommendation(recommendations['Long-Term'])}")
        with col3:
            st.write(f"**Mean Reversion**: {colored_recommendation(recommendations['Mean_Reversion'])}")
            st.write(f"**Breakout**: {colored_recommendation(recommendations['Breakout'])}")
        with col4:
            st.write(f"**Ichimoku Trend**: {colored_recommendation(recommendations['Ichimoku_Trend'])}")
            st.write(f"**Confidence Score**: {recommendations['Score']}/7")
        
        if st.button("🔍 Backtest Swing Strategy"):
            backtest_results = backtest_stock(data)
            if backtest_results:
                st.subheader("📈 Backtest Results (Swing Strategy)")
                st.write(f"**Total Return**: {backtest_results['total_return']:.2f}%")
                st.write(f"**Number of Trades**: {backtest_results['trades']}")
                st.write(f"**Win Rate**: {backtest_results['win_rate']:.2f}%")
                with st.expander("Trade Details"):
                    for trade in backtest_results["trade_details"]:
                        st.write(f"Entry: {trade['entry_date']} @ ₹{trade['entry_price']:.2f}, "
                                 f"Exit: {trade['exit_date']} @ ₹{trade['exit_price']:.2f}")
            else:
                st.warning("⚠️ Insufficient data for backtesting.")
        
        st.subheader("📊 Technical Indicators")
        indicators = [
            ("RSI", data['RSI'].iloc[-1], TOOLTIPS['RSI']),
            ("MACD", data['MACD'].iloc[-1], TOOLTIPS['MACD']),
            ("ATR", data['ATR'].iloc[-1], TOOLTIPS['ATR']),
            ("ADX", data['ADX'].iloc[-1], TOOLTIPS['ADX']),
            ("VWAP", data['VWAP'].iloc[-1], TOOLTIPS['VWAP']),
            ("CMF", data['CMF'].iloc[-1], TOOLTIPS['CMF']),
            ("TRIX", data['TRIX'].iloc[-1], TOOLTIPS['TRIX']),
            ("Ultimate Oscillator", data['Ultimate_Osc'].iloc[-1], TOOLTIPS['Ultimate_Osc']),
            ("Chande Momentum Oscillator", data['CMO'].iloc[-1], TOOLTIPS['CMO']),
            ("Volume Price Trend", data['VPT'].iloc[-1], TOOLTIPS['VPT']),
        ]
        for name, value, tooltip_text in indicators:
            if pd.notnull(value):
                st.write(f"{tooltip(name, tooltip_text)}: {value:.2f}")
        
        st.subheader("📉 Price Chart")
        fig = px.line(data, x=data.index, y='Close', title=f"{symbol} Price Movement")
        fig.add_scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='orange'))
        fig.add_scatter(x=data.index, y=data['SMA_200'], name='SMA 200', line=dict(color='purple'))
        st.plotly_chart(fig)
        
        st.subheader("🔮 Monte Carlo Simulation")
        simulations = monte_carlo_simulation(data)
        simulation_df = pd.DataFrame(simulations).T
        simulation_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(simulation_df))]
        fig_mc = px.line(simulation_df, title="Monte Carlo Price Projections")
        st.plotly_chart(fig_mc)
        
        st.subheader("⚠️ Risk Assessment")
        risk = assess_risk(data)
        st.write(f"**Risk Level**: {risk}")

def main():
    init_database()
    st.set_page_config(page_title="StockGenie Pro", page_icon="📈", layout="wide")
    stock_list = fetch_nse_stock_list()
    all_stocks = sorted(list(set(stock_list)))
    
    st.sidebar.title("⚙️ Controls")
    sector = st.sidebar.selectbox("Select Sector", ["All"] + list(SECTORS.keys()))
    if sector == "All":
        selected_stocks = all_stocks
    else:
        selected_stocks = SECTORS[sector]
    
    symbol = st.sidebar.selectbox("Select Stock", ["None"] + selected_stocks)
    
    if symbol != "None":
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error(f"⚠️ No data available for {symbol}.")
    else:
        display_dashboard(selected_stocks=selected_stocks)

if __name__ == "__main__":
    main()
