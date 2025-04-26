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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    "Agri": [
        "BHAGCHEM.NS", "CHAMBLFERT.NS", "FACT.NS", "GSFC.NS", "INSECTICID.NS",
        "IPL.NS", "KSCL.NS", "MANGCHEFER.NS", "NACLIND.NS", "PARADEEP.NS",
        "PIIND.NS", "RALLIS.NS", "RCF.NS", "SHARDACROP.NS", "UPL.NS"
    ],
    "Alcohol": [
        "ABDL.NS", "GLOBUSSPR.NS", "PICCADIL.NS", "RADICO.NS", "SDBL.NS",
        "TI.NS", "UBL.NS", "UNITDSPR.NS"
    ],
    "Automobile & Ancillaries": [
        "APOLLOTYRE.NS", "ASHOKLEY.NS", "CIEINDIA.NS", "JBMA.NS", "MINDACORP.NS",
        "SONACOMS.NS", "TATAMOTORS.NS", "UNOMINDA.NS"
    ],
    "Aviation": [
        "INDIGO.NS", "SPICEJET.NS"
    ],
    "Bank": [
        "AUBANK.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "CAPITALSFB.NS", "CUB.NS",
        "FEDERALBNK.NS", "FINOPB.NS", "INDIANB.NS", "INDUSINDBK.NS", "KARURVYSYA.NS",
        "KTKBANK.NS", "RBLBANK.NS", "SBIN.NS", "TMB.NS"
    ],
    "Capital Goods": [
        "BHEL.NS", "LT.NS", "SIEMENS.NS"
    ],
    "Chemicals": [
        "AARTIIND.NS", "ACI.NS", "AETHER.NS", "ANURAS.NS", "CHEMPLASTS.NS",
        "GHCL.NS", "GNFC.NS", "HSCL.NS", "JUBLINGREA.NS", "PIDILITIND.NS",
        "SPLPETRO.NS", "SRF.NS", "TATACHEM.NS", "UPL.NS"
    ],
    "Construction Materials": [
        "ACC.NS", "AMBUJACEM.NS", "DECCANCE.NS", "EVERESTIND.NS", "HEIDELBERG.NS",
        "INDIACEM.NS", "JKLAKSHMI.NS", "MANGLMCEM.NS", "NCLIND.NS", "NUVOCO.NS",
        "ORIENTCEM.NS", "RAMCOCEM.NS", "RAMCOIND.NS", "SAGCEM.NS", "STARCEMENT.NS",
        "ULTRACEMCO.NS"
    ],
    "Consumer Durables": [
        "ABFRL.NS", "AVL.NS", "BAJAJELEC.NS", "BOROLTD.NS", "CROMPTON.NS",
        "EPACK.NS", "ETERNAL.NS", "FIRSTCRY.NS", "GOCOLORS.NS", "NYKAA.NS",
        "ORIENTELEC.NS", "PGEL.NS", "SHOPERSTOP.NS", "STOVEKRAFT.NS", "STYLEBAAZA.NS",
        "SWIGGY.NS", "TTKPRESTIG.NS", "VAIBHAVGBL.NS", "VERANDA.NS", "WEL.NS",
        "WHIRLPOOL.NS"
    ],
    "Diamond & Jewellery": [
        "PCJEWELLER.NS", "TITAN.NS"
    ],
    "Diversified": [
        "ANSALBU.NS", "ASAHIINDIA.NS", "CASTROLIND.NS", "CENTURYPLY.NS",
        "FRANKLIN.BO", "JINDALSAW.NS", "JSL.NS", "RKFORGE.NS", "RODIUM.BO",
        "SARDAEN.NS", "SHERVANI.BO", "SHYAMMETL.NS", "SUNDRMFAST.NS", "WELCORP.NS"
    ],
    "Electricals": [
        "KEI.NS", "POLYCAB.NS"
    ],
    "ETF": [
        "NIFTYBEES.NS"
    ],
    "Finance": [
        "ABCAPITAL.NS", "GICRE.NS", "HDFCLIFE.NS", "HUDCO.NS", "ICICIPRULI.NS",
        "JIOFIN.NS", "LICI.NS", "MOTILALOFS.NS", "NAM_INDIA.NS", "PFC.NS",
        "RECLTD.NS", "SBICARD.NS", "SBILIFE.NS", "SHRIRAMFIN.NS"
    ],
    "FMCG": [
        "AVANTIFEED.NS", "BALRAMCHIN.NS", "BIKAJI.NS", "CCL.NS", "DABUR.NS",
        "DEVYANI.NS", "EIDPARRY.NS", "EMAMILTD.NS", "GODREJCP.NS", "ITC.NS",
        "JUBLFOOD.NS", "JYOTHYLAB.NS", "KRBL.NS", "LTFOODS.NS", "MARICO.NS",
        "SAPPHIRE.NS", "TATACONSUM.NS", "TI.NS", "TRIVENI.NS", "UNITDSPR.NS",
        "VBL.NS"
    ],
    "Footwear & Accessories": [
        "BATAINDIA.NS", "CAMPUS.NS", "GOLDIAM.NS", "KALYANKJIL.NS", "KHADIM.NS",
        "LEHA.NS", "LIBERTSHOE.NS", "RAJESHEXPO.NS", "RELAXO.NS", "SENCO.NS",
        "SKYGOLD.NS", "TBZ.NS", "TIMEXWATCH.NS"
    ],
    "Healthcare": [
        "ASTERDM.NS", "AUROPHARMA.NS", "BIOCON.NS", "FORTIS.NS", "KIMS.NS",
        "LAURUSLABS.NS", "NATCOPHARM.NS", "ZYDUSLIFE.NS"
    ],
    "Hospitality": [
        "EIHOTEL.NS", "INDHOTEL.NS"
    ],
    "Industrial Gases & Fuels": [
        "LINDEINDIA.NS"
    ],
    "Infrastructure": [
        "ADANIPORTS.NS", "GMRINFRA.NS"
    ],
    "Insurance": [
        "ICICIGI.NS", "NIACL.NS"
    ],
    "Logistics": [
        "BLUEDART.NS", "CONCOR.NS"
    ],
    "Manufacturing": [
        "ASIANPAINT.NS", "BERGEPAINT.NS"
    ],
    "Media & Entertainment": [
        "SUNTV.NS", "ZEEL.NS"
    ],
    "Metals & Mining (DerivedMS)": [
        "ASAHIINDIA.NS", "BERGEPAINT.NS", "CASTROLIND.NS", "CENTURYPLY.NS",
        "GPIL.NS", "JINDALSAW.NS", "JSL.NS", "JSWSTEEL.NS", "KANSAINER.NS",
        "SHYAMMETL.NS", "SUNDRMFAST.NS", "TATASTEEL.NS", "WELCORP.NS"
    ],
    "Miscellaneous": [
        "ADANIENT.NS", "ADANIGREEN.NS"
    ],
    "Oil & Gas": [
        "BPCL.NS", "CHENNPETRO.NS", "HINDPETRO.NS", "IOC.NS", "MRPL.NS",
        "OIL.NS", "ONGC.NS", "PETRONET.NS", "RELIANCE.NS"
    ],
    "Paper": [
        "JKPAPER.NS", "KUANTUM.NS", "METROGLOBL.NS", "NRAGRINDQ.NS", "PAKKA.NS",
        "PDMJEPAPER.NS", "RUCHIRA.NS", "SANGALPAPR.NS", "SESHAPAPER.NS",
        "SHREYANIND.NS", "STARPAPER.NS", "SUBAMPAPER.NS", "SVJENTERPR.NS",
        "TNPL.NS", "WSTCSTPAPR.NS"
    ],
    "Photographic Products": [
        "JINDALPHOT.NS"
    ],
    "Plastic Products": [
        "SUPREMEIND.NS"
    ],
    "Power": [
        "ACMESOLAR.NS", "ADANIGREEN.NS", "ADANIENSOL.NS", "ADANIPOWER.NS",
        "ATGL.NS", "BPCL.NS", "CHENNPETRO.NS", "EXIDEIND.NS", "GAIL.NS",
        "GSPL.NS", "HBLENGINE.NS", "HINDPETRO.NS", "IEX.NS", "NLCINDIA.NS",
        "NTPC.NS", "OIL.NS", "ONGC.NS", "PETRONET.NS", "POWERGRID.NS",
        "REFEX.NS", "RELINFRA.NS", "TATAPOWER.NS"
    ],
    "Real Estate": [
        "DLF.NS", "GODREJPROP.NS"
    ],
    "Retailing": [
        "DMART.NS", "TRENT.NS"
    ],
    "Ship Building": [
        "ABSMARINE.NS", "COCHINSHIP.NS", "GRSE.NS"
    ],
    "Software & IT Services": [
        "BSOFT.NS", "CYIENT.NS", "EMUDHRA.NS", "FSL.NS", "HAPPSTMNDS.NS",
        "HCLTECH.NS", "HEXT.NS", "INFY.NS", "INTELLECT.NS", "NAZARA.NS",
        "NEWGEN.NS", "NIITMTS.NS", "QUESS.NS", "RATEGAIN.NS", "RSYSTEMS.NS",
        "SONATSOFTW.NS", "TANLA.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS",
        "WIPRO.NS", "ZAGGLE.NS", "ZENSARTECH.NS"
    ],
    "Telecom": [
        "BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"
    ],
    "Textiles": [
        "PAGEIND.NS", "RAYMOND.NS"
    ],
    "Trading": [
        "ADANIPOWER.NS"
    ]
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
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

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

def train_ml_model(data, features=None, target_shift=-1, test_size=0.2, random_state=42):
    """
    Train a Random Forest Classifier to predict price movement and return feature importance.
    
    Parameters:
    - data: DataFrame with OHLCV and technical indicators
    - features: List of feature columns (e.g., ['RSI', 'MACD', 'ATR'])
    - target_shift: Shift for target variable (e.g., -1 for next day's price)
    - test_size: Proportion of data for testing
    - random_state: Seed for reproducibility
    
    Returns:
    - model: Trained Random Forest Classifier
    - feature_importance: Dict mapping features to their importance scores
    - accuracy: Model accuracy on test set
    """
    if data.empty or len(data) < 50:
        return None, {}, 0.0
    
    # Default features if none provided
    if features is None:
        features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'CMF', 'TRIX', 
                   'Ultimate_Osc', 'CMO', 'VPT', 'SlowK', 'SlowD']
    
    # Ensure features exist in data and are numeric
    valid_features = [f for f in features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
    if not valid_features:
        st.warning("⚠️ No valid features available for ML training.")
        return None, {}, 0.0
    
    # Prepare X (features) and y (target)
    X = data[valid_features].dropna()
    if len(X) < 50:
        st.warning("⚠️ Insufficient data after dropping NaNs.")
        return None, {}, 0.0
    
    # Create target: 1 if next day's close > current close, else 0
    y = (data['Close'].shift(target_shift) > data['Close']).astype(int)
    
    # Align X and y
    valid_idx = X.index.intersection(y.index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    if len(X) < 50:
        st.warning("⚠️ Insufficient aligned data for ML training.")
        return None, {}, 0.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state
    )
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.warning(f"⚠️ Failed to train ML model: {str(e)}")
        return None, {}, 0.0
    
    # Evaluate model
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    except Exception as e:
        st.warning(f"⚠️ Failed to evaluate ML model: {str(e)}")
        accuracy = 0.0
    
    # Get feature importance
    feature_importance = dict(zip(valid_features, model.feature_importances_))
    
    return model, feature_importance, accuracy
    

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
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal')
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
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    return score / 3

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
    # Updated Moving Average Section
    try:
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_20'] = None
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
    # New Indicators
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
        data['CMO'] = ta.momentum.ChandeMomentumOscillator(data['Close'], window=14).chande_momentum_oscillator()
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
    if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25:
        atr_multiplier = 3.0
    else:
        atr_multiplier = 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25:
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
        
        # Train ML model to get feature importance
        features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'CMF', 'TRIX', 
                   'Ultimate_Osc', 'CMO', 'VPT', 'SlowK', 'SlowD']
        model, feature_importance, accuracy = train_ml_model(data, features=features)
        
        if not feature_importance:
            st.warning("⚠️ ML model training failed. Using equal weights.")
            feature_importance = {f: 1.0 / len(features) for f in features}
        
        # Normalize feature importance to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
        else:
            feature_importance = {f: 1.0 / len(features) for f in features}
        
        # Calculate weighted signals
        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
            rsi_weight = feature_importance.get('RSI', 1.0 / len(features))
            if isinstance(data['RSI'].iloc[-1], (int, float)):
                if data['RSI'].iloc[-1] < 30:
                    buy_score += 2 * rsi_weight
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2 * rsi_weight
        
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd_weight = feature_importance.get('MACD', 1.0 / len(features))
            if data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None:
                if isinstance(data['MACD'].iloc[-1], (int, float)) and isinstance(data['MACD_signal'].iloc[-1], (int, float)):
                    if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                        buy_score += 1 * macd_weight
                    elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                        sell_score += 1 * macd_weight
        
        # Bollinger Bands
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            bb_weight = feature_importance.get('Lower_Band', 1.0 / len(features))  # Proxy for Bollinger
            if isinstance(data['Close'].iloc[-1], (int, float)) and isinstance(data['Lower_Band'].iloc[-1], (int, float)) and isinstance(data['Upper_Band'].iloc[-1], (int, float)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1 * bb_weight
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1 * bb_weight
        
        # VWAP
        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None:
            vwap_weight = feature_importance.get('VWAP', 1.0 / len(features))
            if isinstance(data['VWAP'].iloc[-1], (int, float)) and isinstance(data['Close'].iloc[-1], (int, float)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1 * vwap_weight
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1 * vwap_weight
        
        # Volume
        if 'Volume' in data.columns and data['Volume'].iloc[-1] is not None:
            volume_weight = feature_importance.get('Volume', 1.0 / len(features))  # Proxy for volume
            avg_volume = data['Volume'].rolling(window=10).mean().iloc[-1]
            if isinstance(data['Volume'].iloc[-1], (int, float)) and isinstance(avg_volume, (int, float)):
                if data['Volume'].iloc[-1] > avg_volume * 1.2:
                    buy_score += 1 * volume_weight
                elif data['Volume'].iloc[-1] < avg_volume * 0.5:
                    sell_score += 1 * volume_weight
        
        # Divergence
        if 'Divergence' in data.columns:
            divergence_weight = feature_importance.get('Divergence', 1.0 / len(features))  # Proxy
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1 * divergence_weight
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1 * divergence_weight
        
        # Ichimoku Cloud
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
            ichimoku_weight = feature_importance.get('Ichimoku_Span_A', 1.0 / len(features))
            if (isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float)) and 
                isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float)) and 
                isinstance(data['Close'].iloc[-1], (int, float))):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 1 * ichimoku_weight
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1 * ichimoku_weight
                    recommendations["Ichimoku_Trend"] = "Sell"
        
        # Chaikin Money Flow
        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
            cmf_weight = feature_importance.get('CMF', 1.0 / len(features))
            if isinstance(data['CMF'].iloc[-1], (int, float)):
                if data['CMF'].iloc[-1] > 0:
                    buy_score += 1 * cmf_weight
                elif data['CMF'].iloc[-1] < 0:
                    sell_score += 1 * cmf_weight
        
        # Donchian Channels
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
            donchian_weight = feature_importance.get('Donchian_Upper', 1.0 / len(features))
            if (isinstance(data['Donchian_Upper'].iloc[-1], (int, float)) and 
                isinstance(data['Donchian_Lower'].iloc[-1], (int, float)) and 
                isinstance(data['Close'].iloc[-1], (int, float))):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1 * donchian_weight
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1 * donchian_weight
                    recommendations["Breakout"] = "Sell"
        
        # Mean Reversion (RSI + Bollinger)
        if 'RSI' in data.columns and 'Lower_Band' in data.columns:
            rsi_weight = feature_importance.get('RSI', 1.0 / len(features))
            bb_weight = feature_importance.get('Lower_Band', 1.0 / len(features))
            combined_weight = (rsi_weight + bb_weight) / 2
            if (isinstance(data['RSI'].iloc[-1], (int, float)) and 
                isinstance(data['Lower_Band'].iloc[-1], (int, float)) and 
                isinstance(data['Close'].iloc[-1], (int, float))):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1]:
                    buy_score += 2 * combined_weight
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2 * combined_weight
                    recommendations["Mean_Reversion"] = "Sell"
        
        # Ichimoku Strong Signals
        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns:
            ichimoku_weight = feature_importance.get('Ichimoku_Tenkan', 1.0 / len(features))
            if (isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float)) and 
                isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float)) and 
                isinstance(data['Close'].iloc[-1], (int, float)) and 
                isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float))):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1 * ichimoku_weight
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1 * ichimoku_weight
                    recommendations["Ichimoku_Trend"] = "Strong Sell"
        
        # Moving Average Crossover
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            sma_weight = feature_importance.get('SMA_20', 1.0 / len(features))
            if (isinstance(data['SMA_20'].iloc[-1], (int, float)) and 
                isinstance(data['SMA_50'].iloc[-1], (int, float)) and 
                isinstance(data['SMA_20'].iloc[-2], (int, float)) and 
                isinstance(data['SMA_50'].iloc[-2], (int, float))):
                if (data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] and 
                    data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]):
                    buy_score += 1 * sma_weight
                elif (data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1] and 
                      data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]):
                    sell_score += 1 * sma_weight
        
        # Keltner Channels
        if 'Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns:
            keltner_weight = feature_importance.get('Keltner_Upper', 1.0 / len(features))
            if (isinstance(data['Keltner_Upper'].iloc[-1], (int, float)) and 
                isinstance(data['Keltner_Lower'].iloc[-1], (int, float)) and 
                isinstance(data['Close'].ilocKILL[-1], (int, float))):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 1 * keltner_weight
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 1 * keltner_weight
        
        # TRIX
        if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None:
            trix_weight = feature_importance.get('TRIX', 1.0 / len(features))
            if isinstance(data['TRIX'].iloc[-1], (int, float)):
                if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                    buy_score += 1 * trix_weight
                elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                    sell_score += 1 * trix_weight
        
        # Ultimate Oscillator
        if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None:
            uo_weight = feature_importance.get('Ultimate_Osc', 1.0 / len(features))
            if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float)):
                if data['Ultimate_Osc'].iloc[-1] < 30:
                    buy_score += 1 * uo_weight
                elif data['Ultimate_Osc'].iloc[-1] > 70:
                    sell_score += 1 * uo_weight
        
        # Chande Momentum Oscillator
        if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None:
            cmo_weight = feature_importance.get('CMO', 1.0 / len(features))
            if isinstance(data['CMO'].iloc[-1], (int, float)):
                if data['CMO'].iloc[-1] < -50:
                    buy_score += 1 * cmo_weight
                elif data['CMO'].iloc[-1] > 50:
                    sell_score += 1 * cmo_weight
        
        # Volume Price Trend
        if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None:
            vpt_weight = feature_importance.get('VPT', 1.0 / len(features))
            if isinstance(data['VPT'].iloc[-1], (int, float)):
                if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                    buy_score += 1 * vpt_weight
                elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                    sell_score += 1 * vpt_weight
        
        # Fundamentals
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            fundamental_weight = 0.1  # Fixed weight for fundamentals
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 1 * fundamental_weight
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 0.5 * fundamental_weight
        
        # Generate recommendations based on scores
        추천사항["Intraday"] = "Strong Buy" if buy_score >= 3 else "Strong Sell" if sell_score >= 3 else "Hold"
        
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        
        # Add ML accuracy to recommendations
        recommendations["ML_Accuracy"] = accuracy if model else 0.0
        recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
        
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    
    return recommendations
    
def backtest_strategy(symbol, data, strategy="Intraday", lookback_period="1y", risk_reward_ratio=3):
    """
    Backtest a trading strategy for a given stock.
    
    Parameters:
    - symbol: Stock symbol (e.g., "RELIANCE.NS")
    - data: Historical OHLCV data (pandas DataFrame)
    - strategy: Strategy to backtest ("Intraday", "Mean_Reversion", "Breakout", "Ichimoku_Trend")
    - lookback_period: Period for backtesting (e.g., "1y")
    - risk_reward_ratio: Risk/reward ratio for target calculation
    
    Returns:
    - Dictionary with backtest metrics (returns, win rate, Sharpe ratio, max drawdown, trades)
    """
    if data.empty or len(data) < 50:
        return {
            "Total Return": 0,
            "Win Rate": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0,
            "Number of Trades": 0,
            "Equity Curve": [],
            "Equity Dates": [],
            "Trades": [],
            "Error": "Insufficient data for backtesting"
        }

    # Filter data to lookback period
    end_date = data.index[-1]
    days = {"1y": 365, "3y": 1095, "5y": 1825}
    start_date = end_date - pd.Timedelta(days=days.get(lookback_period, 365))
    data = data.loc[start_date:end_date].copy()

    # Initialize variables
    trades = []
    portfolio_value = 100000  # Initial capital in INR
    equity_curve = [portfolio_value]
    equity_dates = [data.index[0]]
    daily_returns = []
    position = None
    entry_price = 0
    stop_loss = 0
    target = 0
    transaction_cost = 0.001  # 0.1% per trade (buy + sell = 0.2% total)

    # Simulate trades day by day
    for i in range(1, len(data)):
        # Get subset of data up to current day
        current_data = data.iloc[:i+1]
        recommendations = generate_recommendations(current_data, symbol)

        current_price = current_data['Close'].iloc[-1]
        buy_signal = recommendations[strategy] in ["Buy", "Strong Buy"]
        sell_signal = recommendations[strategy] in ["Sell", "Strong Sell"]

        # Calculate daily return for Sharpe ratio
        prev_portfolio_value = equity_curve[-1]
        if position == "Long":
            unrealized_return = (current_price - entry_price) / entry_price
            daily_return = (portfolio_value * (1 + unrealized_return)) / prev_portfolio_value - 1
        else:
            daily_return = 0
        daily_returns.append(daily_return)

        # Check for exit conditions if in a position
        if position == "Long":
            if current_price <= stop_loss:
                # Hit stop loss
                trade_return = (stop_loss - entry_price) / entry_price
                trade_return -= transaction_cost  # Apply transaction cost on exit
                portfolio_value *= (1 + trade_return)
                trades.append({
                    "Entry Date": current_data.index[-2],
                    "Exit Date": current_data.index[-1],
                    "Entry Price": entry_price,
                    "Exit Price": stop_loss,
                    "Return": trade_return,
                    "Outcome": "Loss"
                })
                position = None
            elif current_price >= target:
                # Hit target
                trade_return = (target - entry_price) / entry_price
                trade_return -= transaction_cost  # Apply transaction cost on exit
                portfolio_value *= (1 + trade_return)
                trades.append({
                    "Entry Date": current_data.index[-2],
                    "Exit Date": current_data.index[-1],
                    "Entry Price": entry_price,
                    "Exit Price": target,
                    "Return": trade_return,
                    "Outcome": "Win"
                })
                position = None
            elif sell_signal:
                # Exit on sell signal
                trade_return = (current_price - entry_price) / entry_price
                trade_return -= transaction_cost  # Apply transaction cost on exit
                portfolio_value *= (1 + trade_return)
                outcome = "Win" if trade_return > 0 else "Loss"
                trades.append({
                    "Entry Date": current_data.index[-2],
                    "Exit Date": current_data.index[-1],
                    "Entry Price": entry_price,
                    "Exit Price": current_price,
                    "Return": trade_return,
                    "Outcome": outcome
                })
                position = None

        # Check for entry conditions
        if position is None and buy_signal and recommendations["Buy At"] is not None:
            entry_price = recommendations["Buy At"]
            stop_loss = recommendations["Stop Loss"]
            target = recommendations["Target"]
            if stop_loss is not None and target is not None and entry_price > stop_loss:
                # Calculate risk-reward ratio
                risk = entry_price - stop_loss
                reward = target - entry_price
                risk_reward = reward / risk if risk > 0 else 0
                if risk_reward >= 2:  # Increased minimum risk-reward ratio to 2:1
                    portfolio_value *= (1 - transaction_cost)  # Apply transaction cost on entry
                    position = "Long"

        equity_curve.append(portfolio_value)
        equity_dates.append(data.index[i])

    # Calculate metrics
    total_return = (portfolio_value - 100000) / 100000
    trade_returns = [trade["Return"] for trade in trades] if trades else [0]
    wins = sum(1 for trade in trades if trade["Outcome"] == "Win")
    win_rate = wins / len(trades) if trades else 0

    # Calculate Sharpe ratio using daily portfolio returns
    daily_returns = np.array(daily_returns)
    trading_days = len(data) - 1
    if trading_days > 0 and np.std(daily_returns) != 0:
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        annualization_factor = 252
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(annualization_factor)
    else:
        sharpe_ratio = 0

    # Calculate maximum drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdowns = (rolling_max - equity_series) / rolling_max
    max_drawdown = drawdowns.max() if not drawdowns.empty else 0

    return {
        "Total Return": round(total_return * 100, 2),
        "Win Rate": round(win_rate * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Number of Trades": len(trades),
        "Equity Curve": equity_curve,
        "Equity Dates": equity_dates,
        "Trades": trades,
        "Error": None
    }
    
def backtest_batch(stocks, strategy="Intraday", lookback_period="1y", progress_callback=None):
    """
    Backtest a strategy across multiple stocks.
    
    Parameters:
    - stocks: List of stock symbols
    - strategy: Strategy to backtest
    - lookback_period: Period for backtesting
    - progress_callback: Function to update progress
    
    Returns:
    - DataFrame with backtest results for each stock
    """
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(backtest_strategy, symbol, fetch_stock_data_cached(symbol), strategy, lookback_period): symbol for symbol in stocks}
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                result = future.result()
                result["Symbol"] = symbol
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Error backtesting {symbol}: {str(e)}")
            if progress_callback:
                progress_callback((i + 1) / len(stocks))
    
    results_df = pd.DataFrame([r for r in results if r["Error"] is None])
    if results_df.empty:
        return pd.DataFrame()
    return results_df.sort_values(by="Total Return", ascending=False)
    
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
        result = {
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
            "ML_Accuracy": recommendations.get("ML_Accuracy", 0.0)
        }
        return result
    return None
def analyze_all_stocks(stock_list, batch_size=50, progress_callback=None):
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
        
def display_backtest_results(backtest_results, symbol=None, batch_results=None):
    """
    Display backtest results in the Streamlit dashboard.
    
    Parameters:
    - backtest_results: Dictionary with backtest metrics for a single stock
    - symbol: Stock symbol (optional, for single stock display)
    - batch_results: DataFrame with backtest results for multiple stocks (optional)
    """
    if symbol and backtest_results and backtest_results["Error"] is None:
        st.subheader(f"📉 Backtest Results for {symbol.split('.')[0]}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{backtest_results['Total Return']}%")
        with col2:
            st.metric("Win Rate", f"{backtest_results['Win Rate']}%")
        with col3:
            st.metric("Sharpe Ratio", f"{backtest_results['Sharpe Ratio']}")
        with col4:
            st.metric("Max Drawdown", f"{backtest_results['Max Drawdown']}%")
        st.write(f"Number of Trades: {backtest_results['Number of Trades']}")
        
        # Plot equity curve using actual dates
        equity_curve = backtest_results["Equity Curve"]
        equity_dates = backtest_results["Equity Dates"]
        if equity_curve and equity_dates:
            equity_df = pd.DataFrame({
                "Date": equity_dates,
                "Portfolio Value": equity_curve
            })
            fig = px.line(equity_df, x="Date", y="Portfolio Value", title="Equity Curve")
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ No equity curve data available for plotting.")
        
        # Display trade log
        if backtest_results["Trades"]:
            st.subheader("📋 Trade Log")
            trade_df = pd.DataFrame(backtest_results["Trades"])
            trade_df["Return"] = trade_df["Return"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(trade_df[["Entry Date", "Exit Date", "Entry Price", "Exit Price", "Return", "Outcome"]])
        
        # Benchmark comparison
        st.subheader("📊 Benchmark Comparison")
        nifty_data = fetch_stock_data_cached("^NSEI", period="1y", interval="1d")
        if not nifty_data.empty:
            nifty_return = (nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[0]) / nifty_data['Close'].iloc[0] * 100
            st.metric("NIFTY 50 Return", f"{nifty_return:.2f}%")
            strategy_return = backtest_results['Total Return']
            alpha = strategy_return - nifty_return
            st.metric("Alpha (vs NIFTY 50)", f"{alpha:.2f}%")
        else:
            st.warning("⚠️ Unable to fetch NIFTY 50 data for comparison.")
    
    if batch_results is not None and not batch_results.empty:
        st.subheader("🏆 Batch Backtest Results")
        st.dataframe(
            batch_results[["Symbol", "Total Return", "Win Rate", "Sharpe Ratio", "Max Drawdown", "Number of Trades"]]
            .style.format({
                "Total Return": "{:.2f}%",
                "Win Rate": "{:.2f}%",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown": "{:.2f}%"
            })
        )

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    # Existing code for top picks and intraday picks
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        results_df = analyze_all_stocks(
            selected_stocks,
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
                    ml_accuracy = row.get('ML_Accuracy', 0.0) * 100
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
                    ML Model Accuracy: {ml_accuracy:.2f}%
                    """, unsafe_allow_html=True)
            # Batch backtest for top picks
            if st.button("📉 Backtest Top Picks"):
                progress_bar = st.progress(0)
                loading_text = st.empty()
                loading_messages = itertools.cycle([
                    "Running backtests...", "Simulating trades...", "Calculating metrics...",
                    "Analyzing performance...", "Finalizing results..."
                ])
                top_stocks = results_df["Symbol"].tolist()
                batch_results = backtest_batch(
                    top_stocks,
                    strategy="Intraday",
                    progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
                )
                progress_bar.empty()
                loading_text.empty()
                display_backtest_results(None, batch_results=batch_results)
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
                    ml_accuracy = row.get('ML_Accuracy', 0.0) * 100
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    ML Model Accuracy: {ml_accuracy:.2f}%
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
        
        # ML Prediction
        st.subheader("🤖 Machine Learning Prediction")
        features = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'CMF', 'TRIX', 
                   'Ultimate_Osc', 'CMO', 'VPT', 'SlowK', 'SlowD']
        valid_features = [f for f in features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
        if valid_features and len(data) > 10:
            latest_data = data[valid_features].iloc[-1:].dropna()
            if not latest_data.empty:
                model, _, _ = train_ml_model(data, features=valid_features)
                if model:
                    prediction = model.predict(latest_data)[0]
                    probability = model.predict_proba(latest_data)[0][1]
                    st.write(f"Prediction: {'Price Increase' if prediction == 1 else 'Price Decrease'}")
                    st.write(f"Probability of Increase: {probability*100:.2f}%")
                else:
                    st.warning("⚠️ Unable to generate ML prediction.")
            else:
                st.warning("⚠️ Insufficient recent data for ML prediction.")
        else:
            st.warning("⚠️ Insufficient features or data for ML prediction.")
        
        # Backtest Section
        st.subheader("📉 Backtest Strategy")
        strategy_to_backtest = st.selectbox(
            "Select Strategy to Backtest",
            options=["Intraday", "Mean_Reversion", "Breakout", "Ichimoku_Trend"],
            key="backtest_strategy"
        )
        lookback_period = st.selectbox(
            "Select Lookback Period",
            options=["1y", "3y", "5y"],
            key="backtest_period"
        )
        data_interval = st.selectbox(
            "Select Data Interval for Backtest",
            options=["1d", "1h"] if strategy_to_backtest == "Intraday" else ["1d"],
            key="backtest_interval"
        )
        if st.button("Run Backtest"):
            progress_bar = st.progress(0)
            loading_text = st.empty()
            loading_messages = itertools.cycle([
                "Running backtest...", "Simulating trades...", "Calculating metrics...",
                "Analyzing performance...", "Finalizing results..."
            ])
            backtest_data = fetch_stock_data_cached(symbol, period=lookback_period, interval=data_interval)
            if not backtest_data.empty:
                backtest_data = analyze_stock(backtest_data)
                backtest_results = backtest_strategy(
                    symbol,
                    backtest_data,
                    strategy=strategy_to_backtest,
                    lookback_period=lookback_period
                )
                progress_bar.progress(1.0)
                loading_text.empty()
                display_backtest_results(backtest_results, symbol)
            else:
                st.warning("⚠️ Failed to load data for backtesting.")
        
        # Updated Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Price Action", "📉 Momentum", "📊 Volatility", 
            "📈 Monte Carlo", "📉 New Indicators", "🤖 ML Insights"
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
        
        with tab6:
            st.subheader("🤖 Machine Learning Insights")
            if recommendations.get('ML_Accuracy', 0.0) > 0:
                st.write(f"ML Model Accuracy: {recommendations['ML_Accuracy']*100:.2f}%")
                # Train model again to get feature importance (or cache it)
                _, feature_importance, _ = train_ml_model(data, features=features)
                if feature_importance:
                    importance_df = pd.DataFrame({
                        'Indicator': list(feature_importance.keys()),
                        'Importance': [v*100 for v in feature_importance.values()]
                    }).sort_values(by='Importance', ascending=False)
                    st.write("Feature Importance (%):")
                    fig = px.bar(importance_df, x='Indicator', y='Importance', 
                               title="Indicator Importance in Price Prediction")
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Unable to compute feature importance.")
            else:
                st.warning("⚠️ ML model not trained or insufficient data.")
    
    elif symbol:
        st.warning("⚠️ No data available for the selected stock.")
        
def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")

def analyze_intraday_stocks(stock_list, batch_size=50, progress_callback=None):
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
    
    all_sectors = list(SECTORS.keys())
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=all_sectors,
        default=all_sectors
    )
    
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
