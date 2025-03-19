import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
import numpy as np
from scipy.signal import find_peaks
import os
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1002411670969")

# Required columns for data validation
REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Cache TTL (Time to Live) in seconds (e.g., 1 hour)
CACHE_TTL = 3600

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
def fetch_nse_stock_list() -> List[str]:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception:
        st.warning("⚠️ Failed to fetch NSE stock list; using fallback list.")
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTIDRUGS.NS",
            "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS", "AATMAJ.NS", "AAVAS.NS",
            "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS", "ABFRL.NS",
        ]

@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol: str, period: str = "1y", interval: str = "1d", _timestamp: int = None) -> pd.DataFrame:
    """
    Fetch stock data with validation and cache invalidation based on freshness.
    """
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logger.warning(f"No data returned for {symbol} from Yahoo Finance.")
            return pd.DataFrame()
        
        # Validate required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        
        # Handle missing values by interpolating, then dropping any remaining NaNs
        data = data[REQUIRED_COLUMNS].astype(np.float32)  # Optimize memory with float32
        data = data.interpolate(method='linear').dropna()
        
        if len(data) < 15:
            logger.warning(f"Insufficient data for {symbol}: only {len(data)} days.")
            return pd.DataFrame()
        
        # Add timestamp for cache freshness
        data.attrs['fetch_time'] = time.time()
        return data
    except Exception as e:
        logger.warning(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def invalidate_cache_if_stale(data: pd.DataFrame) -> bool:
    """Check if cached data is stale based on TTL."""
    fetch_time = data.attrs.get('fetch_time', 0)
    return (time.time() - fetch_time) > CACHE_TTL

def geometric_brownian_motion(data: pd.DataFrame, simulations: int = 1000, days: int = 30) -> List[List[float]]:
    """Monte Carlo simulation using Geometric Brownian Motion."""
    if len(data) < 15:
        logger.warning("Insufficient data for GBM simulation.")
        return []
    
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 2:
        return []
    
    mu = returns.mean()  # Drift
    sigma = returns.std() or 0.01  # Volatility
    last_price = data['Close'].iloc[-1]
    dt = 1 / 252  # Daily time step (assuming 252 trading days per year)
    
    simulation_results = []
    for _ in range(simulations):
        prices = [last_price]
        for _ in range(days):
            drift = (mu - 0.5 * sigma**2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            price = prices[-1] * np.exp(drift + shock)
            prices.append(price)
        simulation_results.append(prices)
    return simulation_results

def rolling_rsi_optimization(data: pd.DataFrame, lookback: int = 50, windows=range(5, 15)) -> int:
    """Dynamically optimize RSI window using a rolling Sharpe ratio."""
    if len(data) < lookback:
        return 9  # Default window
    
    best_window, best_sharpe = 9, -float('inf')
    recent_data = data['Close'].iloc[-lookback:]
    returns = recent_data.pct_change().dropna()
    
    for window in windows:
        if len(recent_data) >= window + 1:
            rsi = ta.momentum.RSIIndicator(recent_data, window=window).rsi()
            signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
            strategy_returns = signals.shift(1) * returns
            sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
            if sharpe > best_sharpe:
                best_sharpe, best_window = sharpe, window
    return best_window

def detect_divergence(data: pd.DataFrame, lookback: int = 20) -> str:
    """Enhanced divergence detection with smoothing and longer lookback."""
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]) or len(data) < lookback:
        return "No Divergence"
    
    # Smooth price and RSI with a 5-day moving average
    price_smooth = data['Close'].rolling(window=5, min_periods=1).mean()
    rsi_smooth = data['RSI'].rolling(window=5, min_periods=1).mean()
    
    recent_highs = price_smooth[-lookback:].idxmax()
    recent_lows = price_smooth[-lookback:].idxmin()
    rsi_highs = rsi_smooth[-lookback:].idxmax()
    rsi_lows = rsi_smooth[-lookback:].idxmin()
    
    bullish_div = (recent_lows > rsi_lows) and (price_smooth[recent_lows] < price_smooth[-1]) and (rsi_smooth[rsi_lows] < rsi_smooth[-1])
    bearish_div = (recent_highs < rsi_highs) and (price_smooth[recent_highs] > price_smooth[-1]) and (rsi_smooth[rsi_highs] > rsi_smooth[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_volume_profile(data: pd.DataFrame, bins: int = None) -> pd.Series:
    """Dynamic binning for volume profile based on price range."""
    if len(data) < 2:
        return pd.Series()
    
    price_range = data['High'].max() - data['Low'].min()
    bins = bins or max(10, int(price_range / (data['Close'].std() or 1)))  # Dynamic bins
    price_bins = np.linspace(data['Low'].min(), data['High'].max(), bins + 1)
    volume_profile = np.zeros(bins)
    for i in range(bins):
        mask = (data['Close'] >= price_bins[i]) & (data['Close'] < price_bins[i + 1])
        volume_profile[i] = data['Volume'][mask].sum()
    # Normalize by total volume
    total_volume = volume_profile.sum()
    return pd.Series(volume_profile / total_volume if total_volume > 0 else volume_profile, index=price_bins[:-1])

def detect_waves(data: pd.DataFrame, min_length: int = 20) -> str:
    """Enhanced wave detection with Fibonacci confirmation."""
    if len(data) < min_length:
        return "No Clear Wave Pattern"
    
    peaks, _ = find_peaks(data['Close'], distance=5)
    troughs, _ = find_peaks(-data['Close'], distance=5)
    
    if len(peaks) < 3 or len(troughs) < 3:
        return "No Clear Wave Pattern"
    
    last_peak = data['Close'].iloc[peaks[-1]]
    last_trough = data['Close'].iloc[troughs[-1]]
    current_price = data['Close'].iloc[-1]
    
    # Fibonacci retracement check
    wave_range = last_peak - last_trough
    fib_618 = last_peak - 0.618 * wave_range
    
    if current_price > last_peak:
        return "Potential Uptrend (Wave 5?)"
    elif current_price < fib_618:
        return "Potential Downtrend (Wave C?) with Fibonacci Confirmation"
    return "No Clear Wave Pattern"

def analyze_stock(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or len(data) < 15:
        logger.warning(f"Insufficient data: only {len(data)} days available.")
        return data
    
    days = len(data)
    windows = {
        'rsi': min(rolling_rsi_optimization(data), max(5, days - 1)),
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
        'keltner': min(20, max(5, days - 1)),
        'supertrend': min(10, max(5, days - 1)),
    }

    try:
        if days >= windows['rsi'] + 1:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=windows['rsi']).rsi()
            data['Divergence'] = detect_divergence(data)
        
        if days >= max(windows['macd_slow'], windows['macd_fast']) + 1:
            macd = ta.trend.MACD(data['Close'], window_slow=windows['macd_slow'], 
                                window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
        
        if days >= windows['sma_20'] + 1:
            sma_20 = ta.trend.SMAIndicator(data['Close'], window=windows['sma_20']).sma_indicator()
            data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=windows['sma_20']).ema_indicator()
            bollinger = ta.volatility.BollingerBands(data['Close'], window=windows['bollinger'], window_dev=2)
            data['Middle_Band'] = sma_20
            data['Upper_Band'] = bollinger.bollinger_hband()
            data['Lower_Band'] = bollinger.bollinger_lband()
        
        if days >= windows['sma_50'] + 1:
            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=windows['sma_50']).sma_indicator()
            data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=windows['sma_50']).ema_indicator()
        
        if days >= windows['sma_200'] + 1:
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=windows['sma_200']).sma_indicator()
        
        if days >= windows['stoch'] + 1:
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], 
                                                    window=windows['stoch'], smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()
        
        if days >= windows['atr'] + 1:
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], 
                                                        window=windows['atr']).average_true_range()
        
        if days >= windows['adx'] + 1:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], 
                                                window=windows['adx']).adx()
        
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        
        if days >= windows['volume'] + 1:
            data['Avg_Volume'] = data['Volume'].rolling(window=windows['volume']).mean()
            data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
        
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - diff * 0.236
        data['Fib_38.2'] = high - diff * 0.382
        data['Fib_50.0'] = high - diff * 0.5
        data['Fib_61.8'] = high - diff * 0.618
        
        if days >= windows['ichimoku_w2'] + 1:
            ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=windows['ichimoku_w1'], 
                                                 window2=windows['ichimoku_w2'], window3=windows['ichimoku_w3'])
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Chikou'] = data['Close'].shift(-min(26, days - 1))
        
        if days >= windows['cmf'] + 1:
            data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], 
                                                              data['Volume'], window=windows['cmf']).chaikin_money_flow()
        
        if days >= windows['donchian'] + 1:
            donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], 
                                                    window=windows['donchian'])
            data['Donchian_Upper'] = donchian.donchian_channel_hband()
            data['Donchian_Lower'] = donchian.donchian_channel_lband()
            data['Donchian_Middle'] = donchian.donchian_channel_mband()
        
        # New indicators
        if days >= windows['keltner'] + 1:
            keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=windows['keltner'])
            data['Keltner_Upper'] = keltner.keltner_channel_hband()
            data['Keltner_Lower'] = keltner.keltner_channel_lband()
            data['Keltner_Middle'] = keltner.keltner_channel_mband()
        
        if days >= windows['supertrend'] + 1:
            supertrend = ta.trend.STCIndicator(data['High'], data['Low'], data['Close'], window=windows['supertrend'])
            data['Supertrend'] = supertrend.stc()
        
        data['Volume_Profile'] = calculate_volume_profile(data)
        data['Wave_Pattern'] = detect_waves(data)
    except Exception as e:
        logger.error(f"Error in analyze_stock: {str(e)}")
    
    return data

def calculate_buy_at(data: pd.DataFrame) -> Optional[float]:
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_rsi):
        buy_at = last_close * 0.99 if last_rsi < 30 else last_close
        return round(buy_at, 2)
    return None

def calculate_stop_loss(data: pd.DataFrame, atr_multiplier: float = 2.5) -> Optional[float]:
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_atr):
        if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
            atr_multiplier = 3.0
        stop_loss = last_close - (atr_multiplier * last_atr)
        return round(stop_loss, 2)
    return None

def calculate_target(data: pd.DataFrame, risk_reward_ratio: float = 3) -> Optional[float]:
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(stop_loss):
        risk = last_close - stop_loss
        if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) and data['ADX'].iloc[-1] > 25:
            risk_reward_ratio = 3
        target = last_close + (risk * risk_reward_ratio)
        return round(target, 2)
    return None

def fetch_fundamentals(symbol: str) -> Dict[str, float]:
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

def generate_recommendations(data: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, any]:
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        return recommendations
    
    last_close = data['Close'].iloc[-1]
    recommendations["Current Price"] = round(float(last_close), 2) if pd.notnull(last_close) else None
    
    # Weighted scores from multiple indicators
    buy_scores = []
    sell_scores = []
    
    if pd.notnull(last_close) and pd.notnull(data['Close'].iloc[0]):
        buy_scores.append(1 if last_close > data['Close'].iloc[0] else 0)
        sell_scores.append(1 if last_close < data['Close'].iloc[0] else 0)
    
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        buy_scores.append(2 if data['RSI'].iloc[-1] < 30 else 0)
        sell_scores.append(2 if data['RSI'].iloc[-1] > 70 else 0)
    
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        if pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            buy_scores.append(1 if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else 0)
            sell_scores.append(1 if data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] else 0)
    
    if 'Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns:
        if pd.notnull(data['Keltner_Upper'].iloc[-1]) and pd.notnull(data['Keltner_Lower'].iloc[-1]):
            buy_scores.append(1 if last_close > data['Keltner_Upper'].iloc[-1] else 0)
            sell_scores.append(1 if last_close < data['Keltner_Lower'].iloc[-1] else 0)
    
    if 'Supertrend' in data.columns and pd.notnull(data['Supertrend'].iloc[-1]):
        buy_scores.append(1 if data['Supertrend'].iloc[-1] > 0 else 0)
        sell_scores.append(1 if data['Supertrend'].iloc[-1] < 0 else 0)
    
    # Aggregate scores with weights (example weights)
    weights = [1.0, 2.0, 1.5, 1.2, 1.3]  # Adjust weights based on indicator reliability
    buy_score = sum(w * s for w, s in zip(weights[:len(buy_scores)], buy_scores))
    sell_score = sum(w * s for w, s in zip(weights[:len(sell_scores)], sell_scores))
    
    if buy_score >= 4:
        recommendations["Intraday"] = "Strong Buy"
        recommendations["Swing"] = "Buy"
        recommendations["Short-Term"] = "Buy"
    elif sell_score >= 4:
        recommendations["Intraday"] = "Strong Sell"
        recommendations["Swing"] = "Sell"
        recommendations["Short-Term"] = "Sell"
    elif buy_score > sell_score + 1:
        recommendations["Intraday"] = "Buy"
        recommendations["Swing"] = "Buy"
    elif sell_score > buy_score + 1:
        recommendations["Intraday"] = "Sell"
        recommendations["Swing"] = "Sell"
    
    recommendations["Buy At"] = calculate_buy_at(data)
    recommendations["Stop Loss"] = calculate_stop_loss(data)
    recommendations["Target"] = calculate_target(data)
    recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    return recommendations

def analyze_batch(stock_batch: List[str]) -> List[Dict[str, any]]:
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {str(e)}")
    return results

def analyze_stock_parallel(symbol: str) -> Optional[Dict[str, any]]:
    data = fetch_stock_data_cached(symbol)
    if not data.empty and len(data) >= 15:
        if invalidate_cache_if_stale(data):
            data = fetch_stock_data_cached(symbol, _timestamp=int(time.time()))  # Force refresh
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
    logger.warning(f"Skipping {symbol}: insufficient data ({len(data)} days).")
    return None

def update_progress(progress_bar, loading_text, progress_value, start_time, total_items, processed_items):
    progress_bar.progress(progress_value)
    elapsed_time = time.time() - start_time
    if processed_items > 0:
        eta = int((elapsed_time / processed_items) * (total_items - processed_items))
        loading_text.text(f"Processing {processed_items}/{total_items} stocks (ETA: {eta}s)")

def analyze_all_stocks(stock_list: List[str], batch_size: int = 50, price_range=None) -> pd.DataFrame:
    if st.session_state.cancel_operation:
        return pd.DataFrame()
    results = []
    total_items = len(stock_list)
    progress_bar = st.progress(0)
    loading_text = st.empty()
    start_time = time.time()
    
    for i in range(0, total_items, batch_size):
        if st.session_state.cancel_operation:
            break
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        processed_items = min(i + len(batch), total_items)
        update_progress(progress_bar, loading_text, processed_items / total_items, start_time, total_items, processed_items)
    
    progress_bar.empty()
    loading_text.empty()
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if price_range:
        results_df = results_df[results_df['Current Price'].notnull() & 
                                (results_df['Current Price'] >= price_range[0]) & 
                                (results_df['Current Price'] <= price_range[1])]
    return results_df.sort_values(by="Score", ascending=False).head(10)

def analyze_intraday_stocks(stock_list: List[str], batch_size: int = 50, price_range=None) -> pd.DataFrame:
    if st.session_state.cancel_operation:
        return pd.DataFrame()
    results = analyze_all_stocks(stock_list, batch_size, price_range)
    intraday_df = results[results["Intraday"].str.contains("Buy", na=False)]
    return intraday_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation: str) -> str:
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    return recommendation

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{"7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"}/sendMessage"
    payload = {"chat_id": "-1002411670969", "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        st.warning("⚠️ Failed to send Telegram message.")

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Select Price Range (₹)", min_value=0, max_value=10000, value=(100, 1000))
    
    if st.button("🚀 Generate Daily Top Picks"):
        st.session_state.cancel_operation = False
        cancel_button = st.button("❌ Cancel Analysis")
        if cancel_button:
            st.session_state.cancel_operation = True
        results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range)
        if not results_df.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Today's Top 10 Stocks")
            telegram_msg = f"*Top 10 Daily Stocks ({datetime.now().strftime('%d %b %Y')})*\nChat ID: {"-1002411670969"}\n\n"
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                    current_price = f"{row['Current Price']:.2f}" if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = f"{row['Buy At']:.2f}" if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = f"{row['Stop Loss']:.2f}" if pd.notnull(row['Stop Loss']) else "N/A"
                    target = f"{row['Target']:.2f}" if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    Current Price: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Long-Term: {colored_recommendation(row['Long-Term'])}
                    """, unsafe_allow_html=True)
                telegram_msg += f"*{row['Symbol']}*: ₹{current_price} - {row['Intraday']} (Score: {row['Score']})\n"
            send_telegram_message(telegram_msg)
            st.success("✅ Top picks sent to Telegram!")
    
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        st.session_state.cancel_operation = False
        cancel_button = st.button("❌ Cancel Intraday Analysis")
        if cancel_button:
            st.session_state.cancel_operation = True
        intraday_results = analyze_intraday_stocks(NSE_STOCKS, price_range=price_range)
        if not intraday_results.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Top 5 Intraday Stocks")
            telegram_msg = f"*Top 5 Intraday Stocks ({datetime.now().strftime('%d %b %Y')})*\nChat ID: {"-1002411670969"}\n\n"
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/7"):
                    current_price = f"{row['Current Price']:.2f}" if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = f"{row['Buy At']:.2f}" if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = f"{row['Stop Loss']:.2f}" if pd.notnull(row['Stop Loss']) else "N/A"
                    target = f"{row['Target']:.2f}" if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    Current Price: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}
                    """, unsafe_allow_html=True)
                telegram_msg += f"*{row['Symbol']}*: ₹{current_price} - {row['Intraday']} (Score: {row['Score']})\n"
            send_telegram_message(telegram_msg)
            st.success("✅ Intraday picks sent to Telegram!")
    
    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = f"{recommendations['Current Price']:.2f}" if recommendations['Current Price'] is not None else "N/A"
            st.metric("Current Price", f"₹{current_price}")
        with col2:
            buy_at = f"{recommendations['Buy At']:.2f}" if recommendations['Buy At'] is not None else "N/A"
            st.metric("Buy At", f"₹{buy_at}")
        with col3:
            stop_loss = f"{recommendations['Stop Loss']:.2f}" if recommendations['Stop Loss'] is not None else "N/A"
            st.metric("Stop Loss", f"₹{stop_loss}")
        with col4:
            target = f"{recommendations['Target']:.2f}" if recommendations['Target'] is not None else "N/A"
            st.metric("Target", f"₹{target}")
        
        st.subheader("📈 Trading Recommendations")
        cols = st.columns(4)
        for col, strategy in zip(cols, ["Intraday", "Swing", "Short-Term", "Long-Term"]):
            with col:
                st.markdown(f"**{strategy}**: {colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Action", "📉 Momentum", "📊 Volatility", "📈 Monte Carlo"])
        with tab1:
            fig = px.line(data, y=['Close', 'SMA_50', 'EMA_20'], title="Price with Moving Averages")
            st.plotly_chart(fig)
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], title="Momentum Indicators")
            st.plotly_chart(fig)
        with tab3:
            fig = px.line(data, y=['ATR', 'Upper_Band', 'Lower_Band', 'Keltner_Upper', 'Keltner_Lower'], title="Volatility Analysis")
            st.plotly_chart(fig)
        with tab4:
            mc_results = geometric_brownian_motion(data)
            if mc_results:
                mc_df = pd.DataFrame(mc_results).T
                fig = px.line(mc_df.mean(axis=1), title="GBM Monte Carlo Mean Path")
                st.plotly_chart(fig)
            else:
                st.write("Insufficient data for Monte Carlo simulation.")

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False

    st.sidebar.title("🔍 Stock Search")
    NSE_STOCKS = fetch_nse_stock_list()
    
    symbol = st.sidebar.selectbox("Choose stock:", [""] + NSE_STOCKS)
    
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 15:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error(f"❌ Failed to load sufficient data for {symbol} (only {len(data)} days).")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()