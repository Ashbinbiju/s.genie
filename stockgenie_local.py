import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time
import requests
import io
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import os
import joblib
import itertools
import telegram
import asyncio
from telegram.error import TelegramError
from statsmodels.tsa.arima.model import ARIMA
import dask.dataframe as dd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
from itertools import cycle
from threading import Lock

warnings.filterwarnings("ignore")

# API Keys
NEWSAPI_KEY = "ed58659895e84dfb8162a8bb47d8525e"
ALPHA_VANTAGE_KEY = "TCAUKYUCIDZ6PI57"
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
TELEGRAM_CHAT_ID = "-1002411670969"

# Tooltips
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Volatility",
    "MACD": "Trend following indicator",
    "Stop Loss": "Risk management level",
}

# Proxy API URL with HTTPS filter
PROXY_API_URL = "https://proxylist.geonode.com/api/proxy-list?limit=500&page=1&sort_by=lastChecked&sort_type=desc&speed=fast&protocols=https"
PROXY_REFRESH_INTERVAL = 3600  # Refresh every hour (in seconds)
PROXY_TEST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"  # Test against NSE
proxy_list = []
proxy_cycle = None
proxy_lock = Lock()
last_proxy_refresh = 0

def fetch_proxies_from_api():
    global proxy_list, proxy_cycle, last_proxy_refresh
    try:
        response = requests.get(PROXY_API_URL, timeout=10)
        response.raise_for_status()
        proxy_data = response.json()['data']
        proxies = [f"{proxy['protocols'][0]}://{proxy['ip']}:{proxy['port']}" 
                   for proxy in proxy_data if 'ip' in proxy and 'port' in proxy and 'protocols' in proxy and 'https' in proxy['protocols']]
        if not proxies:
            st.warning("No HTTPS proxies returned from API.")
            return False
        
        # Validate proxies
        valid_proxies = []
        with ThreadPoolExecutor(max_workers=20) as executor:  # Increased workers for faster validation
            futures = {executor.submit(test_proxy, proxy): proxy for proxy in proxies[:100]}  # Test first 100
            for future in as_completed(futures):
                if future.result():
                    valid_proxies.append(futures[future])
        
        if valid_proxies:
            with proxy_lock:
                proxy_list = valid_proxies
                proxy_cycle = cycle(proxy_list)
                last_proxy_refresh = time.time()
            st.info(f"Fetched and validated {len(valid_proxies)} HTTPS proxies from API.")
            return True
        else:
            st.warning("No valid HTTPS proxies found after testing.")
            return False
    except Exception as e:
        st.warning(f"Failed to fetch proxies from API: {e}")
        return False

def test_proxy(proxy):
    try:
        response = requests.get(PROXY_TEST_URL, proxies={"http": proxy, "https": proxy}, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def refresh_proxies_if_needed():
    global last_proxy_refresh
    current_time = time.time()
    if current_time - last_proxy_refresh > PROXY_REFRESH_INTERVAL or not proxy_list:
        with proxy_lock:
            if current_time - last_proxy_refresh > PROXY_REFRESH_INTERVAL or not proxy_list:
                fetch_proxies_from_api()

def get_proxy(fallback_to_no_proxy=False):
    refresh_proxies_if_needed()
    with proxy_lock:
        if proxy_cycle and proxy_list:
            return next(proxy_cycle)
        elif fallback_to_no_proxy:
            st.warning("No valid proxies available, proceeding without proxy.")
            return None
        else:
            return "http://localhost:8080"  # Fallback proxy (replace if needed)

def retry(max_retries=3, delay=5, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        # Handle case where args might be empty
                        target = args[0] if args else "unknown"
                        st.warning(f"Max retries reached for {target} in {func.__name__}: {str(e)}")
                        return pd.DataFrame() if "fetch_stock_data" in func.__name__ else {'P/E': np.inf, 'EPS': 0, 'DividendYield': 0}
                    if "Too Many Requests" in str(e) or "401" in str(e):
                        sleep_time = delay * (backoff_factor ** retries) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                    else:
                        target = args[0] if args else "unknown"
                        st.warning(f"Error in {func.__name__} for {target}: {str(e)}")
                        return pd.DataFrame() if "fetch_stock_data" in func.__name__ else {'P/E': np.inf, 'EPS': 0, 'DividendYield': 0}
        return wrapper
    return decorator

@retry(max_retries=3, delay=5)
def fetch_nse_stock_list():
    proxy = get_proxy(fallback_to_no_proxy=True)
    proxies = {"http": proxy, "https": proxy} if proxy else None
    response = requests.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv", proxies=proxies, timeout=10)
    response.raise_for_status()
    return [f"{symbol}.NS" for symbol in pd.read_csv(io.StringIO(response.text))['SYMBOL']]

def load_cache(cache_type="data"):
    cache_file = f"stock_{cache_type}_cache.joblib"
    return joblib.load(cache_file) if os.path.exists(cache_file) else {}

def save_cache(cache, cache_type="data"):
    joblib.dump(cache, f"stock_{cache_type}_cache.joblib", compress=3)

@retry(max_retries=3, delay=5)
def fetch_stock_data_batch(symbols, period="5y", interval="1d"):
    cache = load_cache("data")
    data_dict = {}
    symbols_to_fetch = [s for s in symbols if f"{s}_{period}_{interval}" not in cache]
    
    if symbols_to_fetch:
        try:
            proxy = get_proxy(fallback_to_no_proxy=True)
            session = requests.Session()
            session.proxies = {"http": proxy, "https": proxy} if proxy else None
            data = yf.download(symbols_to_fetch + ["^VIX"], period=period, interval=interval, group_by='ticker', threads=True, session=session)
            for symbol in symbols_to_fetch:
                cache_key = f"{symbol}_{period}_{interval}"
                symbol_data = data[symbol].dropna() if symbol in data.columns.levels[0] else pd.DataFrame()
                if not symbol_data.empty:
                    symbol_data['VIX'] = data['^VIX']['Close'].reindex(symbol_data.index, method='ffill') if '^VIX' in data.columns.levels[0] else np.nan
                    cache[cache_key] = symbol_data
                    data_dict[symbol] = symbol_data
                else:
                    data_dict[symbol] = pd.DataFrame()
            save_cache(cache, "data")
        except Exception as e:
            st.warning(f"Batch fetch failed: {e}")
    return {s: cache.get(f"{s}_{period}_{interval}", pd.DataFrame()) for s in symbols}

@retry(max_retries=3, delay=5)
def fetch_fundamentals(symbol):
    cache = load_cache("fundamentals")
    cache_key = f"{symbol}_fundamentals"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        proxy = get_proxy(fallback_to_no_proxy=True)
        session = requests.Session()
        session.proxies = {"http": proxy, "https": proxy} if proxy else None
        stock = yf.Ticker(symbol, session=session)
        info = stock.info
        fundamentals = {
            'P/E': info.get('trailingPE', np.inf),
            'EPS': info.get('trailingEps', 0),
            'DividendYield': info.get('dividendYield', 0) * 100
        }
        cache[cache_key] = fundamentals
        save_cache(cache, "fundamentals")
        return fundamentals
    except Exception as e:
        return {'P/E': np.inf, 'EPS': 0, 'DividendYield': 0}

def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def apply_pca(df, features, n_components=2):
    cleaned_data = df[features].dropna()
    if len(cleaned_data) < 1:
        return pd.DataFrame(index=df.index, columns=[f'PC{i+1}' for i in range(n_components)])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    return pd.DataFrame(pca_result, index=cleaned_data.index, columns=[f'PC{i+1}' for i in range(n_components)])

def analyze_stock(data):
    if data.empty or len(data) < 15:
        return data
    
    data = handle_outliers(data, 'Close')
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX', 'RSI', 'MACD', 'ATR']
    available_features = [f for f in features if f in data.columns and data[f].notna().sum() > 0]
    if len(available_features) >= 2:
        corr_matrix = data[available_features].corr()
        if corr_matrix['Close'].abs().mean() > 0.8:
            available_features = ['Close', 'Volume', 'VIX'] if 'VIX' in data.columns else ['Close', 'Volume']
        pca_df = apply_pca(data, available_features)
        data = pd.concat([data, pca_df], axis=1)
    else:
        data['PC1'] = np.nan
        data['PC2'] = np.nan
    
    return data

def arima_predict(data, days=5):
    try:
        model = ARIMA(data['Close'].dropna(), order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        return forecast.iloc[-1]
    except Exception:
        return None

def calculate_buy_at(data):
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    return round(last_close * 0.99 if last_rsi < 30 else last_close, 2) if pd.notnull(last_close) and pd.notnull(last_rsi) else None

def calculate_stop_loss(data):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    return round(last_close - (2.5 * last_atr), 2) if pd.notnull(last_close) and pd.notnull(last_atr) else None

def calculate_target(data):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    return round(last_close + (3 * (last_close - stop_loss)), 2) if pd.notnull(last_close) and pd.notnull(stop_loss) else None

def generate_recommendations(data, symbol):
    rec = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0,
        "Next Price": None, "Price Direction": "Neutral"
    }
    if data.empty or 'Close' not in data.columns:
        return rec
    
    rec["Current Price"] = round(float(data['Close'].iloc[-1]), 2) if pd.notnull(data['Close'].iloc[-1]) else None
    buy_score = sell_score = 0
    last_close = data['Close'].iloc[-1]

    if pd.notnull(last_close) and pd.notnull(data['Close'].iloc[0]):
        buy_score += 1 if last_close > data['Close'].iloc[0] else 0
        sell_score += 1 if last_close < data['Close'].iloc[0] else 0
    
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        buy_score += 2 if data['RSI'].iloc[-1] < 30 else 0
        sell_score += 2 if data['RSI'].iloc[-1] > 70 else 0
    
    if 'MACD' in data.columns and pd.notnull(data['MACD'].iloc[-1]):
        buy_score += 1 if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else 0
        sell_score += 1 if data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] else 0

    fundamentals = fetch_fundamentals(symbol)
    if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
        buy_score += 1

    if buy_score >= 3:
        rec["Intraday"] = "Strong Buy"
        rec["Swing"] = "Buy"
        rec["Short-Term"] = "Buy"
    elif sell_score >= 3:
        rec["Intraday"] = "Strong Sell"
        rec["Swing"] = "Sell"
        rec["Short-Term"] = "Sell"
    elif buy_score > sell_score:
        rec["Intraday"] = "Buy"
    elif sell_score > buy_score:
        rec["Intraday"] = "Sell"

    rec["Buy At"] = calculate_buy_at(data)
    rec["Stop Loss"] = calculate_stop_loss(data)
    rec["Target"] = calculate_target(data)
    rec["Score"] = max(0, min(buy_score - sell_score, 7))
    
    rec["Next Price"] = arima_predict(data)
    rec["Price Direction"] = "Up" if rec["Next Price"] > last_close else "Down" if rec["Next Price"] < last_close else "Neutral"
    
    return rec

def analyze_stock_parallel(symbol, data):
    if not data.empty:
        data = analyze_stock(data)
        rec = generate_recommendations(data, symbol)
        return {
            "Symbol": symbol, "Current Price": rec["Current Price"], "Buy At": rec["Buy At"],
            "Stop Loss": rec["Stop Loss"], "Target": rec["Target"], "Intraday": rec["Intraday"],
            "Swing": rec["Swing"], "Short-Term": rec["Short-Term"], "Score": rec["Score"],
            "Next Price": rec["Next Price"], "Price Direction": rec["Price Direction"]
        }
    return None

def analyze_batch(stock_batch):
    results = []
    with ThreadPoolExecutor(max_workers=min(20, len(stock_batch))) as executor:
        data_dict = fetch_stock_data_batch(stock_batch)
        valid_batch = [s for s in stock_batch if not data_dict[s].empty and len(data_dict[s].dropna()) >= 15]
        futures = {executor.submit(analyze_stock_parallel, s, data_dict[s]): s for s in valid_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"Error processing {symbol}: {str(e)}")
    return results

def analyze_all_stocks(stock_list, batch_size=15, price_range=None, progress_callback=None):
    if st.session_state.cancel_operation:
        return pd.DataFrame()

    total_items = len(stock_list)
    st.info(f"Analyzing {total_items} stocks (~{total_items // 120} minutes)...")
    results = []
    start_time = time.time()
    
    for i in range(0, total_items, batch_size):
        if st.session_state.cancel_operation:
            break
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        processed_items = min(i + len(batch), total_items)
        if progress_callback:
            progress_callback(processed_items / total_items, start_time, total_items, processed_items)
    
    results_df = dd.from_pandas(pd.DataFrame([r for r in results if r is not None]), npartitions=4)
    if len(results_df.index.compute()) > 0:
        for col in ['Current Price', 'Buy At', 'Stop Loss', 'Target', 'Next Price']:
            results_df[col] = results_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x, meta=(col, 'float64'))
        if price_range:
            results_df = results_df[results_df['Current Price'].notnull() & 
                                   (results_df['Current Price'].between(price_range[0], price_range[1]))]
    return results_df.compute().sort_values(by="Score", ascending=False).head(3)

def colored_recommendation(rec):
    return f"🟢 {rec}" if "Buy" in rec else f"🔴 {rec}" if "Sell" in rec else f"🟡 {rec}"

def update_progress(progress_bar, loading_text, progress_value, loading_messages, start_time, total_items, processed_items):
    progress_bar.progress(progress_value)
    message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    elapsed = time.time() - start_time
    eta = timedelta(seconds=int((elapsed / processed_items) * (total_items - processed_items))) if processed_items else "N/A"
    loading_text.text(f"{message}{dots} | {processed_items}/{total_items} (ETA: {eta})")

async def send_telegram_message(message):
    try:
        bot = telegram.Bot(token="7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps")
        await bot.send_message(chat_id="-1002411670969", text=message, parse_mode='HTML')
        return True
    except TelegramError as e:
        st.error(f"❌ Failed to send Telegram message: {str(e)}")
        return False

def display_stock_analysis(symbol, data, rec):
    st.header(f"📋 {symbol.split('.')[0]} Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"₹{rec['Current Price']:.2f}" if rec['Current Price'] else "N/A")
    with col2:
        st.metric("Buy At", f"₹{rec['Buy At']:.2f}" if rec['Buy At'] else "N/A")
    with col3:
        st.metric("Stop Loss", f"₹{rec['Stop Loss']:.2f}" if rec['Stop Loss'] else "N/A")
    with col4:
        st.metric("Target", f"₹{rec['Target']:.2f}" if rec['Target'] else "N/A")
    
    st.subheader("📈 Recommendations")
    cols = st.columns(3)
    for col, strat in zip(cols, ["Intraday", "Swing", "Short-Term"]):
        with col:
            st.markdown(f"**{strat}**: {colored_recommendation(rec[strat])}", unsafe_allow_html=True)
    
    st.write(f"Next Day Prediction: ₹{rec['Next Price']:.2f} ({rec['Price Direction']})")
    
    if st.button("Plot Price", key=f"plot_{symbol}"):
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close')
        ax.plot(data['SMA_20'], label='SMA 20')
        ax.legend()
        st.pyplot(fig)

def display_dashboard(NSE_STOCKS):
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))
    send_to_telegram = st.sidebar.checkbox("Send to Telegram", value=True)
    
    if st.button("🚀 Best Picks"):
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel", key="cancel")
        loading_messages = itertools.cycle(["Analyzing...", "Fetching...", "Processing..."])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        results_df = analyze_all_stocks(
            NSE_STOCKS, batch_size=15, price_range=price_range,
            progress_callback=lambda p, s, t, pr: update_progress(progress_bar, loading_text, p, loading_messages, s, t, pr)
        )
        progress_bar.empty()
        loading_text.empty()
        
        if not results_df.empty:
            st.subheader("🏆 Best 3 Stock Picks")
            telegram_message = f"<b>🏆 Best 3 Picks - {datetime.now().strftime('%d %b %Y')}</b>\n\n"
            telegram_message += "📊 <b>Stock | Score | Price | Buy | SL | Target | Next Day | Direction</b>\n"
            telegram_message += "─" * 50 + "\n"
            
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}"):
                    st.markdown(f"""
                    Price: ₹{row['Current Price']:.2f} | Buy: ₹{row['Buy At']:.2f}  
                    SL: ₹{row['Stop Loss']:.2f} | Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Next Day: ₹{row['Next Price']:.2f} ({row['Price Direction']})
                    """, unsafe_allow_html=True)
                
                telegram_message += (
                    f"📈 <b>{row['Symbol']}</b> | {row['Score']} | "
                    f"₹{row['Current Price']:.2f} | ₹{row['Buy At']:.2f} | "
                    f"₹{row['Stop Loss']:.2f} | ₹{row['Target']:.2f} | "
                    f"₹{row['Next Price']:.2f} | {row['Price Direction']}\n"
                )
            
            if send_to_telegram:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(send_telegram_message(telegram_message))
                if success:
                    st.success("✅ Sent to Telegram group!")
        else:
            st.warning("⚠️ No valid stock picks found.")

def main():
    try:
        import telegram
    except ImportError:
        st.error("❌ Install python-telegram-bot: pip install python-telegram-bot --upgrade")
        return
    
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    
    # Initial proxy fetch
    fetch_proxies_from_api()
    
    NSE_STOCKS = fetch_nse_stock_list()
    st.sidebar.title("🔍 Stock Search")
    symbol = st.sidebar.selectbox("Choose stock:", [""] + NSE_STOCKS, key="stock_select")
    
    display_dashboard(NSE_STOCKS)
    
    if symbol:
        data_dict = fetch_stock_data_batch([symbol])
        data = data_dict[symbol]
        if not data.empty:
            data = analyze_stock(data)
            rec = generate_recommendations(data, symbol)
            display_stock_analysis(symbol, data, rec)

if __name__ == "__main__":
    main()