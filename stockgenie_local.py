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
import pickle
import itertools
import telegram
import asyncio
from telegram.error import TelegramError

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

def retry(max_retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        st.warning(f"Max retries reached for {args[0]}: {str(e)}")
                        return pd.DataFrame()
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_retries=3, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return [f"{symbol}.NS" for symbol in pd.read_csv(io.StringIO(response.text))['SYMBOL']]

def load_cache():
    cache_file = "stock_data_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open("stock_data_cache.pkl", 'wb') as f:
        pickle.dump(cache, f, protocol=5)

@retry(max_retries=3, delay=5)
def fetch_stock_data_batch(symbols, period="10y", interval="1d"):
    cache = load_cache()
    data_dict = {}
    symbols_to_fetch = [s for s in symbols if f"{s}_{period}_{interval}" not in cache]
    
    if symbols_to_fetch:
        try:
            data = yf.download(symbols_to_fetch, period=period, interval=interval, group_by='ticker', threads=True)
            for symbol in symbols_to_fetch:
                cache_key = f"{symbol}_{period}_{interval}"
                if symbol in data.columns.levels[0]:
                    symbol_data = data[symbol].dropna()
                    if not symbol_data.empty:
                        cache[cache_key] = symbol_data
                        data_dict[symbol] = symbol_data
                    else:
                        data_dict[symbol] = pd.DataFrame()
                else:
                    data_dict[symbol] = pd.DataFrame()
            save_cache(cache)
        except Exception as e:
            st.warning(f"Batch fetch failed: {e}")
    return {s: cache.get(f"{s}_{period}_{interval}", pd.DataFrame()) for s in symbols}

def analyze_stock(data):
    if data.empty or len(data) < 15:
        return data
    
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    return data

def calculate_buy_at(data):
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_rsi):
        return round(last_close * 0.99 if last_rsi < 30 else last_close, 2)
    return None

def calculate_stop_loss(data):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(last_atr):
        return round(last_close - (2.5 * last_atr), 2)
    return None

def calculate_target(data):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    if pd.notnull(last_close) and pd.notnull(stop_loss):
        risk = last_close - stop_loss
        return round(last_close + (3 * risk), 2)
    return None

def generate_recommendations(data, symbol):
    rec = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
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
    return rec

def analyze_stock_parallel(symbol, data):
    if not data.empty:
        data = analyze_stock(data)
        rec = generate_recommendations(data, symbol)
        return {
            "Symbol": symbol, "Current Price": rec["Current Price"], "Buy At": rec["Buy At"],
            "Stop Loss": rec["Stop Loss"], "Target": rec["Target"], "Intraday": rec["Intraday"],
            "Swing": rec["Swing"], "Short-Term": rec["Short-Term"], "Score": rec["Score"]
        }
    return None

def analyze_batch(stock_batch):
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        data_dict = fetch_stock_data_batch(stock_batch)
        futures = {executor.submit(analyze_stock_parallel, s, data_dict[s]): s 
                  for s in stock_batch if not data_dict[s].empty}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"Error processing {futures[future]}: {e}")
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
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if not results_df.empty:
        for col in ['Current Price', 'Buy At', 'Stop Loss', 'Target']:
            results_df[col] = results_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
        if price_range:
            results_df = results_df[results_df['Current Price'].notnull() & 
                                   (results_df['Current Price'].between(price_range[0], price_range[1]))]
    return results_df.sort_values(by="Score", ascending=False).head(3)

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
    
    if st.button("Plot Price", key=f"plot_{symbol}"):
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close')
        ax.plot(data['SMA_20'], label='SMA 20')
        ax.legend()
        st.pyplot(fig)

def display_dashboard(NSE_STOCKS):
    st.title("📊 StockGenie Lite - NSE Analysis")
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
            
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}"):
                    st.markdown(f"""
                    Price: ₹{row['Current Price']:.2f} | Buy: ₹{row['Buy At']:.2f}  
                    SL: ₹{row['Stop Loss']:.2f} | Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}
                    """, unsafe_allow_html=True)
                
                telegram_message += f"<b>{row['Symbol']}</b> (Score: {row['Score']})\n"
                telegram_message += f"Price: ₹{row['Current Price']:.2f} | Buy: ₹{row['Buy At']:.2f}\n"
                telegram_message += f"SL: ₹{row['Stop Loss']:.2f} | Target: ₹{row['Target']:.2f}\n"
                telegram_message += f"Intraday: {row['Intraday']} | Swing: {row['Swing']}\n\n"
            
            if send_to_telegram:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(send_telegram_message(telegram_message))
                if success:
                    st.success("✅ Sent to Telegram group!")

def main():
    try:
        import telegram
    except ImportError:
        st.error("❌ Install python-telegram-bot: pip install python-telegram-bot --upgrade")
        return
    
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    
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