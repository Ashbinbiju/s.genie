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
import numpy as np
import itertools
import asyncio
import telegram
from telegram.error import TelegramError
import pickle
import os

# API Keys (Consider moving to environment variables)
NEWSAPI_KEY = "ed58659895e84dfb8162a8bb47d8525e"
TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"
TELEGRAM_CHAT_ID = "-1002411670969"

# Tooltip explanations (simplified for core indicators)
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "Bollinger": "Price volatility bands",
    "Stop Loss": "Risk management price level",
}

def tooltip(label, explanation):
    return f"{label} ({explanation})"

def retry(max_retries=3, delay=5, backoff_factor=2, jitter=0.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if "Too Many Requests" in str(e) or "rate limited" in str(e).lower():
                        retries += 1
                        if retries == max_retries:
                            st.warning(f"Max retries for {args[0]}: {str(e)}")
                            return pd.DataFrame() if "fetch" in func.__name__ else None
                        sleep_time = delay * (backoff_factor ** retries) + random.uniform(0, jitter)
                        st.warning(f"Rate limited for {args[0]}. Waiting {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        st.warning(f"Error for {args[0]}: {str(e)}")
                        return pd.DataFrame() if "fetch" in func.__name__ else None
        return wrapper
    return decorator

@retry(max_retries=3, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception:
        st.warning("⚠️ Failed to fetch NSE stock list; using fallback.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

def load_cache():
    cache_file = "stock_data_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open("stock_data_cache.pkl", 'wb') as f:
        pickle.dump(cache, f)

@retry(max_retries=3, delay=5)
def fetch_stock_data_bulk(symbols, period="5y", interval="1d"):
    cache = load_cache()
    missing_symbols = [s for s in symbols if f"{s}_{period}_{interval}" not in cache]
    
    if missing_symbols:
        try:
            data = yf.download(missing_symbols, period=period, interval=interval, group_by='ticker', threads=True)
            for symbol in missing_symbols:
                if symbol in data.columns.levels[0]:
                    ticker_data = data[symbol].dropna()
                    if not ticker_data.empty:
                        cache[f"{symbol}_{period}_{interval}"] = ticker_data
            save_cache(cache)
        except Exception as e:
            st.warning(f"⚠️ Bulk fetch failed: {str(e)}")
    
    result = {symbol: cache.get(f"{symbol}_{period}_{interval}", pd.DataFrame()) for symbol in symbols}
    return result

def analyze_stock(data):
    if data.empty or len(data) < 15:
        return data
    
    days = len(data)
    windows = {
        'rsi': min(7, days - 1),  # Simplified to fixed window
        'macd_slow': min(26, days - 1),
        'macd_fast': min(12, days - 1),
        'macd_sign': min(9, days - 1),
        'bollinger': min(20, days - 1),
        'atr': min(14, days - 1),
    }
    
    if days >= windows['rsi'] + 1:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=windows['rsi']).rsi()
    
    if days >= max(windows['macd_slow'], windows['macd_fast']) + 1:
        macd = ta.trend.MACD(data['Close'], window_slow=windows['macd_slow'], 
                            window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
    
    if days >= windows['bollinger'] + 1:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=windows['bollinger'])
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Lower_Band'] = bollinger.bollinger_lband()
    
    if days >= windows['atr'] + 1:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], 
                                                    window=windows['atr']).average_true_range()
    
    return data

def calculate_buy_at(data):
    if 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    if last_rsi < 30:
        return round(last_close * 0.99, 2)
    return round(last_close, 2)

def calculate_stop_loss(data, atr_multiplier=2):
    if 'ATR' not in data.columns or pd.isna(data['ATR'].iloc[-1]):
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    return round(last_close - (atr_multiplier * last_atr), 2)

def calculate_target(data, risk_reward_ratio=2):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    return round(last_close + (risk * risk_reward_ratio), 2)

def generate_recommendations(data, symbol=None):
    rec = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or pd.isna(data['Close'].iloc[-1]):
        return rec
    
    rec["Current Price"] = round(data['Close'].iloc[-1], 2)
    buy_score = sell_score = 0
    last_close = data['Close'].iloc[-1]
    
    if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
    
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        if pd.notna(data['MACD'].iloc[-1]) and pd.notna(data['MACD_signal'].iloc[-1]):
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                buy_score += 1
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                sell_score += 1
    
    if 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
        if pd.notna(data['Lower_Band'].iloc[-1]) and pd.notna(data['Upper_Band'].iloc[-1]):
            if last_close < data['Lower_Band'].iloc[-1]:
                buy_score += 1
            elif last_close > data['Upper_Band'].iloc[-1]:
                sell_score += 1
    
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
        rec["Swing"] = "Buy"
    elif sell_score > buy_score:
        rec["Intraday"] = "Sell"
        rec["Swing"] = "Sell"
    
    rec["Buy At"] = calculate_buy_at(data)
    rec["Stop Loss"] = calculate_stop_loss(data)
    rec["Target"] = calculate_target(data)
    rec["Score"] = min(buy_score - sell_score, 5)
    return rec

def analyze_batch(stock_batch):
    data_dict = fetch_stock_data_bulk(stock_batch)
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:  # Increased workers
        futures = {executor.submit(analyze_stock_parallel, symbol, data_dict[symbol]): symbol 
                  for symbol in stock_batch if not data_dict[symbol].empty and len(data_dict[symbol]) >= 15}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Error processing {symbol}: {str(e)}")
    return results

def analyze_stock_parallel(symbol, data):
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
            "Score": recommendations.get("Score", 0),
        }
    return None

def update_progress(progress_bar, loading_text, progress_value, loading_messages, start_time, total_items, processed_items):
    if int(progress_value * 100) % 5 == 0:  # Update every 5%
        progress_bar.progress(progress_value)
        loading_message = next(loading_messages)
        dots = "." * int((progress_value * 10) % 4)
        elapsed_time = time.time() - start_time
        eta = timedelta(seconds=int((elapsed_time / processed_items) * (total_items - processed_items))) if processed_items > 0 else "N/A"
        loading_text.text(f"{loading_message}{dots} | Processed {processed_items}/{total_items} (ETA: {eta})")

def analyze_all_stocks(stock_list, batch_size=100, price_range=None, progress_callback=None):
    if st.session_state.cancel_operation:
        st.warning("⚠️ Analysis canceled.")
        return pd.DataFrame()
    
    total_items = len(stock_list)
    st.info(f"Analyzing {total_items} NSE stocks (~{total_items // 120} min).")
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
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        for col in ['Current Price', 'Buy At', 'Stop Loss', 'Target']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
        if price_range:
            results_df = results_df[results_df['Current Price'].notnull() & 
                                   results_df['Current Price'].between(price_range[0], price_range[1])]
        return results_df.sort_values(by="Score", ascending=False).head(10)
    return pd.DataFrame()

def analyze_strong_buy_stocks(stock_list, batch_size=100, price_range=None, progress_callback=None):
    if st.session_state.cancel_operation:
        return pd.DataFrame()
    
    total_items = len(stock_list)
    st.info(f"Analyzing {total_items} NSE stocks for buy picks (~{total_items // 120} min).")
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
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        for col in ['Current Price', 'Buy At', 'Stop Loss', 'Target']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
        if price_range:
            results_df = results_df[results_df['Current Price'].notnull() & 
                                   results_df['Current Price'].between(price_range[0], price_range[1])]
        strong_buy_df = results_df[results_df["Intraday"].isin(["Strong Buy", "Buy"])]
        return strong_buy_df.sort_values(by="Score", ascending=False).head(5)
    return pd.DataFrame()

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    return f"🟡 {recommendation}"

async def send_telegram_message(message):
    try:
        bot = telegram.Bot(token="7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps")
        await bot.send_message(chat_id="-1002411670969", text=message, parse_mode='HTML')
        return True
    except TelegramError as e:
        st.error(f"❌ Telegram error: {str(e)}")
        return False

def display_stock_analysis(symbol, data, recommendations):
    with st.container():
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), 
                     f"₹{recommendations['Current Price']:.2f}" if recommendations['Current Price'] else "N/A")
        with col2:
            st.metric("Buy At", f"₹{recommendations['Buy At']:.2f}" if recommendations['Buy At'] else "N/A")
        with col3:
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), 
                     f"₹{stop_loss:.2f}" if (stop_loss := recommendations['Stop Loss']) else "N/A")
        with col4:
            st.metric("Target", f"₹{target:.2f}" if (target := recommendations['Target']) else "N/A")
        
        st.subheader("📈 Recommendations")
        cols = st.columns(3)
        for col, strategy in zip(cols, ["Intraday", "Swing", "Short-Term"]):
            with col:
                st.markdown(f"**{strategy}**: {colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        
        if 'Close' in data.columns:
            fig = px.line(data, y='Close', title="Price Action")
            st.plotly_chart(fig, key=f"price_{symbol}")

def display_dashboard(NSE_STOCKS):
    st.title("📊 StockGenie Lite - NSE Analysis")
    st.subheader(f"📅 {datetime.now().strftime('%d %b %Y')}")
    
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))
    send_to_telegram = st.sidebar.checkbox("Send to Telegram", value=True)
    
    if st.button("🚀 Top Picks"):
        st.session_state.current_view = "daily_top_picks"
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel", key="cancel_daily")
        loading_messages = itertools.cycle(["Analyzing...", "Fetching...", "Processing..."])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        results_df = analyze_all_stocks(
            NSE_STOCKS, batch_size=100, price_range=price_range,
            progress_callback=lambda p, s, t, i: update_progress(progress_bar, loading_text, p, loading_messages, s, t, i)
        )
        progress_bar.empty()
        loading_text.empty()
        
        if not results_df.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Top 10 Stocks")
            telegram_message = f"<b>Top 10 Stocks - {datetime.now().strftime('%d %b %Y')}</b>\n\n"
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    Current Price: ₹{row['Current Price']:.2f}  
                    Buy At: ₹{row['Buy At']:.2f} | Stop Loss: ₹{row['Stop Loss']:.2f}  
                    Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    """, unsafe_allow_html=True)
                telegram_message += f"<b>{row['Symbol']}</b> (Score: {row['Score']})\nPrice: ₹{row['Current Price']:.2f} | Buy: ₹{row['Buy At']:.2f}\nSL: ₹{row['Stop Loss']:.2f} | Target: ₹{row['Target']:.2f}\n\n"
            
            if send_to_telegram:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(send_telegram_message(telegram_message))
                if success:
                    st.success("✅ Sent to Telegram!")
    
    if st.button("🟢 Top Buy Picks"):
        st.session_state.current_view = "strong_buy_picks"
        st.session_state.cancel_operation = False
        progress_bar = st.progress(0)
        loading_text = st.empty()
        cancel_button = st.button("❌ Cancel", key="cancel_buy")
        loading_messages = itertools.cycle(["Scanning...", "Analyzing...", "Filtering..."])
        
        if cancel_button:
            st.session_state.cancel_operation = True
        
        strong_buy_df = analyze_strong_buy_stocks(
            NSE_STOCKS, batch_size=100, price_range=price_range,
            progress_callback=lambda p, s, t, i: update_progress(progress_bar, loading_text, p, loading_messages, s, t, i)
        )
        progress_bar.empty()
        loading_text.empty()
        
        if not strong_buy_df.empty and not st.session_state.cancel_operation:
            st.subheader("🏆 Top 5 Buy Picks")
            telegram_message = f"<b>Top 5 Buy Picks - {datetime.now().strftime('%d %b %Y')}</b>\n\n"
            for _, row in strong_buy_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    Current Price: ₹{row['Current Price']:.2f}  
                    Buy At: ₹{row['Buy At']:.2f} | Stop Loss: ₹{row['Stop Loss']:.2f}  
                    Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    """, unsafe_allow_html=True)
                telegram_message += f"<b>{row['Symbol']}</b> (Score: {row['Score']})\nPrice: ₹{row['Current Price']:.2f} | Buy: ₹{row['Buy At']:.2f}\nSL: ₹{row['Stop Loss']:.2f} | Target: ₹{row['Target']:.2f}\n\n"
            
            if send_to_telegram:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(send_telegram_message(telegram_message))
                if success:
                    st.success("✅ Sent to Telegram!")

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = None

    st.sidebar.title("🔍 Stock Search")
    NSE_STOCKS = fetch_nse_stock_list()
    
    selected_option = st.sidebar.selectbox(
        "Choose stock:", [""] + NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x not in ["", "Custom"] else x
    )
    
    symbol = None
    if selected_option == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g., RELIANCE):")
        if custom_symbol:
            symbol = f"{custom_symbol}.NS"
    elif selected_option:
        symbol = selected_option
    
    display_dashboard(NSE_STOCKS)
    
    if symbol:
        data_dict = fetch_stock_data_bulk([symbol])
        data = data_dict[symbol]
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            st.session_state.current_view = "individual_stock"
            display_stock_analysis(symbol, data, recommendations)
        else:
            st.error(f"❌ Failed to load data for {symbol}")

if __name__ == "__main__":
    main()