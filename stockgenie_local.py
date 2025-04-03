import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
import numpy as np
import itertools
from arch import arch_model

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
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
}

# Define sectors and their stocks (expanded for demonstration)
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
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"],
    "Capital Goods": ["LT.NS", "BHEL.NS", "SIEMENS.NS"],
    "Chemicals": ["PIDILITIND.NS", "SRF.NS", "UPL.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"],
    "Infrastructure": ["ADANIPORTS.NS", "GMRINFRA.NS"],
    "Insurance": ["ICICIGI.NS", "NIACL.NS"],
    "Diversified": ["RELIANCE.NS", "ITC.NS"],
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
    "Alcohol": ["UNITDSPR.NS", "RADICO.NS"],
    "Logistics": ["CONCOR.NS", "BLUEDART.NS"],
    "Plastic Products": ["SUPREMEIND.NS"],
    "Ship Building": ["COCHINSHIP.NS"],
    "Media & Entertainment": ["ZEEL.NS", "SUNTV.NS"],
    "ETF": ["NIFTYBEES.NS"],
    "Footwear": ["BATAINDIA.NS", "RELAXO.NS"],
    "Manufacturing": ["ASIANPAINT.NS", "BERGEPAINT.NS"],
    "Containers & Packaging": ["UFLEX.NS"],
    "Paper": ["JKPAPER.NS"],
    "Photographic Products": []
}

def retry(max_retries=3, delay=1):
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
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
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

def analyze_stock(data):
    if data.empty or len(data) < 27:
        return data
    try:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Lower_Band'] = bollinger.bollinger_lband()
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
    except Exception:
        pass
    return data

def calculate_buy_at(data):
    if 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    return round(last_close * 0.99 if last_rsi < 30 else last_close, 2)

def calculate_stop_loss(data):
    if 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25 else 1.5
    return round(last_close - (atr_multiplier * last_atr), 2)

def calculate_target(data):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    risk_reward_ratio = 3 if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25 else 1.5
    return round(last_close + (risk * risk_reward_ratio), 2)

def generate_recommendations(data):
    recommendations = {
        "Intraday": "Hold", "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        return recommendations
    
    recommendations["Current Price"] = float(data['Close'].iloc[-1])
    buy_score = sell_score = 0
    
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
        if data['RSI'].iloc[-1] < 30: buy_score += 2
        elif data['RSI'].iloc[-1] > 70: sell_score += 2
    if 'MACD' in data.columns and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]: buy_score += 1
    elif 'MACD' in data.columns and data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]: sell_score += 1
    if 'Close' in data.columns and 'Lower_Band' in data.columns and data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]: buy_score += 1
    elif 'Close' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]: sell_score += 1
    
    if buy_score >= 3: recommendations["Intraday"] = "Strong Buy"
    elif sell_score >= 3: recommendations["Intraday"] = "Strong Sell"
    
    recommendations["Buy At"] = calculate_buy_at(data)
    recommendations["Stop Loss"] = calculate_stop_loss(data)
    recommendations["Target"] = calculate_target(data)
    recommendations["Score"] = max(0, min(buy_score - sell_score, 7))
    return recommendations

def analyze_batch(stock_batch):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result: results.append(result)
            except Exception:
                pass
    return results

def analyze_stock_parallel(symbol):
    data = fetch_stock_data_cached(symbol)
    if not data.empty:
        data = analyze_stock(data)
        recommendations = generate_recommendations(data)
        return {
            "Symbol": symbol, "Current Price": recommendations["Current Price"],
            "Buy At": recommendations["Buy At"], "Stop Loss": recommendations["Stop Loss"],
            "Target": recommendations["Target"], "Intraday": recommendations["Intraday"],
            "Score": recommendations["Score"]
        }
    return None

def analyze_all_stocks(stock_list, batch_size=50, progress_callback=None):
    results = []
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
    results_df = pd.DataFrame([r for r in results if r is not None])
    return results_df.sort_values(by="Score", ascending=False).head(10) if not results_df.empty else pd.DataFrame()

def colored_recommendation(recommendation):
    if "Buy" in recommendation: return f"<span style='color:green'>🟢 {recommendation}</span>"
    elif "Sell" in recommendation: return f"<span style='color:red'>🔴 {recommendation}</span>"
    else: return f"<span style='color:orange'>🟡 {recommendation}</span>"

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📈 StockGenie Pro")
    st.markdown(f"**Date:** {datetime.now().strftime('%d %b %Y')}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Market Overview")
        st.info(f"Selected stocks: {len(selected_stocks)}" if selected_stocks else "No sectors selected yet.")
    with col2:
        if st.button("🚀 Top Daily Picks"):  # Removed disabled condition
            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                loading_messages = itertools.cycle(["Fetching", "Processing", "Ranking"])
                results_df = analyze_all_stocks(
                    selected_stocks,
                    progress_callback=lambda x: (progress_bar.progress(x), st.text(f"{next(loading_messages)}..."))
                )
                progress_bar.empty()
                st.text("")
            if not results_df.empty:
                st.subheader("🏆 Top 10 Picks")
                st.dataframe(results_df[["Symbol", "Current Price", "Score"]], use_container_width=True)
                for _, row in results_df.iterrows():
                    with st.expander(f"{row['Symbol']} (Score: {row['Score']})"):
                        st.markdown(f"""
                            Current Price: ₹{row['Current Price'] or 'N/A'}  
                            Buy At: ₹{row['Buy At'] or 'N/A'}  
                            Stop Loss: ₹{row['Stop Loss'] or 'N/A'}  
                            Target: ₹{row['Target'] or 'N/A'}  
                            Intraday: {colored_recommendation(row['Intraday'])}
                        """, unsafe_allow_html=True)
            else:
                st.warning("No picks available. Select sectors to analyze stocks.")

    if symbol and data is not None and recommendations is not None:
        st.markdown("---")
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Current Price", f"₹{recommendations['Current Price'] or 'N/A'}")
        with col2: st.metric("Buy At", f"₹{recommendations['Buy At'] or 'N/A'}")
        with col3: st.metric("Stop Loss", f"₹{recommendations['Stop Loss'] or 'N/A'}")
        with col4: st.metric("Target", f"₹{recommendations['Target'] or 'N/A'}")
        
        st.subheader("📈 Recommendation")
        st.markdown(f"**Intraday**: {colored_recommendation(recommendations['Intraday'])}", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Price & Volatility", "Indicators"])
        with tab1:
            fig = px.line(data, y=['Close', 'Upper_Band', 'Lower_Band'], title="Price & Bollinger Bands")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], title="Momentum Indicators")
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.sidebar.title("🔍 Control Panel")
    with st.sidebar.expander("Sector Selection", expanded=True):
        st.markdown("**Choose Sectors**")
        selected_sectors = []
        col1, col2 = st.columns(2)
        all_sectors = sorted(SECTORS.keys())
        half = len(all_sectors) // 2
        
        # Column 1: First half of sectors
        with col1:
            for sector in all_sectors[:half]:
                if st.checkbox(sector, value=False, key=f"chk_{sector}"):  # No default selection
                    selected_sectors.append(sector)
        
        # Column 2: Second half of sectors
        with col2:
            for sector in all_sectors[half:]:
                if st.checkbox(sector, value=False, key=f"chk_{sector}"):  # No default selection
                    selected_sectors.append(sector)
    
    selected_stocks = list(set([stock for sector in selected_sectors for stock in SECTORS[sector] if stock in fetch_nse_stock_list()]))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Stock Picker")
    symbol = st.sidebar.selectbox(
        "Select Stock",
        options=[""] + selected_stocks,
        format_func=lambda x: x.split('.')[0] if x else "Choose a stock",
        help="Select sectors first to populate this list."
    )
    
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data)
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error(f"❌ Could not load data for {symbol}")
    else:
        display_dashboard(None, None, None, selected_stocks)

if __name__ == "__main__":
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    main()