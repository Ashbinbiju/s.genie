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

# Define sectors and their stocks (expanded for testing)
SECTORS = {
    "Banking and Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS", 
        "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "BANDHANBNK.NS", "INDIANB.NS", 
        "BANKINDIA.NS", "KARURVYSYA.NS", "CUB.NS", "J&KBANK.NS", "LAKSHVILAS.NS", 
        "DCBBANK.NS", "SYNDIBANK.NS", "ALBK.NS", "ANDHRABANK.NS", "CORPBANK.NS", 
        "ORIENTBANK.NS", "UNITEDBNK.NS", "AUBANK.NS"
    ],
    "Software & IT Services": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Finance and Investment Companies": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS"],
    "Automobile & Ancillaries": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"],
    "Healthcare and Pharmaceuticals": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "Fast Moving Consumer Goods (FMCG)": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Power Generation and Distribution": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"],
    "Capital Goods and Engineering": ["LT.NS", "BHEL.NS", "SIEMENS.NS"],
    "Chemicals and Fertilizers": ["UPL.NS", "PIDILITIND.NS", "SRF.NS"],
    "Telecommunication Services": ["BHARTIARTL.NS", "IDEA.NS", "RELIANCE.NS"],
    "Infrastructure Development": ["ADANIPORTS.NS", "GMRINFRA.NS"],
    "Insurance Companies": ["ICICIPRULI.NS", "HDFCAMC.NS"],
    "Diversified Conglomerates": ["RELIANCE.NS", "ADANIENT.NS"],
    "Construction Materials and Cement": ["ULTRACEMCO.NS", "ACC.NS", "AMBUJACEM.NS"],
    "Real Estate and Housing": ["DLF.NS", "GODREJPROP.NS"],
    "Aviation and Airlines": ["INDIGO.NS", "SPICEJET.NS"],
    "Retailing and Consumer Services": ["DMART.NS", "TITAN.NS"],
    "Miscellaneous Industries": ["APOLLOTYRE.NS", "MRF.NS"],
    "Diamond & Jewellery": ["TITAN.NS", "PCJEWELLER.NS"],
    "Consumer Durables and Electronics": ["HAVELLS.NS", "CROMPTON.NS"],
    "Trading and Distribution": ["ADANIENT.NS"],
    "Hospitality and Tourism": ["EIHOTEL.NS", "INDHOTEL.NS"],
    "Agriculture and Agrochemicals": ["UPL.NS", "PIIND.NS"],
    "Textiles and Apparel": ["PAGEIND.NS", "RAYMOND.NS"],
    "Industrial Gases & Fuels": ["LINDEINDIA.NS"],
    "Electricals and Equipment": ["HAVELLS.NS", "POLYCAB.NS"],
    "Alcohol and Beverages": ["UBL.NS", "RADICO.NS"],
    "Logistics and Transportation": ["CONCOR.NS", "BLUEDART.NS"],
    "Plastic Products": ["SUPREMEIND.NS"],
    "Ship Building and Repair": ["COCHINSHIP.NS"],
    "Media & Entertainment": ["ZEEL.NS", "SUNTV.NS"],
    "Exchange Traded Funds (ETF)": ["NIFTYBEES.NS"],
    "Footwear Manufacturing": ["BATAINDIA.NS", "RELAXO.NS"],
    "Manufacturing and Industrials": ["BEL.NS", "HAL.NS"],
    "Containers & Packaging": ["UFLEX.NS"],
    "Paper and Paper Products": ["JKPAPER.NS"],
    "Photographic Products": []  # Empty for testing
}

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

def analyze_stock(data):
    if data.empty or len(data) < 27:
        return data
    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Lower_Band'] = bollinger.bollinger_lband()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        if len(data) >= 27:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
    except Exception:
        pass
    return data

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

def calculate_buy_at(data):
    if 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    return round(last_close * 0.99 if last_rsi < 30 else last_close, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    return round(last_close - (atr_multiplier * last_atr), 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    risk_reward_ratio = 3 if 'ADX' in data.columns and data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    return round(last_close + (risk * risk_reward_ratio), 2)

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        return recommendations
    
    recommendations["Current Price"] = float(data['Close'].iloc[-1])
    buy_score = sell_score = 0
    
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
        if data['RSI'].iloc[-1] < 30: buy_score += 2
        elif data['RSI'].iloc[-1] > 70: sell_score += 2
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None:
        if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]: buy_score += 1
        elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]: sell_score += 1
    if 'Close' in data.columns and 'Lower_Band' in data.columns and data['Upper_Band'] in data.columns:
        if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]: buy_score += 1
        elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]: sell_score += 1
    if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
        if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
            buy_score += 1; recommendations["Ichimoku_Trend"] = "Buy"
        elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
            sell_score += 1; recommendations["Ichimoku_Trend"] = "Sell"
    if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
        if data['CMF'].iloc[-1] > 0: buy_score += 1
        elif data['CMF'].iloc[-1] < 0: sell_score += 1
    if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
        if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
            buy_score += 1; recommendations["Breakout"] = "Buy"
        elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
            sell_score += 1; recommendations["Breakout"] = "Sell"
    
    if buy_score >= 4: recommendations["Intraday"] = "Strong Buy"
    elif sell_score >= 4: recommendations["Intraday"] = "Strong Sell"
    
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
        recommendations = generate_recommendations(data, symbol)
        return {
            "Symbol": symbol, "Current Price": recommendations["Current Price"], "Buy At": recommendations["Buy At"],
            "Stop Loss": recommendations["Stop Loss"], "Target": recommendations["Target"], "Intraday": recommendations["Intraday"],
            "Swing": recommendations["Swing"], "Short-Term": recommendations["Short-Term"], "Long-Term": recommendations["Long-Term"],
            "Mean_Reversion": recommendations["Mean_Reversion"], "Breakout": recommendations["Breakout"],
            "Ichimoku_Trend": recommendations["Ichimoku_Trend"], "Score": recommendations.get("Score", 0),
        }
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
    return results_df.sort_values(by="Score", ascending=False).head(10) if not results_df.empty else pd.DataFrame()

def analyze_intraday_stocks(stock_list, batch_size=50, progress_callback=None):
    results = analyze_all_stocks(stock_list, batch_size, progress_callback)
    return results[results["Intraday"].str.contains("Buy", na=False)].head(5) if not results.empty else pd.DataFrame()

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"<span style='color: green;'>🟢 {recommendation}</span>"
    elif "Sell" in recommendation:
        return f"<span style='color: red;'>🔴 {recommendation}</span>"
    elif "Hold" in recommendation:
        return f"<span style='color: orange;'>🟡 {recommendation}</span>"
    return recommendation

def display_dashboard(symbol=None, data=None, recommendations=None, selected_stocks=None):
    st.title("📈 StockGenie Pro")
    st.markdown(f"**Date:** {datetime.now().strftime('%d %b %Y')}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Market Overview")
        st.info(f"Analyzing {len(selected_stocks)} stocks" if selected_stocks else "No sectors selected.")
    with col2:
        if st.button("🚀 Top Daily Picks", key="daily", disabled=not selected_stocks):
            with st.spinner("Analyzing stocks..."):
                progress_bar = st.progress(0)
                loading_messages = itertools.cycle(["Fetching data", "Processing indicators", "Ranking stocks"])
                results_df = analyze_all_stocks(
                    selected_stocks,
                    progress_callback=lambda x: (progress_bar.progress(x), st.text(f"{next(loading_messages)}..."))
                )
                progress_bar.empty(); st.text("")
            if not results_df.empty:
                st.subheader("🏆 Top 10 Daily Picks")
                for _, row in results_df.iterrows():
                    with st.expander(f"{row['Symbol']} (Score: {row['Score']}/7)"):
                        st.markdown(f"""
                        Current Price: ₹{row['Current Price'] or 'N/A'}  
                        Buy At: ₹{row['Buy At'] or 'N/A'} | Stop Loss: ₹{row['Stop Loss'] or 'N/A'} | Target: ₹{row['Target'] or 'N/A'}  
                        Intraday: {colored_recommendation(row['Intraday'])}  
                        Swing: {colored_recommendation(row['Swing'])}  
                        Short-Term: {colored_recommendation(row['Short-Term'])}  
                        Long-Term: {colored_recommendation(row['Long-Term'])}  
                        Mean Reversion: {colored_recommendation(row['Mean_Reversion'])}  
                        Breakout: {colored_recommendation(row['Breakout'])}  
                        Ichimoku Trend: {colored_recommendation(row['Ichimoku_Trend'])}
                        """, unsafe_allow_html=True)
            else:
                st.warning("No top picks available.")
        if st.button("⚡ Top Intraday Picks", key="intraday", disabled=not selected_stocks):
            with st.spinner("Scanning intraday opportunities..."):
                progress_bar = st.progress(0)
                loading_messages = itertools.cycle(["Detecting signals", "Calculating levels", "Optimizing picks"])
                intraday_results = analyze_intraday_stocks(
                    selected_stocks,
                    progress_callback=lambda x: (progress_bar.progress(x), st.text(f"{next(loading_messages)}..."))
                )
                progress_bar.empty(); st.text("")
            if not intraday_results.empty:
                st.subheader("🏆 Top 5 Intraday Picks")
                for _, row in intraday_results.iterrows():
                    with st.expander(f"{row['Symbol']} (Score: {row['Score']}/7)"):
                        st.markdown(f"""
                        Current Price: ₹{row['Current Price'] or 'N/A'}  
                        Buy At: ₹{row['Buy At'] or 'N/A'} | Stop Loss: ₹{row['Stop Loss'] or 'N/A'} | Target: ₹{row['Target'] or 'N/A'}  
                        Intraday: {colored_recommendation(row['Intraday'])}
                        """, unsafe_allow_html=True)
            else:
                st.warning("No intraday picks available.")
    
    if symbol and data is not None and recommendations is not None:
        st.markdown("---")
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Current Price", f"₹{recommendations['Current Price'] or 'N/A'}")
        with col2: st.metric("Buy At", f"₹{recommendations['Buy At'] or 'N/A'}")
        with col3: st.metric("Stop Loss", f"₹{recommendations['Stop Loss'] or 'N/A'}")
        with col4: st.metric("Target", f"₹{recommendations['Target'] or 'N/A'}")
        
        st.subheader("📈 Recommendations")
        cols = st.columns(4)
        for col, strategy in zip(cols, ["Intraday", "Swing", "Short-Term", "Long-Term"]):
            with col:
                st.markdown(f"**{strategy}**: {colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        cols = st.columns(3)
        for col, strategy in zip(cols, ["Mean_Reversion", "Breakout", "Ichimoku_Trend"]):
            with col:
                st.markdown(f"**{strategy.replace('_', ' ')}**: {colored_recommendation(recommendations[strategy])}", unsafe_allow_html=True)
        
        tabs = st.tabs(["Price Action", "Momentum", "Volatility", "Monte Carlo", "Advanced"])
        with tabs[0]:
            fig = px.line(data, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50'], title="Price Trends")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], title="Momentum")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            fig = px.line(data, y=['ATR', 'Upper_Band', 'Lower_Band', 'Donchian_Upper', 'Donchian_Lower'], title="Volatility")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            mc_results = monte_carlo_simulation(data)
            mc_df = pd.DataFrame(mc_results).T
            mc_df.columns = [f"Sim {i+1}" for i in range(len(mc_results))]
            fig = px.line(mc_df, title="Monte Carlo (30 Days)")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[4]:
            fig = px.line(data, y=['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'CMF'], title="Advanced Indicators")
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.sidebar.title("🔍 Control Panel")
    with st.sidebar.expander("Sector Selection", expanded=False):
        all_sectors = list(SECTORS.keys())
        selected_sectors = []
        st.markdown('<div style="max-height: 200px; overflow-y: auto;">', unsafe_allow_html=True)  # Scrollable container
        for sector in all_sectors:
            if st.checkbox(sector, key=sector):
                selected_sectors.append(sector)
        st.markdown('</div>', unsafe_allow_html=True)
    
    selected_stocks = list(set([stock for sector in selected_sectors for stock in SECTORS[sector] if stock in fetch_nse_stock_list()]))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Stock Picker")
    symbol = st.sidebar.selectbox(
        "Select Stock",
        options=[""] + selected_stocks,
        format_func=lambda x: x.split('.')[0] if x else "Choose a stock",
        disabled=not selected_stocks
    )
    
    if symbol:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, selected_stocks)
        else:
            st.error(f"❌ Could not load data for {symbol}")
    else:
        display_dashboard(None, None, None, selected_stocks)

if __name__ == "__main__":
    st.set_page_config(page_title="StockGenie Pro", layout="wide")
    main()