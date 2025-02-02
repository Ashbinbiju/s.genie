import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
import plotly.express as px
import time
import requests
import io
import random

# Tooltip explanations
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
}


# Tooltip function
def tooltip(label, explanation):
    """Returns a formatted tooltip string"""
    return f"{label} 🛈 ({explanation})"


# Retry decorator for Yahoo Finance requests with jitter
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
                        st.error(f"❌ Max retries reached for function {func.__name__}")
                        raise e
                    # Exponential backoff with jitter
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
            return None

        return wrapper

    return decorator


@retry(max_retries=3, delay=2)
def fetch_nse_stock_list():
    """
    Fetch live NSE stock list from the official NSE website.
    Falls back to predefined list if download fails.
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Use io.StringIO instead of pd.compat.StringIO
        nse_data = pd.read_csv(io.StringIO(response.text))
        # Extract symbol column and append ".NS" for Yahoo Finance compatibility
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        st.success("✅ Fetched live NSE stock list successfully!")
        return stock_list
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch live NSE stock list. Falling back to predefined list. Error: {str(e)}")
        return [
            "20MICRONS.NS", "21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "5PAISA.NS", "63MOONS.NS",
            "A2ZINFRA.NS", "AAATECH.NS", "AADHARHFC.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS",
            "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS",
            "AATMAJ.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABCOTS.NS", "ABDL.NS",
            "ABFRL.NS",
        ]


@lru_cache(maxsize=100)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    """Fetch data with retries and caching"""
    try:
        if ".NS" not in symbol:
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch data for {symbol} after 3 attempts")
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()


def analyze_stock(data):
    """Perform technical analysis on stock data"""
    if data.empty or len(data) < 27:  # Ensure enough data points for ADX calculation
        return data
    try:
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    except Exception as e:
        st.warning(f"⚠️ Error calculating RSI: {e}")
        data['RSI'] = None
    try:
        # MACD
        macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Error calculating MACD: {e}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None
    try:
        # Moving Averages
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Error calculating Moving Averages: {e}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
    try:
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Error calculating Bollinger Bands: {e}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None
    try:
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Error calculating Stochastic Oscillator: {e}")
        data['SlowK'] = None
        data['SlowD'] = None
    try:
        # ATR (Average True Range)
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'],
                                                     window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Error calculating ATR: {e}")
        data['ATR'] = None
    try:
        # ADX (Average Directional Index)
        if len(data) >= 27:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        st.warning(f"⚠️ Error calculating ADX: {e}")
        data['ADX'] = None
    return data


def calculate_stop_loss(data, atr_multiplier=2):
    """Calculate stop-loss level based on ATR"""
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    stop_loss = last_close - (atr_multiplier * last_atr)
    return round(stop_loss, 2)


def calculate_buy_at(data):
    """Calculate optimal buy price based on RSI and current price"""
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    if last_rsi < 30:  # Oversold condition
        buy_at = last_close * 0.99  # Slightly below current price
    else:
        buy_at = last_close  # Buy at current price
    return round(buy_at, 2)


def calculate_target(data, risk_reward_ratio=2):
    """Calculate target price based on risk-reward ratio"""
    if data.empty or 'Close' not in data.columns:
        return None
    last_close = data['Close'].iloc[-1]
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    risk = last_close - stop_loss
    target = last_close + (risk * risk_reward_ratio)
    return round(target, 2)


def generate_recommendations(data):
    """Generate comprehensive trade recommendations"""
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None
    }
    if data.empty:
        return recommendations
    try:
        # Current Price
        recommendations["Current Price"] = data['Close'].iloc[-1]
        # Enhanced scoring system
        score = 0
        # Intraday (RSI + ATR)
        if 'RSI' in data and 'ATR' in data:
            if data['RSI'].iloc[-1] < 30 and data['ATR'].iloc[-1] > data['ATR'].mean():
                recommendations["Intraday"] = "Buy (Oversold with High Volatility)"
                score += 2
            elif data['RSI'].iloc[-1] > 70:
                recommendations["Intraday"] = "Sell (Overbought)"
                score -= 1
        # Swing (MACD + ADX)
        if 'MACD' in data and 'ADX' in data:
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] and data['ADX'].iloc[-1] > 25:
                recommendations["Swing"] = "Buy (Bullish Crossover with Strong Trend)"
                score += 2
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                recommendations["Swing"] = "Sell (Bearish Crossover)"
                score -= 1
        # Short-Term (SMA crossover)
        if 'SMA_50' in data and 'SMA_200' in data:
            if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
                recommendations["Short-Term"] = "Buy (Golden Cross)"
                score += 1
            else:
                recommendations["Short-Term"] = "Sell (Death Cross)"
                score -= 1
        # Long-Term (Bollinger Bands)
        if 'Close' in data and 'Lower_Band' in data:
            if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                recommendations["Long-Term"] = "Buy (Oversold)"
                score += 1
            elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                recommendations["Long-Term"] = "Sell (Overbought)"
                score -= 1
        # Calculate trade levels
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = max(0, min(score, 5))  # Keep score between 0-5
    except Exception as e:
        st.warning(f"⚠️ Recommendation error: {str(e)}")
    return recommendations


def analyze_batch(stock_batch):
    """Analyze a batch of stocks in parallel"""
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Error processing stock: {e}")
    return results


def analyze_stock_parallel(symbol):
    """Analyze a single stock (used in parallel processing)"""
    data = fetch_stock_data_cached(symbol)
    if not data.empty:
        data = analyze_stock(data)
        recommendations = generate_recommendations(data)
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
            "Score": recommendations.get("Score", 0)
        }
    return None


def analyze_all_stocks(stock_list, batch_size=50):
    """Analyze all stocks in the list using batch processing"""
    results = []
    for i in tqdm(range(0, len(stock_list), batch_size), desc="Processing Batches"):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
    results_df = pd.DataFrame(results)
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    results_df = results_df.sort_values(by="Score", ascending=False).head(10)
    return results_df


def colored_recommendation(recommendation):
    """Returns a color-coded recommendation for Streamlit"""
    if "Buy" in recommendation:
        return f"<span style='color:green;'>{recommendation}</span>"
    elif "Sell" in recommendation:
        return f"<span style='color:red;'>{recommendation}</span>"
    elif "Hold" in recommendation:
        return f"<span style='color:orange;'>{recommendation}</span>"
    else:
        return recommendation  # Default case, no color formatting


def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    """Enhanced UI with color coding and tooltips"""
    st.title("📈 StockGenie Pro - NSE Analysis")
    # Current date header
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

    # Daily Suggestions Button
    if st.button("🚀 Generate Daily Top Picks"):
        with st.spinner("🔍 Scanning market..."):
            results_df = analyze_all_stocks(NSE_STOCKS)
            # Display results with color coding
            st.subheader("🏆 Today's Top 10 Stocks")
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}**: {row['Current Price']:.2f}  
                    Buy At**: {row['Buy At']:.2f} | **Stop Loss**: {row['Stop Loss']:.2f}  
                    Target**: {row['Target']:.2f}  
                    Intraday**: {colored_recommendation(row['Intraday'])}  
                    Swing**: {colored_recommendation(row['Swing'])}  
                    Short-Term**: {colored_recommendation(row['Short-Term'])}  
                    Long-Term**: {colored_recommendation(row['Long-Term'])}
                    """, unsafe_allow_html=True)

    # Intraday Suggestions Button
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        with st.spinner("🔍 Scanning market for intraday opportunities..."):
            intraday_results = analyze_intraday_stocks(NSE_STOCKS)
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}**: {row['Current Price']:.2f}  
                    Buy At**: {row['Buy At']:.2f} | **Stop Loss**: {row['Stop Loss']:.2f}  
                    Target**: {row['Target']:.2f}  
                    Intraday**: {colored_recommendation(row['Intraday'])}  
                    """, unsafe_allow_html=True)

    # Individual Stock Analysis
    if symbol:
        st.header(f"📊 {symbol.split('.')[0]} Analysis")
        # Price Levels Card
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']),
                      f"₹{recommendations['Current Price']:.2f}")
        with col2:
            st.metric(tooltip("Buy At", "Recommended entry price"),
                      f"₹{recommendations['Buy At']:.2f}")
        with col3:
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']),
                      f"₹{recommendations['Stop Loss']:.2f}")
        with col4:
            st.metric(tooltip("Target", "Price target based on risk/reward"),
                      f"₹{recommendations['Target']:.2f}")
        # Recommendations Section
        st.subheader("📋 Trading Recommendations")
        cols = st.columns(4)
        strategy_names = ["Intraday", "Swing", "Short-Term", "Long-Term"]
        for col, strategy in zip(cols, strategy_names):
            with col:
                st.markdown(f"**{strategy}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]),
                            unsafe_allow_html=True)
        # Chart Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Price Action", "📊 Indicators", "📉 Volatility"])
        with tab1:
            fig = px.line(data, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20'],
                          title="Price with Moving Averages")
            st.plotly_chart(fig)
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'],
                          title="Momentum Indicators")
            st.plotly_chart(fig)
        with tab3:
            fig = px.line(data, y=['ATR', 'Upper_Band', 'Lower_Band'],
                          title="Volatility Analysis")
            st.plotly_chart(fig)


def analyze_intraday_stocks(stock_list, batch_size=50):
    """Analyze all stocks for intraday trading and return top 5 picks"""
    results = []
    for i in tqdm(range(0, len(stock_list), batch_size), desc="Processing Batches for Intraday"):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
    results_df = pd.DataFrame(results)
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    # Filter for intraday-specific criteria
    intraday_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    intraday_df = intraday_df.sort_values(by="Score", ascending=False).head(5)
    return intraday_df


def main():
    """Main function with enhanced input validation"""
    st.sidebar.title("🔍 Stock Search")
    # Dynamically fetch NSE stock list or use fallback
    NSE_STOCKS = fetch_nse_stock_list()
    # Symbol input with validation
    symbol = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x != "Custom" else x
    )
    if symbol == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g.: RELIANCE):")
        symbol = f"{custom_symbol}.NS" if custom_symbol else None
    if symbol:
        if ".NS" not in symbol:
            symbol += ".NS"
        if symbol not in NSE_STOCKS and not st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable"):
            return
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)  # Pass NSE_STOCKS here
        else:
            st.error("❌ Failed to load data for this symbol")


if __name__ == "__main__":
    main()
