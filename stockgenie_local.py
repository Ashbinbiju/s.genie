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
from textblob import TextBlob

# API Keys
ALPHA_VANTAGE_API_KEY = "TCAUKYUCIDZ6PI57"  # Replace with your Alpha Vantage API key
NEWS_API_KEY = "ed58659895e84dfb8162a8bb47d8525e"  # Replace with your NewsAPI key
GNEWS_API_KEY = "e4f5f1442641400694645433a8f98b94"  # Replace with your GNews API key

# Tooltip explanations
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Current Price": "Latest closing price of the stock",
    "Buy At": "Recommended entry price based on RSI and current price",
    "Target": "Price target based on risk-reward ratio",
    "RVOL": "Relative Volume - Identifies unusual trading activity",
    "Fibonacci": "Support & Resistance levels based on Fibonacci retracement",
    "News Sentiment": "Sentiment score based on recent news articles (0=Negative, 1=Positive)",
}

# Tooltip function
def tooltip(label, explanation):
    """Returns a formatted tooltip string"""
    return f"{label} 📌 ({explanation})"

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
        nse_data = pd.read_csv(io.StringIO(response.text))
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
    if data.empty:
        st.warning("⚠️ No data available for analysis.")
        return data

    # Ensure enough data points for calculations
    min_data_points = max(27, 50)  # ADX requires 27, SMA_200 requires 50
    if len(data) < min_data_points:
        st.warning(f"⚠️ Insufficient data points ({len(data)}). At least {min_data_points} required for full analysis.")
        return data

    try:
        # RSI (Optimized to 9-period for faster reactions)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=9).rsi()
    except Exception as e:
        st.warning(f"⚠️ Error calculating RSI: {e}")
        data['RSI'] = None

    try:
        # MACD (Optimized to (8, 17, 9) for earlier signals)
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
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
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Error calculating Moving Averages: {e}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None

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
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
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

    try:
        # OBV (On-Balance Volume)
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Error calculating OBV: {e}")
        data['OBV'] = None

    try:
        # VWAP Calculation
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Error calculating VWAP: {e}")
        data['VWAP'] = None

    try:
        # Volume Spike Check
        data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.warning(f"⚠️ Error calculating Volume Spike: {e}")
        data['Volume_Spike'] = None

    try:
        # Relative Volume (RVOL)
        data['RVOL'] = data['Volume'] / data['Volume'].rolling(20).mean()
    except Exception as e:
        st.warning(f"⚠️ Error calculating RVOL: {e}")
        data['RVOL'] = None

    try:
        # Fibonacci Retracement Levels
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - 0.236 * diff
        data['Fib_38.2'] = high - 0.382 * diff
        data['Fib_50.0'] = high - 0.500 * diff
        data['Fib_61.8'] = high - 0.618 * diff
        data['Fib_78.6'] = high - 0.786 * diff
    except Exception as e:
        st.warning(f"⚠️ Error calculating Fibonacci Levels: {e}")
        data['Fib_23.6'] = None
        data['Fib_38.2'] = None
        data['Fib_50.0'] = None
        data['Fib_61.8'] = None
        data['Fib_78.6'] = None

    return data

def calculate_stop_loss(data, atr_multiplier=2.5):
    """Calculate stop-loss level based on ATR"""
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    # Adjust multiplier based on trend
    if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25:
        atr_multiplier = 3.0  # Strong trend
    else:
        atr_multiplier = 1.5  # Sideways market
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

def calculate_target(data, risk_reward_ratio=3):
    """Calculate target price based on risk-reward ratio"""
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    # Adjust risk-reward ratio based on trend
    if 'ADX' in data.columns and data['ADX'].iloc[-1] > 25:
        risk_reward_ratio = 3  # Strong trend
    else:
        risk_reward_ratio = 1.5  # Weak trend
    target = last_close + (risk * risk_reward_ratio)
    return round(target, 2)

def fetch_alpha_vantage_sentiment(symbol):
    """
    Fetch news sentiment for a stock using Alpha Vantage.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol.split('.')[0],  # Remove '.NS' from the symbol
        "apikey": "TCAUKYUCIDZ6PI57",
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "feed" in data:
            return data["feed"]
        else:
            st.warning(f"⚠️ No news sentiment data found for {symbol}.")
            return []
    except Exception as e:
        st.error(f"❌ Failed to fetch news sentiment for {symbol}. Error: {str(e)}")
        return []

def get_alpha_vantage_sentiment_score(news_feed):
    """
    Calculate the average sentiment score from Alpha Vantage news feed.
    """
    if not news_feed:
        return None

    total_sentiment = 0
    for article in news_feed:
        sentiment_score = float(article.get("overall_sentiment_score", 0))
        total_sentiment += sentiment_score

    return total_sentiment / len(news_feed)

def generate_recommendations(data, symbol):
    """Generate comprehensive trade recommendations with Alpha Vantage news sentiment analysis"""
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0,
        "News Sentiment": None  # Add news sentiment
    }
    if data.empty:
        st.warning("⚠️ No data available for generating recommendations.")
        return recommendations

    # Ensure Current Price is populated
    if not data['Close'].empty:
        recommendations["Current Price"] = data['Close'].iloc[-1]
    else:
        st.warning("⚠️ No closing price data available.")
        return recommendations

    # Fetch news sentiment from Alpha Vantage
    news_feed = fetch_alpha_vantage_sentiment(symbol)
    if news_feed:
        sentiment_score = get_alpha_vantage_sentiment_score(news_feed)
        recommendations["News Sentiment"] = sentiment_score

    # Add sentiment to the scoring system
    if recommendations["News Sentiment"] is not None:
        if recommendations["News Sentiment"] > 0.6:  # Positive sentiment
            recommendations["Score"] += 1
        elif recommendations["News Sentiment"] < 0.4:  # Negative sentiment
            recommendations["Score"] -= 1

    # Rest of the recommendation logic...
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
            "Score": recommendations.get("Score", 0),
            "News Sentiment": recommendations.get("News Sentiment", None),
        }
    return None

def analyze_all_stocks(stock_list, batch_size=50, price_range=None):
    """Analyze all stocks in the list using batch processing"""
    results = []
    for i in tqdm(range(0, len(stock_list), batch_size), desc="Processing Batches"):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
    results_df = pd.DataFrame(results)
    if "Score" not in results_df.columns:
        results_df["Score"] = 0

    # Filter by price range if provided
    if price_range:
        results_df = results_df[(results_df['Current Price'] >= price_range[0]) & (results_df['Current Price'] <= price_range[1])]
    results_df = results_df.sort_values(by="Score", ascending=False).head(10)
    return results_df

def colored_recommendation(recommendation):
    """Returns a color-coded recommendation for Streamlit"""
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation  # Default case, no color formatting

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    """Enhanced UI with color coding and tooltips"""
    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

    # Display news sentiment
    if recommendations and "News Sentiment" in recommendations:
        sentiment = recommendations["News Sentiment"]
        if sentiment is not None:
            sentiment_label = "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral"
            st.metric("📰 News Sentiment", f"{sentiment_label} ({sentiment:.2f})")

    # Price Range Slider
    price_range = st.sidebar.slider(
        "Select Price Range (₹)",
        min_value=0, max_value=10000, value=(100, 1000)
    )

    # Daily Suggestions Button
    if st.button("🚀 Generate Daily Top Picks"):
        with st.spinner("⏳ Scanning market..."):
            results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range)
            st.subheader("🏆 Today's Top 10 Stocks")
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Current Price'])}: ₹{row['Current Price']:.2f}  
                    Buy At: ₹{row['Buy At']:.2f} | Stop Loss: ₹{row['Stop Loss']:.2f}  
                    Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Long-Term: {colored_recommendation(row['Long-Term'])}
                    """, unsafe_allow_html=True)

    # Intraday Suggestions Button
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        with st.spinner("⏳ Scanning market for intraday opportunities..."):
            intraday_results = analyze_intraday_stocks(NSE_STOCKS, price_range=price_range)
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - Score: {row['Score']}/5"):
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Current Price'])}: ₹{row['Current Price']:.2f}  
                    Buy At: ₹{row['Buy At']:.2f} | Stop Loss: ₹{row['Stop Loss']:.2f}  
                    Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    """, unsafe_allow_html=True)

    # Individual Stock Analysis
    if symbol:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = recommendations.get("Current Price")
            if current_price is not None:
                st.metric(tooltip("Current Price", TOOLTIPS['Current Price']), f"₹{current_price:.2f}")
            else:
                st.metric(tooltip("Current Price", TOOLTIPS['Current Price']), "N/A")
        with col2:
            buy_at = recommendations.get("Buy At")
            if buy_at is not None:
                st.metric(tooltip("Buy At", TOOLTIPS['Buy At']), f"₹{buy_at:.2f}")
            else:
                st.metric(tooltip("Buy At", TOOLTIPS['Buy At']), "N/A")
        with col3:
            stop_loss = recommendations.get("Stop Loss")
            if stop_loss is not None:
                st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss:.2f}")
            else:
                st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), "N/A")
        with col4:
            target = recommendations.get("Target")
            if target is not None:
                st.metric(tooltip("Target", TOOLTIPS['Target']), f"₹{target:.2f}")
            else:
                st.metric(tooltip("Target", TOOLTIPS['Target']), "N/A")

        st.subheader("📈 Trading Recommendations")
        cols = st.columns(4)
        strategy_names = ["Intraday", "Swing", "Short-Term", "Long-Term"]
        for col, strategy in zip(cols, strategy_names):
            with col:
                st.markdown(f"**{strategy}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]), unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Action", "📉 Indicators", "📊 Volatility", "📏 Fibonacci"])
        with tab1:
            fig = px.line(data, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50'], title="Price with Moving Averages")
            st.plotly_chart(fig)
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], title="Momentum Indicators")
            st.plotly_chart(fig)
        with tab3:
            fig = px.line(data, y=['ATR', 'Upper_Band', 'Lower_Band'], title="Volatility Analysis")
            st.plotly_chart(fig)
        with tab4:
            fig = px.line(data, y=['Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_78.6'], title="Fibonacci Retracement Levels")
            st.plotly_chart(fig)


def analyze_intraday_stocks(stock_list, batch_size=50, price_range=None):
    """Analyze all stocks for intraday trading and return top 5 picks"""
    results = []
    for i in tqdm(range(0, len(stock_list), batch_size), desc="Processing Batches for Intraday"):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
    results_df = pd.DataFrame(results)
    if "Score" not in results_df.columns:
        results_df["Score"] = 0

    # Filter by price range if provided
    if price_range:
        results_df = results_df[(results_df['Current Price'] >= price_range[0]) & (results_df['Current Price'] <= price_range[1])]
    intraday_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    intraday_df = intraday_df.sort_values(by="Score", ascending=False).head(5)
    return intraday_df

def main():
    """Main function with enhanced input validation"""
    st.sidebar.title("🔍 Stock Search")
    NSE_STOCKS = fetch_nse_stock_list()

    # Symbol input with validation
    symbol = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=NSE_STOCKS + ["Custom"],
        format_func=lambda x: x.split('.')[0] if x != "Custom" else x
    )
    if symbol == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g.: RELIANCE):")
        if custom_symbol:
            symbol = f"{custom_symbol.strip().upper()}.NS"
        else:
            symbol = None

    if symbol:
        # Validate symbol
        if symbol not in NSE_STOCKS:
            st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable.")

        # Fetch and analyze data
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            data = analyze_stock(data)
            recommendations = generate_recommendations(data, symbol)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error("❌ Failed to load data for this symbol. Please check the symbol and try again.")

if __name__ == "__main__":
    main()