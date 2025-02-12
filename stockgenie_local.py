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
from textblob import TextBlob  # For sentiment analysis

# API Keys
NEWSAPI_KEY = "ed58659895e84dfb8162a8bb47d8525e"
GNEWS_KEY = "e4f5f1442641400694645433a8f98b94"
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

def fetch_us_market_sentiment():
    """Fetch US market sentiment using Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey=TCAUKYUCIDZ6PI57"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "Global Quote" not in data:
            return 0  # Neutral sentiment if data is unavailable
        change_percent = float(data["Global Quote"]["10. change percent"].strip("%"))
        if change_percent > 0:
            return 1  # Positive sentiment
        elif change_percent < 0:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch US market sentiment: {e}")
        return 0

def fetch_news_sentiment(query, api_key, source="newsapi"):
    """Fetch news sentiment using NewsAPI or GNews"""
    try:
        if source == "newsapi":
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey=ed58659895e84dfb8162a8bb47d8525e&language=en&sortBy=publishedAt&pageSize=5"
        elif source == "gnews":
            url = f"https://gnews.io/api/v4/search?q={query}&token=e4f5f1442641400694645433a8f98b94&lang=en&max=5"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        sentiment_scores = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            sentiment = TextBlob(text).sentiment.polarity
            sentiment_scores.append(sentiment)
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch news sentiment for {query}: {e}")
        return 0

def analyze_stock(data):
    """Perform technical analysis on stock data"""
    if data.empty or len(data) < 27:  # Ensure enough data points for ADX calculation
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

def generate_recommendations(data):
    """Generate comprehensive trade recommendations"""
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }
    if data.empty:
        return recommendations
    try:
        # Current Price
        recommendations["Current Price"] = data['Close'].iloc[-1]
        # Multi-Factor Scoring System
        buy_score = 0
        sell_score = 0
        # Condition 1: RSI < 30 (Oversold) or > 70 (Overbought)
        if 'RSI' in data.columns:
            if data['RSI'].iloc[-1] < 30:
                buy_score += 2  # Higher weight for RSI
            elif data['RSI'].iloc[-1] > 70:
                sell_score += 2
        # Condition 2: MACD Crossover
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                buy_score += 1
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                sell_score += 1
        # Condition 3: Bollinger Band Reversion
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:  # Oversold condition
                buy_score += 1
            elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:  # Overbought condition
                sell_score += 1
        # Condition 4: VWAP Trend
        if 'VWAP' in data.columns:
            if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:  # Price above VWAP indicates bullish trend
                buy_score += 1
            elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:  # Price below VWAP indicates bearish trend
                sell_score += 1
        # Condition 5: Volume Confirmation
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].rolling(window=10).mean().iloc[-1]
            if data['Volume'].iloc[-1] > avg_volume * 1.5:  # High volume confirms trend
                buy_score += 1
            elif data['Volume'].iloc[-1] < avg_volume * 0.5:  # Low volume indicates weakness
                sell_score += 1
        # Assign Recommendations Based on Scores
        if buy_score >= 4:  # Require stronger confirmation for "Strong Buy"
            recommendations["Intraday"] = "Strong Buy"
        elif sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
        # Calculate Trade Levels
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Target"] = calculate_target(data)
        # Final Score (Optional)
        recommendations["Score"] = max(0, min(buy_score - sell_score, 5))  # Keep score between 0-5
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
            "Score": recommendations.get("Score", 0),
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
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{row['Current Price']:.2f}  
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
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{row['Current Price']:.2f}  
                    Buy At: ₹{row['Buy At']:.2f} | Stop Loss: ₹{row['Stop Loss']:.2f}  
                    Target: ₹{row['Target']:.2f}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    """, unsafe_allow_html=True)
    # Individual Stock Analysis
    if symbol:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{recommendations['Current Price']:.2f}")
        with col2:
            st.metric(tooltip("Buy At", "Recommended entry price"), f"₹{recommendations['Buy At']:.2f}")
        with col3:
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{recommendations['Stop Loss']:.2f}")
        with col4:
            st.metric(tooltip("Target", "Price target based on risk/reward"), f"₹{recommendations['Target']:.2f}")
        st.subheader("📈 Trading Recommendations")
        cols = st.columns(4)
        strategy_names = ["Intraday", "Swing", "Short-Term", "Long-Term"]
        for col, strategy in zip(cols, strategy_names):
            with col:
                st.markdown(f"**{strategy}**", unsafe_allow_html=True)
                st.markdown(colored_recommendation(recommendations[strategy]), unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📊 Price Action", "📉 Indicators", "📊 Volatility"])
        with tab1:
            fig = px.line(data, y=['Close', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50'], title="Price with Moving Averages")
            st.plotly_chart(fig)
        with tab2:
            fig = px.line(data, y=['RSI', 'MACD', 'MACD_signal'], title="Momentum Indicators")
            st.plotly_chart(fig)
        with tab3:
            fig = px.line(data, y=['ATR', 'Upper_Band', 'Lower_Band'], title="Volatility Analysis")
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
    
    # Initialize symbol as None
    symbol = None
    
    # Symbol input with validation
    selected_option = st.sidebar.selectbox(
        "Choose or enter stock:",
        options=[""] + NSE_STOCKS + ["Custom"],  # Add an empty option at the beginning
        format_func=lambda x: x.split('.')[0] if x != "Custom" and x != "" else x
    )
    
    if selected_option == "Custom":
        custom_symbol = st.sidebar.text_input("Enter NSE Symbol (e.g.: RELIANCE):")
        if custom_symbol:
            symbol = f"{custom_symbol}.NS"
    elif selected_option != "":  # If the user selects a stock from the list
        symbol = selected_option
    
    # Display the dashboard (works regardless of stock selection)
    if symbol:
        if ".NS" not in symbol:
            symbol += ".NS"
        if symbol not in NSE_STOCKS:
            st.sidebar.warning("⚠️ Unverified symbol - data may be unreliable")
        else:
            data = fetch_stock_data_cached(symbol)
            if not data.empty:
                data = analyze_stock(data)
                recommendations = generate_recommendations(data)
                display_dashboard(symbol, data, recommendations, NSE_STOCKS)
            else:
                st.error("❌ Failed to load data for this symbol")
    else:
        # If no stock is selected, display the dashboard without individual stock analysis
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()