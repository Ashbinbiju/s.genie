import yfinance as yf
import pandas as pd
import ta
import streamlit as st
from datetime import datetime
import requests
import io
import random
from textblob import TextBlob

# API Keys
NEWSAPI_KEY = "your_newsapi_key"
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key"

# Initialize Portfolio
portfolio = pd.DataFrame(columns=[
    "Symbol", "Buy Price", "Target Price", "Stop Loss", "Current Price",
    "Quantity", "Status", "Strategy", "Purchase Date"
])

# Fetch Live NSE Stock List
def fetch_nse_stock_list():
    """Fetch live NSE stock list"""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# Fetch Stock Data
def fetch_stock_data(symbol, period="5y", interval="1d"):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period=period, interval=interval)
    except:
        return pd.DataFrame()

# Sentiment Analysis
def fetch_news_sentiment(query):
    """Fetch news sentiment using NewsAPI"""
    url = f'https://gnews.io/api/v4/search?q={query}&token=e4f5f1442641400694645433a8f98b94&lang=en&max=5'
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        sentiments = [TextBlob(a["title"]).sentiment.polarity for a in articles]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except:
        return 0

# Calculate Stop-Loss & Target
def calculate_stop_loss(data):
    if 'ATR' in data.columns:
        return round(data['Close'].iloc[-1] - (2 * data['ATR'].iloc[-1]), 2)
    return None

def calculate_target(data):
    stop_loss = calculate_stop_loss(data)
    if stop_loss:
        return round(data['Close'].iloc[-1] + (3 * (data['Close'].iloc[-1] - stop_loss)), 2)
    return None

# Stock Scoring System
def calculate_stock_score(data, news_sentiment=0):
    buy_score = 0
    if 'RSI' in data and data['RSI'].iloc[-1] < 30:
        buy_score += 2
    if 'MACD' in data and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        buy_score += 1
    if 'Close' in data and 'Lower_Band' in data and data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
        buy_score += 1
    if news_sentiment > 0.2:
        buy_score += 1
    return "Strong Buy" if buy_score >= 5 else "Buy" if buy_score >= 3 else "Hold", buy_score

# Select Top 10 Stocks
def select_top_stocks(stock_list):
    ranked_stocks = []
    for symbol in stock_list:
        data = fetch_stock_data(symbol)
        if not data.empty:
            news_sentiment = fetch_news_sentiment(symbol.split(".")[0])
            recommendation, score = calculate_stock_score(data, news_sentiment)
            ranked_stocks.append({
                "Symbol": symbol,
                "Recommendation": recommendation,
                "Score": score,
                "Buy At": data['Close'].iloc[-1],
                "Stop Loss": calculate_stop_loss(data),
                "Target": calculate_target(data),
                "Current Price": data['Close'].iloc[-1]
            })
    return pd.DataFrame(ranked_stocks).sort_values(by="Score", ascending=False).head(10)

# Auto-Add Best Stocks to Portfolio
def auto_add_to_portfolio():
    global portfolio
    top_stocks = select_top_stocks(fetch_nse_stock_list())
    for _, row in top_stocks.iterrows():
        if row["Recommendation"] in ["Strong Buy", "Buy"]:
            portfolio = pd.concat([portfolio, pd.DataFrame([{
                "Symbol": row['Symbol'],
                "Buy Price": row['Buy At'],
                "Target Price": row['Target'],
                "Stop Loss": row['Stop Loss'],
                "Current Price": row['Current Price'],
                "Quantity": 10,
                "Status": "Active",
                "Strategy": "Intraday",
                "Purchase Date": datetime.now().strftime("%Y-%m-%d"),
            }])], ignore_index=True)
    return portfolio

# Auto-Sell Stocks
def auto_sell(live_prices):
    global portfolio
    for index, row in portfolio.iterrows():
        if row["Status"] == "Active":
            current_price = live_prices.get(row["Symbol"], row["Current Price"])
            if current_price <= row["Stop Loss"]:
                portfolio.at[index, "Status"] = "Sold (Stop Loss)"
            elif current_price >= row["Target Price"]:
                portfolio.at[index, "Status"] = "Sold (Target Achieved)"
    return portfolio

# Display Portfolio in Streamlit
def display_portfolio():
    st.subheader("📈 Virtual Portfolio")
    if portfolio.empty:
        st.warning("No stocks in portfolio yet!")
    else:
        st.dataframe(portfolio)
    active_stocks = portfolio[portfolio["Status"] == "Active"]
    if not active_stocks.empty:
        total_investment = (active_stocks["Buy Price"] * active_stocks["Quantity"]).sum()
        current_value = (active_stocks["Current Price"] * active_stocks["Quantity"]).sum()
        total_profit_loss = current_value - total_investment
        st.metric(label="Total Portfolio Value", value=f"₹{current_value:.2f}")
        st.metric(label="Profit/Loss", value=f"₹{total_profit_loss:.2f}")
    return portfolio

# Main Function
def main():
    st.title("📊 AI-Powered Virtual Portfolio")

    # Fetch Live Prices
    live_prices = {symbol: fetch_stock_data(symbol).iloc[-1]["Close"] for symbol in portfolio["Symbol"].unique()}

    # Generate Top Picks
    if st.button("🚀 Generate Daily Top Stocks"):
        auto_add_to_portfolio()
        st.success("Top stocks added to portfolio!")

    # Show Portfolio
    display_portfolio()

    # Run Auto-Sell Logic
    if st.button("🔴 Auto-Sell Stocks"):
        auto_sell(live_prices)
        st.success("Auto-Sell executed!")

    # Updated Portfolio
    st.subheader("📉 Updated Portfolio After Auto-Sell")
    st.dataframe(portfolio)

# Run Streamlit App
if __name__ == "__main__":
    main()