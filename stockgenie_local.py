import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
import requests
import time

# Load environment variables
load_dotenv()

# Angel One API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEY = os.getenv("API_KEY")

# NSE sector indices and their symbols (as per Angel One's naming convention)
SECTOR_INDICES = {
    "Nifty Bank": "BANKNIFTY",
    "Nifty IT": "CNXIT",
    "Nifty Auto": "NIFTYAUTO",
    "Nifty FMCG": "NIFTYFMCG",
    "Nifty Pharma": "NIFTYPHARMA",
    "Nifty Metal": "NIFTYMETAL",
    "Nifty Realty": "NIFTYREALTY"
}

# Initialize SmartAPI client
def init_smartapi_client():
    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        if data['status']:
            print("Successfully authenticated with SmartAPI")
            return smart_api
        else:
            print(f"Authentication failed: {data['message']}")
            return None
    except Exception as e:
        print(f"Error initializing SmartAPI: {str(e)}")
        return None

# Load instrument tokens from Angel One's scrip master
def load_symbol_token_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        print(f"Failed to load instrument list: {str(e)}")
        return {}

# Fetch LTP or candle data for a given symbol
def fetch_sector_data(smart_api, symbol, token):
    try:
        # Try fetching real-time LTP
        ltp_data = smart_api.ltpData(exchange="NSE", tradingsymbol=symbol, symboltoken=token)
        if ltp_data['status'] and 'data' in ltp_data and ltp_data['data']:
            ltp = ltp_data['data']['ltp']
        else:
            # Fallback to intraday candle data if LTP is unavailable
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            candle_data = smart_api.getCandleData({
                "exchange": "NSE",
                "symboltoken": token,
                "interval": "ONE_DAY",
                "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
                "todate": end_date.strftime("%Y-%m-%d %H:%M")
            })
            if candle_data['status'] and candle_data['data']:
                ltp = candle_data['data'][-1][4]  # Latest close price
            else:
                print(f"No data for {symbol}: {candle_data.get('message', 'Unknown error')}")
                return None, None

        # Fetch previous day's close from candle data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=1)
        candle_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })
        if candle_data['status'] and candle_data['data']:
            prev_close = candle_data['data'][-1][4]  # Previous day's close
            return ltp, prev_close
        else:
            print(f"No previous close data for {symbol}")
            return ltp, None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

# Calculate percentage change and rank sectors
def analyze_sectors():
    smart_api = init_smartapi_client()
    if not smart_api:
        return None

    symbol_token_map = load_symbol_token_map()
    sector_data = []

    for sector_name, symbol in SECTOR_INDICES.items():
        token = symbol_token_map.get(symbol)
        if not token:
            print(f"Token not found for {sector_name} ({symbol})")
            continue

        ltp, prev_close = fetch_sector_data(smart_api, symbol, token)
        if ltp is not None and prev_close is not None:
            try:
                percent_change = ((ltp - prev_close) / prev_close) * 100
                sector_data.append({
                    "Sector": sector_name,
                    "LTP": round(ltp, 2),
                    "Previous Close": round(prev_close, 2),
                    "Percentage Change": round(percent_change, 2)
                })
            except Exception as e:
                print(f"Error calculating percentage change for {sector_name}: {str(e)}")
        time.sleep(1)  # Avoid API rate limits

    if not sector_data:
        print("No valid data retrieved for any sector")
        return None

    # Create DataFrame and rank sectors
    df = pd.DataFrame(sector_data)
    df = df.sort_values(by="Percentage Change", ascending=False)
    
    # Save to CSV
    today = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(f"sector_performance_{today}.csv", index=False)
    print(f"Results saved to sector_performance_{today}.csv")
    
    return df

# Visualize top 3 sectors
def visualize_top_sectors(df):
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    top_3 = df.head(3)
    fig = px.bar(
        top_3,
        x="Sector",
        y="Percentage Change",
        title=f"Top Performing NSE Sectors - {datetime.now().strftime('%d %b %Y')}",
        labels={"Percentage Change": "% Change"},
        color="Percentage Change",
        color_continuous_scale="Viridis",
        text="Percentage Change"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="auto")
    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title="Percentage Change (%)",
        showlegend=False
    )
    fig.show()

# Main function
def main():
    print(f"Running sector analysis for {datetime.now().strftime('%d %b %Y')}")
    df = analyze_sectors()
    
    if df is not None and not df.empty:
        # Display top 3 sectors
        print("\nTop 3 Performing Sectors:")
        for idx, row in df.head(3).iterrows():
            print(f"{idx + 1}. {row['Sector']}: +{row['Percentage Change']}%")
        
        # Visualize results
        visualize_top_sectors(df)
    else:
        print("Failed to retrieve sector performance data")

if __name__ == "__main__":
    main()
