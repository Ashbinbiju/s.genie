"""Comprehensive stock database organized by sectors - NSE Exchange"""

# Note: Token values need to be fetched from Angel One API via search
# This structure provides the organization, actual tokens should be fetched dynamically

STOCKS_BY_SECTOR = {
    "Top 20 Popular": {
        "RELIANCE": "2885",
        "TCS": "11536",
        "INFY": "1594",
        "HDFCBANK": "1333",
        "ICICIBANK": "1330",
        "SBIN": "3045",
        "BHARTIARTL": "10604",
        "ITC": "1660",
        "HINDUNILVR": "1394",
        "LT": "11483",
        "KOTAKBANK": "1922",
        "AXISBANK": "5900",
        "SUNPHARMA": "3351",
        "TATAMOTORS": "3456",
        "TATASTEEL": "3499",
        "ONGC": "2475",
        "NTPC": "11630",
        "POWERGRID": "14977",
        "MARUTI": "10999",
        "WIPRO": "3787"
    },
    "Banking & Finance": {
        "HDFCBANK": "1333",
        "ICICIBANK": "1330",
        "SBIN": "3045",
        "KOTAKBANK": "1922",
        "AXISBANK": "5900",
        "INDUSINDBK": "5258",
        "BANDHANBNK": "579",
        "FEDERALBNK": "1023",
        "PNB": "10666",
        "CANBK": "3045",
        "BANKBARODA": "4668",
        "IDFCFIRSTB": "11184",
        "RBLBANK": "18391",
        "AUBANK": "21238"
    },
    "IT & Software": {
        "TCS": "11536",
        "INFY": "1594",
        "WIPRO": "3787",
        "HCLTECH": "7229",
        "TECHM": "13538",
        "LTIM": "17818",
        "PERSISTENT": "14413",
        "COFORGE": "11543",
        "MPHASIS": "4503",
        "LTTS": "18564"
    },
    "Auto & Auto Ancillary": {
        "TATAMOTORS": "3456",
        "MARUTI": "10999",
        "M&M": "2031",
        "BAJAJ-AUTO": "16669",
        "EICHERMOT": "910",
        "HEROMOTOCO": "1348",
        "BOSCHLTD": "2181",
        "MOTHERSON": "4204",
        "TVSMOTOR": "8479",
        "APOLLOTYRE": "163"
    },
    "Pharma & Healthcare": {
        "SUNPHARMA": "3351",
        "DRREDDY": "881",
        "CIPLA": "694",
        "DIVISLAB": "10940",
        "LUPIN": "10440",
        "AUROPHARMA": "2748",
        "BIOCON": "11373",
        "TORNTPHARM": "3518",
        "ALKEM": "11703",
        "LAURUSLABS": "17818",
        "THYROCARE": "20489"
    },
    "Energy & Power": {
        "RELIANCE": "2885",
        "ONGC": "2475",
        "IOC": "1624",
        "BPCL": "526",
        "NTPC": "11630",
        "POWERGRID": "14977",
        "ADANIPOWER": "25",
        "TATAPOWER": "3426",
        "GAIL": "4717",
        "HINDPETRO": "1348"
    },
    "FMCG & Consumer": {
        "ITC": "1660",
        "HINDUNILVR": "1394",
        "BRITANNIA": "547",
        "NESTLEIND": "17963",
        "DABUR": "772",
        "MARICO": "4067",
        "GODREJCP": "10099",
        "COLGATE": "15141",
        "TATACONSUM": "3432",
        "EMAMILTD": "3920"
    },
    "Metals & Mining": {
        "TATASTEEL": "3499",
        "HINDALCO": "1363",
        "JSWSTEEL": "11723",
        "COALINDIA": "5215",
        "VEDL": "3063",
        "JINDALSTEL": "1316",
        "SAIL": "2963",
        "NATIONALUM": "6364",
        "NMDC": "15332",
        "HINDZINC": "1363"
    },
    "Cement & Construction": {
        "ULTRACEMCO": "2952",
        "GRASIM": "1232",
        "SHREECEM": "3103",
        "AMBUJACEM": "1270",
        "ACC": "22",
        "DALMIBHARA": "8075",
        "RAMCOCEM": "5290",
        "JKCEMENT": "13270",
        "INDIACEM": "1516"
    },
    "Telecom": {
        "BHARTIARTL": "10604",
        "IDEA": "7929",
        "INDUSTOWER": "29135"
    },
    "Realty": {
        "DLF": "966",
        "GODREJPROP": "17875",
        "OBEROIRLTY": "13258",
        "PRESTIGE": "14604",
        "PHOENIXLTD": "14332",
        "BRIGADE": "16669"
    }
}

# Flatten for easy access
ALL_STOCKS = {}
for sector, stocks in STOCKS_BY_SECTOR.items():
    ALL_STOCKS.update(stocks)
