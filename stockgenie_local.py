import pandas as pd
import ta
import logging
import numpy as np
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
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
import warnings
import sqlite3
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from scipy.stats.mstats import winsorize
from streamlit import cache_data
from itertools import cycle


load_dotenv()

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.2478.80 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Brave/124.0.0.0"
]

cache = Cache("stock_data_cache")

TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Parabolic_SAR": "Parabolic Stop and Reverse - Trend reversal indicator",
    "Fib_Retracements": "Fibonacci Retracements - Support and resistance levels",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
    "Keltner": "Keltner Channels - Volatility bands based on EMA and ATR",
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
    "Score": "Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal."
}

SECTORS = {
 
  "Bank": [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS", "BANDHANBNK.NS", "INDIANB.NS",
    "BANKINDIA.NS", "KARURVYSYA.NS", "CUB.NS", "J&KBANK.NS", "DCBBANK.NS",
    "AUBANK.NS", "YESBANK.NS", "IDBI.NS", "SOUTHBANK.NS", "CSBBANK.NS",
    "TMB.NS", "KTKBANK.NS", "EQUITASBNK.NS", "UJJIVANSFB.NS"
  ],

  "Software & IT Services": [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
    "MPHASIS.NS", "FSL.NS", "BSOFT.NS", "NEWGEN.NS", "ZENSARTECH.NS",
    "RATEGAIN.NS", "TANLA.NS", "COFORGE.NS", "PERSISTENT.NS", "CYIENT.NS",
    "SONATSOFTW.NS", "KPITTECH.NS", "BIRLASOFT.NS", "TATAELXSI.NS", "MINDTREE.NS",
    "INTELLECT.NS", "HAPPSTMNDS.NS", "MASTEK.NS", "ECLERX.NS", "NIITLTD.NS",
    "RSYSTEMS.NS", "XCHANGING.NS", "OFSS.NS", "AURIONPRO.NS", "DATAMATICS.NS",
    "QUICKHEAL.NS", "CIGNITITEC.NS","SAGILITY.NS" "ALLSEC.NS"
  ],

  "Finance": [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
    "SBICARD.NS", "M&MFIN.NS", "MUTHOOTFIN.NS", "LICHSGFIN.NS", "IDFCFIRSTB.NS",
    "AUBANK.NS", "POONAWALLA.NS", "SUNDARMFIN.NS", "IIFL.NS", "ABCAPITAL.NS",
    "L&TFH.NS", "CREDITACC.NS", "MANAPPURAM.NS", "DHANI.NS", "JMFINANCIL.NS",
    "EDELWEISS.NS", "INDIASHLTR.NS", "MOTILALOFS.NS", "CDSL.NS", "BSE.NS",
    "MCX.NS", "ANGELONE.NS", "KARURVYSYA.NS", "RBLBANK.NS", "PNB.NS",
    "CANBK.NS", "UNIONBANK.NS", "IOB.NS", "YESBANK.NS", "UCOBANK.NS",
    "BANKINDIA.NS", "CENTRALBK.NS", "IDBI.NS", "J&KBANK.NS", "DCBBANK.NS",
    "FEDERALBNK.NS", "SOUTHBANK.NS", "CSBBANK.NS", "TMB.NS", "KTKBANK.NS",
    "EQUITASBNK.NS", "UJJIVANSFB.NS", "BANDHANBNK.NS", "SURYODAY.NS", "FSL.NS",
    "PSB.NS", "PFS.NS", "HDFCAMC.NS", "NAM-INDIA.NS", "UTIAMC.NS", "ABSLAMC.NS",
    "360ONE.NS", "ANANDRATHI.NS", "PNBHOUSING.NS", "HOMEFIRST.NS", "AAVAS.NS",
    "APTUS.NS", "RECLTD.NS", "PFC.NS", "IREDA.NS", "SMCGLOBAL.NS", "CHOICEIN.NS",
    "KFINTECH.NS", "CAMSBANK.NS", "MASFIN.NS", "TRIDENT.NS", "SBFC.NS",
    "UGROCAP.NS", "FUSION.NS", "PAISALO.NS", "CAPITALSFB.NS", "NSIL.NS",
    "SATIN.NS", "CREDAGRI.NS"
  ],

  "Automobile & Ancillaries": [
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MRF.NS", "BALKRISIND.NS",
    "APOLLOTYRE.NS", "CEATLTD.NS", "JKTYRE.NS", "MOTHERSON.NS", "BHARATFORG.NS",
    "SUNDRMFAST.NS", "EXIDEIND.NS", "AMARAJABAT.NS", "BOSCHLTD.NS", "ENDURANCE.NS",
    "MINDAIND.NS", "WABCOINDIA.NS", "GABRIEL.NS", "SUPRAJIT.NS", "LUMAXTECH.NS",
    "FIEMIND.NS", "SUBROS.NS", "JAMNAAUTO.NS", "SHRIRAMCIT.NS", "ESCORTS.NS",
    "ATULAUTO.NS", "OLECTRA.NS", "GREAVESCOT.NS", "SMLISUZU.NS", "VSTTILLERS.NS",
    "HINDMOTORS.NS", "MAHSCOOTER.NS"
  ],

  "Healthcare": [
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "LUPIN.NS",
    "DIVISLAB.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "IPCALAB.NS", "GLENMARK.NS", "BIOCON.NS", "ABBOTINDIA.NS", "SANOFI.NS",
    "PFIZER.NS", "GLAXO.NS", "NATCOPHARM.NS", "AJANTPHARM.NS", "GRANULES.NS",
    "LAURUSLABS.NS", "STAR.NS", "JUBLPHARMA.NS", "ASTRAZEN.NS", "WOCKPHARDT.NS",
    "FORTIS.NS", "MAXHEALTH.NS", "METROPOLIS.NS", "THYROCARE.NS", "POLYMED.NS",
    "KIMS.NS", "NH.NS", "LALPATHLAB.NS", "MEDPLUS.NS", "ERIS.NS", "INDOCO.NS",
    "CAPLIPOINT.NS", "NEULANDLAB.NS", "SHILPAMED.NS", "SUVENPHAR.NS", "AARTIDRUGS.NS",
    "PGHL.NS", "SYNGENE.NS", "VINATIORGA.NS", "GLAND.NS", "JBCHEPHARM.NS",
    "HCG.NS", "RAINBOW.NS", "ASTERDM.NS", "KRSNAA.NS", "VIJAYA.NS", "MEDANTA.NS",
    "NETMEDS.NS", "BLISSGVS.NS", "MOREPENLAB.NS", "RPGLIFE.NS"
  ],

  "Metals & Mining": [
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS",
    "NMDC.NS", "HINDZINC.NS", "NALCO.NS", "JINDALSTEL.NS", "MOIL.NS",
    "APLAPOLLO.NS", "RATNAMANI.NS", "JSL.NS", "WELCORP.NS", "TINPLATE.NS",
    "SHYAMMETL.NS", "MIDHANI.NS", "GRAVITA.NS", "SARDAEN.NS", "ASHAPURMIN.NS",
    "JTLIND.NS", "RAMASTEEL.NS", "MAITHANALL.NS", "KIOCL.NS", "IMFA.NS",
    "GMDCLTD.NS", "VISHNU.NS", "SANDUMA.NS","VRAJ.NS","COALINDIA.NS ","NILE.BO"
  ],

  "FMCG": [
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "VARBEV.NS", "BRITANNIA.NS",
    "GODREJCP.NS", "DABUR.NS", "COLPAL.NS", "MARICO.NS", "PGHH.NS",
    "EMAMILTD.NS", "GILLETTE.NS", "HATSUN.NS", "JYOTHYLAB.NS", "BAJAJCON.NS",
    "RADICO.NS", "TATACONSUM.NS", "UNITDSPR.NS", "CCL.NS", "AVANTIFEED.NS",
    "BIKAJI.NS", "PATANJALI.NS", "VBL.NS", "ZOMATO.NS", "DOMS.NS",
    "GODREJAGRO.NS", "SAPPHIRE.NS", "VENKEYS.NS", "BECTORFOOD.NS", "KRBL.NS"
  ],

  "Power": [
    "NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "JSWENERGY.NS",
    "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS", "CESC.NS", "ADANIENSOL.NS",
    "INDIAGRID.NS", "POWERMECH.NS", "KEC.NS", "INOXWIND.NS", "KALPATPOWR.NS",
    "SUZLON.NS", "BHEL.NS", "THERMAX.NS", "GEPIL.NS", "VOLTAMP.NS",
    "TRIL.NS", "TDPOWERSYS.NS", "JYOTISTRUC.NS", "IWEL.NS"
  ],

  "Capital Goods": [
    "LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS", "BHEL.NS", "HAL.NS",
    "CUMMINSIND.NS", "THERMAX.NS", "AIAENG.NS", "SKFINDIA.NS", "GRINDWELL.NS",
    "TIMKEN.NS", "KSB.NS", "ELGIEQUIP.NS", "LAKSHMIMACH.NS", "KIRLOSENG.NS",
    "GREAVESCOT.NS", "TRITURBINE.NS", "VOLTAS.NS", "BLUESTARCO.NS", "HAVELLS.NS",
    "DIXON.NS", "KAYNES.NS", "SYRMA.NS", "AMBER.NS", "SUZLON.NS", "CGPOWER.NS",
    "APARINDS.NS", "HBLPOWER.NS", "KEI.NS", "POLYCAB.NS", "RRKABEL.NS",
    "SCHNEIDER.NS", "TDPOWERSYS.NS", "KIRLOSBROS.NS", "JYOTICNC.NS", "DATAPATTNS.NS",
    "INOXWIND.NS", "KALPATPOWR.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS",
    "POWERMECH.NS", "ISGEC.NS", "HPL.NS", "VTL.NS", "DYNAMATECH.NS", "JASH.NS",
    "GMMPFAUDLR.NS", "ESABINDIA.NS", "CENTURYEXT.NS", "SALASAR.NS", "TITAGARH.NS",
    "VGUARD.NS", "WABAG.NS","AZAD"
  ],

  "Oil & Gas": [
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HPCL.NS", "GAIL.NS",
    "PETRONET.NS", "OIL.NS", "IGL.NS", "MGL.NS", "GUJGASLTD.NS", "GSPL.NS",
    "AEGISCHEM.NS", "CHENNPETRO.NS", "MRPL.NS", "GULFOILLUB.NS", "CASTROLIND.NS",
    "SOTL.NS", "PANAMAPET.NS", "GOCLCORP.NS"
  ],

  "Chemicals": [
    "PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS", "ATUL.NS", "AARTIIND.NS",
    "NAVINFLUOR.NS", "VINATIORGA.NS", "FINEORG.NS", "ALKYLAMINE.NS", "BALAMINES.NS",
    "GUJFLUORO.NS", "CLEAN.NS", "JUBLINGREA.NS", "GALAXYSURF.NS", "PCBL.NS",
    "NOCIL.NS", "BASF.NS", "SUDARSCHEM.NS", "NEOGEN.NS", "PRIVISCL.NS",
    "ROSSARI.NS", "LXCHEM.NS", "ANURAS.NS", "JUBLPHARMA.NS", "CHEMCON.NS",
    "DMCC.NS", "TATACHEM.NS", "COROMANDEL.NS", "UPL.NS", "BAYERCROP.NS",
    "SUMICHEM.NS", "PIIND.NS", "DHARAMSI.NS", "EIDPARRY.NS", "CHEMPLASTS.NS",
    "VISHNU.NS", "IGPL.NS", "TIRUMALCHM.NS"
  ],

  "Telecom": [
    "BHARTIARTL.NS", "VODAFONEIDEA.NS", "INDUSTOWER.NS", "TATACOMM.NS",
    "HFCL.NS", "TEJASNET.NS", "STLTECH.NS", "ITI.NS", "ASTEC.NS"
  ],

  "Infrastructure": [
    "LT.NS", "GMRINFRA.NS", "IRB.NS", "NBCC.NS", "RVNL.NS", "KEC.NS",
    "PNCINFRA.NS", "KNRCON.NS", "GRINFRA.NS", "NCC.NS", "HGINFRA.NS",
    "ASHOKA.NS", "SADBHAV.NS", "JWL.NS", "PATELENG.NS", "KALPATPOWR.NS",
    "IRCON.NS", "ENGINERSIN.NS", "AHLUWALIA.NS", "PSPPROJECTS.NS", "CAPACITE.NS",
    "WELSPUNIND.NS", "TITAGARH.NS", "HCC.NS", "MANINFRA.NS", "RIIL.NS",
    "DBREALTY.NS", "JWL.NS"
  ],

  "Insurance": [
    "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS",
    "GICRE.NS", "NIACL.NS", "STARHEALTH.NS", "BAJAJFINSV.NS", "MAXFIN.NS"
  ],

  "Diversified": [
    "ITC.NS", "RELIANCE.NS", "ADANIENT.NS", "GRASIM.NS", "HINDUNILVR.NS",
    "DCMSHRIRAM.NS", "3MINDIA.NS", "CENTURYPLY.NS", "KFINTECH.NS", "BALMERLAWRI.NS",
    "GODREJIND.NS", "VBL.NS", "BIRLACORPN.NS"
  ],

  "Construction Materials": [
    "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS",
    "DALBHARAT.NS", "RAMCOCEM.NS", "NUVOCO.NS", "JKLAKSHMI.NS", "BIRLACORPN.NS",
    "HEIDELBERG.NS", "INDIACEM.NS", "PRISMJOHNS.NS", "STARCEMENT.NS", "SAGCEM.NS",
    "DECCANCE.NS", "KCP.NS", "ORIENTCEM.NS", "HIL.NS", "EVERESTIND.NS",
    "VISAKAIND.NS", "BIGBLOC.NS"
  ],

  "Real Estate": [
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
    "BRIGADE.NS", "SOBHA.NS", "SUNTECK.NS", "MAHLIFE.NS", "ANANTRAJ.NS",
    "KOLTEPATIL.NS", "PURVA.NS", "ARVSMART.NS", "RUSTOMJEE.NS", "DBREALTY.NS",
    "IBREALEST.NS", "OMAXE.NS", "ASHIANA.NS", "ELDEHSG.NS", "TARC.NS"
  ],

  "Aviation": [
    "INDIGO.NS", "SPICEJET.NS", "AAI.NS", "GMRINFRA.NS"
  ],

  "Retailing": [
    "DMART.NS", "TRENT.NS", "ABFRL.NS", "VMART.NS", "SHOPERSTOP.NS",
    "BATAINDIA.NS", "METROBRAND.NS", "ARVINDFASN.NS", "CANTABIL.NS", "ZOMATO.NS",
    "NYKAA.NS", "MANYAVAR.NS", "ELECTRONICSMRKT.NS", "LANDMARK.NS", "V2RETAIL.NS",
    "THANGAMAYL.NS", "KALYANKJIL.NS", "TITAN.NS"
  ],

  "Miscellaneous": [
    "PIDILITIND.NS", "BSE.NS", "CDSL.NS", "MCX.NS", "NAUKRI.NS",
    "JUSTDIAL.NS", "TEAMLEASE.NS", "QUESS.NS", "SIS.NS", "DELHIVERY.NS",
    "PRUDENT.NS", "MEDIASSIST.NS", "AWFIS.NS", "JUBLFOOD.NS", "DEVYANI.NS",
    "WESTLIFE.NS", "SAPPHIRE.NS", "BARBEQUE.NS", "EASEMYTRIP.NS", "THOMASCOOK.NS",
    "MSTC.NS", "IRCTC.NS", "POLICYBZR.NS", "PAYTM.NS", "INFIBEAM.NS",
    "CARTRADE.NS", "HONASA.NS", "ONE97COMM.NS", "SIGNATURE.NS", "RRKABEL.NS",
    "HMAAGRO.NS", "RKFORGE.NS", "CAMPUS.NS", "SENCO.NS", "CONCORDBIO.NS"
  ]
}

def init_smartapi_client():
    try:
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        if data['status']:
            return smart_api
        else:
            st.error(f"⚠️ SmartAPI authentication failed: {data['message']}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error initializing SmartAPI: {str(e)}")
        return None

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

def retry(max_retries=5, delay=5, backoff_factor=2, jitter=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        st.warning(f"Rate limit hit. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@retry(max_retries=5, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}-EQ" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

@retry(max_retries=5, delay=5)
def fetch_stock_data_with_auth(symbol, period="2y", interval="1d"):
    cache_key = f"{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return pd.read_pickle(io.BytesIO(cached_data))

    try:
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"

        smart_api = init_smartapi_client()
        if not smart_api:
            raise ValueError("SmartAPI client initialization failed")

        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=2 * 365)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)

        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")

        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        if not symboltoken:
            st.warning(f"⚠️ Token not found for symbol: {symbol}")
            return pd.DataFrame()

        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symboltoken,
            "interval": api_interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })

        if historical_data['status'] and historical_data['data']:
            data = pd.DataFrame(historical_data['data'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=86400)
            return data
        else:
            raise ValueError(f"No data found for {symbol}: {historical_data.get('message', 'Unknown error')}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning(f"⚠️ Rate limit exceeded for {symbol}. Skipping...")
            return pd.DataFrame()
        raise e
    except Exception as e:
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol, period="2y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    return advances / declines if declines != 0 else 0

def monte_carlo_simulation(data, simulations=1000, days=30, garch_min_obs=80, winsorize_limit=0.01):
    # === Validation ===
    if data is None or not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
        raise ValueError("Input 'data' must be a DataFrame with a 'Close' column.")
    
    close_prices = pd.to_numeric(data['Close'], errors='coerce').dropna()
    if len(close_prices) < 2:
        raise ValueError("Not enough valid 'Close' prices for simulation.")

    returns = close_prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Insufficient return data after computing percentage change.")

    last_price = close_prices.iloc[-1]
    
    # === Optional: Winsorize to reduce effect of outliers ===
    if winsorize_limit > 0:
        returns = pd.Series(winsorize(returns, limits=winsorize_limit))

    # === Use geometric mean return for better forward-looking estimate ===
    log_returns = np.log1p(returns)
    geo_mean = np.expm1(log_returns.mean())  # annualized drift is another option

    std_return = returns.std()

    # === Choose method based on data length ===
    if len(returns) < garch_min_obs:
        # === Simple vectorized Monte Carlo ===
        rand_returns = np.random.normal(geo_mean, std_return, (simulations, days))
        price_paths = last_price * np.cumprod(1 + rand_returns, axis=1)
        price_paths = np.hstack([np.full((simulations, 1), last_price), price_paths])
        return price_paths.tolist()
    
    else:
        # === GARCH Simulation with volatility forecasting ===
        try:
            model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
            garch_fit = model.fit(disp='off')
            forecasts = garch_fit.forecast(horizon=days)
            volatility = np.sqrt(forecasts.variance.iloc[-1].values)

            sim_results = []
            for _ in range(simulations):
                prices = [last_price]
                for i in range(days):
                    shock = np.random.normal(geo_mean, volatility[i])
                    prices.append(prices[-1] * (1 + shock))
                sim_results.append(prices)
            return sim_results

        except Exception as e:
            print(f"GARCH model fitting failed: {e}. Falling back to simple simulation.")
            rand_returns = np.random.normal(geo_mean, std_return, (simulations, days))
            price_paths = last_price * np.cumprod(1 + rand_returns, axis=1)
            price_paths = np.hstack([np.full((simulations, 1), last_price), price_paths])
            return price_paths.tolist()

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def calculate_confidence_score(data):
    # Safety check for empty or None data
    if data is None or len(data) == 0:
        return 0

    score = 0
    max_score = 0

    # --- RSI Oversold Check ---
    if 'RSI' in data.columns:
        latest_rsi = data['RSI'].iloc[-1]
        if not pd.isna(latest_rsi):
            max_score += 1
            if latest_rsi < 30:
                score += 1

    # --- MACD Bullish Crossover Check ---
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        macd = data['MACD'].iloc[-1]
        signal = data['MACD_signal'].iloc[-1]
        if not pd.isna(macd) and not pd.isna(signal):
            max_score += 1
            if macd > signal:
                score += 1

    # --- Ichimoku Span A Check ---
    if 'Ichimoku_Span_A' in data.columns and 'Close' in data.columns:
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        close = data['Close'].iloc[-1]
        if not pd.isna(span_a) and not pd.isna(close):
            max_score += 1
            if close > span_a:
                score += 1

    # --- ATR-based Volatility Check (High Volatility = Good) ---
    if 'ATR' in data.columns and 'Close' in data.columns:
        atr = data['ATR'].iloc[-1]
        close = data['Close'].iloc[-1]
        if not pd.isna(atr) and not pd.isna(close) and close != 0:
            max_score += 0.5
            atr_volatility = atr / close

            if atr_volatility < 0.02:
                score += 0.25  # Very low volatility — minor positive
            elif atr_volatility > 0.05:
                score += 0.5   # High volatility — positive for momentum/breakout

    # Final normalized score between 0 and 1
    if max_score == 0:
        return 0

    return float(np.clip(score / max_score, 0, 1))

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    else:
        return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 15), risk_free_rate=0.025):
    best_window, best_sharpe = 9, -float('inf')

    if data is None or data.empty or 'Close' not in data:
        return best_window

    if len(data) < max(windows) + 20:
        return best_window

    returns = data['Close'].pct_change()

    for window in windows:
        rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()

        # Signal: Buy if RSI < 30, Sell if RSI > 70
        signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)

        # Vectorized position logic: hold previous signal
        positions = signals.replace(0, np.nan).ffill().fillna(0).shift(1)

        # Align positions and returns safely
        aligned_index = returns.index.intersection(positions.index)
        returns_aligned = returns.loc[aligned_index]
        positions_aligned = positions.loc[aligned_index]

        strategy_returns = returns_aligned * positions_aligned
        strategy_returns = strategy_returns.dropna()

        if strategy_returns.std() != 0:
            sharpe = ((strategy_returns.mean() - risk_free_rate / 252) / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        if sharpe > best_sharpe:
            best_sharpe, best_window = sharpe, window

    return best_window
    
def detect_divergence(data, window=10, rsi_threshold=5):
    price = data['Close']
    rsi = data['RSI']
    
    # Ensure enough data
    if len(price) < window or len(rsi) < window:
        return "Insufficient data"

    # Get recent extreme indices
    recent_price_low_idx = price[-window:].idxmin()
    recent_price_high_idx = price[-window:].idxmax()
    recent_rsi_low_idx = rsi[-window:].idxmin()
    recent_rsi_high_idx = rsi[-window:].idxmax()

    # Convert to positional indices
    price_low_pos = price.index.get_loc(recent_price_low_idx)
    price_high_pos = price.index.get_loc(recent_price_high_idx)
    rsi_low_pos = rsi.index.get_loc(recent_rsi_low_idx)
    rsi_high_pos = rsi.index.get_loc(recent_rsi_high_idx)
    latest_pos = len(price) - 1

    # Bullish divergence conditions
    bullish_div = (
        price[recent_price_low_idx] < price.iloc[-1] and                         # Price lower low
        rsi[recent_rsi_low_idx] > rsi.iloc[-1] and                               # RSI higher low
        abs(rsi[recent_rsi_low_idx] - rsi.iloc[-1]) > rsi_threshold and          # RSI change is significant
        price_low_pos > rsi_low_pos and price_low_pos != latest_pos             # RSI low occurs before price low and not current bar
    )

    # Bearish divergence conditions
    bearish_div = (
        price[recent_price_high_idx] > price.iloc[-1] and                        # Price higher high
        rsi[recent_rsi_high_idx] < rsi.iloc[-1] and                              # RSI lower high
        abs(rsi[recent_rsi_high_idx] - rsi.iloc[-1]) > rsi_threshold and         # RSI change is significant
        price_high_pos > rsi_high_pos and price_high_pos != latest_pos          # RSI high occurs before price high and not current bar
    )

    if bullish_div:
        return "Bullish Divergence"
    elif bearish_div:
        return "Bearish Divergence"
    else:
        return "No Divergence"

def calculate_cmo(close, window=14):
    try:
        diff = close.diff()
        up_sum = diff.where(diff > 0, 0).rolling(window=window).sum()
        down_sum = abs(diff.where(diff < 0, 0)).rolling(window=window).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo
    except Exception as e:
        st.warning(f"⚠️ Failed to compute custom CMO: {str(e)}")
        return None

INDICATOR_MIN_LENGTHS = {
    'RSI': 14,
    'MACD': 26,
    'SMA_50': 50,
    'SMA_200': 200,
    'EMA_20': 20,
    'EMA_50': 50,
    'Bollinger': 20,
    'Stochastic': 14,
    'ATR': 14,
    'ADX': 27,
    'OBV': 1,
    'VWAP': 1,
    'Volume_Spike': 10,
    'Parabolic_SAR': 2,
    'Fibonacci': 1,
    'Divergence': 5,
    'Ichimoku': 52,
    'CMF': 20,
    'Donchian': 20,
    'Keltner': 20,
    'TRIX': 15,
    'Ultimate_Osc': 28,
    'CMO': 14,
    'VPT': 1
}

def can_compute_indicator(data, indicator):
    """
    Checks if sufficient data is available for a specific indicator.
    Returns True if computation is possible, False otherwise.
    """
    required_length = INDICATOR_MIN_LENGTHS.get(indicator, 1)
    return len(data) >= required_length

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s: %(message)s")

def validate_data(
    data: pd.DataFrame,
    required_columns=None,
    min_length: int = 50,
    max_volume: float | None = 1e10,
    check_positive_prices: bool = True,
) -> bool:
    """
    Comprehensive OHLCV DataFrame validator.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock price data (must include at least Open/High/Low/Close/Volume columns).
    required_columns : list[str] | None
        Columns that must be present. Defaults to the standard OHLCV set.
    min_length : int
        Minimum number of rows required for the DataFrame.
    max_volume : float | None
        Flag rows with unrealistically large volume figures. Set to None to skip.
    check_positive_prices : bool
        If True, verifies that all price columns are > 0.

    Returns
    -------
    bool
        True if all checks pass; otherwise False (with warnings logged).
    """
    # Default required columns
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 1 — basic integrity
    if data is None or data.empty:
        logging.warning("No data provided for validation.")
        return False
    if len(data) < min_length:
        logging.warning("Insufficient data length: %d rows (minimum %d required).",
                        len(data), min_length)
        return False

    # 2 — schema
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        logging.warning("Missing required columns: %s", ", ".join(missing))
        return False

    # 3 — nulls
    if data[required_columns].isnull().any().any():
        logging.warning("Data contains null values in required columns.")
        return False

    # 4 — positive prices
    price_cols = [c for c in ('Open', 'High', 'Low', 'Close') if c in data.columns]
    if check_positive_prices and (data[price_cols] <= 0).any().any():
        logging.warning("Invalid price values (≤ 0 detected).")
        return False

    # 5 — volume sanity
    if max_volume is not None and 'Volume' in data.columns \
       and data['Volume'].max() > max_volume:
        logging.warning("Abnormal volume values detected (max %.0f > %.0f).",
                        data['Volume'].max(), max_volume)
        return False

    return True

def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    """
    if not validate_data(data, min_length=50):
        columns = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
            'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
            'VWAP', 'Avg_Volume', 'Volume_Spike', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2',
            'Fib_50.0', 'Fib_61.8', 'Divergence', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
            'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF', 'Donchian_Upper',
            'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle', 'Keltner_Lower',
            'TRIX', 'Ultimate_Osc', 'CMO', 'VPT'
        ]
        for col in columns:
            data[col] = None
        return data

    try:
        if can_compute_indicator(data, 'RSI'):
            rsi_window = optimize_rsi_window(data)
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        else:
            data['RSI'] = None
    except Exception as e:
        logging.warning(f"Failed to compute RSI: {str(e)}")
        data['RSI'] = None

    try:
        if can_compute_indicator(data, 'MACD'):
            macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()
        else:
            data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None
    except Exception as e:
        logging.warning(f"Failed to compute MACD: {str(e)}")
        data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None

    try:
        if can_compute_indicator(data, 'SMA_50'):
            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        else:
            data['SMA_50'] = None
        if can_compute_indicator(data, 'SMA_200'):
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        else:
            data['SMA_200'] = None
        if can_compute_indicator(data, 'EMA_20'):
            data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        else:
            data['EMA_20'] = None
        if can_compute_indicator(data, 'EMA_50'):
            data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        else:
            data['EMA_50'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = data['SMA_200'] = data['EMA_20'] = data['EMA_50'] = None

    try:
        if can_compute_indicator(data, 'Bollinger'):
            bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
            data['Upper_Band'] = bollinger.bollinger_hband()
            data['Middle_Band'] = bollinger.bollinger_mavg()
            data['Lower_Band'] = bollinger.bollinger_lband()
        else:
            data['Upper_Band'] = data['Middle_Band'] = data['Lower_Band'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = data['Middle_Band'] = data['Lower_Band'] = None

    try:
        if can_compute_indicator(data, 'Stochastic'):
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()
        else:
            data['SlowK'] = data['SlowD'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = data['SlowD'] = None

    try:
        if can_compute_indicator(data, 'ATR'):
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        else:
            data['ATR'] = None
    except Exception as e:
        logging.warning(f"Failed to compute ATR: {str(e)}")
        data['ATR'] = None

    try:
        if can_compute_indicator(data, 'ADX'):
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        logging.warning(f"Failed to compute ADX: {str(e)}")
        data['ADX'] = None

    try:
        if can_compute_indicator(data, 'OBV'):
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        else:
            data['OBV'] = None
    except Exception as e:
        logging.warning(f"Failed to compute OBV: {str(e)}")
        data['OBV'] = None

    try:
        if can_compute_indicator(data, 'VWAP'):
            data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
            data['Cumulative_Volume'] = data['Volume'].cumsum()
            data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        else:
            data['VWAP'] = None
    except Exception as e:
        logging.warning(f"Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None

    try:
        if can_compute_indicator(data, 'Volume_Spike'):
            data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
            data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
        else:
            data['Avg_Volume'] = data['Volume_Spike'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Volume Spike: {str(e)}")
        data['Avg_Volume'] = data['Volume_Spike'] = None

    try:
        if can_compute_indicator(data, 'Parabolic_SAR'):
            data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        else:
            data['Parabolic_SAR'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None

    try:
        if can_compute_indicator(data, 'Fibonacci'):
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            data['Fib_23.6'] = high - diff * 0.236
            data['Fib_38.2'] = high - diff * 0.382
            data['Fib_50.0'] = high - diff * 0.5
            data['Fib_61.8'] = high - diff * 0.618
        else:
            data['Fib_23.6'] = data['Fib_38.2'] = data['Fib_50.0'] = data['Fib_61.8'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = data['Fib_38.2'] = data['Fib_50.0'] = data['Fib_61.8'] = None

    try:
        if can_compute_indicator(data, 'Divergence'):
            data['Divergence'] = detect_divergence(data)
        else:
            data['Divergence'] = "No Divergence"
    except Exception as e:
        logging.warning(f"Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"

    try:
        if can_compute_indicator(data, 'Ichimoku'):
            ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Chikou'] = data['Close'].shift(-26)
        else:
            data['Ichimoku_Tenkan'] = data['Ichimoku_Kijun'] = data['Ichimoku_Span_A'] = data['Ichimoku_Span_B'] = data['Ichimoku_Chikou'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = data['Ichimoku_Kijun'] = data['Ichimoku_Span_A'] = data['Ichimoku_Span_B'] = data['Ichimoku_Chikou'] = None

    try:
        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        else:
            data['CMF'] = None
    except Exception as e:
        logging.warning(f"Failed to compute CMF: {str(e)}")
        data['CMF'] = None

    try:
        if can_compute_indicator(data, 'Donchian'):
            donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
            data['Donchian_Upper'] = donchian.donchian_channel_hband()
            data['Donchian_Lower'] = donchian.donchian_channel_lband()
            data['Donchian_Middle'] = donchian.donchian_channel_mband()
        else:
            data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None

    try:
        if can_compute_indicator(data, 'Keltner'):
            keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
            data['Keltner_Upper'] = keltner.keltner_channel_hband()
            data['Keltner_Middle'] = keltner.keltner_channel_mband()
            data['Keltner_Lower'] = keltner.keltner_channel_lband()
        else:
            data['Keltner_Upper'] = data['Keltner_Middle'] = data['Keltner_Lower'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = data['Keltner_Middle'] = data['Keltner_Lower'] = None

    try:
        if can_compute_indicator(data, 'TRIX'):
            data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
        else:
            data['TRIX'] = None
    except Exception as e:
        logging.warning(f"Failed to compute TRIX: {str(e)}")
        data['TRIX'] = None

    try:
        if can_compute_indicator(data, 'Ultimate_Osc'):
            data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
                data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
            ).ultimate_oscillator()
        else:
            data['Ultimate_Osc'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = None

    try:
        if can_compute_indicator(data, 'CMO'):
            data['CMO'] = calculate_cmo(data['Close'], window=14)
        else:
            data['CMO'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = None

    try:
        if can_compute_indicator(data, 'VPT'):
            data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
        else:
            data['VPT'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = None

    return data

# === Shared Utility Functions ===

def get_atr_multiplier(adx):
    """
    Determines ATR multiplier based on ADX strength.
    ADX > 25 → Strong trend → Higher stop loss buffer.
    """
    return 3.0 if pd.notnull(adx) and adx > 25 else 1.5

def get_adjusted_rr_ratio(adx, base_ratio=3):
    """
    Adjusts risk-reward ratio based on ADX.
    Higher ADX (trend strength) allows more aggressive targets.
    """
    return min(base_ratio, 5) if pd.notnull(adx) and adx > 25 else min(base_ratio, 3)

def cap_stop_loss(stop_loss, close):
    """
    Ensures stop loss is not too close (no more than 10% below entry).
    """
    return max(stop_loss, close * 0.9)

def cap_target(target, close):
    """
    Limits target to 20% above entry to avoid unrealistic expectations.
    """
    return min(target, close * 1.2)

# === DataFrame-Level Calculations ===

def calculate_buy_at(data):
    """
    Calculates the buy price.
    - If RSI < 30: slight discount (oversold).
    - Else: use the current close.
    """
    if data.empty or 'RSI' not in data.columns or 'Close' not in data.columns:
        st.warning("⚠️ Missing RSI or Close data.")
        return None
    if np.isnan(data['RSI'].iloc[-1]) or np.isnan(data['Close'].iloc[-1]):
        st.warning("⚠️ Latest RSI or Close value is NaN.")
        return None

    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data):
    """
    Calculates stop loss using ATR and trend strength (ADX).
    - Strong trend → wider stop loss (to avoid noise).
    - Ensures SL is not below 10% of close.
    """
    if data.empty or 'ATR' not in data.columns or 'ADX' not in data.columns:
        st.warning("⚠️ Missing ATR or ADX data.")
        return None
    if np.isnan(data['ATR'].iloc[-1]) or np.isnan(data['Close'].iloc[-1]):
        st.warning("⚠️ Invalid ATR or Close values.")
        return None

    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    last_adx = data['ADX'].iloc[-1]
    atr_multiplier = get_atr_multiplier(last_adx)

    stop_loss = last_close - (atr_multiplier * last_atr)
    stop_loss = cap_stop_loss(stop_loss, last_close)
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    """
    Calculates the profit target.
    - Based on risk (Close - Stop Loss).
    - Adjusts target based on ADX strength.
    - Capped at 20% above current price.
    """
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss.")
        return None

    last_close = data['Close'].iloc[-1]
    last_adx = data['ADX'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = get_adjusted_rr_ratio(last_adx, risk_reward_ratio)
    target = last_close + (risk * adjusted_ratio)
    target = cap_target(target, last_close)
    return round(target, 2)

# === Row-Level Calculations ===

def calculate_buy_at_row(row):
    """
    Row-level buy price logic.
    Returns buy price based on RSI < 30 condition.
    """
    if pd.isnull(row.get('RSI')) or pd.isnull(row.get('Close')):
        return None
    return round(row['Close'] * 0.99, 2) if row['RSI'] < 30 else round(row['Close'], 2)

def calculate_stop_loss_row(row):
    """
    Row-level stop loss calculation using ATR and ADX.
    Handles missing or invalid data by returning None.
    """
    if pd.isnull(row.get('ATR')) or pd.isnull(row.get('Close')):
        return None
    atr_multiplier = get_atr_multiplier(row.get('ADX'))
    stop_loss = row['Close'] - (atr_multiplier * row['ATR'])
    stop_loss = cap_stop_loss(stop_loss, row['Close'])
    return round(stop_loss, 2)

def calculate_target_row(row, risk_reward_ratio=3):
    """
    Row-level target calculation based on stop loss and trend strength.
    Risk is calculated as Close - SL, and reward is adjusted by ADX.
    """
    stop_loss = calculate_stop_loss_row(row)
    if stop_loss is None:
        return None
    risk = row['Close'] - stop_loss
    adjusted_ratio = get_adjusted_rr_ratio(row.get('ADX'), risk_reward_ratio)
    target = row['Close'] + (risk * adjusted_ratio)
    target = cap_target(target, row['Close'])
    return round(target, 2)


# Improved strategy logic using adaptive regime detection, signal scoring, and volatility-aware filters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data(data, min_length=50):
    required_cols = {'Close', 'ATR', 'Volume', 'Avg_Volume', 'SMA_50'}
    return (
        isinstance(data, pd.DataFrame) and
        len(data) >= min_length and
        required_cols.issubset(data.columns) and
        data['Close'].notna().sum() >= min_length
    )

def classify_market_regime(data):
    data['ATR_pct'] = data['ATR'] / data['Close']
    atr_pct = data['ATR_pct'].iloc[-1]
    close = data['Close'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]

    if atr_pct > 0.03:
        return 'High Volatility'
    elif close > sma_50:
        return 'Bullish'
    elif close < sma_50:
        return 'Bearish'
    else:
        return 'Neutral'

def compute_signal_score(data, regime, symbol=None):
    score = 0.0
    weights = {
        'RSI': 1.5,
        'MACD': 1.2,
        'Ichimoku': 1.5,
        'CMF': 0.5,
        'ATR_Volatility': 1.0,
        'Breakout': 1.2,
    }

    if data['Volume'].iloc[-1] < data['Avg_Volume'].iloc[-1] * 0.5:
        return -10

    rsi = data.get('RSI', pd.Series([None])).iloc[-1]
    if pd.notnull(rsi):
        rsi_normalized = (rsi - 50) / 50
        if rsi < 30:
            score += weights['RSI'] * max(rsi_normalized, -1)
        elif rsi > 70:
            score -= weights['RSI'] * min(rsi_normalized, 1)
        if rsi > 50 and regime == "Bullish":
            score += weights['RSI'] * 0.5

    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        macd = data['MACD'].iloc[-1]
        signal = data['MACD_signal'].iloc[-1]
        if pd.notnull(macd) and pd.notnull(signal):
            diff = macd - signal
            macd_std = data['MACD'].rolling(20).std().iloc[-1] + 1e-10
            normalized = diff / macd_std
            score += weights['MACD'] * (normalized if diff > 0 else -abs(normalized))

    if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        span_b = data['Ichimoku_Span_B'].iloc[-1]
        close = data['Close'].iloc[-1]
        if pd.notnull(span_a) and pd.notnull(span_b):
            if close > max(span_a, span_b):
                score += weights['Ichimoku']
            elif close < min(span_a, span_b):
                score -= weights['Ichimoku']

    if 'CMF' in data.columns:
        cmf = data['CMF'].iloc[-1]
        if pd.notnull(cmf):
            score += weights['CMF'] * cmf

    if 'ATR' in data.columns:
        atr_pct = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_pct > 0.04:
            score -= weights['ATR_Volatility'] * (atr_pct / 0.04)

    if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
        close = data['Close'].iloc[-1]
        upper = data['Donchian_Upper'].iloc[-1]
        lower = data['Donchian_Lower'].iloc[-1]
        if pd.notnull(upper) and pd.notnull(lower):
            if close > upper:
                score += weights['Breakout']
            elif close < lower:
                score -= weights['Breakout']

    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        if pd.notnull(sma_20) and pd.notnull(sma_50):
            trend_strength = (sma_20 - sma_50) / sma_50
            if abs(trend_strength) < 0.02:
                score *= 0.8

    return min(max(score, -10), 10)

def adaptive_recommendation(data, symbol=None, account_size=100000, max_position_size=100):
    try:
        if not validate_data(data):
            return {**default_output(), "Reason": "Insufficient or invalid data"}

        current_price = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        volume = data['Volume'].iloc[-1]

        regime = classify_market_regime(data)
        score = compute_signal_score(data, regime, symbol)

        # Robust regime transition detection
        regime_stability = 1.0
        if len(data) >= 5:
            recent_regimes = [
                classify_market_regime(data.iloc[i-1:i+1])
                for i in range(len(data) - 4, len(data))
            ]
            regime_stability = len(set(recent_regimes)) / len(recent_regimes)
            if regime_stability > 0.5:
                score *= 0.7

        # Regime-specific confidence threshold
        confidence_threshold = {
            'High Volatility': 1.5,
            'Bullish': 0.8,
            'Bearish': 1.2,
            'Neutral': 1.0
        }.get(regime, 1.0)

        if current_price < 100 or atr < 5 or volume < 5000:
            return {
                **default_output(current_price, score, regime),
                "Reason": "Filtered: low price/ATR/volume",
                "Confidence": confidence_threshold,
                "Regime_Stability": round(regime_stability, 2)
            }

        if score > confidence_threshold:
            recommendation = "Buy"
        elif score < -confidence_threshold:
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        slippage = 0.005
        buy_at = current_price * (1 + slippage) if recommendation == "Buy" else None
        stop_loss = (
            current_price * 0.95 if recommendation == "Buy"
            else current_price * 1.05 if recommendation == "Sell"
            else None
        )
        target = (
            current_price * 1.05 if recommendation == "Buy"
            else current_price * 0.95 if recommendation == "Sell"
            else None
        )

        # Maximum stop loss check (e.g., 8%)
        max_loss_pct = 0.08
        if stop_loss:
            loss_pct = abs(current_price - stop_loss) / current_price
            if loss_pct > max_loss_pct:
                return {
                    **default_output(current_price, score, regime),
                    "Reason": "Stop loss too wide (>8%)",
                    "Confidence": confidence_threshold,
                    "Regime_Stability": round(regime_stability, 2)
                }

        # Volatility-based position sizing
        if stop_loss:
            stop_dist = abs(current_price - stop_loss) / current_price
            stop_dist = max(stop_dist, 0.01)
            position_size = min((account_size * 0.02) / (current_price * stop_dist), max_position_size)
        else:
            position_size = None

        # Dynamic trailing stop based on volatility
        atr_multiplier = 1.5 if regime == "High Volatility" else 2.0
        trailing_stop = (
            current_price - (atr_multiplier * atr) if recommendation == "Buy"
            else current_price + (atr_multiplier * atr) if recommendation == "Sell"
            else None
        )

        reason = f"{recommendation} signal (Score: {score:.2f}) in {regime} regime"

        return {
            "Current Price": current_price,
            "Buy At": buy_at,
            "Stop Loss": stop_loss,
            "Target": target,
            "Recommendation": recommendation,
            "Score": round(score, 2),
            "Regime": regime,
            "Position Size": round(position_size, 2) if position_size else None,
            "Trailing Stop": round(trailing_stop, 2) if trailing_stop else None,
            "Reason": reason,
            "Confidence": confidence_threshold,
            "Regime_Stability": round(regime_stability, 2)
        }

    except Exception as e:
        logging.error(f"Error in adaptive_recommendation: {e}")
        return {
            **default_output(),
            "Reason": f"Error: {str(e)}",
            "Confidence": None,
            "Regime_Stability": None
        }

def default_output(price=None, score=0, regime="Unknown"):
    return {
        "Current Price": price,
        "Buy At": None,
        "Stop Loss": None,
        "Target": None,
        "Recommendation": "Hold",
        "Score": score,
        "Regime": regime,
        "Position Size": None,
        "Trailing Stop": None,
        "Reason": "N/A",
        "Confidence": None,
        "Regime_Stability": None
    }

        
def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }

    if not validate_data(data, min_length=27):
        return recommendations

    if data.empty or len(data) < 27 or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        st.warning("⚠️ Insufficient data for recommendations.")
        return recommendations

    try:
        recommendations["Current Price"] = float(data['Close'].iloc[-1])
        buy_score = 0
        sell_score = 0

        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
            if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['RSI'].iloc[-1] <= 20:
                    buy_score += 4
                elif data['RSI'].iloc[-1] < 30:
                    buy_score += 2
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2

        if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None and len(data['MACD'].dropna()) >= 1:
            if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    buy_score += 1
                elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                    sell_score += 1

        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None and len(data['Lower_Band'].dropna()) >= 1:
            if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1

        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None and len(data['VWAP'].dropna()) >= 1:
            if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1

        if ('Volume' in data.columns and data['Volume'].iloc[-1] is not None and 
            'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None and len(data['Volume'].dropna()) >= 2):
            volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
            if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 2
                elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    sell_score += 2
                elif volume_ratio < 0.5:
                    sell_score += 1

        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None and len(data['Volume_Spike'].dropna()) >= 1:
            if data['Volume_Spike'].iloc[-1] and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 1
                else:
                    sell_score += 1

        if 'Divergence' in data.columns and data['Divergence'].iloc[-1] is not None:
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1

        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None and len(data['Ichimoku_Span_A'].dropna()) >= 1:
            if (isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Sell"

        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None and len(data['CMF'].dropna()) >= 1:
            if isinstance(data['CMF'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMF'].iloc[-1] > 0:
                    buy_score += 1
                elif data['CMF'].iloc[-1] < 0:
                    sell_score += 1

        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Close'].iloc[-1] is not None and len(data['Donchian_Upper'].dropna()) >= 1:
            if (isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Donchian_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"

        if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
            if (isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] >= data['Lower_Band'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"

        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and data['Close'].iloc[-1] is not None and len(data['Ichimoku_Tenkan'].dropna()) >= 1:
            if (isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating))):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"

        if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
            data['Close'].iloc[-1] is not None and len(data['Keltner_Upper'].dropna()) >= 1):
            if (isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 1

        if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None and len(data['TRIX'].dropna()) >= 2:
            if isinstance(data['TRIX'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['TRIX'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                    buy_score += 1
                elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                    sell_score += 1

        if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None and len(data['Ultimate_Osc'].dropna()) >= 1:
            if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Ultimate_Osc'].iloc[-1] < 30:
                    buy_score += 1
                elif data['Ultimate_Osc'].iloc[-1] > 70:
                    sell_score += 1

        if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None and len(data['CMO'].dropna()) >= 1:
            if isinstance(data['CMO'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMO'].iloc[-1] < -50:
                    buy_score += 1
                elif data['CMO'].iloc[-1] > 50:
                    sell_score += 1

        if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None and len(data['VPT'].dropna()) >= 2:
            if isinstance(data['VPT'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['VPT'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                    buy_score += 1
                elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                    sell_score += 1

        if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
            data['Close'].iloc[-1] is not None and len(data['Fib_23.6'].dropna()) >= 1):
            current_price = data['Close'].iloc[-1]
            fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                          data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
            for level in fib_levels:
                if isinstance(level, (int, float, np.integer, np.floating)) and abs(current_price - level) / current_price < 0.01:
                    if current_price > level:
                        buy_score += 1
                    else:
                        sell_score += 1

        if ('Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and 
            data['Close'].iloc[-1] is not None and len(data['Parabolic_SAR'].dropna()) >= 1):
            if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                    sell_score += 1

        if ('OBV' in data.columns and data['OBV'].iloc[-1] is not None and 
            data['OBV'].iloc[-2] is not None and len(data['OBV'].dropna()) >= 2):
            if isinstance(data['OBV'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['OBV'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['OBV'].iloc[-1] > data['OBV'].iloc[-2]:
                    buy_score += 1
                elif data['OBV'].iloc[-1] < data['OBV'].iloc[-2]:
                    sell_score += 1

        net_score = buy_score - sell_score
        if buy_score > sell_score and buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
            recommendations["Swing"] = "Buy" if buy_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Buy" if buy_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Buy" if buy_score >= 1 else "Hold"
        elif sell_score > buy_score and sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
            recommendations["Swing"] = "Sell" if sell_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Sell" if sell_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Sell" if sell_score >= 1 else "Hold"
        elif net_score > 0:
            recommendations["Intraday"] = "Buy" if net_score >= 3 else "Hold"
            recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
            recommendations["Short-Term"] = "Buy" if net_score >= 1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        elif net_score < 0:
            recommendations["Intraday"] = "Sell" if net_score <= -3 else "Hold"
            recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
            recommendations["Short-Term"] = "Sell" if net_score <= -1 else "Hold"
            recommendations["Long-Term"] = "Hold"

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)

        recommendations["Score"] = min(max(buy_score - sell_score, -7), 7)
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

@st.cache_data(ttl=3600)  # Cache results for 1 hour to avoid repeated API hits
def get_top_sectors_cached(rate_limit_delay=2, stocks_per_sector=2):
    sector_scores = {}
    for sector, stocks in SECTORS.items():
        total_score = 0
        count = 0
        for symbol in stocks[:stocks_per_sector]:  # Only analyze top N stocks per sector
            data = fetch_stock_data_cached(symbol)
            if data.empty:
                continue
            data = analyze_stock(data)
            rec = generate_recommendations(data, symbol)
            total_score += rec.get("Score", 0)
            count += 1
            time.sleep(rate_limit_delay)  # Delay per API call
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
        time.sleep(1)  # Optional: delay between sectors
    return sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]

@st.cache_data
@st.cache_data(ttl=3600)
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None):
    results = {
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "trades": 0,
        "win_rate": 0,
        "buy_signals": [],
        "sell_signals": [],
        "trade_details": []
    }
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    
    position = None
    entry_price = 0
    entry_date = None
    trades = []
    returns = []
    
    for i in range(1, len(data)):
        sliced_data = data.iloc[:i+1]
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data)
            signal = rec["Recommendation"]
        else:
            rec = generate_recommendations(sliced_data, symbol)
            signal = rec[strategy] if strategy in rec else "Hold"
        
        current_price = data['Close'].iloc[i]
        current_date = data.index[i]
        
        if signal == "Buy" and position is None:
            position = "Long"
            entry_price = current_price
            entry_date = current_date
            results["buy_signals"].append((current_date, current_price))
        
        elif signal == "Sell" and position == "Long":
            position = None
            profit = current_price - entry_price
            returns.append(profit / entry_price)
            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": current_date,
                "exit_price": current_price,
                "profit": profit
            })
            results["sell_signals"].append((current_date, current_price))
            entry_price = 0
            entry_date = None
    
    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)
        results["total_return"] = sum([t["profit"]/t["entry_price"] for t in trades]) * 100
        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100
        if returns:
            results["annual_return"] = (np.mean(returns) * 252) * 100
            results["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        drawdowns = [t["profit"]/t["entry_price"] for t in trades]
        results["max_drawdown"] = min(drawdowns, default=0) * 100 if drawdowns else 0
    
    return results
    
def init_database():
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_picks (
            date TEXT,
            symbol TEXT,
            score REAL,
            current_price REAL,
            buy_at REAL,
            stop_loss REAL,
            target REAL,
            intraday TEXT,
            swing TEXT,
            short_term TEXT,
            long_term TEXT,
            mean_reversion TEXT,
            breakout TEXT,
            ichimoku_trend TEXT,
            recommendation TEXT,
            regime TEXT,
            position_size REAL,
            trailing_stop REAL,
            reason TEXT,
            pick_type TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.close()

def insert_top_picks(results_df, pick_type="daily"):
    conn = sqlite3.connect('stock_picks.db')
    for _, row in results_df.head(5).iterrows():
        conn.execute('''
            INSERT OR IGNORE INTO daily_picks (
                date, symbol, score, current_price, buy_at, stop_loss, target,
                intraday, swing, short_term, long_term, mean_reversion, breakout,
                ichimoku_trend, recommendation, regime, position_size, trailing_stop,
                reason, pick_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            row.get('Symbol'),
            row.get('Score', 0),
            row.get('Current Price'),
            row.get('Buy At'),
            row.get('Stop Loss'),
            row.get('Target'),
            row.get('Intraday'),
            row.get('Swing'),
            row.get('Short-Term'),
            row.get('Long-Term'),
            row.get('Mean_Reversion'),
            row.get('Breakout'),
            row.get('Ichimoku_Trend'),
            row.get('Recommendation'),
            row.get('Regime'),
            row.get('Position Size'),
            row.get('Trailing Stop'),
            row.get('Reason'),
            pick_type
        ))
    conn.commit()
    conn.close()

def analyze_batch(stock_batch):
    """
    Analyzes a batch of stocks in parallel, aggregating errors for summary reporting.
    Returns a list of valid results.
    """
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg)

    if errors:
        logging.error(f"Batch errors: {len(errors)} total\n" + "\n".join(errors))
        # Display summary warning in main thread
        st.session_state['batch_errors'] = f"Encountered {len(errors)} errors during batch processing. Check logs for details."

    return results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def analyze_stock_parallel(symbol):
    """
    Analyzes a single stock, logging detailed context on errors.
    Returns a dictionary with analysis results or None on failure.
    """
    try:
        logging.info(f"Starting analysis for {symbol}")
        data = fetch_stock_data_cached(symbol)
        if data.empty or len(data) < 50:
            logging.warning(f"No sufficient data for {symbol}: {len(data)} rows")
            return None
        
        data = analyze_stock(data)
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        logging.info(f"Analyzing {symbol} in {recommendation_mode} mode (data shape: {data.shape})")
        
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(data, symbol)
            if not rec or not rec.get('Recommendation'):
                logging.error(f"Invalid adaptive_recommendation output for {symbol}: {rec}")
                return None
            return {
                "Symbol": symbol,
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Recommendation": rec.get("Recommendation", "Hold"),
                "Score": rec.get("Score", 0),
                "Regime": rec.get("Regime"),
                "Position Size": rec.get("Position Size"),
                "Trailing Stop": rec.get("Trailing Stop"),
                "Reason": rec.get("Reason"),
                "Intraday": None,
                "Swing": None,
                "Short-Term": None,
                "Long-Term": None,
                "Mean_Reversion": None,
                "Breakout": None,
                "Ichimoku_Trend": None
            }
        else:
            rec = generate_recommendations(data, symbol)
            if not rec or not rec.get('Intraday'):
                logging.error(f"Invalid generate_recommendations output for {symbol}: {rec}")
                return None
            return {
                "Symbol": symbol,
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Intraday": rec.get("Intraday", "Hold"),
                "Swing": rec.get("Swing", "Hold"),
                "Short-Term": rec.get("Short-Term", "Hold"),
                "Long-Term": rec.get("Long-Term", "Hold"),
                "Mean_Reversion": rec.get("Mean_Reversion", "Hold"),
                "Breakout": rec.get("Breakout", "Hold"),
                "Ichimoku_Trend": rec.get("Ichimoku_Trend", "Hold"),
                "Score": rec.get("Score", 0),
                "Recommendation": None,
                "Regime": None,
                "Position Size": None,
                "Trailing Stop": None,
                "Reason": None
            }
    except Exception as e:
        error_msg = f"Error in analyze_stock_parallel for {symbol}: {str(e)} (data shape: {data.shape if 'data' in locals() else 'N/A'})"
        logging.error(error_msg)
        return None

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend([r for r in batch_results if r is not None])
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        time.sleep(3)
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy|Sell", na=False)]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend([r for r in batch_results if r is not None])
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        time.sleep(3)
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if recommendation is None or not isinstance(recommendation, str):
        return "⚪ N/A"
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    else:
        return f"⚪ {recommendation}"

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)

    try:
        loading_message = next(loading_messages)
    except StopIteration:
        loading_message = "Loading"

    dots = "." * (int(progress_value * 10) % 4)

    # Reliable 5% step tracking (0–20 steps)
    current_step = int(progress_value * 20)

    if not hasattr(update_progress, '_last_text_update'):
        update_progress._last_text_update = -1

    if current_step != update_progress._last_text_update:
        loading_text.text(f"{loading_message}{dots}")
        update_progress._last_text_update = current_step


def display_dashboard(symbol=None, data=None, recommendations=None):
    # Initialize session state
    if 'selected_sectors' not in st.session_state:
        st.session_state.selected_sectors = ["Bank"]
    if 'symbol' not in st.session_state:
        st.session_state.symbol = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'backtest_results_swing' not in st.session_state:
        st.session_state.backtest_results_swing = None
    if 'backtest_results_intraday' not in st.session_state:
        st.session_state.backtest_results_intraday = None
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"

    # Update session state if new data is provided
    if symbol and data is not None and recommendations is not None:
        st.session_state.symbol = symbol
        st.session_state.data = data
        st.session_state.recommendations = recommendations

    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

    # Sector selection
    sector_options = ["All"] + list(SECTORS.keys())
    st.session_state.selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=sector_options,
        default=st.session_state.selected_sectors,
        help="Choose one or more sectors to analyze. Select 'All' to include all sectors."
    )

    if "All" in st.session_state.selected_sectors:
        selected_stocks = list(set([stock for sector in SECTORS.values() for stock in sector]))
    else:
        selected_stocks = list(set([stock for sector in st.session_state.selected_sectors for stock in SECTORS.get(sector, [])]))

    if not selected_stocks:
        st.warning("⚠️ No stocks selected. Please choose at least one sector.")
        return

    # Top sectors button
    if st.button("🔎 Analyze Top Performing Sectors"):
        with st.spinner("🔍 Crunching sector data ..."):
            top_sectors = get_top_sectors_cached(rate_limit_delay=2, stocks_per_sector=2)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            for name, score in top_sectors:
                st.markdown(f"- **{name}**: {score:.2f}/7")

    # Daily top picks button
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        results_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        insert_top_picks(results_df, pick_type="daily")
        progress_bar.empty()
        loading_text.empty()
        if not results_df.empty:
            st.subheader("🏆 Today's Top 5 Stocks")
            for _, row in results_df.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    if st.session_state.recommendation_mode == "Adaptive":
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}  
                        Swing: {colored_recommendation(row.get('Swing', 'N/A'))}  
                        Short-Term: {colored_recommendation(row.get('Short-Term', 'N/A'))}  
                        Long-Term: {colored_recommendation(row.get('Long-Term', 'N/A'))}  
                        Mean Reversion: {colored_recommendation(row.get('Mean_Reversion', 'N/A'))}  
                        Breakout: {colored_recommendation(row.get('Breakout', 'N/A'))}  
                        Ichimoku Trend: {colored_recommendation(row.get('Ichimoku_Trend', 'N/A'))}
                        """)
        else:
            st.warning("⚠️ No top picks available due to data issues.")

    # Intraday top picks button
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        insert_top_picks(intraday_results, pick_type="intraday")
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    if st.session_state.recommendation_mode == "Adaptive":
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}
                        """)
        else:
            st.warning("⚠️ No intraday picks available due to data issues.")

    # Historical picks button
    if st.button("📜 View Historical Picks"):
        conn = sqlite3.connect('stock_picks.db')
        history_df = pd.read_sql_query("SELECT * FROM daily_picks ORDER BY date DESC", conn)
        conn.close()
        if not history_df.empty:
            st.subheader("📜 Historical Top Picks")
            all_dates = sorted(history_df['date'].unique(), reverse=True)
            date_filter = st.selectbox("Filter by Date", ["All"] + all_dates)
            pick_type_filter = st.selectbox("Filter by Pick Type", ["All", "daily", "intraday"])
            filtered_df = history_df.copy()
            if pick_type_filter != "All":
                filtered_df = filtered_df[filtered_df['pick_type'] == pick_type_filter]
            if date_filter != "All":
                filtered_df = filtered_df[filtered_df['date'] == date_filter]
            st.dataframe(filtered_df)
        else:
            st.warning("⚠️ No historical data available.")

    # Display stock analysis if symbol is available
    if st.session_state.symbol and st.session_state.data is not None and st.session_state.recommendations is not None:
        symbol = st.session_state.symbol
        data = st.session_state.data
        recommendations = st.session_state.recommendations

        st.header(f"📋 {symbol.split('-')[0]} Analysis")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            current_price = recommendations.get('Current Price', 'N/A')
            st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
        with col2:
            buy_at = recommendations.get('Buy At', 'N/A')
            st.metric("Buy At", f"₹{buy_at}")
        with col3:
            stop_loss = recommendations.get('Stop Loss', 'N/A')
            st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
        with col4:
            target = recommendations.get('Target', 'N/A')
            st.metric("Target", f"₹{target}")
        with col5:
            regime = recommendations.get('Regime', 'N/A') if st.session_state.recommendation_mode == "Adaptive" else 'N/A'
            st.metric("Market Regime", regime)

        st.subheader("📈 Trading Recommendations")
        if st.session_state.recommendation_mode == "Adaptive":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Recommendation**: {colored_recommendation(recommendations.get('Recommendation', 'N/A'))}")
                st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")
            with col2:
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/7")
                st.write(f"**Position Size (₹)**: {recommendations.get('Position Size', 'N/A')}")
            with col3:
                st.write(f"**Trailing Stop**: ₹{recommendations.get('Trailing Stop', 'N/A')}")
                st.write(f"**Volatility**: {assess_risk(data)}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Intraday**: {colored_recommendation(recommendations.get('Intraday', 'N/A'))}")
                st.write(f"**Swing**: {colored_recommendation(recommendations.get('Swing', 'N/A'))}")
            with col2:
                st.write(f"**Short-Term**: {colored_recommendation(recommendations.get('Short-Term', 'N/A'))}")
                st.write(f"**Long-Term**: {colored_recommendation(recommendations.get('Long-Term', 'N/A'))}")
            with col3:
                st.write(f"**Mean Reversion**: {colored_recommendation(recommendations.get('Mean_Reversion', 'N/A'))}")
                st.write(f"**Breakout**: {colored_recommendation(recommendations.get('Breakout', 'N/A'))}")
                st.write(f"**Ichimoku Trend**: {colored_recommendation(recommendations.get('Ichimoku_Trend', 'N/A'))}")
            st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/7")
            st.write(f"**Volatility**: {assess_risk(data)}")

        # Backtest form
        with st.form(key="backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                swing_button = st.form_submit_button("🔍 Backtest Swing Strategy")
            with col2:
                intraday_button = st.form_submit_button("🔍 Backtest Intraday Strategy")
            
            if swing_button or intraday_button:
                strategy = "Swing" if swing_button else "Intraday"
                with st.spinner(f"Running {strategy} Strategy backtest..."):
                    data_hash = hash(data.to_string())
                    backtest_results = backtest_stock(data, symbol, strategy=strategy, _data_hash=data_hash)
                    if strategy == "Swing":
                        st.session_state.backtest_results_swing = backtest_results
                    else:
                        st.session_state.backtest_results_intraday = backtest_results

        # Backtest results
        for strategy, results_key in [("Swing", "backtest_results_swing"), ("Intraday", "backtest_results_intraday")]:
            backtest_results = st.session_state.get(results_key)
            if backtest_results:
                st.subheader(f"📈 Backtest Results ({strategy} Strategy)")
                st.write(f"**Total Return**: {backtest_results['total_return']:.2f}%")
                st.write(f"**Annualized Return**: {backtest_results['annual_return']:.2f}%")
                st.write(f"**Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}")
                st.write(f"**Max Drawdown**: {backtest_results['max_drawdown']:.2f}%")
                st.write(f"**Number of Trades**: {backtest_results['trades']}")
                st.write(f"**Win Rate**: {backtest_results['win_rate']:.2f}%")
                with st.expander("Trade Details"):
                    for trade in backtest_results["trade_details"]:
                        profit = trade.get("profit", 0)
                        st.write(f"Entry: {trade['entry_date']} @ ₹{trade['entry_price']:.2f}, "
                                 f"Exit: {trade['exit_date']} @ ₹{trade['exit_price']:.2f}, "
                                 f"Profit: ₹{profit:.2f}")

                fig = px.line(data, x=data.index, y='Close', title=f"{symbol.split('-')[0]} Price with Signals")
                if backtest_results["buy_signals"]:
                    buy_dates, buy_prices = zip(*backtest_results["buy_signals"])
                    fig.add_scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals',
                                   marker=dict(color='green', symbol='triangle-up', size=10))
                if backtest_results["sell_signals"]:
                    sell_dates, sell_prices = zip(*backtest_results["sell_signals"])
                    fig.add_scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals',
                                   marker=dict(color='red', symbol='triangle-down', size=10))
                st.plotly_chart(fig, use_container_width=True)

        # Technical Indicators
        st.subheader("📊 Technical Indicators")
        indicators = [
            ("RSI", data['RSI'].iloc[-1], TOOLTIPS['RSI']),
            ("MACD", data['MACD'].iloc[-1], TOOLTIPS['MACD']),
            ("ATR", data['ATR'].iloc[-1], TOOLTIPS['ATR']),
            ("ADX", data['ADX'].iloc[-1], TOOLTIPS['ADX']),
            ("Bollinger Upper", data['Upper_Band'].iloc[-1], TOOLTIPS['Bollinger']),
            ("Bollinger Lower", data['Lower_Band'].iloc[-1], TOOLTIPS['Bollinger']),
            ("VWAP", data['VWAP'].iloc[-1], TOOLTIPS['VWAP']),
            ("Ichimoku Span A", data['Ichimoku_Span_A'].iloc[-1], TOOLTIPS['Ichimoku']),
            ("CMF", data['CMF'].iloc[-1], TOOLTIPS['CMF']),
        ]
        col1, col2 = st.columns(2)
        for i, (name, value, tooltip_text) in enumerate(indicators):
            if i % 2 == 0:
                with col1:
                    value = round(value, 2) if pd.notnull(value) else "N/A"
                    st.write(f"**{tooltip(name, tooltip_text)}**: {value}")
            else:
                with col2:
                    value = round(value, 2) if pd.notnull(value) else "N/A"
                    st.write(f"**{tooltip(name, tooltip_text)}**: {value}")
            
def main():
    init_database()
    st.sidebar.title("🔍 Stock Selection")
    stock_list = fetch_nse_stock_list()

    if 'symbol' not in st.session_state:
        st.session_state.symbol = stock_list[0]
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"

    symbol = st.sidebar.selectbox(
        "Select Stock",
        stock_list,
        key="stock_select",
        index=stock_list.index(st.session_state.symbol) if st.session_state.symbol in stock_list else 0
    )

    recommendation_mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Standard", "Adaptive"],
        index=0 if st.session_state.recommendation_mode == "Standard" else 1,
        help="Standard: Timeframe-specific recommendations. Adaptive: Regime-based with position sizing."
    )
    st.session_state.recommendation_mode = recommendation_mode

    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner("Loading stock data..."):
                data = fetch_stock_data_with_auth(symbol)
                if not data.empty:
                    data = analyze_stock(data)
                    recommendations = (adaptive_recommendation(data) if recommendation_mode == "Adaptive"
                                      else generate_recommendations(data, symbol))
                    st.session_state.symbol = symbol
                    st.session_state.data = data
                    st.session_state.recommendations = recommendations
                    st.session_state.backtest_results_swing = None
                    st.session_state.backtest_results_intraday = None
                    display_dashboard(symbol, data, recommendations)
                else:
                    st.warning("⚠️ No data available for the selected stock.")
    else:
        display_dashboard()
if __name__ == "__main__":
    main()
