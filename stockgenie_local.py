import pandas as pd
import ta
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm # Keep tqdm for potential future console logging, though Streamlit usually replaces it
import plotly.express as px
import time
import requests
import io
import random
import spacy # Keep spacy for text processing tasks if needed, currently unused in core logic
from pytrends.request import TrendReq # Keep pytrends for potential future integration
import itertools
from arch import arch_model
import warnings
import sqlite3
from diskcache import Cache
import os
# from dotenv import load_dotenv # Not needed if using st.secrets directly
from streamlit import cache_data
from supabase import create_client, Client
import json
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator # Ensure ChaikinMoneyFlowIndicator is used
from ratelimit import RateLimitDecorator as RateLimiter
from threading import Lock

data_api_calls = 0
data_api_lock = Lock()

# Configure logging (Moved to top for consistent configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Streamlit secrets for Dhan API
DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]

def dhan_headers():
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

@st.cache_data(ttl=86400)
def load_dhan_instrument_master():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    df = pd.read_csv(url, low_memory=False)
    # Filter for NSE Equity
    df = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_SEGMENT'] == 'E')]
    return df

def get_dhan_security_id(symbol):
    """Get Dhan security ID for a given symbol"""
    df = load_dhan_instrument_master()
    symbol_clean = normalize_symbol_dhan(symbol).upper()

    # Try multiple matching criteria
    for column in ['SEM_SMST_SECURITY_ID', 'SM_SYMBOL_NAME', 'SEM_TRADING_SYMBOL', 'SEM_CUSTOM_SYMBOL']:
        if column in df.columns:
            # Handle potential non-string types by converting to string before .str.upper()
            row = df[df[column].astype(str).str.upper() == symbol_clean]
            if not row.empty:
                security_id = row.iloc[0]['SEM_SMST_SECURITY_ID']
                logging.info(f"Found security_id for {symbol}: {security_id} in column {column}")
                return security_id

    # Log all possible matches for debugging
    possible_matches = df[df['SM_SYMBOL_NAME'].astype(str).str.contains(symbol_clean, case=False, na=False)]
    logging.warning(f"No security_id found for {symbol} ({symbol_clean}). Possible matches: {possible_matches[['SM_SYMBOL_NAME', 'SEM_TRADING_SYMBOL']].to_dict('records')}")
    return None

def test_dhan_connection():
    """Test if Dhan API credentials are working"""
    url = "https://api.dhan.co/v2/charts/intraday" # Intraday endpoint is usually less restrictive for a quick check
    headers = {
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    
    # Test with a known security ID (e.g., for RELIANCE)
    payload = {
        "securityId": "1333",  # RELIANCE security ID for NSE_EQ
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "interval": "1m" # Smallest interval for intraday
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            logging.info("Dhan API connection successful!")
            return True
        else:
            logging.error(f"Dhan API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"Dhan API connection error: {e}")
        return False

def normalize_symbol_dhan(symbol):
    # Remove .NS, .BO, and -EQ
    return symbol.replace(".NS", "").replace(".BO", "").replace("-EQ", "")
    
# --- Supabase setup ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def color_code_action(val):
    if isinstance(val, str):
        if "Book Profit" in val:
            return "color: green; font-weight: bold"
        elif "Review" in val:
            return "color: red; font-weight: bold"
        elif "Hold" in val:
            return "color: orange; font-weight: bold"
    return ""

def color_code_change(val):
    try:
        v = float(val)
        if v > 0:
            return "color: green"
        elif v < 0:
            return "color: red"
        else:
            return ""
    except:
        return ""

def style_picks_df(df):
    return df.style.applymap(color_code_action, subset=["What to do now?"]) \
                   .applymap(color_code_change, subset=["% Change"])

def add_action_and_change(df):
    def action(row):
        try:
            buy_at = float(row['buy_at'])
            current_price = float(row['current_price'])
            if current_price > buy_at * 1.05:
                return "Book Profit / Hold"
            elif current_price < buy_at * 0.97:
                return "Review / Consider Stop Loss"
            else:
                return "Hold"
        except Exception:
            return "N/A"
    df['% Change'] = ((df['current_price'] - df['buy_at']) / df['buy_at'] * 100).round(2)
    df['What to do now?'] = df.apply(action, axis=1)
    return df


warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

def normalize_symbol(symbol):
    if symbol.endswith('.NS'):
        return symbol.replace('.NS', '-EQ')
    elif symbol.endswith('.BO'):
        return symbol.replace('.BO', '-EQ')
    elif '-EQ' in symbol:
        return symbol
    else:
        return symbol + '-EQ'
        
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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

# FIX 1: Added "OBV" to TOOLTIPS dictionary.
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
    "Score": "Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal.",
    "OBV": "On-Balance Volume - Measures buying and selling pressure by adding/subtracting volume based on price changes." # Added missing OBV tooltip
}

SECTORS = {
 
"Day": ["CARTRADE","ACMESOLAR","AGIIL","ALICON","BLUESTARCO","SWIGGY","INDIAMART","ASTERDM","CRIZAC"],
 
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
    "SUNDRMFAST.NS", "EXIDEIND.NS", "ARE&M.NS", "BOSCHLTD.NS", "ENDURANCE.NS",
    "UNOMINDA.NS", "ZFCVINDIA.NS", "GABRIEL.NS", "SUPRAJIT.NS", "LUMAXTECH.NS",
    "FIEMIND.NS", "SUBROS.NS", "JAMNAAUTO.NS", "SHRIRAMFIN.NS", "ESCORTS.NS",
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
    "TRIL.NS", "TDPOWERSYS.NS", "JYOTISTRUC.NS", "IWEL.NS","ACMESOLAR.NS"
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
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS",
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
    "VISHNU.NS", "IGPL.NS", "TIRUMALCHM.NS","RALLIS.NS"
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
    "CARTRADE.NS", "HONASA.NS", "PAYTM.NS", "SIGNATURE.NS", "RRKABEL.NS",
    "HMAAGRO.NS", "RKFORGE.NS", "CAMPUS.NS", "SENCO.NS", "CONCORDBIO.NS"
  ]
}


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
        stock_list = [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        # Fallback to predefined list if NSE data cannot be fetched
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

# Helper function to parse period strings to days
def parse_period_to_days(period):
    """Convert period string (e.g., '5y', '1mo') to number of days."""
    period = period.lower()
    try:
        if period.endswith('y'):
            years = float(period[:-1])
            return int(years * 365)  # Approximate years to days
        elif period.endswith('mo'):
            months = float(period[:-2])
            return int(months * 30)  # Approximate months to days
        elif period.endswith('d'):
            days = float(period[:-1])
            return int(days)
        else:
            logging.warning(f"Unsupported period format: {period}. Defaulting to 30 days.")
            return 30
    except ValueError as e:
        logging.error(f"Invalid period format: {period}, error: {e}. Defaulting to 30 days.")
        return 30
        
@RateLimiter(calls=5, period=1) # 5 calls per second for Dhan API
def fetch_stock_data_with_dhan(symbol, period="5y", interval="1d"):
    global data_api_calls
    with data_api_lock:
        # Check against a lower threshold for warning, and total for hard limit
        if data_api_calls >= 90000: # 90% of daily limit
            logging.warning(f"⚠️ Approaching Data API daily limit: {data_api_calls}/100000")
        if data_api_calls >= 100000: # Max daily limit
            logging.error("⚠️ Reached daily Data API limit of 100,000 requests. Cannot fetch more data.")
            # For Streamlit, display a warning and return empty df
            st.error("⚠️ Reached daily Data API limit of 100,000 requests. Please try again tomorrow.")
            return pd.DataFrame()
        data_api_calls += 1

    security_id = get_dhan_security_id(symbol)
    if not security_id:
        logging.error(f"No security ID found for {symbol}")
        return pd.DataFrame()

    url = "https://api.dhan.co/v2/charts/historical"
    headers = dhan_headers()
    days = parse_period_to_days(period)
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": interval,
        "fromDate": (pd.Timestamp.now() - pd.to_timedelta(days, unit='D')).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d")
    }

    try:
        logging.info(f"Requesting data for {symbol} with payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        data = response.json()

        logging.debug(f"API response for {symbol}: {data}")

        api_data = data.get("data", data) # Some responses wrap data under a "data" key

        if isinstance(api_data, dict) and all(isinstance(v, list) for v in api_data.values()):
            # If api_data is a dict with lists (columnar), convert to list of dicts (row format)
            if not api_data or not next(iter(api_data.values()), []): # Check if any list is empty
                logging.warning(f"Empty data received for {symbol} in columnar format: {api_data}")
                return pd.DataFrame()
            n = len(next(iter(api_data.values())))
            row_data = [
                {k: api_data[k][i] for k in api_data}
                for i in range(n)
            ]
            df = pd.DataFrame(row_data)
        elif isinstance(api_data, list):
            df = pd.DataFrame(api_data)
        else:
            logging.warning(f"Unknown data format for {symbol}: {type(api_data)}. Raw response: {data}")
            return pd.DataFrame()

        if df.empty:
            logging.warning(f"Empty DataFrame created for {symbol} after parsing. Response data: {api_data}")
            return pd.DataFrame()

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "timestamp": "Date"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Validate required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns for {symbol}: {missing_columns}. Columns found: {df.columns.tolist()}")
            return pd.DataFrame()

        # Ensure numeric types
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_columns) # Drop rows where essential columns became NaN

        logging.info(f"Successfully fetched data for {symbol}: {len(df)} rows")
        return df[required_columns]

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {symbol}: {e}, Response: {e.response.text if e.response else 'No response'}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error for {symbol}: {e}")
        return pd.DataFrame()
       
@lru_cache(maxsize=1000) # Keep cache to avoid re-fetching data for the same symbol
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    return fetch_stock_data_with_dhan(symbol, period, interval)

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 2: # Ensure at least two days for comparison
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            elif data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                declines += 1
    return advances / declines if declines != 0 else 0 if advances == 0 else float('inf')


def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return []
    
    # Ensure enough data points for GARCH
    if len(returns) < 50: # Arbitrary but reasonable threshold for statistical models
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Handle cases where std_return might be 0 (e.g., constant price for short period)
        if std_return == 0:
            logging.warning("Standard deviation of returns is zero for Monte Carlo. Using flat projection.")
            simulation_results = [[data['Close'].iloc[-1]] * (days + 1) for _ in range(simulations)]
            return simulation_results

        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    try:
        model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
        garch_fit = model.fit(disp='off')
        forecasts = garch_fit.forecast(horizon=days)
        
        # Ensure 'variance' attribute is available and not empty
        if forecasts.variance.empty:
            logging.warning("GARCH variance forecasts are empty. Falling back to simple simulation.")
            return monte_carlo_simulation(data, simulations, days)
            
        volatility = np.sqrt(forecasts.variance.iloc[-1].values)
        
        # Ensure volatility has values for each day
        if len(volatility) < days:
            logging.warning(f"GARCH volatility forecasts (len={len(volatility)}) less than required days ({days}). Replicating last volatility.")
            volatility = np.pad(volatility, (0, days - len(volatility)), mode='edge')
            
        mean_return = returns.mean()
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for i in range(days):
                # Ensure volatility index i is valid
                current_volatility = volatility[min(i, len(volatility) - 1)]
                price = price_series[-1] * (1 + np.random.normal(mean_return, current_volatility))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    except Exception as e:
        logging.error(f"Error in Monte Carlo GARCH simulation: {e}. Falling back to simple simulation.")
        return monte_carlo_simulation(data, simulations, days)


def extract_entities(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return entities
    except Exception as e:
        logging.error(f"Error in spaCy entity extraction: {e}. Is 'en_core_web_sm' downloaded? (python -m spacy download en_core_web_sm)")
        return []

def get_trending_stocks():
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        trending = pytrends.trending_searches(pn='india')
        return trending
    except Exception as e:
        logging.error(f"Error fetching trending searches with pytrends: {e}")
        return pd.DataFrame()

def assess_risk(data):
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
        atr_ratio = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_ratio > 0.04: # High volatility
            return "High Volatility"
        elif atr_ratio > 0.02: # Moderate volatility
            return "Moderate Volatility"
        else:
            return "Low Volatility"
    return "N/A" # Default if ATR or Close is not available

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

def detect_divergence(data):
    if 'RSI' not in data.columns or len(data) < 5:
        return "Not enough data"
    rsi = data['RSI']
    price = data['Close']
    
    # Simplistic check for bullish divergence (price lower low, RSI higher low)
    if price.iloc[-1] < price.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] and \
       pd.notnull(price.iloc[-1]) and pd.notnull(price.iloc[-3]) and \
       pd.notnull(rsi.iloc[-1]) and pd.notnull(rsi.iloc[-3]):
        return "Potential Bullish Divergence"
    # Simplistic check for bearish divergence (price higher high, RSI lower high)
    elif price.iloc[-1] > price.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] and \
         pd.notnull(price.iloc[-1]) and pd.notnull(price.iloc[-3]) and \
         pd.notnull(rsi.iloc[-1]) and pd.notnull(rsi.iloc[-3]):
        return "Potential Bearish Divergence"
    
    return "No Clear Divergence"

# Ensure ta.momentum.ChandeMomentumOscillator is imported if used
# It is imported at the top of the file.
def calculate_cmo(close, window=14):
    try:
        cmo_indicator = ta.momentum.ChandeMomentumOscillator(close=close, window=window)
        return cmo_indicator.chande_momentum_oscillator()
    except Exception as e:
        logging.warning(f"⚠️ Failed to compute CMO via ta.momentum.ChandeMomentumOscillator: {str(e)}. Check your 'ta' library version.")
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
    'ADX': 14,
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

def validate_data(data, min_length=52):
    """
    Validates that the DataFrame has required columns, sufficient rows, and numeric data.
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not isinstance(data, pd.DataFrame) or data.empty:
        logging.debug("Data is not a DataFrame or is empty")
        return False
    if not all(col in data.columns for col in required_columns):
        logging.debug(f"Missing required columns: {set(required_columns) - set(data.columns)}")
        return False
    if len(data) < min_length:
        logging.debug(f"Insufficient data rows: {len(data)} < {min_length}")
        return False
    # Check for NaN values in required columns for the last `min_length` rows only
    if data[required_columns].tail(min_length).isna().any().any():
        logging.debug("Data contains NaN values in required columns (last min_length rows)")
        return False
    if not data[required_columns].apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
        logging.debug("Non-numeric values found in required columns")
        return False
    return True

def can_compute_indicator(data, indicator_name):
    """
    Checks if data is sufficient to compute the specified indicator.
    """
    required_length = INDICATOR_MIN_LENGTHS.get(indicator_name, 52)
    return validate_data(data, min_length=required_length)

def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    """
    columns = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
        'VWAP', 'Avg_Volume', 'Volume_Spike', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2',
        'Fib_50.0', 'Fib_61.8', 'Divergence', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF', 'Donchian_Upper',
        'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle', 'Keltner_Lower',
        'TRIX', 'Ultimate_Osc', 'CMO', 'VPT', 'ATR_pct', 'DMP', 'DMN', 'ADX_Slope' # Added ADX components for regime
    ]

    # Initialize all indicator columns to NaN before computation
    for col in columns:
        if col not in data.columns:
            data[col] = np.nan # Use np.nan for numeric columns
    
    # Ensure data types are numeric before any operations
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']) # Drop rows with NaNs in core data

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Data validation failed in analyze_stock. Data length: {len(data)}. Returning initial data with NaNs.")
        return data # Return data with NaNs for indicators if validation fails

    try:
        # RSI
        if can_compute_indicator(data, 'RSI'):
            data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()

        # MACD
        if can_compute_indicator(data, 'MACD'):
            macd = MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()

        # SMA
        if can_compute_indicator(data, 'SMA_50'):
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
        if can_compute_indicator(data, 'SMA_200'):
            data['SMA_200'] = data['Close'].rolling(window=200).mean()

        # EMA
        if can_compute_indicator(data, 'EMA_20'):
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        if can_compute_indicator(data, 'EMA_50'):
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # Bollinger Bands
        if can_compute_indicator(data, 'Bollinger'):
            bb = BollingerBands(data['Close'], window=20, window_dev=2)
            data['Upper_Band'] = bb.bollinger_hband()
            data['Middle_Band'] = bb.bollinger_mavg()
            data['Lower_Band'] = bb.bollinger_lband()

        # Stochastic Oscillator
        if can_compute_indicator(data, 'Stochastic'):
            stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()

        # ATR
        if can_compute_indicator(data, 'ATR'):
            data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
            data['ATR_pct'] = data['ATR'] / data['Close'] # Add ATR percentage for regime detection

        # ADX
        if can_compute_indicator(data, 'ADX'):
            adx_ind = ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
            data['ADX'] = adx_ind.adx()
            data['DMP'] = adx_ind.adx_pos() # Positive Directional Movement
            data['DMN'] = adx_ind.adx_neg() # Negative Directional Movement
            data['ADX_Slope'] = data['ADX'].diff(periods=5) # 5-period slope of ADX

        # OBV
        if can_compute_indicator(data, 'OBV'):
            data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

        # Ichimoku Cloud
        if can_compute_indicator(data, 'Ichimoku'):
            ichimoku = IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Chikou'] = data['Close'].shift(-26) # Chikou Span manually

        # CMF
        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        
        # New: TRIX
        if can_compute_indicator(data, 'TRIX'):
            data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()

        # New: Ultimate Oscillator
        if can_compute_indicator(data, 'Ultimate_Osc'):
            data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(data['High'], data['Low'], data['Close']).ultimate_oscillator()
        
        # New: CMO (using ta's implementation)
        if can_compute_indicator(data, 'CMO'):
            # FIX 2: Ensure the ta.momentum.ChandeMomentumOscillator is accessible.
            # If this line causes an error, it points to a 'ta' library issue.
            # Running `pip install --upgrade --force-reinstall ta` might fix it.
            data['CMO'] = ta.momentum.ChandeMomentumOscillator(data['Close'], window=14).chande_momentum_oscillator()

        # New: VPT (Volume Price Trend)
        if can_compute_indicator(data, 'VPT'):
            data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()

        # Volume Spike and Avg Volume
        data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
        if not data['Avg_Volume'].empty and pd.notnull(data['Avg_Volume'].iloc[-1]):
             data['Volume_Spike'] = data['Volume'] > data['Avg_Volume'] * 2
        else:
            data['Volume_Spike'] = False # Default to no spike if avg volume not computable

        return data

    except Exception as e:
        logging.error(f"Error in analyze_stock: {str(e)}")
        # If an error occurs, ensure all indicator columns are present but likely NaN
        # This loop is crucial if an error halts computation before columns are fully populated.
        for col in columns:
            if col not in data.columns:
                data[col] = np.nan
        return data

def calculate_buy_at(data):
    """
    Calculates the buy price based on recent price action and indicators.
    """
    if not validate_data(data, min_length=10) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) else 50
        
        if rsi < 35 and atr > 0: # If oversold, aim for a small dip or current price
            buy_at = close * (1 - 0.5 * (atr / close))
        elif atr > 0: # For normal conditions, a slight discount or market price
            buy_at = close * (1 - 0.2 * (atr / close))
        else: # Fallback
            buy_at = close * 0.99
            
        return round(float(buy_at), 2)
    except Exception as e:
        logging.error(f"Error calculating buy_at: {str(e)}")
        return None

def calculate_stop_loss(data, atr_multiplier=2.5):
    """
    Calculates the stop loss price based on ATR and recent lows.
    """
    if not validate_data(data, min_length=10) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        
        if atr > 0:
            # Adjust multiplier based on ADX for trend strength if available
            adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
            if adx > 25: # Strong trend, might use a tighter stop
                atr_multiplier = 2.0
            elif adx < 20: # Weak trend/sideways, might need a wider stop or different approach
                atr_multiplier = 3.0
            
            stop_loss = close - atr_multiplier * atr
            # Ensure stop loss is not excessively tight or wide. Min 3% max 10%
            if stop_loss < close * 0.90:
                stop_loss = close * 0.90 # Max 10% loss
            elif stop_loss > close * 0.97:
                stop_loss = close * 0.97 # Min 3% loss from current price
        else: # Fallback if ATR is not available
            stop_loss = close * 0.95 # Default 5% stop
            
        return round(float(stop_loss), 2)
    except Exception as e:
        logging.error(f"Error calculating stop_loss: {str(e)}")
        return None

def calculate_target(data, risk_reward_ratio=2.0): # Default R:R of 2
    """
    Calculates the target price based on ATR and a risk-reward ratio.
    """
    if not validate_data(data, min_length=10) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        stop_loss = calculate_stop_loss(data) # Get the calculated stop loss
        
        if stop_loss is None or stop_loss >= close: # Invalid stop loss
            return None
        
        risk_per_share = close - stop_loss # Calculate risk based on stop loss
        
        if risk_per_share <= 0: # Avoid division by zero or negative risk
            return None
        
        # Adjust R:R based on ADX for trend strength
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
        if adx > 25: # Strong trend, can aim for higher R:R
            adjusted_rr = 3.0
        elif adx < 20: # Weak trend/sideways, aim for lower R:R
            adjusted_rr = 1.5
        else:
            adjusted_rr = risk_reward_ratio
            
        target = close + (risk_per_share * adjusted_rr)
        
        # Ensure target is not excessively high
        if target > close * 1.25: # Cap target at 25% for short/medium term
            target = close * 1.25
            
        return round(float(target), 2)
    except Exception as e:
        logging.error(f"Error calculating target: {str(e)}")
        return None

def calculate_trailing_stop(current_price, atr, atr_multiplier=2.5):
    """
    Calculates a simple ATR-based trailing stop.
    This is a static calculation based on current price, not a dynamic trailing stop for an open position.
    For an open position, the trailing stop would only move up.
    """
    if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
        return None
    
    trailing_stop = current_price - atr_multiplier * atr
    
    # Ensure it's not higher than current price and at least 1% below
    trailing_stop = max(current_price * 0.99, trailing_stop) # Ensure a minimum buffer
    
    return round(max(0, trailing_stop), 2)


def classify_market_regime(data):
    """Classifies regime based on volatility, trend strength (ADX), and trend direction (SMA)."""
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['SMA_200']): # Need sufficient data for SMAs and ADX
        return 'Insufficient Data'

    close = data['Close'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    sma_200 = data['SMA_200'].iloc[-1]
    atr_pct = data['ATR_pct'].iloc[-1]
    adx = data['ADX'].iloc[-1]
    adx_slope = data['ADX_Slope'].iloc[-1] # Assuming ADX_Slope is already calculated in analyze_stock

    if pd.isna(close) or pd.isna(sma_50) or pd.isna(sma_200) or \
       pd.isna(atr_pct) or pd.isna(adx) or pd.isna(adx_slope):
        return 'Incomplete Indicator Data'

    # Volatility Check
    if atr_pct > 0.04: # High volatility (e.g., 4% average true range relative to price)
        return 'Highly Volatile'

    # Trend Strength (ADX)
    if adx > 25: # Strong Trend
        if close > sma_50 and close > sma_200: # Bullish setup
            return 'Bullish Trending'
        elif close < sma_50 and close < sma_200: # Bearish setup
            return 'Bearish Trending'
        else: # Strong trend but mixed price action
            return 'Trending (Unclear Direction)'
    elif adx < 20: # Weak/No Trend (Sideways)
        return 'Sideways/Consolidating'
    else: # Developing Trend (20-25)
        if close > sma_50 and close > sma_200:
            return 'Bullish (Developing Trend)'
        elif close < sma_50 and close < sma_200:
            return 'Bearish (Developing Trend)'
        else:
            return 'Neutral'


def compute_signal_score(data, symbol=None):
    """
    Computes a weighted score based on normalized technical indicators and market conditions.
    Returns a score between -10 and 10, with negative scores indicating a bearish bias.
    """
    score = 0.0
    reason_components = []

    weights = {
        'RSI_Oversold': 2.0, 'RSI_Overbought': -2.0,
        'MACD_Bullish': 1.5, 'MACD_Bearish': -1.5,
        'Ichimoku_Bullish': 2.0, 'Ichimoku_Bearish': -2.0,
        'CMF_Buying': 0.5, 'CMF_Selling': -0.5,
        'ADX_StrongBull': 1.0, 'ADX_StrongBear': -1.0,
        'Bollinger_BreakoutUp': 1.0, 'Bollinger_BreakoutDown': -1.0,
        'Volume_SpikeUp': 0.5, 'Volume_SpikeDown': -0.5
    }
    
    max_possible_raw_score = sum([w for k,w in weights.items() if w > 0]) # Sum of positive weights

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Invalid data for scoring {symbol}, returning 0.")
        return 0, ["Insufficient data to compute score."]

    close_price = data['Close'].iloc[-1]
    current_volume = data['Volume'].iloc[-1]

    # --- Basic Filters (pre-score adjustments) ---
    # Filter out very low volume stocks
    if 'Avg_Volume' in data.columns and pd.notnull(data['Avg_Volume'].iloc[-1]):
        if current_volume < data['Avg_Volume'].iloc[-1] * 0.3: # Less than 30% of average volume
            score -= 2 # Penalize low volume significantly
            reason_components.append("Very low volume, caution advised.")

    # --- Indicator Contributions ---

    # RSI
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        if rsi < 30: # Oversold
            score += weights['RSI_Oversold'] * (1 - (rsi / 30)) # Score increases as RSI gets lower (closer to 0)
            reason_components.append(f"RSI({int(rsi)}) is oversold (potential bounce).")
        elif rsi > 70: # Overbought
            score += weights['RSI_Overbought'] * ((rsi - 70) / 30) # Score decreases as RSI gets higher (closer to 100)
            reason_components.append(f"RSI({int(rsi)}) is overbought (potential pullback).")

    # MACD
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        macd_hist = data['MACD_hist'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0: # Bullish crossover and histogram expanding above zero
            score += weights['MACD_Bullish']
            reason_components.append("MACD is bullish (crossover and positive momentum).")
        elif macd < macd_signal and macd_hist < 0: # Bearish crossover and histogram expanding below zero
            score += weights['MACD_Bearish']
            reason_components.append("MACD is bearish (crossover and negative momentum).")
        
    # Ichimoku Cloud
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
        close = data['Close'].iloc[-1]
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        span_b = data['Ichimoku_Span_B'].iloc[-1]
        
        if close > max(span_a, span_b): # Price above cloud
            score += weights['Ichimoku_Bullish']
            reason_components.append("Price is above Ichimoku Cloud (strong bullish trend).")
        elif close < min(span_a, span_b): # Price below cloud
            score += weights['Ichimoku_Bearish']
            reason_components.append("Price is below Ichimoku Cloud (strong bearish trend).")

    # CMF
    if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        if cmf > 0.15: # Strong buying pressure
            score += weights['CMF_Buying']
            reason_components.append(f"CMF({cmf:.2f}) indicates strong buying pressure.")
        elif cmf < -0.15: # Strong selling pressure
            score += weights['CMF_Selling']
            reason_components.append(f"CMF({cmf:.2f}) indicates strong selling pressure.")

    # ADX
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['ADX', 'DMP', 'DMN']):
        adx = data['ADX'].iloc[-1]
        dmp = data['DMP'].iloc[-1]
        dmn = data['DMN'].iloc[-1]

        if adx > 25 and dmp > dmn: # Strong bullish trend
            score += weights['ADX_StrongBull']
            reason_components.append(f"ADX({int(adx)}) shows strong bullish trend (DI+ > DI-).")
        elif adx > 25 and dmn > dmp: # Strong bearish trend
            score += weights['ADX_StrongBear']
            reason_components.append(f"ADX({int(adx)}) shows strong bearish trend (DI- > DI+).")
        # ADX trend strengthening/weakening for more nuanced signals
        if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]):
            adx_slope = data['ADX_Slope'].iloc[-1]
            if adx_slope > 0 and adx < 40: # ADX rising, trend strengthening
                reason_components.append(f"ADX slope positive (trend strengthening).")
            elif adx_slope < 0 and adx > 20: # ADX falling, trend weakening
                reason_components.append(f"ADX slope negative (trend weakening).")

    # Bollinger Bands Breakout / Squeeze
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Upper_Band', 'Lower_Band', 'Middle_Band', 'Close']):
        close = data['Close'].iloc[-1]
        upper_band = data['Upper_Band'].iloc[-1]
        lower_band = data['Lower_Band'].iloc[-1]
        middle_band = data['Middle_Band'].iloc[-1]
        
        if close > upper_band: # Price breaking out above upper band
            score += weights['Bollinger_BreakoutUp']
            reason_components.append("Price breaking out above Bollinger Upper Band (bullish breakout).")
        elif close < lower_band: # Price breaking out below lower band
            score += weights['Bollinger_BreakoutDown']
            reason_components.append("Price breaking out below Bollinger Lower Band (bearish breakout).")
        
        # Bollinger Band Width (volatility squeeze) - implies potential future breakout
        if middle_band != 0:
            bb_width = (upper_band - lower_band) / middle_band
            if len(data['Middle_Band']) >= 20:
                avg_bb_width = ((data['Upper_Band'] - data['Lower_Band']) / data['Middle_Band']).rolling(window=20).mean().iloc[-1]
                if pd.notnull(avg_bb_width) and bb_width < avg_bb_width * 0.7: # Current width is 30% less than avg
                    score += 0.25 # Small positive for potential volatility expansion
                    reason_components.append("Bollinger Bands are squeezing (low volatility, potential breakout soon).")

    # Volume Spike
    if 'Volume_Spike' in data.columns and pd.notnull(data['Volume_Spike'].iloc[-1]) and data['Volume_Spike'].iloc[-1]:
        if close > data['Close'].iloc[-2]: # Price up on spike
            score += weights['Volume_SpikeUp']
            reason_components.append("Significant volume spike confirming upward price move.")
        elif close < data['Close'].iloc[-2]: # Price down on spike
            score += weights['Volume_SpikeDown']
            reason_components.append("Significant volume spike confirming downward price move.")

    # Normalize the score to be between -10 and 10
    # Map score from its potential range to [-10, 10]
    # Rough estimate of possible min/max raw score (sum of most positive/negative weights)
    # The current weights result in max_possible_raw_score = 7.5. Let's scale it to 10.
    scaled_score = (score / max_possible_raw_score) * 10 if max_possible_raw_score > 0 else 0
    final_score = min(max(scaled_score, -10), 10) # Clamp between -10 and 10
    
    final_reason = " | ".join(reason_components) if reason_components else "No strong indicator signals."
    
    logging.info(f"Signal score for {symbol}: {final_score:.2f}, Reason: {final_reason}")
    return final_score, final_reason


def adaptive_recommendation(data, symbol=None, equity=100000, risk_per_trade_pct=1):
    """
    Generates adaptive trading recommendations based on market regime and technical indicators.
    Includes position sizing, trailing stop, and detailed reasons.
    """
    recommendations = {
        "Recommendation": "Hold",
        "Current Price": None,
        "Buy At": None,
        "Stop Loss": None,
        "Target": None,
        "Score": 0,
        "Regime": "N/A",
        "Position Size": {"shares": 0, "value": 0},
        "Trailing Stop": None,
        "Reason": "N/A"
    }

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['SMA_200']): # Needs more data for regime
        recommendations["Reason"] = "Insufficient historical data for comprehensive adaptive analysis."
        logging.warning(f"Insufficient data for {symbol} in adaptive_recommendation.")
        return recommendations

    try:
        current_price = round(float(data['Close'].iloc[-1]), 2)
        recommendations["Current Price"] = current_price
        
        if pd.isna(current_price):
            recommendations["Reason"] = "Current price not available for adaptive analysis."
            return recommendations

        # 1. Classify Market Regime
        market_regime = classify_market_regime(data)
        recommendations["Regime"] = market_regime
        logging.info(f"Market Regime for {symbol}: {market_regime}")

        # 2. Compute Signal Score and Reasons
        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2) # Round score for display
        recommendations["Reason"] = reason_text
        
        # --- Adaptive Recommendation Logic ---
        final_recommendation = "Hold" 

        # Strong buy/sell thresholds for different regimes
        if "Bullish Trending" in market_regime:
            if score >= 6: final_recommendation = "Strong Buy (Trend Continuation)"
            elif score >= 3: final_recommendation = "Buy (Trend Following)"
        elif "Bearish Trending" in market_regime:
            if score <= -6: final_recommendation = "Strong Sell (Trend Continuation)"
            elif score <= -3: final_recommendation = "Sell (Trend Following)"
        elif "Highly Volatile" in market_regime:
            # In volatile markets, mean reversion strategies might be better
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) else 50
            if rsi < 25 and score > 2: final_recommendation = "Strong Buy (Volatile Mean Reversion)"
            elif rsi > 75 and score < -2: final_recommendation = "Strong Sell (Volatile Mean Reversion)"
            elif score >= 4: final_recommendation = "Buy (Volatile Opportunity)"
            elif score <= -4: final_recommendation = "Sell (Volatile Risk)"
        elif "Sideways/Consolidating" in market_regime:
            if score >= 5: final_recommendation = "Buy (Consolidation Breakout Potential)"
            elif score <= -5: final_recommendation = "Sell (Consolidation Breakout Risk)"
            elif score >= 2: final_recommendation = "Buy (Range Low)"
            elif score <= -2: final_recommendation = "Sell (Range High)"
        else: # Neutral or Developing Trend
            if score >= 5: final_recommendation = "Buy (Developing Bullish)"
            elif score <= -5: final_recommendation = "Sell (Developing Bearish)"
            elif score >= 2: final_recommendation = "Consider Buy"
            elif score <= -2: final_recommendation = "Consider Sell"
            
        recommendations["Recommendation"] = final_recommendation

        # --- Calculate Actionable Levels (Buy At, Stop Loss, Target) ---
        buy_at = calculate_buy_at(data)
        stop_loss = calculate_stop_loss(data)
        target = calculate_target(data)

        recommendations["Buy At"] = buy_at
        recommendations["Stop Loss"] = stop_loss
        recommendations["Target"] = target

        # --- Position Sizing ---
        # Calculate risk amount based on equity and risk percentage
        risk_capital = equity * (risk_per_trade_pct / 100)

        # Calculate position size based on Buy At and Stop Loss
        if buy_at is not None and stop_loss is not None and buy_at > stop_loss:
            risk_per_share = buy_at - stop_loss
            if risk_per_share > 0:
                position_shares = int(risk_capital / risk_per_share)
                # Ensure position value is not unreasonably high compared to equity
                position_value = position_shares * current_price
                if position_value > equity * 0.25: # Max 25% of equity per trade
                    position_value = equity * 0.25
                    position_shares = int(position_value / current_price) if current_price > 0 else 0
                
                recommendations["Position Size"] = {"shares": position_shares, "value": round(position_value, 2)}
            else:
                recommendations["Position Size"] = {"shares": 0, "value": 0}
        else:
             recommendations["Position Size"] = {"shares": 0, "value": 0}

        # --- Trailing Stop Calculation ---
        if recommendations["Recommendation"].lower().startswith("buy"): # Only calculate for buy recommendations
             recommendations["Trailing Stop"] = calculate_trailing_stop(current_price, data['ATR'].iloc[-1] if 'ATR' in data.columns else None)
        else:
             recommendations["Trailing Stop"] = None

        logging.info(f"Adaptive recommendations for {symbol}: {recommendations}")
        return recommendations

    except Exception as e:
        logging.error(f"Critical error in adaptive_recommendation for {symbol}: {str(e)}")
        recommendations["Reason"] = f"An unexpected error occurred: {str(e)}"
        return recommendations
        
def generate_recommendations(data, symbol=None):
    """
    Generates trading recommendations based on technical indicators (Standard Mode).
    """
    recommendations = {
        "Intraday": "Hold",
        "Swing": "Hold",
        "Short-Term": "Hold",
        "Long-Term": "Hold",
        "Mean_Reversion": "Hold",
        "Breakout": "Hold",
        "Ichimoku_Trend": "Hold",
        "Current Price": None,
        "Buy At": None,
        "Stop Loss": None,
        "Target": None,
        "Score": 0
    }

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Invalid data for standard recommendations: {symbol}")
        return recommendations

    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        
        # Calculate score and reason for Standard mode as well, using compute_signal_score
        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2)

        buy_score = 0
        sell_score = 0

        # RSI
        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi <= 20:
                buy_score += 4
                recommendations["Mean_Reversion"] = "Strong Buy"
            elif rsi < 30:
                buy_score += 2
                recommendations["Mean_Reversion"] = "Buy"
            elif rsi > 70:
                sell_score += 2
                recommendations["Mean_Reversion"] = "Sell"
            elif rsi > 80:
                sell_score += 4
                recommendations["Mean_Reversion"] = "Strong Sell"

        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
            if macd_diff > 0 and data['MACD'].iloc[-1] > 0: # Bullish cross above zero
                buy_score += 3
                recommendations["Swing"] = "Strong Buy"
            elif macd_diff > 0: # Bullish cross
                buy_score += 2
                recommendations["Swing"] = "Buy"
            elif macd_diff < 0 and data['MACD'].iloc[-1] < 0: # Bearish cross below zero
                sell_score += 3
                recommendations["Swing"] = "Strong Sell"
            elif macd_diff < 0: # Bearish cross
                sell_score += 2
                recommendations["Swing"] = "Sell"

        # Ichimoku
        if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
            close = data['Close'].iloc[-1]
            span_a = data['Ichimoku_Span_A'].iloc[-1]
            span_b = data['Ichimoku_Span_B'].iloc[-1]
            
            if close > span_a and close > span_b: # Price above cloud
                buy_score += 3
                recommendations["Ichimoku_Trend"] = "Buy"
            elif close < span_a and close < span_b: # Price below cloud
                sell_score += 3
                recommendations["Ichimoku_Trend"] = "Sell"

        # Bollinger Bands Breakout
        if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]):
            close = data['Close'].iloc[-1]
            if close > data['Upper_Band'].iloc[-1]:
                buy_score += 2
                recommendations["Breakout"] = "Buy"
            elif close < data['Lower_Band'].iloc[-1]:
                sell_score += 2
                recommendations["Breakout"] = "Sell"

        # CMF
        if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
            cmf = data['CMF'].iloc[-1]
            if cmf > 0.2:
                buy_score += 1
            elif cmf < -0.2:
                sell_score += 1
            
        # Overall recommendation based on net score
        net_score = buy_score - sell_score
        
        if net_score >= 5:
            recommendations["Intraday"] = "Strong Buy"
            recommendations["Swing"] = "Strong Buy"
            recommendations["Short-Term"] = "Strong Buy"
            recommendations["Long-Term"] = "Strong Buy"
        elif net_score >= 2:
            recommendations["Intraday"] = "Buy"
            recommendations["Swing"] = "Buy"
            recommendations["Short-Term"] = "Buy"
            recommendations["Long-Term"] = "Buy"
        elif net_score <= -5:
            recommendations["Intraday"] = "Strong Sell"
            recommendations["Swing"] = "Strong Sell"
            recommendations["Short-Term"] = "Strong Sell"
            recommendations["Long-Term"] = "Strong Sell"
        elif net_score <= -2:
            recommendations["Intraday"] = "Sell"
            recommendations["Swing"] = "Sell"
            recommendations["Short-Term"] = "Sell"
            recommendations["Long-Term"] = "Sell"
        else:
            pass # Keep default "Hold"

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        # Using the scaled score from compute_signal_score
        recommendations["Score"] = round(score, 2) # Score is already scaled to -10 to 10 by compute_signal_score

        logging.info(f"Standard recommendations for {symbol}: {recommendations}")
        return recommendations

@st.cache_data(ttl=3600)
def get_top_sectors_cached(rate_limit_delay=0.2, stocks_per_sector=5):
    sector_scores = {}
    for sector, stocks in SECTORS.items():
        total_score = 0
        count = 0
        for symbol in stocks[:stocks_per_sector]:
            data = fetch_stock_data_cached(symbol)
            if data.empty:
                continue
            data = analyze_stock(data)
            score, _ = compute_signal_score(data, symbol) # Use the unified score
            if score is not None:
                total_score += score
                count += 1
            time.sleep(rate_limit_delay)
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
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
    
    # Ensure enough data for backtesting analysis
    min_data_for_backtest = max(INDICATOR_MIN_LENGTHS.values()) + 1 # At least one bar after indicators compute
    if len(data) < min_data_for_backtest:
        logging.warning(f"Not enough data to backtest {symbol}. Need at least {min_data_for_backtest} rows, got {len(data)}")
        return results

    # Pre-analyze the entire data once to avoid repeated analysis in loop
    full_analyzed_data = analyze_stock(data.copy()) # Use a copy to avoid modifying original

    for i in range(min_data_for_backtest, len(full_analyzed_data)):
        sliced_data = full_analyzed_data.iloc[:i+1] # Use already analyzed data
        
        # Ensure sliced_data is valid for recommendation logic
        if not validate_data(sliced_data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
            continue # Skip if data chunk is insufficient for robust recommendation

        current_price = sliced_data['Close'].iloc[-1]
        current_date = sliced_data.index[-1]

        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data, symbol=symbol)
            signal = rec["Recommendation"]
        else:
            rec = generate_recommendations(sliced_data, symbol)
            signal = rec[strategy] if strategy in rec else "Hold"
        
        if signal is None: # Handle cases where recommendation might be None
            continue

        if "Buy" in signal and position is None:
            position = "Long"
            entry_price = current_price # Or rec["Buy At"] if you want to use that specific level
            entry_date = current_date
            results["buy_signals"].append((current_date, current_price))
        
        # Exit logic: if a sell signal, or if a target/stop loss is hit (for more realistic backtest)
        elif position == "Long":
            # Check for explicit sell signal
            explicit_sell = "Sell" in signal
            
            # Check if stop loss or target is hit (for a more realistic backtest)
            # Note: This simple check assumes intra-day hit. For robustness, check High/Low.
            stop_loss_price = rec.get("Stop Loss")
            target_price = rec.get("Target")
            
            hit_stop_loss = False
            hit_target = False
            
            if stop_loss_price and current_price <= stop_loss_price:
                hit_stop_loss = True
            if target_price and current_price >= target_price:
                hit_target = True

            if explicit_sell or hit_stop_loss or hit_target:
                exit_price = current_price
                if hit_stop_loss:
                    exit_price = stop_loss_price # Exit at stop loss price
                elif hit_target:
                    exit_price = target_price # Exit at target price

                position = None
                profit = exit_price - entry_price
                
                # Only record valid returns to avoid issues if exit_price is None or entry_price is 0
                if entry_price != 0:
                    returns.append(profit / entry_price)
                    
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date, # Date when signal or condition was met
                    "exit_price": exit_price,
                    "profit": profit,
                    "reason": "Sell Signal" if explicit_sell else ("Stop Loss Hit" if hit_stop_loss else "Target Hit")
                })
                results["sell_signals"].append((current_date, exit_price))
                entry_price = 0
                entry_date = None
    
    # Calculate performance metrics
    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)
        
        # Calculate total return from compounded returns, not sum of percentages
        total_growth_factor = 1.0
        for r in returns:
            total_growth_factor *= (1 + r)
        results["total_return"] = (total_growth_factor - 1) * 100

        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100
        
        if returns:
            returns_series = pd.Series(returns)
            # Annualized return needs period, assuming daily data and 252 trading days
            results["annual_return"] = (returns_series.mean() * 252) * 100
            
            # Sharpe ratio
            if returns_series.std() != 0:
                results["sharpe_ratio"] = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                results["sharpe_ratio"] = 0 # No volatility, perfect Sharpe
            
            # Max Drawdown (equity curve based)
            equity_curve = pd.Series([1] + [1 + r for r in returns]).cumprod()
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            results["max_drawdown"] = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            results["annual_return"] = 0
            results["sharpe_ratio"] = 0
            results["max_drawdown"] = 0
    
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
            recommendation TEXT, -- For Adaptive mode
            regime TEXT,        -- For Adaptive mode
            position_size_shares REAL, -- For Adaptive mode
            position_size_value REAL,  -- For Adaptive mode
            trailing_stop REAL, -- For Adaptive mode
            reason TEXT,        -- For Adaptive mode
            pick_type TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.close()


def analyze_batch(stock_batch, progress_callback=None, status_callback=None):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    if "Error" in result: # Check for error key explicitly
                        errors.append(result["Error"])
                    else:
                        results.append(result)
                if status_callback:
                    status_callback(f"✅ Completed: {symbol}")
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                errors.append(error_msg)
                if status_callback:
                    status_callback(f"❌ Failed: {symbol}")
    if errors:
        st.error(f"Encountered {len(errors)} errors during batch processing:\n" + "\n".join(errors))
    return results

def analyze_stock_parallel(symbol):
    try:
        data = fetch_stock_data_with_dhan(symbol)
        
        if data.empty or len(data) < INDICATOR_MIN_LENGTHS['Ichimoku']:
            logging.warning(f"No sufficient data for {symbol} after fetch: {len(data)} rows")
            return None # Return None to indicate failure/insufficient data

        data = analyze_stock(data) # Analyze the data to add indicators
        
        # Re-check data validity AFTER analysis, as some indicators might result in NaNs
        if data.empty or pd.isna(data['Close'].iloc[-1]) or ('ATR' in data.columns and pd.isna(data['ATR'].iloc[-1])):
             logging.warning(f"Final analyzed data for {symbol} is incomplete (missing Close/ATR).")
             return None

        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(data, symbol)
            if not rec or not rec.get('Recommendation'):
                logging.error(f"Invalid adaptive_recommendation output for {symbol}: {rec}")
                return None
            
            position_size = rec.get("Position Size", {"shares": 0, "value": 0})
            
            return {
                "Symbol": symbol,
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Recommendation": rec.get("Recommendation", "Hold"),
                "Score": rec.get("Score", 0),
                "Regime": rec.get("Regime"),
                "Position Size Shares": position_size.get("shares", 0),
                "Position Size Value": position_size.get("value", 0),
                "Trailing Stop": rec.get("Trailing Stop"),
                "Reason": rec.get("Reason"),
                # For consistency with DB schema, add others as None
                "Intraday": None, "Swing": None, "Short-Term": None,
                "Long-Term": None, "Mean_Reversion": None, "Breakout": None,
                "Ichimoku_Trend": None
            }
        else:
            rec = generate_recommendations(data, symbol)
            if not rec or not rec.get('Intraday'): # Using Intraday as a proxy for valid standard recommendations
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
                # For consistency with DB schema, add others as None
                "Recommendation": None, "Regime": None,
                "Position Size Shares": None, "Position Size Value": None, "Trailing Stop": None,
                "Reason": None
            }
    except Exception as e:
        logging.error(f"Error in analyze_stock_parallel for {symbol}: {str(e)}")
        return None # Return None for any exception during analysis

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None, status_callback=None):
    results = []
    total_stocks = len(stock_list)
    processed = 0
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        if status_callback:
            batch_names = ", ".join(batch[:3])
            if len(batch) > 3:
                batch_names += f" and {len(batch)-3} more"
            status_callback(f"🔄 Analyzing: {batch_names}")
        batch_results = analyze_batch(batch, progress_callback=progress_callback, status_callback=status_callback)
        results.extend([r for r in batch_results if r is not None])
        processed += len(batch)
        if progress_callback:
            progress_callback(processed / total_stocks)
        # Add a delay between batches to reduce pressure on API limits, even with RateLimiter
        time.sleep(max(2, batch_size / 5)) # Example: 2 seconds min, or 0.2s per stock
        
    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    
    # Ensure all expected columns are present, even if empty/NaN
    expected_cols = [
        "Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Score",
        "Recommendation", "Regime", "Position Size Shares", "Position Size Value",
        "Trailing Stop", "Reason", "Intraday", "Swing", "Short-Term", "Long-Term",
        "Mean_Reversion", "Breakout", "Ichimoku_Trend"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan # Use NaN for numeric, None for object

    return results_df.sort_values(by="Score", ascending=False) # Do not limit to head(5) here, let display function handle it

    
def analyze_intraday_stocks(stock_list, batch_size=3, progress_callback=None, status_callback=None):
    results = []
    total_stocks = len(stock_list)
    processed = 0
    
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        
        # Update status for current batch
        if status_callback:
            batch_names = ", ".join(batch[:3])  # Show first 3 stocks
            if len(batch) > 3:
                batch_names += f" and {len(batch)-3} more"
            status_callback(f"🔄 Analyzing: {batch_names}")
        
        batch_results = analyze_batch(batch, progress_callback=progress_callback, status_callback=status_callback)
        results.extend([r for r in batch_results if r is not None])
        
        processed += len(batch)
        if progress_callback:
            progress_callback(processed / total_stocks)
        
        time.sleep(30) # Maintain a longer sleep for intraday-specific analysis if it's more frequent
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()
    
    # Ensure all expected columns are present, even if empty/NaN
    expected_cols = [
        "Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Score",
        "Recommendation", "Regime", "Position Size Shares", "Position Size Value",
        "Trailing Stop", "Reason", "Intraday", "Swing", "Short-Term", "Long-Term",
        "Mean_Reversion", "Breakout", "Ichimoku_Trend"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan

    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
        
    return results_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if recommendation is None or not isinstance(recommendation, str):
        return "⚪ N/A"
    if "Strong Buy" in recommendation:
        return f"🌟 {recommendation}"
    elif "Buy" in recommendation or "Consider Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Strong Sell" in recommendation:
        return f"💥 {recommendation}"
    elif "Sell" in recommendation or "Consider Sell" in recommendation:
        return f"🔴 {recommendation}"
    else:
        return f"⚪ {recommendation}"

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")
    
def update_progress_with_status(progress_bar, loading_text, status_text, progress_value, stock_status=None):
    progress_bar.progress(progress_value)
    
    # Calculate percentage
    percentage = int(progress_value * 100)
    
    # Update loading text with percentage
    loading_text.text(f"Progress: {percentage}%")
    
    # Update status text with current stock being analyzed
    if stock_status:
        status_text.text(stock_status)

def insert_top_picks_supabase(results_df, pick_type="daily"):
    # Filter for stocks with positive scores and valid buy/sell signals
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')

    if recommendation_mode == "Adaptive":
        # For adaptive, filter by general Recommendation containing "Buy" or "Strong Buy"
        filtered_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)].head(5)
    else:
        # For standard, filter by Intraday, Swing, Short-Term, or Long-Term containing "Buy"
        filtered_df = results_df[
            (results_df["Intraday"].str.contains("Buy", na=False)) |
            (results_df["Swing"].str.contains("Buy", na=False)) |
            (results_df["Short-Term"].str.contains("Buy", na=False)) |
            (results_df["Long-Term"].str.contains("Buy", na=False))
        ].sort_values(by="Score", ascending=False).head(5)


    for _, row in filtered_df.iterrows():
        data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "symbol": row.get('Symbol') or row.get('symbol', 'Unknown'),
            "score": row.get('Score', 0) or 0,
            "current_price": row.get('Current Price') or row.get('current_price', None),
            "buy_at": row.get('Buy At') or row.get('buy_at', None),
            "stop_loss": row.get('Stop Loss') or row.get('stop_loss', None),
            "target": row.get('Target') or row.get('target', None),
            "intraday": row.get('Intraday', 'Hold'),
            "swing": row.get('Swing', 'Hold'),
            "short_term": row.get('Short-Term', 'Hold'),
            "long_term": row.get('Long-Term', 'Hold'),
            "mean_reversion": row.get('Mean_Reversion', 'Hold'),
            "breakout": row.get('Breakout', 'Hold'),
            "ichimoku_trend": row.get('Ichimoku_Trend', 'Hold'),
            "recommendation": row.get('Recommendation') or row.get('recommendation', 'Hold'),
            "regime": row.get('Regime', 'Unknown'),
            "position_size_shares": row.get('Position Size Shares', None),
            "position_size_value": row.get('Position Size Value', None),
            "trailing_stop": row.get('Trailing Stop', None),
            "reason": row.get('Reason', 'No reason provided'),
            "pick_type": pick_type
        }
        logging.info(f"Inserting to Supabase: {data}")
        try:
            res = supabase.table("daily_picks").upsert(data).execute()
            if hasattr(res, "data") and res.data:
                logging.info(f"Supabase insert successful for {row.get('Symbol')}")
            elif hasattr(res, "error") and res.error:
                logging.error(f"Supabase insert error: {res.error}")
                st.error(f"Supabase insert error for {row.get('Symbol')}: {res.error['message']}")
            else:
                logging.warning(f"Supabase upsert response for {row.get('Symbol')} with no data or error key: {res}")

        except Exception as e:
            logging.error(f"Supabase insert exception for {row.get('Symbol')}: {e}")
            st.error(f"Supabase insert exception for {row.get('Symbol')}: {e}")


@RateLimiter(calls=1, period=1) # 1 call per second
def fetch_latest_price(symbol):
    # Optimizing this: Fetch only 2 days of data for current price
    data = fetch_stock_data_with_dhan(symbol, period="2d", interval="1d")
    if not data.empty and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
        return float(data['Close'].iloc[-1])
    return None

def update_with_latest_prices(df):
    updated_rows = []
    # Using ThreadPoolExecutor for concurrent price fetching
    with ThreadPoolExecutor(max_workers=5) as executor: # Limit concurrent API calls
        future_to_symbol = {executor.submit(fetch_latest_price, row['symbol']): idx for idx, row in df.iterrows()}
        for future in as_completed(future_to_symbol):
            idx = future_to_symbol[future]
            symbol = df.loc[idx, 'symbol']
            try:
                latest_close = future.result()
                if latest_close is not None:
                    df.loc[idx, 'current_price'] = latest_close
            except Exception as e:
                st.warning(f"Could not fetch latest price for {symbol}: {e}")
    return df # Return modified original DataFrame

        
def display_dashboard(symbol=None, data=None, recommendations=None):
    # Initialize session state
    if 'selected_sectors' not in st.session_state:
        st.session_state.selected_sectors = ["All"]
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
    if "show_history" not in st.session_state:
        st.session_state.show_history = False # Initialize for historical picks

    # Update session state if new data is provided (from Analyze Selected Stock button)
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
            top_sectors = get_top_sectors_cached(rate_limit_delay=10, stocks_per_sector=10)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            for name, score in top_sectors:
                st.markdown(f"- **{name}**: {score:.2f}/10") # Score is now out of 10

    # Daily top picks button
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        status_text = st.empty()
        
        # Show initial status
        status_text.text(f"📊 Analyzing {len(selected_stocks)} stocks...")
        
        results_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress_with_status(progress_bar, loading_text, status_text, x),
            status_callback=lambda status: status_text.text(status)
        )
        
        insert_top_picks_supabase(results_df, pick_type="daily") # This handles filtering to top 5 buys before insert
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()
        
        # Display the filtered top 5 buy picks based on the recommendation mode
        display_results_df = pd.DataFrame() # Initialize empty
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        
        if recommendation_mode == "Adaptive":
            display_results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)].sort_values(by="Score", ascending=False).head(5)
        else:
            display_results_df = results_df[
                (results_df["Intraday"].str.contains("Buy", na=False)) |
                (results_df["Swing"].str.contains("Buy", na=False)) |
                (results_df["Short-Term"].str.contains("Buy", na=False)) |
                (results_df["Long-Term"].str.contains("Buy", na=False))
            ].sort_values(by="Score", ascending=False).head(5)


        if not display_results_df.empty:
            st.subheader("🏆 Today's Top 5 Stocks")
            for _, row in display_results_df.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']:.2f}/10"): # Score is now out of 10
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    
                    if st.session_state.recommendation_mode == "Adaptive":
                        pos_shares = row.get('Position Size Shares', 'N/A')
                        pos_value = row.get('Position Size Value', 'N/A')
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size: {pos_shares} shares (~₹{pos_value})  
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
            st.warning("⚠️ No top picks available (or no 'Buy' recommendations) due to data issues or current market conditions.")

    # Intraday top picks button
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        
        status_text = st.empty() # For status updates in analyze_intraday_stocks
        
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages),
            status_callback=lambda status: status_text.text(status)
        )
        
        insert_top_picks_supabase(intraday_results, pick_type="intraday") # This handles filtering to top 5 buys before insert
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()
        
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']:.2f}/10"): # Score is now out of 10
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    if st.session_state.recommendation_mode == "Adaptive":
                        pos_shares = row.get('Position Size Shares', 'N/A')
                        pos_value = row.get('Position Size Value', 'N/A')
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size: {pos_shares} shares (~₹{pos_value})  
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
            st.warning("⚠️ No intraday picks available (or no 'Buy' recommendations) due to data issues.")

    # Historical picks button
    if st.button("📜 View Historical Picks"):
        st.session_state.show_history = not st.session_state.show_history

    if st.session_state.show_history:
        st.markdown("### 📜 Historical Picks")
        if st.button("Close Historical Picks"):
            st.session_state.show_history = False
            st.rerun() # Use st.rerun

        try:
            # Fetch all distinct dates first
            res = supabase.table("daily_picks").select("date").order("date", desc=True).execute()
            if res.data:
                all_dates = sorted({row['date'] for row in res.data}, reverse=True)
                if not all_dates:
                    st.warning("No historical picks found in the database.")
                    return

                date_filter = st.selectbox("Select Date", all_dates, key="history_date")
                
                # Fetch picks for the selected date
                res2 = supabase.table("daily_picks").select("*").eq("date", date_filter).execute()
                if res2.data:
                    df = pd.DataFrame(res2.data)
                    
                    # Ensure numeric columns are actually numeric
                    numeric_cols = ['score', 'current_price', 'buy_at', 'target', 'stop_loss', 'position_size_shares', 'position_size_value', 'trailing_stop']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                    df = update_with_latest_prices(df) # Update prices for selected day's picks
                    df = add_action_and_change(df)
                    
                    # Format for display
                    for col in ['buy_at', 'current_price', 'target', 'stop_loss', 'position_size_value', 'trailing_stop', '% Change']:
                        if col in df.columns:
                            df[col] = df[col].map(lambda x: f"₹{x:.2f}" if pd.notnull(x) else 'N/A')
                    if 'position_size_shares' in df.columns:
                         df['position_size_shares'] = df['position_size_shares'].map(lambda x: f"{int(x)}" if pd.notnull(x) else 'N/A')

                    display_cols = [
                        "symbol", "buy_at", "current_price", "% Change", "What to do now?",
                        "recommendation", "regime", "position_size_shares", "position_size_value",
                        "trailing_stop", "reason", "target", "stop_loss", "pick_type"
                    ]
                    
                    # Filter columns to display based on what's actually in df
                    display_cols_present = [col for col in display_cols if col in df.columns]

                    # Filter for adaptive mode specific columns if adaptive was used for that date
                    # Heuristic: if 'recommendation' column has non-null values, assume adaptive was used.
                    if df['recommendation'].notna().any() and df['recommendation'].isin(['Strong Buy (Trend Continuation)', 'Buy (Trend Following)', 'Strong Buy (Volatile Mean Reversion)', 'Buy (Volatile Opportunity)', 'Buy (Consolidation Breakout Potential)', 'Buy (Range Low)', 'Buy (Developing Bullish)', 'Consider Buy', 'Strong Sell (Trend Continuation)', 'Sell (Trend Following)', 'Strong Sell (Volatile Mean Reversion)', 'Sell (Volatile Risk)', 'Sell (Consolidation Breakout Risk)', 'Sell (Range High)', 'Sell (Developing Bearish)', 'Consider Sell']).any():
                        final_display_df = df[display_cols_present]
                    else: # Standard picks, hide adaptive columns
                        standard_cols = ["symbol", "buy_at", "current_price", "% Change", "What to do now?", "intraday", "swing", "short_term", "long_term", "mean_reversion", "breakout", "ichimoku_trend", "target", "stop_loss", "pick_type"]
                        final_display_df = df[[col for col in standard_cols if col in df.columns]]
                        
                    styled_df = style_picks_df(final_display_df)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.warning("No picks found for the selected date in Supabase.")
            else:
                st.warning("No historical dates found in Supabase.")
        except Exception as e:
            st.error(f"Error fetching historical picks: {e}")
            logging.error(f"Error fetching historical picks: {e}")
        
    # Display stock analysis if symbol is available
    if st.session_state.symbol and st.session_state.data is not None and st.session_state.recommendations is not None:
        symbol = st.session_state.symbol
        data = st.session_state.data
        recommendations = st.session_state.recommendations

        st.header(f"📋 {normalize_symbol_dhan(symbol)} Analysis") # Use normalize symbol for display

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            current_price = recommendations.get('Current Price', 'N/A')
            st.metric("Current Price", f"₹{current_price}")
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
            pos_shares = recommendations.get('Position Size', {}).get('shares', 'N/A')
            pos_value = recommendations.get('Position Size', {}).get('value', 'N/A')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Recommendation**: {colored_recommendation(recommendations.get('Recommendation', 'N/A'))}")
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10") # Score is out of 10
            with col2:
                st.write(f"**Position Size**: {pos_shares} shares")
                st.write(f"**Value**: ₹{pos_value}")
            with col3:
                st.write(f"**Trailing Stop**: ₹{recommendations.get('Trailing Stop', 'N/A')}")
                st.write(f"**Volatility**: {assess_risk(data)}")
            st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")

        else: # Standard Mode
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
            st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10") # Score is now out of 10
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
                with st.spinner(f"Running {strategy} Strategy backtest... (This may take a while for large data sets)"):
                    # Pass a hash of the data to cache_data to invalidate if data changes
                    data_for_hash = data.tail(300) # Only hash a recent portion for cache invalidation
                    data_hash = hash(data_for_hash.to_string())
                    backtest_results = backtest_stock(data, symbol, strategy=strategy, _data_hash=data_hash)
                    if strategy == "Swing":
                        st.session_state.backtest_results_swing = backtest_results
                    else:
                        st.session_state.backtest_results_intraday = backtest_results

        # Backtest results display
        for strategy_name, results_key in [("Swing", "backtest_results_swing"), ("Intraday", "backtest_results_intraday")]:
            backtest_results = st.session_state.get(results_key)
            if backtest_results and backtest_results['trades'] > 0: # Only display if backtest ran and had trades
                st.subheader(f"📈 Backtest Results ({strategy_name} Strategy)")
                st.write(f"**Total Return**: {backtest_results['total_return']:.2f}%")
                st.write(f"**Annualized Return**: {backtest_results['annual_return']:.2f}%")
                st.write(f"**Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}")
                st.write(f"**Max Drawdown**: {backtest_results['max_drawdown']:.2f}%")
                st.write(f"**Number of Trades**: {backtest_results['trades']}")
                st.write(f"**Win Rate**: {backtest_results['win_rate']:.2f}%")
                with st.expander("Trade Details"):
                    for trade in backtest_results["trade_details"]:
                        profit = trade.get("profit", 0)
                        st.write(f"Entry: {trade['entry_date'].strftime('%Y-%m-%d')} @ ₹{trade['entry_price']:.2f}, "
                                 f"Exit: {trade['exit_date'].strftime('%Y-%m-%d')} @ ₹{trade['exit_price']:.2f}, "
                                 f"Profit: ₹{profit:.2f} ({trade['reason']})")

                fig = px.line(data, x=data.index, y='Close', title=f"{normalize_symbol_dhan(symbol)} Price with Signals")
                if backtest_results["buy_signals"]:
                    buy_dates, buy_prices = zip(*backtest_results["buy_signals"])
                    fig.add_scatter(x=list(buy_dates), y=list(buy_prices), mode='markers', name='Buy Signals',
                                   marker=dict(color='green', symbol='triangle-up', size=10))
                if backtest_results["sell_signals"]:
                    sell_dates, sell_prices = zip(*backtest_results["sell_signals"])
                    fig.add_scatter(x=list(sell_dates), y=list(sell_prices), mode='markers', name='Sell Signals',
                                   marker=dict(color='red', symbol='triangle-down', size=10))
                st.plotly_chart(fig, use_container_width=True)
            elif backtest_results: # Backtest ran but no trades or invalid results
                st.info(f"No valid trades generated for {strategy_name} strategy on {normalize_symbol_dhan(symbol)} with available data.")


        # Technical Indicators
        st.subheader(f"📊 Technical Indicators (Latest Values for {normalize_symbol_dhan(symbol)})")
        indicators_to_display = [
            ("RSI", 'RSI', TOOLTIPS['RSI']),
            ("MACD", 'MACD', TOOLTIPS['MACD']),
            ("MACD Signal", 'MACD_signal', TOOLTIPS['MACD']),
            ("MACD Hist", 'MACD_hist', TOOLTIPS['MACD']),
            ("ATR", 'ATR', TOOLTIPS['ATR']),
            ("ADX", 'ADX', TOOLTIPS['ADX']),
            ("DMI+ (DMP)", 'DMP', "Positive Directional Movement Index - indicates upward pressure"),
            ("DMI- (DMN)", 'DMN', "Negative Directional Movement Index - indicates downward pressure"),
            ("ADX Slope (5-period)", 'ADX_Slope', "Slope of ADX over last 5 periods (positive means trend strengthening, negative means weakening)"),
            ("Bollinger Upper", 'Upper_Band', TOOLTIPS['Bollinger']),
            ("Bollinger Middle", 'Middle_Band', TOOLTIPS['Bollinger']),
            ("Bollinger Lower", 'Lower_Band', TOOLTIPS['Bollinger']),
            ("Ichimoku Tenkan", 'Ichimoku_Tenkan', TOOLTIPS['Ichimoku']),
            ("Ichimoku Kijun", 'Ichimoku_Kijun', TOOLTIPS['Ichimoku']),
            ("Ichimoku Span A", 'Ichimoku_Span_A', TOOLTIPS['Ichimoku']),
            ("Ichimoku Span B", 'Ichimoku_Span_B', TOOLTIPS['Ichimoku']),
            ("CMF", 'CMF', TOOLTIPS['CMF']),
            ("OBV", 'OBV', TOOLTIPS['OBV']),
            ("CMO", 'CMO', TOOLTIPS['CMO']),
            ("TRIX", 'TRIX', TOOLTIPS['TRIX']),
            ("Ultimate Oscillator", 'Ultimate_Osc', TOOLTIPS['Ultimate_Osc']),
            ("VPT", 'VPT', TOOLTIPS['VPT'])
        ]

        col1, col2 = st.columns(2)
        current_col = col1
        for i, (display_name, col_name, tooltip_text) in enumerate(indicators_to_display):
            if i % 2 == 0:
                current_col = col1
            else:
                current_col = col2

            if col_name in data.columns and pd.notnull(data[col_name].iloc[-1]):
                value = data[col_name].iloc[-1]
                value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                current_col.write(f"**{tooltip(display_name, tooltip_text)}**: {value_str}")
            else:
                current_col.write(f"**{tooltip(display_name, tooltip_text)}**: N/A")


        # Price Chart
        st.subheader(f"📈 Price Chart with Indicators for {normalize_symbol_dhan(symbol)}")
        fig = px.line(data, x=data.index, y='Close', title=f"{normalize_symbol_dhan(symbol)} Price")
        if 'SMA_50' in data.columns and data['SMA_50'].notnull().any():
            fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange'))
        if 'SMA_200' in data.columns and data['SMA_200'].notnull().any():
            fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red'))
        if 'Upper_Band' in data.columns and data['Upper_Band'].notnull().any():
            fig.add_scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Bollinger Upper', line=dict(color='green', dash='dash'))
        if 'Lower_Band' in data.columns and data['Lower_Band'].notnull().any():
            fig.add_scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Bollinger Lower', line=dict(color='green', dash='dash'))
        if 'Ichimoku_Span_A' in data.columns and data['Ichimoku_Span_A'].notnull().any():
            fig.add_scatter(x=data.index, y=data['Ichimoku_Span_A'], mode='lines', name='Ichimoku Span A', line=dict(color='purple'))
        if 'Ichimoku_Span_B' in data.columns and data['Ichimoku_Span_B'].notnull().any():
            fig.add_scatter(x=data.index, y=data['Ichimoku_Span_B'], mode='lines', name='Ichimoku Span B', line=dict(color='purple', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

        # Monte Carlo Simulation
        st.subheader(f"📊 Monte Carlo Price Projections for {normalize_symbol_dhan(symbol)}")
        simulations = monte_carlo_simulation(data)
        if simulations:
            sim_df = pd.DataFrame(simulations).T
            sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
            fig_sim = px.line(sim_df, title="Monte Carlo Price Projections (30 Days)")
            fig_sim.update_layout(showlegend=False) # Hide legend for clarity on many lines
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Monte Carlo simulation could not be performed due to insufficient data or errors.")

        # RSI and MACD
        st.subheader("📊 RSI and MACD")
        
        if 'RSI' in data.columns and data['RSI'].notnull().any():
            fig_ind = px.line(data, x=data.index, y='RSI', title="RSI")
            fig_ind.add_hline(y=70, line_dash="dash", line_color="red")
            fig_ind.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("RSI data not available.")

        if 'MACD' in data.columns and data['MACD'].notnull().any():
            fig_macd = px.line(data, x=data.index, y=['MACD', 'MACD_signal'], title="MACD")
            fig_macd.add_bar(x=data.index, y=data['MACD_hist'], name='MACD Histogram', marker_color='grey')
            st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.info("MACD data not available.")

        # Volume Analysis
        st.subheader("📊 Volume Analysis")
        if 'Volume' in data.columns and data['Volume'].notnull().any():
            fig_vol = px.bar(data, x=data.index, y='Volume', title="Volume")
            if 'Volume_Spike' in data.columns:
                spike_data = data[data['Volume_Spike'] == True]
                if not spike_data.empty:
                    fig_vol.add_scatter(x=spike_data.index, y=spike_data['Volume'], mode='markers', name='Volume Spike',
                                    marker=dict(color='red', size=10))
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Volume data not available.")
            
def main():
    init_database()
    
    # Test Dhan connection first
    if st.sidebar.button("Test Dhan Connection"):
        with st.spinner("Testing Dhan API connection..."):
            if test_dhan_connection():
                st.success("✅ Dhan API is working!")
            else:
                st.error("❌ Dhan API connection failed. Check your credentials and network.")
                
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
            with st.spinner(f"Loading and analyzing data for {normalize_symbol_dhan(symbol)}..."):
                data = fetch_stock_data_with_dhan(symbol)
                if not data.empty and len(data) >= INDICATOR_MIN_LENGTHS['Ichimoku']: # Check length here again
                    data = analyze_stock(data) # This will fill NaNs for non-computable indicators
                    
                    # Check if analysis resulted in usable data (e.g., Close/ATR not NaN)
                    if data.empty or pd.isna(data['Close'].iloc[-1]) or ('ATR' in data.columns and pd.isna(data['ATR'].iloc[-1])):
                        st.warning(f"⚠️ Could not complete analysis for {normalize_symbol_dhan(symbol)} due to insufficient or invalid data after indicator computation.")
                        st.session_state.symbol = None # Reset to prevent displaying incomplete data
                        return
                    
                    if recommendation_mode == "Adaptive":
                        recommendations = adaptive_recommendation(data, symbol)
                    else:
                        recommendations = generate_recommendations(data, symbol)
                        
                    st.session_state.symbol = symbol
                    st.session_state.data = data
                    st.session_state.recommendations = recommendations
                    st.session_state.backtest_results_swing = None
                    st.session_state.backtest_results_intraday = None
                    display_dashboard(symbol, data, recommendations) # Pass data to display function directly
                else:
                    st.warning(f"⚠️ No sufficient historical data available for {normalize_symbol_dhan(symbol)} to perform a full analysis ({len(data)} rows found, need at least {INDICATOR_MIN_LENGTHS['Ichimoku']}).")
                    st.session_state.symbol = None # Reset to prevent displaying incomplete data
    else:
        # Initial display or after a rerun without explicit analysis button click
        display_dashboard()

if __name__ == "__main__":
    main()
