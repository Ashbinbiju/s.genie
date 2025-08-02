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
import itertools
from arch import arch_model
import warnings
import sqlite3
from diskcache import Cache
import os
from dotenv import load_dotenv
from streamlit import cache_data
from supabase import create_client, Client
import json
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ratelimit import RateLimitDecorator as RateLimiter
from threading import Lock
data_api_calls = 0
data_api_lock = Lock()
# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#load_dotenv()

DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]

def dhan_headers():
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,  # Note: Use "client-id" not "dhan-client-id"
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
            row = df[df[column].astype(str).str.upper() == symbol_clean]
            if not row.empty:
                security_id = row.iloc[0]['SEM_SMST_SECURITY_ID']
                logging.info(f"Found security_id for {symbol}: {security_id} in column {column}")
                return security_id

    # Log all possible matches for debugging
    possible_matches = df[df['SM_SYMBOL_NAME'].str.contains(symbol_clean, case=False, na=False)]
    logging.warning(f"No security_id found for {symbol} ({symbol_clean}). Possible matches: {possible_matches[['SM_SYMBOL_NAME', 'SEM_TRADING_SYMBOL']].to_dict()}")
    st.warning(f"NOT FOUND: {symbol} ({symbol_clean})")
    return None


# Add a test function to verify Dhan connection
def test_dhan_connection():
    """Test if Dhan API credentials are working"""
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    
    # Test with a known security ID (e.g., for RELIANCE)
    payload = {
        "securityId": "1333",  # RELIANCE security ID
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            print("✅ Dhan API connection successful!")
            return True
        else:
            print(f"❌ Dhan API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def normalize_symbol_dhan(symbol):
    # Remove .NS, .BO, and -EQ
    return symbol.replace(".NS", "").replace(".BO", "").replace("-EQ", "")
    
# --- Supabase setup ---
SUPABASE_URL = "https://uwnqchncwvcmvoyalwkt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV3bnFjaG5jd3ZjbXZveWFsd2t0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI5OTE2MDIsImV4cCI6MjA2ODU2NzYwMn0.tADeLmGgiuG3dXJlweNRk8_lmMOcPBYAo6sCxcmaqMs"
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
        
load_dotenv()


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
        
@RateLimiter(calls=5, period=1)
def fetch_stock_data_with_dhan(symbol, period="5y", interval="1d"):
    global data_api_calls
    with data_api_lock:
        if data_api_calls >= 90000:
            st.warning(f"⚠️ Approaching Data API daily limit: {data_api_calls}/100000")
        if data_api_calls >= 100000:
            st.error("⚠️ Reached daily Data API limit of 100,000 requests.")
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

        # Log the full response for debugging
        logging.debug(f"API response for {symbol}: {data}")

        # --- NEW LOGIC STARTS HERE ---
        api_data = data.get("data", data)

        # If api_data is a dict with lists (columnar), convert to list of dicts (row format)
        if isinstance(api_data, dict) and all(isinstance(v, list) for v in api_data.values()):
            n = len(next(iter(api_data.values())))
            row_data = [
                {k: api_data[k][i] for k in api_data}
                for i in range(n)
            ]
            df = pd.DataFrame(row_data)
        elif isinstance(api_data, list):
            df = pd.DataFrame(api_data)
        else:
            logging.warning(f"Unknown data format for {symbol}: {type(api_data)}")
            return pd.DataFrame()
        # --- NEW LOGIC ENDS HERE ---

        if df.empty:
            logging.warning(f"Empty DataFrame created for {symbol}. Response data: {api_data}")
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
            logging.error(f"Missing required columns for {symbol}: {missing_columns}")
            return pd.DataFrame()

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
       
@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    return fetch_stock_data_with_dhan(symbol, period, interval)

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
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
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
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] is not None and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_volatility < 0.02:
            score += 0.5
        elif atr_volatility > 0.05:
            score -= 0.5
    return min(max(score / 3.5, 0), 1)

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    else:
        return "Low Volatility"

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
    rsi = data['RSI']
    price = data['Close']
    recent_highs = price[-5:].idxmax()
    recent_lows = price[-5:].idxmin()
    rsi_highs = rsi[-5:].idxmax()
    rsi_lows = rsi[-5:].idxmin()
    bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1])
    bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

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
    Checks if data is sufficient to compute the specified indicator.
    """
    min_lengths = {
        'RSI': 14,
        'MACD': 26,
        'Ichimoku': 52,
        'ATR': 14,
        'ADX': 14,
        'Bollinger': 20,
        'Stochastic': 14,
        'CMF': 20,
        'OBV': 1
    }
    required_length = min_lengths.get(indicator, 50)
    return validate_data(data, min_length=required_length)


logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s: %(message)s")

def validate_data(data, min_length=50):
    """
    Validates that the DataFrame has required columns, sufficient rows, and numeric data.
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not isinstance(data, pd.DataFrame) or data.empty:
        logging.error("Data is not a DataFrame or is empty")
        return False
    if not all(col in data.columns for col in required_columns):
        logging.error(f"Missing required columns: {set(required_columns) - set(data.columns)}")
        return False
    if len(data) < min_length:
        logging.error(f"Insufficient data rows: {len(data)} < {min_length}")
        return False
    if data[required_columns].isna().any().any():
        logging.warning("Data contains NaN values in required columns")
        return False
    if not data[required_columns].apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
        logging.error("Non-numeric values found in required columns")
        return False
    return True


def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    """
    # Define columns list at function scope to avoid UnboundLocalError
    columns = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
        'VWAP', 'Avg_Volume', 'Volume_Spike', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2',
        'Fib_50.0', 'Fib_61.8', 'Divergence', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF', 'Donchian_Upper',
        'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle', 'Keltner_Lower',
        'TRIX', 'Ultimate_Osc', 'CMO', 'VPT'
    ]

    if not validate_data(data, min_length=50):
        logging.warning("Data validation failed in analyze_stock")
        for col in columns:
            data[col] = None
        return data

    # Ensure data types are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with any NaN values in required columns
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    if len(data) < 50:
        logging.warning(f"Insufficient data after cleaning: {len(data)} rows")
        for col in columns:
            data[col] = None
        return data

    try:
        # RSI
        if can_compute_indicator(data, 'RSI'):
            data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
            logging.info(f"RSI computed: {data['RSI'].iloc[-1]}")
        else:
            data['RSI'] = None

        # MACD
        if can_compute_indicator(data, 'MACD'):
            macd = MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()
            logging.info(f"MACD computed: {data['MACD'].iloc[-1]}")
        else:
            data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None

        # SMA
        if can_compute_indicator(data, 'SMA'):
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            logging.info(f"SMA computed: SMA_50={data['SMA_50'].iloc[-1]}, SMA_200={data['SMA_200'].iloc[-1]}")
        else:
            data['SMA_50'] = data['SMA_200'] = None

        # EMA
        if can_compute_indicator(data, 'EMA'):
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
            logging.info(f"EMA computed: EMA_20={data['EMA_20'].iloc[-1]}")
        else:
            data['EMA_20'] = data['EMA_50'] = None

        # Bollinger Bands
        if can_compute_indicator(data, 'Bollinger'):
            bb = BollingerBands(data['Close'], window=20, window_dev=2)
            data['Upper_Band'] = bb.bollinger_hband()
            data['Middle_Band'] = bb.bollinger_mavg()
            data['Lower_Band'] = bb.bollinger_lband()
            logging.info(f"Bollinger Bands computed: Upper={data['Upper_Band'].iloc[-1]}")
        else:
            data['Upper_Band'] = data['Middle_Band'] = data['Lower_Band'] = None

        # Stochastic Oscillator
        if can_compute_indicator(data, 'Stochastic'):
            stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()
            logging.info(f"Stochastic computed: SlowK={data['SlowK'].iloc[-1]}")
        else:
            data['SlowK'] = data['SlowD'] = None

        # ATR
        if can_compute_indicator(data, 'ATR'):
            data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
            logging.info(f"ATR computed: {data['ATR'].iloc[-1]}")
        else:
            data['ATR'] = None

        # ADX
        if can_compute_indicator(data, 'ADX'):
            data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
            logging.info(f"ADX computed: {data['ADX'].iloc[-1]}")
        else:
            data['ADX'] = None

        # OBV
        if can_compute_indicator(data, 'OBV'):
            data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            logging.info(f"OBV computed: {data['OBV'].iloc[-1]}")
        else:
            data['OBV'] = None

        # Ichimoku Cloud
        if can_compute_indicator(data, 'Ichimoku'):
            ichimoku = IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            # Compute Chikou Span manually (shift Close price backward by 26 periods)
            data['Ichimoku_Chikou'] = data['Close'].shift(-26)
            logging.info(f"Ichimoku computed: Span_A={data['Ichimoku_Span_A'].iloc[-1]}, Chikou={data['Ichimoku_Chikou'].iloc[-1] if pd.notnull(data['Ichimoku_Chikou'].iloc[-1]) else 'None'}")
        else:
            data['Ichimoku_Tenkan'] = data['Ichimoku_Kijun'] = data['Ichimoku_Span_A'] = data['Ichimoku_Span_B'] = data['Ichimoku_Chikou'] = None

        # CMF
        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
            logging.info(f"CMF computed: {data['CMF'].iloc[-1]}")
        else:
            data['CMF'] = None

        # Volume Spike and Avg Volume
        data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Spike'] = data['Volume'] > data['Avg_Volume'] * 2

        # Initialize remaining columns as None
        for col in ['VWAP', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Divergence',
                    'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle',
                    'Keltner_Lower', 'TRIX', 'Ultimate_Osc', 'CMO', 'VPT']:
            data[col] = None

        return data

    except Exception as e:
        logging.error(f"Error in analyze_stock: {str(e)}")
        for col in columns:
            data[col] = None
        return data
    
def calculate_buy_at(data):
    """
    Calculates the buy price based on recent price action and indicators.
    """
    if not validate_data(data, min_length=10):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        buy_at = close * (1 - 0.01 * atr / close) if atr > 0 else close * 0.99
        return round(float(buy_at), 2)
    except Exception as e:
        logging.error(f"Error calculating buy_at: {str(e)}")
        return None

def calculate_stop_loss(data):
    """
    Calculates the stop loss price based on ATR and recent lows.
    """
    if not validate_data(data, min_length=10):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        stop_loss = close - 2 * atr if atr > 0 else close * 0.95
        return round(float(stop_loss), 2)
    except Exception as e:
        logging.error(f"Error calculating stop_loss: {str(e)}")
        return None


def calculate_target(data):
    """
    Calculates the target price based on ATR and recent highs.
    """
    if not validate_data(data, min_length=10):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        target = close + 3 * atr if atr > 0 else close * 1.05
        return round(float(target), 2)
    except Exception as e:
        logging.error(f"Error calculating target: {str(e)}")
        return None

def calculate_buy_at_row(row):
    if pd.notnull(row['RSI']) and row['RSI'] < 30:
        return round(row['Close'] * 0.99, 2)
    return round(row['Close'], 2)

def calculate_stop_loss_row(row, atr_multiplier=2.5):
    if pd.notnull(row['ATR']):
        atr_multiplier = 3.0 if pd.notnull(row['ADX']) and row['ADX'] > 25 else 1.5
        stop_loss = row['Close'] - (atr_multiplier * row['ATR'])
        if stop_loss < row['Close'] * 0.9:
            stop_loss = row['Close'] * 0.9
        return round(stop_loss, 2)
    return None

def calculate_target_row(row, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss_row(row)
    if stop_loss is not None:
        risk = row['Close'] - stop_loss
        adjusted_ratio = min(risk_reward_ratio, 5) if pd.notnull(row['ADX']) and row['ADX'] > 25 else min(risk_reward_ratio, 3)
        target = row['Close'] + (risk * adjusted_ratio)
        if target > row['Close'] * 1.2:
            target = row['Close'] * 1.2
        return round(target, 2)
    return None

# Improved strategy logic using adaptive regime detection, signal scoring, and volatility-aware filters

def classify_market_regime(data):
    """Classifies regime based on volatility and trend"""
    data['ATR_pct'] = data['ATR'] / data['Close']
    if data['ATR_pct'].iloc[-1] > 0.03:
        return 'volatile'
    elif data['Close'].iloc[-1] > data['SMA_50'].iloc[-1]:
        return 'bullish'
    else:
        return 'neutral'

def compute_signal_score(data, symbol=None):
    """
    Computes a weighted score based on normalized technical and fundamental indicators.
    Returns a score between -10 and 10, with negative scores indicating no trade.
    """
    score = 0.0
    weights = {
        'RSI': 1.5,
        'MACD': 1.2,
        'Ichimoku': 1.5,
        'CMF': 0.5,
        'ATR_Volatility': 1.0,
        'Breakout': 1.2,
        'Fundamentals': 1.0
    }

    if not validate_data(data, min_length=50):
        logging.warning(f"Invalid data for scoring: {symbol}")
        return -10

    # Volume filter
    if 'Avg_Volume' in data.columns and pd.notnull(data['Avg_Volume'].iloc[-1]) and data['Volume'].iloc[-1] < data['Avg_Volume'].iloc[-1] * 0.5:
        logging.info(f"Low volume for {symbol}, skipping scoring")
        return -10

    # RSI
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) and isinstance(data['RSI'].iloc[-1], (int, float)):
        rsi = data['RSI'].iloc[-1]
        rsi_normalized = (rsi - 50) / 50
        if rsi < 30:
            score += weights['RSI'] * max(rsi_normalized, -1)
        elif rsi > 70:
            score -= weights['RSI'] * min(rsi_normalized, 1)
        logging.info(f"RSI contribution for {symbol}: {score}")

    # MACD
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
        macd_normalized = macd_diff / (data['MACD'].std() + 1e-10)
        if macd_diff > 0:
            score += weights['MACD'] * max(macd_normalized, 0)
        else:
            score -= weights['MACD'] * min(macd_normalized, 0)
        logging.info(f"MACD contribution for {symbol}: {score}")

    # Ichimoku
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
        close = data['Close'].iloc[-1]
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        span_b = data['Ichimoku_Span_B'].iloc[-1]
        if close > span_a and close > span_b:
            score += weights['Ichimoku'] * 1
        elif close < span_a and close < span_b:
            score -= weights['Ichimoku'] * 1
        logging.info(f"Ichimoku contribution for {symbol}: {score}")

    # CMF
    if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        if cmf > 0:
            score += weights['CMF'] * cmf
        elif cmf < 0:
            score -= weights['CMF'] * abs(cmf)
        logging.info(f"CMF contribution for {symbol}: {score}")

    # ATR Volatility
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        atr = data['ATR'].iloc[-1]
        atr_normalized = atr / data['Close'].iloc[-1]
        score += weights['ATR_Volatility'] * min(atr_normalized, 0.5)
        logging.info(f"ATR contribution for {symbol}: {score}")

    # Breakout (using Bollinger Bands)
    if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]):
        close = data['Close'].iloc[-1]
        if close > data['Upper_Band'].iloc[-1]:
            score += weights['Breakout'] * 1
        elif close < data['Lower_Band'].iloc[-1]:
            score -= weights['Breakout'] * 1
        logging.info(f"Breakout contribution for {symbol}: {score}")

    logging.info(f"Final score for {symbol}: {score}")
    return min(max(score, -10), 10)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def adaptive_recommendation(data, market_regime="normal"):
    """
    Generates adaptive trading recommendations based on market regime and technical indicators.
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
        "Score": 0,
        "Regime": market_regime
    }

    if not validate_data(data, min_length=27):
        logging.warning(f"Invalid data for adaptive recommendations")
        return recommendations

    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        buy_score = 0
        sell_score = 0

        # Adjust weights based on market regime
        rsi_weight = 2.0 if market_regime == "volatile" else 1.5
        macd_weight = 1.5 if market_regime == "trending" else 1.2
        ichimoku_weight = 2.0 if market_regime == "trending" else 1.5

        # RSI
        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) and isinstance(data['RSI'].iloc[-1], (int, float)):
            rsi = data['RSI'].iloc[-1]
            if market_regime == "volatile":
                if rsi <= 15:
                    buy_score += 4 * rsi_weight
                    recommendations["Mean_Reversion"] = "Strong Buy"
                elif rsi < 25:
                    buy_score += 2 * rsi_weight
                    recommendations["Mean_Reversion"] = "Buy"
                elif rsi > 75:
                    sell_score += 2 * rsi_weight
                    recommendations["Mean_Reversion"] = "Sell"
            else:
                if rsi <= 20:
                    buy_score += 4 * rsi_weight
                    recommendations["Mean_Reversion"] = "Buy"
                elif rsi < 30:
                    buy_score += 2 * rsi_weight
                    recommendations["Mean_Reversion"] = "Buy"
                elif rsi > 70:
                    sell_score += 2 * rsi_weight
                    recommendations["Mean_Reversion"] = "Sell"
            logging.info(f"Adaptive RSI score: buy={buy_score}, sell={sell_score}")

        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
            if market_regime == "trending" and macd_diff > 0:
                buy_score += 3 * macd_weight
                recommendations["Swing"] = "Buy"
            elif macd_diff > 0:
                buy_score += 2 * macd_weight
                recommendations["Swing"] = "Buy"
            elif macd_diff < 0:
                sell_score += 2 * macd_weight
                recommendations["Swing"] = "Sell"
            logging.info(f"Adaptive MACD score: buy={buy_score}, sell={sell_score}")

        # Ichimoku
        if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
            close = data['Close'].iloc[-1]
            span_a = data['Ichimoku_Span_A'].iloc[-1]
            span_b = data['Ichimoku_Span_B'].iloc[-1]
            if market_regime == "trending" and close > span_a and close > span_b:
                buy_score += 4 * ichimoku_weight
                recommendations["Ichimoku_Trend"] = "Strong Buy"
            elif close > span_a and close > span_b:
                buy_score += 3 * ichimoku_weight
                recommendations["Ichimoku_Trend"] = "Buy"
            elif close < span_a and close < span_b:
                sell_score += 3 * ichimoku_weight
                recommendations["Ichimoku_Trend"] = "Sell"
            logging.info(f"Adaptive Ichimoku score: buy={buy_score}, sell={sell_score}")

        # Bollinger Bands Breakout
        if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]):
            close = data['Close'].iloc[-1]
            if market_regime == "volatile" and close > data['Upper_Band'].iloc[-1]:
                buy_score += 3
                recommendations["Breakout"] = "Buy"
            elif close > data['Upper_Band'].iloc[-1]:
                buy_score += 2
                recommendations["Breakout"] = "Buy"
            elif close < data['Lower_Band'].iloc[-1]:
                sell_score += 2
                recommendations["Breakout"] = "Sell"
            logging.info(f"Adaptive Breakout score: buy={buy_score}, sell={sell_score}")

        # Generate recommendations based on scores
        net_score = buy_score - sell_score
        if buy_score > sell_score and buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy" if market_regime == "volatile" else "Buy"
            recommendations["Swing"] = "Buy" if buy_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Buy" if buy_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Buy" if buy_score >= 1 else "Hold"
        elif sell_score > buy_score and sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell" if market_regime == "volatile" else "Sell"
            recommendations["Swing"] = "Sell" if sell_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Sell" if sell_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Sell" if sell_score >= 1 else "Hold"

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = min(max(net_score, -7), 7)

        logging.info(f"Adaptive recommendations: {recommendations}")
        return recommendations

    except Exception as e:
        logging.error(f"Error in adaptive_recommendation: {str(e)}")
        return recommendations
        
def generate_recommendations(data, symbol=None):
    """
    Generates trading recommendations based on technical indicators.
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

    if not validate_data(data, min_length=27):
        logging.warning(f"Invalid data for recommendations: {symbol}")
        return recommendations

    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        buy_score = 0
        sell_score = 0

        # RSI
        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) and isinstance(data['RSI'].iloc[-1], (int, float)):
            rsi = data['RSI'].iloc[-1]
            if rsi <= 20:
                buy_score += 4
                recommendations["Mean_Reversion"] = "Buy"
            elif rsi < 30:
                buy_score += 2
                recommendations["Mean_Reversion"] = "Buy"
            elif rsi > 70:
                sell_score += 2
                recommendations["Mean_Reversion"] = "Sell"
            logging.info(f"RSI score for {symbol}: buy={buy_score}, sell={sell_score}")

        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
            if macd_diff > 0:
                buy_score += 2
                recommendations["Swing"] = "Buy"
            elif macd_diff < 0:
                sell_score += 2
                recommendations["Swing"] = "Sell"
            logging.info(f"MACD score for {symbol}: buy={buy_score}, sell={sell_score}")

        # Ichimoku
        if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
            close = data['Close'].iloc[-1]
            span_a = data['Ichimoku_Span_A'].iloc[-1]
            span_b = data['Ichimoku_Span_B'].iloc[-1]
            if close > span_a and close > span_b:
                buy_score += 3
                recommendations["Ichimoku_Trend"] = "Buy"
            elif close < span_a and close < span_b:
                sell_score += 3
                recommendations["Ichimoku_Trend"] = "Sell"
            logging.info(f"Ichimoku score for {symbol}: buy={buy_score}, sell={sell_score}")

        # Bollinger Bands Breakout
        if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]):
            close = data['Close'].iloc[-1]
            if close > data['Upper_Band'].iloc[-1]:
                buy_score += 2
                recommendations["Breakout"] = "Buy"
            elif close < data['Lower_Band'].iloc[-1]:
                sell_score += 2
                recommendations["Breakout"] = "Sell"
            logging.info(f"Breakout score for {symbol}: buy={buy_score}, sell={sell_score}")

        # CMF
        if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
            cmf = data['CMF'].iloc[-1]
            if cmf > 0.2:
                buy_score += 1
            elif cmf < -0.2:
                sell_score += 1
            logging.info(f"CMF score for {symbol}: buy={buy_score}, sell={sell_score}")

        # Generate recommendations based on scores
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

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = min(max(net_score, -7), 7)

        logging.info(f"Recommendations for {symbol}: {recommendations}")
        return recommendations

    except Exception as e:
        logging.error(f"Error generating recommendations for {symbol}: {str(e)}")
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
            rec = generate_recommendations(data, symbol)
            total_score += rec.get("Score", 0)
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
                    if "Error" in result:
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def analyze_stock_parallel(symbol):
    try:
        data = fetch_stock_data_with_dhan(symbol)
        if data.empty or len(data) < 50:
            logging.warning(f"No sufficient data for {symbol}: {len(data)} rows")
            return None
        data = analyze_stock(data)
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
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
        logging.error(f"Error in analyze_stock_parallel for {symbol}: {str(e)}")
        return None
        
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
        time.sleep(max(2, batch_size / 5))
    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    return results_df.sort_values(by="Score", ascending=False).head(5)

    
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
        
        time.sleep(30)
    
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
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")
    
# Add the new function for stock status updates
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
    for _, row in results_df.head(5).iterrows():
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
            "position_size": row.get('Position Size', None),
            "trailing_stop": row.get('Trailing Stop', None),
            "reason": row.get('Reason', 'No reason provided'),
            "pick_type": pick_type
        }
        logging.info(f"Inserting to Supabase: {data}")
        try:
            res = supabase.table("daily_picks").upsert(data).execute()
            if hasattr(res, "error") and res.error:
                logging.error(f"Supabase insert error: {res.error}")
                st.error(f"Supabase insert error: {res.error}")
        except Exception as e:
            logging.error(f"Supabase insert exception: {e}")
            st.error(f"Supabase insert exception: {e}")


@RateLimiter(calls=1, period=1)
def fetch_latest_price(symbol):
    data = fetch_stock_data_with_dhan(symbol, period="1mo", interval="1d")
    if not data.empty:
        return float(data['Close'].iloc[-1])
    return None

def update_with_latest_prices(df):
    for idx, row in df.iterrows():
        symbol = row['symbol']
        try:
            latest_close = fetch_latest_price(symbol)
            if latest_close is not None:
                df.at[idx, 'current_price'] = latest_close
        except Exception as e:
            st.warning(f"Could not fetch latest price for {symbol}: {e}")
    return df
        
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
            top_sectors = get_top_sectors_cached(rate_limit_delay=10, stocks_per_sector=10)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            for name, score in top_sectors:
                st.markdown(f"- **{name}**: {score:.2f}/7")

    # Daily top picks button - PROPERLY INDENTED INSIDE display_dashboard
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
        
        insert_top_picks_supabase(results_df, pick_type="daily")
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()
        
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

    # Intraday top picks button - PROPERLY INDENTED INSIDE display_dashboard
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        
        # If you want to show stock status for intraday too, add status_text
        status_text = st.empty()
        
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages),
            status_callback=lambda status: status_text.text(status)  # Optional: add stock status
        )
        
        insert_top_picks_supabase(intraday_results, pick_type="intraday")
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()  # Clear status text if used
        
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

    # Rest of the display_dashboard function continues here...
    # Historical picks button
    # At the top of your display_dashboard function (or before the button block):

    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    if st.button("📜 View Historical Picks"):
        st.session_state.show_history = not st.session_state.show_history

    if st.session_state.show_history:
        st.markdown("### 📜 Historical Picks")
        if st.button("Close Historical Picks"):
            st.session_state.show_history = False
            st.experimental_rerun()  # This will immediately hide the view

        res = supabase.table("daily_picks").select("date").order("date", desc=True).execute()
        if res.data:
            all_dates = sorted({row['date'] for row in res.data}, reverse=True)
            date_filter = st.selectbox("Select Date", all_dates, key="history_date")
            res2 = supabase.table("daily_picks").select("*").eq("date", date_filter).execute()
            if res2.data:
                df = pd.DataFrame(res2.data)
                df = update_with_latest_prices(df)
                df = add_action_and_change(df)
                for col in ['buy_at', 'current_price', 'target', 'stop_loss', '% Change']:
                    if col in df.columns:
                        df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
            display_cols = [
                "symbol", "buy_at", "current_price", "% Change", "recommendation", "What to do now?", "target", "stop_loss"
            ]
            styled_df = style_picks_df(df[display_cols])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No picks found for this date.")
    else:
        st.warning("No historical data available.")
        
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

        # Price Chart
        st.subheader("📈 Price Chart with Indicators")
        fig = px.line(data, x=data.index, y='Close', title=f"{symbol.split('-')[0]} Price")
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
        st.subheader("📊 Monte Carlo Simulation")
        simulations = monte_carlo_simulation(data)
        sim_df = pd.DataFrame(simulations).T
        sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
        fig_sim = px.line(sim_df, title="Monte Carlo Price Projections (30 Days)")
        st.plotly_chart(fig_sim, use_container_width=True)

        # RSI and MACD
        st.subheader("📊 RSI and MACD")
        fig_ind = px.line(data, x=data.index, y='RSI', title="RSI")
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red")
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_ind, use_container_width=True)

        fig_macd = px.line(data, x=data.index, y=['MACD', 'MACD_signal'], title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

        # Volume Analysis
        st.subheader("📊 Volume Analysis")
        fig_vol = px.bar(data, x=data.index, y='Volume', title="Volume")
        if 'Volume_Spike' in data.columns:
            spike_data = data[data['Volume_Spike'] == True]
            if not spike_data.empty:
                fig_vol.add_scatter(x=spike_data.index, y=spike_data['Volume'], mode='markers', name='Volume Spike',
                                   marker=dict(color='red', size=10))
        st.plotly_chart(fig_vol, use_container_width=True)
    
            
def main():
    init_database()
    
    # Test Dhan connection first
    if st.sidebar.button("Test Dhan Connection"):
        with st.spinner("Testing Dhan API connection..."):
            if test_dhan_connection():
                st.success("✅ Dhan API is working!")
            else:
                st.error("❌ Dhan API connection failed. Check your credentials.")
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
                data = fetch_stock_data_with_dhan(symbol)
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
