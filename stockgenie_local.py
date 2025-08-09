import pandas as pd
import ta
import logging
import numpy as np
from functools import lru_cache
import streamlit as st
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.graph_objects as go # Import plotly.graph_objects for candlesticks and subplots
from plotly.subplots import make_subplots # Import make_subplots
import time
import requests
import io
import random
import numpy as np
from arch import arch_model
import warnings
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

# Load Streamlit secrets for Dhan API
# Ensure DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are correctly set in Streamlit secrets
try:
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
except KeyError:
    st.error("Dhan API credentials not found in Streamlit secrets. Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN.")
    st.stop()


def dhan_headers():
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

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
                        st.warning(f"Rate limit hit. Retrying {func.__name__} after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    st.warning(f"Connection error for {func.__name__}: {e}. Retrying after {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                except Exception as e: # Catch all other exceptions for retry
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    st.warning(f"Unexpected error for {func.__name__}: {e}. Retrying after {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            return None
        return wrapper
    return decorator

@st.cache_data(ttl=86400)
@retry(max_retries=3, delay=5) # Add retry for master file download
def load_dhan_instrument_master():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        df = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_SEGMENT'] == 'E')]
        return df
    except Exception as e:
        logging.error(f"Failed to load Dhan instrument master: {e}")
        return pd.DataFrame()


def get_dhan_security_id(symbol):
    """Get Dhan security ID for a given symbol"""
    df = load_dhan_instrument_master()
    if df.empty:
        logging.error(f"Instrument master data is empty, cannot find security ID for {symbol}")
        return None

    symbol_clean = normalize_symbol_dhan(symbol).upper()

    for column in ['SEM_SMST_SECURITY_ID', 'SM_SYMBOL_NAME', 'SEM_TRADING_SYMBOL', 'SEM_CUSTOM_SYMBOL']:
        if column in df.columns:
            row = df[df[column].astype(str).str.upper() == symbol_clean]
            if not row.empty:
                security_id = row.iloc[0]['SEM_SMST_SECURITY_ID']
                logging.info(f"Found security_id for {symbol}: {security_id} in column {column}")
                return security_id

    possible_matches = df[df['SM_SYMBOL_NAME'].astype(str).str.contains(symbol_clean, case=False, na=False)]
    logging.warning(f"No security_id found for {symbol} ({symbol_clean}). Possible matches: {possible_matches[['SM_SYMBOL_NAME', 'SEM_TRADING_SYMBOL']].to_dict('records')}")
    return None

def test_dhan_connection():
    """Test if Dhan API credentials are working"""
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }

    # Use a highly liquid stock for testing like Reliance, which always has intraday data
    # Security ID for RELIANCE is often 2885 for NSE_EQ.
    test_security_id = get_dhan_security_id("RELIANCE.NS")
    if not test_security_id:
        logging.error("Could not find security ID for RELIANCE.NS, cannot test Dhan connection.")
        return False

    payload = {
        "securityId": str(test_security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "interval": "1m"
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
        v = float(str(val).replace('₹', '').replace('%', '')) # Convert to string then remove currency/percentage
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

    # Ensure buy_at and current_price are numeric
    df['buy_at'] = pd.to_numeric(df['buy_at'], errors='coerce')
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')

    df['% Change'] = ((df['current_price'] - df['buy_at']) / df['buy_at'] * 100).round(2)
    df['What to do now?'] = df.apply(action, axis=1)
    return df


warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

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
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
    "Score": "Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal.",
    "OBV": "On-Balance Volume - Measures buying and selling pressure by adding/subtracting volume based on price changes."
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
    "DBREALTY.NS", "JWL.NS","JAYBARMARU.NS"
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
        # Fallback to predefined sectors if NSE list fetch fails
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

# Helper function to parse period strings to days
def parse_period_to_days(period):
    """Convert period string (e.g., '5y', '1mo') to number of days."""
    period = period.lower()
    try:
        if period.endswith('y'):
            years = float(period[:-1])
            return int(years * 365)
        elif period.endswith('mo'):
            months = float(period[:-2])
            return int(months * 30)
        elif period.endswith('d'):
            days = float(period[:-1])
            return int(days)
        else:
            logging.warning(f"Unsupported period format: {period}. Defaulting to 30 days.")
            return 30
    except ValueError as e:
        logging.error(f"Invalid period format: {period}, error: {e}. Defaulting to 30 days.")
        return 30

# Increased period for a more conservative rate limit
@lru_cache(maxsize=1000)
@RateLimiter(calls=4, period=1)
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
    """
    Fetches stock data from Dhan API, with rate limiting and caching.
    This function wraps the actual API call, allowing its results to be cached in memory.
    """
    global data_api_calls
    with data_api_lock:
        if data_api_calls >= 90000:
            logging.warning(f"⚠️ Approaching Data API daily limit: {data_api_calls}/100000")
        if data_api_calls >= 100000:
            logging.error("⚠️ Reached daily Data API limit of 100,000 requests. Cannot fetch more data.")
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
        # logging.info(f"Requesting data for {symbol} with payload: {payload}") # Too verbose for production
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)

        # Explicitly check for rate limit messages even if status code is not 429
        if response.status_code == 429:
            logging.warning(f"Dhan API returned 429 (Too Many Requests) for {symbol}. Retrying...")
            response.raise_for_status() # This will be caught by the @retry decorator

        response_text = response.text
        # Check for specific "too many calls" message in response body
        if "too many calls" in response_text.lower() or "limit exceeded" in response_text.lower():
            logging.warning(f"Dhan API response text indicates 'too many calls' or 'limit exceeded' for {symbol} (Status: {response.status_code}). Response: {response_text}. Raising error to trigger retry.")
            # If the API sends "too many calls" in a non-429 response, we need to force a retry by raising an exception
            # that the @retry decorator will catch. A generic RequestException is appropriate.
            raise requests.exceptions.RequestException(f"API indicated 'too many calls' in response body for {symbol}")

        response.raise_for_status() # This raises HTTPError for other 4xx/5xx responses
        data = response.json()

        api_data = data.get("data", data)

        if isinstance(api_data, dict) and all(isinstance(v, list) for v in api_data.values()):
            if not api_data or not next(iter(api_data.values()), []):
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

        # --- FIX FOR TIMESTAMP CONVERSION ---
        if pd.api.types.is_numeric_dtype(df['Date']):
            # Attempt to convert as seconds. This matches the Dhan sample response format.
            df["Date"] = pd.to_datetime(df["Date"].astype(np.int64), unit='s', errors='coerce')

            # Additional check/fallback if seconds unit leads to too many NaTs or very old dates,
            # indicating it might actually be milliseconds for some cases.
            # This part is optional but adds robustness if API is inconsistent.
            if df['Date'].isnull().sum() > len(df) / 2 or (not df.empty and df['Date'].iloc[-1].year < 2000):
                 logging.warning(f"Seconds conversion yielded problematic dates for {symbol}. Attempting milliseconds.")
                 df["Date"] = pd.to_datetime(df["Date"].astype(np.int64), unit='ms', errors='coerce')
        else:
            # If it's not numeric, assume it's a string date and convert normally.
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        # Filter out NaT (Not a Time) dates if any failed conversion
        df = df.dropna(subset=['Date'])
        # --- END FIX ---

        df.set_index("Date", inplace=True)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns for {symbol}: {missing_columns}. Columns found: {df.columns.tolist()}")
            return pd.DataFrame()

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_columns)

        # --- NEW ADDITION: Filter out 1970-01-01 dates from the index ---
        # These are usually valid epoch dates (timestamp=0) but not relevant for stock data.
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            # Define a cutoff date just after epoch start to filter problematic dates
            # Ensure this is timezone-naive to match df.index, preventing TypeError.
            epoch_start_threshold = pd.Timestamp('1970-01-02')

            original_len = len(df)
            # Remove dates before the threshold
            df = df[df.index >= epoch_start_threshold]
            if len(df) < original_len:
                logging.warning(f"Filtered out {original_len - len(df)} early epoch dates for {symbol}.")
        # --- END NEW ADDITION ---

        # logging.info(f"Successfully fetched data for {symbol}: {len(df)} rows") # Too verbose

        # Add a small random sleep AFTER successful fetch to reduce burst impact on server-side limits
        time.sleep(random.uniform(0.1, 0.5))

        return df[required_columns]

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {symbol}: {e}, Response: {e.response.text if e.response else 'No response'}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        # This catches both specific RequestException and the custom one we raise
        logging.error(f"Request error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error for {symbol}: {e}")
        return pd.DataFrame()


def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if returns.empty:
        return []

    if len(returns) < 50: # Not enough data for GARCH, fall back to simple historical volatility
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            logging.warning("Standard deviation of returns is zero for Monte Carlo. Using flat projection.")
            simulation_results = [[data['Close'].iloc[-1]] * (days + 1) for _ in range(simulations)]
            return simulation_results

        simulation_results = []
        for _ in range(simulations): # Loop 'simulations' times
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days): # Project 'days' into the future for each simulation
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series) # Append the full price series for one simulation
        return simulation_results

    try:
        # Use GARCH(1,1) model for volatility forecasting if enough data
        # 'Normal' distribution for residuals
        model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
        garch_fit = model.fit(disp='off') # disp='off' suppresses verbose output
        forecasts = garch_fit.forecast(horizon=days)

        if forecasts.variance.empty:
            logging.warning("GARCH variance forecasts are empty. Falling back to simple simulation.")
            return monte_carlo_simulation(data, simulations, days)

        # Get forecasted volatility for each day in the horizon
        # Make sure to get the values from the specific index/column where horizon variance is
        volatility = np.sqrt(forecasts.variance.iloc[-1].values)

        # Pad volatility if the GARCH model couldn't forecast for the full 'days' horizon
        if len(volatility) < days:
            logging.warning(f"GARCH volatility forecasts (len={len(volatility)}) less than required days ({days}). Replicating last volatility.")
            volatility = np.pad(volatility, (0, days - len(volatility)), mode='edge')

        mean_return = returns.mean()
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for i in range(days):
                # Use the forecasted volatility for the current day of the simulation
                current_volatility = volatility[min(i, len(volatility) - 1)] # defensive indexing
                price = price_series[-1] * (1 + np.random.normal(mean_return, current_volatility))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    except Exception as e:
        logging.error(f"Error in Monte Carlo GARCH simulation: {e}. Falling back to simple simulation.")
        return monte_carlo_simulation(data, simulations, days)


def assess_risk(data):
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
        atr_ratio = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_ratio > 0.04:
            return "High Volatility"
        elif atr_ratio > 0.02:
            return "Moderate Volatility"
        else:
            return "Low Volatility"
    return "N/A"

# Updated INDICATOR_MIN_LENGTHS to reflect hardcoded defaults
INDICATOR_MIN_LENGTHS = {
    'RSI': 14,
    'MACD': 26, # window_slow=26
    'SMA_50': 50,
    'SMA_200': 200,
    'EMA_20': 20,
    'EMA_50': 50,
    'Bollinger': 20, # window=20
    'Stochastic': 14, # window=14
    'ATR': 14, # window=14
    'ADX': 14, # window=14
    'OBV': 1,
    'Ichimoku': 52, # window3=52
    'CMF': 20, # window=20
    'TRIX': 15, # window=15
    'Ultimate_Osc': 28, # longest window in UltimateOscillator (default periods 7, 14, 28)
    'VPT': 1,
    'Volume_Spike': 20 # based on default Bollinger window for Avg_Volume
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
    # Check for NaNs only in the recent data relevant for indicator calculation
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

# Reverted analyze_stock to use hardcoded parameters for TA indicators
def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    Uses default TA-lib parameters.
    """
    # Define columns for *actually computed* indicators
    columns_to_compute = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
        'Avg_Volume', 'Volume_Spike', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF',
        'TRIX', 'Ultimate_Osc', 'VPT', 'ATR_pct', 'DMP', 'DMN', 'ADX_Slope'
    ]

    # Initialize all columns to NaN to ensure they exist before computation
    for col in columns_to_compute:
        if col not in data.columns:
            data[col] = np.nan

    # Ensure core columns are numeric and drop rows with NaNs in them
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Ensure sufficient data for Ichimoku as it has the highest min_length
    required_ichimoku_length = INDICATOR_MIN_LENGTHS['Ichimoku']
    if not validate_data(data, min_length=required_ichimoku_length):
        return data

    try:
        # Calculate Average Volume first for Volume_Spike (using default BB window 20)
        if can_compute_indicator(data, 'Volume_Spike'):
            data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()

        if can_compute_indicator(data, 'RSI'):
            data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()

        if can_compute_indicator(data, 'MACD'):
            macd = MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()

        if can_compute_indicator(data, 'SMA_50'):
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
        if can_compute_indicator(data, 'SMA_200'):
            data['SMA_200'] = data['Close'].rolling(window=200).mean()

        if can_compute_indicator(data, 'EMA_20'):
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        if can_compute_indicator(data, 'EMA_50'):
            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        if can_compute_indicator(data, 'Bollinger'):
            bb = BollingerBands(data['Close'], window=20, window_dev=2)
            data['Upper_Band'] = bb.bollinger_hband()
            data['Middle_Band'] = bb.bollinger_mavg()
            data['Lower_Band'] = bb.bollinger_lband()

        if can_compute_indicator(data, 'Stochastic'):
            stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()

        if can_compute_indicator(data, 'ATR'):
            data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
            data['ATR_pct'] = data['ATR'] / data['Close']

        if can_compute_indicator(data, 'ADX'):
            adx_ind = ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
            data['ADX'] = adx_ind.adx()
            data['DMP'] = adx_ind.adx_pos()
            data['DMN'] = adx_ind.adx_neg()
            data['ADX_Slope'] = data['ADX'].diff(periods=5) # 5-period slope for ADX

        if can_compute_indicator(data, 'OBV'):
            data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

        if can_compute_indicator(data, 'Ichimoku'):
            ichimoku = IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Chikou'] = data['Close'].shift(-26) # Chikou needs future data, shifted back by Kijun period

        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20).chaikin_money_flow()

        if can_compute_indicator(data, 'TRIX'):
            data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()

        if can_compute_indicator(data, 'Ultimate_Osc'):
            data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(high=data['High'], low=data['Low'], close=data['Close']).ultimate_oscillator()

        if can_compute_indicator(data, 'VPT'):
            data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()

        # Volume Spike (requires Avg_Volume to be computed first)
        if 'Avg_Volume' in data.columns and not data['Avg_Volume'].empty and pd.notnull(data['Avg_Volume'].iloc[-1]):
             data['Volume_Spike'] = data['Volume'] > data['Avg_Volume'] * 2
        else:
            data['Volume_Spike'] = False


        return data

    except Exception as e:
        logging.error(f"Error in analyze_stock: {str(e)}")
        # Re-initialize only the computed columns to NaN if an error occurred during computation
        for col in columns_to_compute:
            if col not in data.columns:
                data[col] = np.nan
        return data

# Modified calculate_buy_at to accept ATR multiplier
def calculate_buy_at(data, atr_factor=0.2):
    """
    Calculates the buy price based on recent price action and indicators.
    Aims for a slight pullback in an uptrend, or simply near the current close.
    """
    if not validate_data(data, min_length=10) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        ema_20 = data['EMA_20'].iloc[-1] if 'EMA_20' in data.columns and pd.notnull(data['EMA_20'].iloc[-1]) else close

        if atr > 0:
            # Target a price slightly below current close, or near EMA20 if in an uptrend
            # We want to buy a pullback, but not below a strong EMA
            target_pullback_price = close - atr_factor * atr
            if close > ema_20: # In an uptrend, ensure we are not buying too far from EMA
                buy_at = max(target_pullback_price, ema_20 * 0.99) # Allow slight dip below EMA20
                buy_at = min(buy_at, close * 0.995) # Cap max discount from current close
            else: # Not in strong EMA uptrend, just a simple ATR based discount
                buy_at = close * (1 - atr_factor * (atr / close))
        else:
            buy_at = close * 0.99 # Default 1% discount if no ATR

        # Ensure buy_at is not higher than current close (should be a discount)
        buy_at = min(buy_at, close)
        return round(float(buy_at), 2)
    except Exception as e:
        logging.error(f"Error calculating buy_at: {str(e)}")
        return None

# Modified calculate_stop_loss to accept ATR multiplier and ADX thresholds
def calculate_stop_loss(data, atr_multiplier=3.0, adx_high_threshold=30, adx_low_threshold=20):
    """
    Calculates the stop loss price based on ATR and recent lows.
    ADX based dynamic ATR multiplier for adaptive stop loss.
    """
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['ATR']) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1]

        if atr <= 0 or pd.isna(atr): # Handle zero or NaN ATR
            return close * 0.95 # Default 5% stop if ATR not available

        # ADX based dynamic ATR multiplier
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
        if adx > adx_high_threshold: # Strong trend, tighter stop as trend is more predictable
            current_atr_multiplier = 2.0
        elif adx < adx_low_threshold: # Sideways, wider stop to avoid whipsaws
            current_atr_multiplier = 3.5
        else: # Developing trend or neutral
            current_atr_multiplier = atr_multiplier # Use default

        stop_loss = close - current_atr_multiplier * atr

        # Ensure stop loss is not excessively tight (min 2% below close) or loose (max 10% below close)
        stop_loss = max(stop_loss, close * 0.90) # No more than 10% loss typically
        stop_loss = min(stop_loss, close * 0.98) # At least 2% below current price

        return round(float(stop_loss), 2)
    except Exception as e:
        logging.error(f"Error calculating stop_loss: {str(e)}")
        return None

# Modified calculate_target to accept risk-reward ratio and ADX thresholds
def calculate_target(data, risk_reward_ratio=2.5, adx_high_threshold=30, adx_low_threshold=20):
    """
    Calculates the target price based on ATR and a risk-reward ratio.
    ADX based dynamic risk-reward for adaptive target.
    """
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['ATR']) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        # Pass parameters to calculate_stop_loss for consistency
        stop_loss = calculate_stop_loss(data, adx_high_threshold=adx_high_threshold, adx_low_threshold=adx_low_threshold)

        if stop_loss is None or stop_loss >= close:
            return None # Cannot calculate target if stop_loss is invalid or above close

        risk_per_share = close - stop_loss

        if risk_per_share <= 0: # Avoid division by zero or negative risk
            return None

        # ADX based dynamic risk-reward
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
        if adx > adx_high_threshold:
            adjusted_rr = 3.0 # Higher risk-reward in strong trends
        elif adx < adx_low_threshold:
            adjusted_rr = 2.0 # Slightly lower risk-reward in sideways/choppy markets
        else:
            adjusted_rr = risk_reward_ratio

        target = close + (risk_per_share * adjusted_rr)

        # Cap target at 25% profit to be realistic for swing trades
        target = min(target, close * 1.25)

        return round(float(target), 2)
    except Exception as e:
        logging.error(f"Error calculating target: {str(e)}")
        return None

def calculate_trailing_stop(current_price, atr, atr_multiplier=2.0, prev_trailing_stop=None):
    """
    Calculates a simple ATR-based trailing stop.
    If prev_trailing_stop is provided, ensures the new trailing stop never goes down.
    """
    if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
        return prev_trailing_stop # Keep previous if new calculation is invalid

    new_trailing_stop = current_price - atr_multiplier * atr

    # Ensure trailing stop never goes down
    if prev_trailing_stop is not None and new_trailing_stop < prev_trailing_stop:
        return round(float(prev_trailing_stop), 2)

    # Ensure trailing stop is not too close to current price (e.g., min 1% below current)
    # This also helps if ATR is very small leading to an unmanageable stop.
    new_trailing_stop = max(current_price * 0.99, new_trailing_stop)

    return round(max(0, new_trailing_stop), 2) # Ensure it's not negative


# Modified classify_market_regime to accept ADX thresholds
def classify_market_regime(data, adx_strong_trend_threshold=25, adx_no_trend_threshold=20):
    """Classifies regime based on volatility, trend strength (ADX), and trend direction (SMA)."""
    # Need sufficient data for SMA_200 and ATR/ADX
    required_length = max(INDICATOR_MIN_LENGTHS['SMA_200'], INDICATOR_MIN_LENGTHS['ATR'], INDICATOR_MIN_LENGTHS['ADX'])
    if not validate_data(data, min_length=required_length):
        return 'Insufficient Data'

    # Get latest indicator values defensively
    close = data['Close'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns and pd.notnull(data['SMA_50'].iloc[-1]) else np.nan
    sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns and pd.notnull(data['SMA_200'].iloc[-1]) else np.nan
    atr_pct = data['ATR_pct'].iloc[-1] if 'ATR_pct' in data.columns and pd.notnull(data['ATR_pct'].iloc[-1]) else np.nan
    adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else np.nan
    adx_slope = data['ADX_Slope'].iloc[-1] if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]) else np.nan

    if pd.isna(close) or pd.isna(sma_50) or pd.isna(sma_200) or \
       pd.isna(atr_pct) or pd.isna(adx) or pd.isna(adx_slope):
        return 'Incomplete Indicator Data'

    # Volatility Filter (added tighter ranges)
    # These are hardcoded for `classify_market_regime` unless explicitly made configurable for it too
    if atr_pct > 0.045: # Very High Volatility
        return 'Highly Volatile'
    if atr_pct < 0.005: # Very Low Volatility, often means sideways/no movement
        return 'Low Volatility/Tight Range'

    if adx > adx_strong_trend_threshold: # Strong trend
        if close > sma_50 and sma_50 > sma_200 and close > sma_200: # All bullish alignment
            return 'Bullish Trending'
        elif close < sma_50 and sma_50 < sma_200 and close < sma_200: # All bearish alignment
            return 'Bearish Trending'
        else: # Price is trending but mixed signals from MAs
            return 'Trending (Unclear Direction)'
    elif adx < adx_no_trend_threshold: # Weak/No trend
        return 'Sideways/Consolidating'
    else: # Developing trend or neutral
        if close > sma_50 and close > sma_200:
            return 'Bullish (Developing Trend)'
        elif close < sma_50 and close < sma_200:
            return 'Bearish (Developing Trend)'
        else:
            return 'Neutral'


def compute_signal_score(data, symbol=None):
    """
    Computes a weighted score based on normalized technical indicators and market conditions.
    This score is a general sentiment, not the primary decision driver for the refined adaptive strategy.
    Returns a score between -10 and 10.
    """
    score = 0.0
    reason_components = []

    # Weights adjusted to prioritize trend/momentum for the adaptive strategy
    weights = {
        'RSI_Oversold': 1.0, 'RSI_Overbought': -1.0, # Less weight for mean-reversion in trend-following
        'MACD_Bullish': 2.5, 'MACD_Bearish': -2.5,
        'Ichimoku_Bullish': 2.0, 'Ichimoku_Bearish': -2.0,
        'CMF_Buying': 1.0, 'CMF_Selling': -1.0,
        'ADX_StrongBull': 1.5, 'ADX_StrongBear': -1.5,
        'Bollinger_BreakoutUp': 1.0, 'Bollinger_BreakoutDown': -1.0,
        'Volume_SpikeUp': 1.0, 'Volume_SpikeDown': -1.0,
        'SMA_BullishCross': 1.5, 'SMA_BearishCross': -1.5,
        'Price_Above_SMAs': 2.0, 'Price_Below_SMAs': -2.0
    }

    max_possible_raw_score = sum([w for k,w in weights.items() if w > 0]) # Sum of all positive weights

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        # logging.warning(f"Invalid data for scoring {symbol}, returning 0.")
        return 0, ["Insufficient data to compute comprehensive score."]

    close_price = data['Close'].iloc[-1]
    current_volume = data['Volume'].iloc[-1]

    # Check for very low volume
    if 'Avg_Volume' in data.columns and pd.notnull(data['Avg_Volume'].iloc[-1]) and data['Avg_Volume'].iloc[-1] > 0:
        if current_volume < data['Avg_Volume'].iloc[-1] * 0.3:
            score -= 2 # Penalty for very low volume
            reason_components.append("Very low volume, caution advised.")

    # RSI (less impactful for overall score in a trend-following context)
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        if rsi < 30:
            score += weights['RSI_Oversold'] # Potential bounce from oversold
            reason_components.append(f"RSI({int(rsi)}) is oversold.")
        elif rsi > 70:
            score += weights['RSI_Overbought'] # Potential pullback from overbought
            reason_components.append(f"RSI({int(rsi)}) is overbought.")

    # MACD
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        macd_hist = data['MACD_hist'].iloc[-1]

        # Simplified MACD conditions for signal
        if macd > macd_signal and macd > 0: # Bullish crossover above zero
            score += weights['MACD_Bullish']
            reason_components.append("MACD is bullish (crossover above zero).")
        elif macd < macd_signal and macd < 0: # Bearish crossover below zero
            score += weights['MACD_Bearish']
            reason_components.append("MACD is bearish (crossover below zero).")
        elif macd > macd_signal and macd_hist > 0: # Bullish crossover with momentum (even if below zero)
            score += weights['MACD_Bullish'] * 0.5 # Less strong signal
            reason_components.append("MACD is bullish (crossover & positive momentum).")
        elif macd < macd_signal and macd_hist < 0: # Bearish crossover with momentum (even if above zero)
            score += weights['MACD_Bearish'] * 0.5 # Less strong signal
            reason_components.append("MACD is bearish (crossover & negative momentum).")

    # Ichimoku Cloud
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
        cloud_top = max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1])
        cloud_bottom = min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1])

        if close_price > cloud_top: # Price above the cloud (bullish)
            score += weights['Ichimoku_Bullish']
            reason_components.append("Price is above Ichimoku Cloud.")
        elif close_price < cloud_bottom: # Price below the cloud (bearish)
            score += weights['Ichimoku_Bearish']
            reason_components.append("Price is below Ichimoku Cloud.")

    # CMF
    if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        if cmf > 0.10: # Stronger buying pressure threshold
            score += weights['CMF_Buying']
            reason_components.append(f"CMF({cmf:.2f}) indicates buying pressure.")
        elif cmf < -0.10: # Stronger selling pressure threshold
            score += weights['CMF_Selling']
            reason_components.append(f"CMF({cmf:.2f}) indicates selling pressure.")

    # ADX
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['ADX', 'DMP', 'DMN']):
        adx = data['ADX'].iloc[-1]
        dmp = data['DMP'].iloc[-1]
        dmn = data['DMN'].iloc[-1]

        if adx > 25 and dmp > dmn: # Strong trend and DI+ > DI- (bullish)
            score += weights['ADX_StrongBull']
            reason_components.append(f"ADX({int(adx)}) shows strong bullish trend.")
        elif adx > 25 and dmn > dmp: # Strong trend and DI- > DI+ (bearish)
            score += weights['ADX_StrongBear']
            reason_components.append(f"ADX({int(adx)}) shows strong bearish trend.")
        if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]):
            adx_slope = data['ADX_Slope'].iloc[-1]
            if adx_slope > 0 and adx < 40: # ADX still rising but not extremely high
                reason_components.append(f"ADX slope positive (trend strengthening).")
            elif adx_slope < 0 and adx > 20: # ADX falling but still indicating a trend
                reason_components.append(f"ADX slope negative (trend weakening).")

    # Bollinger Bands Breakout / Squeeze (Breakouts contribute to score)
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Upper_Band', 'Lower_Band', 'Close']):
        upper_band = data['Upper_Band'].iloc[-1]
        lower_band = data['Lower_Band'].iloc[-1]

        if close_price > upper_band: # Price breaking above upper band (bullish breakout)
            score += weights['Bollinger_BreakoutUp']
            reason_components.append("Price breaking out above Bollinger Upper Band.")
        elif close_price < lower_band: # Price breaking below lower band (bearish breakout)
            score += weights['Bollinger_BreakoutDown']
            reason_components.append("Price breaking out below Bollinger Lower Band.")

    # Volume Spike
    if 'Volume_Spike' in data.columns and pd.notnull(data['Volume_Spike'].iloc[-1]) and data['Volume_Spike'].iloc[-1]:
        if len(data) >= 2 and close_price > data['Close'].iloc[-2]: # Volume spike with rising price
            score += weights['Volume_SpikeUp']
            reason_components.append("Significant volume spike confirming upward price move.")
        elif len(data) >= 2 and close_price < data['Close'].iloc[-2]: # Volume spike with falling price
            score += weights['Volume_SpikeDown']
            reason_components.append("Significant volume spike confirming downward price move.")

    # SMA Crossovers and Price vs. SMAs
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['SMA_50', 'SMA_200']) and len(data) >= 2:
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        prev_sma_50 = data['SMA_50'].iloc[-2]
        prev_sma_200 = data['SMA_200'].iloc[-2]

        if sma_50 > sma_200 and prev_sma_50 <= prev_sma_200: # Golden Cross (50-day crosses above 200-day)
            score += weights['SMA_BullishCross']
            reason_components.append("Golden Cross (SMA50 > SMA200) observed.")
        elif sma_50 < sma_200 and prev_sma_50 >= prev_sma_200: # Death Cross (50-day crosses below 200-day)
            score += weights['SMA_BearishCross']
            reason_components.append("Death Cross (SMA50 < SMA200) observed.")

        if close_price > sma_50 and close_price > sma_200:
            score += weights['Price_Above_SMAs']
            reason_components.append("Price above SMA50 & SMA200.")
        elif close_price < sma_50 and close_price < sma_200:
            score += weights['Price_Below_SMAs']
            reason_components.append("Price below SMA50 & SMA200.")


    scaled_score = (score / max_possible_raw_score) * 10 if max_possible_raw_score > 0 else 0
    final_score = min(max(scaled_score, -10), 10) # Clamp score between -10 and 10

    final_reason = " | ".join(reason_components) if reason_components else "No strong indicator signals."

    # Added logic for more descriptive neutral score reason
    if abs(final_score) < 0.5: # If score is near zero
        if reason_components: # If there were reasons, they cancelled out
            final_reason += " (conflicting signals led to neutral score)"
        else: # No strong signals either way
            final_reason = "Neutral market conditions observed."

    # logging.info(f"Signal score for {symbol}: {final_score:.2f}, Reason: {final_reason}") # Too verbose
    return final_score, final_reason


# Modified adaptive_recommendation with enhanced strategy logic and accepts strategy params
def adaptive_recommendation(data, symbol=None, equity=100000, risk_per_trade_pct=1,
                            adx_strong_trend_threshold=25, adx_no_trend_threshold=20,
                            rsi_overbought=70, rsi_oversold=30, macd_zero_line_confirm=True,
                            cmf_buy_threshold=0.10, cmf_sell_threshold=-0.10,
                            atr_buy_pullback_factor=0.5, atr_exit_multiplier=2.0,
                            atr_low_volatility_pct=0.005, atr_high_volatility_pct=0.045,
                            risk_reward_ratio=2.5, max_position_pct=0.25): # Added strategy params
    """
    Generates adaptive trading recommendations based on market regime and technical indicators.
    Includes position sizing, trailing stop, and detailed reasons.
    This version implements a more robust Trend-Following strategy.
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

    # Ensure sufficient data for full analysis and regime classification
    required_length_for_adaptive = max(INDICATOR_MIN_LENGTHS['SMA_200'], INDICATOR_MIN_LENGTHS['Ichimoku'])
    if not validate_data(data, min_length=required_length_for_adaptive):
        recommendations["Reason"] = "Insufficient historical data for comprehensive adaptive analysis."
        logging.warning(f"Insufficient data for {symbol} in adaptive_recommendation. Len: {len(data)}")
        return recommendations

    try:
        current_price = round(float(data['Close'].iloc[-1]), 2)
        recommendations["Current Price"] = current_price

        if pd.isna(current_price):
            recommendations["Reason"] = "Current price not available for adaptive analysis."
            return recommendations

        market_regime = classify_market_regime(data, adx_strong_trend_threshold, adx_no_trend_threshold)
        recommendations["Regime"] = market_regime
        # logging.info(f"Market Regime for {symbol}: {market_regime}") # Too verbose

        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2)
        recommendations["Reason"] = reason_text

        final_recommendation = "Hold" # Default

        # Get indicator values defensively
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns and pd.notnull(data['SMA_50'].iloc[-1]) else np.nan
        sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns and pd.notnull(data['SMA_200'].iloc[-1]) else np.nan
        macd = data['MACD'].iloc[-1] if 'MACD' in data.columns and pd.notnull(data['MACD'].iloc[-1]) else np.nan
        macd_signal = data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns and pd.notnull(data['MACD_signal'].iloc[-1]) else np.nan
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else np.nan
        dmp = data['DMP'].iloc[-1] if 'DMP' in data.columns and pd.notnull(data['DMP'].iloc[-1]) else np.nan
        dmn = data['DMN'].iloc[-1] if 'DMN' in data.columns and pd.notnull(data['DMN'].iloc[-1]) else np.nan
        adx_slope = data['ADX_Slope'].iloc[-1] if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]) else np.nan
        cmf = data['CMF'].iloc[-1] if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]) else np.nan
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) else np.nan
        ichimoku_span_a = data['Ichimoku_Span_A'].iloc[-1] if 'Ichimoku_Span_A' in data.columns and pd.notnull(data['Ichimoku_Span_A'].iloc[-1]) else np.nan
        ichimoku_span_b = data['Ichimoku_Span_B'].iloc[-1] if 'Ichimoku_Span_B' in data.columns and pd.notnull(data['Ichimoku_Span_B'].iloc[-1]) else np.nan
        volume_spike = data['Volume_Spike'].iloc[-1] if 'Volume_Spike' in data.columns and pd.notnull(data['Volume_Spike'].iloc[-1]) else False
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        atr_pct = data['ATR_pct'].iloc[-1] if 'ATR_pct' in data.columns and pd.notnull(data['ATR_pct'].iloc[-1]) else np.nan
        ema_20 = data['EMA_20'].iloc[-1] if 'EMA_20' in data.columns and pd.notnull(data['EMA_20'].iloc[-1]) else np.nan


        # --- Market Condition Filters (No-Trade Zones) ---
        # 1. Very Low Volatility (Sideways/Choppy)
        if pd.notnull(atr_pct) and atr_pct < atr_low_volatility_pct:
            recommendations["Reason"] = "Market is in a very low volatility / tight range, not ideal for trend-following."
            return recommendations # Return "Hold" with specific reason
        # 2. Extremely High Volatility (Too Risky)
        if pd.notnull(atr_pct) and atr_pct > atr_high_volatility_pct:
            recommendations["Reason"] = "Market is in an extremely high volatility regime, increased risk."
            return recommendations # Return "Hold" with specific reason
        # 3. No Clear Trend (ADX below threshold and not rising)
        if pd.notnull(adx) and adx < adx_no_trend_threshold and (pd.notnull(adx_slope) and adx_slope <= 0):
            recommendations["Reason"] = f"ADX({int(adx)}) indicates no clear trend, avoiding whipsaws."
            return recommendations # Return "Hold" with specific reason


        # --- Trend-Following Strategy Logic ---
        # Define bullish conditions
        is_uptrend_sma = (current_price > sma_50 and sma_50 > sma_200) and (pd.notnull(sma_50) and pd.notnull(sma_200))
        is_macd_bullish = (macd > macd_signal) and (pd.notnull(macd) and pd.notnull(macd_signal))
        if macd_zero_line_confirm: # Only confirm if above zero line if configured
            is_macd_bullish = is_macd_bullish and (macd > 0)
        is_strong_adx_bullish = (adx > adx_strong_trend_threshold and dmp > dmn and adx_slope > 0) and (pd.notnull(adx) and pd.notnull(dmp) and pd.notnull(dmn) and pd.notnull(adx_slope)) # ADX rising
        is_ichimoku_bullish = (current_price > max(ichimoku_span_a, ichimoku_span_b)) and (pd.notnull(ichimoku_span_a) and pd.notnull(ichimoku_span_b))
        is_cmf_positive = (cmf > cmf_buy_threshold) and pd.notnull(cmf) # Stronger buying pressure
        is_rsi_not_overbought = (rsi < rsi_overbought) and pd.notnull(rsi) # Room to run

        # Define bearish conditions
        is_downtrend_sma = (current_price < sma_50 and sma_50 < sma_200) and (pd.notnull(sma_50) and pd.notnull(sma_200))
        is_macd_bearish = (macd < macd_signal) and (pd.notnull(macd) and pd.notnull(macd_signal))
        if macd_zero_line_confirm: # Only confirm if below zero line if configured
            is_macd_bearish = is_macd_bearish and (macd < 0)
        is_strong_adx_bearish = (adx > adx_strong_trend_threshold and dmn > dmp and adx_slope > 0) and (pd.notnull(adx) and pd.notnull(dmp) and pd.notnull(dmn) and pd.notnull(adx_slope)) # ADX rising in bearish direction
        is_ichimoku_bearish = (current_price < min(ichimoku_span_a, ichimoku_span_b)) and (pd.notnull(ichimoku_span_a) and pd.notnull(ichimoku_span_b))
        is_cmf_negative = (cmf < cmf_sell_threshold) and pd.notnull(cmf) # Stronger selling pressure
        is_rsi_overbought = (rsi > rsi_overbought) and pd.notnull(rsi) # At risk of pullback

        # Check for volume on potential signals
        is_volume_confirming_buy = volume_spike or \
                                   (current_price > data['Close'].iloc[-2] and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]) if len(data) >= 2 and 'Avg_Volume' in data.columns else False
        is_volume_confirming_sell = volume_spike or \
                                    (current_price < data['Close'].iloc[-2] and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]) if len(data) >= 2 and 'Avg_Volume' in data.columns else False


        # --- BUY Logic ---
        # Strong Buy: Confirmed Trend Entry with Pullback/Value
        if (is_uptrend_sma and is_macd_bullish and is_strong_adx_bullish and
            is_ichimoku_bullish and is_cmf_positive and is_rsi_not_overbought and is_volume_confirming_buy):
            # Price must be at a reasonable entry point (e.g., near EMA20/50 after a pullback)
            if pd.notnull(ema_20) and atr > 0 and \
               (abs(current_price - ema_20) / atr < atr_buy_pullback_factor) and \
               (current_price > data['Open'].iloc[-1]): # Ensure today's price is not crashing
                final_recommendation = "Strong Buy (Confirmed Trend & Value)"
        # General Buy: Trend Following
        elif (is_uptrend_sma and is_macd_bullish and adx > adx_no_trend_threshold and is_rsi_not_overbought and is_volume_confirming_buy):
             final_recommendation = "Buy (Trend Following)"
        # Breakout Buy: Price breaking above Bollinger Upper Band with volume
        elif 'Upper_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]) and \
             current_price > data['Upper_Band'].iloc[-1] and volume_spike and is_rsi_not_overbought:
            final_recommendation = "Buy (Breakout with Volume)"
        # Consider Buy: Developing Trend
        elif is_uptrend_sma and (is_macd_bullish or is_ichimoku_bullish) and is_rsi_not_overbought:
            final_recommendation = "Consider Buy (Developing Trend)"


        # --- SELL Logic ---
        # Strong Sell: Confirmed Trend Reversal / Bearish Trend Entry
        if (is_downtrend_sma and is_macd_bearish and is_strong_adx_bearish and
            is_ichimoku_bearish and is_cmf_negative and is_rsi_overbought and is_volume_confirming_sell):
            final_recommendation = "Strong Sell (Confirmed Trend Reversal)"
        # General Sell: Trend Reversal or Weakening
        elif is_downtrend_sma and (is_macd_bearish or is_ichimoku_bearish or is_strong_adx_bearish):
            # If currently recommending Buy/Hold, this downgrades it to Sell
            if final_recommendation.startswith("Buy") or final_recommendation == "Hold":
                final_recommendation = "Sell (Trend Reversal)"
            else:
                final_recommendation = "Strong Sell (Trend Reversal)"
        # Overbought Exit (even if trend is still up, implies profit taking)
        elif is_rsi_overbought and not final_recommendation.startswith("Sell"): # Don't override stronger sell signals
            if final_recommendation.startswith("Buy"):
                final_recommendation = "Hold (Overbought, Consider Profit Booking)"
            elif final_recommendation == "Hold":
                final_recommendation = "Sell (Overbought)"
            else: # Already a sell, reinforce
                final_recommendation = "Strong Sell (Extremely Overbought)"


        recommendations["Recommendation"] = final_recommendation

        # Pass parameters to calculation functions
        buy_at = calculate_buy_at(data, atr_factor=atr_buy_pullback_factor)
        stop_loss = calculate_stop_loss(data, atr_multiplier=atr_exit_multiplier,
                                       adx_high_threshold=adx_strong_trend_threshold,
                                       adx_low_threshold=adx_no_trend_threshold)
        target = calculate_target(data, risk_reward_ratio=risk_reward_ratio,
                                  adx_high_threshold=adx_strong_trend_threshold,
                                  adx_low_threshold=adx_no_trend_threshold)

        recommendations["Buy At"] = buy_at
        recommendations["Stop Loss"] = stop_loss
        recommendations["Target"] = target

        risk_capital = equity * (risk_per_trade_pct / 100)

        # Position sizing calculation
        if buy_at is not None and stop_loss is not None and buy_at > stop_loss:
            risk_per_share = buy_at - stop_loss
            if risk_per_share > 0:
                position_shares = int(risk_capital / risk_per_share)
                position_value = position_shares * current_price

                # Cap position size to max_position_pct of equity to manage concentration risk
                if current_price > 0 and position_value > equity * max_position_pct: # Use max position pct from UI
                    position_value = equity * max_position_pct
                    position_shares = int(position_value / current_price) if current_price > 0 else 0

                recommendations["Position Size"] = {"shares": position_shares, "value": round(position_value, 2)}
            else:
                recommendations["Position Size"] = {"shares": 0, "value": 0}
        else:
             recommendations["Position Size"] = {"shares": 0, "value": 0}

        # Trailing stop only calculated if a potential buy signal (for display purposes)
        if recommendations["Recommendation"].lower().startswith("buy"):
             recommendations["Trailing Stop"] = calculate_trailing_stop(current_price, data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else None, atr_multiplier=atr_exit_multiplier) # Use new atr_exit_multiplier
        else:
             recommendations["Trailing Stop"] = None

        return recommendations

    except Exception as e:
        logging.error(f"Critical error in adaptive_recommendation for {symbol}: {str(e)}")
        recommendations["Reason"] = f"An unexpected error occurred: {str(e)}"
        return recommendations

# ENHANCED generate_recommendations for Standard Mode
def generate_recommendations(data, symbol=None):
    """
    Generates trading recommendations based on technical indicators (Standard Mode).
    Enhanced for better performance, applying trend-following principles with default parameters.
    """
    recommendations = {
        "Intraday": "Hold", # Will be the main consolidated recommendation for standard mode
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
        "Reason": "N/A" # Added reason for standard mode as well
    }

    # Ensure sufficient data for Ichimoku (longest indicator window)
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        recommendations["Reason"] = "Insufficient historical data for comprehensive standard analysis."
        logging.warning(f"Invalid data for standard recommendations: {symbol}")
        return recommendations

    try:
        current_price = round(float(data['Close'].iloc[-1]), 2)
        recommendations["Current Price"] = current_price

        if pd.isna(current_price):
            recommendations["Reason"] = "Current price not available for standard analysis."
            return recommendations

        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2)
        recommendations["Reason"] = reason_text # Initial reason from score components

        # Get indicator values defensively (using hardcoded defaults implicitly from analyze_stock)
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns and pd.notnull(data['SMA_50'].iloc[-1]) else np.nan
        sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns and pd.notnull(data['SMA_200'].iloc[-1]) else np.nan
        macd = data['MACD'].iloc[-1] if 'MACD' in data.columns and pd.notnull(data['MACD'].iloc[-1]) else np.nan
        macd_signal = data['MACD_signal'].iloc[-1] if 'MACD_signal' in data.columns and pd.notnull(data['MACD_signal'].iloc[-1]) else np.nan
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else np.nan
        dmp = data['DMP'].iloc[-1] if 'DMP' in data.columns and pd.notnull(data['DMP'].iloc[-1]) else np.nan
        dmn = data['DMN'].iloc[-1] if 'DMN' in data.columns and pd.notnull(data['DMN'].iloc[-1]) else np.nan
        adx_slope = data['ADX_Slope'].iloc[-1] if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]) else np.nan
        cmf = data['CMF'].iloc[-1] if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]) else np.nan
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]) else np.nan
        ichimoku_span_a = data['Ichimoku_Span_A'].iloc[-1] if 'Ichimoku_Span_A' in data.columns and pd.notnull(data['Ichimoku_Span_A'].iloc[-1]) else np.nan
        ichimoku_span_b = data['Ichimoku_Span_B'].iloc[-1] if 'Ichimoku_Span_B' in data.columns and pd.notnull(data['Ichimoku_Span_B'].iloc[-1]) else np.nan
        volume_spike = data['Volume_Spike'].iloc[-1] if 'Volume_Spike' in data.columns and pd.notnull(data['Volume_Spike'].iloc[-1]) else False
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else 0
        atr_pct = data['ATR_pct'].iloc[-1] if 'ATR_pct' in data.columns and pd.notnull(data['ATR_pct'].iloc[-1]) else np.nan

        # Define default thresholds for standard mode strategy
        std_adx_strong_trend_threshold = 25
        std_adx_no_trend_threshold = 20
        std_rsi_overbought = 70
        std_rsi_oversold = 30
        std_macd_zero_line_confirm = True # Default for standard mode
        std_cmf_buy_threshold = 0.10
        std_cmf_sell_threshold = -0.10
        std_atr_buy_pullback_factor = 0.5 # Default for standard mode
        std_atr_exit_multiplier = 2.0 # Default for standard mode

        # Consolidate general market conditions (copied from adaptive, but with fixed thresholds)
        if pd.notnull(atr_pct) and atr_pct < 0.005: # Very Low Volatility
            recommendations["Reason"] = "Market in low volatility range, standard strategy holds."
            return recommendations
        if pd.notnull(atr_pct) and atr_pct > 0.045: # Extremely High Volatility
            recommendations["Reason"] = "Market in extremely high volatility, standard strategy holds."
            return recommendations
        if pd.notnull(adx) and adx < std_adx_no_trend_threshold and (pd.notnull(adx_slope) and adx_slope <= 0):
            recommendations["Reason"] = f"ADX({int(adx)}) indicates no clear trend, standard strategy holds."
            return recommendations

        # Define standard bullish/bearish conditions based on fixed parameters
        is_uptrend_sma = (current_price > sma_50 and sma_50 > sma_200) and (pd.notnull(sma_50) and pd.notnull(sma_200))
        is_macd_bullish = (macd > macd_signal) and (pd.notnull(macd) and pd.notnull(macd_signal))
        if std_macd_zero_line_confirm: is_macd_bullish = is_macd_bullish and (macd > 0)
        is_strong_adx_bullish = (adx > std_adx_strong_trend_threshold and dmp > dmn and adx_slope > 0) and pd.notnull(adx_slope)
        is_ichimoku_bullish = (current_price > max(ichimoku_span_a, ichimoku_span_b)) and (pd.notnull(ichimoku_span_a) and pd.notnull(ichimoku_span_b))
        is_cmf_positive = (cmf > std_cmf_buy_threshold) and pd.notnull(cmf)
        is_rsi_not_overbought = (rsi < std_rsi_overbought) and pd.notnull(rsi)

        is_downtrend_sma = (current_price < sma_50 and sma_50 < sma_200) and (pd.notnull(sma_50) and pd.notnull(sma_200))
        is_macd_bearish = (macd < macd_signal) and (pd.notnull(macd) and pd.notnull(macd_signal))
        if std_macd_zero_line_confirm: is_macd_bearish = is_macd_bearish and (macd < 0)
        is_strong_adx_bearish = (adx > std_adx_strong_trend_threshold and dmn > dmp and adx_slope > 0) and pd.notnull(adx_slope)
        is_ichimoku_bearish = (current_price < min(ichimoku_span_a, ichimoku_span_b)) and (pd.notnull(ichimoku_span_a) and pd.notnull(ichimoku_span_b))
        is_cmf_negative = (cmf < std_cmf_sell_threshold) and pd.notnull(cmf)
        is_rsi_overbought = (rsi > std_rsi_overbought) and pd.notnull(rsi)

        is_volume_confirming_buy = volume_spike or \
                                   (current_price > data['Close'].iloc[-2] and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]) if len(data) >= 2 and 'Avg_Volume' in data.columns else False
        is_volume_confirming_sell = volume_spike or \
                                    (current_price < data['Close'].iloc[-2] and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1]) if len(data) >= 2 and 'Avg_Volume' in data.columns else False

        final_recommendation = "Hold" # Default

        # BUY Logic (Standard)
        if (is_uptrend_sma and is_macd_bullish and is_strong_adx_bullish and
            is_ichimoku_bullish and is_cmf_positive and is_rsi_not_overbought and is_volume_confirming_buy):
            final_recommendation = "Strong Buy (Trend Confirmation)"
        elif (is_uptrend_sma and (is_macd_bullish or is_ichimoku_bullish) and is_rsi_not_overbought and is_volume_confirming_buy):
            final_recommendation = "Buy (Developing Trend)"
        elif ('Upper_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]) and
              current_price > data['Upper_Band'].iloc[-1] and volume_spike and is_rsi_not_overbought):
            final_recommendation = "Buy (Breakout)"

        # SELL Logic (Standard)
        if (is_downtrend_sma and is_macd_bearish and is_strong_adx_bearish and
            is_ichimoku_bearish and is_cmf_negative):
            # Overrides any previous buy signal
            final_recommendation = "Strong Sell (Trend Reversal)"
        elif (is_downtrend_sma and (is_macd_bearish or is_ichimoku_bearish)):
            if final_recommendation.startswith("Buy") or final_recommendation == "Hold":
                final_recommendation = "Sell (Trend Weakening)"
        elif is_rsi_overbought: # Profit-taking or caution
            if final_recommendation.startswith("Buy"):
                final_recommendation = "Hold (Overbought Caution)"
            elif final_recommendation == "Hold":
                final_recommendation = "Sell (Overbought)"
            # If already a Sell, reinforce or keep stronger signal.

        recommendations["Intraday"] = final_recommendation # Use this for the primary pick
        recommendations["Swing"] = final_recommendation
        recommendations["Short-Term"] = final_recommendation
        recommendations["Long-Term"] = final_recommendation
        recommendations["Mean_Reversion"] = "Hold" # Removed explicit mean reversion logic for simplicity in standard
        recommendations["Breakout"] = "Hold" if "Breakout" not in final_recommendation else final_recommendation # To isolate if Breakout was the main signal
        recommendations["Ichimoku_Trend"] = "Hold" if "Ichimoku" not in final_recommendation else final_recommendation

        # Calculate Buy At, Stop Loss, Target using fixed defaults for standard mode
        recommendations["Buy At"] = calculate_buy_at(data, atr_factor=0.2) # Hardcoded ATR factor for standard mode
        recommendations["Stop Loss"] = calculate_stop_loss(data, atr_multiplier=3.0, # Hardcoded multiplier
                                                           adx_high_threshold=25, adx_low_threshold=20) # Hardcoded ADX thresholds
        recommendations["Target"] = calculate_target(data, risk_reward_ratio=2.5, # Hardcoded R:R
                                                     adx_high_threshold=25, adx_low_threshold=20) # Hardcoded ADX thresholds

        return recommendations

    except Exception as e:
        logging.error(f"Error generating standard recommendations for {symbol}: {str(e)}")
        recommendations["Reason"] = f"An unexpected error occurred during standard analysis: {str(e)}"
        return recommendations

@st.cache_data(ttl=3600)
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None, # Added _data_hash for caching
                   # Removed indicator parameters, only strategy parameters remain
                   adx_strong_trend_threshold=25, adx_no_trend_threshold=20,
                   rsi_overbought=70, rsi_oversold=30, macd_zero_line_confirm=True,
                   cmf_buy_threshold=0.10, cmf_sell_threshold=-0.10,
                   atr_buy_pullback_factor=0.5, atr_exit_multiplier=2.0,
                   atr_low_volatility_pct=0.005, atr_high_volatility_pct=0.045,
                   risk_reward_ratio=2.5, max_position_pct=0.25, risk_per_trade_pct=1.0): # Added strategy params to backtest
    """
    Backtests a given strategy on historical data.
    Now accepts strategy parameters for backtesting.
    """
    results = {
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "trades": 0,
        "win_rate": 0,
        "buy_signals": [],
        "sell_signals": [],
        "trade_details": [],
        "total_profit_amount": 0.0,
        "total_loss_amount": 0.0
    }
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')

    position = None
    entry_price = 0
    entry_date_str = None
    current_trailing_stop = None
    trades = []
    returns = []
    total_profit_amount = 0.0
    total_loss_amount = 0.0

    # Ensure min_data_for_backtest_analysis is calculated based on INDICATOR_MIN_LENGTHS
    min_data_for_backtest_analysis = max(list(INDICATOR_MIN_LENGTHS.values())) + 1

    if len(data) < min_data_for_backtest_analysis:
        logging.warning(f"Not enough data to backtest {symbol}. Need at least {min_data_for_backtest_analysis} rows, got {len(data)}")
        return results

    # Re-analyze data with default parameters for backtesting context
    full_analyzed_data = analyze_stock(data.copy())

    if full_analyzed_data.empty or len(full_analyzed_data) < min_data_for_backtest_analysis:
        logging.warning(f"Analyzed data for {symbol} is insufficient for backtesting. Len: {len(full_analyzed_data)}")
        return results

    # Start backtesting from the point where all indicators are valid
    start_index = full_analyzed_data.first_valid_index()
    if start_index is None: # If no valid index after analysis
        return results

    # Find the integer index corresponding to start_index
    start_int_index = full_analyzed_data.index.get_loc(start_index)

    # Loop from the first valid indicator point up to the second to last day for signals
    for i in range(start_int_index, len(full_analyzed_data) - 1):
        # Data available *up to and including* day 'i' is used to generate the signal for day 'i+1'
        sliced_data_for_signal = full_analyzed_data.iloc[:i+1]

        # Data for day 'i+1' (the trade execution day)
        current_day_data = full_analyzed_data.iloc[i+1]

        # Ensure sliced data is sufficient for signal generation (should be caught by initial validate_data, but defensive)
        if not validate_data(sliced_data_for_signal, min_length=max(list(INDICATOR_MIN_LENGTHS.values()))):
            continue

        current_equity_for_rec = st.session_state.get('initial_capital', 50000) # Use the UI-set capital

        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data_for_signal, symbol=symbol, equity=current_equity_for_rec,
                                          adx_strong_trend_threshold=adx_strong_trend_threshold,
                                          adx_no_trend_threshold=adx_no_trend_threshold,
                                          rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
                                          macd_zero_line_confirm=macd_zero_line_confirm,
                                          cmf_buy_threshold=cmf_buy_threshold, cmf_sell_threshold=cmf_sell_threshold,
                                          atr_buy_pullback_factor=atr_buy_pullback_factor,
                                          atr_exit_multiplier=atr_exit_multiplier,
                                          atr_low_volatility_pct=atr_low_volatility_pct,
                                          atr_high_volatility_pct=atr_high_volatility_pct,
                                          risk_reward_ratio=risk_reward_ratio,
                                          max_position_pct=max_position_pct,
                                          risk_per_trade_pct=risk_per_trade_pct)
            signal_type = rec["Recommendation"]
        else:
            rec = generate_recommendations(sliced_data_for_signal, symbol) # Standard mode uses its own fixed params
            signal_type = rec[strategy] if strategy in rec else "Hold"

        if signal_type is None:
            continue

        trade_open_price = current_day_data['Open']
        trade_high_price = current_day_data['High']
        trade_low_price = current_day_data['Low']
        trade_close_price = current_day_data['Close']
        trade_date_str = current_day_data.name.strftime('%Y-%m-%d')

        slippage_pct = random.uniform(0.001, 0.005)

        # --- Entry Logic ---
        if "Buy" in signal_type and position is None:
            entry_price_with_slippage = trade_open_price * (1 + slippage_pct)
            rec_buy_at = rec.get("Buy At")

            # Refined entry check: only enter if actual open is within a reasonable range of recommended buy_at
            if pd.notnull(rec_buy_at) and (abs(entry_price_with_slippage - rec_buy_at) / rec_buy_at > 0.02):
                continue

            position = "Long"
            entry_price = entry_price_with_slippage
            entry_date_str = trade_date_str
            results["buy_signals"].append((trade_date_str, entry_price))
            current_trailing_stop = None # Reset for new position

        # --- Exit Logic ---
        elif position == "Long":
            exit_reason = None
            exit_price = trade_close_price

            # Update trailing stop daily if in profit
            if 'ATR' in current_day_data and pd.notnull(current_day_data['ATR']) and pd.notnull(current_day_data['Close']):
                # Only trail if current price is above entry price, and ATR > 0
                if current_day_data['Close'] > entry_price and current_day_data['ATR'] > 0:
                    current_trailing_stop = calculate_trailing_stop(current_day_data['Close'], current_day_data['ATR'], atr_multiplier=atr_exit_multiplier, prev_trailing_stop=current_trailing_stop)
                elif current_trailing_stop is None and current_day_data['ATR'] > 0: # Set initial trailing stop if not yet in profit
                     current_trailing_stop = entry_price - (current_day_data['ATR'] * atr_exit_multiplier)

            # Retrieve dynamic stop loss and target from current day's analysis (slice_data_for_signal)
            # Pass relevant parameters for consistent calculation during backtest
            stop_loss_price = calculate_stop_loss(sliced_data_for_signal, atr_multiplier=atr_exit_multiplier,
                                                  adx_high_threshold=adx_strong_trend_threshold,
                                                  adx_low_threshold=adx_no_trend_threshold)
            target_price = calculate_target(sliced_data_for_signal, risk_reward_ratio=risk_reward_ratio,
                                            adx_high_threshold=adx_strong_trend_threshold,
                                            adx_low_threshold=adx_no_trend_threshold)


            # 1. Check for Stop Loss / Trailing Stop / Target Hit (prioritize these)
            # Use 'Low' for Stop Loss checks, 'High' for Target checks
            if stop_loss_price is not None and trade_low_price <= stop_loss_price:
                exit_price = stop_loss_price * (1 - slippage_pct)
                exit_reason = "Stop Loss Hit"
            elif current_trailing_stop is not None and trade_low_price <= current_trailing_stop:
                exit_price = current_trailing_stop * (1 - slippage_pct)
                exit_reason = "Trailing Stop Hit"
            elif target_price is not None and trade_high_price >= target_price:
                exit_price = target_price * (1 - slippage_pct)
                exit_reason = "Target Hit"
            # 2. Check for explicit Sell Signal from the strategy only if other exits not triggered
            elif "Sell" in signal_type and exit_reason is None:
                exit_price = trade_close_price * (1 - slippage_pct)
                exit_reason = "Sell Signal"

            if exit_reason:
                position = None
                profit = exit_price - entry_price

                if entry_price != 0:
                    returns.append(profit / entry_price)

                if profit > 0:
                    total_profit_amount += profit
                else:
                    total_loss_amount += profit

                trades.append({
                    "entry_date": entry_date_str,
                    "entry_price": entry_price,
                    "exit_date": trade_date_str,
                    "exit_price": exit_price,
                    "profit": profit,
                    "reason": exit_reason
                })
                results["sell_signals"].append((trade_date_str, exit_price))
                entry_price = 0
                entry_date_str = None
                current_trailing_stop = None

    if position == "Long":
        final_close_price = full_analyzed_data['Close'].iloc[-1]
        exit_price = final_close_price * (1 - slippage_pct)
        profit = exit_price - entry_price
        if entry_price != 0:
            returns.append(profit / entry_price)

        if profit > 0:
            total_profit_amount += profit
        else:
            total_loss_amount += profit

        trades.append({
            "entry_date": entry_date_str,
            "entry_price": entry_price,
            "exit_date": full_analyzed_data.index[-1].strftime('%Y-%m-%d'),
            "exit_price": exit_price,
            "profit": profit,
            "reason": "Closed at end of period"
        })
        results["sell_signals"].append((full_analyzed_data.index[-1].strftime('%Y-%m-%d'), exit_price))

    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)

        results["total_profit_amount"] = total_profit_amount
        results["total_loss_amount"] = total_loss_amount

        total_growth_factor = 1.0
        for r in returns:
            total_growth_factor *= (1 + r)
        results["total_return"] = (total_growth_factor - 1) * 100

        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100

        if returns:
            returns_series = pd.Series(returns)
            results["annual_return"] = (returns_series.mean() * 252) * 100

            if returns_series.std() != 0:
                results["sharpe_ratio"] = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                results["sharpe_ratio"] = 0

            equity_curve_values = [100]
            for r in returns:
                equity_curve_values.append(equity_curve_values[-1] * (1 + r))
            equity_curve = pd.Series(equity_curve_values)

            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            results["max_drawdown"] = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            results["annual_return"] = 0
            results["sharpe_ratio"] = 0
            results["max_drawdown"] = 0

    return results

def analyze_batch(stock_batch):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    # Check if the result is an error dict
                    if isinstance(result, dict) and "Error" in result:
                        errors.append(f"[{symbol}] {result['Error']}")
                    else: # It's a valid analysis result
                        results.append(result)
            except Exception as e:
                error_msg = f"[{symbol}] Unhandled exception during analysis: {str(e)}"
                logging.error(error_msg)
                errors.append(error_msg)
    return results, errors # Return both results and errors

# Reverted analyze_stock_parallel to use hardcoded indicator parameters
def analyze_stock_parallel(symbol):
    # Retrieve only strategy parameters from session state (if they are still configurable)
    adx_strong_trend_threshold = st.session_state.get('adx_strong_trend_threshold', 25)
    adx_no_trend_threshold = st.session_state.get('adx_no_trend_threshold', 20)
    rsi_overbought = st.session_state.get('rsi_overbought', 70)
    rsi_oversold = st.session_state.get('rsi_oversold', 30)
    macd_zero_line_confirm = st.session_state.get('macd_zero_line_confirm', True)
    cmf_buy_threshold = st.session_state.get('cmf_buy_threshold', 0.10)
    cmf_sell_threshold = st.session_state.get('cmf_sell_threshold', -0.10)
    atr_buy_pullback_factor = st.session_state.get('atr_buy_pullback_factor', 0.5)
    atr_exit_multiplier = st.session_state.get('atr_exit_multiplier', 2.0)
    atr_low_volatility_pct = st.session_state.get('atr_low_volatility_pct', 0.005)
    atr_high_volatility_pct = st.session_state.get('atr_high_volatility_pct', 0.045)
    risk_reward_ratio = st.session_state.get('risk_reward_ratio', 2.5)
    max_position_pct = st.session_state.get('max_position_pct', 0.25)
    risk_per_trade_pct = st.session_state.get('risk_per_trade_pct', 1.0)

    result_dict = {
        "Symbol": symbol,
        "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0,
        "Recommendation": None, "Regime": None,
        "Position Size Shares": None, "Position Size Value": None, "Trailing Stop": None, "Reason": None,
        "Intraday": None, "Swing": None, "Short-Term": None, "Long-Term": None,
        "Mean_Reversion": None, "Breakout": None, "Ichimoku_Trend": None,
        "Error": None
    }
    try:
        data = fetch_stock_data_cached(symbol)

        # required_min_length is now based on fixed INDICATOR_MIN_LENGTHS values
        required_min_length = max(list(INDICATOR_MIN_LENGTHS.values())) + 1

        if data.empty or len(data) < required_min_length:
            error_msg = f"Insufficient data ({len(data)} rows) for comprehensive analysis (need at least {required_min_length})."
            logging.warning(f"No sufficient data for {symbol} after fetch: {error_msg}")
            result_dict["Error"] = error_msg
            return result_dict

        # Call analyze_stock without passing indicator parameters (uses internal defaults)
        data = analyze_stock(data)

        if data.empty or pd.isna(data['Close'].iloc[-1]) or ('ATR' in data.columns and pd.isna(data['ATR'].iloc[-1])):
             error_msg = "Final analyzed data incomplete (missing Close/ATR/other critical indicators)."
             logging.warning(f"Final analyzed data for {symbol} is incomplete: {error_msg}")
             result_dict["Error"] = error_msg
             return result_dict

        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        current_equity_for_rec = st.session_state.get('initial_capital', 50000)


        if recommendation_mode == "Adaptive":
            # Pass all relevant strategy parameters to adaptive_recommendation
            rec = adaptive_recommendation(data, symbol, equity=current_equity_for_rec,
                                          adx_strong_trend_threshold=adx_strong_trend_threshold,
                                          adx_no_trend_threshold=adx_no_trend_threshold,
                                          rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
                                          macd_zero_line_confirm=macd_zero_line_confirm,
                                          cmf_buy_threshold=cmf_buy_threshold, cmf_sell_threshold=cmf_sell_threshold,
                                          atr_buy_pullback_factor=atr_buy_pullback_factor,
                                          atr_exit_multiplier=atr_exit_multiplier,
                                          atr_low_volatility_pct=atr_low_volatility_pct,
                                          atr_high_volatility_pct=atr_high_volatility_pct,
                                          risk_reward_ratio=risk_reward_ratio,
                                          max_position_pct=max_position_pct,
                                          risk_per_trade_pct=risk_per_trade_pct)
            if not rec or not rec.get('Recommendation'):
                error_msg = "Adaptive analysis failed or returned incomplete recommendation."
                logging.error(f"Invalid adaptive_recommendation output for {symbol}: {rec}")
                result_dict["Error"] = error_msg
                return result_dict

            position_size = rec.get("Position Size", {"shares": 0, "value": 0})

            result_dict.update({
                "Current Price": rec.get("Current Price"), "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"), "Target": rec.get("Target"),
                "Recommendation": rec.get("Recommendation", "Hold"), "Score": rec.get("Score", 0),
                "Regime": rec.get("Regime"),
                "Position Size Shares": position_size.get("shares", 0),
                "Position Size Value": position_size.get("value", 0),
                "Trailing Stop": rec.get("Trailing Stop"), "Reason": rec.get("Reason"),
            })
        else: # Standard mode
            rec = generate_recommendations(data, symbol) # Standard mode uses its own fixed params
            if not rec or not rec.get('Intraday'):
                error_msg = "Standard analysis failed or returned incomplete recommendation."
                logging.error(f"Invalid generate_recommendations output for {symbol}: {rec}")
                result_dict["Error"] = error_msg
                return result_dict

            result_dict.update({
                "Current Price": rec.get("Current Price"), "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"), "Target": rec.get("Target"),
                "Intraday": rec.get("Intraday", "Hold"), "Swing": rec.get("Swing", "Hold"),
                "Short-Term": rec.get("Short-Term", "Hold"), "Long-Term": rec.get("Long-Term", "Hold"),
                "Mean_Reversion": None, "Breakout": None, "Ichimoku_Trend": None, # These will now be consolidated
                "Score": rec.get("Score", 0),
                "Reason": rec.get("Reason", "No reason provided") # Include reason for standard mode
            })
        return result_dict

    except Exception as e:
        error_msg = f"Unexpected error during analysis: {str(e)}"
        logging.error(f"Error in analyze_stock_parallel for {symbol}: {error_msg}")
        result_dict["Error"] = error_msg
        return result_dict

# Modified analyze_all_stocks to accept Streamlit UI objects directly
def analyze_all_stocks(stock_list, batch_size=4, progress_bar_obj=None, loading_text_obj=None, status_text_obj=None):
    results = []
    all_errors = []
    total_stocks = len(stock_list)
    processed = 0
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        if status_text_obj:
            batch_names = ", ".join(batch[:3])
            if len(batch) > 3:
                batch_names += f" and {len(batch)-3} more"
            status_text_message = f"🔄 Analyzing: {batch_names}"
            status_text_obj.text(status_text_message)

        batch_results, batch_errors = analyze_batch(batch)
        results.extend([r for r in batch_results if r is not None])
        all_errors.extend(batch_errors)
        processed += len(batch)

        if progress_bar_obj and loading_text_obj:
            progress_value = processed / total_stocks
            progress_bar_obj.progress(progress_value)
            percentage = int(progress_value * 100)
            loading_text_obj.text(f"Progress: {percentage}%")

        time.sleep(max(2, batch_size / 5))

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning("No valid stock data retrieved from batch analysis.")
        return pd.DataFrame(), all_errors

    expected_cols = [
        "Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Score",
        "Recommendation", "Regime", "Position Size Shares", "Position Size Value",
        "Trailing Stop", "Reason", "Intraday", "Swing", "Short-Term", "Long-Term",
        "Mean_Reversion", "Breakout", "Ichimoku_Trend", "Error"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan
        if results_df[col].dtype == 'object':
             results_df[col] = results_df[col].astype(str)

    return results_df.sort_values(by="Score", ascending=False), all_errors

# Modified analyze_intraday_stocks to accept Streamlit UI objects directly
def analyze_intraday_stocks(stock_list, batch_size=3, progress_bar_obj=None, loading_text_obj=None, status_text_obj=None):
    results = []
    all_errors = []
    total_stocks = len(stock_list)
    processed = 0

    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]

        if status_text_obj:
            batch_names = ", ".join(batch[:3])
            if len(batch) > 3:
                batch_names += f" and {len(batch)-3} more"
            status_text_message = f"🔄 Analyzing: {batch_names}"
            status_text_obj.text(status_text_message)

        batch_results, batch_errors = analyze_batch(batch)
        results.extend([r for r in batch_results if r is not None])
        all_errors.extend(batch_errors)

        processed += len(batch)
        if progress_bar_obj and loading_text_obj:
            progress_value = processed / total_stocks
            progress_bar_obj.progress(progress_value)
            percentage = int(progress_value * 100)
            loading_text_obj.text(f"Progress: {percentage}%")

        time.sleep(max(10, batch_size * 2.0 / 3))

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logging.warning("No valid stock data retrieved for intraday analysis.")
        return pd.DataFrame(), all_errors

    expected_cols = [
        "Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Score",
        "Recommendation", "Regime", "Position Size Shares", "Position Size Value",
        "Trailing Stop", "Reason", "Intraday", "Swing", "Short-Term", "Long-Term",
        "Mean_Reversion", "Breakout", "Ichimoku_Trend", "Error"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan
        if results_df[col].dtype == 'object':
             results_df[col] = results_df[col].astype(str)


    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        filtered_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False, case=False) & results_df['Error'].isnull()]
    else:
        if "Intraday" in results_df.columns:
            filtered_df = results_df[results_df["Intraday"].str.contains("Buy", na=False, case=False) & results_df['Error'].isnull()]
        else:
            logging.error("Intraday column not found in results_df during intraday filtering.")
            filtered_df = pd.DataFrame()

    return filtered_df.sort_values(by="Score", ascending=False).head(5), all_errors

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



def insert_top_picks_supabase(results_df, pick_type="daily"):
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')

    if results_df.empty:
        logging.info(f"Input results_df is empty, no {pick_type} picks to insert into Supabase.")
        return

    filtered_df_pre_sort = pd.DataFrame()

    if recommendation_mode == "Adaptive":
        if 'Recommendation' not in results_df.columns:
            results_df['Recommendation'] = np.nan
        buy_condition = results_df["Recommendation"].astype(str).str.contains("Buy", na=False, case=False)
        filtered_df_pre_sort = results_df[buy_condition & results_df['Error'].isnull()]
    else:
        buy_condition = pd.Series([False] * len(results_df), index=results_df.index)
        # In standard mode, we filter by the main "Intraday" recommendation which now consolidates all signals
        if "Intraday" in results_df.columns:
            buy_condition = results_df["Intraday"].astype(str).str.contains("Buy", na=False, case=False)
        filtered_df_pre_sort = results_df[buy_condition & results_df['Error'].isnull()]

    if filtered_df_pre_sort.empty:
        logging.info(f"No 'Buy' signals found after initial filtering, no {pick_type} picks to insert into Supabase.")
        return

    filtered_df = filtered_df_pre_sort.sort_values(by="Score", ascending=False).head(5)

    if filtered_df.empty:
        logging.info(f"Filtered picks became empty after sorting and taking top 5. No {pick_type} picks to insert into Supabase.")
        return

    records_to_insert = []
    for _, row in filtered_df.iterrows():
        record = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "symbol": row.get('Symbol') or row.get('symbol', 'Unknown'),
            "score": (float(row.get('Score')) if pd.notnull(row.get('Score')) else None),
            "current_price": (float(row.get('Current Price')) if pd.notnull(row.get('Current Price')) else None),
            "buy_at": (float(row.get('Buy At')) if pd.notnull(row.get('Buy At')) else None),
            "stop_loss": (float(row.get('Stop Loss')) if pd.notnull(row.get('Stop Loss')) else None),
            "target": (float(row.get('Target')) if pd.notnull(row.get('Target')) else None),
            "intraday": row.get('Intraday', 'Hold'),
            "swing": row.get('Swing', 'Hold'),
            "short_term": row.get('Short-Term', 'Hold'),
            "long_term": row.get('Long-Term', 'Hold'),
            "mean_reversion": row.get('Mean_Reversion', 'Hold'),
            "breakout": row.get('Breakout', 'Hold'),
            "ichimoku_trend": row.get('Ichimoku_Trend', 'Hold'),
            "recommendation": row.get('Recommendation', 'Hold'),
            "regime": row.get('Regime', 'Unknown'),
            "position_size_shares": (float(row.get('Position Size Shares')) if pd.notnull(row.get('Position Size Shares')) else None),
            "position_size_value": (float(row.get('Position Size Value')) if pd.notnull(row.get('Position Size Value')) else None),
            "trailing_stop": (float(row.get('Trailing Stop')) if pd.notnull(row.get('Trailing Stop')) else None),
            "reason": row.get('Reason', 'No reason provided'),
            "pick_type": pick_type
        }
        records_to_insert.append(record)

    clean_records = []
    for record in records_to_insert:
        if not record.get('symbol') or not record.get('date'):
            logging.warning(f"Skipping record due to missing symbol or date: {record}")
            continue
        clean_records.append(record)

    if not clean_records:
        logging.warning("No clean records to insert into Supabase.")
        return

    try:
        logging.info(f"Attempting to upsert {len(clean_records)} records to Supabase table 'daily_picks'.")
        res = supabase.table("daily_picks").upsert(clean_records).execute()
        if hasattr(res, "data") and res.data:
            logging.info(f"Supabase upsert successful for {len(res.data)} records.")
        elif hasattr(res, "error") and res.error:
            logging.error(f"Supabase upsert error: {res.error['message']}")
            st.error(f"Supabase upsert error: {res.error['message']}")
        else:
            logging.warning(f"Supabase upsert response with no data or error key: {res}")

    except Exception as e:
        logging.error(f"Supabase upsert exception: {e}")
        st.error(f"Supabase upsert exception: {e}")


@RateLimiter(calls=1, period=1) # Rate limit this call to prevent burst for latest prices
def fetch_latest_price(symbol):
    data = fetch_stock_data_cached(symbol, period="2d", interval="1d")
    if not data.empty and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
        return float(data['Close'].iloc[-1])
    return None

def update_with_latest_prices(df):
    symbols_to_fetch = df['symbol'].unique()
    latest_prices = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(fetch_latest_price, sym): sym for sym in symbols_to_fetch}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                price = future.result()
                if price is not None:
                    latest_prices[symbol] = price
                else:
                    logging.warning(f"Failed to fetch latest price for {symbol}")
            except Exception as e:
                logging.warning(f"Error fetching latest price for {symbol}: {e}")

    updated_df = df.copy()
    updated_df['current_price'] = updated_df['symbol'].map(latest_prices)

    updated_df['current_price'] = updated_df['current_price'].fillna(df['current_price'])

    return updated_df


def display_dashboard(symbol=None, data=None, recommendations=None):
    # Initialize session state for ALL parameters and UI states
    if 'selected_sectors' not in st.session_state: st.session_state.selected_sectors = ["All"]
    if 'symbol' not in st.session_state: st.session_state.symbol = None
    if 'data' not in st.session_state: st.session_state.data = None
    if 'recommendations' not in st.session_state: st.session_state.recommendations = None
    if 'backtest_results_swing' not in st.session_state: st.session_state.backtest_results_swing = None
    if 'backtest_results_intraday' not in st.session_state: st.session_state.backtest_results_intraday = None
    if 'recommendation_mode' not in st.session_state: st.session_state.recommendation_mode = "Standard"
    if "show_history" not in st.session_state: st.session_state.show_history = False
    if 'initial_capital' not in st.session_state: st.session_state.initial_capital = 50000

    # Strategy Parameters (sidebar inputs) - These are KEPT configurable
    if 'adx_strong_trend_threshold' not in st.session_state: st.session_state.adx_strong_trend_threshold = 25
    if 'adx_no_trend_threshold' not in st.session_state: st.session_state.adx_no_trend_threshold = 20
    if 'rsi_overbought' not in st.session_state: st.session_state.rsi_overbought = 70
    if 'rsi_oversold' not in st.session_state: st.session_state.rsi_oversold = 30
    if 'macd_zero_line_confirm' not in st.session_state: st.session_state.macd_zero_line_confirm = True
    if 'cmf_buy_threshold' not in st.session_state: st.session_state.cmf_buy_threshold = 0.10
    if 'cmf_sell_threshold' not in st.session_state: st.session_state.cmf_sell_threshold = -0.10
    if 'atr_buy_pullback_factor' not in st.session_state: st.session_state.atr_buy_pullback_factor = 0.5
    if 'atr_exit_multiplier' not in st.session_state: st.session_state.atr_exit_multiplier = 2.0
    if 'atr_low_volatility_pct' not in st.session_state: st.session_state.atr_low_volatility_pct = 0.005
    if 'atr_high_volatility_pct' not in st.session_state: st.session_state.atr_high_volatility_pct = 0.045
    if 'risk_reward_ratio' not in st.session_state: st.session_state.risk_reward_ratio = 2.5
    if 'max_position_pct' not in st.session_state: st.session_state.max_position_pct = 0.25
    if 'risk_per_trade_pct' not in st.session_state: st.session_state.risk_per_trade_pct = 1.0


    # Update session state if new data is passed (e.g., from "Analyze Selected Stock" button)
    if symbol and data is not None and recommendations is not None:
        st.session_state.symbol = symbol
        st.session_state.data = data
        st.session_state.recommendations = recommendations

    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

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

    daily_picks_progress_container = st.empty()
    daily_picks_loading_text_container = st.empty()
    daily_picks_status_text_container = st.empty()
    daily_picks_errors_container = st.empty()

    intraday_picks_progress_container = st.empty()
    intraday_picks_loading_text_container = st.empty()
    intraday_picks_status_text_container = st.empty()
    intraday_picks_errors_container = st.empty()

    if st.button("🚀 Generate Daily Top Picks"):
        daily_progress_bar = daily_picks_progress_container.progress(0)
        daily_loading_text = daily_picks_loading_text_container.empty()
        daily_status_text = daily_picks_status_text_container.empty()
        daily_picks_errors_container.empty()

        daily_status_text.text(f"📊 Analyzing {len(selected_stocks)} stocks for Daily Picks...")

        results_df, batch_errors = analyze_all_stocks(
            selected_stocks,
            batch_size=4,
            progress_bar_obj=daily_progress_bar,
            loading_text_obj=daily_loading_text,
            status_text_obj=daily_status_text
        )

        daily_picks_progress_container.empty()
        daily_picks_loading_text_container.empty()
        daily_picks_status_text_container.empty()

        if batch_errors:
            with daily_picks_errors_container.expander(f"⚠️ {len(batch_errors)} stocks encountered errors during analysis"):
                for error_msg in batch_errors:
                    st.write(error_msg)

        if 'Position Size Shares' in results_df.columns:
            results_df['Position Size Shares'] = results_df['Position Size Shares'].replace('None', np.nan)
            results_df['Position Size Shares'] = pd.to_numeric(results_df['Position Size Shares'], errors='coerce')
            results_df['Position Size Shares'] = results_df['Position Size Shares'].fillna(0.0)

        if 'Position Size Value' in results_df.columns:
            results_df['Position Size Value'] = results_df['Position Size Value'].replace('None', np.nan)
            results_df['Position Size Value'] = pd.to_numeric(results_df['Position Size Value'], errors='coerce')
            results_df['Position Size Value'] = results_df['Position Size Value'].fillna(0.0)

        if 'Trailing Stop' in results_df.columns:
            results_df['Trailing Stop'] = results_df['Trailing Stop'].replace('None', np.nan)
            results_df['Trailing Stop'] = pd.to_numeric(results_df['Trailing Stop'], errors='coerce')
            results_df['Trailing Stop'] = results_df['Trailing Stop'].fillna(0.0)

        insert_top_picks_supabase(results_df, pick_type="daily")

        display_results_df = pd.DataFrame()
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')

        if results_df.empty:
            st.warning("⚠️ No valid stock data retrieved to generate Daily Top Picks.")
        else:
            if recommendation_mode == "Adaptive":
                display_results_df = results_df[results_df["Recommendation"].astype(str).str.contains("Buy", na=False, case=False) & results_df['Error'].isnull()].sort_values(by="Score", ascending=False).head(5)
            else:
                # For standard mode, filter explicitly for "Intraday" buy signals
                buy_condition = results_df["Intraday"].astype(str).str.contains("Buy", na=False, case=False)
                display_results_df = results_df[buy_condition & results_df['Error'].isnull()].sort_values(by="Score", ascending=False).head(5)

            if not display_results_df.empty:
                st.subheader("🏆 Today's Top 5 Stocks")
                for _, row in display_results_df.iterrows():
                    with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']:.2f}/10"):
                        current_price = f"₹{row.get('Current Price', np.nan):.2f}" if pd.notnull(row.get('Current Price')) else 'N/A'
                        buy_at = f"₹{row.get('Buy At', np.nan):.2f}" if pd.notnull(row.get('Buy At')) else 'N/A'
                        stop_loss = f"₹{row.get('Stop Loss', np.nan):.2f}" if pd.notnull(row.get('Stop Loss')) else 'N/A'
                        target = f"₹{row.get('Target', np.nan):.2f}" if pd.notnull(row.get('Target')) else 'N/A'

                        if st.session_state.recommendation_mode == "Adaptive":
                            pos_shares = f"{int(row.get('Position Size Shares', 0))}" if pd.notnull(row.get('Position Size Shares')) else 'N/A'
                            pos_value = f"₹{row.get('Position Size Value', 0):.2f}" if pd.notnull(row.get('Position Size Value')) else 'N/A'
                            trailing_stop = f"₹{row.get('Trailing Stop', np.nan):.2f}" if pd.notnull(row.get('Trailing Stop')) else 'N/A'
                            st.markdown(f"""
                            Current Price: {current_price}
                            Buy At: {buy_at} | Stop Loss: {stop_loss}
                            Target: {target}
                            Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}
                            Regime: {row.get('Regime', 'N/A')}
                            Position Size: {pos_shares} shares (~{pos_value})
                            Trailing Stop: {trailing_stop}
                            Reason: {row.get('Reason', 'N/A')}
                            """)
                        else:
                            # Standard mode display for Top 5
                            st.markdown(f"""
                            Current Price: {current_price}
                            Buy At: {buy_at} | Stop Loss: {stop_loss}
                            Target: {target}
                            Recommendation: {colored_recommendation(row.get('Intraday', 'N/A'))}
                            Reason: {row.get('Reason', 'N/A')}
                            """)
            else:
                st.warning("⚠️ No top picks available (or no 'Buy' recommendations) due to data issues or current market conditions.")

    if st.button("⚡ Generate Intraday Top 5 Picks"):
        intraday_progress_bar = intraday_picks_progress_container.progress(0)
        intraday_loading_text = intraday_picks_loading_text_container.empty()
        intraday_status_text = intraday_picks_status_text_container.empty()
        intraday_picks_errors_container.empty()

        intraday_status_text.text(f"📊 Analyzing {len(selected_stocks)} stocks for Intraday Picks...")

        intraday_results, batch_errors = analyze_intraday_stocks(
            selected_stocks,
            batch_size=4,
            progress_bar_obj=intraday_progress_bar,
            loading_text_obj=intraday_loading_text,
            status_text_obj=intraday_status_text
        )

        intraday_picks_progress_container.empty()
        intraday_picks_loading_text_container.empty()
        intraday_picks_status_text_container.empty()

        if batch_errors:
            with intraday_picks_errors_container.expander(f"⚠️ {len(batch_errors)} stocks encountered errors during analysis"):
                for error_msg in batch_errors:
                    st.write(error_msg)

        insert_top_picks_supabase(intraday_results, pick_type="intraday")

        filtered_df = pd.DataFrame()
        if intraday_results.empty:
            st.warning("⚠️ No valid stock data retrieved to generate Intraday Top 5 Picks.")
        else:
            if st.session_state.recommendation_mode == "Adaptive":
                filtered_df = intraday_results[intraday_results["Recommendation"].str.contains("Buy", na=False, case=False) & intraday_results['Error'].isnull()]
            else:
                if "Intraday" in intraday_results.columns:
                    filtered_df = intraday_results[intraday_results["Intraday"].str.contains("Buy", na=False, case=False) & intraday_results['Error'].isnull()]
                else:
                    logging.error("Intraday column not found in intraday_results during filtering for display.")

            if not filtered_df.empty:
                st.subheader("🏆 Top 5 Intraday Stocks")
                for _, row in filtered_df.sort_values(by="Score", ascending=False).head(5).iterrows():
                    with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']:.2f}/10"):
                        current_price = f"₹{row.get('Current Price', np.nan):.2f}" if pd.notnull(row.get('Current Price')) else 'N/A'
                        buy_at = f"₹{row.get('Buy At', np.nan):.2f}" if pd.notnull(row.get('Buy At')) else 'N/A'
                        stop_loss = f"₹{row.get('Stop Loss', np.nan):.2f}" if pd.notnull(row.get('Stop Loss')) else 'N/A'
                        target = f"₹{row.get('Target', np.nan):.2f}" if pd.notnull(row.get('Target')) else 'N/A'
                        if st.session_state.recommendation_mode == "Adaptive":
                            pos_shares = f"{int(row.get('Position Size Shares', 0))}" if pd.notnull(row.get('Position Size Shares')) else 'N/A'
                            pos_value = f"₹{row.get('Position Size Value', 0):.2f}" if pd.notnull(row.get('Position Size Value')) else 'N/A'
                            trailing_stop = f"₹{row.get('Trailing Stop', np.nan):.2f}" if pd.notnull(row.get('Trailing Stop')) else 'N/A'
                            st.markdown(f"""
                            Current Price: {current_price}
                            Buy At: {buy_at} | Stop Loss: {stop_loss}
                            Target: {target}
                            Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}
                            Regime: {row.get('Regime', 'N/A')}
                            Position Size: {pos_shares} shares (~{pos_value})
                            Trailing Stop: {trailing_stop}
                            Reason: {row.get('Reason', 'N/A')}
                            """)
                        else:
                            # Standard mode display for Top 5 Intraday
                            st.markdown(f"""
                            Current Price: {current_price}
                            Buy At: {buy_at} | Stop Loss: {stop_loss}
                            Target: {target}
                            Recommendation: {colored_recommendation(row.get('Intraday', 'N/A'))}
                            Reason: {row.get('Reason', 'N/A')}
                            """)
            else:
                st.warning("⚠️ No intraday picks available (or no 'Buy' recommendations) due to data issues.")

    if st.button("📜 View Historical Picks"):
        st.session_state.show_history = not st.session_state.show_history

    if st.session_state.show_history:
        st.markdown("### 📜 Historical Picks")
        if st.button("Close Historical Picks"):
            st.session_state.show_history = False
            st.rerun()

        try:
            res = supabase.table("daily_picks").select("date").order("date", desc=True).execute()
            if res.data:
                all_dates = sorted(list(set(row['date'] for row in res.data)), reverse=True)
                if not all_dates:
                    st.warning("No historical picks found in the database.")
                    return

                date_filter = st.selectbox("Select Date", all_dates, key="history_date")

                res2 = supabase.table("daily_picks").select("*").eq("date", date_filter).execute()
                if res2.data:
                    df = pd.DataFrame(res2.data)

                    numeric_cols = [
                        'score', 'current_price', 'buy_at', 'target', 'stop_loss',
                        'position_size_shares', 'position_size_value', 'trailing_stop'
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    df = update_with_latest_prices(df)
                    df = add_action_and_change(df)

                    for col in ['buy_at', 'current_price', 'target', 'stop_loss', 'position_size_value', 'trailing_stop']:
                        if col in df.columns:
                            df[col] = df[col].map(lambda x: f"₹{x:.2f}" if pd.notnull(x) else 'N/A')
                    if 'position_size_shares' in df.columns:
                         df['position_size_shares'] = df['position_size_shares'].map(lambda x: f"{int(x)}" if pd.notnull(x) else 'N/A')
                    if '% Change' in df.columns:
                        df['% Change'] = df['% Change'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else 'N/A')

                    is_adaptive_data_present = 'recommendation' in df.columns and \
                                               df['recommendation'].notna().any() and \
                                               df['recommendation'].astype(str).str.contains("Buy|Sell|Hold|N/A", case=False, na=False).any()

                    if is_adaptive_data_present:
                        display_cols = [
                            "symbol", "buy_at", "current_price", "% Change", "What to do now?",
                            "recommendation", "regime", "position_size_shares", "position_size_value",
                            "trailing_stop", "reason", "target", "stop_loss", "pick_type", "score"
                        ]
                        final_display_df = df[[col for col in display_cols if col in df.columns]]
                        if 'recommendation' in final_display_df.columns:
                            final_display_df['recommendation'] = final_display_df['recommendation'].apply(colored_recommendation)
                    else:
                        standard_cols = ["symbol", "buy_at", "current_price", "% Change", "What to do now?",
                                         "intraday", "swing", "short_term", "long_term", "mean_reversion",
                                         "breakout", "ichimoku_trend", "target", "stop_loss", "pick_type", "score", "reason"]
                        final_display_df = df[[col for col in standard_cols if col in df.columns]]
                        for col_name in ["intraday", "swing", "short_term", "long_term", "mean_reversion", "breakout", "ichimoku_trend"]:
                            if col_name in final_display_df.columns:
                                final_display_df[col_name] = final_display_df[col_name].apply(colored_recommendation)

                    styled_df = style_picks_df(final_display_df)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.warning("No picks found for the selected date in Supabase.")
            else:
                st.warning("No historical dates found in Supabase.")
        except Exception as e:
            st.error(f"Error fetching historical picks: {e}")
            logging.error(f"Error fetching historical picks: {e}")

    if st.session_state.symbol and st.session_state.data is not None and st.session_state.recommendations is not None:
        symbol = st.session_state.symbol
        data = st.session_state.data
        recommendations = st.session_state.recommendations

        st.header(f"📋 {normalize_symbol_dhan(symbol)} Analysis")

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
            pos_value = f"₹{recommendations.get('Position Size', {}).get('value', 'N/A'):.2f}" if pd.notnull(recommendations.get('Position Size', {}).get('value')) else 'N/A'

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Recommendation**: {colored_recommendation(recommendations.get('Recommendation', 'N/A'))}")
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10")
            with col2:
                st.write(f"**Position Size**: {pos_shares} shares")
                st.write(f"**Value**: {pos_value}")
            with col3:
                st.write(f"**Trailing Stop**: ₹{recommendations.get('Trailing Stop', 'N/A')}")
                st.write(f"**Volatility**: {assess_risk(data)}")
            st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")

        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Overall Recommendation**: {colored_recommendation(recommendations.get('Intraday', 'N/A'))}") # Intraday is now the consolidated for standard
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10")
                st.write(f"**Volatility**: {assess_risk(data)}")
            with col2:
                st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")
            # Individual strategy recommendations in an expander for standard mode
            with st.expander("Detailed Standard Recommendations"):
                st.write(f"**Intraday**: {colored_recommendation(recommendations.get('Intraday', 'N/A'))}")
                st.write(f"**Swing**: {colored_recommendation(recommendations.get('Swing', 'N/A'))}")
                st.write(f"**Short-Term**: {colored_recommendation(recommendations.get('Short-Term', 'N/A'))}")
                st.write(f"**Long-Term**: {colored_recommendation(recommendations.get('Long-Term', 'N/A'))}")
                st.write(f"**Mean Reversion**: {colored_recommendation(recommendations.get('Mean_Reversion', 'N/A'))}")
                st.write(f"**Breakout**: {colored_recommendation(recommendations.get('Breakout', 'N/A'))}")
                st.write(f"**Ichimoku Trend**: {colored_recommendation(recommendations.get('Ichimoku_Trend', 'N/A'))}")

        backtest_spinner_placeholder = st.empty()

        with st.form(key="backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                swing_button = st.form_submit_button("🔍 Backtest Swing Strategy")
            with col2:
                intraday_button = st.form_submit_button("🔍 Backtest Intraday Strategy")

        if swing_button or intraday_button:
            strategy = "Swing" if swing_button else "Intraday"
            with backtest_spinner_placeholder.container():
                with st.spinner(f"Running {strategy} Strategy backtest... (This may take a while for large data sets)"):
                    if st.session_state.data is not None:
                        data_for_hash = st.session_state.data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(300).to_json()
                        data_hash = hash(data_for_hash)
                        # Pass all configured strategy parameters to backtest_stock
                        backtest_results = backtest_stock(st.session_state.data, st.session_state.symbol, strategy=strategy, _data_hash=data_hash,
                                                          adx_strong_trend_threshold=st.session_state.adx_strong_trend_threshold,
                                                          adx_no_trend_threshold=st.session_state.adx_no_trend_threshold,
                                                          rsi_overbought=st.session_state.rsi_overbought,
                                                          rsi_oversold=st.session_state.rsi_oversold,
                                                          macd_zero_line_confirm=st.session_state.macd_zero_line_confirm,
                                                          cmf_buy_threshold=st.session_state.cmf_buy_threshold,
                                                          cmf_sell_threshold=st.session_state.cmf_sell_threshold,
                                                          atr_buy_pullback_factor=st.session_state.atr_buy_pullback_factor,
                                                          atr_exit_multiplier=st.session_state.atr_exit_multiplier,
                                                          atr_low_volatility_pct=st.session_state.atr_low_volatility_pct,
                                                          atr_high_volatility_pct=st.session_state.atr_high_volatility_pct,
                                                          risk_reward_ratio=st.session_state.risk_reward_ratio,
                                                          max_position_pct=st.session_state.max_position_pct,
                                                          risk_per_trade_pct=st.session_state.risk_per_trade_pct)

                        if strategy == "Swing":
                            st.session_state.backtest_results_swing = backtest_results
                        else:
                            st.session_state.backtest_results_intraday = backtest_results
                    else:
                        st.warning("Cannot backtest: No stock data loaded in session state.")
                        st.session_state.backtest_results_swing = None
                        st.session_state.backtest_results_intraday = None

            backtest_spinner_placeholder.empty()
            st.rerun()

        for strategy_name, results_key in [("Swing", "backtest_results_swing"), ("Intraday", "backtest_results_intraday")]:
            backtest_results = st.session_state.get(results_key)
            if backtest_results and backtest_results['trades'] > 0:
                st.subheader(f"📈 Backtest Results ({strategy_name} Strategy)")
                st.write(f"**Total Return**: {backtest_results['total_return']:.2f}%")
                st.write(f"**Annualized Return**: {backtest_results['annual_return']:.2f}%")
                st.write(f"**Sharpe Ratio**: {backtest_results['sharpe_ratio']:.2f}")
                st.write(f"**Max Drawdown**: {backtest_results['max_drawdown']:.2f}%")
                st.write(f"**Number of Trades**: {backtest_results['trades']}")
                st.write(f"**Win Rate**: {backtest_results['win_rate']:.2f}%")
                st.write(f"**Total Profit Amount**: ₹{backtest_results['total_profit_amount']:.2f}")
                st.write(f"**Total Loss Amount**: ₹{backtest_results['total_loss_amount']:.2f}")

                with st.expander("Trade Details"):
                    for trade in backtest_results["trade_details"]:
                        profit = trade.get("profit", 0)
                        st.write(f"Entry: {trade['entry_date']} @ ₹{trade['entry_price']:.2f}, "
                                 f"Exit: {trade['exit_date']} @ ₹{trade['exit_price']:.2f}, "
                                 f"Profit: ₹{profit:.2f} ({trade['reason']})")

                # --- NEW: INTERACTIVE CHARTS WITH SUBPLOTS ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.08,
                                    row_heights=[0.5, 0.25, 0.25]) # Adjust row heights for price, RSI, MACD

                # Candlestick chart on top subplot
                fig.add_trace(go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'],
                                             name='Candlesticks'), row=1, col=1)

                # Moving Averages
                if 'SMA_50' in data.columns and data['SMA_50'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
                if 'SMA_200' in data.columns and data['SMA_200'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', width=1)), row=1, col=1)
                if 'EMA_20' in data.columns and data['EMA_20'].notnull().any():
                     fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='blue', width=1)), row=1, col=1)
                if 'EMA_50' in data.columns and data['EMA_50'].notnull().any():
                     fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='green', width=1)), row=1, col=1)

                # Bollinger Bands
                if 'Upper_Band' in data.columns and data['Upper_Band'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Bollinger Upper', line=dict(color='gray', dash='dash', width=0.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Bollinger Lower', line=dict(color='gray', dash='dash', width=0.5), fill='tonexty', fillcolor='rgba(192,192,192,0.1)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Middle_Band'], mode='lines', name='Bollinger Middle', line=dict(color='blue', width=0.5)), row=1, col=1)

                # Ichimoku Cloud (plot SPAN A and B, fill between them)
                if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and \
                   data['Ichimoku_Span_A'].notnull().any() and data['Ichimoku_Span_B'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Span_A'], line=dict(color='#2D9DFF', width=0.5), name='Senkou Span A', showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Span_B'], line=dict(color='#F77292', width=0.5), name='Senkou Span B', fill='tonexty', fillcolor='rgba(0,100,0,0.2)' if (data['Ichimoku_Span_A'].iloc[-1] > data['Ichimoku_Span_B'].iloc[-1]) else 'rgba(255,0,0,0.2)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Tenkan'], line=dict(color='green', width=1), name='Tenkan Sen'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Kijun'], line=dict(color='red', width=1), name='Kijun Sen'), row=1, col=1)
                    # Chikou Span is usually shifted back
                    # fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Chikou'], line=dict(color='purple', width=1), name='Chikou Span'), row=1, col=1)

                # Buy/Sell Signals
                epoch_plot_threshold = pd.Timestamp('1970-01-02')

                if backtest_results["buy_signals"]:
                    buy_dates_str, buy_prices = zip(*backtest_results["buy_signals"])
                    buy_dates_dt = pd.to_datetime(list(buy_dates_str), errors='coerce')
                    valid_buy_signals_for_plot = [(d, p) for d, p in zip(buy_dates_dt, buy_prices)
                                                  if pd.notna(d) and d >= epoch_plot_threshold]
                    if valid_buy_signals_for_plot:
                        buy_dates_filtered, buy_prices_filtered = zip(*valid_buy_signals_for_plot)
                        fig.add_trace(go.Scatter(x=list(buy_dates_filtered), y=list(buy_prices_filtered), mode='markers', name='Buy Signals',
                                               marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)

                if backtest_results["sell_signals"]:
                    sell_dates_str, sell_prices = zip(*backtest_results["sell_signals"])
                    sell_dates_dt = pd.to_datetime(list(sell_dates_str), errors='coerce')
                    valid_sell_signals_for_plot = [(d, p) for d, p in zip(sell_dates_dt, sell_prices)
                                                   if pd.notna(d) and d >= epoch_plot_threshold]
                    if valid_sell_signals_for_plot:
                        sell_dates_filtered, sell_prices_filtered = zip(*valid_sell_signals_for_plot)
                        fig.add_trace(go.Scatter(x=list(sell_dates_filtered), y=list(sell_prices_filtered), mode='markers', name='Sell Signals',
                                               marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)


                # RSI on second subplot
                if 'RSI' in data.columns and data['RSI'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
                    fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0,100])

                # MACD on third subplot
                if 'MACD' in data.columns and 'MACD_signal' in data.columns and 'MACD_hist' in data.columns and \
                   data['MACD'].notnull().any():
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='orange')), row=3, col=1)
                    colors = ['green' if val >= 0 else 'red' for val in data['MACD_hist']]
                    fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='MACD Histogram', marker_color=colors), row=3, col=1)
                    fig.update_yaxes(title_text='MACD', row=3, col=1)


                fig.update_layout(title_text=f"{normalize_symbol_dhan(symbol)} Price with Indicators",
                                  xaxis_rangeslider_visible=False,
                                  height=800,
                                  hovermode="x unified")

                fig.update_xaxes(showticklabels=False, row=1, col=1)
                fig.update_xaxes(showticklabels=False, row=2, col=1)
                fig.update_xaxes(showticklabels=True, row=3, col=1, title_text='Date')

                st.plotly_chart(fig, use_container_width=True)


            elif backtest_results:
                st.info(f"No valid trades generated for {strategy_name} strategy on {normalize_symbol_dhan(symbol)} with available data.")


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


        st.subheader(f"📊 Monte Carlo Price Projections for {normalize_symbol_dhan(symbol)}")
        simulations = monte_carlo_simulation(data)
        if simulations:
            sim_df = pd.DataFrame(simulations).T
            sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
            fig_sim = px.line(sim_df, title="Monte Carlo Price Projections (30 Days)")
            fig_sim.update_layout(showlegend=False)
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Monte Carlo simulation could not be performed due to insufficient data or errors.")


def main():

    if st.sidebar.button("Test Dhan Connection"):
        with st.spinner("Testing Dhan API connection..."):
            if test_dhan_connection():
                st.success("✅ Dhan API is working!")
            else:
                st.error("❌ Dhan API connection failed. Check your credentials and network. See logs for details.")

    st.sidebar.title("🔍 Stock Selection")
    stock_list = fetch_nse_stock_list()

    # Initialize all session state variables at the beginning of main()
    if 'symbol' not in st.session_state: st.session_state.symbol = stock_list[0]
    if 'recommendation_mode' not in st.session_state: st.session_state.recommendation_mode = "Standard"
    if 'initial_capital' not in st.session_state: st.session_state.initial_capital = 50000
    # Initialize strategy parameters (they are still configurable via sidebar)
    if 'adx_strong_trend_threshold' not in st.session_state: st.session_state.adx_strong_trend_threshold = 25
    if 'adx_no_trend_threshold' not in st.session_state: st.session_state.adx_no_trend_threshold = 20
    if 'rsi_overbought' not in st.session_state: st.session_state.rsi_overbought = 70
    if 'rsi_oversold' not in st.session_state: st.session_state.rsi_oversold = 30
    if 'macd_zero_line_confirm' not in st.session_state: st.session_state.macd_zero_line_confirm = True
    if 'cmf_buy_threshold' not in st.session_state: st.session_state.cmf_buy_threshold = 0.10
    if 'cmf_sell_threshold' not in st.session_state: st.session_state.cmf_sell_threshold = -0.10
    if 'atr_buy_pullback_factor' not in st.session_state: st.session_state.atr_buy_pullback_factor = 0.5
    if 'atr_exit_multiplier' not in st.session_state: st.session_state.atr_exit_multiplier = 2.0
    if 'atr_low_volatility_pct' not in st.session_state: st.session_state.atr_low_volatility_pct = 0.005
    if 'atr_high_volatility_pct' not in st.session_state: st.session_state.atr_high_volatility_pct = 0.045
    if 'risk_reward_ratio' not in st.session_state: st.session_state.risk_reward_ratio = 2.5
    if 'max_position_pct' not in st.session_state: st.session_state.max_position_pct = 0.25
    if 'risk_per_trade_pct' not in st.session_state: st.session_state.risk_per_trade_pct = 1.0


    # Removed "⚙️ Indicator Parameters" section

    # Strategy parameters (sidebar inputs, relevant for Adaptive Mode)
    with st.sidebar.expander("🧠 Strategy Parameters (Adaptive Mode)"):
        # Corrected pattern: removed `st.session_state.variable =` on the left-hand side
        st.number_input("ADX Strong Trend Threshold", min_value=15, max_value=50, value=st.session_state.adx_strong_trend_threshold, key="adx_strong_trend_threshold", help="ADX value above which trend is considered strong.")
        st.number_input("ADX No Trend Threshold", min_value=5, max_value=25, value=st.session_state.adx_no_trend_threshold, key="adx_no_trend_threshold", help="ADX value below which market is considered sideways.")
        st.number_input("RSI Overbought Level", min_value=60, max_value=90, value=st.session_state.rsi_overbought, key="rsi_overbought", help="RSI value indicating overbought conditions.")
        st.number_input("RSI Oversold Level", min_value=10, max_value=40, value=st.session_state.rsi_oversold, key="rsi_oversold", help="RSI value indicating oversold conditions.")
        st.checkbox("MACD Zero Line Confirmation", value=st.session_state.macd_zero_line_confirm, key="macd_zero_line_confirm", help="Require MACD crossovers to be above/below zero line for stronger signals.")
        st.number_input("CMF Buy Threshold", min_value=0.0, max_value=0.5, value=st.session_state.cmf_buy_threshold, step=0.01, key="cmf_buy_threshold", help="CMF value indicating significant buying pressure.")
        st.number_input("CMF Sell Threshold", min_value=-0.5, max_value=0.0, value=st.session_state.cmf_sell_threshold, step=0.01, key="cmf_sell_threshold", help="CMF value indicating significant selling pressure.")
        st.number_input("ATR Buy Pullback Factor", min_value=0.1, max_value=1.0, value=st.session_state.atr_buy_pullback_factor, step=0.1, key="atr_buy_pullback_factor", help="How many ATRs below current price to target for buy pullback (e.g., 0.5 means 0.5 * ATR below current price).")
        st.number_input("ATR Exit Multiplier", min_value=1.0, max_value=5.0, value=st.session_state.atr_exit_multiplier, step=0.1, key="atr_exit_multiplier", help="Multiplier for ATR to calculate initial Stop Loss / Trailing Stop distance.")
        st.number_input("ATR Low Volatility (%)", min_value=0.001, max_value=0.01, value=st.session_state.atr_low_volatility_pct, step=0.001, format="%.3f", key="atr_low_volatility_pct", help="Below this ATR % of Close, market is too low volatility.")
        st.number_input("ATR High Volatility (%)", min_value=0.01, max_value=0.1, value=st.session_state.atr_high_volatility_pct, step=0.005, format="%.3f", key="atr_high_volatility_pct", help="Above this ATR % of Close, market is too high volatility.")
        st.number_input("Risk-Reward Ratio", min_value=1.0, max_value=5.0, value=st.session_state.risk_reward_ratio, step=0.1, key="risk_reward_ratio", help="Target Profit = Risk * Ratio. (e.g., 2.5 means 2.5x risk for profit).")
        st.number_input("Max Position Size (% of Equity)", min_value=0.05, max_value=1.0, value=st.session_state.max_position_pct, step=0.05, format="%.2f", key="max_position_pct", help="Maximum percentage of total equity to allocate to a single trade (e.g., 0.25 for 25%).")
        st.number_input("Risk per Trade (% of Equity)", min_value=0.5, max_value=5.0, value=st.session_state.risk_per_trade_pct, step=0.1, key="risk_per_trade_pct", help="Percentage of total equity you're willing to lose on a single trade. Used for position sizing.")


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

    st.session_state.initial_capital = st.sidebar.number_input(
        "Initial Capital (for Position Sizing)",
        min_value=10000,
        max_value=10000000,
        value=st.session_state.initial_capital,
        step=5000,
        help="This capital amount is used to calculate position sizes for adaptive recommendations and backtests."
    )

    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner(f"Loading and analyzing data for {normalize_symbol_dhan(symbol)}..."):
                data = fetch_stock_data_cached(symbol)
                # required_min_length is now based on fixed INDICATOR_MIN_LENGTHS values
                required_min_length = max(list(INDICATOR_MIN_LENGTHS.values())) + 1

                if not data.empty and len(data) >= required_min_length:
                    # Call analyze_stock without passing indicator parameters
                    data = analyze_stock(data)

                    if data.empty or pd.isna(data['Close'].iloc[-1]) or ('ATR' in data.columns and pd.isna(data['ATR'].iloc[-1])):
                        st.warning(f"⚠️ Could not complete analysis for {normalize_symbol_dhan(symbol)} due to insufficient or invalid data after indicator computation.")
                        st.session_state.symbol = None
                        return

                    if recommendation_mode == "Adaptive":
                        # Pass all relevant strategy parameters to adaptive_recommendation
                        recommendations = adaptive_recommendation(data, symbol, equity=st.session_state.initial_capital,
                                                                  adx_strong_trend_threshold=st.session_state.adx_strong_trend_threshold,
                                                                  adx_no_trend_threshold=st.session_state.adx_no_trend_threshold,
                                                                  rsi_overbought=st.session_state.rsi_overbought,
                                                                  rsi_oversold=st.session_state.rsi_oversold,
                                                                  macd_zero_line_confirm=st.session_state.macd_zero_line_confirm,
                                                                  cmf_buy_threshold=st.session_state.cmf_buy_threshold,
                                                                  cmf_sell_threshold=st.session_state.cmf_sell_threshold,
                                                                  atr_buy_pullback_factor=st.session_state.atr_buy_pullback_factor,
                                                                  atr_exit_multiplier=st.session_state.atr_exit_multiplier,
                                                                  atr_low_volatility_pct=st.session_state.atr_low_volatility_pct,
                                                                  atr_high_volatility_pct=st.session_state.atr_high_volatility_pct,
                                                                  risk_reward_ratio=st.session_state.risk_reward_ratio,
                                                                  max_position_pct=st.session_state.max_position_pct,
                                                                  risk_per_trade_pct=st.session_state.risk_per_trade_pct)
                    else:
                        recommendations = generate_recommendations(data, symbol) # Standard mode uses its own fixed params

                    st.session_state.symbol = symbol
                    st.session_state.data = data
                    st.session_state.recommendations = recommendations
                    st.session_state.backtest_results_swing = None
                    st.session_state.backtest_results_intraday = None
                    display_dashboard(symbol, data, recommendations)
                else:
                    st.warning(f"⚠️ No sufficient historical data available for {normalize_symbol_dhan(symbol)} to perform a full analysis ({len(data)} rows found, need at least {required_min_length}).")
                    st.session_state.symbol = None
    else:
        display_dashboard()

    if st.sidebar.button("Clear All Caches", help="Clears cached data and restarts the app."):
        st.session_state.clear()
        st.cache_data.clear()
        st.session_state.data = None
        st.session_state.recommendations = None
        st.session_state.backtest_results_swing = None
        st.rerun()

if __name__ == "__main__":
    main()
