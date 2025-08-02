import pandas as pd
import ta
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import random
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
    df = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_SEGMENT'] == 'E')]
    return df

def get_dhan_security_id(symbol):
    """Get Dhan security ID for a given symbol"""
    df = load_dhan_instrument_master()
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
    
    payload = {
        "securityId": "1333", # Example security ID for testing
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
        v = float(val.replace('₹', '').replace('%', '')) # Remove currency and percentage symbols for parsing
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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Brave/124.0.0.0 Safari/537.36"
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
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
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
        
@RateLimiter(calls=1, period=1.5) # 1 call per 1.5 seconds per thread
@lru_cache(maxsize=1000) # Caches results in memory for speed
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
        logging.info(f"Requesting data for {symbol} with payload: {payload}")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        data = response.json()

        # logging.debug(f"API response for {symbol}: {data}") # Too verbose for production logs

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
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns for {symbol}: {missing_columns}. Columns found: {df.columns.tolist()}")
            return pd.DataFrame()

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_columns)

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
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
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

INDICATOR_MIN_LENGTHS = {
    'RSI': 14,
    'MACD': 26, # Requires 26 for slow EMA
    'SMA_50': 50,
    'SMA_200': 200,
    'EMA_20': 20,
    'EMA_50': 50,
    'Bollinger': 20,
    'Stochastic': 14,
    'ATR': 14,
    'ADX': 14,
    'OBV': 1, # Requires only 1 data point conceptually, but needs price movement history
    'Ichimoku': 52, # Longest window for Kijun Sen or Senkou Span B
    'CMF': 20,
    'TRIX': 15, # EMA of EMA of EMA(15) -> 3*15 + ~ some offset for start
    'Ultimate_Osc': 28, # Longest period for oscillator (7, 14, 28)
    'CMO': 14,
    'VPT': 1, # Requires 1 data point conceptually
    'Volume_Spike': 20 # Requires Avg_Volume (20-period moving average)
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

def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    """
    # Define columns for *actually computed* indicators
    columns_to_compute = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
        'Avg_Volume', 'Volume_Spike', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF',
        'TRIX', 'Ultimate_Osc', 'CMO', 'VPT', 'ATR_pct', 'DMP', 'DMN', 'ADX_Slope'
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
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Data validation failed in analyze_stock. Data length: {len(data)}. Returning initial data with NaNs.")
        return data

    try:
        # Calculate Average Volume first for Volume_Spike
        if can_compute_indicator(data, 'Volume_Spike'): # Avg_Volume is part of Volume_Spike calc
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
            data['Ichimoku_Chikou'] = data['Close'].shift(-26) # Needs future data, usually plotted shifted back

        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20).chaikin_money_flow()
        
        if can_compute_indicator(data, 'TRIX'):
            data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()

        if can_compute_indicator(data, 'Ultimate_Osc'):
            data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(high=data['High'], low=data['Low'], close=data['Close']).ultimate_oscillator()
        
        # --- FIX for ChandeMomentumOscillator Error ---
        # The error "module 'ta.momentum' has no attribute 'ChandeMomentumOscillator'" suggests an older/incompatible `ta` version.
        # If your 'ta' library is updated to >= 0.11.0, this should work. If not, you might need to manually install or skip this indicator.
        # For now, I'll keep the call, assuming a compatible version is used.
        # If the error persists, you might need to:
        # 1. Upgrade `python-ta`: `pip install --upgrade ta`
        # 2. If it still fails, manually implement CMO or remove it.
        if can_compute_indicator(data, 'CMO'):
            try:
                data['CMO'] = ta.momentum.ChandeMomentumOscillator(close=data['Close'], window=14).chande_momentum_oscillator()
            except AttributeError:
                logging.error("CMO calculation failed: ChandeMomentumOscillator not found in ta.momentum. Skipping CMO.")
                data['CMO'] = np.nan # Ensure column exists but is NaN
            except Exception as e:
                logging.error(f"Error computing CMO: {e}. Skipping CMO.")
                data['CMO'] = np.nan

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
        
        if rsi < 35 and atr > 0: # Potential for bounce from oversold + volatility
            buy_at = close * (1 - 0.5 * (atr / close))
        elif atr > 0: # Standard entry slightly below current price based on volatility
            buy_at = close * (1 - 0.2 * (atr / close))
        else: # Fallback if ATR is not available
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
            adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
            if adx > 25: # Strong trend, tighter stop
                atr_multiplier = 2.0
            elif adx < 20: # Sideways, wider stop
                atr_multiplier = 3.0
            
            stop_loss = close - atr_multiplier * atr
            if stop_loss < close * 0.90: # Minimum 10% below current price
                stop_loss = close * 0.90
            elif stop_loss > close * 0.97: # Maximum 3% below current price
                stop_loss = close * 0.97
        else:
            stop_loss = close * 0.95 # Default 5% stop if ATR not available
            
        return round(float(stop_loss), 2)
    except Exception as e:
        logging.error(f"Error calculating stop_loss: {str(e)}")
        return None

def calculate_target(data, risk_reward_ratio=2.0):
    """
    Calculates the target price based on ATR and a risk-reward ratio.
    """
    if not validate_data(data, min_length=10) or pd.isna(data['Close'].iloc[-1]):
        return None
    try:
        close = data['Close'].iloc[-1]
        stop_loss = calculate_stop_loss(data)
        
        if stop_loss is None or stop_loss >= close:
            return None # Cannot calculate target if stop_loss is invalid or above close
        
        risk_per_share = close - stop_loss
        
        if risk_per_share <= 0: # Avoid division by zero or negative risk
            return None
        
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns and pd.notnull(data['ADX'].iloc[-1]) else 0
        if adx > 25:
            adjusted_rr = 3.0 # Higher risk-reward in strong trends
        elif adx < 20:
            adjusted_rr = 1.5 # Lower risk-reward in sideways markets
        else:
            adjusted_rr = risk_reward_ratio
            
        target = close + (risk_per_share * adjusted_rr)
        
        if target > close * 1.25: # Cap target at 25% profit
            target = close * 1.25
            
        return round(float(target), 2)
    except Exception as e:
        logging.error(f"Error calculating target: {str(e)}")
        return None

def calculate_trailing_stop(current_price, atr, atr_multiplier=2.5):
    """
    Calculates a simple ATR-based trailing stop.
    """
    if pd.isna(current_price) or pd.isna(atr) or atr <= 0:
        return None
    
    trailing_stop = current_price - atr_multiplier * atr
    
    # Ensure trailing stop is not too close to current price
    trailing_stop = max(current_price * 0.99, trailing_stop) # At least 1% below current price
    
    return round(max(0, trailing_stop), 2) # Ensure it's not negative


def classify_market_regime(data):
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

    if atr_pct > 0.04:
        return 'Highly Volatile'

    if adx > 25: # Strong trend
        if close > sma_50 and close > sma_200:
            return 'Bullish Trending'
        elif close < sma_50 and close < sma_200:
            return 'Bearish Trending'
        else: # Price is trending but mixed signals from MAs
            return 'Trending (Unclear Direction)'
    elif adx < 20: # Weak/No trend
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
    
    max_possible_raw_score = sum([w for k,w in weights.items() if w > 0]) # Sum of all positive weights

    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Invalid data for scoring {symbol}, returning 0.")
        return 0, ["Insufficient data to compute score."]

    close_price = data['Close'].iloc[-1]
    current_volume = data['Volume'].iloc[-1]

    if 'Avg_Volume' in data.columns and pd.notnull(data['Avg_Volume'].iloc[-1]):
        if data['Avg_Volume'].iloc[-1] > 0 and current_volume < data['Avg_Volume'].iloc[-1] * 0.3:
            score -= 2 # Penalty for very low volume
            reason_components.append("Very low volume, caution advised.")

    # RSI
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        if rsi < 30:
            # Stronger positive score for deeper oversold
            score += weights['RSI_Oversold'] * (1 - (rsi / 30))
            reason_components.append(f"RSI({int(rsi)}) is oversold (potential bounce).")
        elif rsi > 70:
            # Stronger negative score for deeper overbought
            score += weights['RSI_Overbought'] * ((rsi - 70) / 30)
            reason_components.append(f"RSI({int(rsi)}) is overbought (potential pullback).")

    # MACD
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        macd_hist = data['MACD_hist'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0: # Bullish crossover and positive histogram (momentum)
            score += weights['MACD_Bullish']
            reason_components.append("MACD is bullish (crossover and positive momentum).")
        elif macd < macd_signal and macd_hist < 0: # Bearish crossover and negative histogram
            score += weights['MACD_Bearish']
            reason_components.append("MACD is bearish (crossover and negative momentum).")
        
    # Ichimoku Cloud
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Ichimoku_Span_A', 'Ichimoku_Span_B', 'Close']):
        close = data['Close'].iloc[-1]
        span_a = data['Ichimoku_Span_A'].iloc[-1]
        span_b = data['Ichimoku_Span_B'].iloc[-1]
        
        if close > max(span_a, span_b): # Price above the cloud (bullish)
            score += weights['Ichimoku_Bullish']
            reason_components.append("Price is above Ichimoku Cloud (strong bullish trend).")
        elif close < min(span_a, span_b): # Price below the cloud (bearish)
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

        if adx > 25 and dmp > dmn: # Strong trend and DI+ > DI- (bullish)
            score += weights['ADX_StrongBull']
            reason_components.append(f"ADX({int(adx)}) shows strong bullish trend (DI+ > DI-).")
        elif adx > 25 and dmn > dmp: # Strong trend and DI- > DI+ (bearish)
            score += weights['ADX_StrongBear']
            reason_components.append(f"ADX({int(adx)}) shows strong bearish trend (DI- > DI+).")
        if 'ADX_Slope' in data.columns and pd.notnull(data['ADX_Slope'].iloc[-1]):
            adx_slope = data['ADX_Slope'].iloc[-1]
            if adx_slope > 0 and adx < 40: # ADX still rising but not extremely high
                reason_components.append(f"ADX slope positive (trend strengthening).")
            elif adx_slope < 0 and adx > 20: # ADX falling but still indicating a trend
                reason_components.append(f"ADX slope negative (trend weakening).")

    # Bollinger Bands Breakout / Squeeze
    if all(col in data.columns and pd.notnull(data[col].iloc[-1]) for col in ['Upper_Band', 'Lower_Band', 'Middle_Band', 'Close']):
        close = data['Close'].iloc[-1]
        upper_band = data['Upper_Band'].iloc[-1]
        lower_band = data['Lower_Band'].iloc[-1]
        middle_band = data['Middle_Band'].iloc[-1]
        
        if close > upper_band: # Price breaking above upper band (bullish breakout)
            score += weights['Bollinger_BreakoutUp']
            reason_components.append("Price breaking out above Bollinger Upper Band (bullish breakout).")
        elif close < lower_band: # Price breaking below lower band (bearish breakout)
            score += weights['Bollinger_BreakoutDown']
            reason_components.append("Price breaking out below Bollinger Lower Band (bearish breakout).")
        
        if middle_band != 0:
            bb_width = (upper_band - lower_band) / middle_band
            if len(data['Middle_Band']) >= 20: # Ensure enough data for rolling mean
                avg_bb_width = ((data['Upper_Band'] - data['Lower_Band']) / data['Middle_Band']).rolling(window=20).mean().iloc[-1]
                if pd.notnull(avg_bb_width) and avg_bb_width > 0 and bb_width < avg_bb_width * 0.7: # Width significantly smaller than average
                    score += 0.25 # Small positive for squeeze (potential breakout soon)
                    reason_components.append("Bollinger Bands are squeezing (low volatility, potential breakout soon).")

    # Volume Spike
    if 'Volume_Spike' in data.columns and pd.notnull(data['Volume_Spike'].iloc[-1]) and data['Volume_Spike'].iloc[-1]:
        if len(data) >= 2 and close_price > data['Close'].iloc[-2]: # Volume spike with rising price
            score += weights['Volume_SpikeUp']
            reason_components.append("Significant volume spike confirming upward price move.")
        elif len(data) >= 2 and close_price < data['Close'].iloc[-2]: # Volume spike with falling price
            score += weights['Volume_SpikeDown']
            reason_components.append("Significant volume spike confirming downward price move.")

    scaled_score = (score / max_possible_raw_score) * 10 if max_possible_raw_score > 0 else 0
    final_score = min(max(scaled_score, -10), 10) # Clamp score between -10 and 10
    
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

    # Ensure sufficient data for full analysis and regime classification
    required_length_for_adaptive = max(INDICATOR_MIN_LENGTHS['SMA_200'], INDICATOR_MIN_LENGTHS['Ichimoku'])
    if not validate_data(data, min_length=required_length_for_adaptive):
        recommendations["Reason"] = "Insufficient historical data for comprehensive adaptive analysis."
        logging.warning(f"Insufficient data for {symbol} in adaptive_recommendation.")
        return recommendations

    try:
        current_price = round(float(data['Close'].iloc[-1]), 2)
        recommendations["Current Price"] = current_price
        
        if pd.isna(current_price):
            recommendations["Reason"] = "Current price not available for adaptive analysis."
            return recommendations

        market_regime = classify_market_regime(data)
        recommendations["Regime"] = market_regime
        logging.info(f"Market Regime for {symbol}: {market_regime}")

        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2)
        recommendations["Reason"] = reason_text
        
        final_recommendation = "Hold" 

        # Adaptive logic based on regime and score
        if "Bullish Trending" in market_regime:
            if score >= 6: final_recommendation = "Strong Buy (Trend Continuation)"
            elif score >= 3: final_recommendation = "Buy (Trend Following)"
        elif "Bearish Trending" in market_regime:
            if score <= -6: final_recommendation = "Strong Sell (Trend Continuation)"
            elif score <= -3: final_recommendation = "Sell (Trend Following)"
        elif "Highly Volatile" in market_regime:
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

        buy_at = calculate_buy_at(data)
        stop_loss = calculate_stop_loss(data)
        target = calculate_target(data)

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
                
                # Cap position size to 25% of equity to manage concentration risk
                if current_price > 0 and position_value > equity * 0.25:
                    position_value = equity * 0.25
                    position_shares = int(position_value / current_price) if current_price > 0 else 0
                
                recommendations["Position Size"] = {"shares": position_shares, "value": round(position_value, 2)}
            else:
                recommendations["Position Size"] = {"shares": 0, "value": 0}
        else:
             recommendations["Position Size"] = {"shares": 0, "value": 0}

        if recommendations["Recommendation"].lower().startswith("buy"):
             recommendations["Trailing Stop"] = calculate_trailing_stop(current_price, data['ATR'].iloc[-1] if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]) else None)
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

    # Ensure sufficient data for Ichimoku
    if not validate_data(data, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
        logging.warning(f"Invalid data for standard recommendations: {symbol}")
        return recommendations

    try:
        recommendations["Current Price"] = round(float(data['Close'].iloc[-1]), 2)
        
        score, reason_text = compute_signal_score(data, symbol)
        recommendations["Score"] = round(score, 2)

        buy_score = 0
        sell_score = 0

        if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi <= 20: # Extremely oversold
                buy_score += 4
                recommendations["Mean_Reversion"] = "Strong Buy"
            elif rsi < 30: # Oversold
                buy_score += 2
                recommendations["Mean_Reversion"] = "Buy"
            elif rsi > 70: # Overbought
                sell_score += 2
                recommendations["Mean_Reversion"] = "Sell"
            elif rsi > 80: # Extremely overbought
                sell_score += 4
                recommendations["Mean_Reversion"] = "Strong Sell"

        if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
            macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
            if macd_diff > 0 and data['MACD'].iloc[-1] > 0: # Bullish crossover above zero line
                buy_score += 3
                recommendations["Swing"] = "Strong Buy"
            elif macd_diff > 0: # Bullish crossover
                buy_score += 2
                recommendations["Swing"] = "Buy"
            elif macd_diff < 0 and data['MACD'].iloc[-1] < 0: # Bearish crossover below zero line
                sell_score += 3
                recommendations["Swing"] = "Strong Sell"
            elif macd_diff < 0: # Bearish crossover
                sell_score += 2
                recommendations["Swing"] = "Sell"

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

        if 'Upper_Band' in data.columns and 'Lower_Band' in data.columns and pd.notnull(data['Upper_Band'].iloc[-1]):
            close = data['Close'].iloc[-1]
            if close > data['Upper_Band'].iloc[-1]: # Price breaking above upper band
                buy_score += 2
                recommendations["Breakout"] = "Buy"
            elif close < data['Lower_Band'].iloc[-1]: # Price breaking below lower band
                sell_score += 2
                recommendations["Breakout"] = "Sell"

        if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
            cmf = data['CMF'].iloc[-1]
            if cmf > 0.2: # Strong buying pressure
                buy_score += 1
            elif cmf < -0.2: # Strong selling pressure
                sell_score += 1
            
        net_score = buy_score - sell_score
        
        # Consolidate recommendations based on net score
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
        # Else, "Hold" remains as initialized

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = round(score, 2)

        logging.info(f"Standard recommendations for {symbol}: {recommendations}")
        return recommendations

    except Exception as e:
        logging.error(f"Error generating standard recommendations for {symbol}: {str(e)}")
        return recommendations

@st.cache_data(ttl=3600)
def get_top_sectors_cached(rate_limit_delay=0.2, stocks_per_sector=5):
    sector_scores = {}
    for sector, stocks in SECTORS.items():
        total_score = 0
        count = 0
        for symbol in stocks[:stocks_per_sector]:
            data = fetch_stock_data_cached(symbol) # Uses cached data
            if data.empty:
                continue
            data = analyze_stock(data)
            score, _ = compute_signal_score(data, symbol)
            if score is not None:
                total_score += score
                count += 1
            # A short sleep to spread out load if not fully rate-limited
            # time.sleep(rate_limit_delay) # Removed here as fetch_stock_data_cached already rate-limits
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
    return sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]

@st.cache_data(ttl=3600)
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None):
    """
    Backtests a given strategy on historical data.
    _data_hash is used for Streamlit caching to invalidate the cache when data changes.
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
        "trade_details": []
    }
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    
    position = None
    entry_price = 0
    entry_date = None
    trades = []
    returns = []
    
    # Need enough data for indicator calculation (longest indicator window) plus one day for current open
    min_data_for_backtest_analysis = max(INDICATOR_MIN_LENGTHS.values()) 
    # We need data up to the signal day (i) AND the next day (i+1) for open/high/low/close
    # So the loop should start from min_data_for_backtest_analysis to ensure sliced_data_for_signal is valid
    # and it should run up to len(full_analyzed_data) - 1 to ensure current_day_data exists.
    if len(data) < min_data_for_backtest_analysis + 1:
        logging.warning(f"Not enough data to backtest {symbol}. Need at least {min_data_for_backtest_analysis + 1} rows, got {len(data)}")
        return results

    full_analyzed_data = analyze_stock(data.copy())
    
    # If analyze_stock failed or returned insufficient data, return empty results
    if full_analyzed_data.empty or len(full_analyzed_data) < min_data_for_backtest_analysis + 1:
        logging.warning(f"Analyzed data for {symbol} is insufficient for backtesting. Len: {len(full_analyzed_data)}")
        return results

    # Start backtesting from the point where all indicators are valid
    # `i` will be the index of the *signal day* (previous day's close for calculation)
    # `i+1` will be the *trade day* (next day's open/high/low/close)
    start_index = min_data_for_backtest_analysis -1 # Corrected start_index for 0-based indexing for `iloc` 
                                                  # so sliced_data_for_signal has at least min_data_for_backtest_analysis rows
                                                  # and current_day_data is available (i+1)

    for i in range(start_index, len(full_analyzed_data) - 1): # Loop up to second to last row
        # Data available *up to and including* day 'i' is used to generate the signal for day 'i+1'
        sliced_data_for_signal = full_analyzed_data.iloc[:i+1]
        
        # Data for day 'i+1' (the trade execution day)
        current_day_data = full_analyzed_data.iloc[i+1]

        # Re-validate sliced data for signal generation, especially at the beginning
        if not validate_data(sliced_data_for_signal, min_length=INDICATOR_MIN_LENGTHS['Ichimoku']):
            continue

        # Generate recommendation based on sliced_data_for_signal (i.e., data *before* the current trade day)
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data_for_signal, symbol=symbol)
            signal = rec["Recommendation"]
        else:
            rec = generate_recommendations(sliced_data_for_signal, symbol)
            signal = rec[strategy] if strategy in rec else "Hold"
        
        if signal is None:
            continue

        # Get prices for the *current* trading day (day after signal generation)
        trade_open_price = current_day_data['Open']
        trade_high_price = current_day_data['High']
        trade_low_price = current_day_data['Low']
        trade_close_price = current_day_data['Close']
        trade_date = current_day_data.name # This is the index (Date)

        # Add small random slippage
        slippage_pct = random.uniform(0.001, 0.005) # 0.1% to 0.5%
        
        if "Buy" in signal and position is None:
            entry_price_with_slippage = trade_open_price * (1 + slippage_pct)
            
            # Ensure we're not buying at a ridiculous price relative to signal
            rec_buy_at = rec.get("Buy At")
            
            # If recommended buy price exists and current open is significantly higher, skip
            if pd.notnull(rec_buy_at) and entry_price_with_slippage > rec_buy_at * 1.02: # Allow small premium
                 continue 

            position = "Long"
            entry_price = entry_price_with_slippage
            entry_date = trade_date
            results["buy_signals"].append((entry_date, entry_price))
        
        elif position == "Long":
            explicit_sell = "Sell" in signal
            
            stop_loss_price = rec.get("Stop Loss")
            target_price = rec.get("Target")
            
            exit_reason = None
            exit_price = trade_close_price # Default exit at close if no trigger

            # Check for intra-candle hits (order matters: stop-loss usually checked first)
            # If both stop and target are within the high/low range, assume stop hits first.
            if stop_loss_price and trade_low_price <= stop_loss_price:
                exit_price = stop_loss_price * (1 - slippage_pct) # Exit at stop with slippage
                exit_reason = "Stop Loss Hit"
            elif target_price and trade_high_price >= target_price:
                exit_price = target_price * (1 - slippage_pct) # Exit at target with slippage
                exit_reason = "Target Hit"
            elif explicit_sell: # Sell signal at close of the day (or market close)
                exit_price = trade_close_price * (1 - slippage_pct) 
                exit_reason = "Sell Signal"

            if exit_reason:
                position = None
                profit = exit_price - entry_price
                
                if entry_price != 0:
                    returns.append(profit / entry_price)
                    
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": trade_date,
                    "exit_price": exit_price,
                    "profit": profit,
                    "reason": exit_reason
                })
                results["sell_signals"].append((trade_date, exit_price))
                entry_price = 0
                entry_date = None
    
    # If a position is still open at the very end of the data, close it at the last available close price
    if position == "Long":
        # Exit on the last day's close
        final_close_price = full_analyzed_data['Close'].iloc[-1]
        exit_price = final_close_price * (1 - slippage_pct)
        profit = exit_price - entry_price
        if entry_price != 0:
            returns.append(profit / entry_price)
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": full_analyzed_data.index[-1],
            "exit_price": exit_price,
            "profit": profit,
            "reason": "Closed at end of period"
        })
        results["sell_signals"].append((full_analyzed_data.index[-1], exit_price))


    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)
        
        total_growth_factor = 1.0
        for r in returns:
            total_growth_factor *= (1 + r)
        results["total_return"] = (total_growth_factor - 1) * 100

        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100
        
        if returns:
            returns_series = pd.Series(returns)
            # Assuming 252 trading days for annualization for daily data
            results["annual_return"] = (returns_series.mean() * 252) * 100
            
            if returns_series.std() != 0:
                results["sharpe_ratio"] = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                results["sharpe_ratio"] = 0 # No volatility, so infinite Sharpe for positive return, 0 otherwise
            
            # Calculate Max Drawdown
            # Convert returns to an equity curve (starting at 100 for readability)
            equity_curve_values = [100] # Starting equity
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
    

def analyze_batch(stock_batch, progress_callback=None, status_callback=None):
    results = []
    errors = []
    # ThreadPoolExecutor to process stocks in parallel
    # The actual data fetching is rate-limited by @RateLimiter on fetch_stock_data_cached
    with ThreadPoolExecutor(max_workers=3) as executor: # Number of parallel analysis tasks
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    if "Error" in result: # Check for custom error messages returned
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
        data = fetch_stock_data_cached(symbol) # Call the cached and rate-limited function
        
        # Ensure data is sufficient before proceeding
        if data.empty or len(data) < INDICATOR_MIN_LENGTHS['Ichimoku']:
            logging.warning(f"No sufficient data for {symbol} after fetch: {len(data)} rows")
            return None

        data = analyze_stock(data)
        
        # Check again after analysis, as some indicators might result in NaNs if not enough history
        # Ensure that latest Close and ATR are valid for recommendations
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
                "Intraday": None, "Swing": None, "Short-Term": None, # Set to None for Adaptive mode
                "Long-Term": None, "Mean_Reversion": None, "Breakout": None,
                "Ichimoku_Trend": None
            }
        else:
            rec = generate_recommendations(data, symbol)
            if not rec or not rec.get('Intraday'): # Using Intraday as a proxy for any standard rec
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
                "Recommendation": None, "Regime": None, # Set to None for Standard mode
                "Position Size Shares": None, "Position Size Value": None, "Trailing Stop": None,
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
            status_text_message = f"🔄 Analyzing: {batch_names}"
            status_callback(status_text_message) # Pass message to status_callback
        batch_results = analyze_batch(batch, progress_callback=progress_callback, status_callback=status_callback)
        results.extend([r for r in batch_results if r is not None])
        processed += len(batch)
        if progress_callback:
            progress_callback(processed / total_stocks)
        # Increased sleep between batches to respect overall API rate limits
        # (1 call/1.5s * 3 workers = 2 calls/sec). So, for a batch of 10, need at least 5 seconds.
        # Adjusted sleep to ensure enough time for concurrent requests to complete and clear rate limit.
        time.sleep(max(10, batch_size * 1.5 / 3)) # Ensure enough time passes for all calls in batch across workers
        
    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    
    # Ensure all expected columns exist, fill with NaN if not
    expected_cols = [
        "Symbol", "Current Price", "Buy At", "Stop Loss", "Target", "Score",
        "Recommendation", "Regime", "Position Size Shares", "Position Size Value",
        "Trailing Stop", "Reason", "Intraday", "Swing", "Short-Term", "Long-Term",
        "Mean_Reversion", "Breakout", "Ichimoku_Trend"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = np.nan

    return results_df.sort_values(by="Score", ascending=False)

    
def analyze_intraday_stocks(stock_list, batch_size=3, progress_callback=None, status_callback=None):
    # Note: This function still uses 1d interval data. If true intraday data (e.g., 1m or 5m candles)
    # is desired for "Intraday Picks", the `fetch_stock_data_cached` call within `analyze_stock_parallel`
    # would need its `interval` parameter changed (e.g., `interval="1m"` and `period="7d"`).
    # This change would drastically increase API calls and processing time.
    
    results = []
    total_stocks = len(stock_list)
    processed = 0
    
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        
        if status_callback:
            batch_names = ", ".join(batch[:3])
            if len(batch) > 3:
                batch_names += f" and {len(batch)-3} more"
            status_text_message = f"🔄 Analyzing: {batch_names}"
            status_callback(status_text_message)
        
        batch_results = analyze_batch(batch, progress_callback=progress_callback, status_callback=status_callback)
        results.extend([r for r in batch_results if r is not None])
        
        processed += len(batch)
        if progress_callback:
            progress_callback(processed / total_stocks)
        
        time.sleep(max(10, batch_size * 1.5 / 3)) # Consistent sleep with analyze_all_stocks
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()
    
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
        # For intraday picks, filter for "Buy" recommendations, regardless of specific type (e.g., strong buy, volatile opportunity)
        filtered_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False, case=False)]
    else:
        # For standard mode, filter explicitly for "Intraday" buy signals
        filtered_df = results_df[results_df["Intraday"].str.contains("Buy", na=False, case=False)]
        
    return filtered_df.sort_values(by="Score", ascending=False).head(5)

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
            
def update_progress_with_status(progress_bar, loading_text, status_text, progress_value, stock_status=None):
    progress_bar.progress(progress_value)
    
    percentage = int(progress_value * 100)
    
    loading_text.text(f"Progress: {percentage}%")
    
    if stock_status:
        status_text.text(stock_status)

def insert_top_picks_supabase(results_df, pick_type="daily"):
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')

    if recommendation_mode == "Adaptive":
        filtered_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False, case=False)].head(5)
    else:
        filtered_df = results_df[
            (results_df["Intraday"].str.contains("Buy", na=False, case=False)) |
            (results_df["Swing"].str.contains("Buy", na=False, case=False)) |
            (results_df["Short-Term"].str.contains("Buy", na=False, case=False)) |
            (results_df["Long-Term"].str.contains("Buy", na=False, case=False))
        ].sort_values(by="Score", ascending=False).head(5)

    if filtered_df.empty:
        logging.info(f"No {pick_type} picks to insert into Supabase.")
        return

    records_to_insert = []
    for _, row in filtered_df.iterrows():
        # Prepare data for Supabase, converting np.nan to None
        record = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "symbol": row.get('Symbol') or row.get('symbol', 'Unknown'),
            "score": float(row.get('Score', 0) or 0),
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
    
    # Filter out records where symbol or date is missing, and ensure numeric fields are correctly None for NaN
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
        # Ensure your Supabase table 'daily_picks' has all these columns with appropriate types.
        # e.g., 'symbol' TEXT, 'score' REAL, 'current_price' REAL, 'recommendation' TEXT etc.
        # The error about 'position_size_shares' missing indicates these columns are needed.
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


@RateLimiter(calls=1, period=1)
def fetch_latest_price(symbol):
    # Fetch 2 days of 1-day interval data to ensure we get today's close if available
    data = fetch_stock_data_cached(symbol, period="2d", interval="1d") # Use cached fetcher
    if not data.empty and 'Close' in data.columns and pd.notnull(data['Close'].iloc[-1]):
        return float(data['Close'].iloc[-1])
    return None

def update_with_latest_prices(df):
    symbols_to_fetch = df['symbol'].unique()
    latest_prices = {}
    
    # Use ThreadPoolExecutor for concurrent price fetching
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

    # Update DataFrame
    updated_df = df.copy()
    updated_df['current_price'] = updated_df['symbol'].map(latest_prices)
    
    # Handle cases where latest price couldn't be fetched (keep original if available, else NaN)
    updated_df['current_price'] = updated_df['current_price'].fillna(df['current_price'])
    
    return updated_df

        
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
        st.session_state.show_history = False

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

    if st.button("🔎 Analyze Top Performing Sectors"):
        with st.spinner("🔍 Crunching sector data ..."):
            top_sectors = get_top_sectors_cached(rate_limit_delay=10, stocks_per_sector=10)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            if top_sectors:
                for name, score in top_sectors:
                    st.markdown(f"- **{name}**: {score:.2f}/10")
            else:
                st.info("No sector data available or able to be processed.")


    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        status_text = st.empty() # Placeholder for current stock status
        
        status_text.text(f"📊 Analyzing {len(selected_stocks)} stocks...")
        
        results_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress_with_status(progress_bar, loading_text, status_text, x),
            status_callback=lambda status: status_text.text(status) # Pass actual status message
        )
        
        insert_top_picks_supabase(results_df, pick_type="daily")
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()
        
        display_results_df = pd.DataFrame()
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        
        if recommendation_mode == "Adaptive":
            display_results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False, case=False)].sort_values(by="Score", ascending=False).head(5)
        else:
            display_results_df = results_df[
                (results_df["Intraday"].str.contains("Buy", na=False, case=False)) |
                (results_df["Swing"].str.contains("Buy", na=False, case=False)) |
                (results_df["Short-Term"].str.contains("Buy", na=False, case=False)) |
                (results_df["Long-Term"].str.contains("Buy", na=False, case=False))
            ].sort_values(by="Score", ascending=False).head(5)


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
                        st.markdown(f"""
                        Current Price: {current_price}  
                        Buy At: {buy_at} | Stop Loss: {stop_loss}  
                        Target: {target}  
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

    if st.button("⚡ Generate Intraday Top 5 Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        status_text = st.empty()
        
        status_text.text(f"📊 Analyzing {len(selected_stocks)} stocks for intraday...")
        
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10, 
            progress_callback=lambda x: update_progress_with_status(progress_bar, loading_text, status_text, x),
            status_callback=lambda status: status_text.text(status)
        )
        
        insert_top_picks_supabase(intraday_results, pick_type="intraday")
        progress_bar.empty()
        loading_text.empty()
        status_text.empty()
        
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks")
            for _, row in intraday_results.iterrows():
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
                        st.markdown(f"""
                        Current Price: {current_price}  
                        Buy At: {buy_at} | Stop Loss: {stop_loss}  
                        Target: {target}  
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}
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
                    
                    # Columns that should be numeric
                    numeric_cols = [
                        'score', 'current_price', 'buy_at', 'target', 'stop_loss',
                        'position_size_shares', 'position_size_value', 'trailing_stop'
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            # Convert to numeric, errors will result in NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                    df = update_with_latest_prices(df) # Get latest prices for historical picks
                    df = add_action_and_change(df)
                    
                    # Format for display AFTER calculations (e.g., % Change)
                    for col in ['buy_at', 'current_price', 'target', 'stop_loss', 'position_size_value', 'trailing_stop']:
                        if col in df.columns:
                            df[col] = df[col].map(lambda x: f"₹{x:.2f}" if pd.notnull(x) else 'N/A')
                    if 'position_size_shares' in df.columns:
                         df['position_size_shares'] = df['position_size_shares'].map(lambda x: f"{int(x)}" if pd.notnull(x) else 'N/A')
                    if '% Change' in df.columns:
                        df['% Change'] = df['% Change'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else 'N/A')

                    # Determine which columns to display based on whether 'recommendation' (Adaptive) exists and has data
                    if 'recommendation' in df.columns and df['recommendation'].notna().any() and \
                       df['recommendation'].str.contains("Buy|Sell|Hold", case=False, na=False).any(): # Check if it contains actual Adaptive recs
                        
                        display_cols = [
                            "symbol", "buy_at", "current_price", "% Change", "What to do now?",
                            "recommendation", "regime", "position_size_shares", "position_size_value",
                            "trailing_stop", "reason", "target", "stop_loss", "pick_type", "score"
                        ]
                        final_display_df = df[[col for col in display_cols if col in df.columns]]
                        # Apply colored recommendations to the 'recommendation' column for consistency
                        if 'recommendation' in final_display_df.columns:
                            final_display_df['recommendation'] = final_display_df['recommendation'].apply(colored_recommendation)
                    else: # Fallback to standard columns if Adaptive mode recommendations are not primary
                        standard_cols = ["symbol", "buy_at", "current_price", "% Change", "What to do now?", 
                                         "intraday", "swing", "short_term", "long_term", "mean_reversion", 
                                         "breakout", "ichimoku_trend", "target", "stop_loss", "pick_type", "score"]
                        final_display_df = df[[col for col in standard_cols if col in df.columns]]
                        # Apply colored recommendations to relevant standard columns
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
        
    # Display analysis for the currently selected stock (if any)
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
            pos_value = recommendations.get('Position Size', {}).get('value', 'N/A')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Recommendation**: {colored_recommendation(recommendations.get('Recommendation', 'N/A'))}")
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10")
            with col2:
                st.write(f"**Position Size**: {pos_shares} shares")
                st.write(f"**Value**: ₹{pos_value}")
            with col3:
                st.write(f"**Trailing Stop**: ₹{recommendations.get('Trailing Stop', 'N/A')}")
                st.write(f"**Volatility**: {assess_risk(data)}")
            st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")

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
            st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/10")
            st.write(f"**Volatility**: {assess_risk(data)}")

        with st.form(key="backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                swing_button = st.form_submit_button("🔍 Backtest Swing Strategy")
            with col2:
                intraday_button = st.form_submit_button("🔍 Backtest Intraday Strategy")
            
            if swing_button or intraday_button:
                strategy = "Swing" if swing_button else "Intraday"
                with st.spinner(f"Running {strategy} Strategy backtest... (This may take a while for large data sets)"):
                    # Use a hash of the raw data to ensure backtest cache invalidates if underlying data changes
                    # Hashing only the latest 300 rows of core OHLCV data
                    data_for_hash = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(300).to_json()
                    data_hash = hash(data_for_hash) 
                    backtest_results = backtest_stock(data, symbol, strategy=strategy, _data_hash=data_hash)
                    if strategy == "Swing":
                        st.session_state.backtest_results_swing = backtest_results
                    else:
                        st.session_state.backtest_results_intraday = backtest_results

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
                data = fetch_stock_data_cached(symbol) # Use the cached fetcher
                # Max length for indicators is Ichimoku 52 periods. Need at least 52+1 for current day for backtesting.
                required_min_length = max(INDICATOR_MIN_LENGTHS.values()) + 1 
                if not data.empty and len(data) >= required_min_length:
                    data = analyze_stock(data)
                    
                    if data.empty or pd.isna(data['Close'].iloc[-1]) or ('ATR' in data.columns and pd.isna(data['ATR'].iloc[-1])):
                        st.warning(f"⚠️ Could not complete analysis for {normalize_symbol_dhan(symbol)} due to insufficient or invalid data after indicator computation.")
                        st.session_state.symbol = None # Reset selected symbol
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
                    display_dashboard(symbol, data, recommendations)
                else:
                    st.warning(f"⚠️ No sufficient historical data available for {normalize_symbol_dhan(symbol)} to perform a full analysis ({len(data)} rows found, need at least {required_min_length}).")
                    st.session_state.symbol = None # Reset selected symbol for clearer state
    else:
        # Initial display or re-run where symbol/data might be in session_state
        display_dashboard()

    # Clear cache button
    if st.sidebar.button("Clear All Caches", help="Clears cached data for stock prices and backtests. Useful if data seems stale or for debugging."):
        st.cache_data.clear() # Clears Streamlit's file-based cache
        fetch_stock_data_cached.cache_clear() # Clears functools.lru_cache for stock data
        st.session_state.data = None
        st.session_state.recommendations = None
        st.session_state.backtest_results_swing = None
        st.session_state.backtest_results_intraday = None
        st.success("All caches cleared!")
        st.rerun() # Rerun the app to reflect the cleared state

if __name__ == "__main__":
    main()
