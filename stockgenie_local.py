import pandas as pd
import ta
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from functools import wraps
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
    # Chrome on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    
    # Chrome on macOS Sonoma
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    
    # Chrome on macOS with Apple Silicon
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    
    # Firefox on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:127.0) Gecko/20100101 Firefox/127.0",
    
    # Safari on macOS Sonoma
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    
    # Edge on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    
    # Chrome on Android 14 (Samsung Galaxy)
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    
    # Chrome on Android 14 (Pixel)
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    
    # Safari on iPhone (iOS 17.5)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    
    # Safari on iPad (iOS 17.5)
    "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    
    # Opera on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 OPR/112.0.0.0",
    
    # Samsung Internet on Android
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/25.0 Chrome/121.0.0.0 Mobile Safari/537.36",
    
    # Brave on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Brave/126.0.0.0",
    
    # Chrome on Linux (Ubuntu)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
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
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
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
    "HINDMOTORS.NS", "MAHSCOOTER.NS","HINDMOTORS.NS"
  ],

  "Healthcare": [
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "LUPIN.NS",
    "DIVISLAB.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "IPCALAB.NS", "GLENMARK.NS", "BIOCON.NS", "ABBOTINDIA.NS", "SANOFI.NS",
    "PFIZER.NS", "GLAXO.NS", "NATCOPHARM.NS", "AJANTPHARM.NS", "GRANULES.NS",
    "LAURUSLABS.NS", "STAR.NS", "JUBLPHARMA.NS", "ASTRAZEN.NS", "WOCKPHARDT.NS","PPLPHARMA.NS"
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
    "LT.NS", "SHAKTIPUMP.NS","SIEMENS.NS", "ABB.NS", "BEL.NS", "BHEL.NS", "HAL.NS",
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        if attempt == max_retries:
                            st.error("Max retries reached due to rate limiting (HTTP 429).")
                            raise
                        sleep_time = (delay * (backoff_factor ** (attempt - 1))) + random.uniform(0, jitter)
                        st.warning(f"Rate limit hit (429). Attempt {attempt}/{max_retries}. Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    else:
                        raise
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries:
                        st.error("Max retries reached due to a network error.")
                        raise
                    sleep_time = (delay * (backoff_factor ** (attempt - 1))) + random.uniform(0, jitter)
                    st.warning(f"Network error: {e}. Attempt {attempt}/{max_retries}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
            # This should never be reached, but added for safety
            raise RuntimeError("Retry mechanism exhausted without success or proper exception raised.")
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

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s: %(message)s")

def validate_data(
    data: pd.DataFrame,
    required_columns=None,
    min_length: int = 30,
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

INDICATOR_MIN_LENGTHS = {
    'RSI': 14,
    'MACD': 26,
    'ATR': 14,
    'ADX': 27,
    'CMF': 20,
    'Donchian': 20,
    'Bollinger': 20,
    'Stochastic': 14,
    'Ichimoku': 52,
    'Volume_Spike': 10
}

def can_compute_indicator(data, indicator):
    required_length = INDICATOR_MIN_LENGTHS.get(indicator, 1)
    return len(data) >= required_length

def analyze_stock(data):
    if not validate_data(data, min_length=50):
        columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                   'ATR', 'ADX', 'CMF',
                   'Upper_Band', 'Middle_Band', 'Lower_Band',
                   'SlowK', 'SlowD',
                   'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle',
                   'Ichimoku_Tenkan', 'Ichimoku_Kijun',
                   'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou',
                   'Volume_Spike', 'Avg_Volume']
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
        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                data['High'], data['Low'], data['Close'], data['Volume'], window=20
            ).chaikin_money_flow()
        else:
            data['CMF'] = None
    except Exception as e:
        logging.warning(f"Failed to compute CMF: {str(e)}")
        data['CMF'] = None

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
        if can_compute_indicator(data, 'Donchian'):
            donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
            data['Donchian_Upper'] = donchian.donchian_channel_hband()
            data['Donchian_Lower'] = donchian.donchian_channel_lband()
            data['Donchian_Middle'] = donchian.donchian_channel_mband()
        else:
            data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Donchian Channels: {str(e)}")
        data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None

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

def fetch_fundamentals(symbol):
    try:
        smart_api = init_smartapi_client()
        if not smart_api:
            return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

# Improved strategy logic using adaptive regime detection, signal scoring, and volatility-aware filters

import pandas as pd
import numpy as np
import logging
import ta

logging.basicConfig(level=logging.INFO)

# ============================ #
#       INDICATOR SETUP        #
# ============================ #

_indicator_cache = {}

def compute_indicators(df, symbol=None):
    cache_key = symbol or id(df)
    if cache_key in _indicator_cache:
        return _indicator_cache[cache_key]

    df = df.copy()
    if df.empty or 'Close' not in df.columns:
        return df

    # Trend indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    # Momentum indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Ichimoku indicators
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    df['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_Span_B'] = ichimoku.ichimoku_b()

    # Volume indicators
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).chaikin_money_flow()

    # Volatility indicators
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close']
    ).average_true_range()

    # Donchian Channels (for breakout)
    df['Donchian_Upper'] = df['High'].rolling(20).max()
    df['Donchian_Lower'] = df['Low'].rolling(20).min()

    # Bollinger Bands (for mean reversion)
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['Bollinger_Upper'] = bb.bollinger_hband()
    df['Bollinger_Lower'] = bb.bollinger_lband()

    # Stochastic Oscillator (secondary momentum)
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Trend strength
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

    # VWAP (Volume Weighted Average Price)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Average volume (for filters or spike detection)
    df['Avg_Volume'] = df['Volume'].rolling(20).mean()

    _indicator_cache[cache_key] = df
    return df

# ============================ #
#       TREND & REGIME         #
# ============================ #

def classify_market_regime(df):
    try:
        sma20 = df['SMA_20'].iloc[-1]
        sma50 = df['SMA_50'].iloc[-1]
        sma200 = df['SMA_200'].iloc[-1]
        close = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        slope = (sma20 - df['SMA_20'].iloc[-5]) / df['SMA_20'].iloc[-5]

        if any(pd.isna([sma20, sma50, close, atr])):
            return "Unknown"

        atr_pct = atr / close

        if atr_pct > 0.03:
            return "High Volatility"
        elif sma20 > sma50 and sma50 > sma200 and slope > 0.01:
            return "Strong Bullish"
        elif sma20 < sma50 and slope < -0.01:
            return "Strong Bearish"
        elif sma20 > sma50:
            return "Bullish"
        elif sma20 < sma50:
            return "Bearish"
        else:
            return "Neutral"
    except:
        return "Unknown"

def detect_regime_instability(df):
    if len(df) < 5:
        return 1.0
    recent = [classify_market_regime(df.iloc[i-1:i+1]) for i in range(len(df)-4, len(df))]
    return len(set(recent)) / len(recent)

def get_signal_correlation(df):
    subset = df[['RSI', 'Stoch_K', 'WilliamsR']].dropna()
    return subset.corr().abs().mean().mean() if not subset.empty else 0

def get_higher_timeframe_trend(df, tf='W'):
    higher = df[['Close']].resample(tf).last()
    return 'Uptrend' if higher['Close'].iloc[-1] > higher['Close'].iloc[-2] else 'Downtrend'

# ============================ #
#       SCORING ENGINE         #
# ============================ #

def compute_signal_score(df, regime):
    w = {
        'RSI': 2.0, 'MACD': 1.5, 'Ichimoku': 2.0, 'CMF': 0.5,
        'ATR_Volatility': 1.0, 'Breakout': 1.2,
        'Stochastic': 1.0,
        'MeanReversion': 1.2, 'TrendStrength': 1.0
    }

    score = 0
    close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    atr_pct = atr / close if close else 0

    rsi_oversold, rsi_overbought = (25, 75) if atr_pct > 0.04 else (30, 70)

    rsi = df['RSI'].iloc[-1]
    if pd.notna(rsi):
        if rsi < rsi_oversold:
            score += w['RSI']
        elif rsi > rsi_overbought:
            score -= w['RSI']
        elif rsi > 50 and 'Bullish' in regime:
            score += w['RSI'] * 0.5

    macd, macd_sig = df['MACD'].iloc[-1], df['MACD_signal'].iloc[-1]
    if pd.notna(macd) and pd.notna(macd_sig):
        score += w['MACD'] * np.sign(macd - macd_sig)

    span_a, span_b = df['Ichimoku_Span_A'].iloc[-1], df['Ichimoku_Span_B'].iloc[-1]
    if pd.notna(span_a) and pd.notna(span_b):
        if close > max(span_a, span_b):
            score += w['Ichimoku']
        elif close < min(span_a, span_b):
            score -= w['Ichimoku']

    cmf = df['CMF'].iloc[-1]
    if pd.notna(cmf):
        score += w['CMF'] * cmf

    if atr_pct > 0.04:
        score -= w['ATR_Volatility'] * (atr_pct / 0.04)

    upper, lower = df['Donchian_Upper'].iloc[-1], df['Donchian_Lower'].iloc[-1]
    if close > upper:
        score += w['Breakout']
    elif close < lower:
        score -= w['Breakout']

    k, d = df['Stoch_K'].iloc[-1], df['Stoch_D'].iloc[-1]
    if pd.notna(k) and pd.notna(d):
        if k > d and k < 20:
            score += w['Stochastic']
        elif k < d and k > 80:
            score -= w['Stochastic']

    bb_upper, bb_lower = df['Bollinger_Upper'].iloc[-1], df['Bollinger_Lower'].iloc[-1]
    if close < bb_lower:
        score += w['MeanReversion']
    elif close > bb_upper:
        score -= w['MeanReversion']

    adx = df['ADX'].iloc[-1]
    if pd.notna(adx):
        if adx > 25:
            score += w['TrendStrength']
        elif adx < 15:
            score -= w['TrendStrength']

    if get_signal_correlation(df) > 0.8:
        score *= 0.85

    return round(np.clip(score, -10, 10), 2)

# ============================ #
#     MAIN RECOMMENDATION      #
# ============================ #

def adaptive_recommendation(df, symbol=None, account_size=30000, max_position_size=100):
    try:
        df = compute_indicators(df, symbol)
        close = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        volume = df['Volume'].iloc[-1]
        avg_vol = df['Avg_Volume'].iloc[-1]

        if any(pd.isna([close, atr, volume, avg_vol])):
            return {"Recommendation": "Hold", "Reason": "Missing values"}

        regime = classify_market_regime(df)
        regime_stability = detect_regime_instability(df)

        score = compute_signal_score(df, regime)
        if regime_stability > 0.5:
            score *= 0.7

        thresholds = {
            'High Volatility': 1.0, 'Bullish': 0.5,
            'Strong Bullish': 0.3, 'Bearish': 0.8,
            'Strong Bearish': 1.0, 'Neutral': 0.6
        }
        threshold = thresholds.get(regime, 1.0)

        if close < 50 or atr < 2 or volume < 1000:
            return {"Recommendation": "Hold", "Reason": "Very illiquid", "Score": score}

        if score > threshold:
            rec = "Buy"
        elif score < -threshold:
            rec = "Sell"
        else:
            rec = "Hold"

        buy_at = close * 1.005 if rec == "Buy" else None
        stop_loss = close * 0.95 if rec == "Buy" else close * 1.05 if rec == "Sell" else None
        target = close * 1.05 if rec == "Buy" else close * 0.95 if rec == "Sell" else None

        max_loss_pct = 0.12
        if stop_loss and abs(close - stop_loss) / close > max_loss_pct:
            return {"Recommendation": "Hold", "Reason": "Stop loss too wide", "Score": score}

        risk_per_trade = 0.02
        stop_pct = abs(close - stop_loss) / close if stop_loss else 0.05
        position_size = min((account_size * risk_per_trade) / (close * stop_pct), max_position_size)

        atr_mult = 1.5 if regime == "High Volatility" else 2.0
        trailing_stop = close - atr_mult * atr if rec == "Buy" else close + atr_mult * atr if rec == "Sell" else None

        return {
            "Current Price": round(close, 2),
            "Buy At": round(buy_at, 2) if buy_at else None,
            "Stop Loss": round(stop_loss, 2) if stop_loss else None,
            "Target": round(target, 2) if target else None,
            "Recommendation": rec,
            "Score": score,
            "Regime": regime,
            "Regime Stability": round(regime_stability, 2),
            "Confidence Threshold": threshold,
            "Position Size": int(position_size),
            "Trailing Stop": round(trailing_stop, 2) if trailing_stop else None
        }
    except Exception as e:
        logging.error(f"Recommendation failed: {e}")
        return {"Error": str(e)}

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

        # RSI logic
        if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if isinstance(rsi, (int, float, np.integer, np.floating)):
                if rsi <= 20:
                    buy_score += 4
                elif rsi < 30:
                    buy_score += 2
                elif rsi > 70:
                    sell_score += 2

        # MACD logic
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_sig = data['MACD_signal'].iloc[-1]
            if pd.notna(macd) and pd.notna(macd_sig):
                if macd > macd_sig:
                    buy_score += 1
                elif macd < macd_sig:
                    sell_score += 1

        # Bollinger Band breakout
        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            close = data['Close'].iloc[-1]
            lower = data['Lower_Band'].iloc[-1]
            upper = data['Upper_Band'].iloc[-1]
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in [close, lower, upper]):
                if close < lower:
                    buy_score += 1
                elif close > upper:
                    sell_score += 1

        # VWAP logic
        if 'VWAP' in data.columns and pd.notna(data['VWAP'].iloc[-1]):
            vwap = data['VWAP'].iloc[-1]
            close = data['Close'].iloc[-1]
            if isinstance(vwap, (int, float, np.integer, np.floating)) and isinstance(close, (int, float, np.integer, np.floating)):
                if close > vwap:
                    buy_score += 1
                elif close < vwap:
                    sell_score += 1

        # Volume/Price movement
        if 'Volume' in data.columns and 'Avg_Volume' in data.columns:
            volume = data['Volume'].iloc[-1]
            avg_vol = data['Avg_Volume'].iloc[-1]
            if pd.notna(volume) and pd.notna(avg_vol) and avg_vol != 0:
                ratio = volume / avg_vol
                close = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2]
                if ratio > 1.5:
                    if close > prev_close:
                        buy_score += 2
                    elif close < prev_close:
                        sell_score += 2
                elif ratio < 0.5:
                    sell_score += 1

        # Volume spike logic
        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1]:
            close = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            if close > prev_close:
                buy_score += 1
            else:
                sell_score += 1

        # Divergence
        if 'Divergence' in data.columns and pd.notna(data['Divergence'].iloc[-1]):
            div = data['Divergence'].iloc[-1]
            if div == "Bullish Divergence":
                buy_score += 1
            elif div == "Bearish Divergence":
                sell_score += 1

        # Ichimoku trend
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
            span_a = data['Ichimoku_Span_A'].iloc[-1]
            span_b = data['Ichimoku_Span_B'].iloc[-1]
            close = data['Close'].iloc[-1]
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in [span_a, span_b, close]):
                if close > max(span_a, span_b):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif close < min(span_a, span_b):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Sell"

        # Ichimoku confirmation (Tenkan/Kijun)
        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns:
            tenkan = data['Ichimoku_Tenkan'].iloc[-1]
            kijun = data['Ichimoku_Kijun'].iloc[-1]
            span_a = data.get('Ichimoku_Span_A', pd.Series([np.nan])).iloc[-1]
            span_b = data.get('Ichimoku_Span_B', pd.Series([np.nan])).iloc[-1]
            close = data['Close'].iloc[-1]
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in [tenkan, kijun, close]):
                if tenkan > kijun and close > span_a:
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif tenkan < kijun and close < span_b:
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"

        # CMF (volume + price strength)
        if 'CMF' in data.columns and pd.notna(data['CMF'].iloc[-1]):
            cmf = data['CMF'].iloc[-1]
            if cmf > 0:
                buy_score += 1
            elif cmf < 0:
                sell_score += 1

        # Donchian channel breakout
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
            upper = data['Donchian_Upper'].iloc[-1]
            lower = data['Donchian_Lower'].iloc[-1]
            close = data['Close'].iloc[-1]
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in [upper, lower, close]):
                if close > upper:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                elif close < lower:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"

        # Mean reversion via RSI and Bollinger Bands
        if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns:
            rsi = data['RSI'].iloc[-1]
            close = data['Close'].iloc[-1]
            lower = data['Lower_Band'].iloc[-1]
            upper = data['Upper_Band'].iloc[-1]
            if all(isinstance(val, (int, float, np.integer, np.floating)) for val in [rsi, close, lower, upper]):
                if rsi < 30 and close >= lower:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif rsi > 70 and close >= upper:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"

        # Fundamentals (optional)
        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 2
            elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
                sell_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 1
            elif fundamentals['RevenueGrowth'] < 0:
                sell_score += 0.5

        # Final recommendations
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

        # Final values
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = min(max(net_score, -7), 7)

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

@st.cache_data(ttl=3600)
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None):
    INITIAL_CAPITAL = 30000
    commission = 0.001  # 0.1% round-trip
    position_size_pct = st.session_state.get('position_size', 1.0)

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
        "equity_curve": []
    }

    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL

    position = None
    entry_price = 0
    entry_date = None
    trade_qty = 0
    trades = []
    returns = []
    equity_curve = []

    for i in range(1, len(data)):
        sliced_data = data.iloc[:i+1]
        current_price = data['Close'].iloc[i]
        current_date = data.index[i]

        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data)
            signal = rec.get("Recommendation", "Hold")
        else:
            rec = generate_recommendations(sliced_data, symbol)
            signal = rec.get(strategy, "Hold")

        # Execute sell first (if needed)
        if signal == "Sell" and position == "Long":
            gross_profit = (current_price - entry_price) * trade_qty
            trade_cost = (entry_price + current_price) * trade_qty * commission / 2
            net_profit = gross_profit - trade_cost
            cash += net_profit
            portfolio_value = cash
            pct_return = net_profit / (entry_price * trade_qty) if entry_price else 0

            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": current_date,
                "exit_price": current_price,
                "profit": net_profit
            })
            returns.append(pct_return)
            results["sell_signals"].append((current_date, current_price))
            position = None
            entry_price = 0
            entry_date = None
            trade_qty = 0

        # Execute buy (if no position)
        if signal == "Buy" and position is None:
            entry_price = current_price
            entry_date = current_date
            allocation = portfolio_value * position_size_pct
            trade_qty = int(allocation // current_price)
            if trade_qty == 0:
                continue
            cost = trade_qty * current_price * (1 + commission / 2)
            cash -= cost
            position = "Long"
            results["buy_signals"].append((current_date, current_price))

        # Track portfolio value *after* trade logic
        portfolio_value = cash + (trade_qty * current_price if position else 0)
        equity_curve.append((current_date, portfolio_value))

    # Close open position at end
    if position == "Long":
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1]
        gross_profit = (current_price - entry_price) * trade_qty
        trade_cost = (entry_price + current_price) * trade_qty * commission / 2
        net_profit = gross_profit - trade_cost
        cash += net_profit
        portfolio_value = cash
        pct_return = net_profit / (entry_price * trade_qty) if entry_price else 0

        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": current_date,
            "exit_price": current_price,
            "profit": net_profit
        })
        returns.append(pct_return)
        results["sell_signals"].append((current_date, current_price))
        equity_curve.append((current_date, portfolio_value))

    # Final stats
    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)
        results["total_return"] = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100

        # Annual return
        holding_periods = [
            (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).days
            for t in trades if t["entry_date"] != t["exit_date"]
        ]
        avg_holding = np.mean(holding_periods) if holding_periods else 1
        annual_factor = 252 / avg_holding if avg_holding else 1
        results["annual_return"] = np.mean(returns) * annual_factor * 100 if returns else 0

        # Sharpe
        std_dev = np.std(returns)
        results["sharpe_ratio"] = (np.mean(returns) / (std_dev + 1e-9)) * np.sqrt(252) if returns else 0

    # Max drawdown
    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    equity_df["RunningMax"] = equity_df["Equity"].cummax()
    equity_df["Drawdown"] = (equity_df["Equity"] - equity_df["RunningMax"]) / equity_df["RunningMax"]
    results["max_drawdown"] = equity_df["Drawdown"].min() * 100  # usually negative
    results["equity_curve"] = equity_df

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
            if not isinstance(rec, dict) or not rec.get('Recommendation'):
                logging.warning(f"No valid adaptive recommendation for {symbol}: {rec}")
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
            if not isinstance(rec, dict) or not rec.get('Intraday'):
                logging.warning(f"No valid standard recommendation for {symbol}: {rec}")
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

def analyze_intraday_stocks(stock_list, batch_size=10, delay=3, top_n=5, progress_callback=None):
    if not stock_list:
        st.warning("Empty stock list provided.")
        return pd.DataFrame()

    results = []
    total = len(stock_list)

    for i in range(0, total, batch_size):
        batch = stock_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            batch_results = analyze_batch(batch)
            if batch_results:
                results.extend([r for r in batch_results if r is not None])
                st.info(f"Batch {batch_num} processed successfully with {len(batch_results)} results.")
            else:
                st.warning(f"Batch {batch_num} returned no results.")
        except Exception as e:
            st.warning(f"Error analyzing batch {batch_num}: {e}")
            continue

        if progress_callback:
            progress = min(1.0, (i + len(batch)) / total)
            progress_callback(progress)

        if i + batch_size < total:
            time.sleep(delay)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("No results found after processing all batches.")
        return pd.DataFrame()

    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None

    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive" and "Recommendation" in results_df.columns:
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", case=False, na=False)]
    elif "Intraday" in results_df.columns:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", case=False, na=False)]
    else:
        st.warning("Neither 'Recommendation' nor 'Intraday' columns found for filtering.")
        return pd.DataFrame()

    return results_df.sort_values(by="Score", ascending=False).head(top_n)

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
            ("VWAP", data['VWAP'].iloc[-1] if 'VWAP' in data.columns and not data['VWAP'].isna().all() else "N/A", TOOLTIPS['VWAP']),
            ("Ichimoku Span A", data['Ichimoku_Span_A'].iloc[-1], TOOLTIPS['Ichimoku']),
            ("CMF", data['CMF'].iloc[-1], TOOLTIPS['CMF']),
        ]
        col1, col2 = st.columns(2)
        for i, (name, value, tooltip_text) in enumerate(indicators):
            value = round(value, 2) if isinstance(value, (int, float, np.integer, np.floating)) else value
            if i % 2 == 0:
                with col1:
                    st.write(f"**{tooltip(name, tooltip_text)}**: {value}")
            else:
                with col2:
                    st.write(f"**{tooltip(name, tooltip_text)}**: {value}")

def main():
    init_database()
    st.sidebar.title("🔍 Stock Selection")

    stock_list = fetch_nse_stock_list()
    if not stock_list:
        st.error("❌ Could not fetch stock list. Please check your internet connection or API status.")
        return

    # Set default session states
    if 'symbol' not in st.session_state:
        st.session_state.symbol = stock_list[0]
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"

    # Stock selection
    selected_index = stock_list.index(st.session_state.symbol) if st.session_state.symbol in stock_list else 0
    symbol = st.sidebar.selectbox(
        "Select Stock",
        stock_list,
        index=selected_index,
        key="stock_select"
    )

    # Recommendation mode selection
    recommendation_mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Standard", "Adaptive"],
        index=0 if st.session_state.recommendation_mode == "Standard" else 1,
        help="Standard: Timeframe-specific recommendations. Adaptive: Regime-based with position sizing."
    )
    st.session_state.recommendation_mode = recommendation_mode

    # Analyze button
    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner("📊 Loading and analyzing stock data..."):
                try:
                    data = fetch_stock_data_with_auth(symbol)
                    if not data.empty:
                        analyzed_data = analyze_stock(data)
                        recommendations = (
                            adaptive_recommendation(analyzed_data)
                            if recommendation_mode == "Adaptive"
                            else generate_recommendations(analyzed_data, symbol)
                        )

                        # Update session state
                        st.session_state.symbol = symbol
                        st.session_state.data = analyzed_data
                        st.session_state.recommendations = recommendations
                        st.session_state.backtest_results_swing = None
                        st.session_state.backtest_results_intraday = None

                        display_dashboard(symbol, analyzed_data, recommendations)
                    else:
                        st.warning("⚠️ No data available for the selected stock.")
                except Exception as e:
                    st.error(f"❌ Error fetching or analyzing data: {str(e)}")
    elif 'data' in st.session_state and 'recommendations' in st.session_state:
        # Display previous results if available
        display_dashboard(
            st.session_state.symbol,
            st.session_state.data,
            st.session_state.recommendations
        )
    else:
        st.info("👈 Select a stock and click 'Analyze Selected Stock' to begin.")

if __name__ == "__main__":
    main()
