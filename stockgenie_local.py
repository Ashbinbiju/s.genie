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
from streamlit import cache_data

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
 "TOP PICKS": [
  "IEX.NS", "MARICO.NS", "NAM-INDIA.NS", "BLUEJET.NS", "BEL.NS", "LLOYDSME.NS", "ITC.NS", "ABSLAMC.NS", "HEXWARE.NS",
  "INDUSTOWER.NS", "TRIL.NS", "VEDL.NS", "HBLPOWER.NS", "WELCORP.NS", "BERGEPAINT.NS", "SUMICHEM.NS", "JYOTICNC.NS",
  "CIPLA.NS", "EIHOTEL.NS", "DRREDDY.NS", "ZENSARTECH.NS", "EMCURE.NS", "UTIAMC.NS", "KPRMILL.NS", "METROBRAND.NS",
  "MAXHEALTH.NS", "LTFOODS.NS", "UNO.NS", "REDINGTON.NS", "INDHOTEL.NS", "EIDPARRY.NS", "INTELLECT.NS", "BPCL.NS",
  "DEEPAKFERT.NS", "MACROTECH.NS", "HINDALCO.NS", "KRISHNAINST.NS", "APTUS.NS", "IPCALAB.NS", "FIRSTSOURCE.NS",
  "ADANIPORTS.NS", "JSWINFRA.NS", "ASHOKLEY.NS", "SAILIFE.NS", "JUBLFOOD.NS", "ASAHIINDIA.NS", "SHYAMMETL.NS",
  "FORTIS.NS", "ICICIPRULI.NS", "AAVAS.NS", "APOLLOTYRE.NS", "CHALET.NS", "MANAPPURAM.NS", "SHRIRAMFIN.NS",
  "JINDALSTEL.NS", "ASTERDM.NS", "GODIGIT.NS", "HPCL.NS", "SBICARD.NS", "CENTURYPLY.NS", "LAURUSLABS.NS",
  "FSN.ECOMMERCE.NS", "JUBILANTPHARMA.NS", "PNBHOUSING.NS", "RELIANCE.NS", "ADANIGREEN.NS", "TATACONSUM.NS",
  "TATASTEEL.NS", "LTFINANCE.NS", "AUBANK.NS", "ACME.NS", "GODREJIND.NS", "JSWSTEEL.NS", "ICICIBANK.NS", "UPL.NS",
  "HDFCLIFE.NS", "KARURVYSYA.NS", "AXISBANK.NS", "FEDERALBNK.NS", "PIRAMAL.NS", "MINDSPACE.NS", "DLF.NS", "SBIN.NS",
  "INDIANB.NS", "BANKBARODA.NS", "BIOCON.NS", "BROOKFIELD.NS", "POONAWALLA.NS",
  "RAMCOCEM.NS", "WOCKPHARMA.NS", "EMBASSY.NS", "ONE97.NS"
],

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
    "QUICKHEAL.NS", "CIGNITITEC.NS","SAGILITY.NS", "ALLSEC.NS"
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
def fetch_stock_data_with_auth(symbol, period="5y", interval="1d"):
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
        if period == "5y":
            start_date = end_date - timedelta(days=5 * 365)
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
def fetch_stock_data_cached(symbol, period="5y", interval="1d"):
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
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None:
        if data['RSI'].iloc[-1] < 30:
            score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None:
        if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
            score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] is not None and data['Ichimoku_Span_A'].iloc[-1] is not None:
        if data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
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

def analyze_stock(data):
    if data.empty or len(data) < 27:
        st.warning("⚠️ Insufficient data to compute indicators.")
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
            if col not in data.columns:
                data[col] = None
        return data

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
        for col in missing_cols:
            data[col] = None
        return data

    try:
        rsi_window = optimize_rsi_window(data)
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute RSI: {str(e)}")
        data['RSI'] = None

    try:
        macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_hist'] = macd.macd_diff()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute MACD: {str(e)}")
        data['MACD'] = None
        data['MACD_signal'] = None
        data['MACD_hist'] = None

    try:
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = None
        data['SMA_200'] = None
        data['EMA_20'] = None
        data['EMA_50'] = None

    try:
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['Upper_Band'] = bollinger.bollinger_hband()
        data['Middle_Band'] = bollinger.bollinger_mavg()
        data['Lower_Band'] = bollinger.bollinger_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = None
        data['Middle_Band'] = None
        data['Lower_Band'] = None

    try:
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['SlowK'] = stoch.stoch()
        data['SlowD'] = stoch.stoch_signal()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = None
        data['SlowD'] = None

    try:
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ATR: {str(e)}")
        data['ATR'] = None

    try:
        if len(data) >= 27:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        st.warning(f"⚠️ Failed to compute ADX: {str(e)}")
        data['ADX'] = None

    try:
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute OBV: {str(e)}")
        data['OBV'] = None

    try:
        data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
    except Exception as e:
        st.warning(f"⚠️ Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None

    try:
        data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
        data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Spike: {str(e)}")
        data['Volume_Spike'] = None

    try:
        data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None

    try:
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        data['Fib_23.6'] = high - diff * 0.236
        data['Fib_38.2'] = high - diff * 0.382
        data['Fib_50.0'] = high - diff * 0.5
        data['Fib_61.8'] = high - diff * 0.618
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = None
        data['Fib_38.2'] = None
        data['Fib_50.0'] = None
        data['Fib_61.8'] = None

    try:
        data['Divergence'] = detect_divergence(data)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"

    try:
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
        data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Chikou'] = data['Close'].shift(-26)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = None
        data['Ichimoku_Kijun'] = None
        data['Ichimoku_Span_A'] = None
        data['Ichimoku_Span_B'] = None
        data['Ichimoku_Chikou'] = None

    try:
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute CMF: {str(e)}")
        data['CMF'] = None

    try:
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
        data['Donchian_Upper'] = donchian.donchian_channel_hband()
        data['Donchian_Lower'] = donchian.donchian_channel_lband()
        data['Donchian_Middle'] = donchian.donchian_channel_mband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = None
        data['Donchian_Lower'] = None
        data['Donchian_Middle'] = None

    try:
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
        data['Keltner_Upper'] = keltner.keltner_channel_hband()
        data['Keltner_Middle'] = keltner.keltner_channel_mband()
        data['Keltner_Lower'] = keltner.keltner_channel_lband()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = None
        data['Keltner_Middle'] = None
        data['Keltner_Lower'] = None

    try:
        data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute TRIX: {str(e)}")
        data['TRIX'] = None

    try:
        data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
            data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
        ).ultimate_oscillator()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = None

    try:
        data['CMO'] = calculate_cmo(data['Close'], window=14)
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = None

    try:
        data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
    except Exception as e:
        st.warning(f"⚠️ Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = None

    return data

def calculate_buy_at(data):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    buy_at = last_close * 0.99 if last_rsi < 30 else last_close
    return round(buy_at, 2)

def calculate_stop_loss(data, atr_multiplier=2.5):
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    atr_multiplier = 3.0 if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else 1.5
    stop_loss = last_close - (atr_multiplier * last_atr)
    if stop_loss < last_close * 0.9:
        stop_loss = last_close * 0.9
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss(data)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = data['Close'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = min(risk_reward_ratio, 5) if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else min(risk_reward_ratio, 3)
    target = last_close + (risk * adjusted_ratio)
    if target > last_close * 1.2:
        target = last_close * 1.2
    return round(target, 2)

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

def fetch_fundamentals(symbol):
    try:
        smart_api = init_smartapi_client()
        if not smart_api:
            return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

def classify_market_regime(data):
    if data.empty:
        return 'unknown'

    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()

    if data['Close'].iloc[-1] > data['SMA_50'].iloc[-1] and data['Close'].iloc[-1] > data['SMA_200'].iloc[-1]:
        trend = 'bullish'
    elif data['Close'].iloc[-1] < data['SMA_50'].iloc[-1] and data['Close'].iloc[-1] < data['SMA_200'].iloc[-1]:
        trend = 'bearish'
    else:
        trend = 'neutral'

    if data['RSI'].iloc[-1] < 30:
        rsi_status = 'oversold'
    elif data['RSI'].iloc[-1] > 70:
        rsi_status = 'overbought'
    else:
        rsi_status = 'neutral'

    if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        macd_status = 'bullish'
    elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
        macd_status = 'bearish'
    else:
        macd_status = 'neutral'

    if data['ADX'].iloc[-1] > 25:
        adx_status = 'strong'
    else:
        adx_status = 'weak'

    if trend == 'bullish' and rsi_status == 'neutral' and macd_status == 'bullish' and adx_status == 'strong':
        return 'strong_bullish'
    elif trend == 'bullish' and rsi_status == 'neutral' and macd_status == 'bullish':
        return 'bullish'
    elif trend == 'bearish' and rsi_status == 'neutral' and macd_status == 'bearish' and adx_status == 'strong':
        return 'strong_bearish'
    elif trend == 'bearish' and rsi_status == 'neutral' and macd_status == 'bearish':
        return 'bearish'
    else:
        return 'neutral'

def compute_signal_score(data):
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

    rsi = data['RSI'].iloc[-1]
    if rsi < 30:
        score += weights['RSI'] * 1
    elif rsi > 70:
        score -= weights['RSI'] * 1

    if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += weights['MACD']
    else:
        score -= weights['MACD']

    close = data['Close'].iloc[-1]
    if close > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
        score += weights['Ichimoku']
    elif close < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
        score -= weights['Ichimoku']

    if data['CMF'].iloc[-1] > 0:
        score += weights['CMF']
    else:
        score -= weights['CMF']

    atr_pct = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
    if atr_pct > 0.04:
        score -= weights['ATR_Volatility']

    if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
        score += weights['Breakout']
    elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
        score -= weights['Breakout']

    return score

def adaptive_recommendation(data, symbol=None):
    if data.empty or len(data) < 27 or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        return {
            "Intraday": "Hold", "Swing": "Hold", "Short-Term": "Hold", "Long-Term": "Hold",
            "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
            "Current Price": None, "Buy At": None, "Stop Loss": None, "Target": None,
            "Score": 0, "Regime": "unknown", "Position Size": 0, "Trailing Stop": None
        }

    regime = classify_market_regime(data)
    score = compute_signal_score(data)
    base_amount = 1000
    position_size = max(0, min(1, score / 7.0)) * base_amount

    if regime == 'strong_bullish':
        threshold = 1.0
    elif regime == 'bullish':
        threshold = 1.5
    elif regime == 'neutral':
        threshold = 2.0
    elif regime == 'bearish':
        threshold = 2.5
    elif regime == 'strong_bearish':
        threshold = 3.0
    else:
        threshold = 2.0

    recommendation = 'Hold'
    if score >= threshold:
        recommendation = 'Buy'
    elif score <= -threshold:
        recommendation = 'Sell'

    highest_close = data['Close'].rolling(22).max().iloc[-1]
    chandelier_stop = highest_close - data['ATR'].iloc[-1] * 3 if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None else None

    buy_score = 0
    sell_score = 0

    # RSI-based signals
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
        if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['RSI'].iloc[-1] <= 20:
                buy_score += 4
            elif data['RSI'].iloc[-1] < 30:
                buy_score += 2
            elif data['RSI'].iloc[-1] > 70:
                sell_score += 2

    # MACD-based signals
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None:
        if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                buy_score += 1
            elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                sell_score += 1

    # Bollinger Bands signals
    if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                sell_score += 1

    # VWAP signals
    if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                sell_score += 1

    # Volume-based signals
    if 'Volume' in data.columns and data['Volume'].iloc[-1] is not None and 'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None:
        volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
        if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
            if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                buy_score += 2
            elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                sell_score += 2
            elif volume_ratio < 0.5:
                sell_score += 1

    # Volume Spike signals
    if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None:
        if data['Volume_Spike'].iloc[-1] and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                buy_score += 1
            else:
                sell_score += 1

    # Divergence signals
    if 'Divergence' in data.columns and data['Divergence'].iloc[-1] is not None:
        if data['Divergence'].iloc[-1] == "Bullish Divergence":
            buy_score += 1
        elif data['Divergence'].iloc[-1] == "Bearish Divergence":
            sell_score += 1

    # Ichimoku Cloud signals
    if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                buy_score += 1
            elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                sell_score += 1

    # CMF signals
    if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None:
        if isinstance(data['CMF'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['CMF'].iloc[-1] > 0:
                buy_score += 1
            elif data['CMF'].iloc[-1] < 0:
                sell_score += 1

    # Donchian Channel signals
    if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Donchian_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                sell_score += 1

    # RSI and Bollinger Bands combined signals
    if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] >= data['Lower_Band'].iloc[-1]:
                buy_score += 2
            elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                sell_score += 2

    # Ichimoku Tenkan/Kijun signals
    if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
                buy_score += 1
            elif data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]:
                sell_score += 1

    # Keltner Channel signals
    if 'Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and data['Close'].iloc[-1] is not None:
        if isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                sell_score += 1

    # TRIX signals
    if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None and len(data['TRIX'].dropna()) >= 2:
        if isinstance(data['TRIX'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['TRIX'].iloc[-2], (int, float, np.integer, np.floating)):
            if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                buy_score += 1
            elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                sell_score += 1

    # Ultimate Oscillator signals
    if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None:
        if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Ultimate_Osc'].iloc[-1] < 30:
                buy_score += 1
            elif data['Ultimate_Osc'].iloc[-1] > 70:
                sell_score += 1

    # CMO signals
    if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None:
        if isinstance(data['CMO'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['CMO'].iloc[-1] < -50:
                buy_score += 1
            elif data['CMO'].iloc[-1] > 50:
                sell_score += 1

    # VPT signals
    if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None and len(data['VPT'].dropna()) >= 2:
        if isinstance(data['VPT'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['VPT'].iloc[-2], (int, float, np.integer, np.floating)):
            if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                buy_score += 1
            elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                sell_score += 1

    # Fibonacci Retracement signals
    if 'Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and data['Close'].iloc[-1] is not None:
        current_price = data['Close'].iloc[-1]
        fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
        for level in fib_levels:
            if isinstance(level, (int, float, np.integer, np.floating)) and abs(current_price - level) / current_price < 0.01:
                if current_price > level:
                    buy_score += 1
                else:
                    sell_score += 1

    # Parabolic SAR signals
    if 'Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
            if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                buy_score += 1
            elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                sell_score += 1

    # Strategy-specific recommendations
    # Intraday: Based on VWAP, RSI, and Volume Spike
    intraday = "Hold"
    if (data['Close'].iloc[-1] > data['VWAP'].iloc[-1] and 
        data['RSI'].iloc[-1] < 70 and 
        data['Volume_Spike'].iloc[-1] and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['VWAP'].iloc[-1], (int, float)) and 
        isinstance(data['RSI'].iloc[-1], (int, float))):
        intraday = "Buy"
    elif (data['Close'].iloc[-1] < data['VWAP'].iloc[-1] or 
          data['RSI'].iloc[-1] > 70 and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['VWAP'].iloc[-1], (int, float)) and 
          isinstance(data['RSI'].iloc[-1], (int, float))):
        intraday = "Sell"

    # Swing: Based on MACD, Bollinger Bands, and Volume
    swing = "Hold"
    if (data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] and 
        data['Close'].iloc[-1] > data['Lower_Band'].iloc[-1] and 
        data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1] and 
        isinstance(data['MACD'].iloc[-1], (int, float)) and 
        isinstance(data['MACD_signal'].iloc[-1], (int, float)) and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['Lower_Band'].iloc[-1], (int, float)) and 
        isinstance(data['Volume'].iloc[-1], (int, float)) and 
        isinstance(data['Avg_Volume'].iloc[-1], (int, float))):
        swing = "Buy"
    elif (data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] or 
          data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1] and 
          isinstance(data['MACD'].iloc[-1], (int, float)) and 
          isinstance(data['MACD_signal'].iloc[-1], (int, float)) and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['Upper_Band'].iloc[-1], (int, float))):
        swing = "Sell"

    # Short-Term: Based on Ichimoku and RSI
    short_term = "Hold"
    if (data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) and 
        data['RSI'].iloc[-1] < 70 and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float)) and 
        isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float)) and 
        isinstance(data['RSI'].iloc[-1], (int, float))):
        short_term = "Buy"
    elif (data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) or 
          data['RSI'].iloc[-1] > 70 and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float)) and 
          isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float)) and 
          isinstance(data['RSI'].iloc[-1], (int, float))):
        short_term = "Sell"

    # Long-Term: Based on SMA_50, SMA_200, and ADX
    long_term = "Hold"
    if (data['Close'].iloc[-1] > data['SMA_50'].iloc[-1] and 
        data['Close'].iloc[-1] > data['SMA_200'].iloc[-1] and 
        data['ADX'].iloc[-1] > 25 and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['SMA_50'].iloc[-1], (int, float)) and 
        isinstance(data['SMA_200'].iloc[-1], (int, float)) and 
        isinstance(data['ADX'].iloc[-1], (int, float))):
        long_term = "Buy"
    elif (data['Close'].iloc[-1] < data['SMA_50'].iloc[-1] or 
          data['Close'].iloc[-1] < data['SMA_200'].iloc[-1] and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['SMA_50'].iloc[-1], (int, float)) and 
          isinstance(data['SMA_200'].iloc[-1], (int, float))):
        long_term = "Sell"

    # Mean Reversion: Based on RSI and Bollinger Bands
    mean_reversion = "Hold"
    if (data['RSI'].iloc[-1] < 30 and 
        data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1] and 
        isinstance(data['RSI'].iloc[-1], (int, float)) and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['Lower_Band'].iloc[-1], (int, float))):
        mean_reversion = "Buy"
    elif (data['RSI'].iloc[-1] > 70 and 
          data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1] and 
          isinstance(data['RSI'].iloc[-1], (int, float)) and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['Upper_Band'].iloc[-1], (int, float))):
        mean_reversion = "Sell"

    # Breakout: Based on Donchian Channels
    breakout = "Hold"
    if (data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1] and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['Donchian_Upper'].iloc[-1], (int, float))):
        breakout = "Buy"
    elif (data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1] and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['Donchian_Lower'].iloc[-1], (int, float))):
        breakout = "Sell"

    # Ichimoku Trend: Based on Tenkan/Kijun crossover and Cloud position
    ichimoku_trend = "Hold"
    if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and 
        data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) and 
        isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float)) and 
        isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float)) and 
        isinstance(data['Close'].iloc[-1], (int, float)) and 
        isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float)) and 
        isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float))):
        ichimoku_trend = "Buy"
    elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and 
          data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]) and 
          isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float)) and 
          isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float)) and 
          isinstance(data['Close'].iloc[-1], (int, float)) and 
          isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float)) and 
          isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float))):
        ichimoku_trend = "Sell"

    # Calculate price levels
    current_price = float(data['Close'].iloc[-1]) if not data.empty else None
    buy_at = calculate_buy_at(data)
    stop_loss = calculate_stop_loss(data)
    target = calculate_target(data)

    return {
        "Intraday": intraday,
        "Swing": swing,
        "Short-Term": short_term,
        "Long-Term": long_term,
        "Mean_Reversion": mean_reversion,
        "Breakout": breakout,
        "Ichimoku_Trend": ichimoku_trend,
        "Current Price": round(current_price, 2) if current_price is not None else None,
        "Buy At": buy_at,
        "Stop Loss": stop_loss,
        "Target": target,
        "Score": round(score, 2),
        "Regime": regime,
        "Position Size": round(position_size, 2),
        "Trailing Stop": round(chandelier_stop, 2) if chandelier_stop is not None else None
    }

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

    # Display stock analysis if symbol is available
    if st.session_state.symbol:
        symbol = st.session_state.symbol

        # Fetch and analyze stock data
        with st.spinner("Loading stock data..."):
            data = fetch_stock_data_with_auth(symbol)
            if not data.empty:
                data = analyze_stock(data)
                recommendations = adaptive_recommendation(data)  # Use adaptive_recommendation
                st.session_state.data = data
                st.session_state.recommendations = recommendations

        # Display recommendations
        if recommendations:
            st.header(f"📋 {symbol.split('-')[0]} Analysis")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_price = recommendations['Current Price'] if recommendations['Current Price'] is not None else "N/A"
                st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
            with col2:
                buy_at = recommendations['Buy At'] if recommendations['Buy At'] is not None else "N/A"
                st.metric("Buy At", f"₹{buy_at}")
            with col3:
                stop_loss = recommendations['Stop Loss'] if recommendations['Stop Loss'] is not None else "N/A"
                st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
            with col4:
                target = recommendations['Target'] if recommendations['Target'] is not None else "N/A"
                st.metric("Target", f"₹{target}")

            st.subheader("📈 Trading Recommendations")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Intraday**: {colored_recommendation(recommendations['Intraday'])}")
                st.write(f"**Swing**: {colored_recommendation(recommendations['Swing'])}")
            with col2:
                st.write(f"**Short-Term**: {colored_recommendation(recommendations['Short-Term'])}")
                st.write(f"**Long-Term**: {colored_recommendation(recommendations['Long-Term'])}")
            with col3:
                st.write(f"**Mean Reversion**: {colored_recommendation(recommendations['Mean_Reversion'])}")
                st.write(f"**Breakout**: {colored_recommendation(recommendations['Breakout'])}")
            with col4:
                st.write(f"**Ichimoku Trend**: {colored_recommendation(recommendations['Ichimoku_Trend'])}")
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations['Score']}/7")

            # Additional analysis and visualization can be added here
            # For example, you can add a price chart, RSI chart, etc.

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

            # Monte Carlo Simulation
            st.subheader("📊 Monte Carlo Simulation")
            simulations = monte_carlo_simulation(data)
            sim_df = pd.DataFrame(simulations).T
            sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
            fig_sim = px.line(sim_df, title="Monte Carlo Price Projections (30 Days)")
            st.plotly_chart(fig_sim, use_container_width=True)

            # Backtest form to isolate button actions
            with st.form(key="backtest_form"):
                col1, col2 = st.columns(2)
                with col1:
                    swing_button = st.form_submit_button("🔍 Backtest Swing Strategy")
                with col2:
                    intraday_button = st.form_submit_button("🔍 Backtest Intraday Strategy")
                
                if swing_button or intraday_button:
                    strategy = "Swing" if swing_button else "Intraday"
                    with st.spinner(f"Running {strategy} Strategy backtest..."):
                        # Simplified hash for caching
                        data_hash = hash(data.to_string())
                        backtest_results = backtest_stock(data, symbol, strategy=strategy, _data_hash=data_hash)
                        if strategy == "Swing":
                            st.session_state.backtest_results_swing = backtest_results
                        else:
                            st.session_state.backtest_results_intraday = backtest_results

            # Display backtest results if available
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

                    # Visualize Buy/Sell Signals
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


def main():
    init_database()
    st.sidebar.title("🔍 Stock Selection")
    stock_list = fetch_nse_stock_list()

    if 'symbol' not in st.session_state:
        st.session_state.symbol = stock_list[0]

    symbol = st.sidebar.selectbox(
        "Select Stock",
        stock_list,
        key="stock_select",
        index=stock_list.index(st.session_state.symbol) if st.session_state.symbol in stock_list else 0
    )

    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner("Loading stock data..."):
                data = fetch_stock_data_with_auth(symbol)
                if not data.empty:
                    data = analyze_stock(data)
                    recommendations = generate_recommendations(data, symbol)
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