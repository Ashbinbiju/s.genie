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

def compute_signal_score(data):
    """Computes weighted score based on indicator strength"""
    score = 0.0
    weights = {
        'RSI': 1.5,
        'MACD': 1.2,
        'Ichimoku': 1.5,
        'CMF': 0.5,
        'ATR_Volatility': 1.0,
        'Breakout': 1.2,
    }
    
    # Skip low volume days
    if data['Volume'].iloc[-1] < data['Avg_Volume'].iloc[-1] * 0.5:
        return -10  # force a "no trade" scenario

    # RSI
    rsi = data['RSI'].iloc[-1]
    if rsi < 30:
        score += weights['RSI'] * 1
    elif rsi > 70:
        score -= weights['RSI'] * 1

    # MACD
    if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += weights['MACD']
    else:
        score -= weights['MACD']

    # Ichimoku Trend
    close = data['Close'].iloc[-1]
    if close > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
        score += weights['Ichimoku']
    elif close < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
        score -= weights['Ichimoku']

    # CMF (money flow)
    if data['CMF'].iloc[-1] > 0:
        score += weights['CMF']
    else:
        score -= weights['CMF']

    # ATR Volatility
    atr_pct = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
    if atr_pct > 0.04:
        score -= weights['ATR_Volatility']

    # Donchian Breakout
    if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
        score += weights['Breakout']
    elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
        score -= weights['Breakout']

    return score

def adaptive_recommendation(data):
    regime = classify_market_regime(data)
    score = compute_signal_score(data)

    # Position sizing
    base_amount = 1000  # example base capital per trade
    position_size = max(0, min(1, score / 7.0)) * base_amount

    if regime == 'volatile':
        threshold = 2.0
    elif regime == 'bullish':
        threshold = 1.0
    else:
        threshold = 1.5

    recommendation = 'Hold'
    if score >= threshold:
        recommendation = 'Buy'
    elif score <= -threshold:
        recommendation = 'Sell'

    # Trailing Stop Loss using Chandelier Exit
    highest_close = data['Close'].rolling(22).max().iloc[-1]
    chandelier_stop = highest_close - data['ATR'].iloc[-1] * 3

    return {
        'Regime': regime,
        'Score': round(score, 2),
        'Recommendation': recommendation,
        'Position Size': round(position_size, 2),
        'Trailing Stop': round(chandelier_stop, 2)
    }


def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }

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
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None):
    if data.empty or len(data) < 200:
        st.warning(f"⚠️ Insufficient data for backtesting {symbol}.")
        return None

    data = data.copy()
    data = analyze_stock(data)

    key_cols = ['Open', 'Close', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'TRIX', 'VPT', 'OBV']
    for col in key_cols:
        if col not in data.columns:
            data[col] = None
    data = data.dropna(subset=['Open', 'Close'])

    if len(data) < 27:
        st.warning(f"⚠️ Not enough valid data for backtesting {symbol}.")
        return None

    initial_capital = 25000
    capital = initial_capital
    position = 0
    trades = []
    buy_signals = []
    sell_signals = []
    slippage = 0.002  # 0.2% slippage per trade
    commission = 0.001  # 0.1% commission per trade

    min_rows = 200
    for i in range(min_rows, len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        sliced_data = data.iloc[:i+1]

        if len(sliced_data['RSI'].dropna()) < 1 or len(sliced_data['MACD'].dropna()) < 1:
            continue

        recommendations = generate_recommendations(sliced_data, symbol=symbol)

        if recommendations[strategy] in ["Buy", "Strong Buy"]:
            buy_signals.append((next_row.name, next_row['Open']))
        elif recommendations[strategy] in ["Sell", "Strong Sell"]:
            sell_signals.append((next_row.name, next_row['Open']))

        if position == 0 and recommendations[strategy] in ["Buy", "Strong Buy"]:
            entry_price = next_row['Open'] * (1 + slippage + commission)
            shares = capital // entry_price
            if shares > 0:
                position = shares
                capital -= shares * entry_price
                trades.append({
                    "entry_date": next_row.name,
                    "entry_price": entry_price,
                    "shares": shares
                })

        elif position > 0 and recommendations[strategy] in ["Sell", "Strong Sell"]:
            exit_price = next_row['Open'] * (1 - slippage - commission)
            capital += position * exit_price
            trades[-1]["exit_date"] = next_row.name
            trades[-1]["exit_price"] = exit_price
            trades[-1]["profit"] = (exit_price - trades[-1]["entry_price"]) * position
            position = 0

    if position > 0:
        exit_price = data['Close'].iloc[-1] * (1 - slippage - commission)
        capital += position * exit_price
        trades[-1]["exit_date"] = data.index[-1]
        trades[-1]["exit_price"] = exit_price
        trades[-1]["profit"] = (exit_price - trades[-1]["entry_price"]) * position

    final_value = capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    wins = sum(1 for trade in trades if trade.get("profit", 0) > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0

    annual_return = ((final_value / initial_capital) ** (252 / len(data))) - 1 if len(data) > 0 else 0
    returns = pd.Series([trade["profit"] / initial_capital for trade in trades if "profit" in trade])
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    max_drawdown = 0
    peak = initial_capital
    for trade in trades:
        if "profit" in trade:
            capital += trade["profit"]
            peak = max(peak, capital)
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "trade_details": trades,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals
    }
    
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
                intraday, swing, short_term, long_term, mean_reversion, breakout, ichimoku_trend, pick_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'), row['Symbol'], row['Score'], row['Current Price'],
            row['Buy At'], row['Stop Loss'], row['Target'], row['Intraday'], row['Swing'],
            row['Short-Term'], row['Long-Term'], row['Mean_Reversion'], row['Breakout'], row['Ichimoku_Trend'], pick_type
        ))
    conn.commit()
    conn.close()

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
                    results.append(result)
            except Exception as e:
                errors.append(f"⚠️ Error processing stock {symbol}: {str(e)}")
    for error in errors:
        st.warning(error)
    return results

def analyze_stock_parallel(symbol):
    data = fetch_stock_data_cached(symbol)
    if not data.empty:
        data = analyze_stock(data)
        recommendations = generate_recommendations(data, symbol)
        return {
            "Symbol": symbol,
            "Current Price": recommendations["Current Price"],
            "Buy At": recommendations["Buy At"],
            "Stop Loss": recommendations["Stop Loss"],
            "Target": recommendations["Target"],
            "Intraday": recommendations["Intraday"],
            "Swing": recommendations["Swing"],
            "Short-Term": recommendations["Short-Term"],
            "Long-Term": recommendations["Long-Term"],
            "Mean_Reversion": recommendations["Mean_Reversion"],
            "Breakout": recommendations["Breakout"],
            "Ichimoku_Trend": recommendations["Ichimoku_Trend"],
            "Score": recommendations.get("Score", 0),
        }
    return None

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        time.sleep(3)
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    return results_df.sort_values(by="Score", ascending=False).head(5)

def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        time.sleep(3)
    
    results_df = pd.DataFrame([r for r in results if r is not None])
    if results_df.empty:
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
    intraday_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    return intraday_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    elif "Hold" in recommendation:
        return f"🟡 {recommendation}"
    else:
        return recommendation

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")

def display_dashboard(symbol=None, data=None, recommendations=None):
    # Initialize session state
    if 'selected_sectors' not in st.session_state:
        st.session_state.selected_sectors = ["Bank"]
    if 'symbol' not in st.session_state:
        st.session_state.symbol = symbol

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
            top_sectors = get_top_sectors_cached(rate_limit_delay=2, stocks_per_sector=2)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            for name, score in top_sectors:
                st.markdown(f"- **{name}**: {score:.2f}/7")

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
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    Swing: {colored_recommendation(row['Swing'])}  
                    Short-Term: {colored_recommendation(row['Short-Term'])}  
                    Long-Term: {colored_recommendation(row['Long-Term'])}  
                    Mean Reversion: {colored_recommendation(row['Mean_Reversion'])}  
                    Breakout: {colored_recommendation(row['Breakout'])}  
                    Ichimoku Trend: {colored_recommendation(row['Ichimoku_Trend'])}
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No top picks available due to data issues.")

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
                    current_price = row['Current Price'] if pd.notnull(row['Current Price']) else "N/A"
                    buy_at = row['Buy At'] if pd.notnull(row['Buy At']) else "N/A"
                    stop_loss = row['Stop Loss'] if pd.notnull(row['Stop Loss']) else "N/A"
                    target = row['Target'] if pd.notnull(row['Target']) else "N/A"
                    st.markdown(f"""
                    {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                    Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                    Target: ₹{target}  
                    Intraday: {colored_recommendation(row['Intraday'])}  
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No intraday picks available due to data issues.")

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

    if symbol and data is not None and recommendations is not None:
        st.session_state.symbol = symbol
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

        # Backtest Swing Strategy
        if st.button("🔍 Backtest Swing Strategy", key="backtest_swing"):
            with st.spinner("Running Swing Strategy backtest..."):
                # Compute a hash for the data to ensure caching works
                data_hash = hash(data.to_string())
                backtest_results = backtest_stock(data, symbol, strategy="Swing", _data_hash=data_hash)
                if backtest_results:
                    st.subheader("📈 Backtest Results (Swing Strategy)")
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
                        fig.add_scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals', marker=dict(color='green', symbol='triangle-up', size=10))
                    if backtest_results["sell_signals"]:
                        sell_dates, sell_prices = zip(*backtest_results["sell_signals"])
                        fig.add_scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals', marker=dict(color='red', symbol='triangle-down', size=10))
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Insufficient data for backtesting.")

        # Backtest Intraday Strategy
        if st.button("🔍 Backtest Intraday Strategy", key="backtest_intraday"):
            with st.spinner("Running Intraday Strategy backtest..."):
                data_hash = hash(data.to_string())
                backtest_results = backtest_stock(data, symbol, strategy="Intraday", _data_hash=data_hash)
                if backtest_results:
                    st.subheader("📈 Backtest Results (Intraday Strategy)")
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
                        fig.add_scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals', marker=dict(color='green', symbol='triangle-up', size=10))
                    if backtest_results["sell_signals"]:
                        sell_dates, sell_prices = zip(*backtest_results["sell_signals"])
                        fig.add_scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals', marker=dict(color='red', symbol='triangle-down', size=10))
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Insufficient data for backtesting.")

        st.subheader("📊 Technical Indicators")
        indicators = [
            ("RSI", data['RSI'].iloc[-1], TOOLTIPS['RSI']),
            ("MACD", data['MACD'].iloc[-1], TOOLTIPS['MACD']),
            ("ATR", data['ATR'].iloc[-1], TOOLTIPS['ATR']),
            ("ADX", data['ADX'].iloc[-1], TOOLTIPS['ADX']),
            ("Bollinger Upper", data['Upper_Band'].iloc[-1], TOOLTIPS['Bollinger']),
            ("Bollinger Lower", data['Lower_Band'].iloc[-1], TOOLTIPS['Bollinger']),
            ("VWAP", data['VWAP'].iloc[-1], TOOLTIPS['VWAP']),
            ("Parabolic SAR", data['Parabolic_SAR'].iloc[-1], TOOLTIPS['Parabolic_SAR']),
            ("Fib 61.8%", data['Fib_61.8'].iloc[-1], TOOLTIPS['Fib_Retracements']),
            ("Ichimoku Span A", data['Ichimoku_Span_A'].iloc[-1], TOOLTIPS['Ichimoku']),
            ("CMF", data['CMF'].iloc[-1], TOOLTIPS['CMF']),
            ("Donchian Upper", data['Donchian_Upper'].iloc[-1], TOOLTIPS['Donchian']),
            ("Keltner Upper", data['Keltner_Upper'].iloc[-1], TOOLTIPS['Keltner']),
            ("TRIX", data['TRIX'].iloc[-1], TOOLTIPS['TRIX']),
            ("Ultimate Oscillator", data['Ultimate_Osc'].iloc[-1], TOOLTIPS['Ultimate_Osc']),
            ("CMO", data['CMO'].iloc[-1], TOOLTIPS['CMO']),
            ("VPT", data['VPT'].iloc[-1], TOOLTIPS['VPT']),
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
        st.plotly_chart(fig)

        st.subheader("📊 Monte Carlo Simulation")
        simulations = monte_carlo_simulation(data)
        sim_df = pd.DataFrame(simulations).T
        sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
        fig_sim = px.line(sim_df, title="Monte Carlo Price Projections (30 Days)")
        st.plotly_chart(fig_sim)

        st.subheader("📊 RSI and MACD")
        fig_ind = px.line(data, x=data.index, y='RSI', title="RSI")
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red")
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_ind)

        fig_macd = px.line(data, x=data.index, y=['MACD', 'MACD_signal'], title="MACD")
        st.plotly_chart(fig_macd)

        st.subheader("📊 Volume Analysis")
        fig_vol = px.bar(data, x=data.index, y='Volume', title="Volume")
        if 'Volume_Spike' in data.columns:
            spike_data = data[data['Volume_Spike'] == True]
            if not spike_data.empty:
                fig_vol.add_scatter(x=spike_data.index, y=spike_data['Volume'], mode='markers', name='Volume Spike', marker=dict(color='red', size=10))
        st.plotly_chart(fig_vol)
        
def main():
    init_database()
    st.sidebar.title("🔍 Stock Selection")
    stock_list = fetch_nse_stock_list()
    symbol = st.sidebar.selectbox("Select Stock", stock_list, key="stock_select", index=stock_list.index(st.session_state.get('symbol', stock_list[0])) if st.session_state.get('symbol') in stock_list else 0)

    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            data = fetch_stock_data_with_auth(symbol)
            if not data.empty:
                data = analyze_stock(data)
                recommendations = generate_recommendations(data, symbol)
                display_dashboard(symbol, data, recommendations)
            else:
                st.warning("⚠️ No data available for the selected stock.")
    else:
        display_dashboard()

if __name__ == "__main__":
    main()
