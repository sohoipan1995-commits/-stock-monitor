import os
import time
import warnings
import threading
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

try:
    from futu import OpenQuoteContext, RET_OK, KLType
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# 基本設定與資料庫初始化
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 撈底監察系統 Pro+", page_icon="📈", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

AUTO_REFRESH_SEC = 1800
OHLCV_TTL = 1800
INFO_TTL = 1800
DB_FILE = "signals.db"  # 改用 SQLite 資料庫持久化
MAX_WORKERS_DATA = 6
MAX_WORKERS_SCORE = 4
FUTU_LOCK = threading.RLock()

# 初始化 SQLite 資料庫
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals
                 (date TEXT, ticker TEXT, total_score REAL, label TEXT, price REAL)''')
    conn.commit()
    conn.close()

init_db()

HK_WATCHLIST = [
    "0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK",
    "0066.HK", "0003.HK", "0002.HK", "0016.HK", "0883.HK", "2318.HK",
    "1299.HK", "0001.HK", "9988.HK", "0175.HK", "0027.HK", "2628.HK",
    "0011.HK", "0688.HK", "3690.HK", "9618.HK", "0981.HK", "9999.HK",
    "2382.HK", "0291.HK", "1211.HK", "0267.HK", "2688.HK", "0762.HK",
    "6862.HK", "0960.HK", "2020.HK", "1810.HK", "1024.HK"
]

US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "ORCL", "ASML", "AMD", "QCOM", "INTC", "AMAT", "LRCX", "MU",
    "SNDK", "SKHY", "JPM", "BAC", "GS", "MS", "BRK-B", "COST", "WMT",
    "HD", "JNJ", "UNH", "PFE", "XOM", "NEE", "UBER", "LITE", "CLX",
    "SPY", "QQQ", "SOXL", "IWM", "NFLX", "SPCX"
]

MACRO_TICKERS = {
    "VIX": "^VIX", "VVIX": "^VVIX", "SPX": "^GSPC", "HSI": "^HSI",
    "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VHSI": "^VHSI", "HYG": "HYG",
    "USDHKD": "USDHKD=X"
}

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
GANN_LEVELS = [0.125, 0.25, 0.333, 0.375, 0.500, 0.625, 0.666, 0.75, 0.875] # 江恩八分與三分法
DROP_LEVELS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]

SECTOR_MAP_FALLBACK = {
    "0005.HK": "HK_BANK", "0011.HK": "HK_BANK", "0939.HK": "HK_BANK",
    "1398.HK": "HK_BANK", "3988.HK": "HK_BANK", "2388.HK": "HK_BANK",
    "0002.HK": "HK_UTIL", "0003.HK": "HK_UTIL", "0006.HK": "HK_UTIL",
    "0016.HK": "HK_PROPERTY", "0688.HK": "HK_PROPERTY", "0012.HK": "HK_PROPERTY",
    "0001.HK": "HK_PROPERTY", "0083.HK": "HK_PROPERTY",
    "0700.HK": "HK_TECH", "9988.HK": "HK_TECH", "3690.HK": "HK_TECH",
    "9618.HK": "HK_TECH", "9999.HK": "HK_TECH", "1810.HK": "HK_TECH"
}

# ─────────────────────────────────────────────────────────────
# UI 樣式與刷新
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0d1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {color:#e6edf3!important;}
.metric-card {background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;margin:4px;}
.volume-alert {background:#1c2c1a;border:2px solid #3fb950;border-radius:10px;padding:16px;margin:8px 0;color:#e6edf3;}
.signal-badge {display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.85em;font-weight:bold;}
.badge-buy {background:#0d2818;color:#3fb950;border:1px solid #3fb950;}
.badge-watch {background:#1c2c1a;color:#d29922;border:1px solid #d29922;}
.badge-observe {background:#161b22;color:#8b949e;border:1px solid #8b949e;}
.badge-none {background:#161b22;color:#6e7681;border:1px solid #30363d;}
.resonance-strong {color:#3fb950;font-weight:bold;}
.resonance-medium {color:#d29922;font-weight:bold;}
.resonance-weak {color:#8b949e;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "error_log" not in st.session_state:
    st.session_state.error_log = []

def add_error(context, ticker=None, exc=None):
    message = f"{context}"
    if ticker: message += f"｜{ticker}"
    if exc: message += f"｜{type(exc).__name__}: {str(exc)[:160]}"
    entry = {"時間": datetime.now().strftime("%H:%M:%S"), "訊息": message}
    logs = st.session_state.get("error_log", [])
    if not any(x["訊息"] == message for x in logs):
        logs.append(entry)
    st.session_state.error_log = logs[-100:]

def clear_errors():
    st.session_state.error_log = []

elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, AUTO_REFRESH_SEC - int(elapsed))
mins, secs = divmod(remaining, 60)
with st.sidebar:
    st.markdown(f"🔄 自動刷新：**{mins:02d}:{secs:02d}**")
if st.button("🔄 立即刷新"):
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    clear_errors()
    st.rerun()
if elapsed >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    clear_errors()
    st.rerun()

# ─────────────────────────────────────────────────────────────
# 數據來源
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_futu():
    if not FUTU_AVAILABLE: return None
    try:
        with FUTU_LOCK:
            ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
            ret, _ = ctx.get_market_snapshot(["HK.00700"])
        return ctx if ret == RET_OK else None
    except Exception as exc:
        add_error("富途初始化失敗", exc=exc)
        return None

quote_ctx = init_futu()

def to_futu(ticker):
    if ticker.endswith(".HK"): return f"HK.{ticker[:-3]}"
    if ticker.replace("-", "").isalpha(): return f"US.{ticker}"
    return ticker

@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    if quote_ctx:
        try:
            with FUTU_LOCK:
                ret, data, _ = quote_ctx.request_history_kline(
                    to_futu(ticker), start=None, end=None,
                    ktype=KLType.K_DAY, max_count=500, extended_time=False
                )
            if ret == RET_OK and data is not None and not data.empty:
                df = data[["time_key", "open", "high", "low", "close", "volume"]].copy()
                df.rename(columns={"time_key": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.columns = [c.lower() for c in df.columns]
                return df.dropna()
        except Exception as exc:
            add_error("富途OHLCV失敗，改用Yahoo", ticker, exc)

    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns): return None
        return df.dropna()
    except Exception as exc:
        add_error("Yahoo OHLCV下載失敗", ticker, exc)
        return None

def fetch_multiple(tickers, period="2y", max_workers=MAX_WORKERS_DATA):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ohlcv, tk, period): tk for tk in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try: results[ticker] = future.result()
            except Exception as exc:
                add_error("平行OHLCV工作失敗", ticker, exc)
                results[ticker] = None
    return results

@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    result = {
        "ticker": ticker, "name": ticker, "pe": None, "pb": None,
        "div_yield": None, "roe": None, "de_ratio": None, "rev_growth": None,
        "eps": None, "book_value": None, "sector": None, "industry": None,
        "hist_5y": None, "info_source": "Yahoo"
    }

    if quote_ctx:
        try:
            with FUTU_LOCK:
                ret, data = quote_ctx.get_market_snapshot([to_futu(ticker)])
            if ret == RET_OK and data is not None and not data.empty:
                row = data.iloc[0]
                pe, pb = row.get("pe_ratio"), row.get("pb_ratio")
                if pd.notna(pe):
                    result["pe"], result["info_source"] = float(pe), "富途+Yahoo"
                if pd.notna(pb):
                    result["pb"], result["info_source"] = float(pb), "富途+Yahoo"
                result["name"] = row.get("stock_name", ticker) or ticker
        except Exception as exc:
            add_error("富途基本面快照失敗", ticker, exc)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if result["name"] == ticker:
            result["name"] = info.get("shortName") or info.get("longName") or ticker
        if result["pe"] is None:
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe is None:
                eps, price = info.get("trailingEps"), info.get("currentPrice")
                if eps and price and eps > 0: pe = price / eps
            result["pe"] = float(pe) if pe is not None else None
        if result["pb"] is None:
            pb = info.get("priceToBook")
            result["pb"] = float(pb) if pb is not None else None
        for field, key in [
            ("div_yield", "dividendYield"), ("roe", "returnOnEquity"),
            ("de_ratio", "debtToEquity"), ("rev_growth", "revenueGrowth"),
            ("eps", "trailingEps"), ("book_value", "bookValue"),
            ("sector", "sector"), ("industry", "industry")
        ]:
            result[field] = info.get(key)
        hist = stock.history(period="5y")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            result["hist_5y"] = hist["Close"].dropna()
    except Exception as exc:
        add_error("Yahoo 基本面/歷史數據失敗", ticker, exc)

    return result

def get_sector_from_info(ticker, info=None):
    info = info or get_full_stock_info(ticker)
    return str(info.get("sector")) if info.get("sector") else SECTOR_MAP_FALLBACK.get(ticker, "OTHER")

def get_futu_capital_flow(ticker):
    if not quote_ctx: return None
    try:
        with FUTU_LOCK:
            ret, data = quote_ctx.get_capital_flow(to_futu(ticker))
        if ret == RET_OK and data is not None and not data.empty:
            return data
    except Exception as exc:
        add_error("富途資金流抓取失敗", ticker, exc)
    return None

# ─────────────────────────────────────────────────────────────
# 指標與平滑評分
# ─────────────────────────────────────────────────────────────
def safe_float(value, default=np.nan):
    try:
        return default if value is None or pd.isna(value) else float(value)
    except Exception: return default

def clip_score(value): return float(np.clip(value, 0, 100))

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_kdj(df, n=9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return k, d, 3 * k - 2 * d

def calc_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def calc_obv(df): return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

def calc_wr(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)

def calc_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

def calc_cmf(df, period=20):
    multiplier = ((2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"]).replace(0, np.nan))
    mfv = multiplier * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)

def calc_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

def calc_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_52w_high(df):
    return float(df["high"].iloc[-252:].max()) if len(df) >= 252 else float(df["high"].max())

def fib_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {f"{int(f * 100)}%": round(swing_high - diff * f, 3) for f in FIB_LEVELS}

def gann_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {f"{f * 100:.1f}%": round(swing_high - diff * f, 3) for f in GANN_LEVELS}

def drop_levels(high_price):
    return {f"-{int(d * 100)}%": round(high_price * (1 - d), 3) for d in DROP_LEVELS}

def volume_zscore(df, period=20):
    vol = df["volume"]
    mean = safe_float(vol.rolling(period).mean().iloc[-1], 0)
    std = safe_float(vol.rolling(period).std().iloc[-1], 0)
    return (safe_float(vol.iloc[-1], 0) - mean) / std if std > 0 else 0.0

def smooth_low_score(value, low, high, max_score):
    if pd.isna(value): return 0.0
    if value <= low: return float(max_score)
    if value >= high: return 0.0
    return float(max_score * (high - value) / (high - low))

def smooth_high_score(value, low, high, max_score):
    if pd.isna(value): return 0.0
    if value <= low: return 0.0
    if value >= high: return float(max_score)
    return float(max_score * (value - low) / (high - low))

def time_decay_oversold(indicator_series, threshold, days_back=5):
    weight_sum = 0.0
    for i in range(min(days_back, len(indicator_series))):
        val = safe_float(indicator_series.iloc[-1 - i], np.nan)
        if np.isnan(val): continue
        if val < threshold: weight_sum += (1 - i / (days_back + 1))
    return weight_sum

# ─────────────────────────────────────────────────────────────
# 技術形態：雙底與MACD底背離
# ─────────────────────────────────────────────────────────────
def local_minima_indices(series, order=3):
    values = np.asarray(series, dtype=float)
    indices = []
    for i in range(order, len(values) - order):
        window = values[i - order:i + order + 1]
        if np.isfinite(values[i]) and values[i] == np.nanmin(window):
            if values[i] < values[i - 1] or values[i] < values[i + 1]:
                indices.append(i)
    return indices

def detect_double_bottom(df, lookback=80, min_gap=10, max_gap=55, tolerance=0.05, min_rebound=0.08):
    if df is None or len(df) < lookback: return False
    recent = df.iloc[-lookback:]
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, order=3)
    if len(candidates) < 2: return False

    for first_i in candidates:
        for second_i in candidates:
            gap = second_i - first_i
            if gap < min_gap or gap > max_gap: continue
            low1, low2 = safe_float(lows.iloc[first_i]), safe_float(lows.iloc[second_i])
            if low1 <= 0 or low2 <= 0: continue
            if abs(low2 - low1) / min(low1, low2) > tolerance: continue
            middle_high = safe_float(recent["high"].iloc[first_i:second_i + 1].max(), np.nan)
            rebound = (middle_high - max(low1, low2)) / max(low1, low2) if middle_high else 0
            if rebound >= min_rebound: return True
    return False

def detect_macd_bullish_divergence(df, lookback=90, min_gap=10, max_gap=60):
    if df is None or len(df) < lookback: return False
    recent = df.iloc[-lookback:].copy()
    _, _, hist = calc_macd(recent["close"])
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, order=3)
    if len(candidates) < 2: return False

    for first_i in candidates:
        for second_i in candidates:
            gap = second_i - first_i
            if gap < min_gap or gap > max_gap: continue
            first_low, second_low = safe_float(lows.iloc[first_i]), safe_float(lows.iloc[second_i])
            first_hist, second_hist = safe_float(hist.iloc[first_i]), safe_float(hist.iloc[second_i])
            if np.isnan(first_hist) or np.isnan(second_hist): continue
            if second_low < first_low * 0.99 and second_hist > first_hist: return True
    return False

# ─────────────────────────────────────────────────────────────
# 評分與估值模型 (核心邏輯升級)
# ─────────────────────────────────────────────────────────────
def score_stock(df, market_state="neutral", macro_10d_ret=0.0):
    if df is None or len(df) < 60:
        return 0, 0, [], 0, "無", 0, 0, 0

    close = df["close"]
    volume = df["volume"]
    rsi_d = calc_rsi(close, 14)
    rsi_w = calc_rsi(close, 70)
    k, d, _ = calc_kdj(df)
    macd, sig, hist = calc_macd(close)
    cci = calc_cci(df)
    obv = calc_obv(df)
    wr = calc_wr(df)
    mfi = calc_mfi(df)
    cmf = calc_cmf(df)
    vwap = calc_vwap(df)
    sma20 = close.rolling(20).mean()
    sma200 = close.rolling(200).mean()

    # 指標最新值提取
    rsi_val = safe_float(rsi_d.iloc[-1], 50)
    rsi_w_val = safe_float(rsi_w.iloc[-1], 50)
    k_val, d_val = safe_float(k.iloc[-1], 50), safe_float(d.iloc[-1], 50)
    cci_val = safe_float(cci.iloc[-1], 0)
    wr_val = safe_float(wr.iloc[-1], -50)
    macd_val, sig_val = safe_float(macd.iloc[-1], 0), safe_float(sig.iloc[-1], 0)
    cmf_val = safe_float(cmf.iloc[-1], 0)
    vwap_val = safe_float(vwap.iloc[-1], safe_float(close.iloc[-1], 0))
    close_v = safe_float(close.iloc[-1], 0)
    obv_now = safe_float(obv.iloc[-1], 0)
    obv_prev = safe_float(obv.iloc[-6], obv_now) if len(obv) >= 6 else obv_now
    vol_z = volume_zscore(df)

    # 1. 短線平滑技術分數 (滿分 80)
    rsi_score = smooth_low_score(rsi_val, 20, 45, 24)
    kdj_score = smooth_low_score((k_val + d_val) / 2, 10, 40, 22)
    cci_score = smooth_low_score(cci_val, -200, -40, 18)
    wr_score = smooth_low_score(wr_val, -95, -50, 16)

    decay_rsi = time_decay_oversold(rsi_d, 30, 5)
    decay_kdj = time_decay_oversold(k, 20, 5)
    decay_cci = time_decay_oversold(cci, -100, 5)
    decay_wr = time_decay_oversold(wr, -85, 5)

    triggers = []
    if rsi_val < 30 or decay_rsi > 1.5: triggers.append("RSI")
    if (k_val < 20 and d_val < 20) or decay_kdj > 1.5: triggers.append("KDJ")
    if cci_val < -100 or decay_cci > 1.5: triggers.append("CCI")
    if wr_val < -85 or decay_wr > 1.5: triggers.append("W%R")

    oversold_count = len(triggers)
    if oversold_count >= 3: resonance, mult = "強", 1.25
    elif oversold_count == 2: resonance, mult = "中", 1.10
    elif oversold_count == 1: resonance, mult = "弱", 1.00
    else: resonance, mult = "無", 0.85

    # 價量確認
    vol_confirm = 1.0
    open_v = safe_float(df["open"].iloc[-1], close_v)
    sma20_v = safe_float(sma20.iloc[-1], close_v)
    if vol_z > 2.0 and close_v > vwap_val: vol_confirm = 1.25
    elif vol_z > 2.0 and close_v > open_v: vol_confirm = 1.15
    elif vol_z < -1.5 and close_v < sma20_v: vol_confirm = 0.75

    short_score = (rsi_score + kdj_score + cci_score + wr_score) * mult * vol_confirm
    signals = []

    # MFI與W%R黃金底背離 (主力吸籌)
    if len(wr) >= 5 and len(mfi) >= 5:
        if all(w < -90 for w in wr.iloc[-5:]):
            if close.iloc[-1] <= close.iloc[-5] and mfi.iloc[-1] > mfi.iloc[-5] + 5:
                short_score += 15
                signals.append("💎W%R鈍化+MFI底背離(吸籌)")

    # 逆市相對強弱 (RS)
    if macro_10d_ret < -3.0 and len(close) >= 10:
        stock_10d_ret = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        if stock_10d_ret > 0:
            short_score += 15
            signals.append("💪逆市抗跌(強RS)")
        elif macro_10d_ret < -5.0 and stock_10d_ret > -2.0:
            short_score += 8
            signals.append("💪相對大盤強勢")

    # 型態與技術加分
    if macd_val > sig_val and macd_val < 0:
        short_score += 8 * vol_confirm
        signals.append("MACD低位金叉")
    if obv_now > obv_prev and close_v <= safe_float(close.iloc[-6], close_v):
        short_score += 8 * vol_confirm
        signals.append("OBV底背離")
    if cmf_val > 0.10:
        short_score += 5
        signals.append("💰CMF吸籌")
    if detect_double_bottom(df):
        short_score += 10
        signals.append("🕳️確認雙底")
    if detect_macd_bullish_divergence(df):
        short_score += 12
        signals.append("📉MACD底背離(確認)")

    # 江恩時價共振加分
    high_52_val = get_52w_high(df)
    if high_52_val > 0 and len(df) >= 30:
        high_52_idx = df["high"].iloc[-252:].idxmax() if len(df) >= 252 else df["high"].idxmax()
        days_from_high = (df.index[-1] - high_52_idx).days
        is_gann_time = any(abs(days_from_high - g) <= 3 for g in [49, 90, 144, 233])
        drop_pct = (close_v - high_52_val) / high_52_val
        is_gann_price = any(abs(drop_pct - p) <= 0.02 for p in [-0.333, -0.5, -0.666])
        if is_gann_time and is_gann_price:
            short_score += 15
            signals.append("⏳江恩時價共振(極強)")

    # 2. 中線評分 (滿分 100：RSI 30 + 乖離 35 + CCI 15 + 周MACD 20)
    mid_score = 0.0
    mid_signals = []
    
    bias200 = 0.0
    sma200_v = safe_float(sma200.iloc[-1], np.nan)
    if not np.isnan(sma200_v) and sma200_v > 0:
        bias200 = (close_v - sma200_v) / sma200_v * 100

    mid_score += smooth_low_score(rsi_w_val, 25, 55, 30)
    mid_score += smooth_low_score(bias200, -35, -3, 35)
    mid_score += smooth_low_score(cci_val, -220, -60, 15)

    # 加入周線MACD動能 (滿分20)
    df_weekly = df["close"].resample('W').last()
    if len(df_weekly) >= 26:
        macd_w, sig_w, hist_w = calc_macd(df_weekly)
        if len(hist_w) >= 2 and hist_w.iloc[-1] > hist_w.iloc[-2] and hist_w.iloc[-1] < 0:
            mid_score += 20
            mid_signals.append("周MACD跌勢收斂")
        elif len(hist_w) >= 1 and hist_w.iloc[-1] > 0:
            mid_score += 15
            if len(hist_w) >= 2 and hist_w.iloc[-1] > hist_w.iloc[-2]:
                mid_score += 5
                mid_signals.append("周MACD轉強")

    if rsi_w_val < 35: mid_signals.append("周RSI超賣")
    if bias200 < -15: mid_signals.append("年線乖離大")
    if rsi_w_val > 60:
        mid_score *= 0.70
        mid_signals.append("⚠️周線偏高(防假底)")

    if market_state == "bear_high_vol":
        short_score *= 1.05
        mid_score *= 1.05
    elif market_state == "bull_low_vol":
        short_score *= 0.90

    signals = list(dict.fromkeys(signals + mid_signals))
    return (
        round(clip_score(short_score), 1), round(clip_score(mid_score), 1), signals,
        oversold_count, resonance, round(cmf_val, 3), round(vwap_val, 2), round(vol_z, 2)
    )

def signal_label(short_score, mid_score):
    if short_score >= 70 or mid_score >= 70: return "🔥 強烈撈底", "buy"
    if short_score >= 50 or mid_score >= 50: return "⭐️ 值得關注", "watch"
    if short_score >= 35 or mid_score >= 35: return "👁️ 觀察中", "observe"
    return "—", "none"

def signal_badge(label):
    if label.startswith("🔥"): return "badge-buy"
    if label.startswith("⭐️"): return "badge-watch"
    if label.startswith("👁️"): return "badge-observe"
    return "badge-none"

# 估值計算保持原樣...
def get_pe_percentile_from_info(info):
    eps = safe_float(info.get("eps"), np.nan)
    hist = info.get("hist_5y")
    if np.isnan(eps) or eps <= 0 or hist is None or len(hist) < 30: return None, None
    series = (hist / eps).dropna()
    if len(series) < 30: return None, None
    current_calc_pe = safe_float(series.iloc[-1], np.nan)
    return float((series < current_calc_pe).mean()), current_calc_pe

def get_pb_percentile_from_info(info):
    bv = safe_float(info.get("book_value"), np.nan)
    hist = info.get("hist_5y")
    if np.isnan(bv) or bv <= 0 or hist is None or len(hist) < 30: return None, None
    series = (hist / bv).dropna()
    if len(series) < 30: return None, None
    current_calc_pb = safe_float(series.iloc[-1], np.nan)
    return float((series < current_calc_pb).mean()), current_calc_pb

def percentile_value_score(percentile, fallback_value, thresholds):
    if percentile is not None:
        if percentile < 0.10: return 90.0
        if percentile < 0.25: return 70.0
        if percentile < 0.50: return 40.0
        return 10.0
    if fallback_value is None or fallback_value <= 0: return None
    a, b, c = thresholds
    if fallback_value < a: return 90.0
    if fallback_value < b: return 70.0
    if fallback_value < c: return 40.0
    return 10.0

def quality_filter(roe, de_ratio, rev_growth):
    flags = []
    penalty = 1.0
    if not np.isnan(roe:=safe_float(roe, np.nan)) and roe < 0.05:
        flags.append("⚠️ ROE<5%"); penalty -= 0.15
    if not np.isnan(de_ratio:=safe_float(de_ratio, np.nan)) and de_ratio > 150:
        flags.append("⚠️ 負債>150%"); penalty -= 0.15
    if not np.isnan(rev_growth:=safe_float(rev_growth, np.nan)) and rev_growth < -0.10:
        flags.append("⚠️ 營收<-10%"); penalty -= 0.10
    if not flags: flags.append("✅ 品質過關")
    return flags, max(0.50, penalty)

def dividend_bonus(div_yield, sector):
    y = safe_float(div_yield, np.nan)
    if np.isnan(y) or y <= 0: return 0.0, "股息率 N/A"
    pct = y * 100 if y <= 1 else y
    cap = 12 if any(x in sector.lower() for x in ["bank", "util", "real estate", "financial", "hk_bank", "hk_util", "hk_property"]) else 6
    bonus = smooth_high_score(pct, 2.0, 7.0, cap)
    return round(bonus, 1), f"股息率 {pct:.2f}% (+{bonus:.1f})"

def valuation_label(score):
    if score >= 70: return "💰 便宜"
    if score >= 40: return "😐 合理"
    return "🔥 偏貴"

def build_sector_peer_cache(scan_list, info_map):
    cache = {}
    for ticker in scan_list:
        info = info_map.get(ticker, {})
        sector = get_sector_from_info(ticker, info)
        pe = safe_float(info.get("pe"), np.nan)
        if not np.isnan(pe) and pe > 0:
            cache.setdefault(sector, []).append((ticker, pe))
    return cache

def sector_relative_valuation(ticker, pe, sector, peer_cache):
    if pe is None or pe <= 0 or sector == "OTHER": return None
    peer_pes = [p for pt, p in peer_cache.get(sector, []) if pt != ticker and p is not None and p > 0]
    if len(peer_pes) < 2: return None
    median_pe = float(np.median(peer_pes))
    return {
        "sector": sector, "median_pe": round(median_pe, 1),
        "rel_pct": round(float((pe - median_pe) / median_pe * 100 if median_pe > 0 else np.nan), 1),
        "n_peers": len(peer_pes)
    }

def fund_flow_detail(df):
    if df is None or len(df) < 20: return 0.0, {}
    close, volume = df["close"], df["volume"]
    mfi_now = safe_float(calc_mfi(df).iloc[-1], 50)
    big_down = close.pct_change() < -0.02
    avg_vol = volume.rolling(20).mean()
    ratios = (volume[big_down] / avg_vol[big_down]).replace([np.inf, -np.inf], np.nan).dropna().tail(5)
    down_ratio = safe_float(ratios.mean(), 0) if not ratios.empty else 0

    detail = {}
    down_score = 30 if 0 < down_ratio < 0.8 else (15 if 0 < down_ratio < 1.1 else 0)
    detail["大跌日"] = (down_score, f"量比{down_ratio:.2f}")
    
    mfi_score = smooth_low_score(mfi_now, 15, 50, 40)
    detail["MFI"] = (round(mfi_score, 1), f"{mfi_now:.1f}")
    return clip_score(down_score + mfi_score), detail

def technical_detail_score(df):
    if df is None or len(df) < 60: return 0.0, {}
    rsi = safe_float(calc_rsi(df["close"]).iloc[-1], 50)
    k, d, _ = calc_kdj(df)
    cci, wr = safe_float(calc_cci(df).iloc[-1], 0), safe_float(calc_wr(df).iloc[-1], -50)
    scores = {
        "RSI(14)": (smooth_low_score(rsi, 20, 45, 25), f"{rsi:.1f}"),
        "KDJ": (smooth_low_score((safe_float(k.iloc[-1],50) + safe_float(d.iloc[-1],50)) / 2, 10, 40, 25), f"K={k.iloc[-1]:.1f}"),
        "CCI": (smooth_low_score(cci, -200, -40, 25), f"{cci:.1f}"),
        "W%R": (smooth_low_score(wr, -95, -50, 25), f"{wr:.1f}")
    }
    return clip_score(sum(v[0] for v in scores.values())), scores

def score_four_dimension(ticker, info, peer_cache, market_state, macro_10d_ret=0.0):
    df = fetch_ohlcv(ticker, period="2y")
    if df is None or len(df) < 60: return None

    name = info.get("name", ticker)
    pe, pb = safe_float(info.get("pe"), np.nan), safe_float(info.get("pb"), np.nan)
    sector = get_sector_from_info(ticker, info)

    tech_total, _ = technical_detail_score(df)
    short_score, mid_score, signals, _, _, _, _, _ = score_stock(df, market_state, macro_10d_ret)

    pe_perc, pe_calc = get_pe_percentile_from_info(info)
    pb_perc, pb_calc = get_pb_percentile_from_info(info)
    pe_score = percentile_value_score(pe_perc, None if np.isnan(pe) else pe, (10, 15, 20))
    pb_score = percentile_value_score(pb_perc, None if np.isnan(pb) else pb, (1, 1.5, 2.5))

    is_financial = any(x in sector.lower() for x in ["bank", "financial", "hk_bank"])
    if pe_score is not None and pb_score is not None:
        val_score = pb_score * 0.60 + pe_score * 0.40 if is_financial else pe_score * 0.60 + pb_score * 0.40
    else:
        val_score = pe_score if pe_score is not None else (pb_score if pb_score is not None else 50.0)

    quality_flags, quality_penalty = quality_filter(info.get("roe"), info.get("de_ratio"), info.get("rev_growth"))
    div_bonus, div_detail = dividend_bonus(info.get("div_yield"), sector)
    val_score = clip_score(val_score * quality_penalty + div_bonus)

    val_detail = f"PE {pe:.1f}｜PB {pb:.2f}｜{div_detail}"

    sector_info = sector_relative_valuation(ticker, None if np.isnan(pe) else pe, sector, peer_cache)
    sector_detail = f"{sector_info['sector']}：中位PE {sector_info['median_pe']} 偏差 {sector_info['rel_pct']:+.1f}%" if sector_info else "同業不足"

    current_price = safe_float(df["close"].iloc[-1], 0)
    high_52 = get_52w_high(df)
    drawdown = (current_price - high_52) / high_52 * 100 if high_52 > 0 else 0
    dd_score = smooth_low_score(drawdown, -45, 0, 90)

    fund_total, _ = fund_flow_detail(df)
    capital_flow = get_futu_capital_flow(ticker)
    capital_bonus = 0.0
    capital_detail = "未取得"
    if capital_flow is not None and not capital_flow.empty:
        inflow = safe_float(capital_flow.iloc[-1].get("in_flow", 0), 0)
        if inflow > 0:
            capital_bonus, capital_detail = 10, f"主力流入 {inflow:.0f}萬"
        else: capital_detail = "主力流出"
    fund_total = clip_score(fund_total + capital_bonus)

    macro = fetch_macro()
    vix = safe_float(macro.get("VIX", {}).get("val"), 20)
    weights = get_dynamic_weights(vix)
    raw_total = (weights["tech"] * tech_total + weights["val"] * val_score + weights["dd"] * dd_score + weights["fund"] * fund_total)
    total_score = round(clip_score(raw_total), 1)
    
    return {
        "ticker": ticker, "name": name, "price": round(current_price, 3),
        "total_score": total_score, "confidence": "高信心" if total_score >= 80 else ("中等信心" if total_score >= 60 else "低信心"),
        "short_score": short_score, "mid_score": mid_score, "signals": "、".join(signals) if signals else "—",
        "tech_total": round(tech_total, 1), "val_score": round(val_score, 1), "val_label": valuation_label(val_score),
        "val_detail": val_detail, "pe_percentile": pe_perc, "pb_percentile": pb_perc,
        "sector": sector, "sector_detail": sector_detail, "quality_flags": "｜".join(quality_flags),
        "quality_penalty": quality_penalty, "drawdown": round(drawdown, 1), "dd_score": round(dd_score, 1),
        "fund_total": round(fund_total, 1), "capital_detail": capital_detail, "weights": weights, "vix": vix,
        "hi52": high_52, "info_source": info.get("info_source", "Yahoo")
    }

# ─────────────────────────────────────────────────────────────
# 市場狀態、信號紀錄與報告 (改用 SQLite)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_macro():
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = fetch_ohlcv(ticker, period="1y")
            if df is None or len(df) < 15: continue
            close = safe_float(df["close"].iloc[-1], 0)
            prev = safe_float(df["close"].iloc[-2], close)
            change = (close - prev) / prev * 100 if prev else 0
            high, low = safe_float(df["high"].max(), close), safe_float(df["low"].min(), close)
            pct = (close - low) / (high - low) * 100 if high != low else 50
            ret_10d = (close - safe_float(df["close"].iloc[-10], close)) / safe_float(df["close"].iloc[-10], close) * 100
            result[name] = {
                "val": close, "chg": change, "pct": pct, "hi": high, "lo": low,
                "rsi": safe_float(calc_rsi(df["close"]).iloc[-1], 50), "ret_10d": ret_10d
            }
        except Exception as exc: add_error("宏觀指標計算失敗", ticker, exc)
    return result

def classify_market_state():
    try:
        spy = fetch_ohlcv("SPY", period="6mo")
        if spy is None or len(spy) < 60: return "unknown", 0, 0
        close = spy["close"]
        ret_60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100
        volatility = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        vix = safe_float(fetch_macro().get("VIX", {}).get("val"), 20)
        if ret_60 < -5 and vix > 25: return "bear_high_vol", ret_60, volatility
        if ret_60 < -5: return "bear_low_vol", ret_60, volatility
        if ret_60 > 5 and vix > 25: return "bull_high_vol", ret_60, volatility
        if ret_60 > 5: return "bull_low_vol", ret_60, volatility
        return "neutral", ret_60, volatility
    except Exception as exc:
        add_error("市場狀態分類失敗", exc=exc)
        return "unknown", 0, 0

def get_dynamic_weights(vix):
    if vix >= 30: return {"tech": 0.40, "val": 0.35, "dd": 0.10, "fund": 0.15}
    if vix >= 25: return {"tech": 0.35, "val": 0.35, "dd": 0.15, "fund": 0.15}
    if vix <= 15: return {"tech": 0.20, "val": 0.50, "dd": 0.15, "fund": 0.15}
    return {"tech": 0.30, "val": 0.40, "dd": 0.15, "fund": 0.15}

def log_signal_to_db(ticker, total_score, label, price, date):
    """寫入 SQLite 資料庫，避免雲端重啟遺失檔案"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM signals WHERE date=? AND ticker=?", (date, ticker))
        if cur.fetchone():
            conn.close()
            return False
        cur.execute("INSERT INTO signals (date, ticker, total_score, label, price) VALUES (?, ?, ?, ?, ?)",
                    (date, ticker, total_score, label, price))
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        add_error("寫入資料庫失敗", exc=exc)
        return False

def calculate_position(price, stop_loss, account_size=100000, risk_pct=0.02):
    risk_amount = account_size * risk_pct
    per_share_risk = abs(price - stop_loss)
    return int(risk_amount / per_share_risk) if per_share_risk > 0 else 0, risk_amount

def generate_pdf_report(results, market_state, vix):
    if not PDF_AVAILABLE: return None
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Daily Bottom-Fishing Report", ln=1, align="C")
        pdf.cell(200, 10, txt=f"Market: {market_state} | VIX: {vix:.1f}", ln=1)
        pdf.ln(8)
        for r in results[:10]:
            pdf.cell(200, 8, txt=f"{r['ticker']} Price {r['price']} Score {r['total_score']}", ln=1)
        return pdf.output(dest="S").encode("latin-1")
    except Exception as exc:
        add_error("PDF生成失敗", exc=exc)
        return None

# ─────────────────────────────────────────────────────────────
# Header / Sidebar
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統 Pro+｜終極穩定版</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據庫：SQLite 持久化保護</p>",
    unsafe_allow_html=True
)
st.divider()

with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    market = st.radio("市場", ["🇭🇰 港股", "🇺🇸 美股", "📋 自選"], index=1)
    custom_input = ""
    if market == "📋 自選":
        custom_input = st.text_area("輸入代碼（每行一個）", "AAPL\nNVDA\n0700.HK\n9988.HK")
    st.divider()
    filter_sig = st.multiselect("篩選信號", ["🔥 強烈撈底", "⭐️ 值得關注", "👁️ 觀察中", "—"], default=["🔥 強烈撈底", "⭐️ 值得關注"])
    min_short = st.slider("最低短線分", 0, 100, 0)
    min_mid = st.slider("最低中線分", 0, 100, 0)
    resonance_filter = st.selectbox("🔍 共振強度篩選", ["全部", "強", "中", "弱"], index=0)
    if quote_ctx: st.success("✅ 富途API 已連線")
    else: st.warning("⚠️ 富途API 未連線，使用 Yahoo 數據")

market_state, market_ret, market_vol = classify_market_state()
macro_data = fetch_macro()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🌍 市場氣氛", "📊 個股掃描", "📐 回撤/江恩", "📈 技術圖表",
    "🎯 四維撈底評分", "📋 信號與回測", "⚖️ 風險管理",
    "🔄 週期投影", "🧪 高階實驗室", "🛠️ 數據庫健康"
])

# ─────────────────────────────────────────────────────────────
# Tab 1 市場氣氛
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")
    vix_now = safe_float(macro_data.get("VIX", {}).get("val"), 20)
    state_map = {
        "bear_high_vol": "🐻 熊市高波動", "bear_low_vol": "🐻 熊市低波動",
        "bull_high_vol": "🐂 牛市高波動", "bull_low_vol": "🐂 牛市低波動",
        "neutral": "😐 中性", "unknown": "❓ 無法判斷"
    }
    st.markdown(f"### 當前市場狀態：{state_map.get(market_state, market_state)}")
    st.caption(f"SPY 60日回報：{market_ret:.1f}% ｜ 年化波動率：{market_vol:.1f}% ｜ VIX：{vix_now:.1f}")

    st.divider()
    kpi_items = [
        ("VIX", "😱 恐慌指數"), ("VVIX", "🌊 波動之波動"),
        ("SPX", "🇺🇸 標普500"), ("HSI", "🇭🇰 恒生指數"),
        ("US10Y", "🏦 美債10年息"), ("DXY", "💵 美元指數"),
        ("HYG", "📉 高收益債"), ("VHSI", "🇭🇰 港股波幅")
    ]
    for group in [kpi_items[:4], kpi_items[4:]]:
        cols = st.columns(4)
        for i, (key, label) in enumerate(group):
            item = macro_data.get(key, {})
            val, chg, pct = safe_float(item.get("val"), 0), safe_float(item.get("chg"), 0), safe_float(item.get("pct"), 0)
            color = C_GREEN if chg >= 0 else C_RED
            emoji, name = label.split()[0], " ".join(label.split()[1:])
            with cols[i]:
                st.markdown(
                    f"<div class='metric-card'><div>{emoji}</div><div style='color:#8b949e;font-size:.7em'>{name}</div>"
                    f"<div style='font-size:1.1em;font-weight:bold'>{val:.2f}</div>"
                    f"<div style='color:{color}'>{chg:+.2f}%</div><div style='color:#8b949e;font-size:.68em'>52W:{pct:.0f}%</div></div>",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────
# Tab 2 個股掃描
# ─────────────────────────────────────────────────────────────
with tab2:
    tickers = HK_WATCHLIST if market == "🇭🇰 港股" else (US_WATCHLIST if market == "🇺🇸 美股" else [x.strip().upper() for x in custom_input.split("\n") if x.strip()] or US_WATCHLIST)
    
    if vix_now >= 30: fc, fi, auto_min_mid, fl = C_GREEN, "🔥", 60, f"VIX {vix_now:.1f} 極度恐慌 — 建議只看中線分≥60"
    elif vix_now >= 25: fc, fi, auto_min_mid, fl = C_ORANGE, "⚠️", 50, f"VIX {vix_now:.1f} 高波動 — 建議只看中線分≥50"
    elif vix_now <= 15: fc, fi, auto_min_mid, fl = C_RED, "😎", 0, f"VIX {vix_now:.1f} 市場偏貪婪 — 注意追高"
    else: fc, fi, auto_min_mid, fl = C_GREY, "😐", 0, f"VIX {vix_now:.1f} 市場中性"

    st.markdown(f"<div style='background:#161b22;border-left:4px solid {fc};border-radius:8px;padding:12px 16px;margin-bottom:12px'>{fi} <span style='color:{fc};font-weight:bold'>市場氣氛濾網</span>：{fl}</div>", unsafe_allow_html=True)
    effective_min_mid = max(min_mid, auto_min_mid)

    st.subheader(f"📊 個股掃描 — {market} ({len(tickers)} 隻)")
    with st.spinner(f"正在並行下載 {len(tickers)} 隻股票數據..."):
        data_map = fetch_multiple(tickers, period="2y")

    rows = []
    failed_tickers = []
    for ticker in tickers:
        df = data_map.get(ticker)
        if df is None or len(df) < 60:
            failed_tickers.append(ticker)
            continue
        try:
            macro_10d_ret = macro_data.get("HSI" if ticker.endswith(".HK") else "SPX", {}).get("ret_10d", 0)
            short_s, mid_s, sigs, oversold_count, resonance, cmf_val, vwap_val, vol_z = score_stock(df, market_state, macro_10d_ret)
            label, stype = signal_label(short_s, mid_s)
            close_v = safe_float(df["close"].iloc[-1], 0)
            high_52 = get_52w_high(df)
            rows.append({
                "代碼": ticker, "現價": round(close_v, 3), 
                "1日漲跌%": round((close_v - safe_float(df["close"].iloc[-2], close_v)) / safe_float(df["close"].iloc[-2], close_v) * 100, 2),
                "距高位%": round((close_v-high_52)/high_52*100, 1) if high_52 else 0,
                "短線分": short_s, "中線分": mid_s, "信號": label, "_type": stype,
                "觸發指標": "、".join(sigs) if sigs else "—",
                "cmf": cmf_val, "vwap": vwap_val, "vol_z": vol_z, "resonance": resonance, "oversold_count": oversold_count
            })
        except Exception as exc:
            add_error("掃描計算失敗", ticker, exc)
            failed_tickers.append(ticker)

    if failed_tickers:
        st.warning(f"⚠️ {len(failed_tickers)} 隻股票數據不足：{', '.join(failed_tickers)}")

    display_rows = [r for r in rows if resonance_filter == "全部" or r["resonance"] == resonance_filter]
    filtered = [r for r in display_rows if r["信號"] in filter_sig and r["短線分"] >= min_short and r["中線分"] >= effective_min_mid]
    
    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 強烈撈底：{sum(r['_type']=='buy' for r in filtered)} 隻**")
    if filtered:
        table = pd.DataFrame(filtered).drop(columns=["_type"])
        st.dataframe(table.sort_values(by=["中線分", "短線分"], ascending=[False, False]), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Tab 3 回撤與江恩
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📐 斐波那契 & 江恩支撐計算器")
    c1, c2, c3 = st.columns(3)
    with c1: tk_input = st.text_input("股票代碼", "NVDA", key="dd_ticker").upper()
    with c2: manual_high = st.number_input("手動輸入高位（0=自動）", min_value=0.0, value=0.0)
    with c3: manual_low = st.number_input("手動輸入低位（0=自動）", min_value=0.0, value=0.0)
    
    if st.button("🔍 計算", type="primary", key="dd_calc"):
        df = fetch_ohlcv(tk_input, period="2y")
        if df is None: st.error("找不到數據。")
        else:
            current = safe_float(df["close"].iloc[-1], 0)
            high = manual_high if manual_high > 0 else get_52w_high(df)
            low = manual_low if manual_low > 0 else safe_float(df["low"].iloc[-252:].min(), 0)
            st.markdown(f"### {tk_input}｜現價：**{current:.3f}** ｜ 52周高：**{high:.3f}** ｜ 52周低：**{low:.3f}**")
            
            col_d, col_f, col_g = st.columns(3)
            with col_d:
                st.markdown("##### 📉 高點回撤位")
                st.dataframe(pd.DataFrame([{"回撤": k, "目標價": v, "距離現價": f"{current-v:+.2f}"} for k, v in drop_levels(high).items()]), hide_index=True)
            with col_f:
                st.markdown("##### 🌀 斐波那契支撐")
                st.dataframe(pd.DataFrame([{"比率": k, "支撐價": v, "距離現價": f"{current-v:+.2f}"} for k, v in fib_levels(low, high).items()]), hide_index=True)
            with col_g:
                st.markdown("##### 📐 江恩八分/三分位")
                st.dataframe(pd.DataFrame([{"江恩比率": k, "支撐價": v, "距離現價": f"{current-v:+.2f}"} for k, v in gann_levels(low, high).items()]), hide_index=True)

# ─────────────────────────────────────────────────────────────
# Tab 4 技術圖表
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart = st.text_input("輸入股票代碼", "AAPL", key="chart_ticker").upper()
    period_map = {"3個月": "3mo", "6個月": "6mo", "1年": "1y", "2年": "2y"}
    period_sel = st.radio("時間範圍", list(period_map.keys()), index=2, horizontal=True)
    df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])
    
    if df_ch is not None and len(df_ch) > 30:
        close = df_ch["close"]
        rsi, weekly_rsi = calc_rsi(close), calc_rsi(close, 70)
        macd, sig, hist = calc_macd(close)
        sma20 = close.rolling(20).mean()
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.02, row_heights=[.5, .15, .15, .2])
        fig.add_trace(go.Candlestick(x=df_ch.index, open=df_ch["open"], high=df_ch["high"], low=df_ch["low"], close=close, name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=sma20, mode="lines", line=dict(color="#f0883e"), name="MA20"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_ch.index, y=df_ch["volume"], marker_color=[C_GREEN if df_ch["close"].iloc[i]>=df_ch["open"].iloc[i] else C_RED for i in range(len(df_ch))]), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=rsi, mode="lines", line=dict(color=C_ORANGE), name="RSI(14)"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=weekly_rsi, mode="lines", line=dict(color=C_PURPLE, dash="dot"), name="Weekly RSI"), row=3, col=1)
        fig.add_trace(go.Bar(x=df_ch.index, y=hist, marker_color=[C_GREEN if x>=0 else C_RED for x in hist.fillna(0)], name="MACD Hist"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=macd, mode="lines", line=dict(color=C_BLUE), name="MACD"), row=4, col=1)
        
        fig.update_layout(height=800, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Tab 5 四維評分
# ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🎯 四維撈底評分模型｜包含江恩與MFI背離")
    
    scan_list = HK_WATCHLIST if market == "🇭🇰 港股" else (US_WATCHLIST if market == "🇺🇸 美股" else [x.strip().upper() for x in custom_input.split("\n") if x.strip()])
    
    if st.button("🔄 執行四維深度掃描", type="primary"):
        clear_errors()
        progress = st.progress(0, text="下載基本面與5年資料...")
        info_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in scan_list}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                try: info_map[ticker] = future.result()
                except Exception as exc: add_error("基本面工作失敗", ticker, exc)
                progress.progress(int((i+1) / len(scan_list) * 35), text=f"基本面資料 {i+1}/{len(scan_list)}")

        progress.progress(40, text="建立同業PE快取...")
        peer_cache = build_sector_peer_cache(scan_list, info_map)

        progress.progress(45, text="平行計算四維評分...")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCORE) as executor:
            futures = {
                executor.submit(score_four_dimension, ticker, info_map.get(ticker, {}), peer_cache, market_state, macro_data.get("HSI" if ticker.endswith(".HK") else "SPX", {}).get("ret_10d", 0)): ticker
                for ticker in scan_list
            }
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                try:
                    res = future.result()
                    if res: results.append(res)
                except Exception as exc: add_error("評分平行工作失敗", ticker, exc)
                progress.progress(45 + int((i+1) / len(scan_list) * 55), text=f"評分計算 {i+1}/{len(scan_list)}")
        progress.empty()

        if results:
            results.sort(key=lambda x: x["total_score"], reverse=True)
            today = datetime.now().strftime("%Y-%m-%d")
            added = sum(1 for r in results if r["total_score"] >= 70 and log_signal_to_db(r["ticker"], r["total_score"], r["confidence"], r["price"], today))
            if added: st.success(f"已將 {added} 筆今日高分信號存入 SQLite 資料庫。")

            st.dataframe(pd.DataFrame([{
                "代碼": r["ticker"], "名稱": r["name"], "現價": r["price"], "總分": r["total_score"],
                "估值標籤": r["val_label"], "技術分": r["tech_total"], "估值分": r["val_score"],
                "回撤分": r["dd_score"], "資金分": r["fund_total"], "短線分": r["short_score"], "中線分": r["mid_score"],
                "觸發信號": r["signals"]
            } for r in results]), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Tab 6 信號追蹤與回測 (SQLite版)
# ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("📋 歷史信號與回測 (SQLite 持久化)")
    try:
        conn = sqlite3.connect(DB_FILE)
        df_log = pd.read_sql_query("SELECT * FROM signals ORDER BY date DESC", conn)
        conn.close()
    except Exception as exc:
        df_log = pd.DataFrame()
        add_error("讀取 SQLite 失敗", exc=exc)

    if df_log.empty:
        st.info("資料庫尚無信號紀錄。請先在「四維撈底評分」完成一次掃描。")
    else:
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        hold_days = st.selectbox("持有交易日", [5, 10, 20, 30], index=1)
        if st.button("計算已成熟信號績效", key="run_backtest"):
            # 回測邏輯與前版相同...
            results = []
            for _, row in df_log.iterrows():
                ticker = row["ticker"]
                entry_date = pd.Timestamp(row["date"]).normalize()
                entry_price = safe_float(row["price"], np.nan)
                if np.isnan(entry_price) or entry_price <= 0: continue
                df = fetch_ohlcv(ticker, period="2y")
                if df is None or df.empty: continue
                try:
                    index_norm = pd.DatetimeIndex(df.index).normalize()
                    future_dates = df.index[index_norm >= entry_date]
                    if len(future_dates) <= hold_days: continue
                    exit_price = safe_float(df.loc[future_dates[hold_days], "close"], np.nan)
                    if np.isnan(exit_price): continue
                    ret = (exit_price - entry_price) / entry_price * 100
                    results.append({"代碼": ticker, "進場日": entry_date.strftime("%Y-%m-%d"), "進場價": entry_price, "出場價": exit_price, "回報%": round(ret, 2)})
                except Exception: pass
            if results:
                bt = pd.DataFrame(results)
                st.dataframe(bt, use_container_width=True, hide_index=True)
                st.metric("平均回報", f"{bt['回報%'].mean():.2f}%")
            else: st.info("目前尚未有足夠成熟的信號可作此持有期回測。")

# ─────────────────────────────────────────────────────────────
# Tab 7 風險管理
# ─────────────────────────────────────────────────────────────
with tab7:
    st.subheader("⚖️ 風險管理與部位計算")
    account_size = st.number_input("帳戶總值（USD）", value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆最大風險（%）", 0.5, 5.0, 2.0) / 100
    ticker = st.text_input("股票代碼", "AAPL", key="risk_ticker").upper()
    if st.button("計算部位"):
        df = fetch_ohlcv(ticker, period="1y")
        if df is not None:
            current = safe_float(df["close"].iloc[-1], 0)
            atr = safe_float(calc_atr(df).iloc[-1], 0)
            stop = current - 2 * atr if atr > 0 else current * 0.92
            shares, risk_amt = calculate_position(current, stop, account_size, risk_pct)
            st.metric("建議股數", f"{shares} 股")
            st.caption(f"參考2倍ATR止損價：{stop:.3f} ｜ 風險金額：${risk_amt:,.2f}")

# ─────────────────────────────────────────────────────────────
# Tab 10 數據庫健康
# ─────────────────────────────────────────────────────────────
with tab10:
    st.subheader("🛠️ 數據與系統健康狀態")
    logs = st.session_state.get("error_log", [])
    if logs:
        st.warning(f"目前有 {len(logs)} 項警告紀錄。")
        st.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("清除錯誤紀錄"):
            clear_errors()
            st.rerun()
    else: st.success("系統運作正常，無錯誤紀錄。")
