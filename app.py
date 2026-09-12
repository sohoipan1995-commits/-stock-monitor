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
# 評分與估值模型
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

    vol_confirm = 1.0
    open_v = safe_float(df["open"].iloc[-1], close_v)
    sma20_v = safe_float(sma20.iloc[-1], close_v)
    if vol_z > 2.0 and close_v > vwap_val: vol_confirm = 1.25
    elif vol_z > 2.0 and close_v > open_v: vol_confirm = 1.15
    elif vol_z < -1.5 and close_v < sma20_v: vol_confirm = 0.75

    short_score = (rsi_score + kdj_score + cci_score + wr_score) * mult * vol_confirm
    signals = []

    if len(wr) >= 5 and len(mfi) >= 5:
        if all(w < -90 for w in wr.iloc[-5:]):
            if close.iloc[-1] <= close.iloc[-5] and mfi.iloc[-1] > mfi.iloc[-5] + 5:
                short_score += 15
                signals.append("💎W%R鈍化+MFI底背離")

    if macro_10d_ret < -3.0 and len(close) >= 10:
        stock_10d_ret = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        if stock_10d_ret > 0:
            short_score += 15
            signals.append("💪逆市抗跌")

    if macd_val > sig_val and macd_val < 0:
        short_score += 8 * vol_confirm
        signals.append("MACD低位金叉")
    if detect_double_bottom(df):
        short_score += 10
        signals.append("🕳️確認雙底")
    if detect_macd_bullish_divergence(df):
        short_score += 12
        signals.append("📉MACD底背離")

    mid_score = 0.0
    mid_signals = []
    
    bias200 = 0.0
    sma200_v = safe_float(sma200.iloc[-1], np.nan)
    if not np.isnan(sma200_v) and sma200_v > 0:
        bias200 = (close_v - sma200_v) / sma200_v * 100

    mid_score += smooth_low_score(rsi_w_val, 25, 55, 30)
    mid_score += smooth_low_score(bias200, -35, -3, 35)
    mid_score += smooth_low_score(cci_val, -220, -60, 15)

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

# 此處省略部分未更動之估值函數，與上一版完全一致...
# ...

# ─────────────────────────────────────────────────────────────
# 市場狀態與紀錄
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
    except Exception: return "unknown", 0, 0

# ─────────────────────────────────────────────────────────────
# Header / Sidebar
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統 Pro+｜終極穩定版</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 模組：加入了多因子選股模型</p>",
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

market_state, market_ret, market_vol = classify_market_state()
macro_data = fetch_macro()

# ★★★ 重點更新：加入第11個 Tab ★★★
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🌍 市場氣氛", "📊 個股掃描", "📐 回撤/江恩", "📈 技術圖表",
    "🎯 四維撈底評分", "📋 信號與回測", "⚖️ 風險管理",
    "🔄 週期投影", "🧪 高階實驗室", "🛠️ 數據庫健康", "🌟 因子選股"
])

# ─────────────────────────────────────────────────────────────
# Tab 1 - 10 保持原狀 (此處簡略以節省版面，請保留你原本的程式碼即可)
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")
    # ...原版氣氛儀表板代碼...

with tab2:
    st.subheader(f"📊 個股掃描 — {market}")
    # ...原版個股掃描代碼...

with tab10:
    st.subheader("🛠️ 數據與系統健康狀態")
    # ...原版健康狀態代碼...

# ─────────────────────────────────────────────────────────────
# ★★★ Tab 11：🌟 多因子量化選股 (Smart Beta) ★★★
# ─────────────────────────────────────────────────────────────
with tab11:
    st.subheader("🌟 多因子量化選股 (Smart Beta / Multi-Factor)")
    st.markdown("這個模組利用學術界證實的核心因子進行全市場 Z-Score 標準化評分，幫你找出**低估值、高質量、強動能、低波動**的「六邊形戰士」。")
    st.info("💡 **實戰提示**：因子投資適合中長線持有 (3-6個月以上)，與左側的「撈底系統」(尋找極度超賣的短線反彈) 邏輯完美互補。")
    
    if st.button("🚀 執行多因子運算", type="primary", key="run_multifactor"):
        # 決定掃描名單
        scan_list = HK_WATCHLIST if market == "🇭🇰 港股" else (US_WATCHLIST if market == "🇺🇸 美股" else [x.strip().upper() for x in custom_input.split("\n") if x.strip()])
        
        with st.spinner("正在並行獲取基本面與歷史數據..."):
            info_map = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
                futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in scan_list}
                for future in as_completed(futures):
                    info_map[futures[future]] = future.result()
            
            # 拉取 1 年歷史資料以計算 6 個月動能與波動率
            data_map = fetch_multiple(scan_list, period="1y")
        
        factor_records = []
        for ticker in scan_list:
            df = data_map.get(ticker)
            info = info_map.get(ticker, {})
            if df is None or len(df) < 130: # 確保有半年以上的數據
                continue
                
            pe = safe_float(info.get("pe"), np.nan)
            roe = safe_float(info.get("roe"), np.nan)
            close = df["close"]
            
            # 計算動能因子：過去 6 個月 (約 126 個交易日) 回報率
            mom_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100
            
            # 計算防禦因子：過去 6 個月每日回報率的年化波動率
            daily_returns = close.pct_change().iloc[-126:]
            volatility_6m = daily_returns.std() * np.sqrt(252) * 100
            
            factor_records.append({
                "代碼": ticker,
                "名稱": info.get("name", ticker),
                "最新價": round(close.iloc[-1], 2),
                "PE": pe,
                "ROE(%)": roe * 100 if pd.notna(roe) else np.nan,
                "動能(6M%)": mom_6m,
                "波動率(%)": volatility_6m
            })
            
        if factor_records:
            f_df = pd.DataFrame(factor_records)
            
            # 資料預處理：將缺失值 (NaN) 替換為中位數，以免 Z-Score 計算失敗
            for col in ["PE", "ROE(%)", "動能(6M%)", "波動率(%)"]:
                f_df[col] = f_df[col].fillna(f_df[col].median())
            
            # Z-Score 計算函數
            def calc_zscore(series, inverse=False):
                std_val = series.std()
                if std_val == 0 or pd.isna(std_val): 
                    return pd.Series(0, index=series.index)
                z = (series - series.mean()) / std_val
                return -z if inverse else z
            
            # 價值因子 (PE越低越好，所以 inverse=True)
            f_df['Z_價值(PE)'] = calc_zscore(f_df['PE'], inverse=True)
            
            # 質量因子 (ROE越高越好)
            f_df['Z_質量(ROE)'] = calc_zscore(f_df['ROE(%)'])
            
            # 動能因子 (回報越高越好)
            f_df['Z_動能(6M)'] = calc_zscore(f_df['動能(6M%)'])
            
            # 防禦因子 (波動率越低越好，所以 inverse=True)
            f_df['Z_防禦(波動)'] = calc_zscore(f_df['波動率(%)'], inverse=True)
            
            # 計算綜合得分 (等權重相加，實戰中可根據市場氣氛調整權重)
            f_df['🏆 綜合得分'] = round(f_df['Z_價值(PE)'] + f_df['Z_質量(ROE)'] + f_df['Z_動能(6M)'] + f_df['Z_防禦(波動)'], 2)
            
            f_df = f_df.sort_values('🏆 綜合得分', ascending=False)
            
            st.markdown(f"### 🏆 {market} 因子選股排名結果")
            cols_to_show = ["代碼", "名稱", "最新價", "🏆 綜合得分", "PE", "ROE(%)", "動能(6M%)", "波動率(%)", "Z_價值(PE)", "Z_質量(ROE)", "Z_動能(6M)", "Z_防禦(波動)"]
            
            # 使用背景漸變色視覺化得分
            st.dataframe(f_df[cols_to_show].style.background_gradient(subset=['🏆 綜合得分'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
            
            # 繪製「價值 vs 動能」散點矩陣
            st.markdown("### 📊 價值與動能分佈矩陣")
            
            # 處理氣泡大小：將質量 Z-Score 正規化為正數
            min_quality = f_df['Z_質量(ROE)'].min()
            point_sizes = [max(1, (x - min_quality + 0.5) * 6) for x in f_df['Z_質量(ROE)']]
            
            fig = px.scatter(f_df, x="PE", y="動能(6M%)", text="代碼", color="🏆 綜合得分", 
                             size=point_sizes, 
                             color_continuous_scale="RdYlGn",
                             title="價值 vs 動能 (氣泡越大代表 ROE 質量越好)")
            
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=600)
            
            # 加入十字參考線 (中位數)
            fig.add_hline(y=f_df["動能(6M%)"].median(), line_dash="dot", line_color="gray", annotation_text="市場動能中位", annotation_position="bottom right")
            fig.add_vline(x=f_df["PE"].median(), line_dash="dot", line_color="gray", annotation_text="市場估值中位", annotation_position="top left")
            
            st.plotly_chart(fig, use_container_width=True)
