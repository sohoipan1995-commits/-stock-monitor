import os
import json
import re
import time
import logging
import sqlite3
import warnings
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(
    page_title="📈 股票監察系統 V5.1｜撈底雷達＋三引擎版",
    page_icon="📈",
    layout="wide"
)

# ==================== 視覺配色與常數 ====================
C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V5.1-BottomFishing-ThreeEngine"
MODEL_VERSION = "bottom-three-engine-v5.1"
ANALYSIS_PERIOD = "5y"
OHLCV_TTL = 1800
INFO_TTL = 7200
DB_FILE = os.getenv("STOCK_DB_FILE", "signals.db")
MAX_WORKERS_INFO = 3
TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 10

# 預設核心美股與港股觀察名單（開箱即用，不依賴外部爬蟲）
CORE_US_WATCHLIST = [
    "ADBE", "CRM", "ORCL", "PYPL", "MSTR", "PFE", "LITE", "UNH", "UBER", "LLY",
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "ASML"
]

EXTENDED_US_WATCHLIST = [
    "QCOM", "INTC", "AMAT", "LRCX", "MU", "JPM", "BAC", "GS", "MS", "BRK-B",
    "COST", "WMT", "HD", "JNJ", "XOM", "NEE", "CLX", "NFLX", "PANW", "SNOW"
]

CORE_HK_WATCHLIST = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "9999.HK", "1024.HK", "0981.HK",
    "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "1299.HK", "2318.HK", "0883.HK",
    "0001.HK", "0016.HK", "0002.HK", "0003.HK", "0066.HK", "1211.HK", "2020.HK", "3033.HK"
]

ETF_TICKERS = {"SPY", "QQQ", "SOXL", "IWM", "3033.HK", "2800.HK", "2828.HK"}
MACRO_TICKERS = {
    "VIX": "^VIX", "SPX": "^GSPC", "NDX": "^NDX", "HSI": "^HSI",
    "US10Y": "^TNX", "DXY": "DX-Y.NYB", "VHSI": "^VHSI", "HYG": "HYG"
}

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
GANN_LEVELS = [0.125, 0.25, 0.333, 0.375, 0.500, 0.625, 0.666, 0.75, 0.875]
DROP_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
AUTO_FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)

SECTOR_MAP_FALLBACK = {
    "0005.HK": "Financial Services", "0011.HK": "Financial Services", "0939.HK": "Financial Services",
    "1398.HK": "Financial Services", "3988.HK": "Financial Services", "2318.HK": "Financial Services",
    "1299.HK": "Financial Services", "0388.HK": "Financial Services",
    "0002.HK": "Utilities", "0003.HK": "Utilities", "0066.HK": "Utilities",
    "0016.HK": "Real Estate", "0001.HK": "Real Estate",
    "0700.HK": "Communication Services", "9999.HK": "Communication Services", "1024.HK": "Communication Services",
    "9988.HK": "Consumer Cyclical", "3690.HK": "Consumer Cyclical", "9618.HK": "Consumer Cyclical",
    "1211.HK": "Consumer Cyclical", "2020.HK": "Consumer Cyclical",
    "1810.HK": "Technology", "0981.HK": "Technology", "0883.HK": "Energy", "3033.HK": "ETF"
}

SECTOR_PE_RANGES = {
    "Technology": (18, 24, 30), "Communication Services": (14, 19, 24),
    "Consumer Cyclical": (13, 18, 23), "Consumer Defensive": (15, 19, 23),
    "Healthcare": (15, 20, 25), "Industrials": (14, 18, 22),
    "Energy": (8, 11, 14), "Basic Materials": (9, 12, 15),
    "Real Estate": (8, 12, 16), "Utilities": (11, 15, 18),
}
DEFAULT_PE_RANGE = (12, 16, 20)
CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
FINANCIAL_KEYWORDS = ("financial", "bank", "insurance", "hk_bank")
HK_MULTIPLE_DISCOUNT = 0.80
FCF_REQUIRED_YIELD = (0.085, 0.065, 0.05)
COST_OF_EQUITY = (0.12, 0.10, 0.085)
LONG_TERM_GROWTH = 0.03

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0d1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {color:#e6edf3!important;}
.decision-card {background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:16px;margin:10px 0;}
.tspr-panel {background:#161b22;border:1px solid #4a148c;border-radius:8px;padding:12px;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

if "error_log" not in st.session_state:
    st.session_state.error_log = []


def hkt_now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong"))


def safe_float(value, default=np.nan):
    try:
        return default if value is None or pd.isna(value) else float(value)
    except Exception:
        return default


def add_error(context, ticker=None, exc=None):
    message = str(context)
    if ticker:
        message += f"｜{ticker}"
    if exc:
        message += f"｜{type(exc).__name__}: {str(exc)[:160]}"
    item = {"時間": hkt_now().strftime("%Y-%m-%d %H:%M:%S"), "訊息": message}
    if threading.current_thread() is not threading.main_thread():
        logging.getLogger("stock_monitor").warning(message)
        return
    logs = st.session_state.get("error_log", [])
    if not any(x["訊息"] == message for x in logs):
        logs.append(item)
    st.session_state.error_log = logs[-200:]


def clear_errors():
    st.session_state.error_log = []


# ==================== 資料庫初始化 ====================
def init_db():
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model_version TEXT NOT NULL,
                decision TEXT,
                total_score REAL,
                price REAL,
                confirmation_price REAL,
                stop_loss REAL,
                target_2r REAL,
                market_regime TEXT,
                sector TEXT,
                features_json TEXT,
                created_at TEXT,
                PRIMARY KEY (signal_date, ticker, model_version)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_snapshots (
                snapshot_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT,
                PRIMARY KEY (snapshot_date, ticker, model_version)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS permanent_watchlist (
                ticker TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
        """)
        default_seeds = [(t, "核心預設名單", hkt_now().isoformat()) for t in CORE_US_WATCHLIST + CORE_HK_WATCHLIST]
        conn.executemany("INSERT OR IGNORE INTO permanent_watchlist VALUES (?,?,?)", default_seeds)


try:
    init_db()
except Exception as exc:
    st.error(f"資料庫初始化異常：{exc}")


def permanent_tickers():
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            return [row[0] for row in conn.execute("SELECT ticker FROM permanent_watchlist ORDER BY ticker")]
    except Exception:
        return sorted(set(CORE_US_WATCHLIST + CORE_HK_WATCHLIST))


def normalize_symbol(value):
    if not value or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.endswith(".HK"):
        num = re.sub(r"\.HK$", "", text)
        if num.isdigit():
            return num.zfill(4) + ".HK"
    if text.isdigit() and len(text) <= 5:
        return text.zfill(4) + ".HK"
    text = text.replace(".", "-")
    return text if re.fullmatch(r"[\^A-Z0-9][A-Z0-9.=-]{0,15}", text) else None


# ==================== 數據清理與批量下載（高容錯） ====================
def clean_price_frame(frame):
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(x).lower() for x in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns):
        return None
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0)
    frame = frame[(frame["close"] > 0) & (frame["open"] > 0) & (frame["low"] > 0)]
    return frame if not frame.empty else None


@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def fetch_ohlcv(ticker, period="2y", interval="1d"):
    try:
        raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                          progress=False, threads=False, timeout=15)
        frame = clean_price_frame(raw)
        if frame is None or frame.empty:
            add_error("無法取得有效日線", ticker)
            return None
        return frame
    except Exception as exc:
        add_error("下載日線失敗", ticker, exc)
        return None


@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def download_analysis_batch(symbols, period="5y"):
    symbols = tuple(sorted(set(symbols)))
    if not symbols:
        return {}
    raw = yf.download(list(symbols), period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False, timeout=20)
    frames = {}
    if raw is None or raw.empty:
        return frames
    for ticker in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    sub = raw[ticker]
                elif ticker in raw.columns.get_level_values(1):
                    sub = raw.xs(ticker, axis=1, level=1)
                else:
                    continue
            elif len(symbols) == 1:
                sub = raw
            else:
                continue
            cleaned = clean_price_frame(sub)
            if cleaned is not None:
                frames[ticker] = cleaned
        except Exception:
            continue
    return frames


def fetch_multiple(tickers, period="5y"):
    results = {}
    tickers = tuple(sorted(set(tickers)))
    for start in range(0, len(tickers), 35):
        chunk = tickers[start:start + 35]
        try:
            frames = download_analysis_batch(chunk, period)
            results.update(frames)
        except Exception as exc:
            add_error("批量日線下載異常", exc=exc)
        for t in chunk:
            if results.get(t) is None:
                single = fetch_ohlcv(t, period=period)
                if single is not None:
                    results[t] = single
    return results


@st.cache_data(ttl=21600, show_spinner=False)
def get_fx_rate(src, dst):
    if not src or not dst or src == dst:
        return 1.0
    fixed_map = {
        ("CNY", "HKD"): 1.08, ("HKD", "CNY"): 0.925,
        ("USD", "HKD"): 7.80, ("HKD", "USD"): 0.128,
        ("CNY", "USD"): 0.138, ("USD", "CNY"): 7.22
    }
    try:
        raw = yf.download(f"{src}{dst}=X", period="5d", interval="1d", progress=False, threads=False, timeout=8)
        frame = clean_price_frame(raw)
        if frame is not None and not frame.empty:
            return safe_float(frame["close"].iloc[-1], fixed_map.get((src, dst), 1.0))
    except Exception:
        pass
    return fixed_map.get((src, dst), 1.0)


@st.cache_data(ttl=INFO_TTL, show_spinner=False)
def get_full_stock_info(ticker):
    result = {
        "ticker": ticker, "name": ticker, "pe": None, "pb": None, "div_yield": None,
        "roe": None, "de_ratio": None, "rev_growth": None, "earnings_growth": None,
        "eps": None, "forward_eps": None, "book_value": None,
        "sector": SECTOR_MAP_FALLBACK.get(ticker, "Technology" if not ticker.endswith(".HK") else "OTHER"),
        "industry": None, "quote_type": "EQUITY",
        "currency": "HKD" if ticker.endswith(".HK") else "USD",
        "financial_currency": "HKD" if ticker.endswith(".HK") else "USD",
        "free_cashflow": None, "total_cash": None, "total_debt": None,
        "ebitda": None, "shares": None, "fx_rate": 1.0, "fx_ok": True,
        "eps_quote": np.nan, "forward_eps_quote": np.nan, "bvps_quote": np.nan,
        "fcf_ps_quote": np.nan, "eps_history_quote": [], "info_error": None
    }
    if ticker in ETF_TICKERS or ticker.startswith("^"):
        result["quote_type"] = "ETF"
        return result
    try:
        info = yf.Ticker(ticker).info or {}
        if not info:
            result["info_error"] = "Yahoo未回傳基本面(自動以技術面評分)"
            return result
        result["name"] = info.get("shortName") or info.get("longName") or ticker
        result["quote_type"] = info.get("quoteType") or "EQUITY"
        result["sector"] = info.get("sector") or result["sector"]
        result["industry"] = info.get("industry")
        result["currency"] = info.get("currency") or result["currency"]
        result["financial_currency"] = info.get("financialCurrency") or result["currency"]

        trailing = safe_float(info.get("trailingPE"), None)
        forward = safe_float(info.get("forwardPE"), None)
        px_now = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        result["pe"] = trailing if (trailing and trailing > 0) else (forward if (forward and forward > 0) else None)
        result["pb"] = safe_float(info.get("priceToBook"), None)
        result["roe"] = safe_float(info.get("returnOnEquity"), None)
        result["de_ratio"] = safe_float(info.get("debtToEquity"), None)
        result["rev_growth"] = safe_float(info.get("revenueGrowth"), None)
        result["earnings_growth"] = safe_float(info.get("earningsGrowth"), None)
        result["free_cashflow"] = safe_float(info.get("freeCashflow"), None)
        result["total_cash"] = safe_float(info.get("totalCash"), None)
        result["total_debt"] = safe_float(info.get("totalDebt"), None)
        result["ebitda"] = safe_float(info.get("ebitda"), None)
        result["shares"] = safe_float(info.get("sharesOutstanding"), None)

        div_rate = safe_float(info.get("dividendRate"))
        if np.isfinite(div_rate) and np.isfinite(px_now) and px_now > 0:
            result["div_yield"] = div_rate / px_now

        fx = get_fx_rate(result["financial_currency"], result["currency"])
        result["fx_rate"] = fx
        eps_raw = safe_float(info.get("trailingEps"))
        feps_raw = safe_float(info.get("forwardEps"))
        bv_raw = safe_float(info.get("bookValue"))

        result["eps_quote"] = (
            px_now / trailing if (np.isfinite(px_now) and trailing and trailing > 0)
            else (eps_raw * fx if np.isfinite(eps_raw) else np.nan)
        )
        result["forward_eps_quote"] = (
            px_now / forward if (np.isfinite(px_now) and forward and forward > 0)
            else (feps_raw * fx if np.isfinite(feps_raw) else np.nan)
        )
        pb_val = result["pb"]
        result["bvps_quote"] = (
            px_now / pb_val if (np.isfinite(px_now) and pb_val and pb_val > 0)
            else (bv_raw * fx if np.isfinite(bv_raw) else np.nan)
        )
        if np.isfinite(safe_float(result["free_cashflow"])) and np.isfinite(safe_float(result["shares"])) and result["shares"] > 0:
            result["fcf_ps_quote"] = result["free_cashflow"] / result["shares"] * fx
    except Exception as exc:
        result["info_error"] = f"基本面降級({type(exc).__name__})"
    return result


# ==================== 技術指標全集 (含週線 WR / KDJ / CCI / SAR / OBV / MFI) ====================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return rsi.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)


def calc_williams_r(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def calc_kdj(df, n=9, m1=3, m2=3):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
    return (tp - sma) / (0.015 * mad)


def calc_psar(df, iaf=0.02, maxaf=0.2):
    length = len(df)
    if length < 3:
        return pd.Series(df["close"], index=df.index), pd.Series(True, index=df.index)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    psar = close.copy()
    bull = np.ones(length, dtype=bool)
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2, length):
        if bull[i - 1]:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull[i - 1]:
            if low[i] < psar[i]:
                bull[i] = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull[i] = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            bull[i] = bull[i - 1]
            if bull[i]:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
    return pd.Series(psar, index=df.index), pd.Series(bull, index=df.index)


def calc_macd(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_mfi(df, period=14):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money = typical * df["volume"]
    pos = money.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg = money.where(typical < typical.shift(1), 0).rolling(period).sum()
    mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return mfi.mask((neg == 0) & (pos > 0), 100.0).mask((neg == 0) & (pos == 0), 50.0)


def calc_obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    obv_ma10 = obv.rolling(10).mean()
    return obv, obv_ma10


def calc_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_avwap_from_recent_low(df, lookback=60):
    recent = df.iloc[-lookback:].copy()
    if recent.empty:
        return pd.Series(dtype=float)
    anchor = recent["low"].idxmin()
    subset = df.loc[anchor:].copy()
    typical = (subset["high"] + subset["low"] + subset["close"]) / 3
    return (typical * subset["volume"]).cumsum() / subset["volume"].cumsum().replace(0, np.nan)


def local_minima_indices(series, order=3):
    values = np.asarray(series, dtype=float)
    result = []
    for i in range(order, len(values) - order):
        window = values[i - order:i + order + 1]
        if np.isfinite(values[i]) and values[i] == np.nanmin(window):
            if values[i] < values[i - 1] or values[i] < values[i + 1]:
                result.append(i)
    return result


def detect_double_bottom(df, lookback=80, min_gap=8, max_gap=55, tolerance=0.045, min_rebound=0.06):
    if df is None or len(df) < lookback:
        return False
    recent = df.iloc[-lookback:]
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, 3)
    for first in candidates:
        for second in candidates:
            gap = second - first
            if gap < min_gap or gap > max_gap:
                continue
            low1, low2 = safe_float(lows.iloc[first]), safe_float(lows.iloc[second])
            if min(low1, low2) <= 0 or abs(low2 - low1) / min(low1, low2) > tolerance:
                continue
            neckline = safe_float(recent["high"].iloc[first:second + 1].max())
            after_low = safe_float(recent["low"].iloc[second + 1:].min(), np.inf)
            still_valid = after_low >= min(low1, low2) * (1 - tolerance / 2) and safe_float(recent["close"].iloc[-1]) > min(low1, low2)
            if still_valid and (neckline - max(low1, low2)) / max(low1, low2) >= min_rebound:
                return True
    return False


def detect_macd_bullish_divergence(df, lookback=90, min_gap=8, max_gap=60):
    if df is None or len(df) < lookback:
        return False
    recent = df.iloc[-lookback:].copy()
    _, _, hist = calc_macd(recent["close"])
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, 3)
    for first in candidates:
        for second in candidates:
            gap = second - first
            if gap < min_gap or gap > max_gap:
                continue
            p1, p2 = safe_float(lows.iloc[first]), safe_float(lows.iloc[second])
            h1, h2 = safe_float(hist.iloc[first]), safe_float(hist.iloc[second])
            if p2 < p1 * 0.995 and h2 > h1:
                return True
    return False


# ==================== TSPR 10日結構支持位 + 破位跌幅統計 ====================
def calc_tspr_system(df, left_bars=10, right_bars=5, box_period=65, num_boxes=3, require_filter=False):
    if df is None or len(df) < left_bars + right_bars + 5:
        return {
            "active_support": np.nan, "is_broken": False, "status": "資料不足",
            "support_events": [], "breakdown_events": [], "support_lines": [],
            "pattern_boxes": [], "stats": {}
        }

    lows = df["low"].values
    highs = df["high"].values
    closes = df["close"].values
    dates = df.index
    n = len(df)

    obv_series, obv_ma10_series = calc_obv(df)
    mfi_series = calc_mfi(df, 14)
    obv_vals = obv_series.values
    obv_ma10_vals = obv_ma10_series.values
    mfi_vals = mfi_series.values

    support_events, breakdown_events, support_lines = [], [], []
    active_support = None
    support_start_idx = None
    is_broken = False
    last_pivot_idx = -999

    for i in range(left_bars + right_bars, n):
        pivot_idx = i - right_bars
        window_lows = lows[pivot_idx - left_bars: i + 1]
        left_window = lows[pivot_idx - left_bars: pivot_idx]

        if (
            np.isfinite(lows[pivot_idx])
            and lows[pivot_idx] == np.nanmin(window_lows)
            and lows[pivot_idx] < np.nanmin(left_window)
            and (pivot_idx - last_pivot_idx) >= right_bars
        ):
            if breakdown_events and breakdown_events[-1].get("end_idx") is None and pivot_idx >= breakdown_events[-1]["idx"]:
                last_brk = breakdown_events[-1]
                b_idx = last_brk["idx"]
                seg_lows = lows[b_idx: pivot_idx + 1]
                min_rel_idx = int(np.nanargmin(seg_lows))
                trough_idx = b_idx + min_rel_idx
                trough_price = float(lows[trough_idx])
                last_brk["end_idx"] = trough_idx
                last_brk["trough_date"] = dates[trough_idx]
                last_brk["trough_price"] = round(trough_price, 3)
                last_brk["post_break_drop_pts"] = round(trough_price - last_brk["price"], 2)
                last_brk["post_break_drop_pct"] = round((trough_price / last_brk["price"] - 1) * 100, 2)
                last_brk["peak_to_trough_drop_pct"] = round((trough_price / last_brk["wave_high"] - 1) * 100, 2)
                last_brk["duration_days"] = int(trough_idx - b_idx)

            if active_support is not None and not is_broken:
                support_lines.append({
                    "start_date": dates[support_start_idx],
                    "end_date": dates[pivot_idx],
                    "price": active_support,
                    "broken": False
                })

            active_support = float(lows[pivot_idx])
            support_start_idx = pivot_idx
            last_pivot_idx = pivot_idx
            is_broken = False
            support_events.append({"idx": pivot_idx, "date": dates[pivot_idx], "price": active_support})

        if active_support is not None and not is_broken and i > support_start_idx:
            if closes[i] < active_support:
                lookback_start = max(0, support_start_idx - 30)
                wave_high = float(np.nanmax(highs[lookback_start: i + 1]))
                wave_low = float(np.nanmin(lows[lookback_start: support_start_idx + 1]))
                gann_50_level = wave_low + (wave_high - wave_low) * 0.50

                gann_broken = closes[i] <= gann_50_level
                obv_weak = np.isfinite(obv_vals[i]) and np.isfinite(obv_ma10_vals[i]) and obv_vals[i] < obv_ma10_vals[i]
                mfi_weak = np.isfinite(mfi_vals[i]) and mfi_vals[i] < 50.0
                flow_confirmed = bool(obv_weak or mfi_weak)

                if require_filter and not (gann_broken or flow_confirmed):
                    continue

                is_broken = True
                confirm_tags = []
                if gann_broken:
                    confirm_tags.append("破江恩50%")
                if obv_weak:
                    confirm_tags.append("OBV走弱")
                if mfi_weak:
                    confirm_tags.append(f"MFI({mfi_vals[i]:.0f})<50")
                confirm_label = "｜".join(confirm_tags) if confirm_tags else "疑似洗盤(資金未明顯流出)"

                breakdown_events.append({
                    "idx": i, "date": dates[i], "price": float(closes[i]), "high": float(highs[i]),
                    "broken_support": round(active_support, 3), "wave_high": round(wave_high, 3),
                    "gann_50": round(gann_50_level, 3), "gann_broken": gann_broken,
                    "flow_confirmed": flow_confirmed, "confirm_label": confirm_label,
                    "end_idx": None, "trough_date": None, "trough_price": None,
                    "post_break_drop_pts": 0.0, "post_break_drop_pct": 0.0,
                    "peak_to_trough_drop_pct": round((closes[i] / wave_high - 1) * 100, 2),
                    "duration_days": 0
                })
                support_lines.append({
                    "start_date": dates[support_start_idx],
                    "end_date": dates[i],
                    "price": active_support,
                    "broken": True
                })

    if breakdown_events and breakdown_events[-1].get("end_idx") is None:
        last_brk = breakdown_events[-1]
        b_idx = last_brk["idx"]
        seg_lows = lows[b_idx: n]
        min_rel_idx = int(np.nanargmin(seg_lows))
        trough_idx = b_idx + min_rel_idx
        trough_price = float(lows[trough_idx])
        last_brk["end_idx"] = trough_idx
        last_brk["trough_date"] = dates[trough_idx]
        last_brk["trough_price"] = round(trough_price, 3)
        last_brk["post_break_drop_pts"] = round(trough_price - last_brk["price"], 2)
        last_brk["post_break_drop_pct"] = round((trough_price / last_brk["price"] - 1) * 100, 2)
        last_brk["peak_to_trough_drop_pct"] = round((trough_price / last_brk["wave_high"] - 1) * 100, 2)
        last_brk["duration_days"] = int(n - 1 - b_idx)

    if active_support is not None and not is_broken:
        support_lines.append({
            "start_date": dates[support_start_idx],
            "end_date": dates[-1],
            "price": active_support,
            "broken": False
        })

    pattern_boxes = []
    for b in range(num_boxes):
        end_idx = n - b * box_period
        start_idx = max(0, end_idx - box_period)
        if end_idx - start_idx < 15:
            break
        sub = df.iloc[start_idx:end_idx]
        b_high = float(sub["high"].max())
        b_low = float(sub["low"].min())
        b_drop_pct = ((b_low - b_high) / b_high * 100) if b_high > 0 else 0.0
        pattern_boxes.append({
            "start_date": sub.index[0], "end_date": sub.index[-1],
            "high": b_high, "low": b_low, "drop_pct": round(b_drop_pct, 2)
        })

    recent_len = min(box_period, n)
    recent_df = df.iloc[-recent_len:]
    cycle_high = float(recent_df["high"].max())
    cycle_low = float(recent_df["low"].min())
    cycle_range = cycle_high - cycle_low
    current_price = float(closes[-1])

    peak_to_trough_drop_pct = ((cycle_low - cycle_high) / cycle_high * 100) if cycle_high > 0 else 0.0
    current_drop_from_high_pct = ((current_price - cycle_high) / cycle_high * 100) if cycle_high > 0 else 0.0
    retracement_ratio_pct = ((cycle_high - current_price) / cycle_range * 100) if cycle_range > 0 else 0.0
    cycle_gann_50 = cycle_low + cycle_range * 0.50
    fib_1236_up = cycle_low + cycle_range * 1.236
    fib_1236_down = cycle_high - cycle_range * 1.236

    active_post_break_drop_pct = 0.0
    if is_broken and breakdown_events:
        latest_brk = breakdown_events[-1]
        active_post_break_drop_pct = round((current_price / latest_brk["price"] - 1) * 100, 2)

    avg_post_break_drop_pct = (
        round(float(np.mean([x["post_break_drop_pct"] for x in breakdown_events])), 2)
        if breakdown_events else 0.0
    )

    obv_now_weak = bool(np.isfinite(obv_vals[-1]) and np.isfinite(obv_ma10_vals[-1]) and obv_vals[-1] < obv_ma10_vals[-1])
    mfi_now = float(mfi_vals[-1]) if np.isfinite(mfi_vals[-1]) else 50.0
    below_gann_50 = current_price < cycle_gann_50

    if active_support is None:
        status_str = "⚪ 尋底觀察中"
    elif is_broken:
        status_str = "🔴 破支持+破江恩50%" if (below_gann_50 and (obv_now_weak or mfi_now < 50)) else "🟠 跌穿10日支持"
    else:
        status_str = "🟢 守穩10日支持"

    stats = {
        "cycle_high": round(cycle_high, 2), "cycle_low": round(cycle_low, 2),
        "cycle_range": round(cycle_range, 2), "peak_to_trough_drop_pct": round(peak_to_trough_drop_pct, 2),
        "current_drop_from_high_pct": round(current_drop_from_high_pct, 2),
        "retracement_pct": round(retracement_ratio_pct, 1),
        "active_support": round(active_support, 3) if active_support else np.nan,
        "active_post_break_drop_pct": active_post_break_drop_pct,
        "avg_post_break_drop_pct": avg_post_break_drop_pct,
        "cycle_gann_50": round(cycle_gann_50, 2), "below_gann_50": below_gann_50,
        "obv_weak": obv_now_weak, "mfi_now": round(mfi_now, 1),
        "fib_1236": round(fib_1236_up, 2), "fib_1236_down": round(max(0.01, fib_1236_down), 2),
        "status": status_str
    }

    return {
        "active_support": round(active_support, 3) if active_support else np.nan,
        "is_broken": is_broken, "status": status_str,
        "support_events": support_events, "breakdown_events": breakdown_events,
        "support_lines": support_lines, "pattern_boxes": pattern_boxes, "stats": stats
    }


# ==================== 動態波段 Fibonacci (Auto Fib) ====================
def af_snapshot(df, left=10, right=5, min_swing_atr=2.0, zone_atr=0.35):
    defaults = {
        "auto_fib_status": "等待波段", "auto_fib_direction": "unavailable",
        "auto_fib_retracement_pct": np.nan, "auto_fib_nearest_ratio": np.nan,
        "auto_fib_nearest_price": np.nan, "auto_fib_distance_atr": np.nan,
        "auto_fib_near_zone": False, "auto_fib_anchor_a": np.nan, "auto_fib_anchor_b": np.nan
    }
    if df is None or len(df) < 40:
        return defaults
    try:
        atr = calc_atr(df, 14)
        highs, lows, closes = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        swings = []
        for i in range(left + right, len(df)):
            p = i - right
            a_val = atr.iloc[i]
            if not np.isfinite(a_val) or a_val <= 0:
                continue
            lo, hi = lows[p - left:i + 1], highs[p - left:i + 1]
            is_low = lows[p] < min(lo[:left]) and lows[p] <= min(lo[left + 1:])
            is_high = highs[p] > max(hi[:left]) and highs[p] >= max(hi[left + 1:])
            if bool(is_low) != bool(is_high):
                kind = "low" if is_low else "high"
                val = float(lows[p] if is_low else highs[p])
                pivot = {"kind": kind, "price": val, "date": df.index[p]}
                if not swings:
                    swings.append(pivot)
                elif swings[-1]["kind"] == kind:
                    if (kind == "low" and val < swings[-1]["price"]) or (kind == "high" and val > swings[-1]["price"]):
                        swings[-1] = pivot
                else:
                    directional = (val > swings[-1]["price"]) if kind == "high" else (val < swings[-1]["price"])
                    if directional and abs(val - swings[-1]["price"]) >= min_swing_atr * a_val:
                        swings.append(pivot)
        if len(swings) >= 2:
            a, b = swings[-2], swings[-1]
            delta = b["price"] - a["price"]
            if delta != 0:
                cur_c = float(closes[-1])
                cur_atr = float(atr.iloc[-1])
                levels = {r: b["price"] - r * delta for r in AUTO_FIB_RATIOS if b["price"] - r * delta > 0}
                near_r = min(levels, key=lambda r: abs(cur_c - levels[r]))
                dist_atr = abs(cur_c - levels[near_r]) / cur_atr if cur_atr > 0 else np.nan
                retr = (b["price"] - cur_c) / delta * 100
                defaults.update({
                    "auto_fib_direction": "up" if delta > 0 else "down",
                    "auto_fib_retracement_pct": round(float(retr), 1),
                    "auto_fib_nearest_ratio": near_r,
                    "auto_fib_nearest_price": round(float(levels[near_r]), 3),
                    "auto_fib_distance_atr": round(float(dist_atr), 2),
                    "auto_fib_near_zone": bool(dist_atr <= zone_atr),
                    "auto_fib_anchor_a": round(a["price"], 3),
                    "auto_fib_anchor_b": round(b["price"], 3),
                    "auto_fib_status": "上升段回調" if delta > 0 else "下跌段反彈"
                })
    except Exception:
        pass
    return defaults


# ==================== 核心升級：🎯 撈底潛力評分引擎 (0-100分) ====================
def calc_bottom_fishing_engine(df, tspr, af_snap, valuation_mos=np.nan, is_value_trap=False):
    """
    綜合 5 大維度計算「撈底潛力分 (0-100)」：
    1. 週線超賣度 (週WR, 週RSI, 週KDJ, 週CCI) - 最高 30 分
    2. 空間回撤與江恩50% / 黃金分割0.618支持共振 - 最高 25 分
    3. 急跌爆量恐慌洗盤後止跌 + OBV/MFI 資金承接 - 最高 25 分
    4. 右側轉勢確認 (週/日 SAR 翻多、MACD底背馳、雙底、TSPR守穩) - 最高 20 分
    5. 估值安全邊際加分 / 價值陷阱扣分
    """
    if df is None or len(df) < 60:
        return {
            "bottom_score": 0.0, "bottom_stage": "⚪ 資料不足", "drop_52w_pct": np.nan,
            "w_rsi": np.nan, "w_wr": np.nan, "w_kdj_j": np.nan, "w_cci": np.nan,
            "w_sar_bull": False, "d_sar_bull": False, "panic_washout": False,
            "bottom_reasons": "日線少於60根"
        }

    close = df["close"]
    price = float(close.iloc[-1])
    high_52w = float(df["high"].iloc[-252:].max()) if len(df) >= 252 else float(df["high"].max())
    low_52w = float(df["low"].iloc[-252:].min()) if len(df) >= 252 else float(df["low"].min())
    drop_52w = (price / high_52w - 1) * 100 if high_52w > 0 else 0.0

    # 1. 週線指標運算 (以週五收市對齊)
    w_df = df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    w_rsi = float(calc_rsi(w_df["close"], 14).iloc[-1]) if len(w_df) >= 15 else 50.0
    w_wr = float(calc_williams_r(w_df, 14).iloc[-1]) if len(w_df) >= 15 else -50.0
    _, _, w_j = calc_kdj(w_df)
    w_j_val = float(w_j.iloc[-1]) if len(w_df) >= 10 and np.isfinite(w_j.iloc[-1]) else 50.0
    w_cci = float(calc_cci(w_df, 20).iloc[-1]) if len(w_df) >= 20 and np.isfinite(calc_cci(w_df, 20).iloc[-1]) else 0.0
    _, w_sar_bull_s = calc_psar(w_df)
    w_sar_bull = bool(w_sar_bull_s.iloc[-1]) if not w_sar_bull_s.empty else False

    # 2. 日線指標與「急跌成交量暴增」事件偵測
    d_rsi = float(calc_rsi(close, 14).iloc[-1])
    mfi = float(calc_mfi(df, 14).iloc[-1])
    obv, obv_ma10 = calc_obv(df)
    obv_strong = bool(obv.iloc[-1] >= obv_ma10.iloc[-1])
    _, d_sar_bull_s = calc_psar(df.iloc[-120:])
    d_sar_bull = bool(d_sar_bull_s.iloc[-1]) if not d_sar_bull_s.empty else False

    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std().replace(0, np.nan)
    vol_z = (df["volume"] - vol_mean) / vol_std
    daily_ret = close.pct_change()
    # 近 12 個交易日內出現單日急跌 > 2.5% 且成交量 Z > 1.6 (恐慌湧出)，且現價未再破近 3 日低點
    panic_days = (daily_ret.iloc[-12:] < -0.025) & (vol_z.iloc[-12:] > 1.6)
    panic_washout = bool(panic_days.any() and price >= float(df["low"].iloc[-3:].min()))

    score = 0.0
    reasons = []

    # --- 維度 1：週線與日線超賣度 (最高 30 分) ---
    if w_wr <= -80:
        score += 12
        reasons.append(f"週WR極超賣({w_wr:.0f})")
    elif w_wr <= -68:
        score += 6

    if w_rsi <= 36 or d_rsi <= 32:
        score += 10
        reasons.append(f"RSI超賣(週{w_rsi:.0f}/日{d_rsi:.0f})")
    elif w_rsi <= 44 or d_rsi <= 38:
        score += 5

    if w_j_val <= 15 or w_cci <= -100:
        score += 8
        reasons.append(f"週KDJ_J({w_j_val:.0f})/CCI({w_cci:.0f})見底區")

    # --- 維度 2：空間回撤 + 江恩/黃金分割共振 (最高 25 分) ---
    if -50 <= drop_52w <= -18:
        score += 12
        reasons.append(f"高點回撤{drop_52w:.1f}%")
    elif drop_52w < -50:
        score += 7

    gann_50 = low_52w + (high_52w - low_52w) * 0.50
    gann_625 = high_52w - (high_52w - low_52w) * 0.625
    near_gann = (abs(price - gann_50) / price <= 0.035) or (abs(price - gann_625) / price <= 0.035)
    near_fib = bool(af_snap.get("auto_fib_near_zone")) and af_snap.get("auto_fib_nearest_ratio") in (0.5, 0.618, 0.786)
    if near_gann or near_fib:
        score += 13
        reasons.append("處於江恩50%/62.5%或Fib關鍵位")

    # --- 維度 3：急跌爆量洗盤 + OBV/MFI 資金承接 (最高 25 分) ---
    if panic_washout:
        score += 12
        reasons.append("🔥急跌爆量洗盤後止跌")
    if mfi <= 28:
        score += 8
        reasons.append(f"MFI超賣({mfi:.0f})")
    if obv_strong:
        score += 10 if mfi >= 42 else 5
        reasons.append("OBV站上MA10承接")

    # --- 維度 4：右側形態與結構確認 (最高 20 分) ---
    has_div = detect_macd_bullish_divergence(df)
    has_db = detect_double_bottom(df)
    if has_div or has_db:
        score += 10
        reasons.append("MACD底背馳/雙底")

    tspr_ok = not tspr.get("is_broken", True) and np.isfinite(safe_float(tspr.get("active_support")))
    if tspr_ok:
        score += 5
        reasons.append("守穩TSPR 10日支持")
    if d_sar_bull or w_sar_bull:
        score += 5
        reasons.append("SAR翻多" if d_sar_bull and not w_sar_bull else "週線SAR多頭")

    # --- 維度 5：估值安全邊際與基本面過濾 ---
    if np.isfinite(valuation_mos) and valuation_mos >= 0.15:
        score += 8
        reasons.append(f"估值折讓{valuation_mos*100:.0f}%")
    if is_value_trap:
        score -= 18
        reasons.append("⚠️基本面紅旗扣分")

    final_score = round(float(np.clip(score, 0, 100)), 1)

    if is_value_trap and final_score >= 55:
        stage = "🟠 技術超賣 (留心基本面)"
    elif final_score >= 68 and (tspr_ok or obv_strong or d_sar_bull or has_div):
        stage = "🟢 右側確認 (優先撈底)"
    elif final_score >= 52:
        stage = "🟡 左側超賣 (分批候位)"
    else:
        stage = "⚪ 未達撈底標準"

    return {
        "bottom_score": final_score,
        "bottom_stage": stage,
        "drop_52w_pct": round(drop_52w, 1),
        "w_rsi": round(w_rsi, 1),
        "w_wr": round(w_wr, 1),
        "w_kdj_j": round(w_j_val, 1),
        "w_cci": round(w_cci, 1),
        "w_sar_bull": w_sar_bull,
        "d_sar_bull": d_sar_bull,
        "panic_washout": panic_washout,
        "bottom_reasons": "｜".join(reasons) if reasons else "暫無明顯見底訊號"
    }


# ==================== 三引擎：估值 / 下行風險 / 入場決策 ====================
def classify_company(ticker, info, sector):
    quote_type = str(info.get("quote_type") or "").upper()
    if ticker in ETF_TICKERS or quote_type in {"ETF", "MUTUALFUND", "INDEX"} or ticker.startswith("^") or sector == "ETF":
        return "ETF／指數"
    s = f"{sector} {info.get('industry') or ''}".lower()
    if any(k in s for k in FINANCIAL_KEYWORDS):
        return "金融"
    if sector in CYCLICAL_SECTORS:
        return "周期"
    eps = safe_float(info.get("eps_quote"))
    if np.isfinite(eps) and eps <= 0:
        return "虧損"
    return "一般盈利"


def quality_engine(info, company_type):
    flags, blockers, trap_points = [], [], 0
    roe = safe_float(info.get("roe"))
    debt = safe_float(info.get("de_ratio"))
    rev = safe_float(info.get("rev_growth"))
    earn = safe_float(info.get("earnings_growth"))
    fcf = safe_float(info.get("free_cashflow"))
    bvps = safe_float(info.get("bvps_quote"))

    if np.isfinite(bvps) and bvps <= 0:
        blockers.append("每股淨資產≤0")
    if np.isfinite(roe) and roe < 0.05:
        flags.append(f"ROE偏低({roe*100:.1f}%)")
        trap_points += 1
    if np.isfinite(rev) and rev < -0.10:
        flags.append(f"營收下跌({rev*100:.1f}%)")
        trap_points += 1
    if np.isfinite(earn) and earn < -0.30:
        flags.append(f"盈利倒退({earn*100:.1f}%)")
        trap_points += 1
    if company_type != "金融":
        if np.isfinite(fcf) and fcf < 0:
            flags.append("自由現金流為負")
            trap_points += 1
        if np.isfinite(debt) and debt > 220:
            flags.append(f"高負債({debt:.0f}%)")
            trap_points += 1
    if np.isfinite(roe) and roe < 0 and np.isfinite(rev) and rev < -0.15:
        blockers.append("盈利與營收雙雙惡化")
    return {
        "flags": flags or ["基本面正常"],
        "blockers": blockers,
        "value_trap": trap_points >= 2,
        "trap_points": trap_points
    }


def valuation_engine(ticker, info, sector, price):
    ctype = classify_company(ticker, info, sector)
    hk = ticker.endswith(".HK")
    out = {
        "company_type": ctype, "methods": [], "assumptions": [],
        "fair_low": np.nan, "fair_base": np.nan, "fair_high": np.nan,
        "margin_of_safety": np.nan, "confidence": "中", "val_level": "⚪ 估值中性",
        "normalized_eps": np.nan
    }
    if not np.isfinite(price) or price <= 0 or ctype in {"ETF／指數", "虧損"}:
        out["val_level"] = f"⚪ {ctype}(不套用PE估值)"
        return out

    disc = HK_MULTIPLE_DISCOUNT if hk else 1.0
    methods = []
    eps_ttm = safe_float(info.get("eps_quote"))
    eps_fwd = safe_float(info.get("forward_eps_quote"))
    candidates = [x for x in [eps_ttm, eps_fwd] if np.isfinite(x) and x > 0]
    norm_eps = float(np.median(candidates)) if candidates else np.nan
    out["normalized_eps"] = norm_eps

    if ctype != "金融" and np.isfinite(norm_eps) and norm_eps > 0:
        lo, base, hi = [m * disc for m in SECTOR_PE_RANGES.get(sector, DEFAULT_PE_RANGE)]
        methods.append(("正常化EPS×行業PE", norm_eps * lo, norm_eps * base, norm_eps * hi))
        out["assumptions"].append(f"EPS {norm_eps:.2f} × PE({lo:.0f}/{base:.0f}/{hi:.0f})")

    fcf_ps = safe_float(info.get("fcf_ps_quote"))
    if ctype != "金融" and np.isfinite(fcf_ps) and fcf_ps > 0:
        add = 0.01 if hk else 0.0
        y = [r + add for r in FCF_REQUIRED_YIELD]
        methods.append(("FCF收益率法", fcf_ps / y[0], fcf_ps / y[1], fcf_ps / y[2]))
        out["assumptions"].append(f"每股FCF {fcf_ps:.2f}")

    bvps, roe = safe_float(info.get("bvps_quote")), safe_float(info.get("roe"))
    if ctype == "金融" and np.isfinite(bvps) and bvps > 0:
        roe_c = float(np.clip(roe if np.isfinite(roe) else 0.10, 0.05, 0.22))
        add = 0.01 if hk else 0.0
        pbs = [max(0.4, (roe_c - LONG_TERM_GROWTH) / (r + add - LONG_TERM_GROWTH)) for r in COST_OF_EQUITY]
        methods.append(("ROE合理PB", bvps * pbs[0], bvps * pbs[1], bvps * pbs[2]))
        out["assumptions"].append(f"BVPS {bvps:.2f}｜ROE {roe_c*100:.1f}%")
    elif sector == "Real Estate" and np.isfinite(bvps) and bvps > 0:
        methods.append(("地產PB折讓", bvps * 0.45, bvps * 0.65, bvps * 0.85))

    methods = [m for m in methods if all(np.isfinite(v) and v > 0 for v in m[1:])]
    if not methods:
        out["val_level"] = "⚪ 基本面待更新(純技術評分)"
        out["confidence"] = "低"
        return out

    out["methods"] = [m[0] for m in methods]
    out["fair_low"] = float(np.median([m[1] for m in methods]))
    out["fair_base"] = float(np.median([m[2] for m in methods]))
    out["fair_high"] = float(np.median([m[3] for m in methods]))
    mos = 1.0 - price / out["fair_base"]
    out["margin_of_safety"] = mos
    out["confidence"] = "高" if len(methods) >= 2 else "中"

    if mos >= 0.30:
        out["val_level"] = "🟢 重度低估"
    elif mos >= 0.15:
        out["val_level"] = "🟠 中度低估"
    elif mos >= 0.05:
        out["val_level"] = "🟡 輕度低估"
    elif mos >= -0.12:
        out["val_level"] = "⚪ 估值合理"
    else:
        out["val_level"] = "🔴 估值偏高"
    return out


def forward_min_drawdown(df, horizon):
    future_min = df["low"].rolling(horizon, min_periods=horizon).min().shift(-horizon)
    return (1 - future_min / df["close"]).clip(lower=0)


def downside_engine(df, tspr):
    price = safe_float(df["close"].iloc[-1])
    atr = safe_float(calc_atr(df).iloc[-1])
    dd20 = forward_min_drawdown(df, 20).dropna()
    dd60 = forward_min_drawdown(df, 60).dropna()

    p80_20 = float(dd20.quantile(0.80)) if len(dd20) >= 40 else np.nan
    p80_60 = float(dd60.quantile(0.80)) if len(dd60) >= 60 else np.nan
    med_20 = float(dd20.median()) if len(dd20) >= 40 else np.nan
    med_60 = float(dd60.median()) if len(dd60) >= 60 else np.nan

    tspr_sup = safe_float(tspr.get("active_support"))
    recent_low = safe_float(df["low"].iloc[-20:].min())
    if np.isfinite(tspr_sup) and not tspr.get("is_broken") and tspr_sup < price:
        stop = max(tspr_sup - 0.35 * atr, price - 2.5 * atr)
    else:
        stop = min(recent_low, price - 2.0 * atr) if np.isfinite(atr) else price * 0.92

    level = "低" if (np.isfinite(p80_60) and p80_60 < 0.14) else ("中" if (np.isfinite(p80_60) and p80_60 < 0.25) else "高")
    return {
        "invalidation": max(0.01, stop),
        "dd20_med": med_20, "dd20_p80": p80_20,
        "dd60_med": med_60, "dd60_p80": p80_60,
        "downside_level": level
    }


def analyze_single_stock(ticker, df, info, benchmark_df=None, regime_name="neutral"):
    if df is None or len(df) < 60:
        return None
    price = safe_float(df["close"].iloc[-1])
    sector = info.get("sector") or SECTOR_MAP_FALLBACK.get(ticker, "OTHER")
    tspr = calc_tspr_system(df.iloc[-450:], left_bars=10, right_bars=5, box_period=65)
    af_snap = af_snapshot(df.iloc[-300:])
    valuation = valuation_engine(ticker, info, sector, price)
    quality = quality_engine(info, valuation["company_type"])
    downside = downside_engine(df, tspr)

    mos = valuation["margin_of_safety"]
    bottom = calc_bottom_fishing_engine(df, tspr, af_snap, valuation_mos=mos, is_value_trap=quality["value_trap"])

    stop = downside["invalidation"]
    risk_per_share = max(price - stop, price * 0.02)
    target_2r = price + 2.0 * risk_per_share
    target_fair = valuation["fair_base"] if (np.isfinite(valuation["fair_base"]) and valuation["fair_base"] > price) else target_2r
    rr = (target_fair - price) / risk_per_share if risk_per_share > 0 else np.nan

    # 綜合決策標籤（兼顧撈底技術面與基本面）
    if quality["blockers"]:
        decision = "排除"
    elif bottom["bottom_stage"].startswith("🟢"):
        decision = "🟢 右側確認可撈"
    elif bottom["bottom_stage"].startswith("🟡") or (np.isfinite(mos) and mos >= 0.20):
        decision = "🟡 左側超賣/低估候位"
    elif quality["value_trap"]:
        decision = "🟠 價值陷阱觀察"
    else:
        decision = "⚪ 觀望/未到價"

    pct = lambda x: round(x * 100, 1) if np.isfinite(safe_float(x)) else np.nan
    r2 = lambda x: round(x, 2) if np.isfinite(safe_float(x)) else np.nan

    return {
        "ticker": ticker,
        "name": info.get("name", ticker),
        "sector": sector,
        "company_type": valuation["company_type"],
        "price": r2(price),
        "last_bar": str(pd.Timestamp(df.index[-1]).date()),
        # 撈底核心欄位
        "bottom_score": bottom["bottom_score"],
        "bottom_stage": bottom["bottom_stage"],
        "drop_52w_pct": bottom["drop_52w_pct"],
        "w_rsi": bottom["w_rsi"],
        "w_wr": bottom["w_wr"],
        "w_kdj_j": bottom["w_kdj_j"],
        "w_cci": bottom["w_cci"],
        "w_sar_bull": bottom["w_sar_bull"],
        "d_sar_bull": bottom["d_sar_bull"],
        "panic_washout": bottom["panic_washout"],
        "bottom_reasons": bottom["bottom_reasons"],
        # 估值與品質欄位
        "val_level": valuation["val_level"],
        "mos_pct": pct(mos),
        "fair_low": r2(valuation["fair_low"]),
        "fair_base": r2(valuation["fair_base"]),
        "fair_high": r2(valuation["fair_high"]),
        "methods": "、".join(valuation["methods"]) or "技術面主導",
        "assumptions": "｜".join(valuation["assumptions"]) or "—",
        "quality_flags": "｜".join(quality["flags"]),
        "value_trap": quality["value_trap"],
        # 下行與交易計劃欄位
        "downside_level": downside["downside_level"],
        "dd20_p80": pct(downside["dd20_p80"]),
        "dd60_p80": pct(downside["dd60_p80"]),
        "stop_loss": r2(stop),
        "target_2r": r2(target_2r),
        "target_fair": r2(target_fair),
        "reward_risk": r2(rr),
        "tspr_status": tspr["status"],
        "tspr_support": r2(tspr["active_support"]),
        "gann_50": r2(tspr["stats"].get("cycle_gann_50")),
        "auto_fib_ratio": af_snap.get("auto_fib_nearest_ratio"),
        "auto_fib_price": af_snap.get("auto_fib_nearest_price"),
        "decision": decision,
        "final_score": bottom["bottom_score"],
        "regime": regime_name,
        "confirmation_price": r2(safe_float(df["high"].iloc[-5:].max()))
    }


def save_snapshot_and_signal(result):
    try:
        payload = pd.Series(result).to_json(force_ascii=False)
        today = result["last_bar"]
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scan_snapshots (snapshot_date,ticker,model_version,payload_json,created_at) VALUES (?,?,?,?,?)",
                (today, result["ticker"], MODEL_VERSION, payload, hkt_now().isoformat())
            )
            if result["decision"] == "🟢 右側確認可撈":
                conn.execute(
                    "INSERT OR IGNORE INTO signals (signal_date,ticker,model_version,decision,total_score,price,confirmation_price,stop_loss,target_2r,market_regime,sector,features_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (today, result["ticker"], MODEL_VERSION, result["decision"], result["bottom_score"], result["price"],
                     result["confirmation_price"], result["stop_loss"], result["target_2r"], result["regime"], result["sector"], payload, hkt_now().isoformat())
                )
    except Exception as exc:
        add_error("儲存訊號失敗", result.get("ticker"), exc)


# ==================== 宏觀與回測輔助 ====================
@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def fetch_macro():
    result = {}
    frames = fetch_multiple(list(MACRO_TICKERS.values()), period="1y")
    for name, ticker in MACRO_TICKERS.items():
        df = frames.get(ticker)
        if df is None or len(df) < 5:
            continue
        close = safe_float(df["close"].iloc[-1])
        prev = safe_float(df["close"].iloc[-2])
        result[name] = {
            "val": close,
            "chg": (close / prev - 1) * 100 if prev > 0 else np.nan,
            "last_bar": str(df.index[-1].date())
        }
    return result


def calculate_position(price, stop_loss, account_size, risk_pct, max_notional_pct=0.15):
    risk_amount = account_size * risk_pct
    risk_per_share = price - stop_loss
    if not np.isfinite(stop_loss) or stop_loss <= 0 or risk_per_share <= 0 or price <= 0:
        return 0, 0, risk_amount
    by_risk = int(risk_amount / risk_per_share)
    by_notional = int((account_size * max_notional_pct) / price)
    shares = max(0, min(by_risk, by_notional))
    return shares, shares * price, risk_amount


def run_walk_forward_backtest(signal_df, hold_days=20, costs_bps=TRANSACTION_COST_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS):
    rows = []
    for _, signal in signal_df.iterrows():
        ticker = signal["ticker"]
        base = {"代碼": ticker, "信號日": str(signal["signal_date"]), "模型": signal.get("model_version", "")}
        df = fetch_ohlcv(ticker, period="5y")
        if df is None or df.empty:
            rows.append({**base, "結果": "無價格資料", "納入統計": False})
            continue
        idx = pd.DatetimeIndex(df.index).normalize()
        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        future = np.where(idx > signal_date)[0]
        if len(future) == 0:
            rows.append({**base, "結果": "未成熟(尚無下一交易日)", "納入統計": False})
            continue
        entry_i = int(future[0])
        last_i = min(entry_i + hold_days - 1, len(df) - 1)
        if last_i <= entry_i:
            rows.append({**base, "結果": "持有期未滿", "納入統計": False})
            continue
        stop, target = safe_float(signal["stop_loss"]), safe_float(signal["target_2r"])
        raw_entry = safe_float(df["open"].iloc[entry_i])
        if not (np.isfinite(stop) and np.isfinite(target) and stop > 0) or raw_entry <= stop:
            rows.append({**base, "結果": "開市已穿止損(取消)", "納入統計": False})
            continue
        entry_fill = raw_entry * (1 + slippage_bps / 10000)
        exit_price, exit_i, outcome = safe_float(df["close"].iloc[last_i]), last_i, "到期出場"
        for i in range(entry_i, last_i + 1):
            low, high = safe_float(df["low"].iloc[i]), safe_float(df["high"].iloc[i])
            if low <= stop:
                exit_price, exit_i, outcome = stop, i, "觸及止損"
                break
            if high >= target:
                exit_price, exit_i, outcome = target, i, "達到2R目標"
                break
        c = costs_bps / 10000
        net = (exit_price * (1 - c) - entry_fill * (1 + c)) / (entry_fill * (1 + c)) * 100
        rows.append({
            **base, "進場價": round(entry_fill, 2), "出場價": round(exit_price, 2),
            "持有日": exit_i - entry_i + 1, "結果": outcome, "淨回報%": round(net, 2), "納入統計": True
        })
    return pd.DataFrame(rows)
