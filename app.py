import os
import time
import warnings
import threading
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
# 基本設定
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
SIGNAL_LOG_FILE = "signal_log.csv"
MAX_WORKERS_DATA = 6
MAX_WORKERS_SCORE = 4
FUTU_LOCK = threading.RLock()

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
DROP_LEVELS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]

# 當 Yahoo sector 無資料時的備援分類
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
    """集中紀錄錯誤，不再靜默吞掉失敗個案。"""
    message = f"{context}"
    if ticker:
        message += f"｜{ticker}"
    if exc:
        message += f"｜{type(exc).__name__}: {str(exc)[:160]}"
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
    if not FUTU_AVAILABLE:
        return None
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
    if ticker.endswith(".HK"):
        return f"HK.{ticker[:-3]}"
    if ticker.replace("-", "").isalpha():
        return f"US.{ticker}"
    return ticker


@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    """優先富途、失敗轉 Yahoo；所有富途呼叫由鎖保護。"""
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
        if df is None or df.empty:
            add_error("Yahoo OHLCV回傳空數據", ticker)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            add_error("Yahoo OHLCV缺少必要欄位", ticker)
            return None
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
            try:
                results[ticker] = future.result()
            except Exception as exc:
                add_error("平行OHLCV工作失敗", ticker, exc)
                results[ticker] = None
    return results


@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    """一次請求整合基本面與5年收市價，避免同一股票重複呼叫 Yahoo。"""
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
                pe = row.get("pe_ratio")
                pb = row.get("pb_ratio")
                if pd.notna(pe):
                    result["pe"] = float(pe)
                    result["info_source"] = "富途+Yahoo"
                if pd.notna(pb):
                    result["pb"] = float(pb)
                    result["info_source"] = "富途+Yahoo"
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
                eps = info.get("trailingEps")
                price = info.get("currentPrice")
                if eps and price and eps > 0:
                    pe = price / eps
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
        else:
            add_error("Yahoo 5年歷史股價為空", ticker)
    except Exception as exc:
        add_error("Yahoo 基本面/歷史數據失敗", ticker, exc)

    return result


def get_sector_from_info(ticker, info=None):
    info = info or get_full_stock_info(ticker)
    sector = info.get("sector")
    if sector:
        return str(sector)
    return SECTOR_MAP_FALLBACK.get(ticker, "OTHER")


def get_futu_capital_flow(ticker):
    if not quote_ctx:
        return None
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
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clip_score(value):
    return float(np.clip(value, 0, 100))


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_kdj(df, n=9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
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


def calc_obv(df):
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


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
    multiplier = ((2 * df["close"] - df["low"] - df["high"]) /
                  (df["high"] - df["low"]).replace(0, np.nan))
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


def drop_levels(high_price):
    return {f"-{int(d * 100)}%": round(high_price * (1 - d), 3) for d in DROP_LEVELS}


def volume_zscore(df, period=20):
    vol = df["volume"]
    mean = safe_float(vol.rolling(period).mean().iloc[-1], 0)
    std = safe_float(vol.rolling(period).std().iloc[-1], 0)
    return (safe_float(vol.iloc[-1], 0) - mean) / std if std > 0 else 0.0


# 平滑化評分：消除舊版臨界值造成的分數跳動/訊號閃爍
def smooth_low_score(value, low, high, max_score):
    """value 越低分越高。value <= low 時滿分；value >= high 時0分。"""
    if pd.isna(value):
        return 0.0
    if value <= low:
        return float(max_score)
    if value >= high:
        return 0.0
    return float(max_score * (high - value) / (high - low))


def smooth_high_score(value, low, high, max_score):
    """value 越高分越高。"""
    if pd.isna(value):
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return float(max_score)
    return float(max_score * (value - low) / (high - low))


def time_decay_oversold(indicator_series, threshold, days_back=5):
    weight_sum = 0.0
    for i in range(min(days_back, len(indicator_series))):
        val = safe_float(indicator_series.iloc[-1 - i], np.nan)
        if np.isnan(val):
            continue
        weight = 1 - i / (days_back + 1)
        if val < threshold:
            weight_sum += weight
    return weight_sum

# ─────────────────────────────────────────────────────────────
# 技術形態：加入低點距離、反彈幅度與局部極值確認
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
    """雙底：兩個局部低點需相隔足夠時間、價差接近、中間反彈足夠。"""
    if df is None or len(df) < lookback:
        return False
    recent = df.iloc[-lookback:]
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, order=3)
    if len(candidates) < 2:
        return False

    for first_i in candidates:
        for second_i in candidates:
            gap = second_i - first_i
            if gap < min_gap or gap > max_gap:
                continue
            low1, low2 = safe_float(lows.iloc[first_i]), safe_float(lows.iloc[second_i])
            if low1 <= 0 or low2 <= 0:
                continue
            if abs(low2 - low1) / min(low1, low2) > tolerance:
                continue
            middle_high = safe_float(recent["high"].iloc[first_i:second_i + 1].max(), np.nan)
            rebound = (middle_high - max(low1, low2)) / max(low1, low2) if middle_high else 0
            if rebound >= min_rebound:
                return True
    return False


def detect_macd_bullish_divergence(df, lookback=90, min_gap=10, max_gap=60):
    """底背離：兩個價格局部低點需相隔10-60日；後低更低、MACD柱更高。"""
    if df is None or len(df) < lookback:
        return False
    recent = df.iloc[-lookback:].copy()
    _, _, hist = calc_macd(recent["close"])
    lows = recent["low"].reset_index(drop=True)
    candidates = local_minima_indices(lows, order=3)
    if len(candidates) < 2:
        return False

    for first_i in candidates:
        for second_i in candidates:
            gap = second_i - first_i
            if gap < min_gap or gap > max_gap:
                continue
            first_low = safe_float(lows.iloc[first_i])
            second_low = safe_float(lows.iloc[second_i])
            first_hist = safe_float(hist.iloc[first_i])
            second_hist = safe_float(hist.iloc[second_i])
            if np.isnan(first_hist) or np.isnan(second_hist):
                continue
            # 價格至少低1%，MACD柱至少改善，降低雜訊誤判
            if second_low < first_low * 0.99 and second_hist > first_hist:
                return True
    return False

# ─────────────────────────────────────────────────────────────
# 評分與估值
# ─────────────────────────────────────────────────────────────
def score_stock(df, market_state="neutral"):
    if df is None or len(df) < 60:
        return 0, 0, [], 0, "無", 0, 0, 0

    close = df["close"]
    volume = df["volume"]
    rsi_d = calc_rsi(close, 14)
    rsi_w = calc_rsi(close, 70)
    k, d, _ = calc_kdj(df)
    macd, sig, _ = calc_macd(close)
    cci = calc_cci(df)
    obv = calc_obv(df)
    wr = calc_wr(df)
    cmf = calc_cmf(df)
    vwap = calc_vwap(df)
    sma20 = close.rolling(20).mean()
    sma200 = close.rolling(200).mean()

    rsi_val = safe_float(rsi_d.iloc[-1], 50)
    rsi_w_val = safe_float(rsi_w.iloc[-1], 50)
    k_val = safe_float(k.iloc[-1], 50)
    d_val = safe_float(d.iloc[-1], 50)
    cci_val = safe_float(cci.iloc[-1], 0)
    wr_val = safe_float(wr.iloc[-1], -50)
    macd_val = safe_float(macd.iloc[-1], 0)
    sig_val = safe_float(sig.iloc[-1], 0)
    cmf_val = safe_float(cmf.iloc[-1], 0)
    vwap_val = safe_float(vwap.iloc[-1], safe_float(close.iloc[-1], 0))
    close_v = safe_float(close.iloc[-1], 0)
    obv_now = safe_float(obv.iloc[-1], 0)
    obv_prev = safe_float(obv.iloc[-6], obv_now) if len(obv) >= 6 else obv_now
    vol_z = volume_zscore(df)

    # 平滑技術分數
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
    if oversold_count >= 3:
        resonance, mult = "強", 1.25
    elif oversold_count == 2:
        resonance, mult = "中", 1.10
    elif oversold_count == 1:
        resonance, mult = "弱", 1.00
    else:
        resonance, mult = "無", 0.85

    vol_confirm = 1.0
    open_v = safe_float(df["open"].iloc[-1], close_v)
    sma20_v = safe_float(sma20.iloc[-1], close_v)
    if vol_z > 2.0 and close_v > vwap_val:
        vol_confirm = 1.25
    elif vol_z > 2.0 and close_v > open_v:
        vol_confirm = 1.15
    elif vol_z < -1.5 and close_v < sma20_v:
        vol_confirm = 0.75

    short_score = (rsi_score + kdj_score + cci_score + wr_score) * mult * vol_confirm
    signals = []

    if macd_val > sig_val and macd_val < 0:
        short_score += 8 * vol_confirm
        signals.append("MACD低位金叉")
    if obv_now > obv_prev and close_v <= safe_float(close.iloc[-6], close_v):
        short_score += 8 * vol_confirm
        signals.append("OBV底背離")
    if cmf_val > 0.10:
        short_score += 5
        signals.append("💰CMF吸籌")
    elif cmf_val < -0.20:
        signals.append("⚠️CMF派發")
    if close_v > vwap_val and vol_z > 1.5:
        short_score += 3
    if vol_z > 2.5:
        signals.append(f"🔥爆量(Z={vol_z:.1f})")
    elif vol_z > 1.5:
        signals.append(f"📈放量(Z={vol_z:.1f})")

    if detect_double_bottom(df):
        short_score += 10
        signals.append("🕳️確認雙底")
    if detect_macd_bullish_divergence(df):
        short_score += 12
        signals.append("📉MACD底背離(確認)")

    mid_score = 0.0
    mid_signals = []
    bias200 = 0.0
    sma200_v = safe_float(sma200.iloc[-1], np.nan)
    if not np.isnan(sma200_v) and sma200_v > 0:
        bias200 = (close_v - sma200_v) / sma200_v * 100

    mid_score += smooth_low_score(rsi_w_val, 25, 55, 30)
    mid_score += smooth_low_score(bias200, -35, -3, 35)
    mid_score += smooth_low_score(cci_val, -220, -60, 15)

    if rsi_w_val < 35:
        mid_signals.append("周RSI超賣")
    if bias200 < -15:
        mid_signals.append("年線乖離偏大")
    if cci_val < -150:
        mid_signals.append("CCI極度超賣")
    if rsi_w_val > 60:
        mid_score *= 0.70
        mid_signals.append("⚠️周線仍強(小心假底)")

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
    if short_score >= 70 or mid_score >= 70:
        return "🔥 強烈撈底", "buy"
    if short_score >= 50 or mid_score >= 50:
        return "⭐️ 值得關注", "watch"
    if short_score >= 35 or mid_score >= 35:
        return "👁️ 觀察中", "observe"
    return "—", "none"


def signal_badge(label):
    if label.startswith("🔥"):
        return "badge-buy"
    if label.startswith("⭐️"):
        return "badge-watch"
    if label.startswith("👁️"):
        return "badge-observe"
    return "badge-none"


def get_pe_percentile_from_info(info):
    """近似PE百分位。以目前EPS回推，需清晰標示限制。"""
    eps = safe_float(info.get("eps"), np.nan)
    hist = info.get("hist_5y")
    if np.isnan(eps) or eps <= 0 or hist is None or len(hist) < 30:
        return None, None
    series = (hist / eps).dropna()
    if len(series) < 30:
        return None, None
    current_calc_pe = safe_float(series.iloc[-1], np.nan)
    percentile = float((series < current_calc_pe).mean())
    return percentile, current_calc_pe


def get_pb_percentile_from_info(info):
    bv = safe_float(info.get("book_value"), np.nan)
    hist = info.get("hist_5y")
    if np.isnan(bv) or bv <= 0 or hist is None or len(hist) < 30:
        return None, None
    series = (hist / bv).dropna()
    if len(series) < 30:
        return None, None
    current_calc_pb = safe_float(series.iloc[-1], np.nan)
    percentile = float((series < current_calc_pb).mean())
    return percentile, current_calc_pb


def percentile_value_score(percentile, fallback_value, thresholds, is_lower_better=True):
    """百分位可用時優先。不可用時才用絕對估值門檻。"""
    if percentile is not None:
        if percentile < 0.10:
            return 90.0
        if percentile < 0.25:
            return 70.0
        if percentile < 0.50:
            return 40.0
        return 10.0
    if fallback_value is None or fallback_value <= 0:
        return None
    a, b, c = thresholds
    if fallback_value < a:
        return 90.0
    if fallback_value < b:
        return 70.0
    if fallback_value < c:
        return 40.0
    return 10.0


def quality_filter(roe, de_ratio, rev_growth):
    """品質警示與估值懲罰，最低保留50%避免過度壓縮。"""
    flags = []
    penalty = 1.0
    roe = safe_float(roe, np.nan)
    de_ratio = safe_float(de_ratio, np.nan)
    rev_growth = safe_float(rev_growth, np.nan)

    if not np.isnan(roe) and roe < 0.05:
        flags.append("⚠️ ROE過低(<5%)")
        penalty -= 0.15
    if not np.isnan(de_ratio) and de_ratio > 150:
        flags.append("⚠️ 負債比過高(>150%)")
        penalty -= 0.15
    if not np.isnan(rev_growth) and rev_growth < -0.10:
        flags.append("⚠️ 營收衰退(<-10%)")
        penalty -= 0.10
    if not flags:
        flags.append("✅ 品質過關")
    return flags, max(0.50, penalty)


def dividend_bonus(div_yield, sector):
    """股息率只作有限加分，避免高息陷阱主導分數。"""
    y = safe_float(div_yield, np.nan)
    if np.isnan(y) or y <= 0:
        return 0.0, "股息率 N/A"
    pct = y * 100 if y <= 1 else y
    # 金融/公用/地產以股息率較有參考價值，科技股加分較低
    income_sector = any(x in sector.lower() for x in ["bank", "util", "real estate", "financial", "hk_bank", "hk_util", "hk_property"])
    cap = 12 if income_sector else 6
    bonus = smooth_high_score(pct, 2.0, 7.0, cap)
    return round(bonus, 1), f"股息率 {pct:.2f}% (+{bonus:.1f})"


def valuation_label(score):
    if score >= 70:
        return "💰 便宜"
    if score >= 40:
        return "😐 合理"
    return "🔥 偏貴"


def build_sector_peer_cache(scan_list, info_map):
    """建立 {sector: [(ticker, pe), ...]}，保留ticker以在比較時排除自己。"""
    cache = {}
    for ticker in scan_list:
        info = info_map.get(ticker, {})
        sector = get_sector_from_info(ticker, info)
        pe = safe_float(info.get("pe"), np.nan)
        if not np.isnan(pe) and pe > 0:
            cache.setdefault(sector, []).append((ticker, pe))
    return cache


def sector_relative_valuation(ticker, pe, sector, peer_cache):
    """必定排除自己，且需至少2個真正同業才顯示。"""
    if pe is None or pe <= 0 or sector == "OTHER":
        return None
    peer_pes = [
        p for peer_ticker, p in peer_cache.get(sector, [])
        if peer_ticker != ticker and p is not None and p > 0
    ]
    if len(peer_pes) < 2:
        return None
    median_pe = float(np.median(peer_pes))
    rel_pct = (pe - median_pe) / median_pe * 100 if median_pe > 0 else np.nan
    return {
        "sector": sector,
        "median_pe": round(median_pe, 1),
        "rel_pct": round(float(rel_pct), 1),
        "n_peers": len(peer_pes)
    }


def fund_flow_detail(df):
    if df is None or len(df) < 20:
        return 0.0, {}
    close, volume = df["close"], df["volume"]
    mfi_series = calc_mfi(df)
    mfi_now = safe_float(mfi_series.iloc[-1], 50)
    ret = close.pct_change()
    big_down = ret < -0.02
    avg_vol = volume.rolling(20).mean()
    ratios = (volume[big_down] / avg_vol[big_down]).replace([np.inf, -np.inf], np.nan).dropna().tail(5)
    down_ratio = safe_float(ratios.mean(), 0) if not ratios.empty else 0

    detail = {}
    down_score = 0.0
    if 0 < down_ratio < 0.8:
        down_score = 30
        detail["大跌日縮量"] = (30, f"量比{down_ratio:.2f}")
    elif 0 < down_ratio < 1.1:
        down_score = 15
        detail["大跌日量比正常"] = (15, f"量比{down_ratio:.2f}")
    else:
        detail["大跌日放量"] = (0, f"量比{down_ratio:.2f}")

    mfi_score = smooth_low_score(mfi_now, 15, 50, 40)
    detail["MFI"] = (round(mfi_score, 1), f"{mfi_now:.1f}")
    trend_score = 0.0
    if len(mfi_series) >= 10:
        start = safe_float(mfi_series.iloc[-10], mfi_now)
        change = mfi_now - start
        if change > 5:
            trend_score = min(10.0, change)
            detail["MFI近期回升"] = (round(trend_score, 1), f"+{change:.1f}")
        elif change < -5:
            detail["MFI近期下降"] = (0, f"{change:.1f}")

    return clip_score(down_score + mfi_score + trend_score), detail


def technical_detail_score(df):
    if df is None or len(df) < 60:
        return 0.0, {}
    rsi = safe_float(calc_rsi(df["close"]).iloc[-1], 50)
    k, d, _ = calc_kdj(df)
    k_val = safe_float(k.iloc[-1], 50)
    d_val = safe_float(d.iloc[-1], 50)
    cci = safe_float(calc_cci(df).iloc[-1], 0)
    wr = safe_float(calc_wr(df).iloc[-1], -50)

    scores = {
        "RSI(14)": (smooth_low_score(rsi, 20, 45, 25), f"{rsi:.1f}"),
        "KDJ": (smooth_low_score((k_val + d_val) / 2, 10, 40, 25), f"K={k_val:.1f}, D={d_val:.1f}"),
        "CCI": (smooth_low_score(cci, -200, -40, 25), f"{cci:.1f}"),
        "W%R": (smooth_low_score(wr, -95, -50, 25), f"{wr:.1f}")
    }
    return clip_score(sum(v[0] for v in scores.values())), scores


def score_four_dimension(ticker, info, peer_cache, market_state):
    """單股票四維評分；所有數據讀取均優先使用批量抓好的info cache。"""
    df = fetch_ohlcv(ticker, period="2y")
    if df is None or len(df) < 60:
        return None

    name = info.get("name", ticker)
    pe = safe_float(info.get("pe"), np.nan)
    pb = safe_float(info.get("pb"), np.nan)
    div_yield = info.get("div_yield")
    roe = info.get("roe")
    de_ratio = info.get("de_ratio")
    rev_growth = info.get("rev_growth")
    sector = get_sector_from_info(ticker, info)

    tech_total, tech_detail = technical_detail_score(df)
    short_score, mid_score, signals, _, _, _, _, _ = score_stock(df, market_state)

    pe_perc, pe_calc = get_pe_percentile_from_info(info)
    pb_perc, pb_calc = get_pb_percentile_from_info(info)
    pe_score = percentile_value_score(pe_perc, None if np.isnan(pe) else pe, (10, 15, 20))
    pb_score = percentile_value_score(pb_perc, None if np.isnan(pb) else pb, (1, 1.5, 2.5))

    is_financial = any(x in sector.lower() for x in ["bank", "financial", "hk_bank"])
    if pe_score is not None and pb_score is not None:
        val_score = pb_score * 0.60 + pe_score * 0.40 if is_financial else pe_score * 0.60 + pb_score * 0.40
    elif pe_score is not None:
        val_score = pe_score
    elif pb_score is not None:
        val_score = pb_score
    else:
        val_score = 50.0

    quality_flags, quality_penalty = quality_filter(roe, de_ratio, rev_growth)
    div_bonus, div_detail = dividend_bonus(div_yield, sector)
    val_score = clip_score(val_score * quality_penalty + div_bonus)

    pe_display = f"PE來源 {pe:.1f}" if not np.isnan(pe) else "PE來源 N/A"
    if pe_perc is not None and pe_calc is not None:
        pe_display += f"｜Yahoo計算PE {pe_calc:.1f}、近似百分位 {pe_perc*100:.0f}%"
    pb_display = f"PB來源 {pb:.2f}" if not np.isnan(pb) else "PB來源 N/A"
    if pb_perc is not None and pb_calc is not None:
        pb_display += f"｜Yahoo計算PB {pb_calc:.2f}、近似百分位 {pb_perc*100:.0f}%"
    val_detail = f"{pe_display} ｜ {pb_display} ｜ {div_detail}"

    sector_info = sector_relative_valuation(ticker, None if np.isnan(pe) else pe, sector, peer_cache)
    if sector_info:
        sector_detail = (
            f"{sector_info['sector']}：真實同業 {sector_info['n_peers']} 家，"
            f"中位PE {sector_info['median_pe']}，相對偏差 {sector_info['rel_pct']:+.1f}%"
        )
    else:
        sector_detail = "同業樣本不足（至少需2家真實同業，且已排除自己）"

    current_price = safe_float(df["close"].iloc[-1], 0)
    high_52 = get_52w_high(df)
    drawdown = (current_price - high_52) / high_52 * 100 if high_52 > 0 else 0
    # 平滑回撤評分：從0%到-45%線性提升，避免-20%附近斷崖跳分
    dd_score = smooth_low_score(drawdown, -45, 0, 90)

    fund_total, fund_detail = fund_flow_detail(df)
    capital_flow = get_futu_capital_flow(ticker)
    capital_bonus = 0.0
    capital_detail = "未取得（需富途連線/權限）"
    if capital_flow is not None and not capital_flow.empty:
        latest = capital_flow.iloc[-1]
        inflow = safe_float(latest.get("in_flow", 0), 0)
        if inflow > 0:
            capital_bonus = 10
            capital_detail = f"主力流入 {inflow:.0f}萬"
        else:
            capital_detail = "主力流出"
    fund_total = clip_score(fund_total + capital_bonus)

    macro = fetch_macro()
    vix = safe_float(macro.get("VIX", {}).get("val"), 20)
    weights = get_dynamic_weights(vix)
    raw_total = (
        weights["tech"] * tech_total + weights["val"] * val_score +
        weights["dd"] * dd_score + weights["fund"] * fund_total
    )
    total_score = round(clip_score(raw_total), 1)
    confidence = "高信心" if total_score >= 80 else ("中等信心" if total_score >= 60 else "低信心")

    return {
        "ticker": ticker, "name": name, "price": round(current_price, 3),
        "total_score": total_score, "confidence": confidence,
        "short_score": short_score, "mid_score": mid_score, "signals": "、".join(signals) if signals else "—",
        "tech_total": round(tech_total, 1), "val_score": round(val_score, 1),
        "val_label": valuation_label(val_score), "val_detail": val_detail,
        "pe_percentile": pe_perc, "pb_percentile": pb_perc,
        "sector": sector, "sector_detail": sector_detail,
        "quality_flags": "｜".join(quality_flags), "quality_penalty": quality_penalty,
        "drawdown": round(drawdown, 1), "dd_score": round(dd_score, 1),
        "fund_total": round(fund_total, 1), "capital_detail": capital_detail,
        "weights": weights, "vix": vix, "hi52": high_52,
        "info_source": info.get("info_source", "Yahoo")
    }

# ─────────────────────────────────────────────────────────────
# 市場狀態、信號紀錄與報告
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_macro():
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = fetch_ohlcv(ticker, period="1y")
            if df is None or len(df) < 5:
                add_error("宏觀數據不足", ticker)
                continue
            close = safe_float(df["close"].iloc[-1], 0)
            prev = safe_float(df["close"].iloc[-2], close)
            change = (close - prev) / prev * 100 if prev else 0
            high = safe_float(df["high"].max(), close)
            low = safe_float(df["low"].min(), close)
            pct = (close - low) / (high - low) * 100 if high != low else 50
            result[name] = {
                "val": close, "chg": change, "pct": pct, "hi": high, "lo": low,
                "rsi": safe_float(calc_rsi(df["close"]).iloc[-1], 50)
            }
        except Exception as exc:
            add_error("宏觀指標計算失敗", ticker, exc)
    return result


def classify_market_state():
    try:
        spy = fetch_ohlcv("SPY", period="6mo")
        if spy is None or len(spy) < 60:
            return "unknown", 0, 0
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
    if vix >= 30:
        return {"tech": 0.40, "val": 0.35, "dd": 0.10, "fund": 0.15}
    if vix >= 25:
        return {"tech": 0.35, "val": 0.35, "dd": 0.15, "fund": 0.15}
    if vix <= 15:
        return {"tech": 0.20, "val": 0.50, "dd": 0.15, "fund": 0.15}
    return {"tech": 0.30, "val": 0.40, "dd": 0.15, "fund": 0.15}


def log_signal(ticker, total_score, label, price, date):
    """每一日每隻股票只記錄一次。"""
    try:
        df_log = pd.read_csv(SIGNAL_LOG_FILE)
    except Exception:
        df_log = pd.DataFrame(columns=["date", "ticker", "total_score", "label", "price"])
    if not df_log.empty:
        exists = ((df_log["date"].astype(str) == str(date)) & (df_log["ticker"] == ticker)).any()
        if exists:
            return False
    new_row = pd.DataFrame([{
        "date": date, "ticker": ticker, "total_score": total_score,
        "label": label, "price": price
    }])
    pd.concat([df_log, new_row], ignore_index=True).to_csv(SIGNAL_LOG_FILE, index=False)
    return True


def calculate_position(price, stop_loss, account_size=100000, risk_pct=0.02):
    risk_amount = account_size * risk_pct
    per_share_risk = abs(price - stop_loss)
    if per_share_risk <= 0:
        return 0, risk_amount
    return int(risk_amount / per_share_risk), risk_amount


def generate_pdf_report(results, market_state, vix):
    if not PDF_AVAILABLE:
        return None
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
    f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據：富途 + Yahoo Finance</p>",
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
    filter_sig = st.multiselect(
        "篩選信號", ["🔥 強烈撈底", "⭐️ 值得關注", "👁️ 觀察中", "—"],
        default=["🔥 強烈撈底", "⭐️ 值得關注"]
    )
    min_short = st.slider("最低短線分", 0, 100, 0)
    min_mid = st.slider("最低中線分", 0, 100, 0)
    resonance_filter = st.selectbox("🔍 共振強度篩選", ["全部", "強", "中", "弱"], index=0)
    if quote_ctx:
        st.success("✅ 富途API 已連線（多線程已加鎖）")
    else:
        st.warning("⚠️ 富途API 未連線，使用 Yahoo Finance 數據")
    st.caption("資料非保證即時；Yahoo Finance 與富途資料可能存在更新延遲或數值差異。")

    st.divider()
    st.markdown("### 🧱 穩定性修正")
    st.markdown("""
    - 富途請求以 Lock 序列化，避免共用連線多線程衝突
    - Yahoo資料可平行抓取，但失敗會列入錯誤清單
    - 同業比較會排除自己，並要求至少2個真實同業
    - 形態加入低點間距及反彈確認
    - 技術/回撤分數採平滑計算，減少訊號閃爍
    - 品質警示與股息率實際納入估值分
    """)

market_state, market_ret, market_vol = classify_market_state()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🌍 市場氣氛", "📊 個股掃描", "📐 回撤計算", "📈 技術圖表",
    "🎯 四維撈底評分", "📋 信號追蹤與績效", "⚖️ 風險管理",
    "🔄 週期投影 (MCPE)", "🧪 實驗性分頁", "🛠️ 數據健康"
])

# ─────────────────────────────────────────────────────────────
# Tab 1 市場氣氛
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")
    macro = fetch_macro()
    vix_now = safe_float(macro.get("VIX", {}).get("val"), 20)
    state_map = {
        "bear_high_vol": "🐻 熊市高波動", "bear_low_vol": "🐻 熊市低波動",
        "bull_high_vol": "🐂 牛市高波動", "bull_low_vol": "🐂 牛市低波動",
        "neutral": "😐 中性", "unknown": "❓ 無法判斷"
    }
    st.markdown(f"### 當前市場狀態：{state_map.get(market_state, market_state)}")
    st.caption(f"SPY 60日回報：{market_ret:.1f}% ｜ 年化波動率：{market_vol:.1f}% ｜ VIX：{vix_now:.1f}")

    if market_state == "bear_high_vol":
        st.info("熊市高波動：可觀察超賣及估值訊號，但採分批策略與嚴格止損。")
    elif market_state == "bull_low_vol":
        st.info("牛市低波動：超賣可能只是短暫回調，避免因單一技術訊號過度撈底。")

    st.divider()
    st.markdown("### 📊 全球宏觀指標")
    kpi_items = [
        ("VIX", "😱 恐慌指數"), ("VVIX", "🌊 波動之波動"),
        ("SPX", "🇺🇸 標普500"), ("HSI", "🇭🇰 恒生指數"),
        ("US10Y", "🏦 美債10年息"), ("DXY", "💵 美元指數"),
        ("HYG", "📉 高收益債"), ("VHSI", "🇭🇰 港股波幅")
    ]
    for group in [kpi_items[:4], kpi_items[4:]]:
        cols = st.columns(4)
        for i, (key, label) in enumerate(group):
            item = macro.get(key, {})
            val = safe_float(item.get("val"), 0)
            chg = safe_float(item.get("chg"), 0)
            pct = safe_float(item.get("pct"), 0)
            color = C_GREEN if chg >= 0 else C_RED
            emoji = label.split()[0]
            name = " ".join(label.split()[1:])
            with cols[i]:
                st.markdown(
                    f"<div class='metric-card'><div>{emoji}</div><div style='color:#8b949e;font-size:.7em'>{name}</div>"
                    f"<div style='font-size:1.1em;font-weight:bold'>{val:.2f}</div>"
                    f"<div style='color:{color}'>{chg:+.2f}%</div><div style='color:#8b949e;font-size:.68em'>52W:{pct:.0f}%</div></div>",
                    unsafe_allow_html=True
                )

    def make_gauge(score, title):
        if score >= 70:
            color, text = C_GREEN, "🔥 極佳撈底視窗"
        elif score >= 55:
            color, text = C_ORANGE, "⚠️ 謹慎撈底機會"
        elif score >= 40:
            color, text = C_GREY, "😐 市場中性"
        else:
            color, text = C_RED, "😎 市場貪婪風險"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text": f"{title}<br><span style='font-size:.7em;color:{color}'>{text}</span>"},
            number={"font": {"color": color, "size": 40}, "suffix": "/100"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": .25},
                   "bgcolor": "#161b22", "bordercolor": "#30363d",
                   "steps": [{"range": [0,25], "color":"#1a1a2e"}, {"range":[25,45],"color":"#1c1a00"},
                             {"range":[45,65],"color":"#161b22"}, {"range":[65,80],"color":"#0d2818"},
                             {"range":[80,100],"color":"#0d3318"}]}
        ))
        fig.update_layout(height=260, paper_bgcolor=C_BG, font=dict(color="#e6edf3"), margin=dict(l=20,r=20,t=60,b=20))
        return fig

    if vix_now >= 30: vix_score = 80
    elif vix_now >= 25: vix_score = 60
    elif vix_now <= 15: vix_score = 20
    else: vix_score = 40
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(make_gauge(vix_score, "🇺🇸 美股撈底機會"), use_container_width=True)
    with c2: st.plotly_chart(make_gauge(vix_score, "🇭🇰 港股撈底機會"), use_container_width=True)
    with c3: st.plotly_chart(make_gauge(vix_score, "🌍 綜合評分"), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Tab 2 個股掃描
# ─────────────────────────────────────────────────────────────
with tab2:
    if market == "🇭🇰 港股":
        tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股":
        tickers = US_WATCHLIST
    else:
        tickers = [x.strip().upper() for x in custom_input.split("\n") if x.strip()] or US_WATCHLIST

    macro = fetch_macro()
    vix_now = safe_float(macro.get("VIX", {}).get("val"), 20)
    if vix_now >= 30:
        fc, fi, auto_min_mid = C_GREEN, "🔥", 60
        fl = f"VIX {vix_now:.1f} 極度恐慌 — 建議只看中線分≥60"
    elif vix_now >= 25:
        fc, fi, auto_min_mid = C_ORANGE, "⚠️", 50
        fl = f"VIX {vix_now:.1f} 高波動 — 建議只看中線分≥50"
    elif vix_now <= 15:
        fc, fi, auto_min_mid = C_RED, "😎", 0
        fl = f"VIX {vix_now:.1f} 市場偏貪婪 — 注意追高"
    else:
        fc, fi, auto_min_mid = C_GREY, "😐", 0
        fl = f"VIX {vix_now:.1f} 市場中性"

    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {fc};border-radius:8px;padding:12px 16px;margin-bottom:12px'>"
        f"{fi} <span style='color:{fc};font-weight:bold'>市場氣氛濾網</span>：{fl}</div>",
        unsafe_allow_html=True
    )
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
            short_s, mid_s, sigs, oversold_count, resonance, cmf_val, vwap_val, vol_z = score_stock(df, market_state)
            label, stype = signal_label(short_s, mid_s)
            close_v = safe_float(df["close"].iloc[-1], 0)
            high_52 = get_52w_high(df)
            prev_close = safe_float(df["close"].iloc[-2], close_v)
            change_1d = (close_v - prev_close) / prev_close * 100 if prev_close else 0
            vol_ma = safe_float(df["volume"].rolling(20).mean().iloc[-1], 1)
            vol_ratio = safe_float(df["volume"].iloc[-1], 0) / vol_ma if vol_ma else 0
            swing_low = safe_float(df["low"].iloc[-126:].min(), close_v)
            weekly_rsi = safe_float(calc_rsi(df["close"], 70).iloc[-1], 50)
            rows.append({
                "代碼": ticker, "現價": round(close_v, 3), "1日漲跌%": round(change_1d, 2),
                "52周高": round(high_52, 3), "距高位%": round((close_v-high_52)/high_52*100, 1) if high_52 else 0,
                "量比": round(vol_ratio, 2), "周線RSI": round(weekly_rsi, 1),
                "短線分": short_s, "中線分": mid_s, "信號": label, "_type": stype,
                "觸發指標": "、".join(sigs) if sigs else "—", "_df": df,
                "_drop": drop_levels(high_52), "_fib": fib_levels(swing_low, high_52),
                "cmf": cmf_val, "vwap": vwap_val, "vol_z": vol_z,
                "resonance": resonance, "oversold_count": oversold_count
            })
        except Exception as exc:
            add_error("掃描計算失敗", ticker, exc)
            failed_tickers.append(ticker)

    if failed_tickers:
        st.warning(f"⚠️ {len(failed_tickers)} 隻股票數據不足或抓取失敗：{', '.join(failed_tickers)}。詳細原因請看「🛠️ 數據健康」。")

    st.markdown("### 🚨 低位大成交量提示")
    alerts = []
    for r in rows:
        df = r["_df"]
        recent_low = safe_float(df["low"].iloc[-20:].min(), 0)
        current = r["現價"]
        pct_above_low = (current - recent_low) / recent_low * 100 if recent_low else 999
        is_positive = safe_float(df["close"].iloc[-1], 0) > safe_float(df["open"].iloc[-1], 0)
        if pct_above_low <= 5 and r["vol_z"] >= 2.0 and is_positive:
            alerts.append((r, pct_above_low, recent_low))
    if alerts:
        cols = st.columns(min(3, len(alerts)))
        for i, (r, pct_low, recent_low) in enumerate(alerts):
            with cols[i % 3]:
                st.markdown(
                    f"<div class='volume-alert'><b>{r['代碼']}</b><br>現價：{r['現價']}<br>"
                    f"近期低點：{recent_low:.3f}<br>距低點：{pct_low:.1f}%<br>"
                    f"量比：{r['量比']}x｜Z-score：{r['vol_z']}<br>信號：{r['信號']}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("目前沒有股票符合「距20日低點≤5% + 大成交量 + 收陽線」條件。")

    display_rows = rows
    if resonance_filter != "全部":
        display_rows = [r for r in rows if r["resonance"] == resonance_filter]
    filtered = [
        r for r in display_rows
        if r["信號"] in filter_sig and r["短線分"] >= min_short and r["中線分"] >= effective_min_mid
    ]
    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 強烈撈底：{sum(r['_type']=='buy' for r in filtered)} 隻**")

    if filtered:
        chart_df = pd.DataFrame([{"代碼": r["代碼"], "短線分": r["短線分"], "中線分": r["中線分"]} for r in filtered])
        fig = px.bar(chart_df.melt(id_vars="代碼", value_vars=["短線分", "中線分"]), x="代碼", y="value", color="variable", barmode="group", height=270,
                     color_discrete_map={"短線分": C_BLUE, "中線分": C_GREEN})
        fig.update_layout(paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig, use_container_width=True)

        sort_by = st.selectbox("排序方式", ["總分（短+中）", "短線分", "中線分", "量比", "距高位%", "周線RSI"], key="tab2_sort")
        sort_map = {
            "總分（短+中）": lambda x: x["短線分"] + x["中線分"],
            "短線分": lambda x: x["短線分"], "中線分": lambda x: x["中線分"],
            "量比": lambda x: x["量比"], "距高位%": lambda x: -x["距高位%"],
            "周線RSI": lambda x: x["周線RSI"]
        }
        for r in sorted(filtered, key=sort_map[sort_by], reverse=sort_by != "距高位%"):
            badge = signal_badge(r["信號"])
            res_class = {"強": "resonance-strong", "中": "resonance-medium", "弱": "resonance-weak"}.get(r["resonance"], "")
            with st.expander(
                f"<span class='signal-badge {badge}'>{r['信號']}</span> {r['代碼']} 現價 {r['現價']} "
                f"({r['1日漲跌%']:+.1f}%) ｜ 短線:{r['短線分']} 中線:{r['中線分']} "
                f"<span class='{res_class}'>{r['resonance']}共振</span>"
            ):
                a, b, c, d = st.columns(4)
                a.metric("CMF", f"{r['cmf']:.3f}")
                b.metric("VWAP", f"{r['vwap']:.2f}")
                c.metric("成交量Z-score", f"{r['vol_z']:.2f}")
                d.metric("共振指標數", f"{r['oversold_count']}/4")
                st.markdown(f"**觸發指標：** {r['觸發指標']}")
                left, right = st.columns(2)
                with left:
                    st.markdown("**📉 從52周高回撤位**")
                    st.dataframe(pd.DataFrame(list(r["_drop"].items()), columns=["回撤", "價位"]), hide_index=True, use_container_width=True)
                with right:
                    st.markdown("**🌀 斐波那契支撐位**")
                    st.dataframe(pd.DataFrame(list(r["_fib"].items()), columns=["比率", "價位"]), hide_index=True, use_container_width=True)
                st.markdown(f"**參考目標+20%：`{r['現價']*1.2:.3f}` ｜ 參考止損-8%：`{r['現價']*0.92:.3f}`**")

    st.divider()
    if rows:
        table = pd.DataFrame([{
            "代碼": r["代碼"], "現價": r["現價"], "漲跌%": r["1日漲跌%"], "量比": r["量比"],
            "CMF": r["cmf"], "VWAP": r["vwap"], "Z-score": r["vol_z"], "共振": r["resonance"],
            "短線分": r["短線分"], "中線分": r["中線分"], "信號": r["信號"]
        } for r in rows])
        st.subheader("📋 全部股票列表")
        st.dataframe(table.sort_values("短線分", ascending=False), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Tab 3 回撤
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📐 回撤 & 斐波那契計算器")
    c1, c2, c3 = st.columns(3)
    with c1: tk_input = st.text_input("股票代碼", "NVDA", key="dd_ticker").upper()
    with c2: manual_high = st.number_input("手動輸入高位（0=自動）", min_value=0.0, value=0.0)
    with c3: manual_low = st.number_input("手動輸入低位（0=自動）", min_value=0.0, value=0.0)
    if st.button("🔍 計算", type="primary", key="dd_calc"):
        df = fetch_ohlcv(tk_input, period="2y")
        if df is None:
            st.error("找不到數據，港股請用 0700.HK 格式。")
        else:
            current = safe_float(df["close"].iloc[-1], 0)
            high = manual_high if manual_high > 0 else get_52w_high(df)
            low = manual_low if manual_low > 0 else safe_float(df["low"].iloc[-252:].min(), 0)
            st.markdown(f"### {tk_input}｜現價：**{current:.3f}** ｜ 52周高：**{high:.3f}** ｜ 52周低：**{low:.3f}**")
            left, right = st.columns(2)
            with left:
                dd_rows = [{"回撤": k, "目標價": v, "現價距離": f"{current-v:+.2f}"} for k, v in drop_levels(high).items()]
                st.dataframe(pd.DataFrame(dd_rows), use_container_width=True, hide_index=True)
            with right:
                fib_rows = [{"比率": k, "支撐價": v, "現價距離": f"{current-v:+.2f}"} for k, v in fib_levels(low, high).items()]
                st.dataframe(pd.DataFrame(fib_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Tab 4 技術圖表
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart = st.text_input("輸入股票代碼", "AAPL", key="chart_ticker").upper()
    period_map = {"1個月": "1mo", "3個月": "3mo", "6個月": "6mo", "1年": "1y", "2年": "2y"}
    period_sel = st.radio("時間範圍", list(period_map.keys()), index=3, horizontal=True)
    df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])
    if df_ch is not None and len(df_ch) > 30:
        close = df_ch["close"]
        rsi = calc_rsi(close)
        weekly_rsi = calc_rsi(close, 70)
        macd, sig, hist = calc_macd(close)
        sma20, sma60, sma200 = close.rolling(20).mean(), close.rolling(60).mean(), close.rolling(200).mean()
        bb_up, bb_dn = sma20 + 2 * close.rolling(20).std(), sma20 - 2 * close.rolling(20).std()
        k, d, j = calc_kdj(df_ch)
        cci, wr = calc_cci(df_ch), calc_wr(df_ch)
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=.02,
                            row_heights=[.34,.10,.13,.11,.11,.11,.10])
        fig.add_trace(go.Candlestick(x=df_ch.index, open=df_ch["open"], high=df_ch["high"], low=df_ch["low"], close=df_ch["close"],
                                     increasing_line_color=C_GREEN, decreasing_line_color=C_RED, name="K線"), row=1, col=1)
        for ma, color, name in [(sma20, "#f0883e", "MA20"), (sma60, C_BLUE, "MA60"), (sma200, C_PURPLE, "MA200")]:
            fig.add_trace(go.Scatter(x=df_ch.index, y=ma, mode="lines", line=dict(color=color, width=1.2), name=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=bb_up, mode="lines", line=dict(color=C_GREY, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=bb_dn, mode="lines", line=dict(color=C_GREY, dash="dot"), fill="tonexty", fillcolor="rgba(139,148,158,.05)", showlegend=False), row=1, col=1)
        volume_colors = [C_GREEN if df_ch["close"].iloc[i] >= df_ch["open"].iloc[i] else C_RED for i in range(len(df_ch))]
        fig.add_trace(go.Bar(x=df_ch.index, y=df_ch["volume"], marker_color=volume_colors, name="成交量", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=rsi, mode="lines", line=dict(color=C_ORANGE), name="RSI日"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=weekly_rsi, mode="lines", line=dict(color=C_PURPLE, dash="dot"), name="RSI慢線"), row=3, col=1)
        for y, color in [(70,C_RED),(50,C_GREY),(30,C_GREEN)]: fig.add_hline(y=y, line_dash="dash", line_color=color, row=3, col=1)
        fig.add_trace(go.Bar(x=df_ch.index, y=hist, marker_color=[C_GREEN if x >= 0 else C_RED for x in hist.fillna(0)], name="MACD Hist", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=macd, mode="lines", line=dict(color=C_BLUE), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=sig, mode="lines", line=dict(color="#f0883e"), name="Signal"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=k, mode="lines", line=dict(color=C_GREEN), name="K"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=d, mode="lines", line=dict(color=C_RED), name="D"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=j, mode="lines", line=dict(color=C_ORANGE), name="J"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=cci, mode="lines", line=dict(color="#79c0ff"), name="CCI"), row=6, col=1)
        fig.add_trace(go.Scatter(x=df_ch.index, y=wr, mode="lines", line=dict(color="#ffa657"), name="W%R"), row=7, col=1)
        fig.update_layout(title=f"{tk_chart} 技術分析", height=1050, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
                          font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=50,b=10))
        for i in range(1, 8):
            fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)
        short, mid, signals, _, _, _, _, _ = score_stock(df_ch, market_state)
        current = safe_float(close.iloc[-1], 0)
        a,b,c,d_col = st.columns(4)
        a.metric("短線平滑評分", f"{short}/100")
        b.metric("中線平滑評分", f"{mid}/100")
        c.metric("日線RSI", f"{safe_float(rsi.iloc[-1],50):.1f}")
        d_col.metric("慢線RSI", f"{safe_float(weekly_rsi.iloc[-1],50):.1f}")
        if signals: st.markdown("**觸發指標：** " + " ｜ ".join(signals))
    else:
        st.warning("找不到足夠數據。")

# ─────────────────────────────────────────────────────────────
# Tab 5 四維評分
# ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🎯 四維撈底評分模型｜終極穩定版")
    st.caption("改善項目：批量資訊快取、同業排除自己、至少2家同業、平滑分數、品質懲罰、股息率加分、富途線程鎖、錯誤透明化。")

    if market == "🇭🇰 港股":
        auto_tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股":
        auto_tickers = US_WATCHLIST
    else:
        auto_tickers = [x.strip().upper() for x in custom_input.split("\n") if x.strip()] or US_WATCHLIST

    left, right = st.columns([1, 2])
    with left:
        scan_auto = st.button("🔄 掃描當前觀察名單", type="primary", key="scan_auto")
    with right:
        with st.expander("✏️ 手動輸入代碼"):
            manual_input = st.text_area("每行一個代碼", "AAPL\nNVDA\n0700.HK", key="tab5_manual")
            scan_manual = st.button("掃描手動清單", key="scan_manual")

    scan_list = auto_tickers if scan_auto else ([x.strip().upper() for x in manual_input.split("\n") if x.strip()] if scan_manual else None)

    if scan_list:
        clear_errors()
        progress = st.progress(0, text="步驟 1/3：批量下載基本面與5年資料...")
        info_map = {}
        failed_info = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in scan_list}
            completed = 0
            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    info_map[ticker] = future.result()
                except Exception as exc:
                    add_error("基本面平行工作失敗", ticker, exc)
                    failed_info.append(ticker)
                progress.progress(int(completed / len(scan_list) * 35), text=f"步驟 1/3：基本面資料 {completed}/{len(scan_list)}")

        progress.progress(40, text="步驟 2/3：建立同業PE快取（已排除自己）...")
        peer_cache = build_sector_peer_cache(scan_list, info_map)

        progress.progress(45, text="步驟 3/3：平行計算四維評分...")
        results = []
        failed_scores = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCORE) as executor:
            futures = {
                executor.submit(score_four_dimension, ticker, info_map.get(ticker, {}), peer_cache, market_state): ticker
                for ticker in scan_list
            }
            completed = 0
            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed_scores.append(ticker)
                except Exception as exc:
                    add_error("四維評分平行工作失敗", ticker, exc)
                    failed_scores.append(ticker)
                progress.progress(45 + int(completed / len(scan_list) * 55), text=f"步驟 3/3：評分計算 {completed}/{len(scan_list)}")
        progress.empty()

        failed = sorted(set(failed_info + failed_scores))
        if failed:
            st.warning(f"⚠️ 以下 {len(failed)} 隻沒有完成評分：{', '.join(failed)}。請到「🛠️ 數據健康」查看原因。")

        if results:
            results.sort(key=lambda x: x["total_score"], reverse=True)
            today = datetime.now().strftime("%Y-%m-%d")
            added = 0
            for r in results:
                if r["total_score"] >= 70:
                    added += int(log_signal(r["ticker"], r["total_score"], r["confidence"], r["price"], today))
            if added:
                st.success(f"已新增 {added} 筆今日高分信號紀錄（同日同股已自動去重）。")

            weights = results[0]["weights"]
            st.caption(f"VIX={results[0]['vix']:.1f} ｜ 權重：技術 {weights['tech']:.0%} / 估值 {weights['val']:.0%} / 回撤 {weights['dd']:.0%} / 資金 {weights['fund']:.0%}")

            table = pd.DataFrame([{
                "代碼": r["ticker"], "名稱": r["name"], "現價": r["price"],
                "總分": r["total_score"], "信心": r["confidence"],
                "估值標籤": r["val_label"], "技術分": r["tech_total"],
                "估值分": r["val_score"], "估值細節": r["val_detail"],
                "同業比較": r["sector_detail"], "品質檢查": r["quality_flags"],
                "品質係數": r["quality_penalty"], "回撤%": r["drawdown"],
                "回撤分": r["dd_score"], "資金分": r["fund_total"],
                "短線分": r["short_score"], "中線分": r["mid_score"],
                "富途資金": r["capital_detail"], "資料源": r["info_source"]
            } for r in results])
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ 下載詳細評分 CSV", data=table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"四維撈底詳細評分_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv"
            )

            a,b,c,d = st.columns(4)
            a.metric("監察數量", f"{len(results)} 隻")
            b.metric("平均總分", f"{np.mean([r['total_score'] for r in results]):.1f}")
            c.metric("最高分", f"{results[0]['ticker']} {results[0]['total_score']}")
            d.metric("最低分", f"{results[-1]['ticker']} {results[-1]['total_score']}")

            plot_df = pd.DataFrame({"代碼": [r["ticker"] for r in results], "總分": [r["total_score"] for r in results]})
            colors = [C_GREEN if x >= 80 else C_ORANGE if x >= 60 else C_RED for x in plot_df["總分"]]
            fig = go.Figure(go.Bar(x=plot_df["總分"], y=plot_df["代碼"], orientation="h", marker_color=colors,
                                   text=[f"{x:.1f}" for x in plot_df["總分"]], textposition="outside"))
            fig.update_layout(height=120 + len(results) * 35, paper_bgcolor=C_BG, plot_bgcolor=C_BG,
                              font=dict(color="#e6edf3"), margin=dict(l=10,r=50,t=10,b=10),
                              xaxis=dict(range=[0,100], gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True)

            if PDF_AVAILABLE:
                pdf = generate_pdf_report(results, market_state, results[0]["vix"])
                if pdf:
                    st.download_button("📄 下載今日報告 PDF", data=pdf,
                                       file_name=f"撈底報告_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
        else:
            st.error("沒有任何股票完成評分。請查看「🛠️ 數據健康」的錯誤紀錄。")
    else:
        st.info("點擊「掃描當前觀察名單」開始。第一次抓取基本面與5年歷史資料會較慢。")

    st.caption("重要限制：歷史PE/PB百分位以『目前EPS/每股淨值』回推歷史股價，並非逐期真實歷史PE/PB；高成長、盈利大幅波動或虧損公司尤其只宜作輔助參考。")

# ─────────────────────────────────────────────────────────────
# Tab 6 信號追蹤與績效
# ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("📋 信號追蹤與績效回測")
    st.caption("同日、同一股票只會記錄一次。注意：本機CSV部署至無持久磁碟的雲端服務時，重啟後可能遺失。")
    try:
        df_log = pd.read_csv(SIGNAL_LOG_FILE)
    except FileNotFoundError:
        df_log = pd.DataFrame()
    except Exception as exc:
        add_error("讀取信號記錄失敗", exc=exc)
        df_log = pd.DataFrame()

    if df_log.empty:
        st.info("尚無信號紀錄。請先在「四維撈底評分」完成一次掃描。")
    else:
        df_log["date"] = pd.to_datetime(df_log["date"], errors="coerce")
        df_log = df_log.dropna(subset=["date"]).sort_values("date", ascending=False)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        hold_days = st.selectbox("持有交易日", [5, 10, 20, 30], index=1)
        if st.button("計算已成熟信號績效", key="run_backtest"):
            results = []
            for _, row in df_log.iterrows():
                ticker = row["ticker"]
                entry_date = pd.Timestamp(row["date"]).normalize()
                entry_price = safe_float(row["price"], np.nan)
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                df = fetch_ohlcv(ticker, period="2y")
                if df is None or df.empty:
                    continue
                try:
                    index_norm = pd.DatetimeIndex(df.index).normalize()
                    future_dates = df.index[index_norm >= entry_date]
                    if len(future_dates) <= hold_days:
                        continue
                    exit_price = safe_float(df.loc[future_dates[hold_days], "close"], np.nan)
                    if np.isnan(exit_price):
                        continue
                    ret = (exit_price - entry_price) / entry_price * 100
                    results.append({
                        "代碼": ticker, "進場日": entry_date.strftime("%Y-%m-%d"),
                        "進場價": round(entry_price, 3), "出場價": round(exit_price, 3),
                        "回報%": round(ret, 2)
                    })
                except Exception as exc:
                    add_error("回測單筆計算失敗", ticker, exc)
            if results:
                bt = pd.DataFrame(results)
                st.dataframe(bt, use_container_width=True, hide_index=True)
                a,b,c = st.columns(3)
                a.metric("樣本數", len(bt))
                b.metric("勝率", f"{(bt['回報%'] > 0).mean()*100:.1f}%")
                c.metric("平均回報", f"{bt['回報%'].mean():.2f}%")
                bt["累積回報"] = (1 + bt["回報%"] / 100).cumprod() - 1
                fig = go.Figure(go.Scatter(x=np.arange(len(bt)), y=bt["累積回報"]*100, mode="lines+markers", line=dict(color=C_GREEN)))
                fig.update_layout(title=f"信號累積回報（持有{hold_days}交易日）", height=380, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("目前尚未有足夠成熟的信號可作此持有期回測。")

# ─────────────────────────────────────────────────────────────
# Tab 7 風險管理
# ─────────────────────────────────────────────────────────────
with tab7:
    st.subheader("⚖️ 風險管理與部位計算")
    account_size = st.number_input("帳戶總值（USD）", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆最大風險（%）", 0.5, 5.0, 2.0) / 100
    ticker = st.text_input("股票代碼", "AAPL", key="risk_ticker").upper()
    if st.button("計算部位", key="risk_calc"):
        df = fetch_ohlcv(ticker, period="2y")
        if df is None:
            st.error("找不到數據。")
        else:
            current = safe_float(df["close"].iloc[-1], 0)
            high = get_52w_high(df)
            low = safe_float(df["low"].iloc[-252:].min(), current)
            atr = safe_float(calc_atr(df).iloc[-1], 0)
            supports = [p for p in fib_levels(low, high).values() if p < current]
            fib_stop = max(supports) if supports else current * 0.92
            atr_stop = current - 2 * atr if atr > 0 else current * 0.92
            # 取較遠止損，避免止損過近，但不作交易建議
            stop = min(fib_stop, atr_stop)
            shares, risk_amount = calculate_position(current, stop, account_size, risk_pct)
            a,b,c,d = st.columns(4)
            a.metric("現價", f"{current:.3f}")
            b.metric("參考止損", f"{stop:.3f}")
            c.metric("每筆風險金額", f"${risk_amount:,.2f}")
            d.metric("建議股數", f"{shares}")
            st.caption("止損綜合斐波那契及2倍ATR，只屬機械化風控參考；仍需考慮流動性、跳空風險、倉位集中度與交易成本。")

# ─────────────────────────────────────────────────────────────
# Tab 8 MCPE
# ─────────────────────────────────────────────────────────────
with tab8:
    st.subheader("🔄 市場週期投影引擎 (MCPE)")
    st.caption("ZigZag波段僅用於視覺化與研究，不應作為單獨交易依據。")
    a,b,c,d = st.columns([1.2,1,1.2,1.5])
    with a: mcpe_ticker = st.text_input("股票代碼 (MCPE)", "0700.HK", key="mcpe_ticker").upper()
    with b: mcpe_period = st.selectbox("分析週期", ["1y","2y","3y","5y"], index=1)
    with c: mcpe_dev = st.number_input("轉折靈敏度 (%)", min_value=1.0, max_value=20.0, value=5.0, step=.5)
    with d:
        st.markdown("<br>", unsafe_allow_html=True)
        run_mcpe = st.button("🚀 執行週期投影", type="primary", use_container_width=True)

    def calculate_zigzag(df, deviation_pct):
        dev = deviation_pct / 100
        pivots = []
        trend = 1
        high_t, high_p = df.index[0], safe_float(df["high"].iloc[0], 0)
        low_t, low_p = df.index[0], safe_float(df["low"].iloc[0], 0)
        for i in range(1, len(df)):
            idx = df.index[i]
            high, low = safe_float(df["high"].iloc[i], 0), safe_float(df["low"].iloc[i], 0)
            if trend == 1:
                if high > high_p:
                    high_p, high_t = high, idx
                elif low < high_p * (1 - dev):
                    pivots.append((high_t, high_p, 1))
                    trend, low_p, low_t = -1, low, idx
            else:
                if low < low_p:
                    low_p, low_t = low, idx
                elif high > low_p * (1 + dev):
                    pivots.append((low_t, low_p, -1))
                    trend, high_p, high_t = 1, high, idx
        pivots.append((high_t, high_p, 1) if trend == 1 else (low_t, low_p, -1))
        out = pd.DataFrame(pivots, columns=["date","price","type"])
        out["days"] = out["date"].diff().dt.days
        out["pct_chg"] = out["price"].pct_change() * 100
        return out

    if run_mcpe or mcpe_ticker:
        df = fetch_ohlcv(mcpe_ticker, period=mcpe_period)
        if df is not None and len(df) > 30:
            pivots = calculate_zigzag(df, mcpe_dev)
            left, right = st.columns([7,3])
            with left:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                             increasing_line_color=C_GREEN, decreasing_line_color=C_RED, name="K線"))
                fig.add_trace(go.Scatter(x=pivots["date"], y=pivots["price"], mode="lines+markers", name="ZigZag",
                                         line=dict(color=C_BLUE, width=2), marker=dict(size=6, color="#e6edf3")))
                fig.update_layout(height=600, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                if len(pivots) >= 3:
                    current_type = pivots["type"].iloc[-1]
                    current_days = safe_float(pivots["days"].iloc[-1], 0)
                    same = pivots[pivots["type"] == current_type].iloc[:-1]
                    avg_days = safe_float(same["days"].mean(), current_days) if not same.empty else current_days
                    completion = min(current_days / avg_days * 100, 100) if avg_days > 0 else 0
                    st.metric("目前結構", "上漲波段" if current_type == 1 else "下跌波段")
                    st.metric("當前運行天數", f"{current_days:.0f} 日")
                    st.metric("歷史同向平均", f"{avg_days:.0f} 日")
                    st.metric("週期完成度", f"{completion:.1f}%")
                    if completion > 80:
                        st.warning("時間窗接近歷史平均，僅代表需提高對轉折的觀察，不代表必然反轉。")
                else:
                    st.info("波段不足，請調低轉折靈敏度。")
        else:
            st.error("無法載入數據。")

# ─────────────────────────────────────────────────────────────
# Tab 9 Beta
# ─────────────────────────────────────────────────────────────
with tab9:
    st.subheader("🧪 進階實驗室｜ATR 動態波段 + 江恩時間窗")
    a,b,c = st.columns([1,1,2])
    with a: beta_ticker = st.text_input("股票代碼 (Beta)", "MSTR", key="beta_ticker").upper()
    with b: atr_mult = st.number_input("ATR 敏感度乘數", min_value=.5, max_value=5.0, value=1.5, step=.1)
    with c:
        st.markdown("<br>", unsafe_allow_html=True)
        run_beta = st.button("🔬 執行高階週期運算", type="primary", use_container_width=True)

    if run_beta or beta_ticker:
        df = fetch_ohlcv(beta_ticker, period="1y")
        if df is not None and len(df) > 60:
            atr = calc_atr(df)
            current = safe_float(df["close"].iloc[-1], 0)
            current_atr = safe_float(atr.iloc[-1], 0)
            dynamic_dev = current_atr * atr_mult / current if current else .05
            st.info(f"動態轉折閾值：ATR={current_atr:.2f}，ATR乘數={atr_mult:.1f}，閾值={dynamic_dev*100:.2f}%")
            pivots = calculate_zigzag(df, dynamic_dev * 100)
            chart_col, data_col = st.columns([7,3])
            with chart_col:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                             increasing_line_color=C_GREEN, decreasing_line_color=C_RED, name="K線"))
                fig.add_trace(go.Scatter(x=pivots["date"], y=pivots["price"], mode="lines+markers", name="ATR ZigZag",
                                         line=dict(color=C_PURPLE, width=2), marker=dict(size=6, color="#e6edf3")))
                fig.update_layout(height=600, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            with data_col:
                if len(pivots) >= 3:
                    days = safe_float(pivots["days"].iloc[-1], 0)
                    gann = [7,21,49,90,144]
                    nearest = min(gann, key=lambda x: abs(x-days))
                    gap = abs(nearest-days)
                    st.metric("當前波段天數", f"{days:.0f} 日")
                    st.metric("最近江恩時間窗", f"{nearest} 日")
                    if gap <= 2:
                        st.warning("貼近江恩時間窗：只作額外觀察提示，不代表買賣訊號。")
                    else:
                        st.info(f"距最近時間窗約 {gap:.0f} 日。")
                else:
                    st.info("波段數不足。")
        else:
            st.error("無法載入足夠數據。")

# ─────────────────────────────────────────────────────────────
# Tab 10 數據健康／錯誤透明化
# ─────────────────────────────────────────────────────────────
with tab10:
    st.subheader("🛠️ 數據健康與錯誤紀錄")
    st.caption("舊版會靜默略過失敗資料；本版會保留錯誤紀錄，方便你識別缺漏資料、Yahoo限流或富途連線問題。")
    logs = st.session_state.get("error_log", [])
    if logs:
        st.warning(f"目前有 {len(logs)} 項錯誤/警告紀錄。")
        st.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("清除錯誤紀錄", key="clear_error_log"):
            clear_errors()
            st.rerun()
    else:
        st.success("目前未記錄到資料抓取或計算錯誤。")

    st.divider()
    st.markdown("### 部署提醒")
    st.markdown("""
    - `signal_log.csv` 是本機檔案。若部署到沒有持久磁碟的雲端容器，重啟後可能遺失。
    - 需要長期回測時，建議改用 SQLite、PostgreSQL、Supabase 或 Google Sheets。
    - Yahoo Finance 不是交易級資料源；遇到限流、修訂或延遲時，請以券商/交易所數據覆核。
    - 估值百分位為近似回推，不能取代逐季財報建構的真實歷史PE/PB。
    """)

st.caption("免責聲明：本工具僅供研究及教育用途，不構成投資建議。任何交易決定請自行核實數據、風險、流動性及個人財務狀況。")
