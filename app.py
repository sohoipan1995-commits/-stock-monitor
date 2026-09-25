import os
import time
import warnings
import threading
import sqlite3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("default", category=RuntimeWarning)

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

st.set_page_config(page_title="📈 股票監察系統 Pro+ V4.4", page_icon="📈", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V4.4-Validation-Portfolio-Risk"
MODEL_VERSION = "decision-v4.4"
AUTO_REFRESH_SEC = 1800
OHLCV_TTL = 1800
INFO_TTL = 3600
DB_FILE = "signals.db"
MAX_WORKERS_DATA = 5
MAX_WORKERS_SCORE = 4
TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 10
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
    "JPM", "BAC", "GS", "MS", "BRK-B", "COST", "WMT", "HD", "JNJ",
    "UNH", "PFE", "XOM", "NEE", "UBER", "LITE", "CLX", "NFLX"
]

ETF_TICKERS = {"SPY", "QQQ", "SOXL", "IWM"}
MACRO_TICKERS = {
    "VIX": "^VIX", "VVIX": "^VVIX", "SPX": "^GSPC", "HSI": "^HSI",
    "NDX": "^NDX", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VHSI": "^VHSI",
    "HYG": "HYG", "USDHKD": "USDHKD=X"
}

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
GANN_LEVELS = [0.125, 0.25, 0.333, 0.375, 0.500, 0.625, 0.666, 0.75, 0.875]
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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0d1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {color:#e6edf3!important;}
.metric-card {background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;margin:4px;}
.decision-card {background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:16px;margin:10px 0;}
.warn-card {background:#2b1a1a;border:1px solid #f85149;border-radius:10px;padding:12px;margin:8px 0;}
.good-card {background:#102419;border:1px solid #3fb950;border-radius:10px;padding:12px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "error_log" not in st.session_state:
    st.session_state.error_log = []


def safe_float(value, default=np.nan):
    try:
        return default if value is None or pd.isna(value) else float(value)
    except Exception:
        return default


def clip_score(value):
    return float(np.clip(value, 0, 100))


def add_error(context, ticker=None, exc=None):
    message = str(context)
    if ticker:
        message += f"｜{ticker}"
    if exc:
        message += f"｜{type(exc).__name__}: {str(exc)[:180]}"
    item = {"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "訊息": message}
    logs = st.session_state.get("error_log", [])
    if not any(x["訊息"] == message for x in logs):
        logs.append(item)
    st.session_state.error_log = logs[-200:]


def clear_errors():
    st.session_state.error_log = []


def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn.commit()
    conn.close()


init_db()


def to_futu(ticker):
    if ticker.endswith(".HK"):
        return f"HK.{ticker[:-3]}"
    if ticker.replace("-", "").isalpha():
        return f"US.{ticker}"
    return ticker


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
        add_error("富途初始化失敗，將使用 Yahoo 備援", exc=exc)
        return None


quote_ctx = init_futu()


def requested_start(period):
    now = datetime.now()
    mapping = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 800, "5y": 1900}
    return (now - timedelta(days=mapping.get(period, 400))).strftime("%Y-%m-%d")


@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    if quote_ctx and interval == "1d":
        try:
            start = requested_start(period)
            with FUTU_LOCK:
                ret, data, _ = quote_ctx.request_history_kline(
                    to_futu(ticker), start=start, end=None,
                    ktype=KLType.K_DAY, max_count=1000, extended_time=False
                )
            if ret == RET_OK and data is not None and not data.empty:
                columns = ["time_key", "open", "high", "low", "close", "volume"]
                df = data[columns].copy()
                df = df.rename(columns={"time_key": "date"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df.columns = [str(c).lower() for c in df.columns]
                return df.sort_index().dropna()
        except Exception as exc:
            add_error("富途 OHLCV 失敗，改用 Yahoo", ticker, exc)
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            add_error("Yahoo 沒有回傳 OHLCV", ticker)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            add_error("OHLCV 欄位不完整", ticker)
            return None
        return df.sort_index().dropna()
    except Exception as exc:
        add_error("Yahoo OHLCV 下載失敗", ticker, exc)
        return None


def fetch_multiple(tickers, period="2y", max_workers=MAX_WORKERS_DATA):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, period): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                add_error("平行 OHLCV 工作失敗", ticker, exc)
                results[ticker] = None
    return results


@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    result = {
        "ticker": ticker, "name": ticker, "pe": None, "pb": None, "div_yield": None,
        "roe": None, "de_ratio": None, "rev_growth": None, "eps": None,
        "book_value": None, "sector": None, "industry": None, "info_source": "Yahoo",
        "fundamental_updated": None
    }
    if quote_ctx:
        try:
            with FUTU_LOCK:
                ret, data = quote_ctx.get_market_snapshot([to_futu(ticker)])
            if ret == RET_OK and data is not None and not data.empty:
                row = data.iloc[0]
                if pd.notna(row.get("pe_ratio")):
                    result["pe"] = float(row.get("pe_ratio"))
                    result["info_source"] = "富途+Yahoo"
                if pd.notna(row.get("pb_ratio")):
                    result["pb"] = float(row.get("pb_ratio"))
                    result["info_source"] = "富途+Yahoo"
                result["name"] = row.get("stock_name", ticker) or ticker
        except Exception as exc:
            add_error("富途基本面快照失敗", ticker, exc)
    try:
        info = yf.Ticker(ticker).info or {}
        if result["name"] == ticker:
            result["name"] = info.get("shortName") or info.get("longName") or ticker
        if result["pe"] is None:
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe is None:
                eps = info.get("trailingEps")
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if eps and price and eps > 0:
                    pe = price / eps
            result["pe"] = safe_float(pe, None)
        if result["pb"] is None:
            result["pb"] = safe_float(info.get("priceToBook"), None)
        field_map = {
            "div_yield": "dividendYield", "roe": "returnOnEquity", "de_ratio": "debtToEquity",
            "rev_growth": "revenueGrowth", "eps": "trailingEps", "book_value": "bookValue",
            "sector": "sector", "industry": "industry", "fundamental_updated": "lastFiscalYearEnd"
        }
        for field, key in field_map.items():
            if result.get(field) is None:
                result[field] = info.get(key)
    except Exception as exc:
        add_error("Yahoo 基本面資料失敗", ticker, exc)
    return result


def get_sector_from_info(ticker, info=None):
    info = info or get_full_stock_info(ticker)
    return str(info.get("sector")) if info.get("sector") else SECTOR_MAP_FALLBACK.get(ticker, "OTHER")


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_weekly_rsi_daily_aligned(close, period=14):
    weekly_close = close.resample("W-FRI").last().dropna()
    weekly_rsi = calc_rsi(weekly_close, period)
    return weekly_rsi.reindex(close.index, method="ffill"), weekly_rsi


def calc_macd(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_cmf(df, period=20):
    multiplier = (2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def calc_mfi(df, period=14):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money = typical * df["volume"]
    pos = money.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg = money.where(typical < typical.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


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
    avwap = (typical * subset["volume"]).cumsum() / subset["volume"].cumsum().replace(0, np.nan)
    return avwap


def get_52w_high(df):
    return safe_float(df["high"].iloc[-252:].max() if len(df) >= 252 else df["high"].max(), np.nan)


def local_minima_indices(series, order=3):
    values = np.asarray(series, dtype=float)
    result = []
    for i in range(order, len(values) - order):
        window = values[i - order:i + order + 1]
        if np.isfinite(values[i]) and values[i] == np.nanmin(window):
            if values[i] < values[i - 1] or values[i] < values[i + 1]:
                result.append(i)
    return result


def detect_double_bottom(df, lookback=80, min_gap=10, max_gap=55, tolerance=0.05, min_rebound=0.08):
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
            if (neckline - max(low1, low2)) / max(low1, low2) >= min_rebound:
                return True
    return False


def detect_macd_bullish_divergence(df, lookback=90, min_gap=10, max_gap=60):
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
            if p2 < p1 * 0.99 and h2 > h1:
                return True
    return False


def calc_relative_strength(stock_df, benchmark_df, days=60):
    if stock_df is None or benchmark_df is None or len(stock_df) < days or len(benchmark_df) < days:
        return np.nan
    stock_ret = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-days] - 1
    bench_ret = benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[-days] - 1
    return float((stock_ret - bench_ret) * 100)


@st.cache_data(ttl=OHLCV_TTL)
def fetch_macro():
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        df = fetch_ohlcv(ticker, period="1y")
        if df is None or len(df) < 15:
            continue
        close = safe_float(df["close"].iloc[-1])
        prev = safe_float(df["close"].iloc[-2], close)
        high = safe_float(df["high"].max(), close)
        low = safe_float(df["low"].min(), close)
        ret10 = (close / safe_float(df["close"].iloc[-10], close) - 1) * 100
        result[name] = {
            "val": close,
            "chg": (close / prev - 1) * 100 if prev else 0,
            "pct": (close - low) / (high - low) * 100 if high > low else 50,
            "ret_10d": ret10,
            "last_bar": str(pd.Timestamp(df.index[-1]).date())
        }
    return result


def classify_market_state(market, macro_data):
    benchmark = "^HSI" if market == "🇭🇰 港股" else "SPY"
    vol_key = "VHSI" if market == "🇭🇰 港股" else "VIX"
    df = fetch_ohlcv(benchmark, period="1y")
    if df is None or len(df) < 205:
        return {"name": "unknown", "benchmark": benchmark, "return_60d": 0.0, "vol": 0.0, "vol_index": np.nan}
    close = df["close"]
    ma50, ma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
    ret60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100
    ann_vol = close.pct_change().iloc[-20:].std() * np.sqrt(252) * 100
    vol_index = safe_float(macro_data.get(vol_key, {}).get("val"), np.nan)
    below_200 = close.iloc[-1] < ma200
    high_vol = (not np.isnan(vol_index) and vol_index >= (25 if market == "🇺🇸 美股" else 28))
    if below_200 and ret60 < -5 and high_vol:
        name = "bear_high_vol"
    elif below_200 and ret60 < 0:
        name = "bear_low_vol"
    elif close.iloc[-1] > ma50 > ma200 and ret60 > 5:
        name = "bull_low_vol" if not high_vol else "bull_high_vol"
    else:
        name = "neutral"
    return {"name": name, "benchmark": benchmark, "return_60d": ret60, "vol": ann_vol, "vol_index": vol_index}


def quality_filter(info, sector):
    flags, blockers = [], []
    roe = safe_float(info.get("roe"))
    debt = safe_float(info.get("de_ratio"))
    revenue_growth = safe_float(info.get("rev_growth"))
    penalty = 1.0
    financial = any(x in sector.lower() for x in ["financial", "bank", "hk_bank"])
    if not np.isnan(roe) and roe < 0.05:
        flags.append("ROE 低於 5%")
        penalty -= 0.15
    if not financial and not np.isnan(debt) and debt > 180:
        flags.append("負債權益比高於 180%")
        penalty -= 0.15
    if not np.isnan(revenue_growth) and revenue_growth < -0.10:
        flags.append("營收按年下跌超過 10%")
        penalty -= 0.12
    if not np.isnan(roe) and roe < 0 and not np.isnan(revenue_growth) and revenue_growth < -0.15:
        blockers.append("盈利能力與收入同時惡化，疑似價值陷阱")
    if not flags:
        flags.append("品質資料未見明顯紅旗")
    return flags, blockers, max(0.55, penalty)


def valuation_score(info, sector):
    pe = safe_float(info.get("pe"))
    pb = safe_float(info.get("pb"))
    div_yield = safe_float(info.get("div_yield"))
    financial = any(x in sector.lower() for x in ["financial", "bank", "hk_bank"])
    parts = []
    if not np.isnan(pe) and pe > 0:
        parts.append(90 if pe < 10 else 70 if pe < 15 else 45 if pe < 23 else 20)
    if not np.isnan(pb) and pb > 0:
        parts.append(90 if pb < 1 else 70 if pb < 1.5 else 45 if pb < 3 else 20)
    if not parts:
        score = 50.0
        detail = "PE/PB 資料不足；估值採中性，不應據此判斷便宜"
    elif financial and not np.isnan(pb):
        score = 0.65 * (90 if pb < 1 else 70 if pb < 1.5 else 45 if pb < 3 else 20) + 0.35 * np.mean(parts)
        detail = f"金融業較重視 PB；PE={pe:.1f}，PB={pb:.2f}"
    else:
        score = float(np.mean(parts))
        detail = f"PE={pe:.1f}，PB={pb:.2f}"
    if not np.isnan(div_yield) and div_yield > 0:
        dividend_pct = div_yield * 100 if div_yield <= 1 else div_yield
        score += min(7, max(0, dividend_pct - 2))
        detail += f"，股息率={dividend_pct:.1f}%"
    return clip_score(score), detail


def classify_undervaluation(score, blockers):
    if blockers:
        return "⛔ 價值陷阱風險"
    if score >= 78:
        return "🟢 重度被低估"
    if score >= 62:
        return "🟠 中度被低估"
    if score >= 48:
        return "🟡 輕度被低估"
    return "⚪ 估值未見吸引"


def volume_zscore(df, period=20):
    volume = df["volume"]
    mean = safe_float(volume.rolling(period).mean().iloc[-1], 0)
    std = safe_float(volume.rolling(period).std().iloc[-1], 0)
    return (safe_float(volume.iloc[-1], 0) - mean) / std if std > 0 else 0.0


def decision_engine(ticker, df, info, regime, benchmark_df):
    if df is None or len(df) < 210:
        return None
    close = df["close"]
    price = safe_float(close.iloc[-1])
    high52 = get_52w_high(df)
    low20 = safe_float(df["low"].iloc[-20:].min())
    high5 = safe_float(df["high"].iloc[-5:].max())
    atr = safe_float(calc_atr(df).iloc[-1])
    rsi = safe_float(calc_rsi(close).iloc[-1], 50)
    weekly_rsi_aligned, weekly_rsi = calc_weekly_rsi_daily_aligned(close)
    weekly_rsi_now = safe_float(weekly_rsi_aligned.iloc[-1], 50)
    macd, macd_signal, macd_hist = calc_macd(close)
    cmf = safe_float(calc_cmf(df).iloc[-1], 0)
    avwap = calc_avwap_from_recent_low(df)
    avwap_now = safe_float(avwap.iloc[-1], price) if not avwap.empty else price
    ma20, ma50, ma200 = close.rolling(20).mean(), close.rolling(50).mean(), close.rolling(200).mean()
    vol_z = volume_zscore(df)
    rs60 = calc_relative_strength(df, benchmark_df, 60)
    sector = get_sector_from_info(ticker, info)
    quality_flags, quality_blockers, quality_penalty = quality_filter(info, sector)
    val_score, val_detail = valuation_score(info, sector)
    val_score = clip_score(val_score * quality_penalty)
    undervaluation = classify_undervaluation(val_score, quality_blockers)

    liquidity_20d = safe_float((df["close"] * df["volume"]).rolling(20).mean().iloc[-1], 0)
    eligibility = []
    blockers = list(quality_blockers)
    if price <= 0:
        blockers.append("現價無效")
    if liquidity_20d <= 0:
        blockers.append("成交額資料不足")
    if atr <= 0:
        blockers.append("ATR 資料不足，無法設定風險")
    if ticker in ETF_TICKERS:
        eligibility.append("ETF：不使用企業基本面品質否決規則")
    else:
        eligibility.append("價格、成交量和風險資料基本可用")

    oversold = 0
    oversold_reasons = []
    if rsi < 35:
        oversold += 1
        oversold_reasons.append(f"RSI(14) {rsi:.1f}")
    if weekly_rsi_now < 42:
        oversold += 1
        oversold_reasons.append(f"真周 RSI(14) {weekly_rsi_now:.1f}")
    drawdown = (price / high52 - 1) * 100 if high52 > 0 else 0
    if drawdown <= -15:
        oversold += 1
        oversold_reasons.append(f"距 52 周高位 {drawdown:.1f}%")
    bottom_score = min(100, oversold * 20 + (15 if detect_double_bottom(df) else 0) + (15 if detect_macd_bullish_divergence(df) else 0))

    confirmation = []
    confirm_score = 0
    confirmation_price = max(high5, safe_float(ma20.iloc[-1], price))
    if price >= avwap_now:
        confirm_score += 20
        confirmation.append("站上近 60 日低點 Anchored VWAP")
    if price >= safe_float(ma20.iloc[-1], price):
        confirm_score += 18
        confirmation.append("收復 MA20")
    if safe_float(ma20.iloc[-1]) > safe_float(ma20.iloc[-5], np.inf):
        confirm_score += 12
        confirmation.append("MA20 開始上彎")
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]):
        confirm_score += 15
        confirmation.append("MACD 金叉")
    if cmf > 0:
        confirm_score += 12
        confirmation.append("CMF 資金流為正")
    if vol_z >= 0.5 and price >= safe_float(df["open"].iloc[-1], price):
        confirm_score += 10
        confirmation.append("量價確認")
    if not np.isnan(rs60) and rs60 > 0:
        confirm_score += 13
        confirmation.append(f"60 日相對強弱 +{rs60:.1f}%")
    confirm_score = min(100, confirm_score)

    trend_risk = 0
    risk_reasons = []
    if price < safe_float(ma200.iloc[-1], price):
        trend_risk += 18
        risk_reasons.append("仍低於 MA200")
    if regime["name"] in {"bear_high_vol", "bear_low_vol"}:
        trend_risk += 18 if regime["name"] == "bear_high_vol" else 10
        risk_reasons.append("市場處於熊市 Regime")
    if not np.isnan(rs60) and rs60 < -10:
        trend_risk += 14
        risk_reasons.append("明顯跑輸大市")
    if atr / price > 0.08:
        trend_risk += 12
        risk_reasons.append("波動率偏高")
    risk_score = min(100, trend_risk)

    structural_stop = min(low20, price - 2 * atr)
    stop_loss = max(0.01, structural_stop)
    per_share_risk = price - stop_loss
    target_2r = price + 2 * per_share_risk
    rr_valid = per_share_risk > 0 and target_2r > price
    if not rr_valid:
        blockers.append("無法計算有效止損／2R 目標")

    regime_min_confirm = 62 if regime["name"] == "bear_high_vol" else 52 if regime["name"].startswith("bear") else 45
    final_score = clip_score(0.28 * val_score + 0.18 * bottom_score + 0.32 * confirm_score + 0.12 * min(100, max(0, 50 + rs60 if not np.isnan(rs60) else 50)) + 0.10 * (100 - risk_score))

    if blockers:
        decision = "不合資格"
        next_step = "先修正資料／品質／風險問題；不建議建立新倉。"
    elif confirm_score >= regime_min_confirm and val_score >= 48 and risk_score <= 45 and rr_valid:
        decision = "可小量試倉"
        next_step = "只在收市站穩確認價或下一交易日可執行價格進場；嚴守止損。"
    elif val_score >= 48 and (bottom_score >= 35 or confirm_score >= 28):
        decision = "等待突破"
        next_step = "建立價格提醒；未突破確認價前不視為買入訊號。"
    elif val_score >= 48:
        decision = "估值觀察"
        next_step = "公司可能有估值吸引力，但尚未看到足夠反轉確認。"
    else:
        decision = "不合資格"
        next_step = "估值與風險回報不足，暫不列作候選。"

    reasons = []
    reasons.extend(oversold_reasons)
    reasons.extend(confirmation)
    reasons.extend(quality_flags)
    if not reasons:
        reasons.append("尚未形成足夠的底部或反轉證據")

    return {
        "ticker": ticker, "name": info.get("name", ticker), "sector": sector,
        "price": round(price, 3), "last_bar": str(pd.Timestamp(df.index[-1]).date()),
        "decision": decision, "next_step": next_step, "final_score": round(final_score, 1),
        "valuation_score": round(val_score, 1), "undervaluation": undervaluation,
        "valuation_detail": val_detail, "bottom_score": round(bottom_score, 1),
        "confirm_score": round(confirm_score, 1), "risk_score": round(risk_score, 1),
        "quality_penalty": round(quality_penalty, 2), "quality_flags": "｜".join(quality_flags),
        "reasons": "｜".join(reasons), "risk_reasons": "｜".join(risk_reasons) if risk_reasons else "未见主要风险扣减",
        "blockers": "｜".join(blockers) if blockers else "—", "confirmation_price": round(confirmation_price, 3),
        "stop_loss": round(stop_loss, 3), "target_2r": round(target_2r, 3),
        "drawdown": round(drawdown, 1), "rsi": round(rsi, 1), "weekly_rsi": round(weekly_rsi_now, 1),
        "cmf": round(cmf, 3), "avwap": round(avwap_now, 3), "rs60": round(rs60, 1) if not np.isnan(rs60) else np.nan,
        "liquidity_20d": round(liquidity_20d, 0), "regime": regime["name"], "model_version": MODEL_VERSION
    }


def save_snapshot_and_signal(result):
    payload = pd.Series(result).to_json(force_ascii=False)
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO scan_snapshots (snapshot_date,ticker,model_version,payload_json,created_at) VALUES (?,?,?,?,?)",
        (today, result["ticker"], MODEL_VERSION, payload, datetime.now().isoformat())
    )
    if result["decision"] == "可小量試倉":
        conn.execute(
            "INSERT OR IGNORE INTO signals (signal_date,ticker,model_version,decision,total_score,price,confirmation_price,stop_loss,target_2r,market_regime,sector,features_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (today, result["ticker"], MODEL_VERSION, result["decision"], result["final_score"], result["price"],
             result["confirmation_price"], result["stop_loss"], result["target_2r"], result["regime"], result["sector"], payload, datetime.now().isoformat())
        )
    conn.commit()
    conn.close()


def calculate_position(price, stop_loss, account_size, risk_pct, max_notional_pct=0.10):
    risk_amount = account_size * risk_pct
    risk_per_share = price - stop_loss
    if risk_per_share <= 0 or price <= 0:
        return 0, 0, risk_amount
    by_risk = int(risk_amount / risk_per_share)
    by_notional = int((account_size * max_notional_pct) / price)
    shares = max(0, min(by_risk, by_notional))
    return shares, shares * price, risk_amount


def run_walk_forward_backtest(signal_df, hold_days=20, costs_bps=TRANSACTION_COST_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS):
    rows = []
    for _, signal in signal_df.iterrows():
        ticker = signal["ticker"]
        df = fetch_ohlcv(ticker, period="2y")
        if df is None or df.empty:
            add_error("回測無價格資料", ticker)
            continue
        idx = pd.DatetimeIndex(df.index).normalize()
        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        future = np.where(idx > signal_date)[0]
        if len(future) == 0:
            continue
        entry_i = int(future[0])
        if entry_i + hold_days >= len(df):
            continue
        raw_entry = safe_float(df["open"].iloc[entry_i])
        entry = raw_entry * (1 + slippage_bps / 10000)
        stop = safe_float(signal["stop_loss"])
        target = safe_float(signal["target_2r"])
        exit_price, exit_date, outcome = None, None, "時間出場"
        for i in range(entry_i, min(entry_i + hold_days + 1, len(df))):
            low, high = safe_float(df["low"].iloc[i]), safe_float(df["high"].iloc[i])
            hit_stop, hit_target = low <= stop, high >= target
            if hit_stop and hit_target:
                exit_price, exit_date, outcome = stop, df.index[i], "同日雙觸發：保守先止損"
                break
            if hit_stop:
                exit_price, exit_date, outcome = stop, df.index[i], "止損"
                break
            if hit_target:
                exit_price, exit_date, outcome = target, df.index[i], "2R 目標"
                break
        if exit_price is None:
            exit_i = entry_i + hold_days
            exit_price, exit_date = safe_float(df["close"].iloc[exit_i]), df.index[exit_i]
        net_return = (exit_price * (1 - costs_bps / 10000) - entry) / entry * 100
        rows.append({
            "代碼": ticker, "信號日": str(signal_date.date()), "進場日": str(pd.Timestamp(df.index[entry_i]).date()),
            "進場價": round(entry, 3), "止損": round(stop, 3), "2R目標": round(target, 3),
            "出場日": str(pd.Timestamp(exit_date).date()), "出場價": round(exit_price, 3),
            "結果": outcome, "淨回報%": round(net_return, 2)
        })
    return pd.DataFrame(rows)


def gann_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {f"{v * 100:.1f}%": round(swing_high - diff * v, 3) for v in GANN_LEVELS}


def fib_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {f"{int(v * 100)}%": round(swing_high - diff * v, 3) for v in FIB_LEVELS}


def drop_levels(high_price):
    return {f"-{int(v * 100)}%": round(high_price * (1 - v), 3) for v in DROP_LEVELS}


def render_metrics(macro_data, regime):
    pairs = [("VIX", "😱 VIX"), ("VVIX", "🌊 VVIX"), ("SPX", "🇺🇸 S&P 500"), ("HSI", "🇭🇰 恒生指數"), ("US10Y", "🏦 美債10年"), ("DXY", "💵 美元"), ("HYG", "📉 高收益債"), ("VHSI", "🇭🇰 VHSI")]
    for group in [pairs[:4], pairs[4:]]:
        cols = st.columns(4)
        for col, (key, label) in zip(cols, group):
            item = macro_data.get(key, {})
            value = safe_float(item.get("val"), 0)
            change = safe_float(item.get("chg"), 0)
            col.metric(label, f"{value:.2f}", f"{change:+.2f}%")
    st.caption(f"本市场 Regime：{regime['name']}｜基准：{regime['benchmark']}｜60日回报：{regime['return_60d']:.1f}%｜年化波动：{regime['vol']:.1f}%")


elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, AUTO_REFRESH_SEC - int(elapsed))
mins, secs = divmod(remaining, 60)

with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    st.markdown(f"🔄 快取刷新倒计时：**{mins:02d}:{secs:02d}**")
    if st.button("🔄 立即刷新"):
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        clear_errors()
        st.rerun()
    market = st.radio("市场", ["🇭🇰 港股", "🇺🇸 美股", "📋 自选"], index=1)
    custom_input = ""
    if market == "📋 自选":
        custom_input = st.text_area("输入代码（每行一个）", "AAPL\nNVDA\n0700.HK\n9988.HK")
    view_mode = st.radio("显示模式", ["新手模式", "进阶模式"], index=0)
    if quote_ctx:
        st.success("✅ 富途已连接：优先使用日线报价")
    else:
        st.info("ℹ️ 富途未连接：使用 Yahoo Finance 备援")
    st.caption("提示：SQLite 在某些云端部署环境不是永久储存；长期信号记录建议迁移至 Supabase/PostgreSQL。")

if elapsed >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    clear_errors()
    st.rerun()

macro_data = fetch_macro()
regime = classify_market_state(market if market != "📋 自选" else "🇺🇸 美股", macro_data)
benchmark_df = fetch_ohlcv(regime["benchmark"], period="2y")

st.markdown("# 📈 股票監察系統 Pro+｜V4.4 驗證與組合風控版")
st.caption(f"模型：{MODEL_VERSION}｜程式版本：{APP_VERSION}｜頁面生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HKT")
st.warning("本系统用于研究与风险管理，不构成投资建议。任何“可小量试仓”都不是预测见底或保证收益；必须以可执行价格、止损与组合风险为准。")

tabs = st.tabs([
    "🌍 市场气氛", "📊 决策扫描", "💎 估值候选", "📈 技术图表", "📐 回撤/江恩",
    "🧪 Walk-forward 回测", "⚖️ 组合风控", "🌟 因子选股", "🛠️ 系统健康"
])

with tabs[0]:
    st.subheader("🌍 宏观市场气氛")
    render_metrics(macro_data, regime)
    st.markdown("""
**如何使用 Regime：** 熊市高波动并不自动提高撈底分数；系统会提高反转确认门槛。牛市也不代表可以追高，仍要检查估值、结构止损和组合风险。

**数据时间说明：** 每个个股结果会显示最后一根价格 bar 日期；页面时间不等于报价时间。
""")

with tabs[1]:
    st.subheader("📊 决策扫描｜资格 → 估值 → 反转确认 → 风险")
    if market == "🇭🇰 港股":
        tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股":
        tickers = US_WATCHLIST
    else:
        tickers = [x.strip().upper() for x in custom_input.splitlines() if x.strip()]
    st.caption(f"本次扫描 {len(tickers)} 个代码。扫描只突出值得进一步研究者；系统仍会保存全部扫描快照供后续核查。")
    if st.button("🚀 执行 V4.4 决策扫描", type="primary", key="decision_scan"):
        clear_errors()
        progress = st.progress(0, text="下载基本面资料…")
        info_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures), start=1):
                ticker = futures[future]
                try:
                    info_map[ticker] = future.result()
                except Exception as exc:
                    add_error("基本面并行工作失败", ticker, exc)
                progress.progress(int(i / max(1, len(tickers)) * 35), text=f"基本面 {i}/{len(tickers)}")
        progress.progress(40, text="下载价格与计算决策…")
        data_map = fetch_multiple(tickers, period="2y")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCORE) as executor:
            futures = {executor.submit(decision_engine, ticker, data_map.get(ticker), info_map.get(ticker, {}), regime, benchmark_df): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures), start=1):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        save_snapshot_and_signal(result)
                except Exception as exc:
                    add_error("决策模型工作失败", ticker, exc)
                progress.progress(40 + int(i / max(1, len(tickers)) * 60), text=f"评分 {i}/{len(tickers)}")
        progress.empty()
        st.session_state["latest_results"] = results

    results = st.session_state.get("latest_results", [])
    if results:
        df_results = pd.DataFrame(results).sort_values(["decision", "final_score"], ascending=[True, False])
        candidate_order = {"可小量試倉": 0, "等待突破": 1, "估值觀察": 2, "不合資格": 3}
        df_results["_rank"] = df_results["decision"].map(candidate_order).fillna(9)
        df_results = df_results.sort_values(["_rank", "final_score"], ascending=[True, False]).drop(columns="_rank")
        candidates = df_results[df_results["decision"].isin(["可小量試倉", "等待突破", "估值觀察"])]
        st.success(f"扫描完成：{len(df_results)} 个有效结果；研究候选 {len(candidates)} 个；可小量试仓 {sum(df_results['decision'] == '可小量試倉')} 个。")
        display_cols = ["ticker", "name", "price", "decision", "undervaluation", "final_score", "valuation_score", "bottom_score", "confirm_score", "risk_score", "confirmation_price", "stop_loss", "target_2r", "last_bar"]
        st.dataframe(candidates[display_cols], use_container_width=True, hide_index=True)
        if view_mode == "进阶模式":
            st.markdown("### 完整审核表")
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        for _, row in candidates.head(8).iterrows():
            color = C_GREEN if row["decision"] == "可小量試倉" else C_ORANGE
            st.markdown(f"<div class='decision-card' style='border-left-color:{color}'><b>{row['ticker']}｜{row['decision']}｜{row['undervaluation']}</b><br>原因：{row['reasons']}<br>确认价：{row['confirmation_price']}｜止损：{row['stop_loss']}｜2R：{row['target_2r']}<br>下一步：{row['next_step']}</div>", unsafe_allow_html=True)
    else:
        st.info("按“执行 V4.4 决策扫描”后会显示研究候选。")

with tabs[2]:
    st.subheader("💎 估值候选｜先看便宜，再看是否值得买")
    results = st.session_state.get("latest_results", [])
    if not results:
        st.info("请先在“决策扫描”完成扫描。")
    else:
        val_df = pd.DataFrame(results)
        val_df = val_df[val_df["undervaluation"].isin(["🟢 重度被低估", "🟠 中度被低估", "🟡 輕度被低估"])]
        st.dataframe(val_df[["ticker", "name", "undervaluation", "valuation_score", "quality_flags", "decision", "confirm_score", "risk_score", "valuation_detail", "blockers"]].sort_values("valuation_score", ascending=False), use_container_width=True, hide_index=True)
        st.markdown("""
- **轻度被低估：** 估值有吸引力，但证据有限或仍须等待技术确认。
- **中度被低估：** 多项现时估值指标处于较低区间，仍必须检查基本面与风险。
- **重度被低估：** 现时估值指标明显偏低且未触发质量否决；不等于必然反弹。

当前版本不把“以今天 EPS/PB 回推过去股价”伪装成真实历史估值百分位；真正的历史估值百分位必须使用每一个历史时点可见的财报数据。
""")

with tabs[3]:
    st.subheader("📈 技术图表｜真实周 RSI 与 Anchored VWAP")
    ticker = st.text_input("股票代码", "AAPL", key="chart_ticker").upper()
    period_map = {"3个月": "3mo", "6个月": "6mo", "1年": "1y", "2年": "2y"}
    period = st.radio("时间范围", list(period_map.keys()), horizontal=True, index=2)
    chart_df = fetch_ohlcv(ticker, period=period_map[period])
    if chart_df is not None and len(chart_df) > 30:
        close = chart_df["close"]
        rsi = calc_rsi(close)
        weekly_rsi, _ = calc_weekly_rsi_daily_aligned(close)
        macd, signal, hist = calc_macd(close)
        avwap = calc_avwap_from_recent_low(chart_df)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.50, 0.15, 0.15, 0.20])
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=close, name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=close.rolling(20).mean(), name="MA20", line=dict(color="#f0883e")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=close.rolling(50).mean(), name="MA50", line=dict(color=C_BLUE)), row=1, col=1)
        if not avwap.empty:
            fig.add_trace(go.Scatter(x=avwap.index, y=avwap, name="Anchored VWAP（近60日低点）", line=dict(color=C_PURPLE, dash="dot")), row=1, col=1)
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["volume"], name="成交量", marker_color=[C_GREEN if c >= o else C_RED for c, o in zip(chart_df["close"], chart_df["open"])]), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=rsi, name="日 RSI(14)", line=dict(color=C_ORANGE)), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=weekly_rsi, name="真周 RSI(14)", line=dict(color=C_PURPLE, dash="dot")), row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=C_GREY, row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=C_GREY, row=3, col=1)
        fig.add_trace(go.Bar(x=chart_df.index, y=hist, name="MACD Histogram", marker_color=[C_GREEN if x >= 0 else C_RED for x in hist.fillna(0)]), row=4, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=macd, name="MACD", line=dict(color=C_BLUE)), row=4, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=signal, name="Signal", line=dict(color=C_ORANGE)), row=4, col=1)
        fig.update_layout(height=850, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"technical_chart_{ticker}_{period}")
        st.caption("真周 RSI：先取每周五收市价，再计算 RSI(14)。Anchored VWAP：从近 60 日低点开始计算的成交量加权均价，用作观察成本密集区，不是保证支撑。")

with tabs[4]:
    st.subheader("📐 回撤、斐波那契与江恩观察位")
    ticker = st.text_input("股票代码", "NVDA", key="levels_ticker").upper()
    if st.button("计算观察位", key="calc_levels"):
        df = fetch_ohlcv(ticker, period="2y")
        if df is None or df.empty:
            st.error("无法取得价格数据。")
        else:
            current = safe_float(df["close"].iloc[-1])
            high = get_52w_high(df)
            low = safe_float(df["low"].iloc[-252:].min())
            cols = st.columns(3)
            cols[0].dataframe(pd.DataFrame([{"回撤": k, "价位": v} for k, v in drop_levels(high).items()]), hide_index=True)
            cols[1].dataframe(pd.DataFrame([{"Fib": k, "价位": v} for k, v in fib_levels(low, high).items()]), hide_index=True)
            cols[2].dataframe(pd.DataFrame([{"江恩比例": k, "价位": v} for k, v in gann_levels(low, high).items()]), hide_index=True)
            st.caption(f"现价：{current:.3f}。这些是价格观察区，不是预测必然反转的目标价。")

with tabs[5]:
    st.subheader("🧪 严格 Walk-forward 回测")
    st.markdown("信号在收市后产生；回测以**下一交易日开盘价加滑点**进场。纳入止损、2R 目标、固定持有期、成本；若同一日同时触及止损与目标，保守地按先止损处理。")
    try:
        conn = sqlite3.connect(DB_FILE)
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC", conn)
        conn.close()
    except Exception as exc:
        add_error("读取回测信号失败", exc=exc)
        signals = pd.DataFrame()
    if signals.empty:
        st.info("尚无“可小量试仓”信号。先运行决策扫描，系统会按信号日、代码和模型版本去重记录。")
    else:
        st.dataframe(signals[["signal_date", "ticker", "decision", "total_score", "price", "confirmation_price", "stop_loss", "target_2r", "market_regime", "model_version"]], use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hold_days = st.selectbox("最长持有交易日", [10, 20, 30, 45], index=1)
        with c2:
            cost_bps = st.number_input("单边交易成本（bps）", min_value=0, max_value=100, value=TRANSACTION_COST_BPS)
        with c3:
            slip_bps = st.number_input("入场滑点（bps）", min_value=0, max_value=100, value=DEFAULT_SLIPPAGE_BPS)
        if st.button("运行严格回测", type="primary", key="strict_bt"):
            bt = run_walk_forward_backtest(signals, hold_days, cost_bps, slip_bps)
            if bt.empty:
                st.info("现有信号尚未成熟，或历史价格不足以完成所选持有期。")
            else:
                st.dataframe(bt, use_container_width=True, hide_index=True)
                cols = st.columns(4)
                cols[0].metric("成熟样本", len(bt))
                cols[1].metric("胜率", f"{(bt['淨回報%'] > 0).mean() * 100:.1f}%")
                cols[2].metric("平均净回报", f"{bt['淨回報%'].mean():.2f}%")
                cols[3].metric("中位数净回报", f"{bt['淨回報%'].median():.2f}%")
                st.caption("样本数少、幸存者偏差、数据缺漏、公司行动和未覆盖的流动性约束都会使回测偏乐观；应累积足够点时可见的日度快照后再调权重。")

with tabs[6]:
    st.subheader("⚖️ 组合风控与部位计算")
    c1, c2, c3 = st.columns(3)
    with c1:
        account_size = st.number_input("账户总值（USD）", min_value=1000.0, value=100000.0, step=1000.0)
    with c2:
        risk_pct = st.slider("单笔最大账户风险 (%)", 0.25, 3.0, 1.0, 0.25) / 100
    with c3:
        max_notional = st.slider("单一持仓名义上限 (%)", 3, 25, 10) / 100
    ticker = st.text_input("计划交易代码", "AAPL", key="portfolio_ticker").upper()
    if st.button("计算风险部位", key="position_calc"):
        df = fetch_ohlcv(ticker, period="1y")
        if df is None or len(df) < 30:
            st.error("价格资料不足。")
        else:
            price = safe_float(df["close"].iloc[-1])
            atr = safe_float(calc_atr(df).iloc[-1])
            stop = min(safe_float(df["low"].iloc[-20:].min()), price - 2 * atr)
            shares, notional, risk_amount = calculate_position(price, stop, account_size, risk_pct, max_notional)
            target = price + 2 * (price - stop)
            cols = st.columns(4)
            cols[0].metric("建议最大股数", f"{shares:,}")
            cols[1].metric("名义金额", f"${notional:,.0f}")
            cols[2].metric("若止损最大亏损", f"${min(risk_amount, shares * max(0, price-stop)):,.0f}")
            cols[3].metric("2R 目标", f"{target:.3f}")
            st.warning("港股下单前必须再按每手股数、货币、佣金及实际盘口流动性调整；部位模型不能替代你的总组合行业／相关性限制。")

with tabs[7]:
    st.subheader("🌟 多因子选股｜价值、质量、动能、低波动")
    st.caption("此模块适合 3–6 个月以上研究；与短线反转系统不同。金融、亏损企业和 ETF 不适合只用 PE 与 ROE 横向比较。")
    if st.button("执行多因子运算", type="primary", key="multi_factor"):
        if market == "🇭🇰 港股":
            tickers = HK_WATCHLIST
        elif market == "🇺🇸 美股":
            tickers = US_WATCHLIST
        else:
            tickers = [x.strip().upper() for x in custom_input.splitlines() if x.strip()]
        info_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    info_map[ticker] = future.result()
                except Exception as exc:
                    add_error("因子基本面失败", ticker, exc)
        data_map = fetch_multiple(tickers, period="1y")
        records = []
        for ticker in tickers:
            df = data_map.get(ticker)
            info = info_map.get(ticker, {})
            if df is None or len(df) < 130 or ticker in ETF_TICKERS:
                continue
            close = df["close"]
            returns = close.pct_change().iloc[-126:]
            records.append({
                "代码": ticker, "名称": info.get("name", ticker), "PE": safe_float(info.get("pe")),
                "ROE%": safe_float(info.get("roe")) * 100, "6M动能%": (close.iloc[-1] / close.iloc[-126] - 1) * 100,
                "年化波动%": returns.std() * np.sqrt(252) * 100
            })
        if records:
            mf = pd.DataFrame(records)
            for col in ["PE", "ROE%", "6M动能%", "年化波动%"]:
                mf[col] = mf[col].replace([np.inf, -np.inf], np.nan)
                mf[col] = mf[col].fillna(mf[col].median())
            def zscore(series, inverse=False):
                std = series.std()
                z = pd.Series(0.0, index=series.index) if std == 0 or pd.isna(std) else (series - series.mean()) / std
                return -z if inverse else z
            mf["价值Z"] = zscore(mf["PE"], True)
            mf["质量Z"] = zscore(mf["ROE%"])
            mf["动能Z"] = zscore(mf["6M动能%"])
            mf["防御Z"] = zscore(mf["年化波动%"], True)
            mf["综合分"] = (mf[["价值Z", "质量Z", "动能Z", "防御Z"]].sum(axis=1)).round(2)
            mf = mf.sort_values("综合分", ascending=False)
            st.dataframe(mf, use_container_width=True, hide_index=True)
            fig = px.scatter(mf, x="PE", y="6M动能%", text="代码", color="综合分", size=np.maximum(6, (mf["质量Z"] - mf["质量Z"].min() + 0.5) * 8), color_continuous_scale="RdYlGn")
            fig.update_layout(template="plotly_dark", plot_bgcolor=C_BG, paper_bgcolor=C_BG, height=550)
            st.plotly_chart(fig, use_container_width=True, key="factor_scatter")

with tabs[8]:
    st.subheader("🛠️ 系统健康、数据覆盖与错误日志")
    st.markdown("### 存储状态")
    st.warning("目前使用本地 SQLite。若部署在 Streamlit Community Cloud 或无持久磁盘环境，重启或部署后数据库可能被清空。正式长期运行请迁移至 Supabase PostgreSQL。")
    try:
        conn = sqlite3.connect(DB_FILE)
        signal_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM signals", conn).iloc[0]["n"]
        snapshot_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM scan_snapshots", conn).iloc[0]["n"]
        conn.close()
        cols = st.columns(2)
        cols[0].metric("已记录可试仓信号", int(signal_count))
        cols[1].metric("已记录扫描快照", int(snapshot_count))
    except Exception as exc:
        add_error("读取数据库健康状态失败", exc=exc)
    logs = st.session_state.get("error_log", [])
    if logs:
        st.warning(f"本次页面运行记录到 {len(logs)} 项问题。请先修复数据覆盖与错误，不要仅忽略。")
        st.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("清除本次错误日志", key="clear_logs"):
            clear_errors()
            st.rerun()
    else:
        st.success("本次页面运行暂未记录错误。注意：这不代表所有历史数据完整或策略已验证。")
    st.markdown("""
### 下一阶段建议
1. 使用 GitHub Actions 在港、美股收市后写入每日数据与模型快照。
2. 将 SQLite 迁移至 PostgreSQL，保存可追溯的价格、基本面、特征和模型版本。
3. 累积足够成熟样本后，按市场、行业、Regime 与决策层级校准真实胜率和期望值。
4. 对江恩时间窗进行独立样本外检验；未经验证前，它只能作为风险日历。
""")
