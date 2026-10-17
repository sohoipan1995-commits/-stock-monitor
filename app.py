import os
import json
import re
import logging
import requests
from io import StringIO
from zoneinfo import ZoneInfo
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

st.set_page_config(page_title="📈 股票監察系統 V4.7 四指數自動股票池", page_icon="📈", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V4.7-AutoPool-Top30"
MODEL_VERSION = "decision-v4.7-data-quality"
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
    "6862.HK", "0960.HK", "2020.HK", "1810.HK", "1024.HK", "3033.HK"
]

US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "ORCL", "ASML", "AMD", "QCOM", "INTC", "AMAT", "LRCX", "MU",
    "JPM", "BAC", "GS", "MS", "BRK-B", "COST", "WMT", "HD", "JNJ",
    "UNH", "PFE", "XOM", "NEE", "UBER", "LITE", "CLX", "NFLX"
]

ETF_TICKERS = {"SPY", "QQQ", "SOXL", "IWM", "3033.HK", "2800.HK", "2828.HK"}
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
    "9618.HK": "HK_TECH", "9999.HK": "HK_TECH", "1810.HK": "HK_TECH",
    "3033.HK": "HK_TECH_ETF"
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
.tspr-panel {background:#161b22;border:1px solid #4a148c;border-radius:8px;padding:12px;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "error_log" not in st.session_state:
    st.session_state.error_log = []


def hkt_now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong"))


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
        return f"HK.{ticker[:-3].zfill(5)}"
    if ticker.replace("-", "").isalpha():
        return f"US.{ticker}"
    return ticker


@st.cache_resource
def init_futu():
    if not FUTU_AVAILABLE:
        return None
    try:
        with FUTU_LOCK:
            ctx = OpenQuoteContext(host=os.getenv("FUTU_HOST", "127.0.0.1"), port=int(os.getenv("FUTU_PORT", "11111")))
            ret, _ = ctx.get_market_snapshot(["HK.00700"])
        return ctx if ret == RET_OK else None
    except Exception as exc:
        add_error("富途初始化失敗，將使用 Yahoo 備援", exc=exc)
        return None


quote_ctx = None


def requested_start(period):
    now = datetime.now()
    mapping = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 800, "5y": 1900}
    return (now - timedelta(days=mapping.get(period, 400))).strftime("%Y-%m-%d")


@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    if quote_ctx and interval == "1d" and not ticker.startswith("^") and "=" not in ticker:
        try:
            start = requested_start(period)
            with FUTU_LOCK:
                ret, data, _ = quote_ctx.request_history_kline(
                    to_futu(ticker), start=start, end=None,
                    ktype=KLType.K_DAY, max_count=1000, extended_time=False
                )
            if ret == RET_OK and data is not None and not data.empty:
                columns = ["time_key", "open", "high", "low", "close", "volume"]
                if "turnover" in data.columns:
                    columns.append("turnover")
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
    if quote_ctx and not ticker.startswith("^"):
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
    avwap = (typical * subset["volume"]).cumsum() / subset["volume"].cumsum().replace(0, np.nan)
    return avwap


# ==================== V4.6 升級：TSPR 10日支持位 + 下跌幅度計算 + 江恩50%/OBV/MFI 確認模組 ====================
def calc_tspr_system(df, left_bars=10, right_bars=5, box_period=65, num_boxes=3, require_filter=False):
    """
    計算 TSPR (Trend & Support Pattern Recognition) V4.6
    1. 捕捉 10日結構支持位 (Pivot Low)
    2. 檢測收市跌穿 10日支持位 -> 同步驗證：
       - 江恩 50% 分水嶺 (波段前低至波段前高的 50% 中軸)
       - OBV 能量潮是否跌穿 10日均線 / MFI(14) 是否低於 50
    3. 精確計算每次破位的「下跌幅度」：
       - 破位後最大跌幅 (%) 與跌點 (由破位收市價計至下一波段最低點)
       - 波段頂部至低點總跌幅 (%) (由破位前波段最高位計至波段最低點)
       - 下跌持續交易日數
    """
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

    support_events = []
    breakdown_events = []
    support_lines = []

    active_support = None
    support_start_idx = None
    is_broken = False
    last_pivot_idx = -999

    for i in range(left_bars + right_bars, n):
        pivot_idx = i - right_bars
        window_lows = lows[pivot_idx - left_bars: i + 1]
        left_window = lows[pivot_idx - left_bars: pivot_idx]

        # 1. 確認 10 日波段低點 (Pivot Low)
        if (
            np.isfinite(lows[pivot_idx])
            and lows[pivot_idx] == np.nanmin(window_lows)
            and lows[pivot_idx] < np.nanmin(left_window)
            and (pivot_idx - last_pivot_idx) >= right_bars
        ):
            # 若上一個破位事件正在進行，先結算該次破位至新低點之間的「最終下跌幅度」
            if breakdown_events and breakdown_events[-1].get("end_idx") is None:
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
            support_events.append({
                "idx": pivot_idx,
                "date": dates[pivot_idx],
                "price": active_support
            })

        # 2. 檢測收市價是否跌穿當前生效的 10 日支持位
        if active_support is not None and not is_broken and i > support_start_idx:
            if closes[i] < active_support:
                # 計算破位前波段最高位與江恩 50% 分水嶺
                lookback_start = max(0, support_start_idx - 30)
                wave_high = float(np.nanmax(highs[lookback_start: i + 1]))
                wave_low = float(np.nanmin(lows[lookback_start: support_start_idx + 1]))
                gann_50_level = wave_low + (wave_high - wave_low) * 0.50

                # 檢查江恩 50% 與 OBV/MFI 資金流確認
                gann_broken = closes[i] <= gann_50_level
                obv_weak = (
                    np.isfinite(obv_vals[i]) and np.isfinite(obv_ma10_vals[i]) and obv_vals[i] < obv_ma10_vals[i]
                )
                mfi_weak = np.isfinite(mfi_vals[i]) and mfi_vals[i] < 50.0
                flow_confirmed = bool(obv_weak or mfi_weak)

                # 若開啟嚴格過濾，則要求必須跌穿江恩50%或資金流轉弱才算正式破位
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
                confirm_label = "｜".join(confirm_tags) if confirm_tags else "疑似震倉(未破江恩/資金未流出)"

                breakdown_events.append({
                    "idx": i,
                    "date": dates[i],
                    "price": float(closes[i]),
                    "high": float(highs[i]),
                    "broken_support": round(active_support, 3),
                    "wave_high": round(wave_high, 3),
                    "gann_50": round(gann_50_level, 3),
                    "gann_broken": gann_broken,
                    "flow_confirmed": flow_confirmed,
                    "confirm_label": confirm_label,
                    "end_idx": None,
                    "trough_date": None,
                    "trough_price": None,
                    "post_break_drop_pts": 0.0,
                    "post_break_drop_pct": 0.0,
                    "peak_to_trough_drop_pct": round((closes[i] / wave_high - 1) * 100, 2),
                    "duration_days": 0
                })
                support_lines.append({
                    "start_date": dates[support_start_idx],
                    "end_date": dates[i],
                    "price": active_support,
                    "broken": True
                })

    # 結算最後一個尚未結束的破位事件（計算截至最新交易日的下跌幅度）
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

    # 建立多段週期紫色形態箱體 (Pattern Boxes) + 計算每個箱體的頂底跌幅
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
            "start_date": sub.index[0],
            "end_date": sub.index[-1],
            "high": b_high,
            "low": b_low,
            "drop_pct": round(b_drop_pct, 2)
        })

    # 計算當前週期量化面板統計（包含圖二所有下跌幅度與江恩50%指標）
    recent_len = min(box_period, n)
    recent_df = df.iloc[-recent_len:]
    cycle_high = float(recent_df["high"].max())
    cycle_low = float(recent_df["low"].min())
    cycle_range = cycle_high - cycle_low
    current_price = float(closes[-1])

    # 1. 週期頂部至低點最大跌幅 (%) (對應圖二「頂部至低點跌幅」)
    peak_to_trough_drop_pct = ((cycle_low - cycle_high) / cycle_high * 100) if cycle_high > 0 else 0.0
    # 2. 現價距離週期頂部下跌幅度 (%)
    current_drop_from_high_pct = ((current_price - cycle_high) / cycle_high * 100) if cycle_high > 0 else 0.0
    # 3. 調整比率 (現價在整個週期波幅中的回調比例 %)
    retracement_ratio_pct = ((cycle_high - current_price) / cycle_range * 100) if cycle_range > 0 else 0.0
    # 4. 江恩 50% 中軸位與黃金分割 1.236 倍目標位
    cycle_gann_50 = cycle_low + cycle_range * 0.50
    fib_1236_up = cycle_low + cycle_range * 1.236
    fib_1236_down = cycle_high - cycle_range * 1.236

    # 5. 若當前處於跌穿 10 日支持狀態，計算自跌穿日起算的已下跌幅度 (%)
    active_post_break_drop_pct = 0.0
    if is_broken and breakdown_events:
        latest_brk = breakdown_events[-1]
        active_post_break_drop_pct = round((current_price / latest_brk["price"] - 1) * 100, 2)

    # 6. 歷史破位平均再跌幅度 (%)
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
        if below_gann_50 and (obv_now_weak or mfi_now < 50):
            status_str = "🔴 跌穿支持+破江恩50% (深度調整)"
        else:
            status_str = "🟠 跌穿10日支持 (展開調整)"
    else:
        status_str = "🟢 守穩10日支持 (升勢/結構維持)"

    stats = {
        "cycle_high": round(cycle_high, 2),
        "cycle_low": round(cycle_low, 2),
        "cycle_range": round(cycle_range, 2),
        "peak_to_trough_drop_pct": round(peak_to_trough_drop_pct, 2),
        "current_drop_from_high_pct": round(current_drop_from_high_pct, 2),
        "retracement_pct": round(retracement_ratio_pct, 1),
        "active_support": round(active_support, 3) if active_support else np.nan,
        "active_post_break_drop_pct": active_post_break_drop_pct,
        "avg_post_break_drop_pct": avg_post_break_drop_pct,
        "cycle_gann_50": round(cycle_gann_50, 2),
        "below_gann_50": below_gann_50,
        "obv_weak": obv_now_weak,
        "mfi_now": round(mfi_now, 1),
        "fib_1236": round(fib_1236_up, 2),
        "fib_1236_down": round(max(0.01, fib_1236_down), 2),
        "status": status_str
    }

    return {
        "active_support": round(active_support, 3) if active_support else np.nan,
        "is_broken": is_broken,
        "status": status_str,
        "support_events": support_events,
        "breakdown_events": breakdown_events,
        "support_lines": support_lines,
        "pattern_boxes": pattern_boxes,
        "stats": stats
    }


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
    df = trim_completed(df, completed_session("HK" if market == "🇭🇰 港股" else "US"))
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
    mfi_now = safe_float(calc_mfi(df, 14).iloc[-1], 50)
    obv_s, obv_ma10_s = calc_obv(df)
    obv_bullish = safe_float(obv_s.iloc[-1], 0) >= safe_float(obv_ma10_s.iloc[-1], 0)

    avwap = calc_avwap_from_recent_low(df)
    avwap_now = safe_float(avwap.iloc[-1], price) if not avwap.empty else price
    ma20, ma50, ma200 = close.rolling(20).mean(), close.rolling(50).mean(), close.rolling(200).mean()
    vol_z = volume_zscore(df)
    rs60 = calc_relative_strength(df, benchmark_df, 60)

    # 計算 TSPR V4.6 (含下跌幅度、江恩50%及OBV/MFI確認)
    tspr = calc_tspr_system(df, left_bars=10, right_bars=5, box_period=65)
    tspr_sup = tspr["active_support"]
    tspr_broken = tspr["is_broken"]
    tspr_status = tspr["status"]
    st_stats = tspr["stats"]
    fib_1236 = st_stats.get("fib_1236", np.nan)
    cycle_gann_50 = st_stats.get("cycle_gann_50", np.nan)
    below_gann_50 = st_stats.get("below_gann_50", False)
    cycle_drop_pct = st_stats.get("current_drop_from_high_pct", 0.0)
    peak_to_trough_drop_pct = st_stats.get("peak_to_trough_drop_pct", 0.0)
    post_break_drop_pct = st_stats.get("active_post_break_drop_pct", 0.0)

    sector = get_sector_from_info(ticker, info)
    quality_flags, quality_blockers, quality_penalty = quality_filter(info, sector)
    val_score, val_detail = valuation_score(info, sector)
    val_score = clip_score(val_score * quality_penalty)
    undervaluation = classify_undervaluation(val_score, quality_blockers)
    valuation_available = any(np.isfinite(safe_float(info.get(k))) and safe_float(info.get(k)) > 0 for k in ("pe", "pb"))
    if not valuation_available:
        undervaluation = "⚪ 估值資料不足"

    liquidity_20d = safe_float((df["close"] * df["volume"]).rolling(20).mean().iloc[-1], 0)
    eligibility = []
    blockers = list(quality_blockers) if ticker not in ETF_TICKERS else []
    if not valuation_available:
        blockers.append("PE/PB有效資料不足，不能以中性分視作低估")
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
        confirm_score += 16
        confirmation.append("站上近 60 日低點 Anchored VWAP")
    if price >= safe_float(ma20.iloc[-1], price):
        confirm_score += 14
        confirmation.append("收復 MA20")
    if safe_float(ma20.iloc[-1]) > safe_float(ma20.iloc[-5], np.inf):
        confirm_score += 10
        confirmation.append("MA20 開始上彎")
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]):
        confirm_score += 12
        confirmation.append("MACD 金叉")
    if cmf > 0 and obv_bullish and mfi_now >= 50:
        confirm_score += 14
        confirmation.append(f"OBV>MA10 且 MFI({mfi_now:.0f})資金淨流入")
    elif cmf > 0 or obv_bullish:
        confirm_score += 8
        confirmation.append("量價資金流初步回穩")
    if vol_z >= 0.5 and price >= safe_float(df["open"].iloc[-1], price):
        confirm_score += 10
        confirmation.append("量價確認")
    if not np.isnan(rs60) and rs60 > 0:
        confirm_score += 10
        confirmation.append(f"60 日相對強弱 +{rs60:.1f}%")
    if not np.isnan(tspr_sup) and not tspr_broken:
        confirm_score += 14
        confirmation.append(f"守穩 TSPR 10日支持 ({tspr_sup:.2f})")
    if not below_gann_50 and not np.isnan(cycle_gann_50):
        Confirmation_gann = f"守穩週期江恩50% ({cycle_gann_50:.2f})"
        confirmation.append(Confirmation_gann)
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
        trend_risk += 12
        risk_reasons.append("明顯跑輸大市")
    if atr / price > 0.08:
        trend_risk += 10
        risk_reasons.append("波動率偏高")
    if tspr_broken:
        trend_risk += 15
        risk_reasons.append(f"跌穿10日支持({tspr_sup:.2f})，破位後跌幅 {post_break_drop_pct:.1f}%")
    if below_gann_50:
        trend_risk += 10
        risk_reasons.append(f"失守週期江恩50%分水嶺({cycle_gann_50:.2f})")
    if not obv_bullish and mfi_now < 45:
        trend_risk += 8
        risk_reasons.append(f"OBV<MA10 且 MFI({mfi_now:.0f})偏弱")
    risk_score = min(100, trend_risk)

    base_stop = min(low20, price - 2 * atr)
    if not np.isnan(tspr_sup) and not tspr_broken and tspr_sup < price:
        structural_stop = max(base_stop, tspr_sup * 0.995)
    else:
        structural_stop = base_stop

    stop_loss = max(0.01, structural_stop)
    per_share_risk = price - stop_loss
    target_2r = price + 2 * per_share_risk
    rr_valid = per_share_risk > 0 and target_2r > price
    if not rr_valid:
        blockers.append("無法計算有效止損／2R 目標")

    regime_min_confirm = 62 if regime["name"] == "bear_high_vol" else 52 if regime["name"].startswith("bear") else 45
    final_score = clip_score(
        0.28 * val_score
        + 0.18 * bottom_score
        + 0.32 * confirm_score
        + 0.12 * min(100, max(0, 50 + rs60 if not np.isnan(rs60) else 50))
        + 0.10 * (100 - risk_score)
    )

    if blockers:
        decision = "不合資格"
        next_step = "先修正資料／品質／風險問題；不建議建立新倉。"
    elif confirm_score >= regime_min_confirm and val_score >= 48 and risk_score <= 45 and rr_valid and not tspr_broken:
        decision = "可小量試倉"
        next_step = "收市守穩 TSPR 10日支持與江恩 50% 分水嶺可分批執行；跌穿 10日支持嚴格止損。"
    elif val_score >= 48 and (bottom_score >= 35 or confirm_score >= 28):
        decision = "等待突破"
        next_step = "建立價格提醒；等待重新站上江恩 50% 或突破確認價前不視為買入訊號。"
    elif val_score >= 48:
        decision = "估值觀察"
        next_step = "公司有估值吸引力，但正處於破位調整浪或資金流尚未確認回流。"
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
        "tspr_status": tspr_status, "tspr_support": tspr_sup, "gann_50": cycle_gann_50,
        "cycle_drop_pct": cycle_drop_pct, "peak_to_trough_drop_pct": peak_to_trough_drop_pct,
        "post_break_drop_pct": post_break_drop_pct, "fib_1236": fib_1236,
        "valuation_score": round(val_score, 1), "undervaluation": undervaluation,
        "valuation_detail": val_detail, "bottom_score": round(bottom_score, 1),
        "confirm_score": round(confirm_score, 1), "risk_score": round(risk_score, 1),
        "quality_penalty": round(quality_penalty, 2), "quality_flags": "｜".join(quality_flags),
        "reasons": "｜".join(reasons), "risk_reasons": "｜".join(risk_reasons) if risk_reasons else "未見主要風險扣減",
        "blockers": "｜".join(blockers) if blockers else "—", "confirmation_price": round(confirmation_price, 3),
        "stop_loss": round(stop_loss, 3), "target_2r": round(target_2r, 3),
        "drawdown": round(drawdown, 1), "rsi": round(rsi, 1), "weekly_rsi": round(weekly_rsi_now, 1),
        "mfi": round(mfi_now, 1), "cmf": round(cmf, 3), "avwap": round(avwap_now, 3),
        "rs60": round(rs60, 1) if not np.isnan(rs60) else np.nan,
        "liquidity_20d": round(liquidity_20d, 0), "regime": regime["name"], "model_version": MODEL_VERSION
    }


def save_snapshot_and_signal(result):
    payload = pd.Series(result).to_json(force_ascii=False)
    today = result["last_bar"]
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO scan_snapshots (snapshot_date,ticker,model_version,payload_json,created_at) VALUES (?,?,?,?,?)",
        (today, result["ticker"], MODEL_VERSION, payload, hkt_now().isoformat())
    )
    if result["decision"] == "可小量試倉" and result.get("formal_signal_eligible", True):
        conn.execute(
            "INSERT OR IGNORE INTO signals (signal_date,ticker,model_version,decision,total_score,price,confirmation_price,stop_loss,target_2r,market_regime,sector,features_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (today, result["ticker"], MODEL_VERSION, result["decision"], result["final_score"], result["price"],
             result["confirmation_price"], result["stop_loss"], result["target_2r"], result["regime"], result["sector"], payload, hkt_now().isoformat())
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
    st.caption(f"本市場 Regime：{regime['name']}｜基準：{regime['benchmark']}｜60日回報：{regime['return_60d']:.1f}%｜年化波動：{regime['vol']:.1f}%")



# ==================== V4.7 自動股票池 ====================
INDEX_SPECS = {
    "DJIA": ("道指", "US", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", 30, 30),
    "NDX": ("納斯達克100", "US", "https://en.wikipedia.org/wiki/Nasdaq-100", 95, 115),
    "SP500": ("標普500", "US", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 490, 520),
    "HSI": ("恒生指數", "HK", "https://en.wikipedia.org/wiki/Hang_Seng_Index", 50, 150),
}

def init_pool_db():
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS auto_pool_state (state_key TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS auto_pool_history (index_key TEXT, market_date TEXT, payload TEXT, updated_at TEXT, PRIMARY KEY(index_key,market_date))")
        conn.execute("CREATE TABLE IF NOT EXISTS permanent_watchlist (ticker TEXT PRIMARY KEY, source TEXT NOT NULL, added_at TEXT NOT NULL)")
        seeds = [(t, "原有永久名單", hkt_now().isoformat()) for t in HK_WATCHLIST + US_WATCHLIST]
        conn.executemany("INSERT OR IGNORE INTO permanent_watchlist VALUES (?,?,?)", seeds)


def state_get(key):
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute("SELECT payload FROM auto_pool_state WHERE state_key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def state_put(key, payload):
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("INSERT OR REPLACE INTO auto_pool_state VALUES (?,?,?)", (key, json.dumps(payload, ensure_ascii=False, allow_nan=False), hkt_now().isoformat()))


def permanent_tickers():
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        return [x[0] for x in conn.execute("SELECT ticker FROM permanent_watchlist ORDER BY ticker")]


def normalize_ticker(value, hk=False):
    value = re.sub(r"\[[^\]]*\]", "", str(value)).strip()
    if hk:
        match = re.fullmatch(r"(?:SEHK\s*:\s*|HKG\s*:\s*)?(\d{1,5})(?:\.HK)?", value, re.I)
        return str(int(match[1])).zfill(4) + ".HK" if match else None
    value = value.upper().replace(".", "-")
    return value if re.fullmatch(r"[A-Z]{1,6}(?:-[A-Z])?", value) else None


def parse_constituents(tables, index_key):
    spec = INDEX_SPECS[index_key]
    candidates = []
    for table in tables:
        df = table.copy()
        df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
        for col in df.columns:
            if col.lower().strip() not in {"symbol", "ticker", "ticker symbol"}:
                continue
            symbols = sorted({s for value in df[col] if (s := normalize_ticker(value, spec[1] == "HK"))})
            if spec[3] <= len(symbols) <= spec[4]:
                candidates.append(symbols)
    if len(candidates) != 1:
        raise ValueError(f"{index_key} 成分表驗證失敗：找到 {len(candidates)} 個合格表格；未使用固定熱門股替代")
    return candidates[0]


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_constituents(index_key):
    url = INDEX_SPECS[index_key][2]
    response = requests.get(url, timeout=30, headers={"User-Agent": "StockMonitor/4.7 (personal research; constituent table reader)"})
    response.raise_for_status()
    symbols = parse_constituents(pd.read_html(StringIO(response.text)), index_key)
    return {"symbols": symbols, "source": url, "fetched_at": hkt_now().isoformat(), "effective_date_verified": False}


def completed_session(market_code, now=None):
    import exchange_calendars as xcals
    now = pd.Timestamp(now or datetime.now(ZoneInfo("UTC")))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    calendar = xcals.get_calendar("XHKG" if market_code == "HK" else "XNYS")
    schedule = calendar.schedule.loc[(now - pd.Timedelta(days=35)).strftime("%Y-%m-%d"):now.strftime("%Y-%m-%d")]
    close_col = "close" if "close" in schedule.columns else "market_close"
    closes = pd.to_datetime(schedule[close_col], utc=True)
    # 收市後額外等候15分鐘，涵蓋收市競價及供應商更新緩衝。
    eligible = schedule.index[closes + pd.Timedelta(minutes=15) <= now]
    if len(eligible) == 0:
        raise ValueError("交易日曆找不到已完成交易日；停止排名，不以星期一至五猜測")
    return str(pd.Timestamp(eligible[-1]).date())


def normalize_daily_frame(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    df.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rank_prices(symbols, market_date):
    output = {}
    target = pd.Timestamp(market_date)
    start = (target - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    end = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for offset in range(0, len(symbols), 80):
        chunk = list(symbols[offset:offset+80])
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=False,
                                  group_by="ticker", threads=3, progress=False)
                break
            except Exception as exc:
                if attempt == 1:
                    add_error("批量排名區塊失敗；計入缺失而非冒充完整", exc=exc)
                else:
                    time.sleep(1)
        if raw is None or raw.empty:
            continue
        for ticker in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        frame = raw[ticker]
                    else:
                        frame = raw.xs(ticker, axis=1, level=1)
                else:
                    frame = raw if len(chunk) == 1 else None
                frame = normalize_daily_frame(frame)
                if frame is not None and target in frame.index:
                    row = frame.loc[target]
                    close, vol = safe_float(row.get("close")), safe_float(row.get("volume"))
                    if np.isfinite(close) and close > 0 and np.isfinite(vol) and vol > 0:
                        output[ticker] = {"estimate": float(close * vol), "price_date": market_date}
            except (KeyError, ValueError, TypeError):
                continue
    return output


def snapshot_turnovers(symbols, market_date):
    output = {}
    if not quote_ctx:
        return output
    for offset in range(0, len(symbols), 300):
        chunk = list(symbols[offset:offset+300])
        reverse = {to_futu(t): t for t in chunk}
        try:
            with FUTU_LOCK:
                ret, frame = quote_ctx.get_market_snapshot(list(reverse))
            if ret != RET_OK or frame is None:
                continue
            for _, row in frame.iterrows():
                ticker = reverse.get(row.get("code"))
                amount = safe_float(row.get("turnover"))
                update = str(row.get("update_time", ""))[:10]
                if ticker and update == market_date and np.isfinite(amount) and amount > 0:
                    output[ticker] = {"amount": float(amount), "price_date": update}
        except Exception as exc:
            add_error("成交額快照不可用，改用明確標示的估算額", exc=exc)
    return output


def select_rank(symbols, estimates, actuals, market_date):
    complete_actual = all(t in actuals and actuals[t]["price_date"] == market_date for t in symbols)
    method = "實際成交額（富途快照）" if complete_actual else "估算成交額（未復權收市價×成交量）"
    rows = []
    for ticker in symbols:
        if complete_actual:
            amount = actuals[ticker]["amount"]
        elif ticker in estimates and estimates[ticker]["price_date"] == market_date:
            amount = estimates[ticker]["estimate"]
        else:
            continue
        rows.append({"ticker": ticker, "amount": amount})
    rows.sort(key=lambda x: (-x["amount"], x["ticker"]))
    top = [dict(row, rank=i+1) for i, row in enumerate(rows[:30])]
    return {"market_date": market_date, "expected": len(symbols), "available": len(rows),
            "complete": len(rows) == len(symbols), "method": method, "top": top}


def build_auto_pools(index_keys, progress=None):
    dates, constituents, errors = {}, {}, []
    for key in index_keys:
        try:
            dates[key] = completed_session(INDEX_SPECS[key][1])
            constituents[key] = fetch_constituents(key)
            state_put("constituents:"+key, constituents[key])
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    quotes, actuals = {}, {}
    for date in sorted(set(dates.values())):
        syms = tuple(sorted({t for k, c in constituents.items() if dates.get(k) == date for t in c["symbols"]}))
        if not syms:
            continue
        if progress:
            progress.text(f"批量取得 {date} 的成交資料：{len(syms)} 個成分股…")
        try:
            quotes[date] = fetch_rank_prices(syms, date)
        except Exception as exc:
            errors.append(f"{date} 排名價格下載失敗：{exc}")
            quotes[date] = {}
        actuals[date] = snapshot_turnovers(syms, date)
    pools = []
    for key in index_keys:
        rank = None
        if key in constituents and key in dates:
            rank = select_rank(constituents[key]["symbols"], quotes.get(dates[key], {}), actuals.get(dates[key], {}), dates[key])
            rank.update({"index_key": key, "market": INDEX_SPECS[key][1],
                         "constituent_source": constituents[key]["source"],
                         "constituents_fetched_at": constituents[key]["fetched_at"],
                         "effective_date_verified": False, "generated_at": hkt_now().isoformat()})
        if rank and rank["complete"]:
            rank["status"] = "完整覆蓋；成分生效日未經官方核實"
            rank["fallback"] = False
            state_put("ranking:"+key, rank)
            with sqlite3.connect(DB_FILE, timeout=30) as conn:
                conn.execute("INSERT OR REPLACE INTO auto_pool_history VALUES (?,?,?,?)", (key, rank["market_date"], json.dumps(rank, ensure_ascii=False), hkt_now().isoformat()))
        else:
            previous = state_get("ranking:"+key)
            if previous:
                previous.update({"fallback": True, "status": "回退：上一份完整榜單（不是本次更新）",
                                 "requested_date": dates.get(key), "current_available": rank["available"] if rank else 0})
                rank = previous
            elif rank:
                rank.update({"fallback": False, "status": "不完整排名：只供參考，不作完整Top30"})
            else:
                rank = {"index_key": key, "market": INDEX_SPECS[key][1], "market_date": dates.get(key, "未知"),
                        "expected": 0, "available": 0, "complete": False, "method": "無可用資料", "top": [],
                        "fallback": False, "status": "成分或行情更新失敗；沒有固定熱門股替代"}
        pools.append(rank)
    return pools, errors


def run_decision_scan(tickers, pools, progress):
    meta = {}
    for pool in pools:
        for row in pool["top"]:
            meta.setdefault(row["ticker"], []).append({"index": pool["index_key"], "rank": row["rank"],
                "amount": row["amount"], "currency": "HKD" if pool["market"] == "HK" else "USD",
                "ranking_date": pool["market_date"], "method": pool["method"],
                "ranking_complete": pool["complete"], "fallback": pool["fallback"]})
    contexts = {}
    for code in sorted({"HK" if t.endswith(".HK") or t == "^HSI" else "US" for t in tickers}):
        label = "🇭🇰 港股" if code == "HK" else "🇺🇸 美股"
        target = completed_session(code)
        regime_local = classify_market_state(label, fetch_macro())
        benchmark = fetch_ohlcv(regime_local["benchmark"], "2y")
        benchmark = trim_completed(benchmark, target)
        contexts[code] = (target, regime_local, benchmark)
    results, errors, infos = [], [], {}
    core_set = set(permanent_tickers())
    progress.progress(10, text="下載入選股票基本面（保留基本面快取）…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
        futures = {executor.submit(get_full_stock_info, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                infos[ticker] = future.result()
            except Exception as exc:
                errors.append(f"{ticker} 基本面：{exc}")
    progress.progress(35, text="下載入選股日線…")
    data = fetch_multiple(tickers, period="2y")
    for i, ticker in enumerate(tickers):
        code = "HK" if ticker.endswith(".HK") or ticker == "^HSI" else "US"
        target, regime_local, benchmark = contexts[code]
        try:
            frame = trim_completed(data.get(ticker), target)
            if frame is None or frame.empty or str(frame.index[-1].date()) != target:
                raise ValueError(f"缺少最新完整交易日 {target} 日線；不產生舊資料新訊號")
            if benchmark is None or benchmark.empty or str(benchmark.index[-1].date()) != target:
                raise ValueError("基準價格未更新，暫停本市場訊號")
            result = decision_engine(ticker, frame, infos.get(ticker, {}), regime_local, benchmark)
            if result is None:
                raise ValueError("歷史不足210根日線")
            result.update({"market": code, "scanned_at": hkt_now().isoformat(),
                           "pool_memberships": meta.get(ticker, []),
                           "pool_sources": " / ".join(f"{x['index']} #{x['rank']} ({x['ranking_date']})" for x in meta.get(ticker, [])) or "永久／自選名單"})
            memberships = meta.get(ticker, [])
            current_pool_ok = any(x["ranking_complete"] and not x["fallback"] and x["ranking_date"] == target for x in memberships)
            result["formal_signal_eligible"] = current_pool_ok or ticker in core_set or not pools
            save_snapshot_and_signal(result)
            results.append(result)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
        progress.progress(35 + int((i+1)/max(1,len(tickers))*65), text=f"決策分析 {i+1}/{len(tickers)}")
    return results, errors


def trim_completed(frame, target):
    frame = normalize_daily_frame(frame)
    return frame.loc[frame.index <= pd.Timestamp(target)].copy() if frame is not None else None


def render_pool_report(pools):
    if not pools:
        return
    report = [{"指數": p["index_key"], "交易日": p["market_date"], "預期成分": p["expected"],
               "有效行情": p["available"], "缺失": p["expected"]-p["available"], "計算方法": p["method"], "本次要求日期": p.get("requested_date", p["market_date"]), "本次有效行情": p.get("current_available", p["available"]), "狀態": p["status"]} for p in pools]
    st.dataframe(pd.DataFrame(report), hide_index=True, use_container_width=True)
    st.caption("成分來源：Wikipedia 公開表格，非官方授權成分服務；抓取時間不等於成分生效日。完整覆蓋只指該次取得的成分表，不保證官方成分及生效日已核實。估算額不等於逐筆成交金額加總。")
    with st.expander("各指數Top30、幣種、資料時間與成分來源"):
        for pool in pools:
            st.write(f"{pool['index_key']}｜{pool['market_date']}｜{pool['method']}｜{pool['status']}")
            st.caption(f"成分來源：{pool.get('constituent_source', '不可用')}｜抓取：{pool.get('constituents_fetched_at','未知')}")
            if pool["top"]:
                frame = pd.DataFrame(pool["top"])
                frame["currency"] = "HKD" if pool["market"] == "HK" else "USD"
                st.dataframe(frame, hide_index=True, use_container_width=True)


init_pool_db()


elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, AUTO_REFRESH_SEC - int(elapsed))
mins, secs = divmod(remaining, 60)
refresh_requested = False
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    st.caption("刷新會更新四指數Top30及深度掃描；改選頁面本身不會重掃。")
    market = st.radio("市場", ["🌐 四指數", "🇭🇰 港股", "🇺🇸 美股", "📋 自選"], index=0)
    custom_input = ""
    if market == "📋 自選":
        custom_input = st.text_area("輸入代碼（每行一個）", "0700.HK\nNVDA")
    include_core = st.checkbox("合併永久自選／持倉名單", value=True)
    force_full = st.checkbox("強制重抓成分及基本面（較慢）", value=False)
    refresh_requested = st.button("🔄 刷新資料並自動掃描", type="primary")
    view_mode = st.radio("顯示模式", ["新手模式", "進階模式"], index=0)
    st.caption("預設：各市場最新完整交易日；收市後另等15分鐘。此版不包含盤中榜及背景排程。")
    st.caption("富途已連接" if quote_ctx else "富途未連接：使用Yahoo；成交額不足時明確標示估算。")
    with st.expander("永久名單管理／備份"):
        additions = st.text_area("新增永久代碼（每行一個；港股如0700.HK）", key="core_additions")
        if st.button("加入永久名單"):
            valid = [normalize_ticker(t.strip(), t.strip().upper().endswith(".HK")) for t in additions.splitlines() if t.strip()]
            if any(t is None for t in valid):
                st.error("有不合法代碼；本次未寫入。")
            else:
                with sqlite3.connect(DB_FILE, timeout=30) as conn:
                    conn.executemany("INSERT OR IGNORE INTO permanent_watchlist VALUES (?,?,?)", [(t,"手動新增／持倉",hkt_now().isoformat()) for t in valid])
                st.success("已加入；按刷新將納入掃描。")
        st.download_button("匯出永久名單備份", "\n".join(permanent_tickers()), "permanent_watchlist.txt", mime="text/plain")
        st.caption("本地SQLite並非雲端永久儲存保證；持倉股請加入這份名單，未接券商持倉同步。")

context_key = json.dumps([market, custom_input, include_core], ensure_ascii=False)
if st.session_state.get("result_context") != context_key:
    st.session_state["latest_results"] = []
    st.session_state["pool_report"] = []
    st.session_state["scan_errors"] = []
    st.session_state["result_context"] = context_key
if refresh_requested:
    st.session_state.last_refresh = time.time()
    fetch_ohlcv.clear()
    fetch_macro.clear()
    fetch_rank_prices.clear()
    if force_full:
        fetch_constituents.clear()
        get_full_stock_info.clear()
    clear_errors()

macro_data = fetch_macro()
regime = classify_market_state(market if market != "📋 自選" else "🇺🇸 美股", macro_data)
benchmark_df = fetch_ohlcv(regime["benchmark"], period="2y")

st.markdown("# 📈 股票監察系統 Pro+｜V4.7 四指數Top30自動掃描")
st.caption(f"模型：{MODEL_VERSION}｜程式版本：{APP_VERSION}｜頁面生成：{hkt_now().strftime('%Y-%m-%d %H:%M:%S %Z')} HKT")
st.warning("本系統用於研究與風險管理，不構成投資建議。任何「可小量試倉」都不是預測見底或保證收益；必須以可執行價格、10日結構支持止損與組合風險為準。")

tabs = st.tabs([
    "🌍 市場氣氛", "📊 決策掃描", "💎 估值候選", "📈 技術圖表 (含TSPR及跌幅統計)", "📐 回撤/江恩",
    "🧪 歷史訊號回測", "⚖️ 組合風控", "🌟 因子選股", "🛠️ 系統健康"
])

with tabs[0]:
    st.subheader("🌍 宏觀市場氣氛")
    render_metrics(macro_data, regime)
    st.markdown("""
**如何使用 Regime 與 TSPR V4.6：** 系統現已整合 **TSPR 10日結構支持位**、**江恩 50% 分水嶺** 與 **OBV / MFI 資金流破位確認**，並自動計算每次跌穿支持後的 **最大下跌幅度 (%)** 與 **頂部至低點總跌幅 (%)**。
""")

with tabs[1]:
    st.subheader("📊 決策掃描｜資格 → 估值 → TSPR 10日支持/江恩50%/OBV確認 → 下跌幅度與風險")
    manual_scan = st.button("🚀 更新股票池並執行決策掃描", type="primary", key="decision_scan")
    st.caption("四指數各取成交額Top30 → 合併去重 → 加入永久名單 → 個別市場基準分析。納指預設Nasdaq-100，不是綜合指數。")
    if refresh_requested or manual_scan:
        st.session_state["latest_results"] = []
        st.session_state["scan_errors"] = []
        progress = st.progress(0, text="檢查交易日與成分股…")
        pools, errors = [], []
        try:
            if market == "📋 自選":
                tickers = sorted({t.strip().upper() for t in custom_input.splitlines() if t.strip()})
            else:
                keys = ["DJIA", "NDX", "SP500", "HSI"] if market == "🌐 四指數" else (["HSI"] if market == "🇭🇰 港股" else ["DJIA","NDX","SP500"])
                pools, errors = build_auto_pools(keys, progress)
                tickers = sorted({row["ticker"] for pool in pools for row in pool["top"]})
                if include_core:
                    core = permanent_tickers()
                    if market == "🇭🇰 港股":
                        core = [t for t in core if t.endswith(".HK")]
                    elif market == "🇺🇸 美股":
                        core = [t for t in core if not t.endswith(".HK")]
                    tickers = sorted(set(tickers) | set(core))
            if not tickers:
                raise ValueError("沒有可掃描股票；請檢查成分及行情來源，沒有以固定名單冒充Top30。")
            results, scan_errors = run_decision_scan(tickers, pools, progress)
            errors.extend(scan_errors)
            st.session_state["latest_results"] = results
            st.session_state["scan_summary"] = {"requested":len(tickers), "successful":len(results), "finished_at":hkt_now().isoformat()}
        except Exception as exc:
            errors.append(f"掃描中止：{type(exc).__name__}: {exc}")
        finally:
            progress.empty()
            st.session_state["pool_report"] = pools
            st.session_state["scan_errors"] = errors
    render_pool_report(st.session_state.get("pool_report", []))
    st.caption("formal_signal_eligible=False 表示只供研究；不會寫入正式試倉訊號。回退榜及不完整榜的新動態候選會被此閘門攔截。")
    for message in st.session_state.get("scan_errors", []):
        st.warning(message)
    if st.session_state.get("latest_results"):
        summary = st.session_state.get("scan_summary", {})
        st.caption(f"本輪請求 {summary.get('requested',0)} 隻｜有效 {summary.get('successful',0)} 隻｜完成 {summary.get('finished_at','未知')}")

    results = st.session_state.get("latest_results", [])
    if results:
        df_results = pd.DataFrame(results).sort_values(["decision", "final_score"], ascending=[True, False])
        candidate_order = {"可小量試倉": 0, "等待突破": 1, "估值觀察": 2, "不合資格": 3}
        df_results["_rank"] = df_results["decision"].map(candidate_order).fillna(9)
        df_results = df_results.sort_values(["_rank", "final_score"], ascending=[True, False]).drop(columns="_rank")
        candidates = df_results[df_results["decision"].isin(["可小量試倉", "等待突破", "估值觀察"])]
        st.success(f"掃描完成：{len(df_results)} 個有效結果；研究候選 {len(candidates)} 個；模型試倉候選 {sum(df_results['decision'] == '可小量試倉')} 個（須另看正式訊號資格）。")
        display_cols = [
            "ticker", "name", "pool_sources", "formal_signal_eligible", "price", "decision", "tspr_status", "tspr_support", "gann_50",
            "cycle_drop_pct", "peak_to_trough_drop_pct", "post_break_drop_pct",
            "undervaluation", "final_score", "confirm_score", "risk_score",
            "confirmation_price", "stop_loss", "target_2r", "last_bar"
        ]
        st.dataframe(candidates[display_cols], use_container_width=True, hide_index=True)
        if view_mode == "進階模式":
            st.markdown("### 完整審核表")
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        for _, row in candidates.head(8).iterrows():
            color = C_GREEN if row["decision"] == "可小量試倉" else C_ORANGE
            st.markdown(
                f"<div class='decision-card' style='border-left-color:{color}'>"
                f"<b>{row['ticker']} ({row['name']})｜{row['decision']}｜{row['tspr_status']}｜{row['undervaluation']}</b><br>"
                f"原因：{row['reasons']}<br>"
                f"📉 <b>下跌幅度統計</b>：現價距週期高點 <b>{row['cycle_drop_pct']}%</b>｜週期頂至底最大跌幅 <b>{row['peak_to_trough_drop_pct']}%</b>｜破位後已跌 <b>{row['post_break_drop_pct']}%</b><br>"
                f"🎯 <b>關鍵位置</b>：10日支持位：{row['tspr_support']}｜江恩50%：{row['gann_50']}｜確認價：{row['confirmation_price']}｜止損：{row['stop_loss']}｜2R：{row['target_2r']}<br>"
                f"下一步：{row['next_step']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("按「執行 V4.6 決策掃描」後會顯示研究候選。")

with tabs[2]:
    st.subheader("💎 估值候選｜先看便宜，再看是否值得買")
    results = st.session_state.get("latest_results", [])
    if not results:
        st.info("請先在「決策掃描」完成掃描。")
    else:
        val_df = pd.DataFrame(results)
        val_df = val_df[val_df["undervaluation"].isin(["🟢 重度被低估", "🟠 中度被低估", "🟡 輕度被低估"])]
        st.dataframe(
            val_df[[
                "ticker", "name", "undervaluation", "valuation_score", "tspr_status",
                "tspr_support", "cycle_drop_pct", "peak_to_trough_drop_pct",
                "quality_flags", "decision", "confirm_score", "risk_score", "valuation_detail", "blockers"
            ]].sort_values("valuation_score", ascending=False),
            use_container_width=True, hide_index=True
        )

with tabs[3]:
    st.subheader("📈 技術圖表｜TSPR 10日支持位、下跌幅度統計、江恩 50% 與 OBV/MFI 確認")

    col_q1, col_q2, col_q3 = st.columns([2, 2, 3])
    with col_q1:
        quick_pick = st.selectbox(
            "快速切換大市指數 / 個股",
            ["自訂輸入", "^HSI (恒生指數)", "3033.HK (南方恒生科技)", "^NDX (納斯達克100)", "^GSPC (標普500)", "0700.HK (騰訊)", "NVDA (Nvidia)"],
            index=1
        )
    with col_q2:
        default_tk = "^HSI" if quick_pick == "^HSI (恒生指數)" else quick_pick.split(" ")[0] if quick_pick != "自訂輸入" else "AAPL"
        ticker = st.text_input("股票或指數代碼", default_tk, key="chart_ticker").upper()
    with col_q3:
        period_map = {"3個月": "3mo", "6個月": "6mo", "1年": "1y", "2年": "2y"}
        period = st.radio("時間範圍", list(period_map.keys()), horizontal=True, index=2)

    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
    show_tspr_labels = t_col1.checkbox("顯示 10日支持與破位跌幅標籤", value=True)
    show_tspr_boxes = t_col2.checkbox("顯示紫色週期形態框", value=True)
    require_gann_obv = t_col3.checkbox("開啟江恩50%+OBV破位過濾", value=False, help="開啟後，只有同時跌穿江恩50%或OBV/MFI轉弱的真破位才會觸發紅色破位標籤")
    left_bars_input = t_col4.number_input("支持位左側週期 (日)", min_value=5, max_value=30, value=10)
    box_period_input = t_col5.number_input("形態箱體週期 (日)", min_value=30, max_value=120, value=65)

    chart_df = fetch_ohlcv(ticker, period=period_map[period])
    if chart_df is not None and len(chart_df) > 30:
        close = chart_df["close"]
        rsi = calc_rsi(close)
        weekly_rsi, _ = calc_weekly_rsi_daily_aligned(close)
        macd, signal, hist = calc_macd(close)
        avwap = calc_avwap_from_recent_low(chart_df)
        obv_s, obv_ma10_s = calc_obv(chart_df)
        mfi_s = calc_mfi(chart_df, 14)

        # 執行 TSPR V4.6 計算
        tspr_res = calc_tspr_system(
            chart_df,
            left_bars=int(left_bars_input),
            right_bars=5,
            box_period=int(box_period_input),
            num_boxes=3,
            require_filter=require_gann_obv
        )
        st_data = tspr_res["stats"]

        # 頂部復刻圖二右上角 TSPR 量化統計面板（含完整下跌幅度指標）
        st.markdown("<div class='tspr-panel'>", unsafe_allow_html=True)
        r1_cols = st.columns(5)
        r1_cols[0].metric("週期最高", f"{st_data.get('cycle_high', 0):,.2f}")
        r1_cols[1].metric("週期最低", f"{st_data.get('cycle_low', 0):,.2f}")
        r1_cols[2].metric("週期波幅 (點數)", f"{st_data.get('cycle_range', 0):,.2f}")
        r1_cols[3].metric("頂部至低點跌幅", f"{st_data.get('peak_to_trough_drop_pct', 0):.2f}%", delta=f"現價距頂 {st_data.get('current_drop_from_high_pct', 0):.2f}%", delta_color="inverse")
        r1_cols[4].metric("調整比率 (波幅佔比)", f"{st_data.get('retracement_pct', 0):.1f}%")

        r2_cols = st.columns(5)
        r2_cols[0].metric("10日關鍵支持位", f"{st_data.get('active_support', np.nan):,.2f}" if not np.isnan(st_data.get("active_support", np.nan)) else "N/A")
        r2_cols[1].metric("週期江恩 50% 分水嶺", f"{st_data.get('cycle_gann_50', 0):,.2f}", delta="已失守" if st_data.get("below_gann_50") else "守穩中", delta_color="inverse" if st_data.get("below_gann_50") else "normal")
        r2_cols[2].metric("歷次破位平均再跌幅度", f"{st_data.get('avg_post_break_drop_pct', 0):.2f}%", delta=f"當前破位已跌 {st_data.get('active_post_break_drop_pct', 0):.2f}%")
        r2_cols[3].metric("黃金比率 1.236倍 (上/下)", f"{st_data.get('fib_1236', 0):,.0f} / {st_data.get('fib_1236_down', 0):,.0f}")
        r2_cols[4].metric("TSPR 趨勢狀態", st_data.get("status", "N/A"))
        st.markdown("</div>", unsafe_allow_html=True)

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.50, 0.12, 0.12, 0.13, 0.13],
            subplot_titles=("K線 + TSPR 10日支持與破位跌幅 + 紫色週期框", "成交量", "OBV 能量潮 vs MA10", "日 RSI / 真周 RSI / MFI(14)", "MACD")
        )
        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"], close=close,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="K線"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=close.rolling(20).mean(), name="MA20", line=dict(color="#f0883e", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=close.rolling(50).mean(), name="MA50", line=dict(color=C_BLUE, width=1.2)), row=1, col=1)
        if not avwap.empty:
            fig.add_trace(go.Scatter(x=avwap.index, y=avwap, name="Anchored VWAP（近60日低點）", line=dict(color=C_PURPLE, dash="dot")), row=1, col=1)

        # 繪製當前週期江恩 50% 水平參考線
        fig.add_hline(
            y=st_data.get("cycle_gann_50", 0), line_dash="dot", line_color="#d29922",
            annotation_text=f"江恩50% ({st_data.get('cycle_gann_50', 0):.1f})",
            annotation_position="bottom right", row=1, col=1
        )

        # 繪製 TSPR 紫色形態箱體 (連同箱體頂底跌幅標註)
        if show_tspr_boxes:
            for idx_b, box in enumerate(tspr_res["pattern_boxes"]):
                fig.add_shape(
                    type="rect",
                    x0=box["start_date"], y0=box["low"],
                    x1=box["end_date"], y1=box["high"],
                    line=dict(color="rgba(188, 140, 255, 0.75)", width=1.5, dash="dash"),
                    fillcolor="rgba(74, 20, 140, 0.16)",
                    layer="below", row=1, col=1
                )
                fig.add_annotation(
                    x=box["start_date"], y=box["high"],
                    text=f"週期框#{idx_b+1} 頂底跌幅: {box['drop_pct']:.1f}%",
                    showarrow=False, xanchor="left", yanchor="bottom",
                    font=dict(color="#bc8cff", size=10),
                    bgcolor="rgba(22, 27, 34, 0.8)",
                    row=1, col=1
                )

        # 繪製 TSPR 10日支持水平虛線、綠色支持標籤與紅色跌穿標籤（直接顯示破位跌幅 %）
        if show_tspr_labels:
            for s_line in tspr_res["support_lines"]:
                line_col = "#ef5350" if s_line["broken"] else "#3fb950"
                fig.add_shape(
                    type="line",
                    x0=s_line["start_date"], y0=s_line["price"],
                    x1=s_line["end_date"], y1=s_line["price"],
                    line=dict(color=line_col, width=1.5, dash="dash"),
                    row=1, col=1
                )
            for ev in tspr_res["support_events"]:
                fig.add_annotation(
                    x=ev["date"], y=ev["price"],
                    text=f"10日支持<br>{ev['price']:.1f}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                    arrowcolor="#2e7d32", ax=0, ay=28,
                    bgcolor="#1b5e20", bordercolor="#4caf50",
                    font=dict(color="white", size=9),
                    row=1, col=1
                )
            for brk in tspr_res["breakdown_events"]:
                tag_color = "#b71c1c" if (brk["gann_broken"] or brk["flow_confirmed"]) else "#bf360c"
                fig.add_annotation(
                    x=brk["date"], y=brk["high"],
                    text=(
                        f"跌穿支持 {brk['price']:.1f}<br>"
                        f"破位再跌: {brk['post_break_drop_pct']:.1f}% (頂跌:{brk['peak_to_trough_drop_pct']:.1f}%)<br>"
                        f"{brk['confirm_label']}"
                    ),
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
                    arrowcolor="#ef5350", ax=0, ay=-38,
                    bgcolor=tag_color, bordercolor="#ef5350",
                    font=dict(color="white", size=9),
                    row=1, col=1
                )

        # 第 2 行：成交量
        fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["volume"], name="成交量", marker_color=[C_GREEN if c >= o else C_RED for c, o in zip(chart_df["close"], chart_df["open"])]), row=2, col=1)
        # 第 3 行：OBV 與 OBV 10日均線
        fig.add_trace(go.Scatter(x=chart_df.index, y=obv_s, name="OBV", line=dict(color=C_BLUE, width=1.3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=obv_ma10_s, name="OBV MA10", line=dict(color=C_ORANGE, dash="dot", width=1.2)), row=3, col=1)
        # 第 4 行：RSI / 周RSI / MFI
        fig.add_trace(go.Scatter(x=chart_df.index, y=rsi, name="日 RSI(14)", line=dict(color=C_ORANGE)), row=4, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=weekly_rsi, name="真周 RSI(14)", line=dict(color=C_PURPLE, dash="dot")), row=4, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=mfi_s, name="MFI(14)", line=dict(color="#26a69a", width=1)), row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=C_GREY, row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=C_GREY, row=4, col=1)
        # 第 5 行：MACD
        fig.add_trace(go.Bar(x=chart_df.index, y=hist, name="MACD Histogram", marker_color=[C_GREEN if x >= 0 else C_RED for x in hist.fillna(0)]), row=5, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=macd, name="MACD", line=dict(color=C_BLUE)), row=5, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=signal, name="Signal", line=dict(color=C_ORANGE)), row=5, col=1)

        fig.update_layout(height=1050, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, key=f"technical_chart_{ticker}_{period}")

        # 新增：歷次跌穿 10 日支持位之下跌幅度明細表
        st.markdown("### 📉 歷次跌穿 10 日支持位之下跌幅度與確認統計表")
        if tspr_res["breakdown_events"]:
            brk_rows = []
            for b_ev in tspr_res["breakdown_events"][::-1]:
                brk_rows.append({
                    "跌穿日期": str(pd.Timestamp(b_ev["date"]).date()),
                    "原10日支持位": b_ev["broken_support"],
                    "跌穿日收市價": round(b_ev["price"], 2),
                    "破位前波段高點": b_ev["wave_high"],
                    "江恩50%分水嶺": b_ev["gann_50"],
                    "調整最低點日期": str(pd.Timestamp(b_ev["trough_date"]).date()) if b_ev["trough_date"] is not None else "進行中",
                    "調整最低價": b_ev["trough_price"],
                    "破位後再跌點數": b_ev["post_break_drop_pts"],
                    "破位後最大跌幅 (%)": f"{b_ev['post_break_drop_pct']:.2f}%",
                    "頂部至低點總跌幅 (%)": f"{b_ev['peak_to_trough_drop_pct']:.2f}%",
                    "下跌持續日數": f"{b_ev['duration_days']} 日",
                    "江恩50% / OBV / MFI 確認狀態": b_ev["confirm_label"]
                })
            st.dataframe(pd.DataFrame(brk_rows), use_container_width=True, hide_index=True)
        else:
            st.info("所選時間範圍內未出現跌穿 10 日支持位事件。")

with tabs[4]:
    st.subheader("📐 回撤、斐波那契與江恩觀察位")
    ticker = st.text_input("股票代碼", "NVDA", key="levels_ticker").upper()
    if st.button("計算觀察位", key="calc_levels"):
        df = fetch_ohlcv(ticker, period="2y")
        if df is None or df.empty:
            st.error("無法取得價格數據。")
        else:
            current = safe_float(df["close"].iloc[-1])
            high = get_52w_high(df)
            low = safe_float(df["low"].iloc[-252:].min())
            tspr_info = calc_tspr_system(df)
            st_i = tspr_info["stats"]
            cols = st.columns(3)
            cols[0].dataframe(pd.DataFrame([{"高點回撤比例": k, "目標價位": v} for k, v in drop_levels(high).items()]), hide_index=True)
            cols[1].dataframe(pd.DataFrame([{"斐波那契回調": k, "目標價位": v} for k, v in fib_levels(low, high).items()]), hide_index=True)
            cols[2].dataframe(pd.DataFrame([{"江恩回調比例": k, "目標價位": v} for k, v in gann_levels(low, high).items()]), hide_index=True)
            st.caption(
                f"現價：{current:.3f}｜TSPR 10日支持位：{tspr_info['active_support']}｜週期江恩50%：{st_i.get('cycle_gann_50')}｜"
                f"週期頂至底跌幅：{st_i.get('peak_to_trough_drop_pct')}%｜狀態：{tspr_info['status']}"
            )

with tabs[5]:
    st.subheader("🧪 歷史訊號追蹤回測（不是滾動訓練／調參驗證）")
    st.markdown("信號在收市後產生；回測以**下一交易日開盤價加滑點**進場。納入止損、2R 目標、固定持有期、成本；若同一日同時觸及止損與目標，保守地按先止損處理。")
    try:
        conn = sqlite3.connect(DB_FILE)
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC", conn)
        conn.close()
    except Exception as exc:
        add_error("讀取回測信號失敗", exc=exc)
        signals = pd.DataFrame()
    if signals.empty:
        st.info("尚無「可小量試倉」信號。先運行決策掃描，系統會按信號日、代碼和模型版本去重記錄。")
    else:
        st.dataframe(signals[["signal_date", "ticker", "decision", "total_score", "price", "confirmation_price", "stop_loss", "target_2r", "market_regime", "model_version"]], use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hold_days = st.selectbox("最長持有交易日", [10, 20, 30, 45], index=1)
        with c2:
            cost_bps = st.number_input("單邊交易成本（bps）", min_value=0, max_value=100, value=TRANSACTION_COST_BPS)
        with c3:
            slip_bps = st.number_input("入場滑點（bps）", min_value=0, max_value=100, value=DEFAULT_SLIPPAGE_BPS)
        if st.button("運行嚴格回測", type="primary", key="strict_bt"):
            bt = run_walk_forward_backtest(signals, hold_days, cost_bps, slip_bps)
            if bt.empty:
                st.info("現有信號尚未成熟，或歷史價格不足以完成所選持有期。")
            else:
                st.dataframe(bt, use_container_width=True, hide_index=True)
                cols = st.columns(4)
                cols[0].metric("成熟樣本", len(bt))
                cols[1].metric("勝率", f"{(bt['淨回報%'] > 0).mean() * 100:.1f}%")
                cols[2].metric("平均淨回報", f"{bt['淨回報%'].mean():.2f}%")
                cols[3].metric("中位數淨回報", f"{bt['淨回報%'].median():.2f}%")

with tabs[6]:
    st.subheader("⚖️ 組合風控與部位計算")
    c1, c2, c3 = st.columns(3)
    with c1:
        account_size = st.number_input("賬戶總值（USD/HKD）", min_value=1000.0, value=100000.0, step=1000.0)
    with c2:
        risk_pct = st.slider("單筆最大賬戶風險 (%)", 0.25, 3.0, 1.0, 0.25) / 100
    with c3:
        max_notional = st.slider("單一持倉名義上限 (%)", 3, 25, 10) / 100
    ticker = st.text_input("計劃交易代碼", "AAPL", key="portfolio_ticker").upper()
    if st.button("計算風險部位", key="position_calc"):
        df = fetch_ohlcv(ticker, period="1y")
        if df is None or len(df) < 30:
            st.error("價格資料不足。")
        else:
            price = safe_float(df["close"].iloc[-1])
            atr = safe_float(calc_atr(df).iloc[-1])
            tspr_res = calc_tspr_system(df)
            tspr_sup = tspr_res["active_support"]
            base_stop = min(safe_float(df["low"].iloc[-20:].min()), price - 2 * atr)
            stop = max(base_stop, tspr_sup * 0.995) if (not np.isnan(tspr_sup) and not tspr_res["is_broken"] and tspr_sup < price) else base_stop
            shares, notional, risk_amount = calculate_position(price, stop, account_size, risk_pct, max_notional)
            target = price + 2 * (price - stop)
            cols = st.columns(5)
            cols[0].metric("建議最大股數", f"{shares:,}")
            cols[1].metric("名義金額", f"${notional:,.0f}")
            cols[2].metric("結構/TSPR 止損位", f"{stop:.3f}")
            cols[3].metric("若止損最大虧損", f"${min(risk_amount, shares * max(0, price-stop)):,.0f}")
            cols[4].metric("2R 目標", f"{target:.3f}")
            st.warning("港股下單前必須再按每手股數、貨幣、佣金及實際盤口流動性調整；部位模型不能替代你的總組合行業／相關性限制。")

with tabs[7]:
    st.subheader("🌟 多因子選股｜價值、質量、動能、低波動")
    st.caption("此模組適合 3–6 個月以上研究；與短線反轉系統不同。金融、虧損企業和 ETF 不適合只用 PE 與 ROE 橫向比較。")
    if st.button("執行多因子運算", type="primary", key="multi_factor"):
        tickers = sorted({r["ticker"] for r in st.session_state.get("latest_results", [])})
        if not tickers:
            st.info("請先刷新並完成決策掃描；因子選股會共用同一股票池。")
        info_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            futures = {executor.submit(get_full_stock_info, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    info_map[ticker] = future.result()
                except Exception as exc:
                    add_error("因子基本面失敗", ticker, exc)
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
                "代碼": ticker, "名稱": info.get("name", ticker), "PE": safe_float(info.get("pe")),
                "ROE%": safe_float(info.get("roe")) * 100, "6M動能%": (close.iloc[-1] / close.iloc[-126] - 1) * 100,
                "年化波動%": returns.std() * np.sqrt(252) * 100
            })
        if records:
            mf = pd.DataFrame(records)
            for col in ["PE", "ROE%", "6M動能%", "年化波動%"]:
                mf[col] = mf[col].replace([np.inf, -np.inf], np.nan)
                mf[col] = mf[col].fillna(mf[col].median())

            def zscore(series, inverse=False):
                std = series.std()
                z = pd.Series(0.0, index=series.index) if std == 0 or pd.isna(std) else (series - series.mean()) / std
                return -z if inverse else z

            mf["價值Z"] = zscore(mf["PE"], True)
            mf["質量Z"] = zscore(mf["ROE%"])
            mf["動能Z"] = zscore(mf["6M動能%"])
            mf["防禦Z"] = zscore(mf["年化波動%"], True)
            mf["綜合分"] = (mf[["價值Z", "質量Z", "動能Z", "防禦Z"]].sum(axis=1)).round(2)
            mf = mf.sort_values("綜合分", ascending=False)
            st.dataframe(mf, use_container_width=True, hide_index=True)
            fig = px.scatter(mf, x="PE", y="6M動能%", text="代碼", color="綜合分", size=np.maximum(6, (mf["質量Z"] - mf["質量Z"].min() + 0.5) * 8), color_continuous_scale="RdYlGn")
            fig.update_layout(template="plotly_dark", plot_bgcolor=C_BG, paper_bgcolor=C_BG, height=550)
            st.plotly_chart(fig, use_container_width=True, key="factor_scatter")

with tabs[8]:
    st.subheader("🛠️ 系統健康、數據覆蓋與錯誤日誌")
    st.markdown("### 存儲狀態")
    st.warning("目前使用本地 SQLite。若部署在 Streamlit Community Cloud 或無持久磁盤環境，重啟或部署後數據庫可能被清空。正式長期運行請遷移至 Supabase PostgreSQL。")
    try:
        conn = sqlite3.connect(DB_FILE)
        signal_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM signals", conn).iloc[0]["n"]
        snapshot_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM scan_snapshots", conn).iloc[0]["n"]
        conn.close()
        cols = st.columns(2)
        cols[0].metric("已記錄可試倉信號", int(signal_count))
        cols[1].metric("已記錄掃描快照", int(snapshot_count))
    except Exception as exc:
        add_error("讀取數據庫健康狀態失敗", exc=exc)
    logs = st.session_state.get("error_log", [])
    if logs:
        st.warning(f"本次頁面運行記錄到 {len(logs)} 項問題。請先修復數據覆蓋與錯誤，不要僅忽略。")
        st.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("清除本次錯誤日誌", key="clear_logs"):
            clear_errors()
            st.rerun()
    else:
        st.success("本次頁面運行暫未記錄錯誤。注意：這不代表所有歷史數據完整或策略已驗證。")
