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

st.set_page_config(page_title="📈 股票監察系統 V5.0 三引擎版", page_icon="📈", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V5.0-Three-Engine"
MODEL_VERSION = "three-engine-v5.0"
LEGACY_MODEL_VERSIONS = ("decision-v4.7.2-yahoo",)
ANALYSIS_PERIOD = "5y"
OHLCV_TTL = 1800
INFO_TTL = 3600
DB_FILE = os.getenv("STOCK_DB_FILE", "signals.db")
MAX_WORKERS_DATA = 3
MAX_WORKERS_SCORE = 4
TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 10

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
    conn = sqlite3.connect(DB_FILE, timeout=30)
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
    required = {"signal_date","ticker","model_version","decision","total_score","price","confirmation_price",
                "stop_loss","target_2r","market_regime","sector","features_json","created_at"}
    for table, cols in (("signals", required), ("scan_snapshots", {"snapshot_date","ticker","model_version","payload_json","created_at"})):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = cols - existing
        if missing:
            conn.close()
            raise RuntimeError(f"{table} 表缺少欄位 {sorted(missing)}；舊資料庫不相容，請先備份 {DB_FILE} 再遷移")
    conn.commit()
    conn.close()


try:
    init_db()
except Exception as exc:
    st.error(f"資料庫初始化失敗：{exc}。請先備份signals.db，再檢查磁盤權限及資料表結構。")
    st.stop()













@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    try:
        raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                          progress=False, threads=False, timeout=15)
        frame = clean_price_frame(raw)
        if frame is None or frame.empty:
            add_error("Yahoo未回傳可用OHLCV", ticker)
            return None
        return frame
    except Exception as exc:
        add_error("Yahoo OHLCV下載失敗", ticker, exc)
        return None


def fetch_multiple(tickers, period="2y", max_workers=MAX_WORKERS_DATA):
    results = {}
    tickers = tuple(sorted(set(tickers)))
    for start in range(0,len(tickers),50):
        chunk = tickers[start:start+50]
        try:
            frames = download_analysis_batch(chunk, period)
            results.update(frames)
        except Exception as exc:
            add_error("批量日線下載失敗", exc=exc)
        for ticker in chunk:
            results.setdefault(ticker,None)
    return results


@st.cache_data(ttl=INFO_TTL, show_spinner=False)
def get_full_stock_info(ticker):
    result = {"ticker":ticker, "name":ticker, "pe":None, "pb":None, "div_yield":None,
              "roe":None, "de_ratio":None, "rev_growth":None, "eps":None,
              "book_value":None, "sector":None, "industry":None,
              "info_source":"Yahoo Finance", "fundamental_updated":None, "info_error":None,
              "pe_basis":"未知", "fetched_at":hkt_now().isoformat()}
    try:
        info = yf.Ticker(ticker).info or {}
        result["name"] = info.get("shortName") or info.get("longName") or ticker
        trailing = safe_float(info.get("trailingPE"),None)
        forward = safe_float(info.get("forwardPE"),None)
        if trailing is not None and trailing > 0:
            result["pe"], result["pe_basis"] = trailing, "TTM"
        elif forward is not None and forward > 0:
            result["pe"], result["pe_basis"] = forward, "Forward（預測）"
        else:
            eps = safe_float(info.get("trailingEps"),None)
            price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"),None)
            if eps is not None and eps > 0 and price is not None and price > 0:
                result["pe"], result["pe_basis"] = price / eps, "價格/TTM EPS"
        result["pb"] = safe_float(info.get("priceToBook"),None)
        fields = {"roe":"returnOnEquity", "de_ratio":"debtToEquity", "rev_growth":"revenueGrowth",
                  "eps":"trailingEps", "book_value":"bookValue", "sector":"sector", "industry":"industry",
                  "fundamental_updated":"lastFiscalYearEnd"}
        for key, source in fields.items():
            result[key] = info.get(source)
        # Yahoo dividendYield口徑可能變動；優先用每股年息/現價計算比例。
        dividend_rate = safe_float(info.get("dividendRate"))
        market_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if np.isfinite(dividend_rate) and np.isfinite(market_price) and market_price > 0:
            result["div_yield"] = dividend_rate / market_price
        else:
            result["div_yield"] = None
        result.update({"quote_type": info.get("quoteType"), "currency": info.get("currency"),
                       "financial_currency": info.get("financialCurrency") or info.get("currency"),
                       "earnings_growth": info.get("earningsGrowth"), "free_cashflow": info.get("freeCashflow"),
                       "operating_cashflow": info.get("operatingCashflow"), "total_cash": info.get("totalCash"),
                       "total_debt": info.get("totalDebt"), "ebitda": info.get("ebitda"),
                       "shares": info.get("sharesOutstanding"), "forward_eps": info.get("forwardEps")})
        # 以報價貨幣表示每股數據：優先用 價格/倍數 反推，避免港股財報貨幣(CNY/USD)與HKD混用
        px_now = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        fx = get_fx_rate(result["financial_currency"], result["currency"])
        result["fx_rate"], result["fx_ok"] = fx, bool(np.isfinite(fx))
        tpe, fpe, pb = safe_float(info.get("trailingPE")), safe_float(info.get("forwardPE")), safe_float(info.get("priceToBook"))
        eps_raw, feps_raw, bv_raw = safe_float(info.get("trailingEps")), safe_float(info.get("forwardEps")), safe_float(info.get("bookValue"))
        conv = fx if np.isfinite(fx) else np.nan
        result["eps_quote"] = px_now / tpe if np.isfinite(px_now) and np.isfinite(tpe) and tpe > 0 else (eps_raw * conv if np.isfinite(eps_raw) else np.nan)
        result["forward_eps_quote"] = px_now / fpe if np.isfinite(px_now) and np.isfinite(fpe) and fpe > 0 else (feps_raw * conv if np.isfinite(feps_raw) else np.nan)
        result["bvps_quote"] = px_now / pb if np.isfinite(px_now) and np.isfinite(pb) and pb > 0 else (bv_raw * conv if np.isfinite(bv_raw) else np.nan)
        shares = safe_float(info.get("sharesOutstanding"))
        fcf = safe_float(info.get("freeCashflow"))
        result["fcf_ps_quote"] = fcf / shares * conv if np.isfinite(fcf) and np.isfinite(shares) and shares > 0 else np.nan
        history = get_earnings_history(ticker) if np.isfinite(shares) and shares > 0 else []
        result["eps_history_quote"] = [v / shares * conv for v in history if np.isfinite(conv)]
        if not info:
            result["info_error"] = "Yahoo基本面為空"
    except Exception as exc:
        result["info_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        add_error("Yahoo基本面失敗",ticker,exc)
    return result


def get_sector_from_info(ticker, info=None):
    info = info or get_full_stock_info(ticker)
    return str(info.get("sector")) if info.get("sector") else SECTOR_MAP_FALLBACK.get(ticker, "OTHER")


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)
    return rsi.where(delta.notna().cumsum() >= period)


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
            after_low = safe_float(recent["low"].iloc[second + 1:].min(), np.inf)
            still_valid = after_low >= min(low1, low2) * (1 - tolerance / 2) and safe_float(recent["close"].iloc[-1]) > min(low1, low2)
            if still_valid and (neckline - max(low1, low2)) / max(low1, low2) >= min_rebound:
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
    if stock_df is None or benchmark_df is None:
        return np.nan
    joined = pd.concat([stock_df["close"].rename("s"), benchmark_df["close"].rename("b")], axis=1, join="inner").dropna()
    if len(joined) < days + 1:
        return np.nan
    stock_ret = joined["s"].iloc[-1] / joined["s"].iloc[-days - 1] - 1
    bench_ret = joined["b"].iloc[-1] / joined["b"].iloc[-days - 1] - 1
    return float((stock_ret - bench_ret) * 100)


@st.cache_data(ttl=OHLCV_TTL,show_spinner=False)
def fetch_macro():
    result={}
    frames=fetch_multiple(list(MACRO_TICKERS.values()),period="1y")
    for name,ticker in MACRO_TICKERS.items():
        target=completed_session("HK" if name in {"HSI","VHSI"} else "US")
        df=trim_completed(frames.get(ticker),target)
        if df is None or len(df)<15 or str(df.index[-1].date())!=target:
            continue
        close=safe_float(df["close"].iloc[-1])
        previous=safe_float(df["close"].iloc[-2])
        high,low=safe_float(df["high"].max()),safe_float(df["low"].min())
        result[name]={"val":close,"chg":(close/previous-1)*100 if previous>0 else np.nan,
                      "pct":(close-low)/(high-low)*100 if high>low else 50,
                      "ret_10d":(close/df["close"].iloc[-10]-1)*100,"last_bar":target}
    return result


def classify_market_state(market, macro_data):
    benchmark = "^HSI" if market == "🇭🇰 港股" else "SPY"
    vol_key = "VHSI" if market == "🇭🇰 港股" else "VIX"
    target = completed_session("HK" if market=="🇭🇰 港股" else "US")
    df = trim_completed(fetch_ohlcv(benchmark, period="2y"),target)
    if df is None or len(df) < 205 or str(df.index[-1].date()) != target:
        return {"name": "unknown", "benchmark": benchmark, "return_60d": 0.0, "vol": 0.0, "vol_index": np.nan,
                "vol_known": False, "data_date": None if df is None or df.empty else str(df.index[-1].date())}
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
    vol_known = bool(np.isfinite(vol_index))
    if not vol_known and name.startswith("bull"):
        name = "bull_vol_unknown"
    return {"name": name, "benchmark": benchmark, "return_60d": ret60, "vol": ann_vol, "vol_index": vol_index,
            "vol_known": vol_known, "data_date": str(df.index[-1].date())}


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


# ==================== V4.8 已確認波段與動態 Fibonacci ====================
AUTO_FIB_FEATURE_VERSION = "confirmed-swing-fib-v1"
AUTO_FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)


def af_validate(df):
    if df is None or df.empty:
        raise ValueError("沒有可用日線")
    cols = ["open", "high", "low", "close", "volume"]
    d = df[cols].copy().apply(pd.to_numeric, errors="coerce")
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index))
    if d.index.has_duplicates or not d.index.is_monotonic_increasing:
        raise ValueError("日期必須遞增且不重複")
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError("OHLCV含缺失或無限值")
    if ((d[cols[:4]] <= 0).any().any() or (d.volume < 0).any()
        or (d.high < d[["open", "close", "low"]].max(axis=1)).any()
        or (d.low > d[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("OHLCV價格範圍不合法")
    return d


def af_atr(df, period=14):
    prev = df.close.shift()
    tr = pd.concat([df.high-df.low, (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    values = np.full(len(df), np.nan)
    if len(df) >= period:
        values[period-1] = tr.iloc[:period].mean()
        for i in range(period, len(df)):
            values[i] = (values[i-1]*(period-1)+tr.iloc[i])/period
    return pd.Series(values, index=df.index)


def af_replay(df, left=10, right=5, min_swing_atr=2.0, zone_atr=0.35):
    if int(left) != left or int(right) != right or left < 1 or right < 1:
        raise ValueError("左右窗口必須是正整數")
    if not np.isfinite(min_swing_atr) or not np.isfinite(zone_atr) or min_swing_atr < 0 or zone_atr <= 0:
        raise ValueError("ATR參數不合法")
    left, right = int(left), int(right)
    d = af_validate(df)
    atr = af_atr(d)
    highs, lows, closes = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    swings, events, history = [], [], []
    for i in range(len(d)):
        p = i-right
        if p >= left and np.isfinite(atr.iloc[i]) and atr.iloc[i] > 0:
            lo, hi = lows[p-left:i+1], highs[p-left:i+1]
            is_low = lows[p] < min(lo[:left]) and lows[p] <= min(lo[left+1:])
            is_high = highs[p] > max(hi[:left]) and highs[p] >= max(hi[left+1:])
            if bool(is_low) != bool(is_high):
                kind = "low" if is_low else "high"
                value = float(lows[p] if is_low else highs[p])
                pivot = {"kind":kind, "price":value, "pivot_index":p, "pivot_date":d.index[p],
                         "confirmed_index":i, "confirmed_at":d.index[i]}
                accepted, replace = not swings, False
                if swings:
                    last = swings[-1]
                    if last["kind"] == kind:
                        replace = value < last["price"] if kind == "low" else value > last["price"]
                        accepted = replace
                    else:
                        directional = value > last["price"] if kind == "high" else value < last["price"]
                        accepted = directional and abs(value-last["price"]) >= min_swing_atr*atr.iloc[i]
                if accepted:
                    if replace:
                        swings[-1] = pivot
                    else:
                        swings.append(pivot)
                    events.append(dict(pivot, action="replace" if replace else "append"))
        row = {"asof":d.index[i], "close":float(closes[i]), "atr":float(atr.iloc[i]),
               "direction":"unavailable", "anchor_a":np.nan, "anchor_b":np.nan,
               "anchor_a_date":pd.NaT, "anchor_b_date":pd.NaT, "available_at":pd.NaT,
               "retracement_pct":np.nan, "price_change_from_b_pct":np.nan,
               "nearest_ratio":np.nan, "nearest_price":np.nan, "distance_atr":np.nan,
               "near_zone":False, "invalidated":False, "status":"等待兩個已確認波段點"}
        if len(swings) >= 2:
            a,b = swings[-2:]
            delta = b["price"]-a["price"]
            if delta != 0:
                levels = {r:b["price"]-r*delta for r in AUTO_FIB_RATIOS if b["price"]-r*delta > 0}
                near = min(levels,key=lambda r:abs(closes[i]-levels[r]))
                dist = abs(closes[i]-levels[near])/atr.iloc[i] if atr.iloc[i] > 0 else np.nan
                retr = (b["price"]-closes[i])/delta*100
                row.update(direction="up" if delta > 0 else "down",
                           anchor_a=a["price"],anchor_b=b["price"],
                           anchor_a_date=a["pivot_date"],anchor_b_date=b["pivot_date"],
                           available_at=b["confirmed_at"],retracement_pct=float(retr),
                           price_change_from_b_pct=float((closes[i]/b["price"]-1)*100),
                           nearest_ratio=near,nearest_price=levels[near],distance_atr=float(dist),
                           near_zone=bool(dist <= zone_atr),invalidated=bool(retr >= 100),
                           status="原波段起點已被跨越" if retr >= 100 else
                                  "已突破原波段終點" if retr < 0 else
                                  "上升段後回調" if delta > 0 else "下跌段後反彈")
        history.append(row)
    h = pd.DataFrame(history).set_index("asof")
    for col in ("anchor_a_date","anchor_b_date","available_at"):
        h[col] = pd.to_datetime(h[col])
    return {"data":d, "history":h, "events":pd.DataFrame(events), "swings":swings,
            "left":left,"right":right,"min_swing_atr":min_swing_atr,"zone_atr":zone_atr}


def af_levels(result):
    h = result["history"].iloc[-1]
    if h.direction == "unavailable":
        return pd.DataFrame()
    rows = []
    for r in AUTO_FIB_RATIOS:
        p = h.anchor_b-r*(h.anchor_b-h.anchor_a)
        if p <= 0:
            continue
        width = result["zone_atr"]*h.atr
        rows.append({"ratio":r,"price":p,"lower":p-width,"upper":p+width,
                     "meaning":"跨越起點的反向延伸" if r > 1 else "起點" if r == 1 else
                               "上升段回調" if h.direction == "up" else "下跌段反彈"})
    return pd.DataFrame(rows)


def af_snapshot(df):
    defaults = {"auto_fib_version":AUTO_FIB_FEATURE_VERSION,"auto_fib_status":"資料不足",
                "auto_fib_direction":"unavailable","auto_fib_retracement_pct":None,
                "auto_fib_nearest_ratio":None,"auto_fib_nearest_price":None,
                "auto_fib_distance_atr":None,"auto_fib_near_zone":False,
                "auto_fib_invalidated":False,"auto_fib_confirmed_at":None,
                "auto_fib_anchor_a":None,"auto_fib_anchor_b":None,
                "auto_fib_anchor_a_date":None,"auto_fib_anchor_b_date":None,
                "auto_fib_price_change_from_b_pct":None,"auto_fib_error":None,
                "auto_fib_parameters":"left=10;right=5;min_swing_atr=2;zone_atr=0.35"}
    try:
        h = af_replay(df)["history"].iloc[-1]
        defaults.update(auto_fib_status=h.status,auto_fib_direction=h.direction,
                        auto_fib_near_zone=bool(h.near_zone),auto_fib_invalidated=bool(h.invalidated))
        for key in ("retracement_pct","nearest_ratio","nearest_price","distance_atr",
                    "anchor_a","anchor_b","price_change_from_b_pct"):
            v = h[key]
            defaults["auto_fib_"+key] = float(v) if pd.notna(v) and np.isfinite(v) else None
        for src,dst in (("available_at","confirmed_at"),("anchor_a_date","anchor_a_date"),("anchor_b_date","anchor_b_date")):
            defaults["auto_fib_"+dst] = str(pd.Timestamp(h[src]).date()) if pd.notna(h[src]) else None
    except Exception as exc:
        defaults.update(auto_fib_status="波段計算失敗；不改動原評分",auto_fib_error=f"{type(exc).__name__}: {str(exc)[:160]}")
    return defaults


def af_figure(result, bars=240):
    d = result["data"].iloc[-bars:]
    h = result["history"].iloc[-1]
    fig = go.Figure(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name="完整日線"))
    if h.direction != "unavailable":
        start = max(h.available_at,d.index[0])
        for row in af_levels(result).itertuples():
            color = C_PURPLE if row.ratio in (.5,.618) else C_GREY
            fig.add_trace(go.Scatter(x=[start,d.index[-1]],y=[row.price,row.price],mode="lines",
                                    name=f"{row.ratio:.3f} | {row.price:,.2f}",line=dict(color=color,width=1)))
            if row.ratio in (.5,.618):
                fig.add_shape(type="rect",x0=start,x1=d.index[-1],y0=row.lower,y1=row.upper,
                              fillcolor=C_PURPLE,opacity=.14,line_width=0,layer="below")
    events = result["events"]
    if not events.empty:
        events = events[events.confirmed_at >= d.index[0]]
        for kind,symbol,color in (("low","triangle-up",C_GREEN),("high","triangle-down",C_RED)):
            e = events[events.kind == kind]
            fig.add_trace(go.Scatter(x=e.confirmed_at,y=e.price,mode="markers",name=f"{kind}確認日",
                                    customdata=e.pivot_date.astype(str),
                                    hovertemplate="確認日：%{x}<br>原轉折日：%{customdata}<br>轉折價：%{y}<extra></extra>",
                                    marker=dict(symbol=symbol,color=color,size=9)))
    fig.update_layout(template="plotly_dark",height=650,xaxis_rangeslider_visible=False,
                      title="當前有效比例水平：只從確認日起畫線",legend=dict(orientation="h"))
    return fig


def render_auto_fib_panel():
    st.subheader("📐 動態波段／黃金比率與調整幅度")
    st.caption("沿用主系統Yahoo快取與完整交易日檢查。此模組不改動原評分、買賣分類、止損或目標。")
    with st.form("auto_fib_form"):
        a,b,c = st.columns(3)
        tk = a.text_input("股票／指數代號","^HSI",key="auto_fib_ticker").strip().upper()
        left = int(b.number_input("左側比較交易日",3,30,10,key="auto_fib_left"))
        right = int(c.number_input("右側確認交易日",2,15,5,key="auto_fib_right"))
        a,b = st.columns(2)
        min_atr = a.number_input("最小反向波段 ATR 倍數",0.0,5.0,2.0,.25,key="auto_fib_min_atr")
        zone = b.number_input("候選區域半寬 ATR 倍數",.1,1.0,.35,.05,key="auto_fib_zone")
        requested = st.form_submit_button("載入／更新波段研究")
    context = (tk,left,right,min_atr,zone)
    if requested:
        st.session_state.pop("auto_fib_research",None)
        try:
            if tk not in {"^HSI","^HSCE","^GSPC","^NDX","^DJI"} and not re.fullmatch(r"[A-Z0-9][A-Z0-9.=-]{0,19}",tk):
                raise ValueError("請輸入港股、美股或支援的港美指數代號")
            code = "HK" if tk.endswith(".HK") or tk in {"^HSI","^HSCE"} else "US"
            target = completed_session(code)
            if not target:
                raise ValueError("未能確認完整交易日："+str(calendar_info(code).get("error")))
            with st.spinner("讀取完整日線並重播波段…"):
                fetch_ohlcv.clear()
                frame = trim_completed(fetch_ohlcv(tk,"2y"),target)
                if frame is None or frame.empty or str(frame.index[-1].date()) != target:
                    raise ValueError(f"缺少{target}完整日線；不以舊價格作最新研究")
                result = af_replay(frame,left,right,min_atr,zone)
                st.session_state["auto_fib_research"] = {"context":context,"ticker":tk,"target":target,"result":result}
        except Exception as exc:
            add_error("動態波段研究失敗",tk,exc)
            st.error(f"未能載入：{exc}")
    saved = st.session_state.get("auto_fib_research")
    if not saved or saved["context"] != context:
        st.info("按『載入／更新波段研究』才會下載；初次開頁不會新增行情請求。")
        return
    result = saved["result"]
    h = result["history"].iloc[-1]
    st.caption(f"{saved['ticker']}｜研究資料截至 {saved['target']}｜復權日線｜右側確認延遲 {right} 個交易日；非盤中報價")
    cols = st.columns(4)
    cols[0].metric("收市價",f"{h.close:,.3f}")
    cols[1].metric("ATR(14)",f"{h.atr:,.3f}")
    cols[2].metric("波段回吐／收復比例",f"{h.retracement_pct:.1f}%" if pd.notna(h.retracement_pct) else "未建立")
    cols[3].metric("相對波段終點價格變化",f"{h.price_change_from_b_pct:+.2f}%" if pd.notna(h.price_change_from_b_pct) else "未建立")
    st.info(h.status)
    if h.direction != "unavailable":
        st.write(f"錨點 {h.anchor_a:,.3f} → {h.anchor_b:,.3f}；由 {pd.Timestamp(h.available_at).date()} 收市起可用。")
        st.write(f"最近比例 {h.nearest_ratio:.3f}：{h.nearest_price:,.3f}；距離 {h.distance_atr:.2f} ATR。")
        if h.invalidated:
            st.warning("已跨越原波段起點；勿再沿用原方向的回調／反彈假設。")
        elif h.near_zone:
            st.warning("接近候選價格區域；並非已確認反轉或買入訊號。")
    st.plotly_chart(af_figure(result),use_container_width=True,key="auto_fib_chart")
    st.dataframe(af_levels(result),use_container_width=True,hide_index=True)
    with st.expander("波段確認事件及逐日特徵匯出"):
        st.dataframe(result["events"].tail(40),use_container_width=True,hide_index=True)
        st.download_button("匯出逐日可見特徵CSV",result["history"].to_csv().encode("utf-8-sig"),
                           file_name=f"{tk}_confirmed_swing_fib.csv",mime="text/csv",key="auto_fib_export")
    st.caption("三角形橫座標為確認日，縱座標為較早轉折價格，不是當日高低價。1.272/1.618為越過起點的反向延伸，不是順勢目標。")
    st.caption("掃描固定使用10/5、2 ATR、0.35 ATR；本頁參數只供研究，不改變掃描。未估計成功率，未驗證績效優於原系統。")

# ==================== V5.0 三引擎：估值 / 下行風險 / 入場決策 ====================
ENGINE_VERSION = "three-engine-v5.0"

# 行業合理PE假設（保守／基準／樂觀）。這是可審核的預設假設，不是市場事實；可按需要修改。
SECTOR_PE_RANGES = {
    "Technology": (18, 24, 30), "Communication Services": (14, 19, 24),
    "Consumer Cyclical": (13, 18, 23), "Consumer Defensive": (15, 19, 23),
    "Healthcare": (15, 20, 25), "Industrials": (14, 18, 22),
    "Energy": (8, 11, 14), "Basic Materials": (9, 12, 15),
    "Real Estate": (9, 13, 17), "Utilities": (12, 15, 18),
}
DEFAULT_PE_RANGE = (12, 16, 20)
CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
FINANCIAL_KEYWORDS = ("financial", "bank", "insurance", "hk_bank")
HK_MULTIPLE_DISCOUNT = 0.80          # 港股估值倍數折讓假設
FCF_REQUIRED_YIELD = (0.09, 0.07, 0.055)   # 保守／基準／樂觀 要求自由現金流收益率
COST_OF_EQUITY = (0.12, 0.10, 0.09)        # 金融股要求回報（保守／基準／樂觀）
LONG_TERM_GROWTH = 0.03
MOS_LEVELS = [(0.35, "🟢 重度低估"), (0.20, "🟠 中度低估"), (0.10, "🟡 輕度低估")]
ENTRY_ACTIONS = ["符合入場條件", "等待入場確認", "價值陷阱觀察", "價格未夠吸引", "排除"]


@st.cache_data(ttl=21600, show_spinner=False)
def get_fx_rate(src, dst):
    """財報貨幣→報價貨幣匯率；失敗回傳NaN，不作猜測。"""
    if not src or not dst or src == dst:
        return 1.0
    try:
        raw = yf.download(f"{src}{dst}=X", period="10d", interval="1d", progress=False, threads=False, timeout=10)
        frame = clean_price_frame(raw)
        return safe_float(frame["close"].iloc[-1]) if frame is not None else np.nan
    except Exception:
        return np.nan


@st.cache_data(ttl=INFO_TTL * 6, show_spinner=False)
def get_earnings_history(ticker):
    """讀取年度淨利潤（財報貨幣）。回傳由新至舊的列表。"""
    try:
        stmt = yf.Ticker(ticker).income_stmt
        if stmt is None or stmt.empty:
            return []
        for row in ("Net Income Common Stockholders", "Net Income", "Net Income From Continuing Operation Net Minority Interest"):
            if row in stmt.index:
                values = [safe_float(v) for v in stmt.loc[row].values]
                return [v for v in values if np.isfinite(v)][:5]
    except Exception:
        return []
    return []


def classify_company(ticker, info, sector):
    quote_type = str(info.get("quote_type") or "").upper()
    if ticker in ETF_TICKERS or quote_type in {"ETF", "MUTUALFUND", "INDEX"} or ticker.startswith("^"):
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
    """基本面品質：紅旗、否決項、價值陷阱。"""
    flags, blockers, trap_points = [], [], 0
    roe = safe_float(info.get("roe"))
    debt = safe_float(info.get("de_ratio"))
    rev = safe_float(info.get("rev_growth"))
    earn = safe_float(info.get("earnings_growth"))
    fcf = safe_float(info.get("free_cashflow"))
    ebitda = safe_float(info.get("ebitda"))
    net_debt = safe_float(info.get("total_debt"), 0) - safe_float(info.get("total_cash"), 0)
    bvps = safe_float(info.get("bvps_quote"))
    if np.isfinite(bvps) and bvps <= 0:
        blockers.append("每股淨資產≤0（負淨資產），不以PB判斷便宜")
    if np.isfinite(roe) and roe < 0.05:
        flags.append(f"ROE偏低 {roe*100:.1f}%"); trap_points += 1
    if np.isfinite(rev) and rev < -0.10:
        flags.append(f"營收按年 {rev*100:.1f}%"); trap_points += 1
    if np.isfinite(earn) and earn < -0.30:
        flags.append(f"盈利按年 {earn*100:.1f}%"); trap_points += 1
    if company_type != "金融":
        if np.isfinite(fcf) and fcf < 0:
            flags.append("自由現金流為負"); trap_points += 1
        if np.isfinite(debt) and debt > 200:
            flags.append(f"負債權益比 {debt:.0f}%"); trap_points += 1
        if np.isfinite(ebitda) and ebitda > 0 and net_debt / ebitda > 4:
            flags.append(f"淨負債/EBITDA {net_debt/ebitda:.1f}倍"); trap_points += 1
    if np.isfinite(roe) and roe < 0 and np.isfinite(rev) and rev < -0.15:
        blockers.append("盈利能力與收入同時惡化")
    return {"flags": flags or ["未見主要基本面紅旗"], "blockers": blockers,
            "value_trap": trap_points >= 2, "trap_points": trap_points}


def valuation_engine(ticker, info, sector, price):
    """輸出合理價區間（保守/基準/樂觀）、安全邊際、可信度及假設。"""
    ctype = classify_company(ticker, info, sector)
    hk = ticker.endswith(".HK")
    out = {"company_type": ctype, "methods": [], "assumptions": [], "fair_low": np.nan, "fair_base": np.nan,
           "fair_high": np.nan, "margin_of_safety": np.nan, "confidence": "低", "confidence_score": 0,
           "val_level": "⚪ 無法可靠估值", "normalized_eps": np.nan, "valuation_note": ""}
    if not np.isfinite(price) or price <= 0:
        out["valuation_note"] = "現價無效"
        return out
    if ctype in {"ETF／指數", "虧損"}:
        out["valuation_note"] = f"{ctype}：不套用正盈利企業估值模型"
        return out
    fx_ok = info.get("fx_ok", True)
    disc = HK_MULTIPLE_DISCOUNT if hk else 1.0
    methods = []
    eps_ttm, eps_fwd = safe_float(info.get("eps_quote")), safe_float(info.get("forward_eps_quote"))
    hist = [x for x in info.get("eps_history_quote", []) if np.isfinite(x)]
    candidates = []
    if ctype == "周期":
        if len(hist) >= 3:
            candidates = [float(np.mean(hist))]
            out["assumptions"].append(f"周期股：使用 {len(hist)} 年平均EPS，避免把景氣高峰低PE當便宜")
    else:
        candidates = [x for x in [eps_ttm, eps_fwd, float(np.mean(hist[:3])) if len(hist) >= 3 else np.nan] if np.isfinite(x) and x > 0]
    norm_eps = float(np.median(candidates)) if candidates else np.nan
    out["normalized_eps"] = norm_eps

    if ctype != "金融" and np.isfinite(norm_eps) and norm_eps > 0:
        lo, base, hi = [m * disc for m in SECTOR_PE_RANGES.get(sector, DEFAULT_PE_RANGE)]
        methods.append(("正常化盈利×行業PE", norm_eps * lo, norm_eps * base, norm_eps * hi))
        out["assumptions"].append(f"正常化EPS {norm_eps:.3f}｜PE假設 {lo:.1f}/{base:.1f}/{hi:.1f}" + ("（含港股折讓）" if hk else ""))

    fcf_ps = safe_float(info.get("fcf_ps_quote"))
    if ctype not in {"金融"} and np.isfinite(fcf_ps) and fcf_ps > 0:
        add = 0.01 if hk else 0.0
        y = [r + add for r in FCF_REQUIRED_YIELD]
        methods.append(("自由現金流收益率", fcf_ps / y[0], fcf_ps / y[1], fcf_ps / y[2]))
        out["assumptions"].append(f"每股FCF {fcf_ps:.3f}｜要求收益率 {y[0]*100:.1f}%/{y[1]*100:.1f}%/{y[2]*100:.1f}%")

    bvps, roe = safe_float(info.get("bvps_quote")), safe_float(info.get("roe"))
    if ctype == "金融" and np.isfinite(bvps) and bvps > 0 and np.isfinite(roe):
        roe_c = float(np.clip(roe, 0, 0.25))
        add = 0.01 if hk else 0.0
        pbs = [max(0.0, (roe_c - LONG_TERM_GROWTH) / (r + add - LONG_TERM_GROWTH)) for r in COST_OF_EQUITY]
        methods.append(("ROE合理PB", bvps * pbs[0], bvps * pbs[1], bvps * pbs[2]))
        out["assumptions"].append(f"BVPS {bvps:.3f}｜ROE {roe*100:.1f}%｜合理PB {pbs[0]:.2f}/{pbs[1]:.2f}/{pbs[2]:.2f}（g={LONG_TERM_GROWTH*100:.0f}%）")
    if sector == "Real Estate" and np.isfinite(bvps) and bvps > 0:
        methods.append(("資產折讓（PB）", bvps * 0.5, bvps * 0.7, bvps * 0.9))
        out["assumptions"].append("地產：賬面資產按0.5/0.7/0.9倍PB折讓")

    methods = [m for m in methods if all(np.isfinite(v) and v > 0 for v in m[1:])]
    if not methods:
        out["valuation_note"] = "資料不足以估算合理價；不視作低估"
        return out
    out["methods"] = [m[0] for m in methods]
    lows, bases, highs = [m[1] for m in methods], [m[2] for m in methods], [m[3] for m in methods]
    out["fair_low"], out["fair_base"], out["fair_high"] = float(np.median(lows)), float(np.median(bases)), float(np.median(highs))
    mos = 1 - price / out["fair_base"]
    out["margin_of_safety"] = mos
    dispersion = max(bases) / min(bases) if len(bases) > 1 else np.nan

    score = {1: 25, 2: 40}.get(len(methods), 45)
    score += 20 if len(hist) >= 3 else 8 if hist else 0
    score += 10 if fx_ok else -15
    score += 15 if np.isfinite(dispersion) and dispersion < 1.5 else (5 if not np.isfinite(dispersion) else -5)
    score += 10 if np.isfinite(roe) and np.isfinite(safe_float(info.get("de_ratio"))) else 0
    if np.isfinite(eps_ttm) and np.isfinite(eps_fwd) and eps_ttm > 0 and abs(eps_fwd / eps_ttm - 1) > 0.5:
        score -= 10
        out["assumptions"].append("預測EPS與TTM相差超過50%，盈利波動大")
    if not fx_ok:
        out["assumptions"].append("財報貨幣與報價貨幣未能確認匯率")
    score = int(np.clip(score, 0, 100))
    out["confidence_score"] = score
    out["confidence"] = "高" if score >= 70 else "中" if score >= 45 else "低"

    level = "⚪ 估值合理" if mos > -0.10 else "🔴 估值偏貴"
    for threshold, label in MOS_LEVELS:
        if mos >= threshold:
            level = label
            break
    if out["confidence"] == "低" and mos >= 0.10:
        level = "🟡 相對偏低（低可信）"
    out["val_level"] = level
    out["valuation_note"] = f"方法：{'、'.join(out['methods'])}" + (f"｜方法分歧 {dispersion:.2f}倍" if np.isfinite(dispersion) else "")
    return out


def confirmed_pivot_lows(df, left=10, right=5):
    lows = df["low"].to_numpy()
    out = []
    for p in range(left, len(lows) - right):
        if lows[p] < lows[p - left:p].min() and lows[p] <= lows[p + 1:p + right + 1].min():
            out.append((df.index[p], float(lows[p])))
    return out


def forward_min_drawdown(df, horizon):
    """D(t,h)=max(0,1-min(L[t+1..t+h])/P[t])；最後h日沒有完整未來資料→NaN。"""
    future_min = df["low"].rolling(horizon, min_periods=horizon).min().shift(-horizon)
    return (1 - future_min / df["close"]).clip(lower=0)


def historical_downside(df, horizon):
    close = df["close"]
    dd = forward_min_drawdown(df, horizon)
    ma200 = close.rolling(200, min_periods=120).mean()
    rv20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    trend = (close > ma200).where(ma200.notna())
    q1, q2 = rv20.quantile(1 / 3), rv20.quantile(2 / 3)
    vol_bucket = pd.cut(rv20, [-np.inf, q1, q2, np.inf], labels=["低", "中", "高"])
    frame = pd.DataFrame({"dd": dd, "trend": trend, "vol": vol_bucket}).dropna()
    result = {"horizon": horizon, "n": 0, "effective_n": 0.0, "median": np.nan, "p80": np.nan, "p95": np.nan,
              "condition": "樣本不足", "oos_exceed_rate": np.nan, "reliable": False}
    if frame.empty or pd.isna(trend.iloc[-1]) or pd.isna(vol_bucket.iloc[-1]):
        return result
    now_trend, now_vol = bool(trend.iloc[-1]), vol_bucket.iloc[-1]
    similar = frame[(frame["trend"] == now_trend) & (frame["vol"] == now_vol)]
    label = f"{'MA200以上' if now_trend else 'MA200以下'}＋{now_vol}波動"
    sample = similar if len(similar) / horizon >= 6 else frame
    if sample is frame:
        label = "相似情境樣本不足，改用全部歷史"
    eff = len(sample) / horizon
    result.update({"n": int(len(sample)), "effective_n": round(eff, 1), "condition": label,
                   "median": float(sample["dd"].median()), "p80": float(sample["dd"].quantile(0.80)),
                   "p95": float(sample["dd"].quantile(0.95)), "reliable": eff >= 6})
    split = int(len(frame) * 0.7)
    if split > horizon * 4 and len(frame) - split > horizon * 2:
        p80_train = frame["dd"].iloc[:split].quantile(0.80)
        result["oos_exceed_rate"] = float((frame["dd"].iloc[split:] > p80_train).mean())
    return result


def downside_engine(df, info, valuation, tspr):
    price = safe_float(df["close"].iloc[-1])
    atr = safe_float(calc_atr(df).iloc[-1])
    # 1) 技術結構失效位置（只用已被右側5根確認的低點）
    pivots = [p for p in confirmed_pivot_lows(df.iloc[-500:]) if p[1] < price]
    levels = []
    tspr_sup = tspr.get("active_support", np.nan)
    if np.isfinite(safe_float(tspr_sup)) and not tspr.get("is_broken") and tspr_sup < price:
        levels.append(("TSPR 10日支持", float(tspr_sup)))
    for date, value in sorted(pivots, key=lambda x: -x[1]):
        if all(abs(value / v - 1) > 0.02 for _, v in levels):
            levels.append((f"已確認低點 {pd.Timestamp(date).date()}", value))
        if len(levels) >= 3:
            break
    tech = [{"位置": name, "價格": round(v, 3), "距現價%": round((v / price - 1) * 100, 2),
             "區域": f"{v - 0.35*atr:.3f}–{v + 0.35*atr:.3f}" if np.isfinite(atr) else "N/A"} for name, v in levels[:3]]
    invalidation = levels[0][1] - 0.5 * atr if levels and np.isfinite(atr) else price - 2 * atr if np.isfinite(atr) else np.nan
    # 2) 歷史情境
    h20, h60 = historical_downside(df, 20), historical_downside(df, 60)
    # 3) 基本面壓力
    stress = []
    ctype = valuation["company_type"]
    pe_lo = SECTOR_PE_RANGES.get(info.get("sector"), DEFAULT_PE_RANGE)[0] * (HK_MULTIPLE_DISCOUNT if info.get("ticker", "").endswith(".HK") else 1)
    neps = valuation.get("normalized_eps", np.nan)
    eps_ttm = safe_float(info.get("eps_quote"))
    if ctype in {"一般盈利", "周期"} and np.isfinite(eps_ttm) and eps_ttm > 0:
        cur_pe = price / eps_ttm
        stress.append(("盈利-20%，PE不變", eps_ttm * 0.8 * cur_pe))
        if np.isfinite(neps) and neps > 0:
            stress.append(("盈利不變，PE降至保守倍數", neps * min(pe_lo, cur_pe)))
            stress.append(("盈利-20%＋PE降至保守倍數", neps * 0.8 * min(pe_lo, cur_pe)))
    elif ctype == "金融":
        bvps, roe = safe_float(info.get("bvps_quote")), safe_float(info.get("roe"))
        if np.isfinite(bvps) and bvps > 0:
            cur_pb = price / bvps
            stress.append(("淨資產-10%，PB不變", bvps * 0.9 * cur_pb))
            if np.isfinite(roe):
                pb_s = max(0.0, (max(roe - 0.03, 0) - LONG_TERM_GROWTH) / (COST_OF_EQUITY[0] - LONG_TERM_GROWTH))
                stress.append(("ROE-3個百分點＋保守要求回報", bvps * 0.9 * min(pb_s, cur_pb)))
    stress_rows = [{"情境": n, "情境價": round(v, 3), "下行%": round(max(0.0, 1 - v / price) * 100, 1)} for n, v in stress if np.isfinite(v)]
    worst_stress = max([r["下行%"] for r in stress_rows], default=np.nan)
    p80_60 = h60["p80"] * 100 if np.isfinite(h60["p80"]) else np.nan
    if not np.isfinite(p80_60):
        level = "資料不足"
    else:
        level = "低" if p80_60 < 12 else "中" if p80_60 < 25 else "高"
        if np.isfinite(worst_stress) and worst_stress > 35 and level != "高":
            level = "中" if level == "低" else "高"
    return {"tech_levels": tech, "invalidation": invalidation, "h20": h20, "h60": h60,
            "stress": stress_rows, "worst_stress_pct": worst_stress, "downside_level": level}


def technical_confirmations(df, benchmark_df, tspr):
    close = df["close"]
    price = safe_float(close.iloc[-1])
    ma20 = close.rolling(20).mean()
    macd, sig, _ = calc_macd(close)
    obv, obv_ma = calc_obv(df)
    mfi = safe_float(calc_mfi(df).iloc[-1], 50)
    avwap = calc_avwap_from_recent_low(df)
    rs = calc_relative_strength(df, benchmark_df, 60)
    checks = [
        ("收市站上MA20", price >= safe_float(ma20.iloc[-1], np.inf)),
        ("MA20上彎", safe_float(ma20.iloc[-1]) > safe_float(ma20.iloc[-6], np.inf)),
        ("MACD高於訊號線", safe_float(macd.iloc[-1]) > safe_float(sig.iloc[-1])),
        ("OBV>MA10且MFI≥50", safe_float(obv.iloc[-1]) >= safe_float(obv_ma.iloc[-1], np.inf) and mfi >= 50),
        ("站上近60日低點AVWAP", not avwap.empty and price >= safe_float(avwap.iloc[-1], np.inf)),
        ("60日跑贏基準（同日對齊）", np.isfinite(rs) and rs > 0),
        ("TSPR支持未破", not tspr.get("is_broken", True) and np.isfinite(safe_float(tspr.get("active_support")))),
    ]
    passed = [n for n, ok in checks if ok]
    return passed, [n for n, ok in checks if not ok], rs, mfi


def entry_engine(valuation, quality, downside, passed, regime, price, blockers):
    need = 5 if regime["name"] == "bear_high_vol" else 4 if str(regime["name"]).startswith("bear") else 3
    mos = valuation["margin_of_safety"]
    stop = downside["invalidation"]
    target = valuation["fair_base"]
    rr = (target - price) / (price - stop) if np.isfinite(stop) and np.isfinite(target) and price > stop > 0 else np.nan
    reasons = []
    h20p80 = downside["h20"]["p80"]
    if blockers or quality["blockers"]:
        action = "排除"; reasons += blockers + quality["blockers"]
    elif not np.isfinite(mos):
        action = "排除"; reasons.append(valuation["valuation_note"] or "無法可靠估值")
    elif mos < 0.10:
        action = "價格未夠吸引"; reasons.append(f"安全邊際 {mos*100:.1f}% 未達10%")
    elif quality["value_trap"]:
        action = "價值陷阱觀察"; reasons.append("估值便宜但有多項基本面紅旗：" + "、".join(quality["flags"]))
    else:
        waits = []
        if valuation["confidence"] == "低":
            waits.append("估值可信度低")
        if len(passed) < need:
            waits.append(f"技術確認 {len(passed)}/{need}（{regime['name']}）")
        if not np.isfinite(rr) or rr < 2:
            waits.append(f"回報風險比 {rr:.2f} < 2" if np.isfinite(rr) else "無有效失效位")
        if np.isfinite(h20p80) and h20p80 > 0.25:
            waits.append(f"20日較差情境下行 {h20p80*100:.1f}% 過大")
        if downside["downside_level"] == "資料不足":
            waits.append("歷史下行樣本不足")
        action = "等待入場確認" if waits else "符合入場條件"
        reasons += waits or [f"安全邊際 {mos*100:.1f}%｜確認 {len(passed)}/{need}｜回報風險比 {rr:.2f}"]
    return {"action": action, "action_reasons": reasons, "required_confirmations": need,
            "stop_loss": stop, "target_fair": target, "reward_risk": rr}


def decision_engine(ticker, df, info, regime, benchmark_df):
    """V5.0：三引擎決策。估值與技術分開，總分只作排序，不代表見底機率。"""
    if df is None or len(df) < 250:
        return None
    price = safe_float(df["close"].iloc[-1])
    sector = get_sector_from_info(ticker, info)
    info = dict(info, ticker=ticker, sector=sector)
    blockers = []
    atr = safe_float(calc_atr(df).iloc[-1])
    liquidity_20d = safe_float((df["close"] * df["volume"]).rolling(20).mean().iloc[-1], 0)
    if price <= 0: blockers.append("現價無效")
    if liquidity_20d <= 0: blockers.append("成交額資料不足")
    if not np.isfinite(atr) or atr <= 0: blockers.append("ATR資料不足")
    tspr = calc_tspr_system(df.iloc[-500:], left_bars=10, right_bars=5, box_period=65)
    valuation = valuation_engine(ticker, info, sector, price)
    quality = quality_engine(info, valuation["company_type"])
    downside = downside_engine(df, info, valuation, tspr)
    passed, failed, rs60, mfi = technical_confirmations(df, benchmark_df, tspr)
    entry = entry_engine(valuation, quality, downside, passed, regime, price, blockers)
    mos = valuation["margin_of_safety"]
    conf_w = {"高": 1.0, "中": 0.7, "低": 0.35}[valuation["confidence"]]
    h60p80 = downside["h60"]["p80"]
    research_rank = (mos * 100 * conf_w if np.isfinite(mos) else -99) - (0.5 * h60p80 * 100 if np.isfinite(h60p80) else 10)
    target_2r = price + 2 * (price - entry["stop_loss"]) if np.isfinite(entry["stop_loss"]) else np.nan
    pct = lambda x: round(x * 100, 1) if np.isfinite(safe_float(x)) else np.nan
    r3 = lambda x: round(x, 3) if np.isfinite(safe_float(x)) else np.nan
    return {
        **af_snapshot(df),
        "ticker": ticker, "name": info.get("name", ticker), "sector": sector, "currency": info.get("currency"),
        "price": r3(price), "last_bar": str(pd.Timestamp(df.index[-1]).date()),
        "company_type": valuation["company_type"], "val_level": valuation["val_level"],
        "confidence": valuation["confidence"], "confidence_score": valuation["confidence_score"],
        "fair_low": r3(valuation["fair_low"]), "fair_base": r3(valuation["fair_base"]), "fair_high": r3(valuation["fair_high"]),
        "mos_pct": pct(mos), "methods": "、".join(valuation["methods"]) or "—",
        "assumptions": "｜".join(valuation["assumptions"]) or "—", "valuation_note": valuation["valuation_note"],
        "quality_flags": "｜".join(quality["flags"]), "value_trap": quality["value_trap"],
        "dd20_median": pct(downside["h20"]["median"]), "dd20_p80": pct(downside["h20"]["p80"]), "dd20_p95": pct(downside["h20"]["p95"]),
        "dd60_median": pct(downside["h60"]["median"]), "dd60_p80": pct(downside["h60"]["p80"]), "dd60_p95": pct(downside["h60"]["p95"]),
        "dd_condition": downside["h60"]["condition"], "dd60_effective_n": downside["h60"]["effective_n"],
        "dd60_oos_exceed": pct(downside["h60"]["oos_exceed_rate"]), "dd_reliable": downside["h60"]["reliable"],
        "downside_level": downside["downside_level"], "stress_worst_pct": downside["worst_stress_pct"],
        "tech_levels_json": json.dumps(downside["tech_levels"], ensure_ascii=False),
        "stress_json": json.dumps(downside["stress"], ensure_ascii=False),
        "decision": entry["action"], "action_reasons": "｜".join(entry["action_reasons"]),
        "confirmations": f"{len(passed)}/{entry['required_confirmations']}", "passed_checks": "、".join(passed) or "—",
        "failed_checks": "、".join(failed) or "—",
        "stop_loss": r3(entry["stop_loss"]), "target_fair": r3(entry["target_fair"]), "target_2r": r3(target_2r),
        "reward_risk": round(entry["reward_risk"], 2) if np.isfinite(entry["reward_risk"]) else np.nan,
        "confirmation_price": r3(max(safe_float(df["high"].iloc[-5:].max()), safe_float(df["close"].rolling(20).mean().iloc[-1]))),
        "final_score": round(research_rank, 1),
        "tspr_status": tspr["status"], "tspr_support": tspr["active_support"],
        "rs60": round(rs60, 1) if np.isfinite(rs60) else np.nan, "mfi": round(mfi, 1),
        "liquidity_20d": round(liquidity_20d, 0), "regime": regime["name"], "model_version": MODEL_VERSION,
        "blockers": "｜".join(blockers + quality["blockers"]) or "—",
    }


def save_snapshot_and_signal(result):
    payload = pd.Series(result).to_json(force_ascii=False)
    today = result["last_bar"]
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.execute(
        "INSERT OR REPLACE INTO scan_snapshots (snapshot_date,ticker,model_version,payload_json,created_at) VALUES (?,?,?,?,?)",
        (today, result["ticker"], MODEL_VERSION, payload, hkt_now().isoformat())
    )
    if result["decision"] == "符合入場條件" and result.get("formal_signal_eligible",False):
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
    if not np.isfinite(stop_loss) or stop_loss <= 0 or risk_per_share <= 0 or price <= 0:
        return 0, 0, risk_amount
    by_risk = int(risk_amount / risk_per_share)
    by_notional = int((account_size * max_notional_pct) / price)
    shares = max(0, min(by_risk, by_notional))
    return shares, shares * price, risk_amount


def run_walk_forward_backtest(signal_df, hold_days=20, costs_bps=TRANSACTION_COST_BPS, slippage_bps=DEFAULT_SLIPPAGE_BPS):
    """V5.0 追蹤回測：
    - 訊號日必須有日線，入場為其後首個交易日開市（相隔不超過7個曆日），否則標示資料不足。
    - 入場價已低於止損或高於目標：計劃失效，不計入統計。
    - 跳空穿止損／目標以開市價成交；同日雙觸發保守先止損。
    - 雙邊成本：買入及賣出各扣 costs_bps；持有期＝包括入場日共 hold_days 個交易日。"""
    rows = []
    for _, signal in signal_df.iterrows():
        ticker = signal["ticker"]
        base = {"代碼": ticker, "信號日": str(signal["signal_date"]), "模型": signal.get("model_version", "")}
        df = fetch_ohlcv(ticker, period="5y")
        if df is None or df.empty:
            rows.append({**base, "結果": "無價格資料", "納入統計": False}); continue
        idx = pd.DatetimeIndex(df.index).normalize()
        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        if signal_date not in idx:
            rows.append({**base, "結果": "資料不足：訊號日沒有日線", "納入統計": False}); continue
        future = np.where(idx > signal_date)[0]
        if len(future) == 0:
            rows.append({**base, "結果": "未成熟：尚未有入場日", "納入統計": False}); continue
        entry_i = int(future[0])
        if (idx[entry_i] - signal_date).days > 7:
            rows.append({**base, "結果": "資料不足：入場日缺口過大", "納入統計": False}); continue
        last_i = entry_i + hold_days - 1
        if last_i >= len(df):
            rows.append({**base, "結果": "未成熟", "納入統計": False}); continue
        stop, target = safe_float(signal["stop_loss"]), safe_float(signal["target_2r"])
        raw_entry = safe_float(df["open"].iloc[entry_i])
        if not (np.isfinite(stop) and np.isfinite(target) and stop > 0) or raw_entry <= stop or raw_entry >= target:
            rows.append({**base, "進場日": str(idx[entry_i].date()), "開市價": round(raw_entry, 3), "止損": stop, "目標": target,
                         "結果": "計劃失效：開市已穿止損或目標，取消交易", "納入統計": False}); continue
        entry_fill = raw_entry * (1 + slippage_bps / 10000)
        exit_price, exit_i, outcome = None, last_i, "時間出場"
        for i in range(entry_i, last_i + 1):
            o, low, high = (safe_float(df[c].iloc[i]) for c in ("open", "low", "high"))
            if i > entry_i and o <= stop:
                exit_price, exit_i, outcome = o, i, "跳空穿止損：開市價出場"; break
            if i > entry_i and o >= target:
                exit_price, exit_i, outcome = o, i, "跳空越目標：開市價出場"; break
            if low <= stop:
                exit_price, exit_i, outcome = stop, i, "止損" if high < target else "同日雙觸發：保守先止損"; break
            if high >= target:
                exit_price, exit_i, outcome = target, i, "目標"; break
        if exit_price is None:
            exit_price = safe_float(df["close"].iloc[last_i])
        c = costs_bps / 10000
        net = (exit_price * (1 - c) - entry_fill * (1 + c)) / (entry_fill * (1 + c)) * 100
        rows.append({**base, "進場日": str(idx[entry_i].date()), "進場價": round(entry_fill, 3), "止損": round(stop, 3),
                     "目標": round(target, 3), "出場日": str(idx[exit_i].date()), "出場價": round(exit_price, 3),
                     "持有日": exit_i - entry_i + 1, "結果": outcome, "淨回報%": round(net, 2), "納入統計": True})
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
            value = safe_float(item.get("val"))
            change = safe_float(item.get("chg"))
            col.metric(label, f"{value:.2f}" if np.isfinite(value) else "N/A", f"{change:+.2f}%" if np.isfinite(change) else None)
    st.caption(f"本市場 Regime：{regime['name']}｜基準：{regime['benchmark']}｜60日回報：{regime['return_60d']:.1f}%｜年化波動：{regime['vol']:.1f}%")



# ==================== Yahoo獨立版：交易日／自動股票池 ====================
INDEX_SPECS = {
    "DJIA": {"name":"道瓊工業指數", "market":"US", "url":"https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "min":30, "max":30},
    "NDX": {"name":"納斯達克100", "market":"US", "url":"https://en.wikipedia.org/wiki/Nasdaq-100", "min":95, "max":115},
    "SP500": {"name":"標普500", "market":"US", "url":"https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "min":490, "max":520},
    "HSI": {"name":"恒生指數", "market":"HK", "url":"https://en.wikipedia.org/wiki/Hang_Seng_Index", "min":50, "max":150},
}


def clean_price_frame(frame):
    if frame is None or frame.empty:
        return None
    frame=frame.copy()
    if isinstance(frame.columns,pd.MultiIndex):
        frame.columns=frame.columns.get_level_values(0)
    frame.columns=[str(x).lower() for x in frame.columns]
    required=["open","high","low","close","volume"]
    if not set(required).issubset(frame.columns):
        return None
    frame=frame[required].apply(pd.to_numeric,errors="coerce")
    idx=pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index=idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    frame=frame[~frame.index.duplicated(keep="last")].sort_index()
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=required)
    frame=frame[(frame["close"]>0)&(frame["open"]>0)&(frame["low"]>0)&(frame["volume"]>=0)]
    tol=1.002
    valid=(frame["high"]*tol>=frame[["open","close","low"]].max(axis=1))&(frame["low"]<=frame[["open","close","high"]].min(axis=1)*tol)
    frame=frame[valid]
    return frame if not frame.empty else None


def extract_download(raw, symbols):
    frames={}
    if raw is None or raw.empty:
        return frames
    for ticker in symbols:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    frame=raw[ticker]
                elif ticker in raw.columns.get_level_values(1):
                    frame=raw.xs(ticker,axis=1,level=1)
                else:
                    continue
            elif len(symbols)==1:
                frame=raw
            else:
                continue
            frame=clean_price_frame(frame)
            if frame is not None:
                frames[ticker]=frame
        except (KeyError,ValueError,TypeError):
            continue
    return frames


@st.cache_data(ttl=OHLCV_TTL,show_spinner=False)
def download_analysis_batch(symbols,period):
    raw=yf.download(list(symbols),period=period,interval="1d",group_by="ticker",
                    auto_adjust=True,threads=3,progress=False,timeout=15)
    return extract_download(raw,symbols)


@st.cache_data(ttl=300,show_spinner=False)
def calendar_info(market_code):
    try:
        import exchange_calendars as xcals
    except Exception as exc:
        return {"date":None,"verified":False,"error":f"交易日曆不可用：{type(exc).__name__}: {str(exc)[:160]}。請同步更新requirements.txt並重新部署。"}
    try:
        return calendar_session(market_code,xcals,datetime.now(ZoneInfo("UTC")))
    except Exception as exc:
        return {"date":None,"verified":False,"error":f"交易日判斷失敗：{str(exc)[:160]}；停止正式排名，沒有用普通工作日猜測。"}


def calendar_session(market_code,xcals,now):
    now=pd.Timestamp(now)
    now=now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    calendar=xcals.get_calendar("XHKG" if market_code=="HK" else "XNYS")
    schedule=calendar.schedule.loc[(now-pd.Timedelta(days=35)).strftime("%Y-%m-%d"):now.strftime("%Y-%m-%d")]
    close_col="close" if "close" in schedule.columns else "market_close"
    closes=pd.to_datetime(schedule[close_col],utc=True)
    eligible=schedule.index[closes+pd.Timedelta(minutes=15)<=now]
    if len(eligible)==0:
        raise ValueError("沒有已完成交易日")
    return {"date":str(pd.Timestamp(eligible[-1]).date()),"verified":True,
            "error":None,"calendar":"XHKG" if market_code=="HK" else "XNYS","buffer_minutes":15}


def completed_session(market_code):
    return calendar_info(market_code)["date"]


def trim_completed(frame,target):
    frame=clean_price_frame(frame)
    if frame is None or target is None:
        return None
    return frame.loc[frame.index<=pd.Timestamp(target)].copy()


def init_auto_db():
    with sqlite3.connect(DB_FILE,timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS auto_pool_state (state_key TEXT PRIMARY KEY,payload TEXT NOT NULL,updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS auto_pool_history (index_key TEXT,market_date TEXT,payload TEXT,updated_at TEXT,PRIMARY KEY(index_key,market_date))")
        conn.execute("CREATE TABLE IF NOT EXISTS permanent_watchlist (ticker TEXT PRIMARY KEY,source TEXT NOT NULL,added_at TEXT NOT NULL)")
        conn.executemany("INSERT OR IGNORE INTO permanent_watchlist VALUES (?,?,?)",
                         [(t,"原有永久名單",hkt_now().isoformat()) for t in HK_WATCHLIST+US_WATCHLIST])


def state_get(key):
    with sqlite3.connect(DB_FILE,timeout=30) as conn:
        row=conn.execute("SELECT payload FROM auto_pool_state WHERE state_key=?",(key,)).fetchone()
    return json.loads(row[0]) if row else None


def state_put(key,payload):
    with sqlite3.connect(DB_FILE,timeout=30) as conn:
        conn.execute("INSERT OR REPLACE INTO auto_pool_state VALUES (?,?,?)",
                      (key,json.dumps(payload,ensure_ascii=False,allow_nan=False),hkt_now().isoformat()))


def permanent_tickers():
    with sqlite3.connect(DB_FILE,timeout=30) as conn:
        return [row[0] for row in conn.execute("SELECT ticker FROM permanent_watchlist ORDER BY ticker")]


def normalize_symbol(value,hk=False):
    if pd.isna(value):
        return None
    text=re.sub(r"\[[^\]]*\]","",str(value)).strip()
    if hk:
        text=re.sub(r"^(?:SEHK|HKG|HKEX)\s*:\s*","",text,flags=re.I)
        text=re.sub(r"\.HK$","",text,flags=re.I)
        if re.fullmatch(r"\d{1,5}\.0",text):
            text=text[:-2]
        return str(int(text)).zfill(4)+".HK" if re.fullmatch(r"\d{1,5}",text) and int(text)>0 else None
    text=text.upper().replace(".","-")
    return text if re.fullmatch(r"[A-Z]{1,6}(?:-[A-Z])?",text) else None


def parse_constituents(tables,key):
    spec=INDEX_SPECS[key]
    choices=[]
    for table in tables:
        frame=table.copy()
        frame.columns=[" ".join(map(str,c)) if isinstance(c,tuple) else str(c) for c in frame.columns]
        for column in frame.columns:
            normalized=re.sub(r"\[[^\]]*\]","",column).strip().lower()
            if normalized not in {"symbol","ticker","ticker symbol"}:
                continue
            symbols=sorted({t for v in frame[column] if (t:=normalize_symbol(v,spec["market"]=="HK"))})
            if spec["min"]<=len(symbols)<=spec["max"]:
                choices.append(symbols)
    # 相同表重覆出現可去重；不同合格表不可自行猜測。
    choices=list({tuple(x):x for x in choices}.values())
    if len(choices)!=1:
        raise ValueError(f"{key}成分表驗證失敗：找到{len(choices)}個合格表。沒有以固定熱門股替代。")
    return choices[0]


@st.cache_data(ttl=21600,show_spinner=False)
def fetch_constituents(key):
    url=INDEX_SPECS[key]["url"]
    response=requests.get(url,headers={"User-Agent":"Mozilla/5.0 (compatible; StockMonitorResearch/4.7.2)","Accept":"text/html"},timeout=(8,20))
    response.raise_for_status()
    symbols=parse_constituents(pd.read_html(StringIO(response.text)),key)
    return {"symbols":symbols,"source":url,"fetched_at":hkt_now().isoformat(),"official_effective_date_verified":False}


@st.cache_data(ttl=900,show_spinner=False)
def download_rank_batch(symbols,target):
    target=pd.Timestamp(target)
    raw=yf.download(list(symbols),start=(target-pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                    end=(target+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),interval="1d",
                    auto_adjust=False,group_by="ticker",threads=3,progress=False,timeout=15)
    return extract_download(raw,symbols)


def rank_candidates(symbols,frames,target):
    rows=[]
    date=pd.Timestamp(target)
    for ticker in symbols:
        frame=frames.get(ticker)
        if frame is None or date not in frame.index:
            continue
        row=frame.loc[date]
        price,vol=safe_float(row.get("close")),safe_float(row.get("volume"))
        if np.isfinite(price) and price>0 and np.isfinite(vol) and vol>0:
            rows.append({"ticker":ticker,"estimated_turnover":float(price*vol)})
    rows.sort(key=lambda r:(-r["estimated_turnover"],r["ticker"]))
    return {"market_date":target,"expected":len(symbols),"available":len(rows),
            "complete":bool(symbols) and len(rows)==len(symbols),
            "method":"估算成交額：未復權收市價×成交量（Yahoo）",
            "top":[dict(row,rank=i+1) for i,row in enumerate(rows[:30])]}


def build_auto_pools(keys,status):
    dates,lists,errors,frames={},{},[],{}
    for key in keys:
        try:
            info=calendar_info(INDEX_SPECS[key]["market"])
            if not info["verified"] or not info["date"]:
                raise ValueError(info["error"])
            dates[key]=info["date"]
            status.text(f"讀取{INDEX_SPECS[key]['name']}成分表…")
            lists[key]=fetch_constituents(key)
        except Exception as exc:
            errors.append(f"{key}更新失敗：{type(exc).__name__}: {str(exc)[:200]}")
    for date in sorted(set(dates.values())):
        symbols=sorted({t for k,info in lists.items() if dates.get(k)==date for t in info["symbols"]})
        bucket={}
        for start in range(0,len(symbols),60):
            chunk=tuple(symbols[start:start+60])
            status.text(f"下載{date}排名行情 {min(start+60,len(symbols))}/{len(symbols)}…")
            try:
                bucket.update(download_rank_batch(chunk,date))
            except Exception as exc:
                errors.append(f"{date}排名區塊下載失敗：{type(exc).__name__}: {str(exc)[:140]}")
        frames[date]=bucket
    pools=[]
    for key in keys:
        rank=None
        if key in lists:
            rank=rank_candidates(lists[key]["symbols"],frames.get(dates[key],{}),dates[key])
            rank.update({"index_key":key,"market":INDEX_SPECS[key]["market"],"source":lists[key]["source"],
                         "constituents_fetched_at":lists[key]["fetched_at"],"generated_at":hkt_now().isoformat(),
                         "official_effective_date_verified":False,"fallback":False})
        if rank and rank["complete"]:
            rank["status"]="完整行情覆蓋／估算榜；官方成分生效日未核實"
            state_put("yahoo_ranking:"+key,rank)
            with sqlite3.connect(DB_FILE,timeout=30) as conn:
                conn.execute("INSERT OR REPLACE INTO auto_pool_history VALUES (?,?,?,?)",
                             ("YAHOO:"+key,rank["market_date"],json.dumps(rank,ensure_ascii=False),hkt_now().isoformat()))
        else:
            previous=state_get("yahoo_ranking:"+key)
            if previous:
                rank=previous
                rank.update({"fallback":True,"status":"回退：上一份完整估算榜，並非本次更新",
                             "requested_date":dates.get(key),"current_available":len([t for t in lists.get(key,{}).get("symbols",[]) if t in frames.get(dates.get(key),{})])})
            elif rank:
                rank["status"]="不完整估算榜，只供研究；不是完整Top30"
            else:
                rank={"index_key":key,"market":INDEX_SPECS[key]["market"],"market_date":dates.get(key),
                      "expected":0,"available":0,"complete":False,"top":[],"fallback":False,
                      "method":"無可用排名","status":"更新失敗且沒有已保存榜單"}
        pools.append(rank)
    return pools,errors


def combine_scan_universe(pools,core,market):
    symbols={r["ticker"] for p in pools for r in p["top"]}
    if market=="🇭🇰 港股":
        core=[t for t in core if t.endswith(".HK")]
    elif market=="🇺🇸 美股":
        core=[t for t in core if not t.endswith(".HK")]
    return sorted(symbols|set(core))


def get_scan_metadata(pools):
    metadata={}
    for pool in pools:
        for row in pool["top"]:
            metadata.setdefault(row["ticker"],[]).append({"index":pool["index_key"],"rank":row["rank"],
                "estimated_turnover":row["estimated_turnover"],"ranking_date":pool["market_date"],
                "currency":"HKD" if pool["market"]=="HK" else "USD","complete":pool["complete"],"fallback":pool["fallback"]})
    return metadata


def scan_universe(tickers,pools,progress,custom=False,formal_set=None):
    results,errors,info_map=[],[],{}
    metadata=get_scan_metadata(pools)
    core=set(permanent_tickers())
    market_codes=sorted({"HK" if t.endswith(".HK") or t=="^HSI" else "US" for t in tickers})
    if not any(completed_session(m) for m in market_codes):
        return [],["交易日曆不可用：請更新requirements.txt並重新部署；本次未下載行情或產生正式訊號。"]
    macro=fetch_macro()
    st.session_state["macro_data"]=macro
    contexts={}
    for market_code in market_codes:
        target=completed_session(market_code)
        if not target:
            errors.append(f"{market_code}：交易日曆不可用；暫停本市場正式掃描，技術圖仍可手動載入。")
            continue
        label="🇭🇰 港股" if market_code=="HK" else "🇺🇸 美股"
        local_regime=classify_market_state(label,macro)
        benchmark=trim_completed(fetch_ohlcv(local_regime["benchmark"],ANALYSIS_PERIOD),target)
        if benchmark is None or benchmark.empty or str(benchmark.index[-1].date())!=target:
            errors.append(f"{market_code}：缺少{target}基準日線；暫停本市場。")
            continue
        if local_regime["name"]=="unknown":
            errors.append(f"{market_code}：市場狀態資料不足；暫停正式掃描。")
            continue
        contexts[market_code]=(target,local_regime,benchmark)
    tickers=[t for t in tickers if ("HK" if t.endswith(".HK") or t=="^HSI" else "US") in contexts]
    if not tickers:
        return results,errors
    progress.progress(15,text=f"取得{len(tickers)}隻入選股票基本面…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
        futures={executor.submit(get_full_stock_info,t):t for t in tickers}
        for i,future in enumerate(as_completed(futures),1):
            ticker=futures[future]
            try:
                info_map[ticker]=future.result()
                if info_map[ticker].get("info_error"):
                    errors.append(f"{ticker}基本面：{info_map[ticker]['info_error']}")
            except Exception as exc:
                errors.append(f"{ticker}基本面：{type(exc).__name__}: {str(exc)[:150]}")
            progress.progress(15+int(i/len(tickers)*30),text=f"基本面 {i}/{len(tickers)}；缺失不當作低估")
    prices={}
    for start in range(0,len(tickers),40):
        chunk=tickers[start:start+40]
        progress.progress(45,text=f"下載五年日線 {min(start+40,len(tickers))}/{len(tickers)}…")
        prices.update(fetch_multiple(chunk,ANALYSIS_PERIOD))
    for i,ticker in enumerate(tickers):
        market_code="HK" if ticker.endswith(".HK") or ticker=="^HSI" else "US"
        target,local_regime,benchmark=contexts[market_code]
        try:
            frame=trim_completed(prices.get(ticker),target)
            if frame is None or frame.empty or str(frame.index[-1].date())!=target:
                raise ValueError(f"缺少{target}個股日線，不以舊資料產生新訊號")
            result=decision_engine(ticker,frame,info_map.get(ticker,{}),local_regime,benchmark)
            if result is None:
                raise ValueError("歷史不足250根日線，下行情境樣本不足")
            entries=metadata.get(ticker,[])
            formal=custom or ticker in core or (formal_set is not None and ticker in formal_set) or any(x["complete"] and not x["fallback"] and x["ranking_date"]==target for x in entries)
            result.update({"market":market_code,"data_source":"Yahoo Finance","scanned_at":hkt_now().isoformat(),
                "pool_memberships":entries,"formal_signal_eligible":bool(formal),
                "pool_sources":" / ".join(f"{x['index']} #{x['rank']} ({x['ranking_date']})" for x in entries) or "永久／自選名單"})
            save_snapshot_and_signal(result)
            results.append(result)
        except Exception as exc:
            errors.append(f"{ticker}：{type(exc).__name__}: {str(exc)[:180]}")
        progress.progress(45+int((i+1)/len(tickers)*55),text=f"模型分析 {i+1}/{len(tickers)}")
    return results,errors


def render_pool_report(pools):
    if not pools:
        return
    rows=[{"指數":p["index_key"],"榜單交易日":p["market_date"],"預期成分":p["expected"],
           "有效行情":p["available"],"缺失":p["expected"]-p["available"],"計算方法":p["method"],
           "本次要求日期":p.get("requested_date",p["market_date"]),"狀態":p["status"]} for p in pools]
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption("全部榜單均為估算成交額，不是交易所正式成交額榜。成分表來自Wikipedia，官方生效日未核實；完整只表示當次成分表行情覆蓋。")
    with st.expander("查看各指數Top30、來源及抓取時間"):
        for pool in pools:
            st.write(f"{pool['index_key']}｜{pool['market_date']}｜{pool['status']}")
            st.caption(f"來源：{pool.get('source','不可用')}｜成分抓取時間：{pool.get('constituents_fetched_at','未知')}")
            if pool["top"]:
                frame=pd.DataFrame(pool["top"])
                frame["currency"]="HKD" if pool["market"]=="HK" else "USD"
                st.dataframe(frame,hide_index=True,use_container_width=True)


try:
    init_auto_db()
except Exception as exc:
    st.error(f"自動股票池資料庫初始化失敗：{exc}。請先備份資料庫，再檢查權限與schema。")
    st.stop()



refresh_requested=False
with st.sidebar:
    st.markdown("## ⚙️ Yahoo獨立版控制面板")
    st.success("使用Yahoo Finance；沒有券商連線及OpenD初始化。")
    market=st.radio("市場",["🌐 四指數","🇭🇰 港股","🇺🇸 美股","📋 自選"],index=0)
    custom_input=""
    if market=="📋 自選":
        custom_input=st.text_area("輸入代碼（每行一個）","0700.HK\nNVDA")
    include_core=st.checkbox("合併永久自選／持倉名單",value=True)
    pool_mode=st.radio("股票池",["活躍池：各指數成交額Top30","估值池：全部成分股（較慢）"],index=0,
                       help="估值池不受成交額限制，避免漏掉冷門低估股；S&P500全掃可能需時十多分鐘。")
    force_full=st.checkbox("強制更新成分及基本面（較慢）",value=False)
    refresh_requested=st.button("🔄 刷新資料並自動掃描",type="primary")
    view_mode=st.radio("顯示模式",["新手模式","進階模式"],index=0)
    st.caption("納指採Nasdaq-100；各指數各取30後去重。使用最新完整交易日，收市後等待15分鐘。")
    st.caption("首次開頁不抓行情。按刷新／掃描才下載；本版沒有背景排程或盤中榜。")
    with st.expander("永久名單管理及備份"):
        additions=st.text_area("加入永久名單（每行一個，例如0700.HK或AAPL）",key="core_additions")
        if st.button("加入永久名單"):
            valid=[normalize_symbol(t.strip(),t.strip().upper().endswith(".HK")) for t in additions.splitlines() if t.strip()]
            if not valid or any(t is None for t in valid):
                st.error("沒有有效代碼或包含不合法代碼；本次未寫入。")
            else:
                with sqlite3.connect(DB_FILE,timeout=30) as conn:
                    conn.executemany("INSERT OR IGNORE INTO permanent_watchlist VALUES (?,?,?)",[(t,"手動新增／持倉",hkt_now().isoformat()) for t in valid])
                st.success("已加入；按刷新後納入掃描。")
        st.download_button("匯出永久名單","\n".join(permanent_tickers()),"permanent_watchlist.txt",mime="text/plain")
        st.caption("持倉股請自行加入永久名單；沒有券商持倉同步。本地SQLite需另作備份。")
    with st.expander("交易日曆檢查"):
        if st.button("檢查交易日曆"):
            for calendar_market in ("HK","US"):
                ci=calendar_info(calendar_market)
                if ci["date"]:
                    st.write(f"{calendar_market} 最新完整交易日：{ci['date']}")
                else:
                    st.warning(ci["error"])

context_key=json.dumps([APP_VERSION,market,custom_input,include_core,pool_mode],ensure_ascii=False)
if st.session_state.get("result_context")!=context_key:
    st.session_state["result_context"]=context_key
    st.session_state["latest_results"]=[]
    st.session_state["pool_report"]=[]
    st.session_state["scan_errors"]=[]
    st.session_state["scan_summary"]={}
if refresh_requested:
    clear_errors()
    fetch_ohlcv.clear()
    fetch_macro.clear()
    download_analysis_batch.clear()
    download_rank_batch.clear()
    calendar_info.clear()
    if force_full:
        fetch_constituents.clear()
        get_full_stock_info.clear()

macro_data=st.session_state.get("macro_data",{})
regime={"name":"尚未更新","benchmark":"^HSI" if market=="🇭🇰 港股" else "SPY",
        "return_60d":0.0,"vol":0.0,"vol_index":np.nan}

st.markdown("# 📈 股票監察系統 Pro+｜V5.0 三引擎版（估值／下行風險／入場決策）")
st.caption(f"模型：{MODEL_VERSION}｜程式版本：{APP_VERSION}｜頁面生成：{hkt_now().strftime('%Y-%m-%d %H:%M:%S %Z')} HKT")
st.warning("本系統用於研究與風險管理，不構成投資建議。合理價區間建基於列明的假設；「符合入場條件」不是預測見底或保證收益，歷史下行情境亦不保證未來。")

tabs = st.tabs([
    "🏠 三引擎決策首頁", "💎 估值引擎", "📉 下行風險引擎", "🌍 市場氣氛", "📈 技術圖表 (TSPR)", "📐 回撤/江恩",
    "🧪 歷史訊號回測", "⚖️ 組合風控", "🌟 因子選股", "🛠️ 系統健康", "📐 動態波段研究"
])

ACTION_COLORS = {"符合入場條件": C_GREEN, "等待入場確認": C_ORANGE, "價值陷阱觀察": C_RED, "價格未夠吸引": C_GREY, "排除": C_GREY}


def run_scan_flow():
    clear_errors()
    st.session_state["latest_results"] = []
    st.session_state["scan_summary"] = {}
    pools, errors, results, tickers, formal_set = [], [], [], [], None
    progress = st.progress(0, text="檢查交易日與自動成分表…")
    try:
        if market == "📋 自選":
            tickers = sorted({t.strip().upper() for t in custom_input.splitlines() if t.strip()})
        else:
            keys = ["DJIA", "NDX", "SP500", "HSI"] if market == "🌐 四指數" else (["HSI"] if market == "🇭🇰 港股" else ["DJIA", "NDX", "SP500"])
            pools, errors = build_auto_pools(keys, progress)
            tickers = combine_scan_universe(pools, permanent_tickers() if include_core else [], market)
            if pool_mode.startswith("估值池"):
                formal_set = set()
                for key in keys:
                    try:
                        formal_set |= set(fetch_constituents(key)["symbols"])
                    except Exception as exc:
                        errors.append(f"{key}估值池成分讀取失敗：{type(exc).__name__}")
                tickers = sorted(set(tickers) | formal_set)
        if not tickers:
            raise ValueError("沒有可掃描股票；成分／行情來源失敗時不會虛構股票池。")
        progress.progress(10, text=f"更新宏觀及市場基準…（共 {len(tickers)} 隻）")
        results, scan_errors = scan_universe(tickers, pools, progress, custom=market == "📋 自選", formal_set=formal_set)
        errors.extend(scan_errors)
        st.session_state["latest_results"] = results
        st.session_state["scan_summary"] = {"requested": len(tickers), "successful": len(results), "finished_at": hkt_now().isoformat(), "pool_mode": pool_mode}
    except Exception as exc:
        errors.append(f"本輪掃描中止：{type(exc).__name__}: {str(exc)[:240]}")
    finally:
        progress.empty()
        st.session_state["pool_report"] = pools
        st.session_state["scan_errors"] = errors


def render_stock_card(row):
    color = ACTION_COLORS.get(row["decision"], C_BLUE)
    st.markdown(
        f"<div class='decision-card' style='border-left-color:{color}'>"
        f"<b>{row['ticker']}（{row['name']}）｜{row['decision']}｜{row['val_level']}｜可信度 {row['confidence']}</b><br>"
        f"<b>為何便宜？</b> 現價 {row['price']} vs 基準合理價 {row['fair_base']}，安全邊際 {row['mos_pct']}%<br>"
        f"<b>估值方法：</b>{row['methods']}（{row['company_type']}）<br>"
        f"<b>合理價區間：</b>保守 {row['fair_low']}｜基準 {row['fair_base']}｜樂觀 {row['fair_high']}<br>"
        f"<b>主要假設：</b>{row['assumptions']}<br>"
        f"<b>20日下行情境：</b>中位 {row['dd20_median']}%｜較差(80%) {row['dd20_p80']}%｜尾部(95%) {row['dd20_p95']}%<br>"
        f"<b>60日下行情境：</b>中位 {row['dd60_median']}%｜較差(80%) {row['dd60_p80']}%｜尾部(95%) {row['dd60_p95']}%｜{row['dd_condition']}｜有效樣本 {row['dd60_effective_n']}<br>"
        f"<b>基本面壓力最差：</b>{row['stress_worst_pct']}%｜<b>品質：</b>{row['quality_flags']}<br>"
        f"<b>入場：</b>技術確認 {row['confirmations']}｜失效位 {row['stop_loss']}｜回報風險比 {row['reward_risk']}<br>"
        f"<b>判斷原因：</b>{row['action_reasons']}</div>", unsafe_allow_html=True)


with tabs[0]:
    st.subheader("🏠 三引擎決策｜值得研究？為何低估？會跌幾多？現在應否行動？")
    manual_scan = st.button("🚀 更新股票池並執行三引擎掃描", type="primary", key="decision_scan")
    st.caption(f"股票池：{pool_mode}。估值池掃描全部成分股，時間較長並可能觸發 Yahoo 限流。")
    if refresh_requested or manual_scan:
        run_scan_flow()
    render_pool_report(st.session_state.get("pool_report", []))
    errors = st.session_state.get("scan_errors", [])
    if errors:
        st.warning(f"本輪有 {len(errors)} 項資料／掃描問題。")
        with st.expander("查看本輪問題", expanded=not bool(st.session_state.get("latest_results"))):
            for message in errors:
                st.write(message)
    summary = st.session_state.get("scan_summary", {})
    if summary:
        st.caption(f"本輪請求 {summary.get('requested',0)} 隻｜有效 {summary.get('successful',0)} 隻｜完成 {summary.get('finished_at','未知')}")
    results = st.session_state.get("latest_results", [])
    if not results:
        st.info("按「刷新資料並自動掃描」或上方按鈕後顯示結果。")
    else:
        df_r = pd.DataFrame(results)
        order = {a: i for i, a in enumerate(ENTRY_ACTIONS)}
        df_r["_o"] = df_r["decision"].map(order).fillna(9)
        df_r = df_r.sort_values(["_o", "final_score"], ascending=[True, False]).drop(columns="_o")
        counts = df_r["decision"].value_counts()
        cols = st.columns(5)
        for col, action in zip(cols, ENTRY_ACTIONS):
            col.metric(action, int(counts.get(action, 0)))
        show_actions = st.multiselect("顯示行動類別", ENTRY_ACTIONS, default=ENTRY_ACTIONS[:3])
        view = df_r[df_r["decision"].isin(show_actions)]
        table = view[["ticker", "name", "company_type", "price", "val_level", "confidence", "mos_pct", "fair_low", "fair_base", "fair_high",
                      "dd20_p80", "dd60_p80", "downside_level", "stress_worst_pct", "quality_flags", "confirmations", "reward_risk",
                      "decision", "formal_signal_eligible", "last_bar"]].rename(columns={
            "ticker": "代號", "name": "名稱", "company_type": "類型", "price": "現價", "val_level": "估值狀態", "confidence": "估值可信度",
            "mos_pct": "安全邊際%", "fair_low": "保守合理價", "fair_base": "基準合理價", "fair_high": "樂觀合理價",
            "dd20_p80": "20日較差下行%", "dd60_p80": "60日較差下行%", "downside_level": "下行風險", "stress_worst_pct": "壓力最差下行%",
            "quality_flags": "基本面風險", "confirmations": "技術確認", "reward_risk": "回報風險比", "decision": "行動",
            "formal_signal_eligible": "正式訊號資格", "last_bar": "資料日"})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("研究排序＝安全邊際×可信度權重−0.5×60日較差下行；只用於排序，不是見底機率或回報預測。歷史下行為過去相似情境統計，不保證未來。")
        if not view.empty:
            pick = st.selectbox("查看個股三引擎卡", view["ticker"].tolist(), key="home_pick")
            render_stock_card(view[view["ticker"] == pick].iloc[0])
        if view_mode == "進階模式":
            with st.expander("完整審核表（全部欄位）"):
                st.dataframe(df_r, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("💎 估值引擎｜按公司類型估算合理價區間")
    st.markdown("""
| 類型 | 方法 | 備註 |
|---|---|---|
| 一般盈利 | 正常化EPS（TTM、預測、3年平均的中位數）× 行業PE區間；每股FCF ÷ 要求收益率 | 港股倍數折讓 20%、FCF要求收益率 +1% |
| 金融 | 合理PB＝(ROE−g)/(r−g) × 每股淨資產 | 負淨資產直接排除 |
| 周期 | 只用多年平均EPS | 避免景氣高峰低PE陷阱 |
| 地產 | 盈利法＋資產折讓PB | |
| 虧損／ETF | 不估值 | 不會以中性分當低估 |
""")
    st.caption("行業PE、要求回報率等是程式頂部常數，屬可審核假設，請按你的判斷修改。可信度受方法數量、盈利歷史、貨幣一致性、方法分歧影響。")
    results = st.session_state.get("latest_results", [])
    if results:
        vdf = pd.DataFrame(results)
        only = st.checkbox("只顯示安全邊際 ≥ 10%", value=True)
        if only:
            vdf = vdf[pd.to_numeric(vdf["mos_pct"], errors="coerce") >= 10]
        st.dataframe(vdf[["ticker", "name", "company_type", "price", "val_level", "mos_pct", "confidence", "confidence_score",
                          "fair_low", "fair_base", "fair_high", "methods", "assumptions", "quality_flags", "value_trap", "valuation_note"]]
                     .sort_values("mos_pct", ascending=False), use_container_width=True, hide_index=True)
        plot_df = vdf.dropna(subset=["mos_pct", "dd60_p80"])
        if not plot_df.empty:
            fig = px.scatter(plot_df, x="dd60_p80", y="mos_pct", color="confidence", text="ticker",
                             labels={"dd60_p80": "60日較差情境下行%", "mos_pct": "安全邊際%"},
                             color_discrete_map={"高": C_GREEN, "中": C_ORANGE, "低": C_GREY})
            fig.update_traces(textposition="top center")
            fig.update_layout(template="plotly_dark", plot_bgcolor=C_BG, paper_bgcolor=C_BG, height=520, title="安全邊際 vs 下行風險（左上最理想）")
            st.plotly_chart(fig, use_container_width=True, key="val_scatter")
    else:
        st.info("請先在首頁完成掃描。")

with tabs[2]:
    st.subheader("📉 下行風險引擎｜技術失效位、歷史情境、基本面壓力")
    st.latex(r"D_{t,h}=\max\left(0,\,1-\frac{\min(L_{t+1},\ldots,L_{t+h})}{P_t}\right)")
    st.caption("歷史情境：取與現在相同『MA200上/下＋波動三分位』的過去日子，計算其後20／60個交易日內最多曾跌多深。重疊窗口不是獨立樣本，故以 樣本數÷h 作有效樣本；少於6不顯示為可靠。樣本外檢驗：用前70%算80分位，看後30%超越比例（理想約20%）。")
    one = st.text_input("單股三引擎分析（例如 0700.HK、AAPL）", "0700.HK", key="downside_ticker").upper().strip()
    if st.button("分析", key="downside_run"):
        with st.spinner("下載五年日線及基本面…"):
            mk = "HK" if one.endswith(".HK") else "US"
            target = completed_session(mk)
            frame = trim_completed(fetch_ohlcv(one, period=ANALYSIS_PERIOD), target) if target else None
            reg = classify_market_state("🇭🇰 港股" if mk == "HK" else "🇺🇸 美股", st.session_state.get("macro_data", {}))
            bench = trim_completed(fetch_ohlcv(reg["benchmark"], period=ANALYSIS_PERIOD), target) if target else None
            res = decision_engine(one, frame, get_full_stock_info(one), reg, bench) if frame is not None else None
            st.session_state["downside_single"] = (one, res, frame)
    saved = st.session_state.get("downside_single")
    if saved and saved[0] == one:
        _, res, frame = saved
        if res is None:
            st.error("資料不足（需要至少250根已完成日線），或交易日曆不可用。")
        else:
            if str(res["last_bar"]) != str(completed_session("HK" if one.endswith(".HK") else "US")):
                st.warning("個股最新日線並非最新完整交易日。")
            render_stock_card(res)
            c1, c2 = st.columns(2)
            c1.markdown("**技術結構失效位置**")
            c1.dataframe(pd.DataFrame(json.loads(res["tech_levels_json"])), hide_index=True, use_container_width=True)
            c2.markdown("**基本面壓力情境**")
            c2.dataframe(pd.DataFrame(json.loads(res["stress_json"])), hide_index=True, use_container_width=True)
            st.caption(f"60日樣本外：前70%的80分位，在後30%被超越比例 {res['dd60_oos_exceed']}%（明顯高於20%代表近期風險上升，模型偏樂觀）。")
            dd20, dd60 = forward_min_drawdown(frame, 20).dropna() * 100, forward_min_drawdown(frame, 60).dropna() * 100
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dd20, name="20日內最大下行%", opacity=0.6, marker_color=C_BLUE))
            fig.add_trace(go.Histogram(x=dd60, name="60日內最大下行%", opacity=0.6, marker_color=C_RED))
            fig.update_layout(barmode="overlay", template="plotly_dark", plot_bgcolor=C_BG, paper_bgcolor=C_BG, height=420,
                              title="全部歷史：其後期間內最大下行分佈")
            st.plotly_chart(fig, use_container_width=True, key="dd_hist")
    results = st.session_state.get("latest_results", [])
    if results:
        st.markdown("### 本輪掃描下行風險總表")
        ddf = pd.DataFrame(results)
        st.dataframe(ddf[["ticker", "name", "price", "downside_level", "dd20_median", "dd20_p80", "dd20_p95", "dd60_median", "dd60_p80",
                          "dd60_p95", "dd_condition", "dd60_effective_n", "dd60_oos_exceed", "stress_worst_pct", "stop_loss", "tspr_status"]]
                     .sort_values("dd60_p80"), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("🌍 宏觀市場氣氛")
    if st.button("更新市場氣氛",key="update_macro"):
        with st.spinner("下載宏觀行情…"):
            macro_data=fetch_macro()
            st.session_state["macro_data"]=macro_data
            regime=classify_market_state("🇭🇰 港股" if market=="🇭🇰 港股" else "🇺🇸 美股",macro_data)
            st.session_state["dashboard_regime"]={"market":market,"regime":regime}
    saved_regime=st.session_state.get("dashboard_regime",{})
    if saved_regime.get("market")==market:
        regime=saved_regime["regime"]
    if macro_data:
        render_metrics(macro_data,regime)
        st.caption("四指數／自選模式的上方Regime面板使用美股基準；決策掃描仍按每隻股票的市場分別處理。宏觀面板及正式掃描均嘗試截取至各市場完整交易日；資料未齊時顯示N/A或暫停本市場。")
    else:
        st.info("按『更新市場氣氛』載入宏觀報價；初次開頁不會自動下載。")
    st.markdown("""
**如何使用 Regime 與 TSPR V4.6：** 系統現已整合 **TSPR 10日結構支持位**、**江恩 50% 分水嶺** 與 **OBV / MFI 資金流破位確認**，並自動計算每次跌穿支持後的 **最大下跌幅度 (%)** 與 **頂部至低點總跌幅 (%)**。
""")

with tabs[4]:
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

    chart_context=(ticker,period_map[period])
    if st.button("載入／更新技術圖",key="load_chart"):
        with st.spinner("下載技術圖日線…"):
            st.session_state["chart_data"]=fetch_ohlcv(ticker,period=period_map[period])
            st.session_state["chart_context"]=chart_context
    chart_df=st.session_state.get("chart_data") if st.session_state.get("chart_context")==chart_context else None
    if chart_df is None:
        st.info("按『載入／更新技術圖』才會下載。圖表最新日線可能尚未收市；正式決策掃描會另行排除。")
    st.caption("TSPR低點需右側5根日線確認；圖上的低點日期不是當日已知的訊號日期。破位後跌幅為事後統計，不是預測。")
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

with tabs[5]:
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

with tabs[6]:
    st.subheader("🧪 歷史訊號追蹤回測")
    st.caption("此頁不是滾動訓練／調參的樣本外Walk-forward驗證；交易成本與跳空成交等假設仍需進一步驗證。")
    st.markdown("信號在收市後產生；以**下一交易日開市價加滑點**進場。開市已穿止損或目標＝計劃失效不計；跳空以開市價成交；同日雙觸發先止損；成本雙邊扣除。")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30)
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC", conn)
        conn.close()
    except Exception as exc:
        add_error("讀取回測信號失敗", exc=exc)
        signals = pd.DataFrame()
    if signals.empty:
        st.info("尚無正式訊號。V5.0 只把『符合入場條件』且有正式資格的結果記錄為訊號。")
    else:
        versions = sorted(signals["model_version"].dropna().unique().tolist())
        chosen = st.selectbox("模型版本（新舊模型分開統計）", versions, index=versions.index(MODEL_VERSION) if MODEL_VERSION in versions else 0)
        signals = signals[signals["model_version"] == chosen]
        st.dataframe(signals[["signal_date", "ticker", "decision", "total_score", "price", "confirmation_price", "stop_loss", "target_2r", "market_regime", "model_version"]], use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hold_days = st.selectbox("最長持有交易日（包括入場日）", [10, 20, 30, 45], index=1)
        with c2:
            cost_bps = st.number_input("每邊交易成本（bps，買入及賣出各扣）", min_value=0, max_value=100, value=TRANSACTION_COST_BPS)
        with c3:
            slip_bps = st.number_input("入場滑點（bps）", min_value=0, max_value=100, value=DEFAULT_SLIPPAGE_BPS)
        if st.button("運行歷史訊號回測", type="primary", key="strict_bt"):
            bt = run_walk_forward_backtest(signals, hold_days, cost_bps, slip_bps)
            if bt.empty:
                st.info("沒有訊號可回測。")
            else:
                st.dataframe(bt, use_container_width=True, hide_index=True)
                excluded = int((~bt["納入統計"]).sum())
                bt = bt[bt["納入統計"]]
                st.caption(f"{excluded} 個訊號因未成熟、資料不足或計劃失效而不納入統計。")
            if not bt.empty:
                cols = st.columns(4)
                cols[0].metric("成熟樣本", len(bt))
                cols[1].metric("勝率", f"{(bt['淨回報%'] > 0).mean() * 100:.1f}%")
                cols[2].metric("平均淨回報", f"{bt['淨回報%'].mean():.2f}%")
                cols[3].metric("中位數淨回報", f"{bt['淨回報%'].median():.2f}%")

with tabs[7]:
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

with tabs[8]:
    st.subheader("🌟 多因子選股｜價值、質量、動能、低波動")
    st.caption("此模組適合 3–6 個月以上研究；與短線反轉系統不同。金融、虧損企業和 ETF 不適合只用 PE 與 ROE 橫向比較。")
    if st.button("執行多因子運算", type="primary", key="multi_factor"):
        tickers=sorted({r["ticker"] for r in st.session_state.get("latest_results",[])})
        if not tickers:
            st.info("請先完成決策掃描；因子選股使用同一個已掃描股票池。")
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
            pe_value=safe_float(info.get("pe"))
            if not np.isfinite(pe_value) or pe_value<=0:
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

with tabs[9]:
    st.subheader("🛠️ 系統健康、數據覆蓋與錯誤日誌")
    st.markdown("### 存儲狀態")
    st.warning("目前使用本地 SQLite。若部署在 Streamlit Community Cloud 或無持久磁盤環境，重啟或部署後數據庫可能被清空。正式長期運行請遷移至 Supabase PostgreSQL。")
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30)
        signal_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM signals", conn).iloc[0]["n"]
        snapshot_count = pd.read_sql_query("SELECT COUNT(*) AS n FROM scan_snapshots", conn).iloc[0]["n"]
        conn.close()
        cols = st.columns(2)
        cols[0].metric("已記錄正式訊號（全部模型）", int(signal_count))
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


with tabs[10]:
    render_auto_fib_panel()
