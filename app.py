from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
import os
import re
import sqlite3
import threading
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(
    page_title="📈 股票監察系統 V5.3｜週期反彈回調＋大盤指數三引擎版",
    page_icon="📈",
    layout="wide",
)

# ==================== 視覺配色與常數 ====================
C_RED = "#ef5350"
C_GREEN = "#26a69a"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V5.3-CycleRebound-IndexEdition"
MODEL_VERSION = "bottom-three-engine-v5.3"
ANALYSIS_PERIOD = "5y"
OHLCV_TTL = 1800
INFO_TTL = 7200
DB_FILE = os.getenv("STOCK_DB_FILE", "signals.db")
MAX_WORKERS_INFO = 5

# 🌟 新增：大盤指數與核心衍生/指數 ETF 池 (支援恒指、標普500、納指、道指及代表性ETF)
INDEX_ETF_WATCHLIST = [
    "^HSI",
    "^GSPC",
    "^NDX",
    "^DJI",
    "QQQ",
    "SPY",
    "IWM",
    "SOXL",
    "2800.HK",
    "3033.HK",
    "2828.HK",
]

INDEX_NAME_MAP = {
    "^HSI": "恒生指數 (HSI)",
    "^GSPC": "標普 500 指數 (S&P 500)",
    "^NDX": "納斯達克 100 指數 (NDX)",
    "^DJI": "道瓊斯工業指數 (DJI)",
    "QQQ": "Invesco 納指100 ETF",
    "SPY": "SPDR 標普500 ETF",
    "IWM": "iShares 羅素2000 ETF",
    "SOXL": "Direxion 3倍做多半導體",
    "2800.HK": "盈富基金 (追蹤恒指)",
    "3033.HK": "南方恒生科技 ETF",
    "2828.HK": "恒生中國企業 ETF",
}

# 🌟 美股三大指數最具代表性前 40 大成分股 + 核心觀察股
CORE_US_WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "WMT",
    "V",
    "UNH",
    "XOM",
    "MA",
    "PG",
    "JNJ",
    "COST",
    "HD",
    "ORCL",
    "BAC",
    "ABBV",
    "CVX",
    "CRM",
    "NFLX",
    "AMD",
    "PEP",
    "KO",
    "TMUS",
    "DIS",
    "CSCO",
    "MCD",
    "IBM",
    "CAT",
    "BA",
    "GS",
    "HON",
    "AMGN",
    "AXP",
    "ADBE",
    "PYPL",
    "MSTR",
    "PFE",
    "LITE",
    "UBER",
]

# 🌟 港股最具代表性前 30 大成分股
CORE_HK_WATCHLIST = [
    "0700.HK",
    "9988.HK",
    "3690.HK",
    "1810.HK",
    "9618.HK",
    "9999.HK",
    "1024.HK",
    "0981.HK",
    "0005.HK",
    "0939.HK",
    "1398.HK",
    "3988.HK",
    "0388.HK",
    "1299.HK",
    "2318.HK",
    "2388.HK",
    "0883.HK",
    "0857.HK",
    "0386.HK",
    "0941.HK",
    "0762.HK",
    "1211.HK",
    "2020.HK",
    "1093.HK",
    "1177.HK",
    "0001.HK",
    "0016.HK",
    "0002.HK",
    "0003.HK",
    "0066.HK",
]

ETF_TICKERS = set(INDEX_ETF_WATCHLIST)
AUTO_FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)

SECTOR_MAP_FALLBACK = {
    "0005.HK": "Financial Services",
    "0939.HK": "Financial Services",
    "1398.HK": "Financial Services",
    "3988.HK": "Financial Services",
    "2318.HK": "Financial Services",
    "1299.HK": "Financial Services",
    "0388.HK": "Financial Services",
    "2388.HK": "Financial Services",
    "0002.HK": "Utilities",
    "0003.HK": "Utilities",
    "0066.HK": "Utilities",
    "0016.HK": "Real Estate",
    "0001.HK": "Real Estate",
    "0700.HK": "Communication Services",
    "9999.HK": "Communication Services",
    "1024.HK": "Communication Services",
    "9988.HK": "Consumer Cyclical",
    "3690.HK": "Consumer Cyclical",
    "9618.HK": "Consumer Cyclical",
    "1211.HK": "Consumer Cyclical",
    "2020.HK": "Consumer Cyclical",
    "1810.HK": "Technology",
    "0981.HK": "Technology",
    "0883.HK": "Energy",
    "0857.HK": "Energy",
    "0386.HK": "Energy",
    "0941.HK": "Communication Services",
    "0762.HK": "Communication Services",
    "1093.HK": "Healthcare",
    "1177.HK": "Healthcare",
}

SECTOR_PE_RANGES = {
    "Technology": (18, 24, 30),
    "Communication Services": (14, 19, 24),
    "Consumer Cyclical": (13, 18, 23),
    "Consumer Defensive": (15, 19, 23),
    "Healthcare": (15, 20, 25),
    "Industrials": (14, 18, 22),
    "Energy": (8, 11, 14),
    "Basic Materials": (9, 12, 15),
    "Real Estate": (8, 12, 16),
    "Utilities": (11, 15, 18),
}
DEFAULT_PE_RANGE = (12, 16, 20)
HK_MULTIPLE_DISCOUNT = 0.80

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {background:#0d1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {color:#e6edf3!important;}
.decision-card {background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:16px;margin:10px 0;}
.cycle-card {background:#161b22;border:1px solid #26a69a;border-radius:10px;padding:14px;margin-bottom:12px;}
</style>
""",
    unsafe_allow_html=True,
)


def hkt_now():
  return datetime.now(ZoneInfo("Asia/Hong_Kong"))


def safe_float(value, default=np.nan):
  try:
    return default if value is None or pd.isna(value) else float(value)
  except Exception:
    return default


def normalize_symbol(value):
  if not value or pd.isna(value):
    return None
  text = str(value).strip().upper()
  if text.startswith("^"):
    return text
  if text.endswith(".HK"):
    num = re.sub(r"\.HK$", "", text)
    if num.isdigit():
      return num.zfill(4) + ".HK"
  if text.isdigit() and len(text) <= 5:
    return text.zfill(4) + ".HK"
  text = text.replace(".", "-")
  return text if re.fullmatch(r"[\^A-Z0-9][A-Z0-9.=-]{0,15}", text) else None


# ==================== 數據清理與批量下載 ====================
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
  frame.index = (
      idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
  )
  frame = frame[~frame.index.duplicated(keep="last")].sort_index()
  frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
      subset=["open", "high", "low", "close"]
  )
  frame["volume"] = frame["volume"].fillna(0)
  frame = frame[(frame["close"] > 0) & (frame["open"] > 0) & (frame["low"] > 0)]
  return frame if not frame.empty else None


@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def fetch_ohlcv(ticker, period="5y", interval="1d"):
  try:
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    return clean_price_frame(raw)
  except Exception:
    return None


@st.cache_data(ttl=OHLCV_TTL, show_spinner=False)
def download_analysis_batch(symbols, period="5y"):
  symbols = tuple(sorted(set(symbols)))
  if not symbols:
    return {}
  raw = yf.download(
      list(symbols),
      period=period,
      interval="1d",
      group_by="ticker",
      auto_adjust=True,
      threads=True,
      progress=False,
      timeout=20,
  )
  frames = {}
  if raw is None or raw.empty:
    return frames
  for ticker in symbols:
    try:
      if isinstance(raw.columns, pd.MultiIndex):
        sub = (
            raw[ticker]
            if ticker in raw.columns.get_level_values(0)
            else raw.xs(ticker, axis=1, level=1)
        )
      else:
        sub = raw if len(symbols) == 1 else None
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
    chunk = tickers[start : start + 35]
    try:
      results.update(download_analysis_batch(chunk, period))
    except Exception:
      pass
    for t in chunk:
      if results.get(t) is None:
        single = fetch_ohlcv(t, period=period)
        if single is not None:
          results[t] = single
  return results


@st.cache_data(ttl=INFO_TTL, show_spinner=False)
def get_full_stock_info(ticker):
  is_idx_or_etf = ticker in ETF_TICKERS or ticker.startswith("^")
  result = {
      "ticker": ticker,
      "name": INDEX_NAME_MAP.get(ticker, ticker),
      "pe_ttm": np.nan,
      "pb": None,
      "roe": None,
      "rev_growth": None,
      "sector": (
          "Index / ETF"
          if is_idx_or_etf
          else SECTOR_MAP_FALLBACK.get(
              ticker, "Technology" if not ticker.endswith(".HK") else "OTHER"
          )
      ),
      "quote_type": "INDEX/ETF" if is_idx_or_etf else "EQUITY",
      "eps_quote": np.nan,
  }
  if ticker.startswith("^"):
    return result
  try:
    info = yf.Ticker(ticker).info or {}
    if not info:
      return result
    result["name"] = INDEX_NAME_MAP.get(ticker) or (
        info.get("shortName") or info.get("longName") or ticker
    )
    result["sector"] = info.get("sector") or result["sector"]
    trailing = safe_float(info.get("trailingPE"), None)
    forward = safe_float(info.get("forwardPE"), None)
    px_now = safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    result["pe_ttm"] = (
        trailing
        if (trailing and trailing > 0)
        else (forward if (forward and forward > 0) else np.nan)
    )
    result["pb"] = safe_float(info.get("priceToBook"), None)
    result["roe"] = safe_float(info.get("returnOnEquity"), None)
    result["rev_growth"] = safe_float(info.get("revenueGrowth"), None)
    eps_raw = safe_float(info.get("trailingEps"))
    result["eps_quote"] = (
        px_now / trailing
        if (np.isfinite(px_now) and trailing and trailing > 0)
        else eps_raw
    )
  except Exception:
    pass
  return result


# ==================== 技術指標集 ====================
def calc_rsi(series, period=14):
  delta = series.diff()
  gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
  loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
  rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
  return rsi.mask((loss == 0) & (gain > 0), 100.0).mask(
      (loss == 0) & (gain == 0), 50.0
  )


def calc_williams_r(df, period=14):
  hh = df["high"].rolling(period).max()
  ll = df["low"].rolling(period).min()
  return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def calc_kdj(df, n=9, m1=3, m2=3):
  low_n = df["low"].rolling(n).min()
  high_n = df["high"].rolling(n).max()
  rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100).fillna(
      50.0
  )
  k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
  d = k.ewm(alpha=1 / m2, adjust=False).mean()
  return k, d, 3 * k - 2 * d


def calc_cci(df, period=20):
  tp = (df["high"] + df["low"] + df["close"]) / 3.0
  sma = tp.rolling(period).mean()
  mad = (
      tp.rolling(period)
      .apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
      .replace(0, np.nan)
  )
  return (tp - sma) / (0.015 * mad)


def calc_mfi(df, period=14):
  typical = (df["high"] + df["low"] + df["close"]) / 3
  money = typical * df["volume"]
  pos = money.where(typical > typical.shift(1), 0).rolling(period).sum()
  neg = money.where(typical < typical.shift(1), 0).rolling(period).sum()
  mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
  return mfi.mask((neg == 0) & (pos > 0), 100.0).mask(
      (neg == 0) & (pos == 0), 50.0
  )


def calc_obv(df):
  direction = np.sign(df["close"].diff()).fillna(0)
  obv = (direction * df["volume"]).cumsum()
  return obv, obv.rolling(10).mean()


def calc_atr(df, period=14):
  tr = pd.concat(
      [
          df["high"] - df["low"],
          (df["high"] - df["close"].shift()).abs(),
          (df["low"] - df["close"].shift()).abs(),
      ],
      axis=1,
  ).max(axis=1)
  return tr.ewm(alpha=1 / period, adjust=False).mean()


# ==================== 🌟 新增：N週期反彈與回調儀表板引擎 (CL 指標升級版) ====================
def calc_rebound_correction_dashboard(
    df, window=120, bull_thresh=50.0, bear_thresh=-25.0
):
  """計算 N 週期高低位反彈% (Rebound) 與 回調% (Correction)

  支援日線 (預設120日/250日) 與 週線 (預設52週)，並偵測「紅線深跌勾頭 + 綠線破12%」見底轉勢訊號
  """
  if df is None or len(df) < 10:
    return df, {
        "rebound_pct": 0.0,
        "correction_pct": 0.0,
        "max_rebound": 0.0,
        "max_correction": 0.0,
        "cycle_status": "⚪ 資料不足",
        "turn_bull_signal": False,
        "bull_dyn": bull_thresh,
        "bear_dyn": bear_thresh,
    }

  out = df.copy()
  roll_low = out["low"].rolling(window=window, min_periods=5).min()
  roll_high = out["high"].rolling(window=window, min_periods=5).max()

  out["rebound_pct"] = (
      (out["close"] - roll_low) / roll_low.replace(0, np.nan)
  ) * 100.0
  out["correction_pct"] = (
      (out["close"] - roll_high) / roll_high.replace(0, np.nan)
  ) * 100.0

  # 計算歷史動態分位數 (適應指數與高波動個股差異：85分位為過熱，15分位為超賣)
  valid_reb = out["rebound_pct"].dropna()
  valid_corr = out["correction_pct"].dropna()
  bull_dyn = (
      float(np.nanpercentile(valid_reb, 85))
      if len(valid_reb) > 20
      else bull_thresh
  )
  bear_dyn = (
      float(np.nanpercentile(valid_corr, 15))
      if len(valid_corr) > 20
      else bear_thresh
  )

  # 見底轉勢訊號：過去3期內回調曾跌破超賣線(或動態超賣線)，當前期回調勾頭向上，且反彈突破 10%~12%
  eff_bear = max(bear_thresh, bear_dyn)  # 對指數更靈敏
  recent_oversold = out["correction_pct"].rolling(4).min() <= eff_bear
  corr_hook_up = out["correction_pct"] > out["correction_pct"].shift(1)
  reb_breakout = (out["rebound_pct"] >= 10.0) & (
      out["rebound_pct"].shift(1) < 15.0
  )
  out["turn_bull_signal"] = recent_oversold & corr_hook_up & reb_breakout

  # 見頂回落訊號：反彈曾突破過熱線，且回調開始跌破 -6%
  recent_overbought = out["rebound_pct"].rolling(4).max() >= min(
      bull_thresh, bull_dyn
  )
  corr_drop = (out["correction_pct"] <= -6.0) & (
      out["correction_pct"].shift(1) > -6.0
  )
  out["turn_bear_signal"] = recent_overbought & corr_drop

  curr_reb = float(out["rebound_pct"].iloc[-1])
  curr_corr = float(out["correction_pct"].iloc[-1])
  prev_corr = (
      float(out["correction_pct"].iloc[-2]) if len(out) > 1 else curr_corr
  )
  prev_reb = float(out["rebound_pct"].iloc[-2]) if len(out) > 1 else curr_reb
  is_turn_now = bool(out["turn_bull_signal"].iloc[-3:].any())

  if is_turn_now:
    status = "🟢 見底轉勢 (紅線勾頭+綠線抬頭)"
  elif curr_corr <= eff_bear and curr_corr > prev_corr:
    status = "🟡 超賣勾頭 (左側觀察區)"
  elif curr_corr <= eff_bear:
    status = "🔴 極度超賣 (尋底勿接刀)"
  elif curr_reb >= bull_thresh and curr_corr <= -6.0:
    status = "⚠️ 牛末回調 (紅線開口食糊)"
  elif curr_reb >= bull_thresh:
    status = "🔥 主升過熱 (紅線貼0續抱)"
  elif curr_corr >= -3.5 and curr_reb >= 20.0:
    status = "🚀 強勢破頂推進"
  else:
    status = "⚪ 週期中段震盪"

  summary = {
      "rebound_pct": round(curr_reb, 2),
      "correction_pct": round(curr_corr, 2),
      "max_rebound": round(float(out["rebound_pct"].tail(250).max()), 2),
      "max_correction": round(float(out["correction_pct"].tail(250).min()), 2),
      "bull_dyn": round(bull_dyn, 1),
      "bear_dyn": round(bear_dyn, 1),
      "cycle_status": status,
      "turn_bull_signal": is_turn_now,
  }
  return out, summary


# ==================== TSPR 結構與信號運算 ====================
def calc_tspr_system(df, left_bars=10, right_bars=5):
  if df is None or len(df) < left_bars + right_bars + 5:
    return {
        "active_support": np.nan,
        "is_broken": False,
        "status": "資料不足",
        "stats": {},
    }

  recent_len = min(504, len(df))
  recent_df = df.iloc[-recent_len:]
  cycle_high = float(recent_df["high"].max())
  cycle_low = float(recent_df["low"].min())
  cycle_range = cycle_high - cycle_low
  current_price = float(df["close"].iloc[-1])
  cycle_gann_50 = cycle_low + cycle_range * 0.50

  lows, closes = df["low"].values, df["close"].values
  active_support = None
  is_broken = False

  for i in range(left_bars + right_bars, len(df)):
    pivot_idx = i - right_bars
    if lows[pivot_idx] == np.nanmin(lows[pivot_idx - left_bars : i + 1]):
      active_support = float(lows[pivot_idx])
      is_broken = False
    if active_support is not None and not is_broken:
      if closes[i] < active_support:
        is_broken = True

  status_str = (
      "⚪ 尋底觀察中"
      if active_support is None
      else ("🔴 跌穿10日支持" if is_broken else "🟢 守穩10日支持")
  )
  return {
      "active_support": round(active_support, 3) if active_support else np.nan,
      "is_broken": is_broken,
      "status": status_str,
      "stats": {
          "cycle_high": cycle_high,
          "cycle_low": cycle_low,
          "cycle_gann_50": cycle_gann_50,
      },
  }


def af_snapshot(df):
  if df is None or len(df) < 40:
    return {}
  atr = calc_atr(df, 14)
  hi, lo = df["high"].max(), df["low"].min()
  rng = hi - lo
  if rng <= 0:
    return {}
  c = df["close"].iloc[-1]
  levels = {r: hi - r * rng for r in AUTO_FIB_RATIOS}
  near_r = min(levels, key=lambda r: abs(c - levels[r]))
  return {
      "auto_fib_near_zone": abs(c - levels[near_r]) <= atr.iloc[-1] * 0.35,
      "auto_fib_nearest_ratio": near_r,
  }


# ==================== 🎯 核心撈底潛力評分引擎 (整合N週期反彈與回調) ====================
def calc_bottom_fishing_engine(
    df, tspr, af_snap, cycle_sum, valuation_mos=np.nan, is_value_trap=False
):
  if df is None or len(df) < 60:
    return {
        "bottom_score": 0.0,
        "bottom_stage": "⚪ 資料不足",
        "drop_2y_pct": np.nan,
        "bottom_reasons": "資料不足",
    }

  close = df["close"]
  price = float(close.iloc[-1])

  lookback_2y = min(504, len(df))
  high_2y = float(df["high"].iloc[-lookback_2y:].max())
  low_2y = float(df["low"].iloc[-lookback_2y:].min())
  drop_2y = (price / high_2y - 1) * 100 if high_2y > 0 else 0.0

  w_df = (
      df.resample("W-FRI")
      .agg({
          "open": "first",
          "high": "max",
          "low": "min",
          "close": "last",
          "volume": "sum",
      })
      .dropna()
  )
  w_rsi = float(calc_rsi(w_df["close"], 14).iloc[-1]) if len(w_df) >= 15 else 50.0
  w_wr = float(calc_williams_r(w_df, 14).iloc[-1]) if len(w_df) >= 15 else -50.0
  _, _, w_j = calc_kdj(w_df)
  w_j_val = float(w_j.iloc[-1]) if len(w_df) >= 10 else 50.0
  w_cci = float(calc_cci(w_df, 20).iloc[-1]) if len(w_df) >= 20 else 0.0

  d_rsi = float(calc_rsi(close, 14).iloc[-1])
  mfi = float(calc_mfi(df, 14).iloc[-1])
  obv, obv_ma10 = calc_obv(df)
  obv_strong = bool(obv.iloc[-1] >= obv_ma10.iloc[-1])

  vol_z = (df["volume"] - df["volume"].rolling(20).mean()) / df[
      "volume"
  ].rolling(20).std()
  panic_washout = bool(
      ((close.pct_change().iloc[-12:] < -0.025) & (vol_z.iloc[-12:] > 1.6)).any()
      and price >= df["low"].iloc[-3:].min()
  )

  score = 0.0
  reasons = []

  # 維度 1：週線超賣度
  if w_wr <= -80:
    score += 12
    reasons.append(f"週WR極超賣({w_wr:.0f})")
  elif w_wr <= -68:
    score += 6
  if w_rsi <= 36 or d_rsi <= 32:
    score += 10
    reasons.append("RSI超賣")
  elif w_rsi <= 44 or d_rsi <= 38:
    score += 5
  if w_j_val <= 15 or w_cci <= -100:
    score += 8
    reasons.append("週KDJ_J/CCI見底區")

  # 🌟 維度 2：N週期反彈與回調轉勢訊號 + 兩年回撤
  if cycle_sum.get("turn_bull_signal"):
    score += 15
    reasons.append("🎯週期回調勾頭+反彈破10%轉勢")
  elif cycle_sum.get("correction_pct", 0) <= -20.0:
    score += 8
    reasons.append(f"半年週期回調{cycle_sum['correction_pct']:.1f}%")

  if -75 <= drop_2y <= -22:
    score += 10
    reasons.append(f"兩年高位回撤{drop_2y:.1f}%")

  gann_50 = low_2y + (high_2y - low_2y) * 0.50
  if abs(price - gann_50) / price <= 0.035 or af_snap.get("auto_fib_near_zone"):
    score += 10
    reasons.append("處於江恩50%或Fib關鍵位")

  # 維度 3：爆量承接
  if panic_washout:
    score += 12
    reasons.append("🔥急跌爆量洗盤後止跌")
  if mfi <= 28:
    score += 8
    reasons.append(f"MFI超賣({mfi:.0f})")
  if obv_strong:
    score += 10 if mfi >= 42 else 5
    reasons.append("OBV站上MA10承接")

  # 維度 4：右側確認與估值
  if not tspr.get("is_broken", True):
    score += 5
    reasons.append("守穩TSPR支持")
  if np.isfinite(valuation_mos) and valuation_mos >= 0.15:
    score += 8
    reasons.append(f"估值折讓{valuation_mos*100:.0f}%")
  if is_value_trap:
    score -= 18
    reasons.append("⚠️基本面紅旗扣分")

  final_score = round(float(np.clip(score, 0, 100)), 1)

  if is_value_trap and final_score >= 55:
    stage = "🟠 技術超賣 (留心基本面)"
  elif final_score >= 68:
    stage = "🟢 右側確認 (優先撈底)"
  elif final_score >= 52:
    stage = "🟡 左側超賣 (分批候位)"
  else:
    stage = "⚪ 未達撈底標準"

  return {
      "bottom_score": final_score,
      "bottom_stage": stage,
      "drop_2y_pct": round(drop_2y, 1),
      "w_rsi": round(w_rsi, 1),
      "w_wr": round(w_wr, 1),
      "w_kdj_j": round(w_j_val, 1),
      "panic_washout": panic_washout,
      "bottom_reasons": "｜".join(reasons) or "暫無明顯見底訊號",
  }


def quality_engine(info):
  roe, rev = safe_float(info.get("roe")), safe_float(info.get("rev_growth"))
  trap = bool(
      (np.isfinite(roe) and roe < 0.05) and (np.isfinite(rev) and rev < -0.10)
  )
  return {"value_trap": trap}


def valuation_engine(ticker, info, sector, price):
  if ticker in ETF_TICKERS or ticker.startswith("^"):
    return {
        "val_level": "📊 指數/ETF",
        "margin_of_safety": np.nan,
        "fair_base": np.nan,
    }
  eps = safe_float(info.get("eps_quote"))
  if (
      not np.isfinite(price)
      or price <= 0
      or not np.isfinite(eps)
      or eps <= 0
  ):
    return {
        "val_level": "⚪ 估值中性",
        "margin_of_safety": np.nan,
        "fair_base": np.nan,
    }
  pe_base = SECTOR_PE_RANGES.get(sector, DEFAULT_PE_RANGE)[1] * (
      HK_MULTIPLE_DISCOUNT if ticker.endswith(".HK") else 1.0
  )
  fair_base = eps * pe_base
  mos = 1.0 - price / fair_base
  level = (
      "🟢 重度低估"
      if mos >= 0.30
      else (
          "🟠 中度低估"
          if mos >= 0.15
          else ("⚪ 估值合理" if mos >= -0.12 else "🔴 估值偏高")
      )
  )
  return {"val_level": level, "margin_of_safety": mos, "fair_base": fair_base}


def analyze_single_stock(ticker, df, info):
  if df is None or len(df) < 60:
    return None
  price = safe_float(df["close"].iloc[-1])
  sector = info.get("sector") or SECTOR_MAP_FALLBACK.get(ticker, "OTHER")
  pe_ttm = safe_float(info.get("pe_ttm"))

  tspr = calc_tspr_system(df.iloc[-504:])
  af_snap = af_snapshot(df)
  # 🌟 計算 120 日 (半年) 週期反彈與回調指標
  _, cycle_sum = calc_rebound_correction_dashboard(
      df,
      window=120,
      bull_thresh=40.0 if (ticker.startswith("^") or ticker in ETF_TICKERS) else 50.0,
      bear_thresh=-18.0 if (ticker.startswith("^") or ticker in ETF_TICKERS) else -25.0,
  )
  quality = quality_engine(info)
  valuation = valuation_engine(ticker, info, sector, price)
  bottom = calc_bottom_fishing_engine(
      df,
      tspr,
      af_snap,
      cycle_sum,
      valuation["margin_of_safety"],
      quality["value_trap"],
  )

  atr = safe_float(calc_atr(df).iloc[-1])
  stop = (
      max(tspr.get("active_support", price - 2 * atr), price * 0.90)
      if np.isfinite(tspr.get("active_support", np.nan))
      else price - 2 * atr
  )
  target_2r = price + 2.0 * max(price - stop, price * 0.02)

  return {
      "ticker": ticker,
      "name": info.get("name", ticker),
      "sector": sector,
      "pe_ttm": round(pe_ttm, 2) if np.isfinite(pe_ttm) else np.nan,
      "price": round(price, 2),
      "last_bar": str(pd.Timestamp(df.index[-1]).date()),
      "bottom_score": bottom["bottom_score"],
      "bottom_stage": bottom["bottom_stage"],
      "rebound_pct": cycle_sum["rebound_pct"],  # 🌟 新增：週期反彈%
      "correction_pct": cycle_sum["correction_pct"],  # 🌟 新增：週期回調%
      "cycle_status": cycle_sum["cycle_status"],  # 🌟 新增：週期狀態
      "drop_2y_pct": bottom["drop_2y_pct"],
      "w_wr": bottom["w_wr"],
      "w_rsi": bottom["w_rsi"],
      "w_kdj_j": bottom["w_kdj_j"],
      "panic_washout": bottom["panic_washout"],
      "bottom_reasons": bottom["bottom_reasons"],
      "val_level": valuation["val_level"],
      "mos_pct": (
          round(valuation["margin_of_safety"] * 100, 1)
          if np.isfinite(valuation["margin_of_safety"])
          else np.nan
      ),
      "fair_base": (
          round(valuation["fair_base"], 2)
          if np.isfinite(valuation["fair_base"])
          else np.nan
      ),
      "stop_loss": round(stop, 2),
      "target_2r": round(target_2r, 2),
      "tspr_status": tspr["status"],
  }


# ==================== 側邊欄與掃描流程 ====================
with st.sidebar:
  st.markdown("## ⚙️ 撈底與週期雷達面板")
  pool_choice = st.radio(
      "選擇掃描股票池",
      [
          "🔥 全景池 (大盤指數ETF + 美股 + 港股)",
          "📊 大盤指數與核心 ETF 專區 (HSI/標普/納指/QQQ/2800)",
          "🇺🇸 美股核心池 (三大指數代表股)",
          "🇭🇰 港股核心池 (港股前30大)",
          "📋 自訂輸入代碼",
      ],
      index=0,
  )
  custom_input = ""
  if pool_choice == "📋 自訂輸入代碼":
    custom_input = st.text_area(
        "每行一個代碼 (支援 ^HSI, ^GSPC, ^NDX, QQQ, 2800.HK)",
        "^HSI\n^GSPC\n^NDX\nQQQ\n2800.HK\nGOOGL\nNFLX",
    )

  fetch_fundamentals = st.checkbox("同步抓取 Yahoo 基本面估值 (含PE)", value=True)
  scan_btn = st.button("🚀 立即執行掃描", type="primary", use_container_width=True)


def get_selected_tickers():
  if pool_choice == "📋 自訂輸入代碼":
    return sorted(
        {
            normalize_symbol(x)
            for x in custom_input.splitlines()
            if normalize_symbol(x)
        }
    )
  if pool_choice.startswith("📊"):
    return sorted(set(INDEX_ETF_WATCHLIST))
  if pool_choice.startswith("🇺🇸"):
    return sorted(set(CORE_US_WATCHLIST))
  if pool_choice.startswith("🇭🇰"):
    return sorted(set(CORE_HK_WATCHLIST))
  return sorted(
      set(INDEX_ETF_WATCHLIST + CORE_US_WATCHLIST + CORE_HK_WATCHLIST)
  )


def execute_scan():
  tickers = get_selected_tickers()
  if not tickers:
    st.warning("股票池為空。")
    return
  progress = st.progress(5, text=f"正在下載 {len(tickers)} 隻指數與股票資料…")
  prices = fetch_multiple(tickers, period="5y")

  info_map = {}
  if fetch_fundamentals:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_INFO) as executor:
      futures = {executor.submit(get_full_stock_info, t): t for t in tickers}
      for f in as_completed(futures):
        info_map[futures[f]] = f.result()

  results = []
  for t in tickers:
    df = prices.get(t)
    if df is not None and not df.empty:
      info = (
          info_map.get(t) or get_full_stock_info(t)
          if fetch_fundamentals
          else {"ticker": t, "name": INDEX_NAME_MAP.get(t, t)}
      )
      res = analyze_single_stock(t, df, info)
      if res:
        results.append(res)

  progress.empty()
  st.session_state["latest_results"] = results
  st.session_state["price_cache"] = prices


st.markdown("# 📈 股票監察系統 V5.3｜🎯 週期反彈回調儀表板 ＋ 大盤指數雷達")
st.caption(
    "整合「N週期反彈與回調儀表板 (Rebound & Correction Dashboard)」，支援恒指(^HSI)、標普500(^GSPC)、納指100(^NDX)、QQQ、盈富(2800.HK) 及美港核心股週期轉勢監測。"
)

if scan_btn or "latest_results" not in st.session_state:
  execute_scan()

# ==================== 四大分頁 ====================
tabs = st.tabs([
    "🎯 1. 撈底雷達 & 週期排行榜",
    "📈 2. 指數與個股『週期反彈/回調』深度診斷",
    "⚖️ 3. 倉位計算",
    "🛠️ 4. 操作口訣與設定",
])

with tabs[0]:
  results = st.session_state.get("latest_results", [])
  if results:
    df_all = pd.DataFrame(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已掃描總數", f"{len(df_all)} 隻")
    c2.metric(
        "🟢 右側確認 (優先撈底)",
        f"{(df_all['bottom_stage'].str.startswith('🟢')).sum()} 隻",
    )
    c3.metric(
        "🔄 週期見底/超賣勾頭",
        f"{(df_all['cycle_status'].str.startswith(('🟢', '🟡'))).sum()} 隻",
    )
    c4.metric(
        "💎 具備估值安全邊際",
        f"{(pd.to_numeric(df_all['mos_pct'], errors='coerce') >= 15).sum()} 隻",
    )

    st.markdown("---")
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
      rank_mode = st.radio(
          "排行排序模式",
          [
              "🎯 按『撈底潛力總分』排序",
              "📉 按『當前期週期回調%』由深至淺排序",
              "📉 按『過去兩年高位回撤幅度』排序",
              "📊 按『市盈率 (PE TTM)』由低至高",
              "💎 按『估值安全邊際%』排序",
          ],
          horizontal=False,
      )
    with col_f2:
      market_filter = st.radio(
          "市場篩選",
          ["全部", "📊 大盤指數與ETF", "🇺🇸 美股個股", "🇭🇰 港股個股"],
          horizontal=True,
      )

    view_df = df_all.copy()
    if market_filter == "📊 大盤指數與ETF":
      view_df = view_df[
          view_df["ticker"].isin(INDEX_ETF_WATCHLIST)
          | view_df["ticker"].str.startswith("^")
      ]
    elif market_filter == "🇺🇸 美股個股":
      view_df = view_df[
          (~view_df["ticker"].str.endswith(".HK"))
          & (~view_df["ticker"].isin(INDEX_ETF_WATCHLIST))
          & (~view_df["ticker"].str.startswith("^"))
      ]
    elif market_filter == "🇭🇰 港股個股":
      view_df = view_df[
          view_df["ticker"].str.endswith(".HK")
          & (~view_df["ticker"].isin(INDEX_ETF_WATCHLIST))
      ]

    if rank_mode.startswith("🎯"):
      view_df = view_df.sort_values(
          ["bottom_score", "mos_pct"], ascending=[False, False]
      )
    elif rank_mode.startswith("📉 按『當前期週期回調"):
      view_df = view_df.sort_values("correction_pct", ascending=True)
    elif rank_mode.startswith("📉 按『過去兩年"):
      view_df = view_df.sort_values("drop_2y_pct", ascending=True)
    elif rank_mode.startswith("📊"):
      view_df = view_df.sort_values("pe_ttm", ascending=True)
    else:
      view_df = view_df.sort_values(
          ["mos_pct", "bottom_score"], ascending=[False, False]
      )

    display_cols = {
        "ticker": "代號",
        "name": "名稱",
        "sector": "分類",
        "price": "現價",
        "bottom_score": "🎯 總分",
        "bottom_stage": "撈底階段",
        "rebound_pct": "週期反彈%",
        "correction_pct": "週期回調%",
        "cycle_status": "週期位置狀態",
        "drop_2y_pct": "兩年回撤%",
        "w_wr": "週WR",
        "pe_ttm": "PE(TTM)",
        "val_level": "估值狀態",
        "tspr_status": "TSPR支持",
        "bottom_reasons": "觸發理由",
    }
    st.dataframe(
        view_df[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
        height=550,
    )

# ==================== 🌟 Tab 2: 指數與個股『週期反彈/回調』深度診斷 ====================
with tabs[1]:
  st.markdown("### 📊 大盤指數與個股：N週期反彈與回調儀表板 (TradingView 升級版)")

  results = st.session_state.get("latest_results", [])
  price_cache = st.session_state.get("price_cache", {})

  # 頂部快速顯示大盤指數健康度總覽
  if results:
    df_idx = pd.DataFrame(results)
    df_idx = df_idx[
        df_idx["ticker"].isin(INDEX_ETF_WATCHLIST)
        | df_idx["ticker"].str.startswith("^")
    ]
    if not df_idx.empty:
      with st.expander(
          "🌍 點擊展開：全球核心指數與 ETF 週期位置總表 (^HSI / ^GSPC / ^NDX / QQQ / 2800.HK)",
          expanded=True,
      ):
        idx_cols = {
            "ticker": "指數/ETF",
            "name": "名稱",
            "price": "最新點數/股價",
            "rebound_pct": "週期反彈%",
            "correction_pct": "週期回調%",
            "cycle_status": "牛熊週期診斷",
            "drop_2y_pct": "兩年高位回撤%",
            "w_wr": "週線WR",
            "bottom_score": "撈底分",
        }
        st.dataframe(
            df_idx[list(idx_cols.keys())].rename(columns=idx_cols),
            use_container_width=True,
            hide_index=True,
        )

  col_ctl1, col_ctl2, col_ctl3, col_ctl4 = st.columns([2, 1.5, 1.5, 2])
  available_tickers = (
      [r["ticker"] for r in results] if results else INDEX_ETF_WATCHLIST
  )
  default_idx = (
      available_tickers.index("^HSI")
      if "^HSI" in available_tickers
      else (available_tickers.index("QQQ") if "QQQ" in available_tickers else 0)
  )

  with col_ctl1:
    sel_ticker = st.selectbox(
        "選擇診斷標的 (支援指數、ETF或個股)",
        available_tickers,
        index=default_idx,
        format_func=lambda x: f"{x} - {INDEX_NAME_MAP.get(x, '')}",
    )
  with col_ctl2:
    tf_mode = st.radio(
        "K線時間框架", ["週K (Weekly - 推薦長線)", "日K (Daily - 波段)"], index=0
    )
  with col_ctl3:
    default_win = 52 if tf_mode.startswith("週K") else 120
    lookback_n = st.number_input(
        "回溯週期 N (週K建議52，日K建議120)",
        min_value=10,
        max_value=300,
        value=default_win,
        step=5,
    )
  with col_ctl4:
    is_idx = sel_ticker.startswith("^") or sel_ticker in ETF_TICKERS
    bear_line = st.slider(
        "超賣警戒線 (-%)",
        min_value=-50,
        max_value=-10,
        value=-20 if is_idx else -25,
        step=5,
    )
    bull_line = st.slider(
        "過熱警戒線 (+%)",
        min_value=20,
        max_value=120,
        value=40 if is_idx else 50,
        step=5,
    )

  df_raw = price_cache.get(sel_ticker) or fetch_ohlcv(sel_ticker, period="5y")
  if df_raw is not None and not df_raw.empty:
    if tf_mode.startswith("週K"):
      df_plot = (
          df_raw.resample("W-FRI")
          .agg({
              "open": "first",
              "high": "max",
              "low": "min",
              "close": "last",
              "volume": "sum",
          })
          .dropna()
      )
    else:
      df_plot = df_raw.copy()

    df_calc, cyc_info = calc_rebound_correction_dashboard(
        df_plot,
        window=int(lookback_n),
        bull_thresh=float(bull_line),
        bear_thresh=float(bear_line),
    )
    tspr_info = calc_tspr_system(df_raw.iloc[-504:])

    # 頂部五格實時儀表板卡片
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        f"{sel_ticker} 最新收市",
        f"{df_calc['close'].iloc[-1]:,.2f}",
        f"{df_calc['close'].pct_change().iloc[-1]*100:+.2f}%",
    )
    m2.metric(
        f"當前期反彈 ({lookback_n}期)",
        f"+{cyc_info['rebound_pct']:.2f}%",
        f"近250期極值 +{cyc_info['max_rebound']:.1f}%",
    )
    m3.metric(
        f"當前期回調 ({lookback_n}期)",
        f"{cyc_info['correction_pct']:.2f}%",
        f"近250期極值 {cyc_info['max_correction']:.1f}%",
    )
    m4.metric(
        "動態歷史分位閾值",
        f"超賣 {cyc_info['bear_dyn']}%",
        f"過熱 +{cyc_info['bull_dyn']}%",
    )
    m5.metric("當前週期轉勢判定", cyc_info["cycle_status"])

    # 繪製雙層互動圖表 (主圖 K 線 + 副圖 Rebound & Correction Dashboard)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            f"{sel_ticker} ({INDEX_NAME_MAP.get(sel_ticker, sel_ticker)}) - 價格與兩年江恩50%/TSPR支持",
            f"N-Period Rebound & Correction Dashboard (N={lookback_n}｜超賣線={bear_line}%｜過熱線=+{bull_line}%)",
        ),
    )

    # 1. 主圖 Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_calc.index,
            open=df_calc["open"],
            high=df_calc["high"],
            low=df_calc["low"],
            close=df_calc["close"],
            name="K線",
            increasing_line_color=C_GREEN,
            decreasing_line_color=C_RED,
        ),
        row=1,
        col=1,
    )

    # 畫出兩年江恩 50% 中軸與 TSPR 支持位
    gann50 = tspr_info.get("stats", {}).get("cycle_gann_50", np.nan)
    if np.isfinite(gann50):
      fig.add_hline(
          y=gann50,
          line_dash="dash",
          line_color=C_ORANGE,
          annotation_text=f"兩年江恩50% ({gann50:,.2f})",
          row=1,
          col=1,
      )
    sup_lvl = tspr_info.get("active_support", np.nan)
    if np.isfinite(sup_lvl):
      fig.add_hline(
          y=sup_lvl,
          line_dash="dot",
          line_color=C_BLUE,
          annotation_text=f"TSPR支持 ({sup_lvl:,.2f})",
          row=1,
          col=1,
      )

    # 2. 副圖：週期反彈% (松石青綠 + 零軸面積填充)
    fig.add_trace(
        go.Scatter(
            x=df_calc.index,
            y=df_calc["rebound_pct"],
            name="週期反彈 %",
            line=dict(color=C_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.18)",
        ),
        row=2,
        col=1,
    )

    # 3. 副圖：週期回調% (珊瑚紅 + 零軸面積填充)
    fig.add_trace(
        go.Scatter(
            x=df_calc.index,
            y=df_calc["correction_pct"],
            name="週期回調 %",
            line=dict(color=C_RED, width=2),
            fill="tozeroy",
            fillcolor="rgba(239, 83, 80, 0.22)",
        ),
        row=2,
        col=1,
    )

    # 副圖：過熱線、零軸、超賣線
    fig.add_hline(
        y=bull_line,
        line_dash="dot",
        line_color=C_GREEN,
        annotation_text=f"過熱線 (+{bull_line}%)",
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="#555555", row=2, col=1)
    fig.add_hline(
        y=bear_line,
        line_dash="dot",
        line_color=C_RED,
        annotation_text=f"超賣線 ({bear_line}%)",
        row=2,
        col=1,
    )

    # 副圖：標記「見底轉勢」綠色向上三角形
    bull_turns = df_calc[df_calc["turn_bull_signal"]]
    if not bull_turns.empty:
      fig.add_trace(
          go.Scatter(
              x=bull_turns.index,
              y=bull_turns["correction_pct"] * 1.08,
              mode="markers",
              name="見底轉勢訊號",
              marker=dict(
                  symbol="triangle-up",
                  size=12,
                  color="#00e676",
                  line=dict(width=1, color="white"),
              ),
          ),
          row=2,
          col=1,
      )

    fig.update_layout(
        template="plotly_dark",
        height=760,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        legend=dict(orientation="h", y=1.03, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
  st.markdown("### ⚖️ 風險定倉計算機 (以損定倉)")
  c_p1, c_p2, c_p3 = st.columns(3)
  capital = c_p1.number_input("總投資本金 ($)", value=100000.0, step=10000.0)
  risk_pct = c_p2.number_input("單筆最大承受虧損 (%)", value=1.5, step=0.5)
  entry_px = c_p3.number_input("預計買入價 ($)", value=100.0, step=1.0)
  stop_px = st.number_input("預計止蝕價 ($)", value=92.0, step=1.0)
  if entry_px > stop_px > 0:
    max_loss = capital * (risk_pct / 100.0)
    shares = int(max_loss // (entry_px - stop_px))
    st.success(
        f"✅ **建議買入股數：{shares:,} 股** ｜ 預計動用資金：`${shares * entry_px:,.2f}` ｜ 觸及止蝕最大虧損：`${shares * (entry_px - stop_px):,.2f}`"
    )

with tabs[3]:
  st.markdown("""
    ### 🛠️ 週期反彈與回調指標 — 實戰操作四句口訣
    1. **紅線深跌先觀察**：當 `週期回調%` 跌破超賣線（大盤指數約 `-18% ~ -20%`，大型股約 `-25% ~ -30%`），代表進入左側觀察區，但綠線貼在 `0%` 時切忌盲目接飛刀。
    2. **綠線抬頭才買入**：等紅線停止創新低並向上勾頭，同時 `週期反彈%`（綠線）從 `0%` 向上突破 `10% ~ 12%`（圖表出現綠色向上三角形），確認見底轉勢才扣扳機。
    3. **綠線狂升莫急賣**：大牛市或強勢股主升浪中，只要紅線一直緊貼 `0%`（代表不斷創階段新高），就算綠線衝過 `+50%` 過熱線亦可續抱。
    4. **紅線開口才食糊**：當綠線在高位掉頭向下，同時紅線從 `0%` 向下跌穿 `-6% ~ -8%`，代表升浪動能衰竭，可執行移動止賺。
    """)
