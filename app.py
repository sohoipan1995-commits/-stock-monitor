import os
import time
import warnings
import threading
import sqlite3
import math
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

st.set_page_config(page_title="📈 終極撈底系統 V5.0 (大市氣氛 + 江恩轉勢 + 回調階梯)", page_icon="🎯", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

APP_VERSION = "V5.0-Bottom-Fishing"
MODEL_VERSION = "decision-v5.0-bottom-fishing"
AUTO_REFRESH_SEC = 1800
OHLCV_TTL = 1800
INFO_TTL = 3600
MAX_WORKERS_DATA = 5
MAX_WORKERS_SCORE = 4
FUTU_LOCK = threading.RLock()

HK_WATCHLIST = [
    "0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK",
    "0066.HK", "0003.HK", "0002.HK", "0016.HK", "0883.HK", "2318.HK",
    "1299.HK", "0001.HK", "9988.HK", "0175.HK", "0027.HK", "2628.HK",
    "0011.HK", "0688.HK", "3690.HK", "9618.HK", "0981.HK", "9999.HK",
    "2382.HK", "0291.HK", "1211.HK", "0267.HK", "2688.HK", "0762.HK",
    "6862.HK", "0960.HK", "2020.HK", "1810.HK", "1024.HK", "3033.HK", "2800.HK"
]

US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "ORCL", "ASML", "AMD", "QCOM", "INTC", "AMAT", "LRCX", "MU",
    "JPM", "BAC", "GS", "MS", "BRK-B", "COST", "WMT", "HD", "JNJ",
    "UNH", "PFE", "XOM", "NEE", "UBER", "LITE", "CLX", "NFLX", "ADBE", "CRM"
]

ETF_TICKERS = {"SPY", "QQQ", "SOXL", "IWM", "3033.HK", "2800.HK", "2828.HK"}
MACRO_TICKERS = {
    "VIX": "^VIX", "VVIX": "^VVIX", "SPX": "^GSPC", "HSI": "^HSI",
    "NDX": "^NDX", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VHSI": "^VHSI",
    "HYG": "HYG", "USDHKD": "USDHKD=X"
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0d1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3,h4,h5,h6,p,label,.stMarkdown {color:#e6edf3!important;}
.metric-card {background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;margin:4px;}
.decision-card {background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:16px;margin:10px 0;}
.warn-card {background:#2b1a1a;border:1px solid #f85149;border-radius:10px;padding:12px;margin:8px 0;}
.good-card {background:#102419;border:1px solid #3fb950;border-radius:10px;padding:16px;margin:12px 0;}
.ladder-box {background:#21262d; border:1px solid #30363d; border-radius:6px; padding:12px; margin-top:10px; font-family: monospace; font-size: 14px;}
.tspr-panel {background:#161b22;border:1px solid #4a148c;border-radius:8px;padding:12px;margin-bottom:12px;}
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

def add_error(context, ticker=None, exc=None):
    message = str(context)
    if ticker: message += f"｜{ticker}"
    if exc: message += f"｜{type(exc).__name__}: {str(exc)[:180]}"
    st.session_state.error_log.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "訊息": message})
    st.session_state.error_log = st.session_state.error_log[-50:]

def clear_errors():
    st.session_state.error_log = []

def to_futu(ticker):
    if ticker.endswith(".HK"): return f"HK.{ticker[:-3].zfill(5)}"
    if ticker.replace("-", "").isalpha(): return f"US.{ticker}"
    return ticker

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
                ret, data, _ = quote_ctx.request_history_kline(to_futu(ticker), start=start, end=None, ktype=KLType.K_DAY, max_count=1000)
            if ret == RET_OK and data is not None and not data.empty:
                df = data[["time_key", "open", "high", "low", "close", "volume"]].rename(columns={"time_key": "date"})
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index().dropna()
        except Exception:
            pass
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty: return None
        
        # 處理 yfinance 多重索引
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # 【核心修復】去除重複的欄位名 (避免 df["low"] 變成二維 DataFrame 導致後續報錯)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        return df.sort_index().dropna(subset=["open", "high", "low", "close", "volume"])
    except Exception as exc:
        add_error("Yahoo OHLCV 失敗", ticker, exc)
        return None

def fetch_multiple(tickers, period="2y", max_workers=MAX_WORKERS_DATA):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, period): ticker for ticker in tickers}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results

@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    result = {"ticker": ticker, "name": ticker, "pe": None, "pb": None, "div_yield": None, "roe": None, "de_ratio": None, "sector": None}
    try:
        info = yf.Ticker(ticker).info or {}
        result["name"] = info.get("shortName") or ticker
        pe = info.get("trailingPE") or info.get("forwardPE")
        result["pe"] = safe_float(pe)
        result["pb"] = safe_float(info.get("priceToBook"))
        result["div_yield"] = safe_float(info.get("dividendYield"))
        result["roe"] = safe_float(info.get("returnOnEquity"))
        result["de_ratio"] = safe_float(info.get("debtToEquity"))
        result["sector"] = info.get("sector")
    except Exception:
        pass
    return result

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def calc_weekly_rsi_daily_aligned(close, period=14):
    weekly_close = close.resample("W-FRI").last().dropna()
    weekly_rsi = calc_rsi(weekly_close, period)
    return weekly_rsi.reindex(close.index, method="ffill"), weekly_rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def get_52w_high(df):
    high_col = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    highs = np.asarray(high_col, dtype=float)
    if len(highs) >= 252:
        window = highs[-252:]
    else:
        window = highs
    if len(window) > 0 and not np.all(np.isnan(window)):
        return float(np.nanmax(window))
    return np.nan

# ==================== TSPR + 江恩系統模組 ====================
def calc_gann_time_and_cloud_system(df, direction="auto", manual_scale=0.0):
    # 【防呆處理】確保所有特徵都是一維 Series，防止意外成為 DataFrame
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    high = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    low = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    vol = df["volume"].iloc[:, 0] if isinstance(df["volume"], pd.DataFrame) else df["volume"]
    
    n = len(df)
    ema_fast_1, ema_mid_1, ema_slow_1 = close.ewm(span=5).mean(), close.ewm(span=21).mean(), close.ewm(span=55).mean()
    cloud_bullish = bool(ema_fast_1.iloc[-1] > ema_mid_1.iloc[-1] > ema_slow_1.iloc[-1])
    
    vol_ma20 = vol.rolling(20).mean()
    rvol = vol / vol_ma20.replace(0, np.nan)
    vqc_mask = (rvol >= 1.8) & (vol > 0)

    lookback = min(n, 140)
    sub_df_high = high.iloc[-lookback:]
    sub_df_low = low.iloc[-lookback:]
    offset_idx = n - lookback
    
    highs_sub = np.asarray(sub_df_high, dtype=float)
    lows_sub = np.asarray(sub_df_low, dtype=float)
    
    if not np.all(np.isnan(highs_sub)):
        anchor_rel_hi = int(np.nanargmax(highs_sub))
        anchor_price_hi = float(highs_sub[anchor_rel_hi])
    else:
        anchor_rel_hi, anchor_price_hi = 0, 0.0
        
    if not np.all(np.isnan(lows_sub)):
        anchor_rel_lo = int(np.nanargmin(lows_sub))
        anchor_price_lo = float(lows_sub[anchor_rel_lo])
    else:
        anchor_rel_lo, anchor_price_lo = 0, 0.0
    
    if direction == "up" or (direction == "auto" and close.iloc[-1] > anchor_price_hi * 0.96):
        anchor_rel = anchor_rel_lo
        anchor_price = anchor_price_lo
        is_down = False
    else:
        anchor_rel = anchor_rel_hi
        anchor_price = anchor_price_hi
        is_down = True

    anchor_idx = offset_idx + anchor_rel
    bars_elapsed = max(1, (n - 1) - anchor_idx)
    
    if manual_scale > 0:
        price_per_bar = float(manual_scale)
    else:
        valid_range = anchor_price_hi - anchor_price_lo
        price_per_bar = round(float(valid_range) / max(20, lookback * 0.75), 6)
        
    sign = -1.0 if is_down else 1.0
    current_gann_1x1 = anchor_price + sign * price_per_bar * (n - 1 - anchor_idx)
    
    fan_lines = []
    for label, mult in [("2x1", 2.0), ("1x1", 1.0), ("1x2", 0.5)]:
        y_vals = anchor_price + sign * (price_per_bar * mult) * (np.arange(anchor_idx, n) - anchor_idx)
        fan_lines.append({"label": label, "dates": df.index[anchor_idx:], "values": y_vals})
        
    return {
        "ema_fast_1": ema_fast_1, "ema_mid_1": ema_mid_1, "ema_slow_1": ema_slow_1,
        "cloud_bullish": cloud_bullish, "vqc_dates": df.index[vqc_mask].tolist(), 
        "anchor_date": df.index[anchor_idx], "anchor_price": anchor_price, "price_per_bar": price_per_bar,
        "current_gann_1x1": round(current_gann_1x1, 3), "fan_lines": fan_lines, "bars_elapsed": bars_elapsed,
        "gann_direction": "向下" if is_down else "向上"
    }

def predict_gann_turn_dates(df, forward_trading_days=45):
    if df is None or len(df) < 60: return {"top_turns": []}
    dates = pd.DatetimeIndex(df.index)
    
    # 【防呆處理】強制降維
    high_col = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    low_col = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    
    highs = np.asarray(high_col, dtype=float)
    lows = np.asarray(low_col, dtype=float)
    n = len(df)
    
    pivots = []
    offset = n - 180 if n > 180 else 0
    
    for i in range(7, (180 if n > 180 else n) - 7):
        g_idx = offset + i
        w_highs = highs[g_idx - 7 : g_idx + 8]
        w_lows = lows[g_idx - 7 : g_idx + 8]
        
        if len(w_highs) > 0 and not np.all(np.isnan(w_highs)):
            if highs[g_idx] == np.nanmax(w_highs): 
                pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "頂"})
        if len(w_lows) > 0 and not np.all(np.isnan(w_lows)):
            if lows[g_idx] == np.nanmin(w_lows): 
                pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "底"})
        
    pivots = sorted({p["idx"]: p for p in pivots}.values(), key=lambda x: x["idx"])
    future_bdays = pd.bdate_range(start=dates[-1] + timedelta(days=1), periods=forward_trading_days)
    
    trading_cycles = {15: 1.5, 21: 2.0, 30: 2.5, 45: 3.5, 60: 3.0, 90: 4.5, 120: 3.5}
    gann_seasonal_dates = [(3, 21), (6, 21), (9, 22), (12, 21), (2, 4), (5, 5), (8, 7), (11, 7)]
    
    records = []
    for step, f_date in enumerate(future_bdays, start=1):
        future_idx = (n - 1) + step
        score, reasons = 0.0, []
        
        for pv in pivots[-5:]:
            el_td = future_idx - pv["idx"]
            if el_td in trading_cycles:
                score += trading_cycles[el_td]
                reasons.append(f"{el_td}交易日循環")
            el_cd = (f_date - pv["date"]).days
            if any(abs(el_cd - cc) <= 1 for cc in [30, 45, 60, 90, 180]):
                score += 2.0
                reasons.append(f"自然日循環")
                
        for m, d in gann_seasonal_dates:
            if abs((f_date - pd.Timestamp(year=f_date.year, month=m, day=d)).days) <= 1:
                score += 3.5
                reasons.append("江恩節氣轉折")
                
        if score > 0:
            records.append({
                "預測轉勢日期": f_date.strftime("%Y-%m-%d"),
                "星期": ["一", "二", "三", "四", "五", "六", "日"][f_date.weekday()],
                "共振強度分": score,
                "時間周期依據": " + ".join(set(reasons))
            })
            
    hit_df = pd.DataFrame(records).sort_values(["共振強度分", "預測轉勢日期"], ascending=[False, True]) if records else pd.DataFrame()
    return {"top_turns": hit_df.head(5).to_dict("records") if not hit_df.empty else []}

def calc_tspr_system(df, left_bars=10):
    if df is None or len(df) < 30: return {"active_support": np.nan, "is_broken": False, "status": "資料不足"}
    
    # 【防呆處理】確保提取的絕對是 1D 陣列
    low_col = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    close_col = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    
    lows = np.asarray(low_col, dtype=float)
    closes = np.asarray(close_col, dtype=float)
    active_support = None
    is_broken = False
    
    for i in range(left_bars + 5, len(df)):
        pivot_idx = i - 5
        start_idx = max(0, pivot_idx - left_bars)
        window = lows[start_idx : i + 1]
        
        if len(window) > 0 and not np.all(np.isnan(window)):
            if not np.isnan(lows[pivot_idx]) and lows[pivot_idx] == np.nanmin(window):
                active_support = float(lows[pivot_idx])
                is_broken = False
                
        if active_support is not None and closes[i] < active_support:
            is_broken = True
            
    status_str = "🔴 跌穿10日支持" if is_broken else ("🟢 守穩10日支持" if active_support else "⚪ 尋底中")
    return {"active_support": round(active_support, 3) if active_support else np.nan, "is_broken": is_broken, "status": status_str}

# ==================== 核心：溫度計與撈底過濾引擎 ====================
@st.cache_data(ttl=OHLCV_TTL)
def fetch_macro():
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        df = fetch_ohlcv(ticker, period="1mo")
        if df is not None and not df.empty:
            # 確保提取為 1D
            close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
            result[name] = {"val": safe_float(close.iloc[-1]), "chg": (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close)>1 else 0}
    return result

def classify_market_state(market, macro_data):
    benchmark = "^HSI" if market == "🇭🇰 港股" else "SPY"
    vol_key = "VHSI" if market == "🇭🇰 港股" else "VIX"
    df = fetch_ohlcv(benchmark, period="1y")
    if df is None or len(df) < 205: return {"name": "未知", "strategy": "數據不足，觀望為主。"}
    
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    ma50, ma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    rsi_14 = calc_rsi(close).iloc[-1]
    vol_index = safe_float(macro_data.get(vol_key, {}).get("val"), 20)
    
    if price < ma200 and vol_index > 28 and rsi_14 < 35:
        name, strategy = "🥶 冰點 (恐慌拋售區)", "大市極度恐慌，優質股被錯殺。此時是【分批撈底高價值股】黃金時期，重點看基本面與 MACD 底背馳。"
    elif price < ma200 and vol_index <= 28:
        name, strategy = "🧊 冷卻 (反覆築底區)", "大市處於跌勢但恐慌減退。關注江恩轉勢日，等待 TSPR 10日支持位確立才小注試倉，嚴格止損。"
    elif price > ma200 and rsi_14 > 70:
        name, strategy = "🔥 過熱 (狂熱泡沫區)", "大市超買。停止追入新倉，現有持倉應利用江恩 1x1 線或 10日支持位作移動止盈，準備鎖定利潤。"
    elif price > ma200:
        name, strategy = "☀️ 溫和 (穩健牛市區)", "大市趨勢向上。持有優質股，逢回調至 MA50 或江恩 50% 分水嶺可加倉。"
    else:
        name, strategy = "⛅ 中性 (震盪上落區)", "區間上落。高估值不追，低估值可買，嚴格控制注碼。"
        
    return {"name": name, "benchmark": benchmark, "strategy": strategy, "vol": vol_index}

def decision_engine(ticker, df, info):
    if df is None or len(df) < 210: return None
    
    close_col = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    close = close_col
    price = safe_float(close.iloc[-1])
    high52 = get_52w_high(df)
    
    rsi, weekly_rsi = safe_float(calc_rsi(close).iloc[-1], 50), safe_float(calc_weekly_rsi_daily_aligned(close)[0].iloc[-1], 50)
    macd, macd_signal, _ = calc_macd(close)
    
    tspr = calc_tspr_system(df)
    gann_cloud = calc_gann_time_and_cloud_system(df)
    turn_info = predict_gann_turn_dates(df, 30)
    
    nt = turn_info["top_turns"][0] if turn_info["top_turns"] else None
    next_turn_str = f"{nt['預測轉勢日期']} ({nt['共振強度分']}分)" if nt else "—"

    # 1. 估值評分
    pe, pb = safe_float(info.get("pe")), safe_float(info.get("pb"))
    val_parts = []
    if pe > 0: val_parts.append(90 if pe < 10 else 70 if pe < 15 else 45 if pe < 23 else 20)
    if pb > 0: val_parts.append(90 if pb < 1 else 70 if pb < 1.5 else 45 if pb < 3 else 20)
    val_score = float(np.mean(val_parts)) if val_parts else 50.0
    
    if val_score >= 78: undervaluation = "🟢 重度被低估"
    elif val_score >= 62: undervaluation = "🟠 中度被低估"
    elif val_score >= 48: undervaluation = "🟡 輕度被低估"
    else: undervaluation = "⚪ 估值未見吸引"
    
    roe = safe_float(info.get("roe"))
    blockers = []
    if roe < 0.05: blockers.append("ROE < 5% (價值陷阱)")

    # 2. 技術見底特徵 (Bottom Score)
    oversold, reasons = 0, []
    if rsi < 35: oversold += 1; reasons.append("日線 RSI 超賣")
    if weekly_rsi < 42: oversold += 1; reasons.append("周線 RSI 偏低")
    drawdown = (price / high52 - 1) * 100 if high52 > 0 and not np.isnan(high52) else 0
    if drawdown <= -20: oversold += 1; reasons.append(f"從高位回調 {-drawdown:.1f}%")
    bottom_score = oversold * 20
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]) and safe_float(macd.iloc[-1]) < 0:
        bottom_score += 20; reasons.append("MACD 低位金叉/背馳")

    # 止損位邏輯
    tspr_sup = tspr["active_support"]
    stop_loss_val = min(price * 0.95, tspr_sup if not pd.isna(tspr_sup) else price * 0.95)

    return {
        "ticker": ticker, "name": info.get("name", ticker), "price": round(price, 3), "high52": round(high52, 3) if not np.isnan(high52) else price,
        "tspr_status": tspr["status"], "cloud_status": "🟢 三雲多頭" if gann_cloud["cloud_bullish"] else "🔴 三雲空頭",
        "next_turn_date": next_turn_str, "tspr_support": tspr_sup,
        "undervaluation": undervaluation, "valuation_score": round(val_score, 1),
        "bottom_score": bottom_score, "drawdown": round(drawdown, 2),
        "reasons": "｜".join(reasons) if reasons else "未見明顯超賣",
        "blockers": "｜".join(blockers), "stop_loss": round(stop_loss_val, 3)
    }

# ==================== Streamlit UI ====================
elapsed = time.time() - st.session_state.last_refresh
if elapsed >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    clear_errors()
    st.rerun()

with st.sidebar:
    st.markdown("## ⚙️ 系統控制")
    market = st.radio("市場選擇", ["🇭🇰 港股", "🇺🇸 美股"], index=0)
    if quote_ctx: st.success("✅ 富途實時行情已連接")
    else: st.info("ℹ️ 使用 Yahoo 備援報價")
    if st.button("🔄 立即刷新快取"):
        st.cache_data.clear(); st.rerun()

macro_data = fetch_macro()
regime = classify_market_state(market, macro_data)
benchmark_df = fetch_ohlcv(regime["benchmark"], period="1y")

st.markdown("# 🎯 終極撈底系統 V5.0")
st.caption(f"極簡化版｜專為價值投資者與右側摸底設計｜頁面生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HKT")

# 核心 3 個 Tabs
tabs = st.tabs(["🌍 大市指南針 (Dashboard)", "🎯 撈底黃金雷達 (Screener)", "📈 深度技術圖表 (Charts)"])

# ==================== Tab 1: 大市指南針 ====================
with tabs[0]:
    st.subheader(f"大市氣氛溫度計：{regime['name']}")
    st.info(f"💡 **當前資金對策：** {regime['strategy']}")
    
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("VIX", "😱 美股 VIX"), ("VHSI", "🇭🇰 港股 VHSI"), ("SPX", "🇺🇸 S&P 500"), ("HSI", "🇭🇰 恒生指數")]):
        item = macro_data.get(key, {})
        col.metric(label, f"{safe_float(item.get('val'), 0):.2f}", f"{safe_float(item.get('chg'), 0):+.2f}%")
        
    st.markdown("### 📅 大市(基準)下個江恩轉勢日預測")
    turn_res = predict_gann_turn_dates(benchmark_df, forward_trading_days=30)
    top_turns = turn_res.get("top_turns", [])
    
    if top_turns:
        high_prob = [t for t in top_turns if t["共振強度分"] >= 6.5]
        med_prob = [t for t in top_turns if 3.5 <= t["共振強度分"] < 6.5]
        c1, c2 = st.columns(2)
        with c1:
            st.error("🔥 高機率轉勢日 (極易變盤)")
            st.dataframe(pd.DataFrame(high_prob)[["預測轉勢日期", "星期", "時間周期依據"]] if high_prob else pd.DataFrame([{"狀態": "近期無高機率變盤"}]), hide_index=True)
        with c2:
            st.warning("⚡ 中度機率轉勢日 (需配合技術支持/阻力)")
            st.dataframe(pd.DataFrame(med_prob)[["預測轉勢日期", "星期", "時間周期依據"]] if med_prob else pd.DataFrame([{"狀態": "近期無中度變盤"}]), hide_index=True)

# ==================== Tab 2: 撈底黃金雷達 (含黃金回調階梯) ====================
with tabs[1]:
    st.subheader("🎯 撈底黃金雷達 (基本面平價 + 技術面見底)")
    st.caption("自動過濾：估值分 >= 60 (中度/重度低估) 且 見底分 >= 35 (超賣/背馳)，並自動計算下方回調目標。")
    
    tickers = HK_WATCHLIST if market == "🇭🇰 港股" else US_WATCHLIST
    
    if st.button("🚀 啟動全市場撈底掃描", type="primary"):
        progress = st.progress(0, text="下載資料中...")
        info_map, data_map = {}, fetch_multiple(tickers, period="1y")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            for fut in as_completed({executor.submit(get_full_stock_info, tk): tk for tk in tickers}):
                info_map[fut.result()["ticker"]] = fut.result()
                
        progress.progress(50, text="運算江恩與估值模型...")
        results = []
        for i, tk in enumerate(tickers):
            res = decision_engine(tk, data_map.get(tk), info_map.get(tk, {}))
            if res: results.append(res)
            progress.progress(50 + int((i+1)/len(tickers)*50))
        progress.empty()
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            bottom_fish_df = df_results[
                (df_results["valuation_score"] >= 60) & 
                (df_results["bottom_score"] >= 35) & 
                (~df_results["blockers"].str.contains("價值陷阱", na=False))
            ].copy()
            
            if bottom_fish_df.empty:
                st.warning("目前沒有符合【平價 + 極度超賣】雙重標準的股票。請耐心等待大市回調，切勿夾硬買入。")
            else:
                st.success(f"發現 {len(bottom_fish_df)} 隻潛在撈底名單！")
                bottom_fish_df = bottom_fish_df.sort_values("bottom_score", ascending=False)
                
                for _, row in bottom_fish_df.iterrows():
                    drawdown = row['drawdown'] 
                    high52 = row['high52']
                    
                    next_milestone_pct = math.floor(drawdown / 10) * 10
                    if next_milestone_pct > -10: next_milestone_pct = -10
                    next_price = high52 * (1 + next_milestone_pct/100)
                    
                    ladder_html = ""
                    for pct in range(10, 100, 10):
                        price_at_pct = high52 * (1 - pct/100)
                        if -pct == next_milestone_pct:
                            ladder_html += f"<span style='color:#f85149; font-weight:bold; margin-right:14px;'>▼{pct}%: {price_at_pct:.2f}</span>"
                        else:
                            ladder_html += f"<span style='color:#8b949e; margin-right:14px;'>▼{pct}%: {price_at_pct:.2f}</span>"
                            
                        if pct == 40 or pct == 80: 
                            ladder_html += "<br>"

                    st.markdown(
                        f"<div class='good-card'>"
                        f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 現價: <b>{row['price']}</b> ｜ 狀態: <b>{row['tspr_status']}</b><br>"
                        f"<b>為何選中：</b> 估值 {row['undervaluation']} 且 技術面顯示見底訊號 ({row['reasons']})<br>"
                        f"<b>行動建議：</b> 配合江恩轉勢日 <b>{row['next_turn_date']}</b>，企穩 TSPR 10日支持（{row['tspr_support']}）可建首注。破位止損：<b>{row['stop_loss']}</b><br>"
                        f"<div class='ladder-box'>"
                        f"<b>📉 從 52周高位({high52}) 目前已回調：<span style='color:#f85149;'>{drawdown}%</span></b><br>"
                        f"<b>🎯 下一個 10% 跌幅目標接貨價：<span style='color:#f85149; font-size:16px;'>{next_price:.2f}</span></b><br><br>"
                        f"📊 <b>完整跌幅回調階梯：</b><br>{ladder_html}"
                        f"</div>"
                        f"</div>", unsafe_allow_html=True
                    )

# ==================== Tab 3: 深度技術圖表 (TSPR + 江恩) ====================
with tabs[2]:
    st.subheader("📈 深度技術圖表｜右側確認專用")
    st.caption("在撈底雷達找到心水股票後，來這裡確認是否企穩 TSPR 支持或江恩扇形線，並觀察 MACD 資金流向。")
    
    col_q1, col_q2 = st.columns([1, 4])
    with col_q1: default_tk = st.selectbox("快捷選擇", ["0700.HK", "2800.HK", "NVDA", "AAPL"])
    with col_q2: ticker = st.text_input("輸入股票代碼以查看圖表", default_tk, key="chart_tk").upper()
    
    if ticker:
        chart_df = fetch_ohlcv(ticker, period="1y")
        if chart_df is not None and len(chart_df) > 30:
            close = chart_df["close"].iloc[:, 0] if isinstance(chart_df["close"], pd.DataFrame) else chart_df["close"]
            gc_res = calc_gann_time_and_cloud_system(chart_df)
            tspr_res = calc_tspr_system(chart_df)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=close, name="K線"), row=1, col=1)
            bull_upper = np.where(gc_res["ema_fast_1"] >= gc_res["ema_slow_1"], gc_res["ema_fast_1"], np.nan)
            fig.add_trace(go.Scatter(x=chart_df.index, y=bull_upper, fill="tonexty", fillcolor="rgba(38, 166, 154, 0.2)", line=dict(width=0), name="多頭雲"), row=1, col=1)
            
            main_fan = [fl for fl in gc_res["fan_lines"] if fl["label"] == "1x1"][0]
            fig.add_trace(go.Scatter(x=main_fan["dates"], y=main_fan["values"], line=dict(color="#80deea", width=2), name="Gann 1x1 主線"), row=1, col=1)
            
            if not pd.isna(tspr_res["active_support"]):
                fig.add_hline(y=tspr_res["active_support"], line_dash="dash", line_color="#3fb950" if not tspr_res["is_broken"] else "#ef5350", annotation_text="TSPR 10日支持", row=1, col=1)

            vol = chart_df["volume"].iloc[:, 0] if isinstance(chart_df["volume"], pd.DataFrame) else chart_df["volume"]
            op = chart_df["open"].iloc[:, 0] if isinstance(chart_df["open"], pd.DataFrame) else chart_df["open"]
            vol_colors = ["#e040fb" if dt in gc_res["vqc_dates"] else ("#3fb950" if c>=o else "#f85149") for dt, c, o in zip(chart_df.index, close, op)]
            fig.add_trace(go.Bar(x=chart_df.index, y=vol, marker_color=vol_colors, name="成交量 (紫=爆量)"), row=2, col=1)
            
            macd, sig, hist = calc_macd(close)
            fig.add_trace(go.Bar(x=chart_df.index, y=hist, marker_color=[C_GREEN if x>=0 else C_RED for x in hist.fillna(0)], name="MACD Hist"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=macd, line=dict(color=C_BLUE), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=sig, line=dict(color=C_ORANGE), name="Signal"), row=3, col=1)
            
            fig.update_layout(height=800, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
