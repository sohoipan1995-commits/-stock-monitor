import os
import time
import warnings
import threading
import sqlite3
import math
import random
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
    from futu import OpenQuoteContext, RET_OK, KLType, AuType
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False

st.set_page_config(page_title="📈 終極撈底系統 V9.0 (精準防串線報價 + 智能倉位)", page_icon="🎯", layout="wide")

# ==================== 系統常數與配色 ====================
C_RED, C_GREEN, C_ORANGE = "#f85149", "#3fb950", "#d29922"
C_BLUE, C_PURPLE, C_GREY, C_BG = "#58a6ff", "#bc8cff", "#8b949e", "#0d1117"
AUTO_REFRESH_SEC, OHLCV_TTL, INFO_TTL = 1800, 1800, 3600
MAX_WORKERS_INFO = 4
FUTU_LOCK = threading.RLock()
YF_LOCK = threading.RLock()  # 💡 核心修復：防止 yfinance 多線程報價互相覆蓋的專屬鎖

# ==================== 專屬名單 (嚴格保留, 絕不刪減) ====================
HK_PINNED = ["9988.HK", "0700.HK", "0124.HK", "3690.HK", "1810.HK", "0388.HK"]
US_PINNED = ["NFLX", "NVDA", "GOOGL", "MSFT", "META", "MU", "TSM", "SPCX", "AVGO", 
             "TSLA", "AAPL", "AMD", "AMZN", "ORCL", "PLTR", "UNH", "PDD"]

HK_OTHERS = ["0005.HK", "0939.HK", "1398.HK", "3988.HK", "0066.HK", "0003.HK", 
             "0002.HK", "0016.HK", "0883.HK", "2318.HK", "1299.HK", "0001.HK", 
             "0175.HK", "0027.HK", "2628.HK", "0011.HK", "0688.HK", "9618.HK", 
             "0981.HK", "9999.HK", "2382.HK", "0291.HK", "1211.HK", "0267.HK", 
             "2688.HK", "0762.HK", "3033.HK", "2800.HK"]
HK_WATCHLIST = HK_PINNED + [x for x in HK_OTHERS if x not in HK_PINNED]

US_OTHERS = ["ASML", "QCOM", "INTC", "AMAT", "LRCX", "JPM", "BAC", "GS", "MS", 
             "BRK-B", "COST", "WMT", "HD", "JNJ", "PFE", "XOM", "NEE", "UBER", 
             "LITE", "CLX", "CRM", "MSTR", "LLY", "PYPL"]
US_WATCHLIST = US_PINNED + [x for x in US_OTHERS if x not in US_PINNED]

MACRO_TICKERS = {
    "VIX": "^VIX", "VVIX": "^VVIX", "SPX": "^GSPC", "HSI": "^HSI",
    "NDX": "^NDX", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "VHSI": "^VHSI"
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
.ladder-box {background:#21262d; border:1px solid #30363d; border-radius:6px; padding:12px; margin-top:10px; font-family: monospace; font-size: 14px; line-height: 1.8;}
.wave-box {background:#1a202c; border:1px solid #4a148c; border-radius:8px; padding:16px; margin-top:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
.extended-wave {border-color: #d29922; background: #2b2210;}
.pos-box {background:rgba(88,166,255,0.1); border:1px dashed #58a6ff; border-radius:6px; padding:10px; margin-top:10px; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state: st.session_state.last_refresh = time.time()
if "error_log" not in st.session_state: st.session_state.error_log = []

def safe_float(value, default=np.nan):
    try: return default if value is None or pd.isna(value) else float(value)
    except Exception: return default

def clear_errors(): st.session_state.error_log = []

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
    except Exception: return None
quote_ctx = init_futu()

# ==================== 數據獲取模組 (徹底修復報價串線問題) ====================
def requested_start(period):
    mapping = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 800}
    return (datetime.now() - timedelta(days=mapping.get(period, 400))).strftime("%Y-%m-%d")

def _clean_ohlcv_df(df, source_name="Yahoo"):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    needed = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in needed):
        return None
    df = df[needed].sort_index().dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    if df.empty: return None
    df.attrs["source"] = source_name
    return df

@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y"):
    # 1. 優先嘗試富途前復權日線
    if quote_ctx and not ticker.startswith("^") and "=" not in ticker:
        try:
            start = requested_start(period)
            with FUTU_LOCK:
                ret, data, _ = quote_ctx.request_history_kline(
                    to_futu(ticker), start=start, end=None, 
                    ktype=KLType.K_DAY, max_count=1000, 
                    autype=AuType.QFQ if 'AuType' in globals() else None
                )
            if ret == RET_OK and data is not None and not data.empty:
                df = data[["time_key", "open", "high", "low", "close", "volume"]].rename(columns={"time_key": "date"})
                df["date"] = pd.to_datetime(df["date"])
                cleaned = _clean_ohlcv_df(df.set_index("date"), "Futu")
                if cleaned is not None: return cleaned
        except Exception: pass

    # 2. 使用 Yahoo 獨立 Ticker.history() + 線程鎖，徹底杜絕多線程錯價串線
    try:
        with YF_LOCK:
            time.sleep(0.05)
            hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        return _clean_ohlcv_df(hist, "Yahoo")
    except Exception:
        return None

@st.cache_data(ttl=OHLCV_TTL)
def fetch_batch_ohlcv(tickers_tuple, period="1y"):
    """
    💡 核心修復：使用單次批量分組下載 (group_by='ticker')，
    一次過安全獲取所有股票報價，100% 避免不同股票價格互相覆蓋！
    """
    tickers = list(tickers_tuple)
    results = {}
    remaining = []

    # 1. 先檢查富途連接
    for tk in tickers:
        if quote_ctx and not tk.startswith("^") and "=" not in tk:
            df_f = fetch_ohlcv(tk, period)
            if df_f is not None and df_f.attrs.get("source") == "Futu":
                results[tk] = df_f
                continue
        remaining.append(tk)

    if not remaining:
        return results

    # 2. 對剩餘股票執行單次批量安全下載
    try:
        with YF_LOCK:
            raw = yf.download(
                remaining, period=period, interval="1d", 
                group_by="ticker", auto_adjust=True, progress=False, threads=True
            )
        if raw is not None and not raw.empty:
            if len(remaining) == 1:
                tk = remaining[0]
                results[tk] = _clean_ohlcv_df(raw, "Yahoo")
            else:
                for tk in remaining:
                    if tk in raw.columns.get_level_values(0):
                        sub_df = raw[tk].copy()
                        results[tk] = _clean_ohlcv_df(sub_df, "Yahoo")
    except Exception:
        pass

    # 3. 針對批量中漏掉的個別股票進行安全單獨補抓
    for tk in remaining:
        if results.get(tk) is None:
            results[tk] = fetch_ohlcv(tk, period)

    return results

@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    result = {"ticker": ticker, "name": ticker, "pe": None, "pb": None, "roe": None}
    try:
        with YF_LOCK:
            info = yf.Ticker(ticker).info or {}
        result["name"] = info.get("shortName") or info.get("longName") or ticker
        result["pe"] = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        result["pb"] = safe_float(info.get("priceToBook"))
        result["roe"] = safe_float(info.get("returnOnEquity"))
    except Exception: pass
    return result

# ==================== 核心指標與引擎庫 (加入收市價防錯價保護) ====================
def extract_col(df, col_name):
    return df[col_name].iloc[:, 0] if isinstance(df[col_name], pd.DataFrame) else df[col_name]

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def calc_macd(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_obv(df):
    close, vol = extract_col(df, "close"), extract_col(df, "volume")
    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    return obv, obv.rolling(10).mean()

def calc_cmf(df, period=20):
    high, low, close, vol = [extract_col(df, c) for c in ["high", "low", "close", "volume"]]
    mf_vol = (((close - low) - (high - close)) / (high - low).replace(0, np.nan)) * vol
    return mf_vol.rolling(period).sum() / vol.rolling(period).sum()

def get_52w_high(df):
    """
    💡 精準 52 周高位計算：結合收市價與合理盤中高位，杜絕錯價針
    """
    if df is None or df.empty: return np.nan
    highs = np.asarray(extract_col(df, "high"), dtype=float)
    closes = np.asarray(extract_col(df, "close"), dtype=float)
    
    window_hi = highs[-252:] if len(highs) >= 252 else highs
    window_cl = closes[-252:] if len(closes) >= 252 else closes
    
    if len(window_cl) > 0 and not np.all(np.isnan(window_cl)):
        max_cl = float(np.nanmax(window_cl))
        max_hi = float(np.nanmax(window_hi)) if not np.all(np.isnan(window_hi)) else max_cl
        # 如果最高影線超過最高收市價 12% 以上，視為異常錯價針，採用最高收市價
        if max_hi > max_cl * 1.12:
            return max_cl
        return max_hi
    return np.nan

def calc_tspr_system(df, left_bars=10):
    if df is None or len(df) < 30: return {"active_support": np.nan, "is_broken": False, "status": "資料不足"}
    lows = np.asarray(extract_col(df, "low"), dtype=float)
    closes = np.asarray(extract_col(df, "close"), dtype=float)
    active_support, is_broken = None, False
    for i in range(left_bars + 5, len(df)):
        pivot_idx = i - 5
        window = lows[max(0, pivot_idx - left_bars) : i + 1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            if not np.isnan(lows[pivot_idx]) and lows[pivot_idx] == np.nanmin(window):
                active_support, is_broken = float(lows[pivot_idx]), False
        if active_support is not None and closes[i] < active_support: is_broken = True
    status_str = "🔴 跌穿10日支持" if is_broken else ("🟢 守穩10日支持" if active_support else "⚪ 尋底中")
    return {"active_support": round(active_support, 3) if active_support else np.nan, "is_broken": is_broken, "status": status_str}

def predict_gann_turn_dates(df, forward_trading_days=45):
    if df is None or len(df) < 60: return {"top_turns": []}
    dates = pd.DatetimeIndex(df.index)
    highs, lows = np.asarray(extract_col(df, "high"), dtype=float), np.asarray(extract_col(df, "low"), dtype=float)
    n = len(df)
    pivots, offset = [], max(0, n - 180)
    for i in range(7, n - offset - 7):
        g_idx = offset + i
        w_hi, w_lo = highs[g_idx - 7 : g_idx + 8], lows[g_idx - 7 : g_idx + 8]
        if len(w_hi) > 0 and highs[g_idx] == np.nanmax(w_hi): pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "波段頂部", "price": highs[g_idx]})
        if len(w_lo) > 0 and lows[g_idx] == np.nanmin(w_lo): pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "波段底部", "price": lows[g_idx]})
            
    pivots = sorted({p["idx"]: p for p in pivots}.values(), key=lambda x: x["idx"])
    future_bdays = pd.bdate_range(start=dates[-1] + timedelta(days=1), periods=forward_trading_days)
    
    trading_cycles = {15: 1.5, 21: 2.0, 30: 2.5, 45: 3.5, 60: 3.0, 90: 4.5, 120: 3.5}
    gann_seasonal_dates = [(3, 21, "春分(0°)"), (6, 21, "夏至(90°)"), (9, 22, "秋分(180°)"), (12, 21, "冬至(270°)"), (2, 4, "立春"), (5, 5, "立夏"), (8, 7, "立秋"), (11, 7, "立冬")]
    
    records = []
    for step, f_date in enumerate(future_bdays, start=1):
        future_idx = (n - 1) + step
        score, reasons = 0.0, []
        for pv in pivots[-4:]:
            el_td = future_idx - pv["idx"]
            if el_td in trading_cycles:
                score += trading_cycles[el_td]
                reasons.append(f"基於 {pv['date'].strftime('%Y-%m-%d')} {pv['type']}({pv['price']:.1f}) 疊加 {el_td} 交易日週期")
            el_cd = (f_date - pv["date"]).days
            if any(abs(el_cd - cc) <= 1 for cc in [30, 45, 60, 90]):
                score += 1.5
                reasons.append(f"基於 {pv['date'].strftime('%Y-%m-%d')} {pv['type']} 疊加自然日循環")
                
        for m, d, name in gann_seasonal_dates:
            if abs((f_date - pd.Timestamp(year=f_date.year, month=m, day=d)).days) <= 1:
                score += 3.5; reasons.append(f"觸發江恩節氣：{name}")
                
        if score > 0:
            records.append({
                "預測轉勢日期": f_date.strftime("%Y-%m-%d"),
                "星期": ["一", "二", "三", "四", "五", "六", "日"][f_date.weekday()],
                "共振強度分": round(score, 1),
                "時間周期依據": " ｜ ".join(set(reasons))
            })
    hit_df = pd.DataFrame(records).sort_values(["共振強度分", "預測轉勢日期"], ascending=[False, True]) if records else pd.DataFrame()
    return {"top_turns": hit_df.to_dict("records") if not hit_df.empty else []}

def analyze_wave_cycles(df):
    if df is None or len(df) < 60: return None
    dates = pd.DatetimeIndex(df.index)
    highs, lows = np.asarray(extract_col(df, "high"), dtype=float), np.asarray(extract_col(df, "low"), dtype=float)
    pivots, n, today = [], len(df), datetime.now()
    for i in range(5, n - 5):
        w_hi, w_lo = highs[i-5:i+6], lows[i-5:i+6]
        if len(w_hi) > 0 and highs[i] == np.nanmax(w_hi): pivots.append({"idx": i, "date": dates[i], "type": "頂部", "price": highs[i]})
        if len(w_lo) > 0 and lows[i] == np.nanmin(w_lo): pivots.append({"idx": i, "date": dates[i], "type": "底部", "price": lows[i]})
    
    clean_pivots = []
    for p in pivots:
        if not clean_pivots: clean_pivots.append(p)
        else:
            last_p = clean_pivots[-1]
            if last_p["type"] == p["type"]:
                if p["type"] == "頂部" and p["price"] > last_p["price"]: clean_pivots[-1] = p
                elif p["type"] == "底部" and p["price"] < last_p["price"]: clean_pivots[-1] = p
            else: clean_pivots.append(p)
                
    if len(clean_pivots) < 3: return None
    
    down_days, down_pcts, up_days, up_pcts = [], [], [], []
    for i in range(1, len(clean_pivots)):
        p1, p2 = clean_pivots[i-1], clean_pivots[i]
        days_diff = (p2["date"] - p1["date"]).days
        if p1["type"] == "頂部" and p2["type"] == "底部":
            down_days.append(days_diff); down_pcts.append((p1["price"] - p2["price"]) / p1["price"] * 100)
        elif p1["type"] == "底部" and p2["type"] == "頂部":
            up_days.append(days_diff); up_pcts.append((p2["price"] - p1["price"]) / p1["price"] * 100)
            
    avg_down_days, avg_down_pct = int(np.mean(down_days)) if down_days else 20, np.mean(down_pcts) if down_pcts else 10.0
    avg_up_days, avg_up_pct = int(np.mean(up_days)) if up_days else 25, np.mean(up_pcts) if up_pcts else 15.0
    
    last_p, prev_p = clean_pivots[-1], clean_pivots[-2]
    current_trend = "回調中(跌浪)" if last_p["type"] == "頂部" else "上升中(升浪)"
    is_extended = False
    
    if current_trend == "回調中(跌浪)":
        target_date = last_p["date"] + timedelta(days=avg_down_days)
        fib_0618 = last_p["price"] - (last_p["price"] - prev_p["price"]) * 0.618
        avg_target_price = last_p["price"] * (1 - avg_down_pct / 100)
        if target_date < today:
            is_extended = True
            ext_days = int(avg_down_days * 1.618)
            target_date_ext = last_p["date"] + timedelta(days=ext_days)
            if target_date_ext < today: target_date_ext = today + timedelta(days=3)
            desc = (f"⚠️ **延長跌浪警告**：原定尋底窗 ({target_date.strftime('%Y-%m-%d')}) 已過失效。<br>"
                    f"市場進入 **1.618 斐波那契時間擴展**，下一個極限轉勢窗為 <b style='color:#d29922;'>{target_date_ext.strftime('%Y-%m-%d')}</b>。<br>"
                    f"🎯 **極限支持區**：平均跌幅位 **{avg_target_price:.2f}** ｜ 黃金分割 0.618 **{fib_0618:.2f}**")
        else:
            desc = (f"🔍 **正常波浪推演**：預計尋底目標日期 <b style='color:#3fb950;'>{target_date.strftime('%Y-%m-%d')}</b>。<br>"
                    f"🎯 **預期支持區**：平均跌幅位 **{avg_target_price:.2f}** ｜ 黃金分割 0.618 **{fib_0618:.2f}**")
    else:
        target_date = last_p["date"] + timedelta(days=avg_up_days)
        fib_1618 = last_p["price"] + (last_p["price"] - prev_p["price"]) * 1.618
        avg_target_price = last_p["price"] * (1 + avg_up_pct / 100)
        if target_date < today:
            is_extended, ext_days = True, int(avg_up_days * 1.618)
            target_date_ext = last_p["date"] + timedelta(days=ext_days)
            if target_date_ext < today: target_date_ext = today + timedelta(days=3)
            desc = (f"🔥 **延伸升浪警告**：原定見頂窗 ({target_date.strftime('%Y-%m-%d')}) 已過，進入 **1.618 時間擴展**。<br>"
                    f"下一個極限見頂窗為 <b style='color:#f85149;'>{target_date_ext.strftime('%Y-%m-%d')}</b>。<br>"
                    f"🎯 **極限阻力區**：平均升幅位 **{avg_target_price:.2f}** ｜ 黃金擴展 1.618 **{fib_1618:.2f}**")
        else:
            desc = (f"🔍 **正常波浪推演**：預計見頂目標日期 <b style='color:#f85149;'>{target_date.strftime('%Y-%m-%d')}</b>。<br>"
                    f"🎯 **預期阻力區**：平均升幅位 **{avg_target_price:.2f}** ｜ 黃金擴展 1.618 **{fib_1618:.2f}**")
            
    return {
        "avg_down_days": avg_down_days, "avg_down_pct": round(avg_down_pct, 1),
        "avg_up_days": avg_up_days, "avg_up_pct": round(avg_up_pct, 1),
        "current_trend": current_trend, "is_extended": is_extended, "prediction_desc": desc
    }

def generate_drawdown_ladder_html(high52, current_price):
    if np.isnan(high52) or high52 <= 0: return ""
    drawdown_pct = (current_price / high52 - 1) * 100
    next_milestone_pct = math.floor(drawdown_pct / 10) * 10
    if next_milestone_pct > -10: next_milestone_pct = -10
    
    ladder_html = "<div class='ladder-box'>"
    ladder_html += f"<b>📐 全景回調接貨階梯 (52周高位: {high52:.2f} ｜ 現價: {current_price:.2f})：</b><br>"
    for pct in range(10, 100, 10):
        price_at_pct = high52 * (1 - pct/100)
        if -pct == next_milestone_pct:
            ladder_html += f"<span style='color:#f85149; font-weight:bold; font-size:15px; margin-right:14px; border-bottom: 2px solid #f85149;'>▼{pct}%({price_at_pct:.2f})</span> "
        else:
            ladder_html += f"<span style='color:#8b949e; margin-right:14px;'>▼{pct}%({price_at_pct:.2f})</span> "
        if pct == 40 or pct == 80: ladder_html += "<br>"
    ladder_html += "</div>"
    return ladder_html

def generate_drawdown_ladder_text(high52, current_price):
    if np.isnan(high52) or high52 <= 0: return ""
    drawdown_pct = (current_price / high52 - 1) * 100
    next_milestone_pct = math.floor(drawdown_pct / 10) * 10
    if next_milestone_pct > -10: next_milestone_pct = -10
    parts = []
    for pct in range(10, 100, 10):
        price_at_pct = high52 * (1 - pct/100)
        if -pct == next_milestone_pct: parts.append(f"🎯▼{pct}%({price_at_pct:.2f})")
        else: parts.append(f"▼{pct}%({price_at_pct:.2f})")
    return " | ".join(parts)

# ==================== 決策與見底概率引擎 ====================
def decision_engine(ticker, df, info, benchmark_df):
    if df is None or len(df) < 60: return None
    close = extract_col(df, "close")
    price = safe_float(close.iloc[-1])
    high52 = get_52w_high(df)
    last_bar = str(pd.Timestamp(df.index[-1]).date())
    source = df.attrs.get("source", "Yahoo")
    
    rsi = safe_float(calc_rsi(close).iloc[-1], 50)
    macd, macd_signal, _ = calc_macd(close)
    obv_s, obv_ma = calc_obv(df)
    cmf = safe_float(calc_cmf(df).iloc[-1], 0)
    
    df_weekly = df.resample('W').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    w_rsi = safe_float(calc_rsi(df_weekly['close']).iloc[-1], 50) if len(df_weekly) > 14 else 50
    
    if benchmark_df is not None and len(benchmark_df) >= 60 and len(df) >= 60:
        bench_close = extract_col(benchmark_df, "close")
        stock_ret = close.iloc[-1] / close.iloc[-60] - 1
        bench_ret = bench_close.iloc[-1] / bench_close.iloc[-60] - 1
        rs_60 = (stock_ret - bench_ret) * 100
    else:
        rs_60 = 0.0

    obv_status = "流入" if safe_float(obv_s.iloc[-1]) > safe_float(obv_ma.iloc[-1]) else "流出"
    drawdown = (price / high52 - 1) * 100 if high52 > 0 and not np.isnan(high52) else 0
    
    tspr = calc_tspr_system(df)
    turn_info = predict_gann_turn_dates(df, 30)
    nt = turn_info["top_turns"][0] if turn_info["top_turns"] else None
    next_turn_str = f"{nt['預測轉勢日期']} ({nt['共振強度分']}分)" if nt else "—"

    # 見底百分比評分系統
    bottom_prob = 0
    if drawdown <= -40: bottom_prob += 25
    elif drawdown <= -25: bottom_prob += 15
    elif drawdown <= -15: bottom_prob += 10
    
    if rsi < 30: bottom_prob += 15
    elif rsi < 40: bottom_prob += 10
    
    if w_rsi < 35: bottom_prob += 15
    elif w_rsi < 45: bottom_prob += 5
    
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]) and safe_float(macd.iloc[-1]) < 0:
        bottom_prob += 15
        
    if cmf > 0.05: bottom_prob += 10
    if obv_status == "流入": bottom_prob += 10
    if rs_60 > 10: bottom_prob += 10
    
    if not tspr["is_broken"] and not pd.isna(tspr["active_support"]):
        dist_to_supp = (price / tspr["active_support"] - 1) * 100
        if 0 <= dist_to_supp <= 5: bottom_prob += 10 
        
    if drawdown <= -40 and cmf < -0.15: bottom_prob -= 30 
    if rs_60 < -20: bottom_prob -= 15 
        
    bottom_prob = max(0, min(99, bottom_prob)) 
    
    pe = safe_float(info.get("pe"))
    pe_str = f"{pe:.1f}" if pe > 0 else "N/A"
    
    flow_tags = []
    if obv_status == "流入": flow_tags.append("OBV增籌")
    if cmf > 0.1: flow_tags.append(f"CMF({cmf:.2f})流入")
    elif cmf < -0.1: flow_tags.append(f"CMF({cmf:.2f})流出")
    
    flow_str = " ｜ ".join(flow_tags) if flow_tags else "資金平淡"
    rs_mtf_str = f"周RSI:{w_rsi:.0f} ｜ 60日大市相對強弱:{rs_60:+.1f}%"
    
    tspr_sup = tspr["active_support"]
    stop_loss_val = min(price * 0.95, tspr_sup if not pd.isna(tspr_sup) else price * 0.95)

    return {
        "ticker": ticker, "name": info.get("name", ticker), 
        "price": round(price, 2), "high52": round(high52, 2),
        "last_bar": last_bar, "source": source,
        "pe": pe_str, "bottom_prob": bottom_prob, 
        "flow_detail": flow_str, "rs_mtf": rs_mtf_str, 
        "tspr_status": tspr["status"], "tspr_support": tspr_sup,
        "next_turn_date": next_turn_str, "drawdown": round(drawdown, 2),
        "stop_loss": round(stop_loss_val, 2), 
        "ladder_html": generate_drawdown_ladder_html(high52, price), 
        "ladder_text": generate_drawdown_ladder_text(high52, price)
    }

# ==================== Streamlit UI 主結構 ====================
if time.time() - st.session_state.last_refresh >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time(); st.cache_data.clear(); clear_errors(); st.rerun()

with st.sidebar:
    st.markdown("## ⚙️ 系統控制")
    market = st.radio("市場選擇", ["🇺🇸 核心與美股", "🇭🇰 港股"], index=0)
    if quote_ctx: st.success("✅ 富途實時行情已連接")
    else: st.info("ℹ️ 使用 Yahoo 批量安全報價")
    if st.button("🔄 立即刷新快取 (清除舊報價)", type="primary"):
        st.cache_data.clear()
        if "latest_results" in st.session_state:
            del st.session_state["latest_results"]
        st.rerun()

idx_tk = "^GSPC" if market == "🇺🇸 核心與美股" else "^HSI"
benchmark_df = fetch_ohlcv(idx_tk, period="1y")

st.markdown("# 🎯 終極撈底系統 V9.0")
st.caption(f"防串線精準報價引擎 + 智能倉位 + 多時空共振｜頁面生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HKT")

tabs = st.tabs([
    "🌍 大市深度分析 (三大指數推演)", 
    "📊 關注名單總排行 (置頂+階梯)", 
    "🎯 撈底黃金雷達 (含智能倉位)", 
    "📈 深度技術圖表 (時空視覺化)"
])

# ==================== Tab 1: 大市深度分析 ====================
with tabs[0]:
    st.subheader("🌍 宏觀指數深度拆解 (恒指 / 標普 / 納指)")
    idx_map = {"恒生指數 (HSI)": "^HSI", "標普500 (SPX)": "^GSPC", "納斯達克 (NDX)": "^NDX"}
    cols = st.columns(3)
    
    for col, (idx_name, idx_tk) in zip(cols, idx_map.items()):
        with col:
            st.markdown(f"### {idx_name}")
            df_idx = fetch_ohlcv(idx_tk, period="1y")
            if df_idx is not None:
                close = extract_col(df_idx, "close")
                ma200 = close.rolling(200).mean().iloc[-1]
                rsi_val = calc_rsi(close).iloc[-1]
                if close.iloc[-1] < ma200 and rsi_val < 35: state = "🥶 冰點 (極度恐慌)"
                elif close.iloc[-1] < ma200: state = "🧊 冷卻 (弱勢反覆)"
                elif rsi_val > 70: state = "🔥 過熱 (動能耗竭警戒)"
                else: state = "☀️ 溫和 (健康趨勢)"
                
                st.info(f"**現價：** {close.iloc[-1]:.2f} ｜ **狀態：** {state}")
                
                turn_res = predict_gann_turn_dates(df_idx, 30)
                if turn_res.get("top_turns"):
                    t = turn_res["top_turns"][0]
                    st.markdown(f"**🔥 下個高機率變盤日：** {t['預測轉勢日期']}")
                    st.markdown("<ul style='font-size:13px; color:#8b949e;'>" + "".join([f"<li>{r}</li>" for r in t['時間周期依據'].split(' ｜ ')]) + "</ul>", unsafe_allow_html=True)
                
                wave_res = analyze_wave_cycles(df_idx)
                if wave_res:
                    box_class = "wave-box extended-wave" if wave_res["is_extended"] else "wave-box"
                    st.markdown(
                        f"<div class='{box_class}'>"
                        f"<b>🌊 歷史波浪均值統計：</b><br>"
                        f"平均跌浪：<b>{wave_res['avg_down_days']}日</b> (跌幅 <b>{wave_res['avg_down_pct']}%</b>)<br>"
                        f"平均升浪：<b>{wave_res['avg_up_days']}日</b> (升幅 <b>{wave_res['avg_up_pct']}%</b>)<br>"
                        f"<hr style='margin:8px 0; border-color:#30363d;'>"
                        f"{wave_res['prediction_desc']}"
                        f"</div>", unsafe_allow_html=True
                    )
            else:
                st.error("無法取得指數數據。")

# ==================== Tab 2: 關注名單總排行 ====================
with tabs[1]:
    st.subheader("📊 關注名單：技術評分總排行")
    st.caption("已啟用防串線批量校準報價，直接顯示最新報價日期與 52 周精準高位。")
    
    tickers = US_WATCHLIST if market == "🇺🇸 核心與美股" else HK_WATCHLIST
    active_pinned = US_PINNED if market == "🇺🇸 核心與美股" else HK_PINNED
    
    if st.button("🚀 載入並分析全名單", type="primary", key="rank_btn"):
        progress = st.progress(0, text="正在安全批量校準全市場報價...")
        # 💡 使用全新的防串線批量下載
        data_map = fetch_batch_ohlcv(tuple(tickers), period="1y")
        progress.progress(45, text="下載基本面估值資料...")
        
        info_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_INFO) as executor:
            for fut in as_completed({executor.submit(get_full_stock_info, tk): tk for tk in tickers}):
                res_info = fut.result()
                info_map[res_info["ticker"]] = res_info
                
        progress.progress(75, text="運算見底概率與江恩模型...")
        results = []
        for tk in tickers:
            res = decision_engine(tk, data_map.get(tk), info_map.get(tk, {}), benchmark_df)
            if res: results.append(res)
        progress.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results["置頂"] = df_results["ticker"].apply(lambda x: "📌" if x in active_pinned else "")
            df_results = df_results.sort_values(["置頂", "bottom_prob"], ascending=[False, False])
            st.session_state["latest_results"] = df_results
            
            display_df = df_results[[
                "置頂", "ticker", "name", "price", "high52", "drawdown", "bottom_prob", "ladder_text", 
                "pe", "flow_detail", "rs_mtf", "tspr_status", "next_turn_date", "stop_loss", "last_bar", "source"
            ]].rename(columns={
                "ticker": "代碼", "name": "名稱", "price": "現價", "high52": "52周高位", "drawdown": "回調%",
                "bottom_prob": "見底概率%", "ladder_text": "🎯 全景回調階梯", 
                "pe": "PE", "flow_detail": "資金流", "rs_mtf": "動能與強弱", "tspr_status": "TSPR 狀態", 
                "next_turn_date": "下個轉勢日", "stop_loss": "止損位", "last_bar": "報價日期", "source": "來源"
            })
            
            st.success("報價校準與排名完成！你的專屬名單已永久置頂。")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 📌 置頂專屬名單：深度回調階梯與見底分析")
            pinned_results = df_results[df_results["置頂"] == "📌"]
            for _, row in pinned_results.iterrows():
                prob_color = "#3fb950" if row['bottom_prob'] >= 60 else ("#d29922" if row['bottom_prob'] >= 30 else "#f85149")
                st.markdown(
                    f"<div class='decision-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 現價: <b>{row['price']}</b> (52周高: {row['high52']} / {row['drawdown']}%) ｜ 見底概率: <b style='color:{prob_color}; font-size:16px;'>{row['bottom_prob']}%</b> ｜ PE: <b>{row['pe']}</b> <span style='color:#8b949e; font-size:12px;'>[{row['source']} {row['last_bar']}]</span><br>"
                    f"<b>資金動向：</b> {row['flow_detail']} ｜ <b>時空動能：</b> {row['rs_mtf']}<br>"
                    f"<b>行動建議：</b> 配合轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。<br>"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 3: 撈底黃金雷達 (含智能倉位計算機) ====================
with tabs[2]:
    st.subheader("🎯 撈底黃金雷達 & 智能倉位計算機")
    st.caption("自動過濾出全市場：見底概率 >= 45% 的超跌優質股，並針對你的資金規模給出嚴格的買入股數建議。")
    
    c1, c2 = st.columns(2)
    with c1: acc_size = st.number_input("組合總資金 (如 100000 USD/HKD)", min_value=1000.0, value=100000.0, step=5000.0)
    with c2: risk_limit = st.slider("單筆交易最大可承受虧損 (%)", 0.5, 3.0, 1.0, 0.1)
    
    max_risk_amount = acc_size * (risk_limit / 100)
    st.info(f"🛡️ 嚴格風控：每筆交易如果觸及止損，你的最大虧損將被控制在 **{max_risk_amount:.0f}** 內。")

    if "latest_results" not in st.session_state:
        st.warning("請先在『Tab 2: 關注名單』點擊載入，雷達掃描名單會自動為你過濾生成。")
    else:
        df_results = st.session_state["latest_results"]
        bottom_fish_df = df_results[df_results["bottom_prob"] >= 45].copy()
        if bottom_fish_df.empty:
            st.warning("目前沒有符合【高見底概率】的股票。切勿夾硬買入。")
        else:
            st.success(f"雷達鎖定 {len(bottom_fish_df)} 隻高潛力撈底目標！")
            for _, row in bottom_fish_df.iterrows():
                risk_per_share = row['price'] - row['stop_loss']
                if risk_per_share > 0:
                    suggested_shares = int(max_risk_amount / risk_per_share)
                    cost = suggested_shares * row['price']
                    pos_html = f"<div class='pos-box'>💰 <b>智能注碼計算</b>：建議買入 <b>{suggested_shares} 股</b> (動用資金: {cost:.0f}) ｜ 止損位: <b>{row['stop_loss']}</b></div>"
                else:
                    pos_html = f"<div class='pos-box' style='border-color:#f85149;'>⚠️ 現價低於或太接近止損位 ({row['stop_loss']})，無法計算安全注碼，請等待企穩。</div>"

                st.markdown(
                    f"<div class='good-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 現價: <b>{row['price']}</b> ｜ 見底概率: <b style='color:#3fb950;'>{row['bottom_prob']}%</b> <span style='color:#8b949e; font-size:12px;'>[{row['source']} {row['last_bar']}]</span><br>"
                    f"<b>資金動向：</b> {row['flow_detail']} ｜ <b>時空動能：</b> {row['rs_mtf']}<br>"
                    f"<b>行動建議：</b> 配合轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。<br>"
                    f"{pos_html}"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 4: 深度技術圖表 ====================
with tabs[3]:
    st.subheader("📈 深度技術圖表｜時空交匯視覺化")
    st.caption("圖表會自動畫出 **TSPR 支持線** (綠虛線)，並以 **垂直黃虛線** 標示未來預測的江恩轉勢日。")
    
    col_q1, col_q2 = st.columns([1, 4])
    with col_q1: default_tk = st.selectbox("快捷選擇", ["0700.HK", "9988.HK", "NVDA", "NFLX"])
    with col_q2: ticker = st.text_input("輸入股票代碼以查看圖表", default_tk, key="chart_tk").upper()
    
    if ticker:
        chart_df = fetch_ohlcv(ticker, period="1y")
        if chart_df is not None and len(chart_df) > 30:
            close = extract_col(chart_df, "close")
            tspr_res = calc_tspr_system(chart_df)
            turn_res = predict_gann_turn_dates(chart_df, 45)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=chart_df.index, open=extract_col(chart_df, "open"), high=extract_col(chart_df, "high"), low=extract_col(chart_df, "low"), close=close, name="K線"), row=1, col=1)
            
            if not pd.isna(tspr_res["active_support"]):
                line_col = "#3fb950" if not tspr_res["is_broken"] else "#ef5350"
                fig.add_hline(y=tspr_res["active_support"], line_dash="dash", line_color=line_col, annotation_text=f"TSPR 支持 ({tspr_res['active_support']})", row=1, col=1)

            if turn_res.get("top_turns"):
                for t in turn_res["top_turns"][:2]:
                    target_dt = pd.to_datetime(t["預測轉勢日期"])
                    fig.add_vline(x=target_dt, line_dash="dot", line_color="#d29922", annotation_text="轉勢窗", annotation_position="top right", row=1, col=1)

            vol = extract_col(chart_df, "volume")
            op = extract_col(chart_df, "open")
            vol_colors = ["#3fb950" if c>=o else "#f85149" for c, o in zip(close, op)]
            fig.add_trace(go.Bar(x=chart_df.index, y=vol, marker_color=vol_colors, name="成交量"), row=2, col=1)
            
            macd, sig, hist = calc_macd(close)
            fig.add_trace(go.Bar(x=chart_df.index, y=hist, marker_color=[C_GREEN if x>=0 else C_RED for x in hist.fillna(0)], name="MACD Hist"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=macd, line=dict(color=C_BLUE), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=sig, line=dict(color=C_ORANGE), name="Signal"), row=3, col=1)
            
            last_date = chart_df.index[-1]
            future_date = last_date + timedelta(days=25)
            fig.update_layout(xaxis=dict(range=[chart_df.index[0], future_date]), height=800, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
