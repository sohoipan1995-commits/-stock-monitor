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

st.set_page_config(page_title="📈 終極撈底系統 V7.0 (波浪推演 + 見底機率)", page_icon="🎯", layout="wide")

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREY = "#8b949e"
C_BG = "#0d1117"

AUTO_REFRESH_SEC = 1800
OHLCV_TTL = 1800
INFO_TTL = 3600
MAX_WORKERS_DATA = 5
FUTU_LOCK = threading.RLock()

# 💡 你的專屬永久置頂名單 (未經同意不刪減)
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
.wave-box {background:#1a202c; border:1px dashed #bc8cff; border-radius:6px; padding:12px; margin-top:8px;}
</style>
""", unsafe_allow_html=True)

if "last_refresh" not in st.session_state: st.session_state.last_refresh = time.time()
if "error_log" not in st.session_state: st.session_state.error_log = []

def safe_float(value, default=np.nan):
    try:
        return default if value is None or pd.isna(value) else float(value)
    except Exception:
        return default

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
    except Exception:
        return None

quote_ctx = init_futu()

def requested_start(period):
    mapping = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 800}
    return (datetime.now() - timedelta(days=mapping.get(period, 400))).strftime("%Y-%m-%d")

@st.cache_data(ttl=OHLCV_TTL)
def fetch_ohlcv(ticker, period="1y"):
    if quote_ctx and not ticker.startswith("^") and "=" not in ticker:
        try:
            start = requested_start(period)
            with FUTU_LOCK:
                ret, data, _ = quote_ctx.request_history_kline(to_futu(ticker), start=start, end=None, ktype=KLType.K_DAY, max_count=1000)
            if ret == RET_OK and data is not None and not data.empty:
                df = data[["time_key", "open", "high", "low", "close", "volume"]].rename(columns={"time_key": "date"})
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index().dropna()
        except Exception: pass
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df.sort_index().dropna(subset=["open", "high", "low", "close", "volume"])
    except Exception: return None

def fetch_multiple(tickers, period="1y"):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
        futures = {executor.submit(fetch_ohlcv, tk, period): tk for tk in tickers}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results

@st.cache_data(ttl=INFO_TTL)
def get_full_stock_info(ticker):
    result = {"ticker": ticker, "name": ticker, "pe": None, "pb": None, "roe": None}
    try:
        info = yf.Ticker(ticker).info or {}
        result["name"] = info.get("shortName") or ticker
        result["pe"] = safe_float(info.get("trailingPE") or info.get("forwardPE"))
        result["pb"] = safe_float(info.get("priceToBook"))
        result["roe"] = safe_float(info.get("returnOnEquity"))
    except Exception: pass
    return result

# ==================== 技術與波浪推演引擎 ====================
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
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    vol = df["volume"].iloc[:, 0] if isinstance(df["volume"], pd.DataFrame) else df["volume"]
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).cumsum()
    return obv, obv.rolling(10).mean()

def calc_mfi(df, period=14):
    high, low, close, vol = [df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c] for c in ["high", "low", "close", "volume"]]
    typical = (high + low + close) / 3
    money = typical * vol
    pos = money.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg = money.where(typical < typical.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))

def calc_cmf(df, period=20):
    high, low, close, vol = [df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c] for c in ["high", "low", "close", "volume"]]
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_multiplier * vol
    return mf_volume.rolling(period).sum() / vol.rolling(period).sum()

def get_52w_high(df):
    highs = np.asarray(df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"], dtype=float)
    window = highs[-252:] if len(highs) >= 252 else highs
    return float(np.nanmax(window)) if len(window) > 0 and not np.all(np.isnan(window)) else np.nan

# 💡 新增：波浪回調日數與幅度推演引擎
def analyze_wave_cycles(df):
    if df is None or len(df) < 60: return None
    dates = pd.DatetimeIndex(df.index)
    highs = np.asarray(df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"], dtype=float)
    lows = np.asarray(df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"], dtype=float)
    n = len(df)
    
    # 抓取 Pivot
    pivots = []
    for i in range(5, n - 5):
        w_hi, w_lo = highs[i-5:i+6], lows[i-5:i+6]
        if len(w_hi) > 0 and highs[i] == np.nanmax(w_hi): pivots.append({"idx": i, "date": dates[i], "type": "頂部", "price": highs[i]})
        if len(w_lo) > 0 and lows[i] == np.nanmin(w_lo): pivots.append({"idx": i, "date": dates[i], "type": "底部", "price": lows[i]})
    
    # 清理連續相同類型的 Pivot (只留極值)
    clean_pivots = []
    for p in pivots:
        if not clean_pivots: clean_pivots.append(p)
        else:
            last_p = clean_pivots[-1]
            if last_p["type"] == p["type"]:
                if p["type"] == "頂部" and p["price"] > last_p["price"]: clean_pivots[-1] = p
                elif p["type"] == "底部" and p["price"] < last_p["price"]: clean_pivots[-1] = p
            else:
                clean_pivots.append(p)
                
    if len(clean_pivots) < 2: return None
    
    down_days, down_pcts, up_days, up_pcts = [], [], [], []
    for i in range(1, len(clean_pivots)):
        p1, p2 = clean_pivots[i-1], clean_pivots[i]
        days_diff = (p2["date"] - p1["date"]).days
        if p1["type"] == "頂部" and p2["type"] == "底部":
            down_days.append(days_diff)
            down_pcts.append((p1["price"] - p2["price"]) / p1["price"] * 100)
        elif p1["type"] == "底部" and p2["type"] == "頂部":
            up_days.append(days_diff)
            up_pcts.append((p2["price"] - p1["price"]) / p1["price"] * 100)
            
    avg_down_days = int(np.mean(down_days)) if down_days else 20
    avg_down_pct = np.mean(down_pcts) if down_pcts else 10.0
    avg_up_days = int(np.mean(up_days)) if up_days else 25
    avg_up_pct = np.mean(up_pcts) if up_pcts else 15.0
    
    last_pivot = clean_pivots[-1]
    last_price = safe_float((df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]).iloc[-1])
    current_trend = "回調中(跌浪)" if last_pivot["type"] == "頂部" else "上升中(升浪)"
    
    if current_trend == "回調中(跌浪)":
        target_date = last_pivot["date"] + timedelta(days=avg_down_days)
        target_price = last_pivot["price"] * (1 - avg_down_pct / 100)
        desc = f"預計尋底目標日期：**{target_date.strftime('%Y-%m-%d')}**，預計底部支持價：**{target_price:.2f}**"
    else:
        target_date = last_pivot["date"] + timedelta(days=avg_up_days)
        target_price = last_pivot["price"] * (1 + avg_up_pct / 100)
        desc = f"預計見頂目標日期：**{target_date.strftime('%Y-%m-%d')}**，預計頂部阻力價：**{target_price:.2f}**"
        
    return {
        "avg_down_days": avg_down_days, "avg_down_pct": round(avg_down_pct, 1),
        "avg_up_days": avg_up_days, "avg_up_pct": round(avg_up_pct, 1),
        "current_trend": current_trend, "last_pivot": last_pivot,
        "prediction_desc": desc
    }

# 💡 升級版：深度理據化江恩預測 (明確列出基準點)
def predict_gann_turn_dates(df, forward_trading_days=45):
    if df is None or len(df) < 60: return {"top_turns": []}
    dates = pd.DatetimeIndex(df.index)
    highs = np.asarray(df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"], dtype=float)
    lows = np.asarray(df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"], dtype=float)
    n = len(df)
    
    pivots = []
    offset = n - 180 if n > 180 else 0
    for i in range(7, (180 if n > 180 else n) - 7):
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
        
        # 理據化 1: 依據歷史高低點
        for pv in pivots[-4:]:
            el_td = future_idx - pv["idx"]
            if el_td in trading_cycles:
                score += trading_cycles[el_td]
                reasons.append(f"基於 {pv['date'].strftime('%Y-%m-%d')} {pv['type']}({pv['price']:.1f}) 疊加 {el_td} 交易日江恩週期")
            
            el_cd = (f_date - pv["date"]).days
            if any(abs(el_cd - cc) <= 1 for cc in [30, 45, 60, 90]):
                score += 1.5
                reasons.append(f"基於 {pv['date'].strftime('%Y-%m-%d')} {pv['type']} 疊加自然日循環")
                
        # 理據化 2: 自然節氣
        for m, d, name in gann_seasonal_dates:
            if abs((f_date - pd.Timestamp(year=f_date.year, month=m, day=d)).days) <= 1:
                score += 3.5
                reasons.append(f"觸發江恩四方形關鍵節氣：{name}")
                
        if score > 0:
            records.append({
                "預測轉勢日期": f_date.strftime("%Y-%m-%d"),
                "星期": ["一", "二", "三", "四", "五", "六", "日"][f_date.weekday()],
                "共振強度分": round(score, 1),
                "時間周期依據": " ｜ ".join(set(reasons))
            })
            
    hit_df = pd.DataFrame(records).sort_values(["共振強度分", "預測轉勢日期"], ascending=[False, True]) if records else pd.DataFrame()
    return {"top_turns": hit_df.to_dict("records") if not hit_df.empty else []}

def calc_tspr_system(df, left_bars=10):
    if df is None or len(df) < 30: return {"active_support": np.nan, "is_broken": False, "status": "資料不足"}
    lows = np.asarray(df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"], dtype=float)
    closes = np.asarray(df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"], dtype=float)
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

def generate_drawdown_ladder_html(high52, current_price):
    if np.isnan(high52) or high52 <= 0: return ""
    drawdown_pct = (current_price / high52 - 1) * 100
    next_milestone_pct = math.floor(drawdown_pct / 10) * 10
    if next_milestone_pct > -10: next_milestone_pct = -10
    
    ladder_html = "<div class='ladder-box'>"
    ladder_html += "<b>📐 全景回調接貨階梯 (▼10% 至 ▼90%)：</b><br>"
    for pct in range(10, 100, 10):
        price_at_pct = high52 * (1 - pct/100)
        if -pct == next_milestone_pct:
            ladder_html += f"<span style='color:#f85149; font-weight:bold; font-size:15px; margin-right:14px; border-bottom: 2px solid #f85149;'>▼{pct}%({price_at_pct:.1f})</span> "
        else:
            ladder_html += f"<span style='color:#8b949e; margin-right:14px;'>▼{pct}%({price_at_pct:.1f})</span> "
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
        if -pct == next_milestone_pct: parts.append(f"🎯▼{pct}%({price_at_pct:.1f})")
        else: parts.append(f"▼{pct}%({price_at_pct:.1f})")
    return " | ".join(parts)

# ==================== 決策與見底概率引擎 ====================
def decision_engine(ticker, df, info):
    if df is None or len(df) < 130: return None
    
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    price = safe_float(close.iloc[-1])
    high52 = get_52w_high(df)
    
    rsi = safe_float(calc_rsi(close).iloc[-1], 50)
    macd, macd_signal, _ = calc_macd(close)
    obv_s, obv_ma = calc_obv(df)
    mfi = safe_float(calc_mfi(df).iloc[-1], 50)
    cmf = safe_float(calc_cmf(df).iloc[-1], 0)
    
    obv_status = "流入" if safe_float(obv_s.iloc[-1]) > safe_float(obv_ma.iloc[-1]) else "流出"
    drawdown = (price / high52 - 1) * 100 if high52 > 0 and not np.isnan(high52) else 0
    
    tspr = calc_tspr_system(df)
    turn_info = predict_gann_turn_dates(df, 30)
    nt = turn_info["top_turns"][0] if turn_info["top_turns"] else None
    next_turn_str = f"{nt['預測轉勢日期']} ({nt['共振強度分']}分)" if nt else "—"

    # 💡 5. 新增：股票見底百分比評分系統 (Bottom Probability %)
    bottom_prob = 0
    if drawdown <= -40: bottom_prob += 35
    elif drawdown <= -25: bottom_prob += 20
    elif drawdown <= -15: bottom_prob += 10
    
    if rsi < 30: bottom_prob += 20
    elif rsi < 40: bottom_prob += 10
    
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]) and safe_float(macd.iloc[-1]) < 0:
        bottom_prob += 15
        
    if cmf > 0.05: bottom_prob += 15
    if obv_status == "流入": bottom_prob += 10
    
    if not tspr["is_broken"] and not pd.isna(tspr["active_support"]):
        dist_to_supp = (price / tspr["active_support"] - 1) * 100
        if 0 <= dist_to_supp <= 5: bottom_prob += 15 # 接近支持位
        
    bottom_prob = min(99, bottom_prob) # 封頂 99%
    
    # 其他細節標籤
    pe = safe_float(info.get("pe"))
    pe_str = f"{pe:.1f}" if pe > 0 else "N/A"
    
    flow_tags = []
    if obv_status == "流入": flow_tags.append("OBV增籌")
    if cmf > 0.1: flow_tags.append(f"CMF({cmf:.2f})流入")
    elif cmf < -0.1: flow_tags.append(f"CMF({cmf:.2f})流出")
    flow_str = " ｜ ".join(flow_tags) if flow_tags else "資金平淡"
    
    tspr_sup = tspr["active_support"]
    stop_loss_val = min(price * 0.95, tspr_sup if not pd.isna(tspr_sup) else price * 0.95)

    return {
        "ticker": ticker, "name": info.get("name", ticker), "price": round(price, 3), "high52": round(high52, 3),
        "pe": pe_str, "bottom_prob": bottom_prob, 
        "flow_detail": flow_str, "tspr_status": tspr["status"], "tspr_support": tspr_sup,
        "next_turn_date": next_turn_str, "drawdown": round(drawdown, 2),
        "stop_loss": round(stop_loss_val, 3), 
        "ladder_html": generate_drawdown_ladder_html(high52, price), 
        "ladder_text": generate_drawdown_ladder_text(high52, price)
    }

# ==================== Streamlit UI ====================
if time.time() - st.session_state.last_refresh >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time(); st.cache_data.clear(); clear_errors(); st.rerun()

with st.sidebar:
    st.markdown("## ⚙️ 系統控制")
    market = st.radio("市場選擇", ["🇺🇸 核心與美股", "🇭🇰 港股"], index=0)
    if quote_ctx: st.success("✅ 富途實時行情已連接")
    else: st.info("ℹ️ 使用 Yahoo 備援報價")
    if st.button("🔄 立即刷新快取"): st.cache_data.clear(); st.rerun()

st.markdown("# 🎯 終極撈底系統 V7.0")
st.caption(f"波浪推演 + 深度江恩 + 見底機率引擎｜頁面生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HKT")

tabs = st.tabs([
    "🌍 大市深度分析 (三指數並列)", 
    "📊 關注名單總排行 (優先回調階梯)", 
    "🎯 撈底黃金雷達 (Screener)", 
    "📈 深度技術圖表 (Charts)"
])

# ==================== Tab 1: 大市深度分析 (三重拆解) ====================
with tabs[0]:
    st.subheader("🌍 大市深度分析 (恒指 / 標普 / 納指)")
    st.caption("每個指數獨立進行：溫度計判斷、江恩轉勢日推算，以及歷史波浪回調日數/幅度推演。")
    
    idx_map = {"恒生指數 (HSI)": "^HSI", "標普500 (SPX)": "^GSPC", "納斯達克 (NDX)": "^NDX"}
    cols = st.columns(3)
    
    for col, (idx_name, idx_tk) in zip(cols, idx_map.items()):
        with col:
            st.markdown(f"### {idx_name}")
            df_idx = fetch_ohlcv(idx_tk, period="1y")
            if df_idx is not None:
                # 溫度計
                close = df_idx["close"].iloc[:, 0] if isinstance(df_idx["close"], pd.DataFrame) else df_idx["close"]
                ma200 = close.rolling(200).mean().iloc[-1]
                rsi_val = calc_rsi(close).iloc[-1]
                if close.iloc[-1] < ma200 and rsi_val < 35: state = "🥶 冰點 (恐慌撈底區)"
                elif close.iloc[-1] < ma200: state = "🧊 冷卻 (反覆築底)"
                elif rsi_val > 70: state = "🔥 過熱 (泡沫區)"
                else: state = "☀️ 溫和 (穩健區)"
                
                st.info(f"**大市狀態：** {state}")
                
                # 江恩轉勢
                turn_res = predict_gann_turn_dates(df_idx, 30)
                if turn_res.get("top_turns"):
                    t = turn_res["top_turns"][0]
                    st.markdown(f"**🔥 下個高機率變盤日：** {t['預測轉勢日期']}")
                    st.caption(f"理據：{t['時間周期依據']}")
                
                # 波浪推演
                wave_res = analyze_wave_cycles(df_idx)
                if wave_res:
                    st.markdown(
                        f"<div class='wave-box'>"
                        f"<b>🌊 歷史波浪均值統計：</b><br>"
                        f"平均跌浪：<b>{wave_res['avg_down_days']}日</b> (跌幅 <b>{wave_res['avg_down_pct']}%</b>)<br>"
                        f"平均升浪：<b>{wave_res['avg_up_days']}日</b> (升幅 <b>{wave_res['avg_up_pct']}%</b>)<br>"
                        f"<hr style='margin:8px 0; border-color:#30363d;'>"
                        f"<b>🔮 AI 推演預測：</b><br>"
                        f"{wave_res['prediction_desc']}"
                        f"</div>", unsafe_allow_html=True
                    )
            else:
                st.error("無法取得指數數據。")

# ==================== Tab 2: 關注名單總排行 (置頂+回調階梯優先) ====================
with tabs[1]:
    st.subheader("📊 關注名單：技術評分總排行")
    st.caption("以『見底概率% (Bottom Probability)』為核心。專屬名單永久置頂，全景回調階梯優先顯示。")
    
    tickers = US_WATCHLIST if market == "🇺🇸 核心與美股" else HK_WATCHLIST
    active_pinned = US_PINNED if market == "🇺🇸 核心與美股" else HK_PINNED
    
    if st.button("🚀 載入並分析全名單", type="primary", key="rank_btn"):
        progress = st.progress(0, text="下載全市場資料中...")
        info_map, data_map = {}, fetch_multiple(tickers, period="1y")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            for fut in as_completed({executor.submit(get_full_stock_info, tk): tk for tk in tickers}):
                info_map[fut.result()["ticker"]] = fut.result()
                
        progress.progress(50, text="運算見底概率與江恩模型...")
        results = []
        for i, tk in enumerate(tickers):
            res = decision_engine(tk, data_map.get(tk), info_map.get(tk, {}))
            if res: results.append(res)
            progress.progress(50 + int((i+1)/len(tickers)*50))
        progress.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results["置頂"] = df_results["ticker"].apply(lambda x: "📌" if x in active_pinned else "")
            
            # 💡 排序：置頂優先，然後按【見底概率%】由高至低
            df_results = df_results.sort_values(["置頂", "bottom_prob"], ascending=[False, False])
            st.session_state["latest_results"] = df_results
            
            # 💡 調整顯示順序，把梯隊字串移到前面
            display_df = df_results[[
                "置頂", "ticker", "name", "bottom_prob", "ladder_text", 
                "pe", "flow_detail", "tspr_status", "next_turn_date", "drawdown", "stop_loss"
            ]].rename(columns={
                "ticker": "代碼", "name": "名稱", "bottom_prob": "見底概率%", "ladder_text": "🎯 全景回調階梯", 
                "pe": "PE", "flow_detail": "資金流", "tspr_status": "TSPR 狀態", 
                "next_turn_date": "下個轉勢日", "drawdown": "高位回調%", "stop_loss": "止損位"
            })
            
            st.success("排名完成！你的專屬名單已永久置頂，其餘按見底概率排序。")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 📌 置頂專屬名單：深度回調階梯與見底分析")
            pinned_results = df_results[df_results["置頂"] == "📌"]
            
            for _, row in pinned_results.iterrows():
                # 見底顏色判斷
                prob_color = "#3fb950" if row['bottom_prob'] >= 60 else ("#d29922" if row['bottom_prob'] >= 30 else "#f85149")
                
                st.markdown(
                    f"<div class='decision-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 見底概率: <b style='color:{prob_color}; font-size:16px;'>{row['bottom_prob']}%</b> ｜ 現時PE: <b>{row['pe']}</b><br>"
                    f"<b>資金動向：</b> {row['flow_detail']}<br>"
                    f"<b>行動建議：</b> 配合轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。<br>"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 3: 撈底黃金雷達 ====================
with tabs[2]:
    st.subheader("🎯 撈底黃金雷達")
    st.caption("自動過濾出全市場：見底概率 >= 45% 的超跌優質股，適合進行右側交易。")
    
    if "latest_results" not in st.session_state:
        st.info("請先在『Tab 2: 關注名單』點擊載入，雷達掃描名單會自動為你過濾生成。")
    else:
        df_results = st.session_state["latest_results"]
        bottom_fish_df = df_results[df_results["bottom_prob"] >= 45].copy()
        
        if bottom_fish_df.empty:
            st.warning("目前沒有符合【高見底概率】的股票。切勿夾硬買入。")
        else:
            st.success(f"雷達鎖定 {len(bottom_fish_df)} 隻高潛力撈底目標！")
            for _, row in bottom_fish_df.iterrows():
                st.markdown(
                    f"<div class='good-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 見底概率: <b style='color:#3fb950;'>{row['bottom_prob']}%</b><br>"
                    f"<b>資金動向：</b> {row['flow_detail']}<br>"
                    f"<b>行動建議：</b> 配合轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。<br>"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 4: 深度技術圖表 ====================
with tabs[3]:
    st.subheader("📈 深度技術圖表｜右側確認專用")
    
    col_q1, col_q2 = st.columns([1, 4])
    with col_q1: default_tk = st.selectbox("快捷選擇", ["0700.HK", "9988.HK", "NVDA", "NFLX"])
    with col_q2: ticker = st.text_input("輸入股票代碼以查看圖表", default_tk, key="chart_tk").upper()
    
    if ticker:
        chart_df = fetch_ohlcv(ticker, period="1y")
        if chart_df is not None and len(chart_df) > 30:
            close = chart_df["close"].iloc[:, 0] if isinstance(chart_df["close"], pd.DataFrame) else chart_df["close"]
            tspr_res = calc_tspr_system(chart_df)
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
            
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=close, name="K線"), row=1, col=1)
            if not pd.isna(tspr_res["active_support"]):
                fig.add_hline(y=tspr_res["active_support"], line_dash="dash", line_color="#3fb950" if not tspr_res["is_broken"] else "#ef5350", annotation_text="TSPR 10日支持", row=1, col=1)

            vol = chart_df["volume"].iloc[:, 0] if isinstance(chart_df["volume"], pd.DataFrame) else chart_df["volume"]
            op = chart_df["open"].iloc[:, 0] if isinstance(chart_df["open"], pd.DataFrame) else chart_df["open"]
            vol_colors = ["#3fb950" if c>=o else "#f85149" for c, o in zip(close, op)]
            fig.add_trace(go.Bar(x=chart_df.index, y=vol, marker_color=vol_colors, name="成交量"), row=2, col=1)
            
            macd, sig, hist = calc_macd(close)
            fig.add_trace(go.Bar(x=chart_df.index, y=hist, marker_color=[C_GREEN if x>=0 else C_RED for x in hist.fillna(0)], name="MACD Hist"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=macd, line=dict(color=C_BLUE), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=sig, line=dict(color=C_ORANGE), name="Signal"), row=3, col=1)
            
            fig.update_layout(height=800, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
