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

st.set_page_config(page_title="📈 終極撈底系統 V6.6 (置頂名單 + 全景階梯)", page_icon="🎯", layout="wide")

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

# 💡 你的專屬永久置頂名單
HK_PINNED = ["9988.HK", "0700.HK", "0124.HK", "3690.HK", "1810.HK", "0388.HK"]
US_PINNED = ["NFLX", "NVDA", "GOOGL", "MSFT", "META", "MU", "TSM", "SPCX", "AVGO", 
             "TSLA", "AAPL", "AMD", "AMZN", "ORCL", "PLTR", "UNH", "PDD"]

# 將置頂名單與其他觀察清單合併 (利用 set 去重複，並確保置頂名單在前)
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
    except Exception:
        return None

quote_ctx = init_futu()

def requested_start(period):
    now = datetime.now()
    mapping = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 800}
    return (now - timedelta(days=mapping.get(period, 400))).strftime("%Y-%m-%d")

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
        except Exception:
            pass
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df.sort_index().dropna(subset=["open", "high", "low", "close", "volume"])
    except Exception as exc:
        add_error("OHLCV 失敗", ticker, exc)
        return None

def fetch_multiple(tickers, period="1y"):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, period): ticker for ticker in tickers}
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
    except Exception:
        pass
    return result

# ==================== 技術與資金流指標計算 ====================
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
    high = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    low = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    vol = df["volume"].iloc[:, 0] if isinstance(df["volume"], pd.DataFrame) else df["volume"]
    
    typical = (high + low + close) / 3
    money = typical * vol
    pos = money.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg = money.where(typical < typical.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))

def calc_cmf(df, period=20):
    high = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    low = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    vol = df["volume"].iloc[:, 0] if isinstance(df["volume"], pd.DataFrame) else df["volume"]
    
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_multiplier * vol
    cmf = mf_volume.rolling(period).sum() / vol.rolling(period).sum()
    return cmf

def get_52w_high(df):
    high_col = df["high"].iloc[:, 0] if isinstance(df["high"], pd.DataFrame) else df["high"]
    highs = np.asarray(high_col, dtype=float)
    window = highs[-252:] if len(highs) >= 252 else highs
    return float(np.nanmax(window)) if len(window) > 0 and not np.all(np.isnan(window)) else np.nan

# 💡 全景回調階梯生成器 (視覺化 HTML 卡片用)
def generate_drawdown_ladder_html(high52, current_price):
    if np.isnan(high52) or high52 <= 0: return ""
    drawdown_pct = (current_price / high52 - 1) * 100
    next_milestone_pct = math.floor(drawdown_pct / 10) * 10
    if next_milestone_pct > -10: next_milestone_pct = -10
    
    ladder_html = "<div class='ladder-box'>"
    ladder_html += f"<b>📉 52周高位：<span style='color:#58a6ff'>{high52:.2f}</span> ｜ 現價：<span style='color:#58a6ff'>{current_price:.2f}</span> ｜ 目前跌幅：<span style='color:#f85149'>{drawdown_pct:.1f}%</span></b><br>"
    ladder_html += "<b>📐 全景回調接貨階梯 (▼10% 至 ▼90%)：</b><br>"
    
    for pct in range(10, 100, 10):
        price_at_pct = high52 * (1 - pct/100)
        if -pct == next_milestone_pct:
            ladder_html += f"<span style='color:#f85149; font-weight:bold; font-size:15px; margin-right:16px; border-bottom: 2px solid #f85149;'>▼{pct}%: {price_at_pct:.2f}</span> "
        else:
            ladder_html += f"<span style='color:#8b949e; margin-right:16px;'>▼{pct}%: {price_at_pct:.2f}</span> "
        if pct == 40 or pct == 80:
            ladder_html += "<br>"
    ladder_html += "</div>"
    return ladder_html

# 💡 全景回調階梯生成器 (Dataframe 壓縮字串用)
def generate_drawdown_ladder_text(high52, current_price):
    if np.isnan(high52) or high52 <= 0: return ""
    drawdown_pct = (current_price / high52 - 1) * 100
    next_milestone_pct = math.floor(drawdown_pct / 10) * 10
    if next_milestone_pct > -10: next_milestone_pct = -10
    
    parts = []
    for pct in range(10, 100, 10):
        price_at_pct = high52 * (1 - pct/100)
        if -pct == next_milestone_pct:
            parts.append(f"🎯▼{pct}%({price_at_pct:.1f})")
        else:
            parts.append(f"▼{pct}%({price_at_pct:.1f})")
    return " | ".join(parts)

# ==================== 江恩與 TSPR ====================
def calc_tspr_system(df, left_bars=10):
    if df is None or len(df) < 30: return {"active_support": np.nan, "is_broken": False, "status": "資料不足"}
    low_col = df["low"].iloc[:, 0] if isinstance(df["low"], pd.DataFrame) else df["low"]
    close_col = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    lows, closes = np.asarray(low_col, dtype=float), np.asarray(close_col, dtype=float)
    active_support, is_broken = None, False
    
    for i in range(left_bars + 5, len(df)):
        pivot_idx = i - 5
        window = lows[max(0, pivot_idx - left_bars) : i + 1]
        if len(window) > 0 and not np.all(np.isnan(window)):
            if not np.isnan(lows[pivot_idx]) and lows[pivot_idx] == np.nanmin(window):
                active_support, is_broken = float(lows[pivot_idx]), False
        if active_support is not None and closes[i] < active_support:
            is_broken = True
            
    status_str = "🔴 跌穿10日支持" if is_broken else ("🟢 守穩10日支持" if active_support else "⚪ 尋底中")
    return {"active_support": round(active_support, 3) if active_support else np.nan, "is_broken": is_broken, "status": status_str}

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
        w_highs, w_lows = highs[g_idx - 7 : g_idx + 8], lows[g_idx - 7 : g_idx + 8]
        if len(w_highs) > 0 and not np.all(np.isnan(w_highs)) and highs[g_idx] == np.nanmax(w_highs): 
            pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "頂部", "price": highs[g_idx]})
        if len(w_lows) > 0 and not np.all(np.isnan(w_lows)) and lows[g_idx] == np.nanmin(w_lows): 
            pivots.append({"idx": g_idx, "date": dates[g_idx], "type": "底部", "price": lows[g_idx]})
            
    pivots = sorted({p["idx"]: p for p in pivots}.values(), key=lambda x: x["idx"])
    future_bdays = pd.bdate_range(start=dates[-1] + timedelta(days=1), periods=forward_trading_days)
    
    trading_cycles = {15: 1.5, 21: 2.0, 30: 2.5, 45: 3.5, 60: 3.0, 90: 4.5, 120: 3.5}
    gann_seasonal_dates = [(3, 21, "春分"), (6, 21, "夏至"), (9, 22, "秋分"), (12, 21, "冬至"), (2, 4, "立春"), (5, 5, "立夏"), (8, 7, "立秋"), (11, 7, "立冬")]
    
    records = []
    for step, f_date in enumerate(future_bdays, start=1):
        future_idx = (n - 1) + step
        score, reasons = 0.0, []
        
        for pv in pivots[-5:]:
            el_td = future_idx - pv["idx"]
            if el_td in trading_cycles:
                score += trading_cycles[el_td]
                reasons.append(f"由 {pv['date'].strftime('%m-%d')} {pv['type']}起算 {el_td} 交易日")
            el_cd = (f_date - pv["date"]).days
            if any(abs(el_cd - cc) <= 1 for cc in [30, 45, 60, 90, 180]):
                score += 2.0
                reasons.append(f"由 {pv['date'].strftime('%m-%d')} {pv['type']}起算自然日循環")
                
        for m, d, name in gann_seasonal_dates:
            if abs((f_date - pd.Timestamp(year=f_date.year, month=m, day=d)).days) <= 1:
                score += 3.5
                reasons.append(f"江恩四方形節氣：{name}")
                
        if score > 0:
            records.append({
                "預測轉勢日期": f_date.strftime("%Y-%m-%d"),
                "星期": ["一", "二", "三", "四", "五", "六", "日"][f_date.weekday()],
                "共振強度分": score,
                "時間周期依據": " ｜ ".join(set(reasons))
            })
            
    hit_df = pd.DataFrame(records).sort_values(["共振強度分", "預測轉勢日期"], ascending=[False, True]) if records else pd.DataFrame()
    return {"top_turns": hit_df.to_dict("records") if not hit_df.empty else []}

# ==================== 決策引擎 (包含資金流強化) ====================
@st.cache_data(ttl=OHLCV_TTL)
def fetch_macro():
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        df = fetch_ohlcv(ticker, period="1mo")
        if df is not None and not df.empty:
            close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
            result[name] = {"val": safe_float(close.iloc[-1]), "chg": (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close)>1 else 0}
    return result

def classify_market_state(market, macro_data):
    benchmark = "^HSI" if market == "🇭🇰 港股" else "SPY"
    vol_key = "VHSI" if market == "🇭🇰 港股" else "VIX"
    df = fetch_ohlcv(benchmark, period="1y")
    if df is None or len(df) < 205: return {"name": "未知", "strategy": "數據不足，觀望為主。"}
    
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    ma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    rsi_14 = calc_rsi(close).iloc[-1]
    vol_index = safe_float(macro_data.get(vol_key, {}).get("val"), 20)
    
    if price < ma200 and vol_index > 28 and rsi_14 < 35:
        name, strategy = "🥶 冰點 (恐慌拋售區)", "大市極度恐慌，優質股被錯殺。分批撈底高價值股黃金時期，重點看技術面嚴重超賣及資金逆勢流入的個股。"
    elif price < ma200 and vol_index <= 28:
        name, strategy = "🧊 冷卻 (反覆築底區)", "大市恐慌減退。關注江恩轉勢日，等待 TSPR 10日支持位確立才小注試倉，嚴格止損。"
    elif price > ma200 and rsi_14 > 70:
        name, strategy = "🔥 過熱 (狂熱泡沫區)", "大市超買。停止追入新倉，現有持倉應利用 10日支持位作移動止盈，準備鎖定利潤。"
    elif price > ma200:
        name, strategy = "☀️ 溫和 (穩健牛市區)", "大市趨勢向上。持有優質股，逢回調至 TSPR 支持位可加倉。"
    else:
        name, strategy = "⛅ 中性 (震盪上落區)", "區間上落。高估值不追，低估值可買，嚴格控制注碼。"
        
    return {"name": name, "benchmark": benchmark, "strategy": strategy}

def decision_engine(ticker, df, info):
    if df is None or len(df) < 130: return None
    
    close = df["close"].iloc[:, 0] if isinstance(df["close"], pd.DataFrame) else df["close"]
    price = safe_float(close.iloc[-1])
    high52 = get_52w_high(df)
    
    # 指標計算
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

    # 1. 估值評估
    pe = safe_float(info.get("pe"))
    pe_str = f"{pe:.1f}" if pe > 0 else "N/A"
    
    # 2. 超賣程度
    os_score = 0
    os_desc = []
    if rsi < 35: os_score += 40; os_desc.append(f"RSI({rsi:.1f})嚴重超賣")
    elif rsi < 45: os_score += 20; os_desc.append(f"RSI({rsi:.1f})偏弱")
    if drawdown <= -30: os_score += 40; os_desc.append(f"回調過深({drawdown:.1f}%)")
    elif drawdown <= -15: os_score += 20; os_desc.append(f"正常回調({drawdown:.1f}%)")
    if safe_float(macd.iloc[-1]) > safe_float(macd_signal.iloc[-1]) and safe_float(macd.iloc[-1]) < 0:
        os_score += 20; os_desc.append("MACD底背馳")
    os_score = min(100, os_score)
    os_str = " ｜ ".join(os_desc) if os_desc else "未見明顯超賣"
    
    # 3. 資金流強化分析
    flow_score = 0
    flow_tags = []
    if obv_status == "流入": flow_score += 35; flow_tags.append("OBV大戶增籌")
    else: flow_tags.append("OBV量能減弱")
        
    if mfi > 60: flow_score += 35; flow_tags.append(f"MFI({mfi:.0f})熱度高")
    elif mfi < 40: flow_tags.append(f"MFI({mfi:.0f})資金枯竭")
        
    if cmf > 0.1: flow_score += 30; flow_tags.append(f"CMF({cmf:.2f})強勢流入")
    elif cmf < -0.1: flow_tags.append(f"CMF({cmf:.2f})顯著流出")
        
    flow_str = " ｜ ".join(flow_tags)
    
    # 4. 總技術評分
    struct_score = 100 if not tspr["is_broken"] and not pd.isna(tspr["active_support"]) else 0
    tech_score = (os_score * 0.4) + (flow_score * 0.3) + (struct_score * 0.3)
    
    tspr_sup = tspr["active_support"]
    stop_loss_val = min(price * 0.95, tspr_sup if not pd.isna(tspr_sup) else price * 0.95)
    ladder_html = generate_drawdown_ladder_html(high52, price)
    ladder_text = generate_drawdown_ladder_text(high52, price)

    return {
        "ticker": ticker, "name": info.get("name", ticker), "price": round(price, 3), "high52": round(high52, 3),
        "pe": pe_str, "tech_score": round(tech_score, 1),
        "oversold_detail": os_str, "flow_detail": flow_str,
        "tspr_status": tspr["status"], "tspr_support": tspr_sup,
        "next_turn_date": next_turn_str, "drawdown": round(drawdown, 2),
        "stop_loss": round(stop_loss_val, 3), "ladder_html": ladder_html, "ladder_text": ladder_text
    }

# ==================== Streamlit UI ====================
elapsed = time.time() - st.session_state.last_refresh
if elapsed >= AUTO_REFRESH_SEC:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear(); clear_errors(); st.rerun()

with st.sidebar:
    st.markdown("## ⚙️ 系統控制")
    market = st.radio("市場選擇", ["🇺🇸 核心與美股", "🇭🇰 港股"], index=0)
    if quote_ctx: st.success("✅ 富途實時行情已連接")
    else: st.info("ℹ️ 使用 Yahoo 備援報價")
    if st.button("🔄 立即刷新快取"): st.cache_data.clear(); st.rerun()

macro_data = fetch_macro()
regime = classify_market_state(market, macro_data)
benchmark_df = fetch_ohlcv(regime["benchmark"], period="1y")

st.markdown("# 🎯 終極撈底系統 V6.6")
st.caption(f"專屬置頂名單 + 全景回調階梯雙模式顯示｜頁面生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HKT")

tabs = st.tabs([
    "🌍 大市深度剖析 (Macro & Gann)", 
    "📊 關注名單：技術評分總排行", 
    "🎯 撈底黃金雷達 (Screener)", 
    "📈 深度技術圖表 (Charts)"
])

# ==================== Tab 1: 大市深度剖析 ====================
with tabs[0]:
    st.subheader(f"大市氣氛溫度計：{regime['name']}")
    st.info(f"💡 **資金對策：** {regime['strategy']}")
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("VIX", "😱 美股 VIX"), ("VHSI", "🇭🇰 港股 VHSI"), ("SPX", "🇺🇸 S&P 500"), ("HSI", "🇭🇰 恒生指數")]):
        item = macro_data.get(key, {})
        col.metric(label, f"{safe_float(item.get('val'), 0):.2f}", f"{safe_float(item.get('chg'), 0):+.2f}%")
        
    st.markdown("### 📅 大市(基準)下個江恩轉勢日 深度理據剖析")
    turn_res = predict_gann_turn_dates(benchmark_df, forward_trading_days=45)
    top_turns = turn_res.get("top_turns", [])
    if top_turns:
        best_turn = top_turns[0]
        st.markdown(
            f"<div class='decision-card' style='border-left-color:#f85149'>"
            f"<h3 style='margin-top:0;'>🔥 最強共振變盤日：{best_turn['預測轉勢日期']} (星期{best_turn['星期']})</h3>"
            f"<b>共振強度分：</b> {best_turn['共振強度分']} 分<br><br>"
            f"<b>📊 深度推演理據 (為何這天容易轉勢？)：</b><br>"
            f"<ul>" + "".join([f"<li>{r}</li>" for r in best_turn['時間周期依據'].split(' ｜ ')]) + "</ul>"
            f"</div>", unsafe_allow_html=True
        )

# ==================== Tab 2: 關注名單：技術評分總排行 ====================
with tabs[1]:
    st.subheader("📊 關注名單：技術評分總排行 (專屬名單永久置頂)")
    st.caption("結合超賣程度與資金流向（OBV/MFI/CMF）進行綜合評分。專屬名單已固定在最上方，並附帶全景回調階梯。")
    
    tickers = US_WATCHLIST if market == "🇺🇸 核心與美股" else HK_WATCHLIST
    active_pinned = US_PINNED if market == "🇺🇸 核心與美股" else HK_PINNED
    
    if st.button("🚀 載入並排行所有股票", type="primary", key="rank_btn"):
        progress = st.progress(0, text="下載全市場資料中...")
        info_map, data_map = {}, fetch_multiple(tickers, period="1y")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DATA) as executor:
            for fut in as_completed({executor.submit(get_full_stock_info, tk): tk for tk in tickers}):
                info_map[fut.result()["ticker"]] = fut.result()
                
        progress.progress(50, text="運算技術與資金流模型...")
        results = []
        for i, tk in enumerate(tickers):
            res = decision_engine(tk, data_map.get(tk), info_map.get(tk, {}))
            if res: results.append(res)
            progress.progress(50 + int((i+1)/len(tickers)*50))
        progress.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            # 💡 判斷是否為永久置頂名單
            df_results["置頂"] = df_results["ticker"].apply(lambda x: "📌" if x in active_pinned else "")
            
            # 💡 排序邏輯：優先按【置頂】排，其次按【技術評分】降序
            df_results = df_results.sort_values(["置頂", "tech_score"], ascending=[False, False])
            st.session_state["latest_results"] = df_results
            
            # Dataframe 概覽 (包含文字版回調階梯)
            display_df = df_results[[
                "置頂", "ticker", "name", "tech_score", "pe", 
                "oversold_detail", "flow_detail", 
                "tspr_status", "next_turn_date", "drawdown", "stop_loss", "ladder_text"
            ]].rename(columns={
                "ticker": "代碼", "name": "名稱", "tech_score": "技術評分", 
                "pe": "現時PE", "oversold_detail": "超賣細節", "flow_detail": "資金流細節(OBV/MFI/CMF)",
                "tspr_status": "TSPR 狀態", "next_turn_date": "下個轉勢日", 
                "drawdown": "高位回調%", "stop_loss": "止損位", "ladder_text": "全景回調階梯 (▼10% 至 ▼90%)"
            })
            
            st.success("排名完成！你的專屬名單已永久置頂，其餘按技術評分排序。")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 💡 為置頂專屬名單生成視覺化「全景回調階梯卡片」
            st.markdown("### 📌 永久置頂專屬名單 (深度回調階梯與分析)")
            pinned_results = df_results[df_results["置頂"] == "📌"]
            
            for _, row in pinned_results.iterrows():
                st.markdown(
                    f"<div class='decision-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 技術評分: <b style='color:#f85149;'>{row['tech_score']}</b> ｜ 現時PE: <b>{row['pe']}</b><br>"
                    f"<b>資金動向：</b> <span style='color:#3fb950; font-weight:bold;'>{row['flow_detail']}</span><br>"
                    f"<b>行動建議：</b> 配合轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。止損：<b>{row['stop_loss']}</b><br>"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 3: 撈底黃金雷達 ====================
with tabs[2]:
    st.subheader("🎯 撈底黃金雷達")
    st.caption("自動過濾出全市場：技術評分 >= 40 且 有明顯資金流入的股票，適合進行右側交易。")
    
    if "latest_results" not in st.session_state:
        st.info("請先在『Tab 2: 關注名單』點擊載入，雷達掃描名單會自動為你過濾生成。")
    else:
        df_results = st.session_state["latest_results"]
        bottom_fish_df = df_results[df_results["tech_score"] >= 40].copy()
        
        if bottom_fish_df.empty:
            st.warning("目前沒有符合【高超賣 + 資金流入】標準的股票。切勿夾硬買入。")
        else:
            st.success(f"雷達鎖定 {len(bottom_fish_df)} 隻高潛力撈底目標！")
            for _, row in bottom_fish_df.iterrows():
                st.markdown(
                    f"<div class='good-card'>"
                    f"⭐ <b style='font-size:18px; color:#58a6ff;'>{row['ticker']} {row['name']}</b> ｜ 技術評分: <b style='color:#f85149;'>{row['tech_score']}</b> ｜ 現時PE: <b>{row['pe']}</b><br>"
                    f"<b>超賣細節：</b> {row['oversold_detail']}<br>"
                    f"<b>資金動向：</b> <span style='color:#3fb950; font-weight:bold;'>{row['flow_detail']}</span><br>"
                    f"<b>行動建議：</b> 配合江恩轉勢日 <b>{row['next_turn_date']}</b>，企穩 <b>{row['tspr_status']}</b>（{row['tspr_support']}）可建首注。<br>"
                    f"{row['ladder_html']}"
                    f"</div>", unsafe_allow_html=True
                )

# ==================== Tab 4: 深度技術圖表 ====================
with tabs[3]:
    st.subheader("📈 深度技術圖表｜右側確認專用")
    st.caption("輸入代碼確認 TSPR 支持線與資金流 (成交量/MACD) 的實際圖表表現。")
    
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
