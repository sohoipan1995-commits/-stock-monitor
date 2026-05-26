import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="📈 撈底監察系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
import time

# ── 自動刷新：每30分鐘重新載入 ──────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, 1800 - int(elapsed))
mins, secs = divmod(remaining, 60)

with st.sidebar:
    st.markdown(f"🔄 自動刷新：**{mins:02d}:{secs:02d}**")
if st.button("🔄 立即刷新"):
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()
st.divider()

if elapsed >= 1800:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()
st.markdown("""
<style>
  [data-testid="stAppViewContainer"]{background:#0d1117;}
  [data-testid="stSidebar"]{background:#161b22;}
  h1,h2,h3,h4,h5,h6,p,label,.stMarkdown{color:#e6edf3!important;}
  .metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
    padding:16px;text-align:center;margin:4px;}
  div[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:10px;}
</style>
""", unsafe_allow_html=True)

# ── 觀察名單（永久硬編碼）────────────────────────────────────────────────────
HK_WATCHLIST = [
    "0700.HK","0005.HK","0939.HK","1398.HK","3988.HK",
    "0388.HK","0066.HK","0003.HK","0002.HK","0016.HK",
    "0883.HK","2318.HK","1299.HK","0001.HK","9988.HK",
    "0175.HK","0027.HK","2628.HK","0011.HK","0688.HK",
    "3690.HK","9618.HK","0981.HK","9999.HK","2382.HK",
    "0291.HK","1211.HK","0267.HK","2688.HK","0762.HK",
    "6862.HK","0960.HK","2020.HK","1810.HK","1024.HK",
]

US_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO",
    "ORCL","ASML","AMD","QCOM","INTC","AMAT","LRCX",
    "JPM","BAC","GS","MS","BRK-B",
    "COST","WMT","HD","JNJ","UNH",
    "PFE","XOM","NEE","UBER","LITE",
    "CLX","SPY","QQQ","SOXL","IWM",
]

MACRO_TICKERS = {
    "VIX":"^VIX","VVIX":"^VVIX","SPX":"^GSPC","HSI":"^HSI",
    "DXY":"DX-Y.NYB","US10Y":"^TNX","VHSI":"^VHSI",
    "HYG":"HYG","USDHKD":"USDHKD=X",
}

FIB_LEVELS  = [0.236,0.382,0.500,0.618,0.786]
DROP_LEVELS = [0.10,0.20,0.25,0.30,0.35,0.40]

# ── 數據抓取 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

# ── 技術指標 ─────────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1+gain/loss.replace(0,np.nan)))

def calc_kdj(df, n=9):
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv    = (df["close"]-low_n)/(high_n-low_n).replace(0,np.nan)*100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    return K, D, 3*K-2*D

def calc_macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    m  = ef-es; s = m.ewm(span=signal, adjust=False).mean()
    return m, s, m-s

def calc_cci(df, period=20):
    tp  = (df["high"]+df["low"]+df["close"])/3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-sma)/(0.015*mad.replace(0,np.nan))

def calc_obv(df):
    return (np.sign(df["close"].diff()).fillna(0)*df["volume"]).cumsum()

def calc_wr(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100*(hh-df["close"])/(hh-ll).replace(0,np.nan)

def calc_mfi(df, period=14):
    tp  = (df["high"]+df["low"]+df["close"])/3
    mf  = tp*df["volume"]
    pos = mf.where(tp>tp.shift(1),0).rolling(period).sum()
    neg = mf.where(tp<tp.shift(1),0).rolling(period).sum()
    return 100-(100/(1+pos/neg.replace(0,np.nan)))

def get_52w_high(df):
    return float(df["high"].iloc[-252:].max()) if len(df)>=252 else float(df["high"].max())

def fib_levels(swing_low, swing_high):
    d = swing_high-swing_low
    return {f"{int(f*100)}%": round(swing_high-d*f,3) for f in FIB_LEVELS}

def drop_levels(high_price):
    return {f"-{int(d*100)}%": round(high_price*(1-d),3) for d in DROP_LEVELS}

# ── 核心評分函數（整合雙周期RSI + 成交量深度）──────────────────────────────
def score_stock(df):
    if df is None or len(df)<60: return 0,0,[]
    close   = df["close"]
    volume  = df["volume"]
    rsi_d   = calc_rsi(close,14)
    rsi_w   = calc_rsi(close,70)
    K,D,_   = calc_kdj(df)
    macd,sig,_ = calc_macd(close)
    cci     = calc_cci(df)
    obv     = calc_obv(df)
    wr      = calc_wr(df)
    mfi     = calc_mfi(df)
    sma200  = close.rolling(200).mean()
    sma20   = close.rolling(20).mean()
    vol_ma20= volume.rolling(20).mean()

    def r(s):  return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 50
    def rv(s): return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0

    rsi_val  = r(rsi_d); rsi_w_val = r(rsi_w)
    k_val    = r(K);     d_val     = r(D)
    cci_val  = rv(cci);  wr_val    = rv(wr);  mfi_val = r(mfi)
    macd_val = rv(macd); sig_val   = rv(sig)
    obv_now  = rv(obv)
    obv_prev = float(obv.iloc[-6]) if len(obv)>=6 else obv_now
    sma200_v = rv(sma200); sma20_v = rv(sma20)
    close_v  = float(close.iloc[-1])
    bias200  = (close_v-sma200_v)/sma200_v*100 if sma200_v else 0

    # ① 雙周期RSI共振判斷
    weekly_warning = rsi_w_val > 60
    dual_signal = False; dual_desc = ""
    if rsi_val<35 and 28<=rsi_w_val<=50:
        dual_signal = True
        dual_desc = f"⭐雙周期RSI共振(日{rsi_val:.0f}/周{rsi_w_val:.0f})"
    elif rsi_val<30 and rsi_w_val<55:
        dual_signal = True
        dual_desc = f"日線極度超賣(日{rsi_val:.0f}/周{rsi_w_val:.0f})"

    # ② 成交量深度分析
    vol_signals = []; vol_bonus = 0
    if len(volume)>=10:
        vol_now   = float(volume.iloc[-1])
        vol_1d    = float(volume.iloc[-2]) if len(volume)>=2 else vol_now
        vol_2d    = float(volume.iloc[-3]) if len(volume)>=3 else vol_now
        vol_3d    = float(volume.iloc[-4]) if len(volume)>=4 else vol_now
        vol_ma20v = float(vol_ma20.iloc[-1]) if not pd.isna(vol_ma20.iloc[-1]) else vol_now
        vol_ma5v  = float(volume.rolling(5).mean().iloc[-1])
        vol_ratio = vol_now/vol_ma20v if vol_ma20v>0 else 1
        price_5d  = (close_v-float(close.iloc[-5]))/float(close.iloc[-5])*100 if len(close)>=5 else 0

        # A) 連續3日縮量整理
        if (vol_1d<vol_ma20v*0.8 and vol_2d<vol_ma20v*0.8 and
                vol_3d<vol_ma20v*0.8 and close_v<sma20_v):
            vol_signals.append("連續3日縮量整理")
            vol_bonus += 15

        # B) 放量下跌後縮量（賣壓衰竭）
        if len(volume)>=6:
            if (float(volume.iloc[-4])>vol_ma20v*1.5 and
                    float(close.iloc[-4])<float(close.iloc[-5]) and
                    vol_now<vol_ma20v*0.8):
                vol_signals.append("放量跌後縮量(賣壓衰竭)")
                vol_bonus += 20

        # C) 爆量陽線吸籌
        if vol_ratio>=2.0 and float(close.iloc[-1])>float(df["open"].iloc[-1]):
            vol_signals.append(f"爆量陽線吸籌({vol_ratio:.1f}x均量)")
            vol_bonus += 25

        # D) 價跌量縮（賣壓不強）
        if price_5d<-3 and vol_ma5v<vol_ma20v*0.7:
            vol_signals.append("價跌量縮(賣壓漸弱)")
            vol_bonus += 12

        # E) OBV量先於價
        if len(obv)>=10:
            obv_5  = float(obv.iloc[-5])
            obv_10 = float(obv.iloc[-10])
            if obv_now>obv_5>obv_10 and close_v<=float(close.iloc[-5]):
                vol_signals.append("OBV持續上升(量先於價)")
                vol_bonus += 20

        # F) MFI資金流
        if mfi_val<25:
            vol_signals.append(f"MFI極低({mfi_val:.0f})資金大量流出")
            vol_bonus += 15
        elif mfi_val<35:
            vol_signals.append(f"MFI偏低({mfi_val:.0f})")
            vol_bonus += 8

        # G) 52周極度縮量
        if len(volume)>=252:
            vol_52w_min = float(volume.iloc[-252:].min())
            if vol_now<=vol_52w_min*1.15:
                vol_signals.append("成交量接近52周最低")
                vol_bonus += 18

    # 短線評分
    rsi_mult = 0.5 if weekly_warning else 1.0
    short_score = 0; short_sig = []

    if rsi_val<30:
        short_score += int(15*rsi_mult)
        short_sig.append(f"日RSI超賣({rsi_val:.0f})" + ("⚠️" if weekly_warning else ""))
    elif rsi_val<40:
        short_score += int(8*rsi_mult)

    if k_val<20 and d_val<20:
        short_score += 15; short_sig.append(f"KDJ超賣({k_val:.0f})")
    elif k_val<30: short_score += 7

    if macd_val>sig_val and macd_val<0:
        short_score += 12; short_sig.append("MACD低位金叉")

    if cci_val<-100:
        short_score += 10; short_sig.append(f"CCI超賣({cci_val:.0f})")

    if wr_val<-85:
        short_score += 8; short_sig.append(f"WR超賣({wr_val:.0f})")

    if dual_signal:
        short_score += 20; short_sig.append(dual_desc)

    short_score += min(vol_bonus, 30)
    short_sig.extend(vol_signals)

    # 中線評分
    mid_score = 0; mid_sig = []

    if rsi_w_val<35:
        mid_score += 25; mid_sig.append(f"周RSI超賣({rsi_w_val:.0f})")
    elif rsi_w_val<45:
        mid_score += 12
    if weekly_warning:
        mid_score = int(mid_score*0.6)
        mid_sig.append("⚠️周線仍強(小心假底)")

    if k_val<25 and d_val<25:
        mid_score += 20; mid_sig.append("周KDJ低位")

    if bias200<-20:
        mid_score += 20; mid_sig.append(f"年線乖離{bias200:.1f}%")
    elif bias200<-10:
        mid_score += 10

    if obv_now>obv_prev and close_v<=float(close.iloc[-6]):
        mid_score += 20; mid_sig.append("OBV底背離吸籌")

    if cci_val<-150:
        mid_score += 15; mid_sig.append(f"CCI極度超賣({cci_val:.0f})")

    if dual_signal:
        mid_score += 15
        if dual_desc not in mid_sig: mid_sig.append(dual_desc)

    mid_score += min(vol_bonus//2, 20)
    for vs in vol_signals:
        if vs not in mid_sig: mid_sig.append(vs)

    signals = list(dict.fromkeys(short_sig+mid_sig))
    return min(short_score,100), min(mid_score,100), signals

def signal_label(short, mid):
    if short>=70 or mid>=70: return "🔥 強烈撈底","buy"
    if short>=50 or mid>=50: return "⭐️ 值得關注","watch"
    if short>=35 or mid>=35: return "👁️ 觀察中","observe"
    return "—","none"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro():
    result = {}
    for name,tk in MACRO_TICKERS.items():
        df = fetch_ohlcv(tk, period="1y")
        if df is not None and len(df)>1:
            c   = float(df["close"].iloc[-1])
            p   = float(df["close"].iloc[-2])
            chg = (c-p)/p*100
            hi  = float(df["high"].iloc[-252:].max()) if len(df)>=252 else float(df["high"].max())
            lo  = float(df["low"].iloc[-252:].min())  if len(df)>=252 else float(df["low"].min())
            pct = (c-lo)/(hi-lo)*100 if hi!=lo else 50
            result[name] = {
                "val":c,"chg":chg,"pct":pct,"hi":hi,"lo":lo,
                "rsi":float(calc_rsi(df["close"]).iloc[-1]),
                "close_series":df["close"].tolist()[-60:],
                "vol_ratio": (float(df["volume"].iloc[-1]) / float(df["volume"].rolling(20).mean().iloc[-1])
    if ("volume" in df.columns and len(df) >= 20
        and float(df["volume"].rolling(20).mean().iloc[-1]) > 0) else 1.0),
            }
    return result

@st.cache_data(ttl=3600, show_spinner=False)
def scan_stocks(tickers, vix_val=20.0):
    rows = []
    for tk in tickers:
        df = fetch_ohlcv(tk, period="2y")
        if df is None or len(df)<60: continue
        close_v = float(df["close"].iloc[-1])
        hi52    = get_52w_high(df)
        chg1d   = (close_v-float(df["close"].iloc[-2]))/float(df["close"].iloc[-2])*100
        vol_ma  = float(df["volume"].rolling(20).mean().iloc[-1]) or 1
        vol_rat = float(df["volume"].iloc[-1])/vol_ma
        short_s,mid_s,sigs = score_stock(df)
        label,stype = signal_label(short_s,mid_s)
        swing_lo = float(df["low"].iloc[-126:].min())
        rsi_w_v  = float(calc_rsi(df["close"],70).iloc[-1])

        if vix_val>=30:   vix_env = "🔥 極度恐慌"
        elif vix_val>=25: vix_env = "⚠️ 高波動"
        elif vix_val<=15: vix_env = "😎 市場貪婪"
        else:             vix_env = "😐 中性"

        rows.append({
            "代碼":tk,"現價":round(close_v,3),
            "1日漲跌%":round(chg1d,2),
            "52周高":round(hi52,3),
            "距高位%":round((close_v-hi52)/hi52*100,1),
            "量比":round(vol_rat,2),
            "周線RSI":round(rsi_w_v,1),
            "短線分":short_s,"中線分":mid_s,
            "信號":label,"_type":stype,
            "VIX環境":vix_env,
            "觸發指標":"、".join(sigs) if sigs else "—",
            "_drop":drop_levels(hi52),
            "_fib":fib_levels(swing_lo,hi52),
            "_df":df
        })
    return rows

# ════════════ HEADER ══════════════════════════════════════════════════════════
st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據：Yahoo Finance</p>", unsafe_allow_html=True)
st.divider()

# ════════════ SIDEBAR ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    market = st.radio("市場", ["🇭🇰 港股","🇺🇸 美股","📋 自選"], index=1)
    custom_input = ""
    if market == "📋 自選":
        custom_input = st.text_area("輸入代碼（每行一個）","AAPL\nNVDA\n0700.HK\n9988.HK")
    st.divider()
    filter_sig = st.multiselect("篩選信號",
        ["🔥 強烈撈底","⭐️ 值得關注","👁️ 觀察中","—"],
        default=["🔥 強烈撈底","⭐️ 值得關注"])
    min_short = st.slider("最低短線分",0,100,0)
    min_mid   = st.slider("最低中線分",0,100,0)
    st.divider()
    st.markdown("### 📌 評分說明")
    st.markdown("""
**短線分（0-100）** 5-15日操作
- 雙周期RSI共振（日<35+周28-50）
- KDJ/CCI/WR超賣
- MACD低位金叉
- 爆量陽線/縮量整理/量先於價

**中線分（0-100）** 1-3個月操作
- 周RSI < 35
- 200日均線乖離 < -20%
- OBV底背離吸籌
- 成交量底部形態

**⭐ 最強入場信號**
日線RSI<35 + 周線RSI在30-50
= 雙周期共振，真底部概率最高

**⚠️ 注意**
周線RSI>60時日線超賣
= 只是短暫彈跳，不是真底
    """)

tab1, tab2, tab3, tab4 = st.tabs(["🌍 市場氣氛","📊 個股掃描","📐 回撤計算","📈 技術圖表"])

# ════════════ TAB 1: 市場氣氛 ═════════════════════════════════════════════════
with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_market_sentiment():
        result = {}
        macro_map = {
            "VIX":"^VIX","VVIX":"^VVIX","SPX":"^GSPC","HSI":"^HSI",
            "DXY":"DX-Y.NYB","US10Y":"^TNX","VHSI":"^VHSI",
            "HYG":"HYG","USDHKD":"USDHKD=X",
        }
        for name,tk in macro_map.items():
            try:
                df = yf.download(tk, period="1y", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if df.empty or len(df)<5: continue
                c   = float(df["close"].iloc[-1])
                p   = float(df["close"].iloc[-2])
                hi  = float(df["high"].max())
                lo  = float(df["low"].min())
                chg = (c-p)/p*100
                pct = (c-lo)/(hi-lo)*100 if hi!=lo else 50
                result[name] = {
                    "val":c,"chg":chg,"pct":pct,"hi":hi,"lo":lo,
                    "rsi":float(calc_rsi(df["close"]).iloc[-1]),
                    "close_series":df["close"].tolist()[-60:],
                    "vol_ratio": (float(df["volume"].iloc[-1]) / float(df["volume"].rolling(20).mean().iloc[-1])
    if ("volume" in df.columns and len(df) >= 20
        and float(df["volume"].rolling(20).mean().iloc[-1]) > 0) else 1.0),
                }
            except: pass

        sp500_sample = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
                        "JPM","BAC","XOM","JNJ","UNH","PG","AVGO","ORCL",
                        "HD","MA","V","COST","MRK"]
                oversold_count = 0
        valid_count = 0
        for tk in sp500_sample:
            try:
                df = yf.download(tk, period="3mo", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if df.empty or len(df)<20: continue
                rsi_val = float(calc_rsi(df["close"]).iloc[-1])
                if pd.isna(rsi_val): continue
                valid_count += 1
                if rsi_val < 30: oversold_count += 1
            except: pass
        result["breadth_oversold"] = (oversold_count/valid_count*100) if valid_count > 0 else None
        return result

    with st.spinner("載入市場氣氛數據..."):
        mkt = fetch_market_sentiment()

    def safe_get(key, subkey="val", default=0):
        v = mkt.get(key,{})
        return v.get(subkey,default) if isinstance(v,dict) else default

    def calc_opportunity_score():
        score = 50
        vix_v  = safe_get("VIX")
        vhsi_v = safe_get("VHSI")
        hyg_chg= safe_get("HYG","chg")
        dxy_pct= safe_get("DXY","pct")
        breadth= mkt.get("breadth_oversold",0)
        hsi_rsi= safe_get("HSI","rsi")

        if vix_v>=40:   score-=30
        elif vix_v>=30: score-=20
        elif vix_v>=22: score-=8
        elif vix_v<=15: score+=15
        elif vix_v<=18: score+=8

        if vhsi_v>=35:   score-=20
        elif vhsi_v>=25: score-=10
        elif vhsi_v<=18: score+=10

        if breadth>=40:   score-=25
        elif breadth>=25: score-=12
        elif breadth<=5:  score+=10

        if hyg_chg<=-1.5: score-=10
        elif hyg_chg>=0.5: score+=5

        if dxy_pct>=80:  score-=8
        elif dxy_pct<=30: score+=5

        if hsi_rsi<=30:  score-=15
        elif hsi_rsi<=40: score-=5
        elif hsi_rsi>=65: score+=10

        return max(0,min(100,score))

    opportunity_score = 100 - calc_opportunity_score()

    # KPI 卡
    st.markdown("### 📊 全球宏觀指標")
    kpi_items = [
        ("VIX","😱 恐慌指數"),("VVIX","🌊 波動之波動"),
        ("SPX","🇺🇸 標普500"),("HSI","🇭🇰 恒生指數"),
        ("US10Y","🏦 美債10年息"),("DXY","💵 美元指數"),
        ("HYG","📉 高收益債"),("VHSI","🇭🇰 港股波幅"),
    ]
    for row_items in [kpi_items[:4], kpi_items[4:]]:
        cols_kpi = st.columns(4)
    for i,(key,label) in enumerate(row_items):
            val = safe_get(key)
            chg = safe_get(key,"chg")
            pct = safe_get(key,"pct")
            color = "#3fb950" if chg>=0 else "#f85149"
            arrow = "▲" if chg>=0 else "▼"
            emoji = label.split()[0]
            name  = ' '.join(label.split()[1:])
            val_str = f"{val:.2f}"
            chg_str = f"{chg:+.2f}%"
            pct_str = f"{pct:.0f}%"
            with cols_kpi[i]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='font-size:1em'>{emoji}</div>"
                    f"<div style='color:#8b949e;font-size:0.7em'>{name}</div>"
                    f"<div style='font-size:1.1em;font-weight:bold;color:#e6edf3;margin:4px 0'>{val_str}</div>"
                    f"<div style='color:{color};font-size:0.82em'>{arrow} {chg_str}</div>"
                    f"<div style='color:#8b949e;font-size:0.68em'>52W:{pct_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

    st.divider()

    # Gauge 儀表盤
    st.markdown("### 🎯 撈底機會總評")
    g_col1, g_col2 = st.columns([1,1])
    with g_col1:
        if opportunity_score>=70:   gc="#3fb950"; gt="🔥 極佳撈底視窗"
        elif opportunity_score>=55: gc="#d29922"; gt="⚠️ 謹慎撈底機會"
        elif opportunity_score>=40: gc="#8b949e"; gt="😐 市場中性"
        else:                       gc="#f85149"; gt="😎 市場貪婪風險"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=opportunity_score,
            title={"text":f"撈底機會分<br><span style='font-size:0.7em;color:{gc}'>{gt}</span>"},
            number={"font":{"color":gc,"size":48},"suffix":"/100"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gc,"thickness":0.25},
                "bgcolor":"#161b22","bordercolor":"#30363d",
                "steps":[
                    {"range":[0,25],"color":"#1a1a2e"},{"range":[25,45],"color":"#1c1a00"},
                    {"range":[45,65],"color":"#161b22"},{"range":[65,80],"color":"#0d2818"},
                    {"range":[80,100],"color":"#0d3318"},
                ],
                "threshold":{"line":{"color":"#ffffff","width":3},"thickness":0.8,"value":opportunity_score}
            }
        ))
        fig_gauge.update_layout(height=280,paper_bgcolor="#0d1117",
                                 font=dict(color="#e6edf3"),margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g_col2:
        levels = [
            ("80-100","🟢 極佳撈底視窗","市場極度恐慌，VIX高企，歷史上最強分批建倉時機"),
            ("60-79", "🟢 良好撈底機會","市場明顯悲觀，技術超賣，可輕倉試探"),
            ("40-59", "⚪ 市場中性",    "正常波動，選擇個股技術信號操作"),
            ("20-39", "🟠 市場樂觀",    "情緒偏熱，不宜追高，等回調"),
            ("0-19",  "🔴 極度貪婪",    "市場過熱，VIX極低，宜減倉觀望"),
        ]
        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
        for sr,label,desc in levels:
            lo,hi_s = map(int,sr.split("-"))
            is_cur = lo<=opportunity_score<=hi_s
            bg  = "#0d1f3c" if is_cur else "#161b22"
            brd = "2px solid #58a6ff" if is_cur else "1px solid #30363d"
            st.markdown(
                f"<div style='background:{bg};border:{brd};border-radius:8px;padding:8px 12px;margin:4px 0'>"
                f"<span style='color:#e6edf3;font-weight:bold'>{label}</span> "
                f"<span style='color:#8b949e;font-size:0.75em'>({sr}分)</span><br>"
                f"<span style='color:#8b949e;font-size:0.8em'>{desc}</span></div>",
                unsafe_allow_html=True)

    st.divider()
st.markdown("### 📉 VIX 恐慌指數近30日走勢")
vix_series = mkt.get("VIX", {}).get("close_series", []) if isinstance(mkt.get("VIX"), dict) else []
if len(vix_series) >= 10:
    vix_plot = vix_series[-30:]
    vix_mean = sum(vix_plot) / len(vix_plot)
    vix_now_val = vix_plot[-1]
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(
        y=vix_plot, mode="lines+markers",
        line=dict(color="#f85149", width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.08)",
        name="VIX"
    ))
    fig_vix.add_hline(y=30, line_dash="dash", line_color="#3fb950",
        annotation_text="30 恐慌區", annotation_position="right")
    fig_vix.add_hline(y=20, line_dash="dot", line_color="#d29922",
        annotation_text="20 警戒線", annotation_position="right")
    fig_vix.add_hline(y=15, line_dash="dot", line_color="#f85149",
        annotation_text="15 貪婪區", annotation_position="right")
    fig_vix.add_hline(y=vix_mean, line_dash="dash", line_color="#8b949e",
        annotation_text=f"30日均 {vix_mean:.1f}", annotation_position="left")
    fig_vix.update_layout(
        height=220, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), showlegend=False,
        margin=dict(l=10, r=80, t=15, b=10),
        xaxis=dict(gridcolor="#21262d", showticklabels=False),
        yaxis=dict(gridcolor="#21262d", title="VIX")
    )
    vix_color = "#3fb950" if vix_now_val>=30 else ("#d29922" if vix_now_val>=20 else "#f85149")
    vix_label = "🔥 恐慌區" if vix_now_val>=30 else ("⚠️ 警戒" if vix_now_val>=20 else "😎 貪婪")
    st.markdown(f"<span style='color:{vix_color};font-weight:bold'>現值 {vix_now_val:.2f} — {vix_label}</span> ｜ 30日均值：{vix_mean:.2f}", unsafe_allow_html=True)
    st.plotly_chart(fig_vix, use_container_width=True)
    # 美股 vs 港股 詳細氣氛
    st.markdown("### 📡 美股 vs 港股 詳細氣氛")
    us_col, hk_col = st.columns(2)

    vix_v    = safe_get("VIX"); vvix_v = safe_get("VVIX")
    breadth  = mkt.get("breadth_oversold",0)
    dxy_v    = safe_get("DXY"); dxy_pct = safe_get("DXY","pct")
    us10y_v  = safe_get("US10Y"); hyg_chg = safe_get("HYG","chg")
    spx_rsi  = safe_get("SPX","rsi"); vhsi_v = safe_get("VHSI")
    hsi_rsi_v= safe_get("HSI","rsi"); hsi_pct = safe_get("HSI","pct")
    hsi_vol_r= safe_get("HSI","vol_ratio"); usdhkd_v = safe_get("USDHKD")
    hsi_close= mkt.get("HSI",{}).get("close_series",[]) if isinstance(mkt.get("HSI"),dict) else []
    hsi_20bias = 0
    if hsi_close and len(hsi_close)>=20:
        ma20 = sum(hsi_close[-20:])/20
        hsi_20bias = (hsi_close[-1]-ma20)/ma20*100 if ma20 else 0

    def ind_row(label, value, desc, sc, st_text, fmt=".2f"):
        return (f"<div style='background:#161b22;border-radius:8px;padding:10px 14px;"
                f"margin:5px 0;border-left:3px solid {sc}'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:#e6edf3;font-weight:bold'>{label}</span>"
                f"<span style='color:{sc};font-weight:bold'>{format(value,fmt)}</span></div>"
                f"<div style='color:#8b949e;font-size:0.77em;margin-top:2px'>{desc}</div>"
                f"<div style='color:{sc};font-size:0.77em'>{st_text}</div></div>")

    with us_col:
        st.markdown("#### 🇺🇸 美股市場氣氛")
        vc = "#3fb950" if vix_v>=30 else ("#d29922" if vix_v>=20 else "#f85149")
        st.markdown(ind_row("😱 VIX 恐慌指數",vix_v,
            "市場預期30日波動率。>30=恐慌撈底區；<15=過度貪婪",vc,
            "🔥 恐慌區，撈底機會" if vix_v>=30 else ("⚠️ 警戒區" if vix_v>=20 else "😎 低波動")),
            unsafe_allow_html=True)

        vc2 = "#3fb950" if vvix_v>=120 else ("#d29922" if vvix_v>=100 else "#8b949e")
        st.markdown(ind_row("🌊 VVIX 波動率的波動率",vvix_v,
            "VIX本身的波動。>120=恐慌極端；是市場見底的先行指標",vc2,
            "🌊 極端不穩，恐慌頂部" if vvix_v>=120 else ("⚠️ 波動加劇" if vvix_v>=100 else "—正常")),
            unsafe_allow_html=True)
        if breadth is None:
            st.markdown(ind_row("📊 市場寬度（超賣佔比）", 0,
                "S&P500抽樣RSI<30佔比。>40%=系統性拋售，歷史底部區域", "#8b949e",
                "N/A ⚠️ 數據暫時無法獲取", ".0f"),
                unsafe_allow_html=True)
        else:
        bc = "#3fb950" if breadth>=40 else ("#d29922" if breadth>=20 else "#8b949e")
        st.markdown(ind_row("📊 市場寬度（超賣佔比）",breadth,
            "S&P500抽樣RSI<30佔比。>40%=系統性拋售，歷史底部區域",bc,
            f"🔥 {breadth:.0f}%股票超賣" if breadth>=40 else f"—{breadth:.0f}%超賣",".0f"),
            unsafe_allow_html=True)

        dc = "#f85149" if dxy_pct>=75 else ("#d29922" if dxy_pct>=55 else "#3fb950")
        st.markdown(ind_row("💵 美元指數 DXY",dxy_v,
            "美元強弱影響資金流向。強美元壓制股市；弱美元利好風險資產",dc,
            "⚠️ 美元強勢，壓制股市" if dxy_pct>=75 else ("中性" if dxy_pct>=45 else "✅ 美元偏弱，利好股市")),
            unsafe_allow_html=True)

        tc = "#f85149" if us10y_v>=4.5 else ("#d29922" if us10y_v>=4.0 else "#3fb950")
        st.markdown(ind_row("🏦 美債10年息",us10y_v,
            "無風險回報基準。快速上升壓制股票估值；下行=資金回流股市",tc,
            "⚠️ 高息環境，估值承壓" if us10y_v>=4.5 else ("中性" if us10y_v>=3.5 else "✅ 低息環境")),
            unsafe_allow_html=True)

        hc = "#3fb950" if hyg_chg<=-1.5 else ("#d29922" if hyg_chg<=-0.5 else "#8b949e")
        st.markdown(ind_row("📉 高收益債ETF (HYG) 日變化",hyg_chg,
            "信用市場風向標。HYG下跌=信用風險上升，通常先於股市反映恐慌",hc,
            "🔥 信用市場恐慌" if hyg_chg<=-1.5 else ("⚠️ 信用輕微壓力" if hyg_chg<=-0.5 else "—穩定"),"+.2f"),
            unsafe_allow_html=True)

        sr_c = "#3fb950" if spx_rsi<35 else ("#f85149" if spx_rsi>65 else "#8b949e")
        st.markdown(ind_row("📈 標普500 RSI",spx_rsi,
            "大盤14日RSI。<30=指數超賣；>70=超買。配合VIX一起看效果最佳",sr_c,
            "🔥 大盤超賣" if spx_rsi<35 else ("⚠️ 大盤超買" if spx_rsi>65 else "—中性")),
            unsafe_allow_html=True)

    with hk_col:
        st.markdown("#### 🇭🇰 港股市場氣氛")
        if vhsi_v == 0 or pd.isna(vhsi_v):
            st.markdown(ind_row("😱 VHSI 港股波幅指數", 0,
                "港版VIX。>30=市場恐慌；底部常出現VHSI尖頂後掉頭", "#8b949e",
                "N/A ⚠️ 數據暫時無法獲取"),
                unsafe_allow_html=True)
        else:
        vh_c = "#3fb950" if vhsi_v>=30 else ("#d29922" if vhsi_v>=22 else "#f85149")
        st.markdown(ind_row("😱 VHSI 港股波幅指數",vhsi_v,
            "港版VIX。>30=市場恐慌；底部常出現VHSI尖頂後掉頭",vh_c,
            "🔥 港股恐慌，撈底機會" if vhsi_v>=30 else ("⚠️ 波動加劇" if vhsi_v>=22 else "😎 波動低")),
            unsafe_allow_html=True)

        hr_c = "#3fb950" if hsi_rsi_v<30 else ("#f85149" if hsi_rsi_v>70 else "#8b949e")
        st.markdown(ind_row("📊 恒指 RSI（日線）",hsi_rsi_v,
            "恒指14日RSI。<30=短線超賣，考慮撈底；>70=超買，不宜追",hr_c,
            "🔥 日線超賣，短線機會" if hsi_rsi_v<30 else ("⚠️ 超買" if hsi_rsi_v>70 else "—中性")),
            unsafe_allow_html=True)

        hp_c = "#3fb950" if hsi_pct<=25 else ("#d29922" if hsi_pct<=45 else "#f85149")
        st.markdown(ind_row("📍 恒指52周水位",hsi_pct,
            "現價在52周高低點中的位置。<25%=年度低位，底部機會區",hp_c,
            "🔥 52周低位區" if hsi_pct<=25 else ("⚠️ 中等水位" if hsi_pct<=55 else "😎 高位區"),".1f"),
            unsafe_allow_html=True)

        hv_c = "#3fb950" if hsi_vol_r>=1.5 else ("#8b949e" if hsi_vol_r>=0.8 else "#d29922")
        st.markdown(ind_row("📦 恒指成交量比（vs20日均）",hsi_vol_r,
            "今日量vs20日均量。>1.5=放量，資金介入；底部爆量止跌=真底部",hv_c,
            "🔥 放量，資金介入" if hsi_vol_r>=1.5 else ("—正常" if hsi_vol_r>=0.8 else "⚠️ 縮量")),
            unsafe_allow_html=True)

        hb_c = "#3fb950" if hsi_20bias<=-8 else ("#f85149" if hsi_20bias>=8 else "#8b949e")
        st.markdown(ind_row("📐 恒指20日均線乖離率",hsi_20bias,
            "現價與20日均線偏離。<-8%=嚴重超跌，均值回歸拉力極強",hb_c,
            "🔥 嚴重超跌" if hsi_20bias<=-8 else ("⚠️ 偏高" if hsi_20bias>=8 else "—在均線附近"),"+.1f"),
            unsafe_allow_html=True)

        hd_c = "#f85149" if usdhkd_v>=7.83 else ("#d29922" if usdhkd_v>=7.80 else "#3fb950")
        st.markdown(ind_row("💱 港元匯率 USDHKD",usdhkd_v,
            "接近7.85弱方兌換保證=資金外流壓力。越低越穩定，利好港股",hd_c,
            "⚠️ 接近弱方保證" if usdhkd_v>=7.83 else ("注意走弱" if usdhkd_v>=7.80 else "✅ 港元穩定"),".4f"),
            unsafe_allow_html=True)

                try:
            df_3032 = yf.download("3032.HK", period="5d", auto_adjust=True, progress=False)
            if isinstance(df_3032.columns, pd.MultiIndex):
                df_3032.columns = df_3032.columns.get_level_values(0)
            df_3032.columns = [c.lower() for c in df_3032.columns]
            df_3032 = df_3032.dropna(subset=["Close"] if "Close" in df_3032.columns else ["close"])
            if not df_3032.empty and len(df_3032) >= 2:
                nw_chg = (float(df_3032["close"].iloc[-1]) - float(df_3032["close"].iloc[-2])) / float(df_3032["close"].iloc[-2]) * 100
                if pd.isna(nw_chg):
                    raise ValueError("nw_chg is nan")
                nw_c = "#3fb950" if nw_chg >= 0.3 else ("#f85149" if nw_chg <= -0.3 else "#8b949e")
                st.markdown(ind_row("🌊 北水方向（港股通ETF替代）", nw_chg,
                    "以3032.HK港股通ETF估算北水動向。連續流入=內地資金撐盤", nw_c,
                    "✅ 北水淨流入" if nw_chg >= 0.3 else ("⚠️ 北水淨流出" if nw_chg <= -0.3 else "—小幅變動"), "+.2f"),
                    unsafe_allow_html=True)
            else:
                st.markdown(ind_row("🌊 北水方向（港股通ETF替代）", 0,
                    "以3032.HK港股通ETF估算北水動向。連續流入=內地資金撐盤", "#8b949e",
                    "N/A ⚠️ 數據不足", "+.2f"),
                    unsafe_allow_html=True)
        except:
            st.markdown(ind_row("🌊 北水方向（港股通ETF替代）", 0,
                "以3032.HK港股通ETF估算北水動向。連續流入=內地資金撐盤", "#8b949e",
                "N/A ⚠️ 暫時無法獲取", "+.2f"),
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📈 指數走勢圖（近60日）")
    col_a, col_b = st.columns(2)
    for col,tk,title in [(col_a,"^GSPC","🇺🇸 標普500"),(col_b,"^HSI","🇭🇰 恒生指數")]:
        with col:
            df_idx = fetch_ohlcv(tk, period="3mo")
            if df_idx is not None and len(df_idx)>5:
                sma20_idx = df_idx["close"].rolling(20).mean()
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_idx.index,open=df_idx["open"],high=df_idx["high"],
                    low=df_idx["low"],close=df_idx["close"],
                    increasing_line_color="#3fb950",decreasing_line_color="#f85149",name="K線"))
                fig.add_trace(go.Scatter(x=df_idx.index,y=sma20_idx,mode="lines",
                    line=dict(color="#f0883e",width=1.5),name="MA20"))
                fig.update_layout(title=title,height=320,paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",font=dict(color="#e6edf3"),
                    xaxis_rangeslider_visible=False,legend=dict(bgcolor="#161b22"),
                    margin=dict(l=5,r=5,t=40,b=5))
                fig.update_xaxes(gridcolor="#21262d")
                fig.update_yaxes(gridcolor="#21262d")
                st.plotly_chart(fig, use_container_width=True)

# ════════════ TAB 2: 個股掃描 ════════════════════════════════════════════════
with tab2:
    if market=="🇭🇰 港股":   tickers = HK_WATCHLIST
    elif market=="🇺🇸 美股": tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        tickers = raw if raw else US_WATCHLIST

    # ③ VIX 氣氛濾網
    macro_now = fetch_macro()
    vix_now = macro_now.get("VIX",{}).get("val",20) if isinstance(macro_now.get("VIX"),dict) else 20

    if vix_now>=30:
        fc="#3fb950"; fi="🔥"
        fl=f"VIX {vix_now:.1f} 極度恐慌 — 只顯示最強信號（建議中線分≥60）"
        auto_min_mid=60
    elif vix_now>=25:
        fc="#d29922"; fi="⚠️"
        fl=f"VIX {vix_now:.1f} 高波動市場 — 建議只操作中線分≥50的股票"
        auto_min_mid=50
    elif vix_now<=15:
        fc="#f85149"; fi="😎"
        fl=f"VIX {vix_now:.1f} 市場過度貪婪 — 注意追高風險，降低倉位"
        auto_min_mid=0
    else:
        fc="#8b949e"; fi="😐"
        fl=f"VIX {vix_now:.1f} 市場中性 — 正常操作"
        auto_min_mid=0

    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {fc};"
        f"border-radius:8px;padding:12px 16px;margin-bottom:12px'>"
        f"<span style='font-size:1.1em'>{fi}</span> "
        f"<span style='color:{fc};font-weight:bold'>市場氣氛濾網</span>: "
        f"<span style='color:#e6edf3'>{fl}</span></div>",
        unsafe_allow_html=True)

    effective_min_mid = max(min_mid, auto_min_mid)
    if auto_min_mid>0 and auto_min_mid>min_mid:
        st.caption(f"💡 VIX濾網已自動將最低中線分提升至 {auto_min_mid}（可在側邊欄手動覆蓋）")

    st.subheader(f"📊 個股掃描 — {market} ({len(tickers)} 隻)")
    with st.spinner(f"正在掃描 {len(tickers)} 隻股票..."):
        rows = scan_stocks(tuple(tickers), float(vix_now))

    filtered = [r for r in rows
                if r["信號"] in filter_sig
                and r["短線分"]>=min_short
                and r["中線分"]>=effective_min_mid]

    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 🔥 強烈撈底：{sum(1 for r in filtered if r['_type']=='buy')} 隻**")

    if filtered:
        df_plot = pd.DataFrame([{"代碼":r["代碼"],"短線分":r["短線分"],"中線分":r["中線分"]} for r in filtered])
        fig_bar = px.bar(df_plot.melt(id_vars="代碼",value_vars=["短線分","中線分"]),
                         x="代碼",y="value",color="variable",barmode="group",
                         color_discrete_map={"短線分":"#388bfd","中線分":"#3fb950"},height=260)
        fig_bar.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                               font=dict(color="#e6edf3"),legend_title="",
                               margin=dict(l=5,r=5,t=10,b=5))
        fig_bar.update_xaxes(gridcolor="#21262d")
        fig_bar.update_yaxes(gridcolor="#21262d",range=[0,105])
        st.plotly_chart(fig_bar, use_container_width=True)

        sort_col1, _ = st.columns([2,4])
        with sort_col1:
            sort_by = st.selectbox("排序方式", ["總分（短+中）","短線分","中線分","量比","距高位%","周線RSI"], key="sort_by")
        sort_key_map = {
            "總分（短+中）": lambda x: x["短線分"]+x["中線分"],
            "短線分":  lambda x: x["短線分"],
            "中線分":  lambda x: x["中線分"],
            "量比":    lambda x: x["量比"],
            "距高位%": lambda x: -x["距高位%"],
            "周線RSI": lambda x: x["周線RSI"],
        }
        sorted_filtered = sorted(filtered, key=sort_key_map[sort_by],
                                 reverse=sort_by not in ["距高位%"])
        for r in sorted_filtered:
            weekly_warn_str = " ⚠️周線仍強" if r.get("周線RSI",50)>60 else ""
            with st.expander(
                f"{r['信號']}  {r['代碼']}  現價 {r['現價']}  "
                f"({r['1日漲跌%']:+.1f}%)  ｜ 短線:{r['短線分']}  中線:{r['中線分']}{weekly_warn_str}"
            ):
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("現價",r["現價"])
                c2.metric("52周高",r["52周高"])
                c3.metric("距高位",f"{r['距高位%']}%")
                c4.metric("量比",r["量比"])
                c5.metric("周線RSI",f"{r['周線RSI']:.1f}")

                st.markdown(f"**觸發指標：** {r['觸發指標']}")
                st.markdown(f"**VIX環境：** {r['VIX環境']}")

                # 成交量形態分析
                df_vol = r["_df"]
                if df_vol is not None and len(df_vol)>=20:
                    vol_s    = df_vol["volume"]
                    vol_m20  = vol_s.rolling(20).mean()
                    vol_ma20v = float(vol_m20.iloc[-1])
                    vol_ratios = (vol_s.iloc[-5:]/vol_m20.iloc[-5:].replace(0,np.nan)).round(2)
                    is_shrinking = all(float(vol_s.iloc[-i])<vol_ma20v*0.85 for i in range(1,4))
                    is_expanding = float(vol_s.iloc[-1])>vol_ma20v*1.5
                    is_diverging = (float(vol_s.iloc[-1])<vol_ma20v*0.7 and
                                    float(df_vol["close"].iloc[-1])<float(df_vol["close"].iloc[-2]))

                    vol_tags = []
                    if is_shrinking: vol_tags.append("<span style='background:#0d2818;color:#3fb950;padding:3px 8px;border-radius:4px;font-size:0.8em'>📉 縮量整理</span>")
                    if is_expanding: vol_tags.append("<span style='background:#1c2c1a;color:#3fb950;padding:3px 8px;border-radius:4px;font-size:0.8em'>📈 放量介入</span>")
                    if is_diverging:  vol_tags.append("<span style='background:#0d1f3c;color:#58a6ff;padding:3px 8px;border-radius:4px;font-size:0.8em'>🔵 量縮價跌(動能衰竭)</span>")
                    st.markdown("**量能形態：** " + (" ".join(vol_tags) if vol_tags else "正常"), unsafe_allow_html=True)

                    # 近5日量比小圖
                    dates_5 = [str(d)[:10] for d in df_vol.index[-5:]]
                    vr_list = vol_ratios.fillna(1).tolist()
                    bar_c5  = ["#3fb950" if v>=1 else "#f85149" for v in vr_list]
                    fig_mini = go.Figure(go.Bar(
                        x=dates_5,y=vr_list,marker_color=bar_c5,
                        text=[f"{v:.1f}x" for v in vr_list],
                        textposition="outside",textfont=dict(color="#e6edf3",size=10)
                    ))
                    fig_mini.add_hline(y=1.0,line_dash="dash",line_color="#8b949e",annotation_text="均量")
                    fig_mini.add_hline(y=1.5,line_dash="dot",line_color="#d29922",annotation_text="1.5x")
                    fig_mini.update_layout(title="近5日量比",height=200,
                        paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                        font=dict(color="#e6edf3"),margin=dict(l=5,r=5,t=35,b=5),showlegend=False)
                    fig_mini.update_xaxes(gridcolor="#21262d")
                    fig_mini.update_yaxes(gridcolor="#21262d")
                    st.plotly_chart(fig_mini, use_container_width=True)

                d1,d2 = st.columns(2)
                with d1:
                    st.markdown("**📉 從52周高回撤位**")
                    drop_df = pd.DataFrame(list(r["_drop"].items()),columns=["回撤","價位"])
                    drop_df["狀態"] = drop_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x-r["現價"])/r["現價"]<0.03 else "")
                    st.dataframe(drop_df, use_container_width=True, hide_index=True)
                with d2:
                    st.markdown("**🌀 斐波那契支撐位**")
                    fib_df = pd.DataFrame(list(r["_fib"].items()),columns=["比率","價位"])
                    fib_df["狀態"] = fib_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x-r["現價"])/r["現價"]<0.03 else "")
                    st.dataframe(fib_df, use_container_width=True, hide_index=True)

                st.markdown(f"**🎯 目標+20%：`{round(r['現價']*1.2,3)}`  ｜  止損-8%：`{round(r['現價']*0.92,3)}`**")

    st.divider()
    st.subheader("📋 全部股票列表")
    if rows:
        table_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
        st.dataframe(table_df.sort_values("短線分",ascending=False),
                     use_container_width=True, hide_index=True)

# ════════════ TAB 3: 回撤計算 ════════════════════════════════════════════════
with tab3:
    st.subheader("📐 回撤 & 斐波那契計算器")
    col_in1,col_in2,col_in3 = st.columns(3)
    with col_in1: tk_input    = st.text_input("股票代碼","NVDA").upper()
    with col_in2: manual_high = st.number_input("手動輸入高位（0=自動）",min_value=0.0,value=0.0)
    with col_in3: manual_low  = st.number_input("手動輸入低位（0=自動）",min_value=0.0,value=0.0)

    if st.button("🔍 計算",type="primary"):
        with st.spinner("抓取數據..."):
            df_c = fetch_ohlcv(tk_input, period="2y")
        if df_c is not None:
            close_now = float(df_c["close"].iloc[-1])
            hi = manual_high if manual_high>0 else get_52w_high(df_c)
            lo = manual_low  if manual_low>0  else float(df_c["low"].iloc[-252:].min())

            st.markdown(f"### {tk_input}  現價：**{close_now:.3f}**  ｜  52周高：**{hi:.3f}**  ｜  52周低：**{lo:.3f}**")

            ca,cb = st.columns(2)
            with ca:
                st.markdown("#### 📉 從高位回撤價位")
                rows_d = []
                for pct,price in drop_levels(hi).items():
                    diff = close_now-price
                    status = "✅ 已達到" if close_now<=price*1.02 else f"還需跌 {abs(diff):.2f}"
                    rows_d.append({"回撤幅度":pct,"目標價位":price,"現價距離":f"{diff:+.2f}","狀態":status})
                st.dataframe(pd.DataFrame(rows_d),use_container_width=True,hide_index=True)
            with cb:
                st.markdown("#### 🌀 斐波那契黃金分割")
                rows_f = []; fibs_v = fib_levels(lo,hi)
                for ratio,price in fibs_v.items():
                    diff = close_now-price
                    status = ("◀ 當前附近" if abs(diff)/close_now<0.03
                              else ("✅ 已跌穿" if close_now<price else f"距離 {diff:+.2f}"))
                    rows_f.append({"比率":ratio,"支撐價位":price,"現價距離":f"{diff:+.2f}","狀態":status})
                st.dataframe(pd.DataFrame(rows_f),use_container_width=True,hide_index=True)

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(x=df_c.index[-120:],y=df_c["close"].iloc[-120:],
                mode="lines",name="收盤價",line=dict(color="#58a6ff",width=2)))
            ls = ["#3fb950","#d29922","#f85149","#8957e5","#79c0ff"]
            for i,(ratio,price) in enumerate(fibs_v.items()):
                fig_ret.add_hline(y=price,line_dash="dash",line_color=ls[i%len(ls)],
                    annotation_text=f"Fib {ratio} ({price:.2f})",annotation_position="right")
            fig_ret.add_hline(y=hi,line_color="#ff7b72",line_width=2,
                annotation_text=f"52W High {hi:.2f}",annotation_position="right")
            fig_ret.update_layout(title=f"{tk_input} 斐波那契支撐圖",height=450,
                paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(color="#e6edf3"),
                margin=dict(l=10,r=130,t=40,b=10))
            fig_ret.update_xaxes(gridcolor="#21262d")
            fig_ret.update_yaxes(gridcolor="#21262d")
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.error("找不到數據，港股請用 0700.HK 格式。")

    st.divider()

    # 全觀察名單回撤儀表板
    st.subheader("📡 全觀察名單 回撤深度儀表板")
    st.markdown("自動掃描所有觀察股，計算回撤深度及**即將到達的斐波那契支撐位**。")

    if market=="🇭🇰 港股":   retrace_tickers = HK_WATCHLIST
    elif market=="🇺🇸 美股": retrace_tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        retrace_tickers = raw if raw else US_WATCHLIST

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_retrace_data(tickers):
        results = []
        for tk in tickers:
            df = fetch_ohlcv(tk, period="2y")
            if df is None or len(df)<30: continue
            try:
                close_v = float(df["close"].iloc[-1])
                hi52 = get_52w_high(df)
                lo52 = float(df["low"].iloc[-252:].min()) if len(df)>=252 else float(df["low"].min())
                drop_pct = (close_v-hi52)/hi52*100
                fibs = fib_levels(lo52,hi52)
                fib_values = sorted(fibs.items(),key=lambda x:float(x[0].replace("%","")),reverse=True)

                next_fib_label="—"; next_fib_price=None; next_fib_pct=None; passed_fibs=[]
                for label,price in fib_values:
                    if close_v>price:
                        if next_fib_price is None:
                            next_fib_label=label; next_fib_price=price
                            next_fib_pct=(price-close_v)/close_v*100
                    else:
                        passed_fibs.append(label)

                next_drop_label="—"; next_drop_price=None; next_drop_pct=None
                for dk,dp in drop_levels(hi52).items():
                    if close_v>dp and next_drop_price is None:
                        next_drop_label=dk; next_drop_price=dp
                        next_drop_pct=(dp-close_v)/close_v*100

                results.append({
                    "代碼":tk,"現價":round(close_v,3),"52W高":round(hi52,3),
                    "回撤深度%":round(drop_pct,1),
                    "下一回撤目標":next_drop_label,
                    "目標價":round(next_drop_price,3) if next_drop_price else "—",
                    "距目標%":round(next_drop_pct,1) if next_drop_pct else 0,
                    "下一Fib支撐":next_fib_label,
                    "Fib支撐價":round(next_fib_price,3) if next_fib_price else "—",
                    "距Fib%":round(next_fib_pct,1) if next_fib_pct else 0,
                    "已穿Fib":", ".join(passed_fibs) if passed_fibs else "—",
                    "_hi":hi52,"_lo":lo52,"_close":close_v,
                })
            except: continue
        return results

    with st.spinner("掃描回撤數據..."):
        retrace_data = fetch_retrace_data(tuple(retrace_tickers))

    if retrace_data:
        df_retrace = pd.DataFrame(retrace_data)

        # 圖表一：回撤深度條形圖
        st.markdown("#### 📉 各股從52周高位回撤深度")
        df_sorted = df_retrace.sort_values("回撤深度%")
        bar_colors = []
        for v in df_sorted["回撤深度%"]:
            if v<=-30: bar_colors.append("#3fb950")
            elif v<=-15: bar_colors.append("#d29922")
            elif v<=-10: bar_colors.append("#f0883e")
            else: bar_colors.append("#8b949e")

        fig_bar_ret = go.Figure(go.Bar(
            x=df_sorted["回撤深度%"],y=df_sorted["代碼"],orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in df_sorted["回撤深度%"]],
            textposition="outside",textfont=dict(color="#e6edf3",size=11),
            hovertemplate="<b>%{y}</b><br>回撤:%{x:.1f}%<extra></extra>"))
        for xv,lbl,lc in [(-10,"-10%","#f0883e"),(-20,"-20%","#d29922"),(-30,"-30%","#3fb950")]:
            fig_bar_ret.add_vline(x=xv,line_dash="dash",line_color=lc,
                annotation_text=lbl,annotation_font_color=lc)
        fig_bar_ret.update_layout(
            height=max(400,len(df_sorted)*22),
            paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d",zeroline=True,zerolinecolor="#30363d",range=[-65,5]),
            yaxis=dict(gridcolor="#21262d"),margin=dict(l=10,r=80,t=20,b=20),showlegend=False)
        st.plotly_chart(fig_bar_ret, use_container_width=True)

        st.divider()

        # 圖表二：泡泡圖
        st.markdown("#### 🌀 距離下一個斐波那契支撐位的距離")
        st.caption("X軸=回撤深度，Y軸=距下一個Fib支撐%。越接近0%代表越快到達支撐位。")
        df_bubble = df_retrace[df_retrace["距Fib%"]!=0].copy()
        if not df_bubble.empty:
            bc_list = []
            for v in df_bubble["回撤深度%"].tolist():
                if v<=-30: bc_list.append("#3fb950")
                elif v<=-15: bc_list.append("#d29922")
                elif v<=-10: bc_list.append("#f0883e")
                else: bc_list.append("#8b949e")

            fig_bubble = go.Figure(go.Scatter(
                x=df_bubble["回撤深度%"],y=df_bubble["距Fib%"],
                mode="markers+text",text=df_bubble["代碼"],
                textposition="top center",textfont=dict(color="#e6edf3",size=10),
                marker=dict(size=14,color=bc_list,line=dict(color="#30363d",width=1)),
                hovertemplate="<b>%{text}</b><br>回撤:%{x:.1f}%<br>距Fib:%{y:.1f}%<extra></extra>"
            ))
            fig_bubble.add_hline(y=-3,line_dash="dot",line_color="#3fb950",
                annotation_text="⚡ 即將到支撐(3%內)",annotation_font_color="#3fb950")
            fig_bubble.add_vline(x=-30,line_dash="dash",line_color="#3fb950",
                annotation_text="深度超跌區",annotation_font_color="#3fb950")
            fig_bubble.update_layout(title="回撤深度 vs 距Fib支撐距離",height=480,
                paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(color="#e6edf3"),
                xaxis=dict(title="從52W高位回撤(%)",gridcolor="#21262d"),
                yaxis=dict(title="距下一Fib支撐(%)",gridcolor="#21262d"),
                margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig_bubble, use_container_width=True)

        st.divider()

        # 圖表三：熱力圖
        st.markdown("#### 🗺️ 回撤里程碑熱力圖")
        milestones = ["-10%","-20%","-25%","-30%","Fib23.6%","Fib38.2%","Fib50%","Fib61.8%","Fib78.6%"]
        heat_matrix = []
        for row in retrace_data:
            hi_v=row["_hi"]; lo_v=row["_lo"]; cv=row["_close"]
            fibs_heat = fib_levels(lo_v,hi_v)
            heat_row = []
            for m in milestones:
                if m.startswith("-"):
                    pct_v = float(m.replace("%",""))/100
                    heat_row.append(1 if cv<=hi_v*(1+pct_v) else 0)
                else:
                    fk = m.replace("Fib","").replace("%","")+"%"
                    fp = fibs_heat.get(fk,hi_v)
                    heat_row.append(1 if cv<=fp else 0)
            heat_matrix.append(heat_row)

        df_heat = pd.DataFrame(heat_matrix,
            index=[r["代碼"] for r in retrace_data],columns=milestones)
        fig_heat = go.Figure(go.Heatmap(
            z=df_heat.values,x=df_heat.columns.tolist(),y=df_heat.index.tolist(),
            colorscale=[[0,"#161b22"],[1,"#238636"]],showscale=False,
            text=[["✅" if v else "·" for v in row] for row in df_heat.values],
            texttemplate="%{text}",textfont=dict(size=14),
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>"))
        fig_heat.update_layout(height=max(350,len(retrace_data)*22),
            paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",font=dict(color="#e6edf3"),
            xaxis=dict(side="top",gridcolor="#21262d"),yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # 詳細數據表
        st.markdown("#### 📋 回撤詳細數據表")
        display_cols = ["代碼","現價","52W高","回撤深度%","下一回撤目標","目標價","距目標%",
                        "下一Fib支撐","Fib支撐價","距Fib%","已穿Fib"]
        st.dataframe(df_retrace[display_cols].sort_values("回撤深度%"),
                     use_container_width=True,hide_index=True)

        st.divider()

        # 即將到達Fib支撐警示
        st.markdown("#### ⚡ 即將到達斐波那契支撐位（距離 ≤ 5%）")
        alert_stocks = [r for r in retrace_data
                        if isinstance(r["距Fib%"],(int,float)) and -5<=r["距Fib%"]<=0]
        if alert_stocks:
            a_cols = st.columns(min(len(alert_stocks),4))
            for i,r in enumerate(alert_stocks):
                with a_cols[i%4]:
                    st.markdown(f"""
                    <div style='background:#0d2818;border:1px solid #238636;border-radius:10px;
                    padding:14px;text-align:center;margin:4px'>
                      <div style='font-size:1.3em;font-weight:bold;color:#3fb950'>{r["代碼"]}</div>
                      <div style='color:#8b949e;font-size:0.8em'>現價：{r["現價"]}</div>
                      <div style='color:#e6edf3'>Fib {r["下一Fib支撐"]} 支撐</div>
                      <div style='color:#3fb950;font-weight:bold'>支撐價：{r["Fib支撐價"]}</div>
                      <div style='color:#d29922;font-size:0.85em'>距離：{r["距Fib%"]:+.1f}%</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("目前沒有股票即將到達斐波那契支撐位（5%範圍內）。")

# ════════════ TAB 4: 技術圖表 ═════════════════════════════════════════════════
with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart = st.text_input("輸入股票代碼","AAPL",key="chart_tk").upper()
    period_map = {"1個月":"1mo","3個月":"3mo","6個月":"6mo","1年":"1y","2年":"2y"}
    period_sel = st.radio("時間範圍",list(period_map.keys()),index=3,horizontal=True)

    with st.spinner("載入圖表..."):
        df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])

    if df_ch is not None and len(df_ch)>30:
        close_ch = df_ch["close"]
        rsi_ch   = calc_rsi(close_ch)
        rsi_w_ch = calc_rsi(close_ch,70)
        macd_ch,sig_ch,hist_ch = calc_macd(close_ch)
        sma20  = close_ch.rolling(20).mean()
        sma60  = close_ch.rolling(60).mean()
        sma200 = close_ch.rolling(200).mean()
        bb_up  = sma20+2*close_ch.rolling(20).std()
        bb_dn  = sma20-2*close_ch.rolling(20).std()

        fig_tech = make_subplots(rows=7,cols=1,shared_xaxes=True,
    row_heights=[0.34,0.10,0.13,0.11,0.11,0.11,0.10],vertical_spacing=0.02)

        fig_tech.add_trace(go.Candlestick(
            x=df_ch.index,open=df_ch["open"],high=df_ch["high"],
            low=df_ch["low"],close=df_ch["close"],
            increasing_line_color="#3fb950",decreasing_line_color="#f85149",name="K線"),row=1,col=1)

        for ma,mc,nm in [(sma20,"#f0883e","MA20"),(sma60,"#58a6ff","MA60"),(sma200,"#bc8cff","MA200")]:
            fig_tech.add_trace(go.Scatter(x=df_ch.index,y=ma,mode="lines",
                line=dict(color=mc,width=1.2),name=nm),row=1,col=1)

        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=bb_up,mode="lines",
            line=dict(color="#8b949e",dash="dot",width=1),showlegend=False),row=1,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=bb_dn,mode="lines",
            line=dict(color="#8b949e",dash="dot",width=1),fill="tonexty",
            fillcolor="rgba(139,148,158,0.05)",showlegend=False),row=1,col=1)

        vol_colors = ["#3fb950" if df_ch["close"].iloc[i]>=df_ch["open"].iloc[i] else "#f85149"
                      for i in range(len(df_ch))]
        fig_tech.add_trace(go.Bar(x=df_ch.index,y=df_ch["volume"],
            marker_color=vol_colors,name="成交量",showlegend=False),row=2,col=1)

        # 日線RSI
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=rsi_ch,mode="lines",
            line=dict(color="#d29922",width=1.5),name="RSI日"),row=3,col=1)
        # 周線模擬RSI
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=rsi_w_ch,mode="lines",
            line=dict(color="#8957e5",width=1.5,dash="dot"),name="RSI周(模擬)"),row=3,col=1)
        for y,c in [(70,"#f85149"),(50,"#8b949e"),(30,"#3fb950")]:
            fig_tech.add_hline(y=y,line_dash="dash",line_color=c,row=3,col=1)

        # MACD
        hist_colors = ["#3fb950" if v>=0 else "#f85149" for v in hist_ch.fillna(0)]
        fig_tech.add_trace(go.Bar(x=df_ch.index,y=hist_ch,marker_color=hist_colors,
            name="MACD Hist",showlegend=False),row=4,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=macd_ch,mode="lines",
            line=dict(color="#58a6ff",width=1.2),name="MACD"),row=4,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=sig_ch,mode="lines",
            line=dict(color="#f0883e",width=1.2),name="Signal"),row=4,col=1)
        fig_tech.add_hline(y=0,line_dash="dash",line_color="#8b949e",row=4,col=1)

        # KDJ
        K_ch, D_ch, J_ch = calc_kdj(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=K_ch,mode="lines",
            line=dict(color="#3fb950",width=1.2),name="K"),row=5,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=D_ch,mode="lines",
            line=dict(color="#f85149",width=1.2),name="D"),row=5,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=J_ch,mode="lines",
            line=dict(color="#d29922",width=1.2),name="J"),row=5,col=1)
        for y in [20,80]:
            fig_tech.add_hline(y=y,line_dash="dash",line_color="#8b949e",row=5,col=1)
        
        # CCI
        cci_ch = calc_cci(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=cci_ch,mode="lines",
            line=dict(color="#79c0ff",width=1.2),name="CCI"),row=6,col=1)
        for y in [100,-100,0]:
            fig_tech.add_hline(y=y,line_dash="dash",line_color="#8b949e",row=6,col=1)

        # Williams %R
        wr_ch = calc_wr(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=wr_ch,mode="lines",
            line=dict(color="#ffa657",width=1.2),name="W%R"),row=7,col=1)
        fig_tech.add_hline(y=-20,line_dash="dash",line_color="#f85149",row=7,col=1)
        fig_tech.add_hline(y=-80,line_dash="dash",line_color="#3fb950",row=7,col=1)

        fig_tech.update_layout(
            title=f"{tk_chart} 技術分析（K線 / 成交量 / RSI日+周 / MACD / KDJ / CCI / W%R）",
            height=1050,paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="#161b22",bordercolor="#30363d"),
            margin=dict(l=10,r=10,t=50,b=10)
        )
        for i in range(1,8):
            fig_tech.update_xaxes(gridcolor="#21262d",row=i,col=1)
            fig_tech.update_yaxes(gridcolor="#21262d",row=i,col=1)

        st.plotly_chart(fig_tech, use_container_width=True)

        # 評分 + 指標總結
        short_s,mid_s,sigs = score_stock(df_ch)
        label,_ = signal_label(short_s,mid_s)
        close_v = float(df_ch["close"].iloc[-1])
        rsi_now = float(calc_rsi(df_ch["close"]).iloc[-1])
        rsi_w_now = float(calc_rsi(df_ch["close"],70).iloc[-1])

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("短線評分",f"{short_s}/100")
        c2.metric("中線評分",f"{mid_s}/100")
        c3.metric("信號",label)
        c4.metric("日線RSI",f"{rsi_now:.1f}")
        c5.metric("周線RSI(模擬)",f"{rsi_w_now:.1f}")
        c6.metric("目標+20% / 止損-8%",f"{round(close_v*1.2,3)} / {round(close_v*0.92,3)}")

        if sigs:
            st.markdown("**觸發指標：** " + " ｜ ".join(sigs))

        # 雙周期RSI共振說明
        if rsi_now<35 and 28<=rsi_w_now<=50:
            st.markdown(
                "<div style='background:#0d2818;border:1px solid #238636;border-radius:8px;"
                "padding:12px;margin-top:8px'>"
                "<b style='color:#3fb950'>⭐ 雙周期RSI共振信號</b><br>"
                "<span style='color:#e6edf3'>日線RSI已超賣，同時周線RSI處於30-50合理區間，"
                "代表這不只是短暫彈跳，而是有中線底部支撐的真實超賣機會。</span></div>",
                unsafe_allow_html=True)
        elif rsi_w_now>60 and rsi_now<35:
            st.markdown(
                "<div style='background:#1c1a00;border:1px solid #9e6a03;border-radius:8px;"
                "padding:12px;margin-top:8px'>"
                "<b style='color:#d29922'>⚠️ 注意：周線RSI仍強</b><br>"
                "<span style='color:#e6edf3'>日線雖然超賣，但周線RSI仍在60以上，"
                "代表下跌趨勢未完，日線超賣可能只是短暫彈跳，建議等周線RSI回落至50以下再操作。</span></div>",
                unsafe_allow_html=True)
    else:
        st.warning("找不到足夠數據，請確認代碼（港股用 0700.HK 格式）。")

st.divider()
st.caption("⚠️ 本系統僅供技術分析參考，不構成投資建議。數據來自 Yahoo Finance，存在延遲。投資涉及風險，買賣前請自行評估。")
