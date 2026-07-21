import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings, time, math, os, json, threading
warnings.filterwarnings("ignore")

# 可选套件
try:
    from futu import OpenQuoteContext, RET_OK, KLType
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(page_title="📈 撈底監察系統 Pro+", page_icon="📈", layout="wide")

# ── 自动刷新逻辑 ──
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
if elapsed >= 1800:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

# ── 样式 ──
st.markdown("""
<style>
  [data-testid="stAppViewContainer"]{background:#0d1117;}
  [data-testid="stSidebar"]{background:#161b22;}
  h1,h2,h3,h4,h5,h6,p,label,.stMarkdown{color:#e6edf3!important;}
  .metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;margin:4px;}
  .volume-alert{background:#1c2c1a;border:2px solid #3fb950;border-radius:10px;padding:16px;margin:8px 0;color:#e6edf3;}
  .signal-badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.85em;font-weight:bold;}
  .badge-buy{background:#0d2818;color:#3fb950;border:1px solid #3fb950;}
  .badge-watch{background:#1c2c1a;color:#d29922;border:1px solid #d29922;}
  .badge-observe{background:#161b22;color:#8b949e;border:1px solid #8b949e;}
  .badge-none{background:#161b22;color:#6e7681;border:1px solid #30363d;}
  .resonance-strong{color:#3fb950;font-weight:bold;}
  .resonance-medium{color:#d29922;font-weight:bold;}
  .resonance-weak{color:#8b949e;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

C_RED="#f85149"; C_GREEN="#3fb950"; C_ORANGE="#d29922"; C_BLUE="#58a6ff"; C_PURPLE="#bc8cff"; C_GREY="#8b949e"; C_BG="#0d1117"

HK_WATCHLIST = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","0066.HK","0003.HK","0002.HK","0016.HK","0883.HK","2318.HK","1299.HK","0001.HK","9988.HK","0175.HK","0027.HK","2628.HK","0011.HK","0688.HK","3690.HK","9618.HK","0981.HK","9999.HK","2382.HK","0291.HK","1211.HK","0267.HK","2688.HK","0762.HK","6862.HK","0960.HK","2020.HK","1810.HK","1024.HK"]
US_WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","ASML","AMD","QCOM","INTC","AMAT","LRCX","MU","SNDK","SKHY","JPM","BAC","GS","MS","BRK-B","COST","WMT","HD","JNJ","UNH","PFE","XOM","NEE","UBER","LITE","CLX","SPY","QQQ","SOXL","IWM","NFLX","SPCX"]
MACRO_TICKERS = {"VIX":"^VIX","VVIX":"^VVIX","SPX":"^GSPC","HSI":"^HSI","DXY":"DX-Y.NYB","US10Y":"^TNX","VHSI":"^VHSI","HYG":"HYG","USDHKD":"USDHKD=X"}
FIB_LEVELS=[0.236,0.382,0.500,0.618,0.786]; DROP_LEVELS=[0.10,0.20,0.25,0.30,0.35,0.40]

# 富途连线
@st.cache_resource
def init_futu():
    if not FUTU_AVAILABLE: return None
    try:
        ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        ret, _ = ctx.get_market_snapshot(['HK.00700'])
        return ctx if ret == RET_OK else None
    except: return None
quote_ctx = init_futu()

def to_futu(t): return f'HK.{t[:-3]}' if t.endswith('.HK') else (f'US.{t}' if t.isalpha() else t)

@st.cache_data(ttl=3600)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    if quote_ctx:
        try:
            ret, data, _ = quote_ctx.request_history_kline(to_futu(ticker), start=None, end=None, ktype=KLType.K_DAY, max_count=500, extended_time=False)
            if ret == RET_OK and not data.empty:
                df = data[['time_key','open','high','low','close','volume']].copy()
                df.rename(columns={'time_key':'date'}, inplace=True); df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True); df.columns = [c.lower() for c in df.columns]
                return df
        except: pass
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def fetch_multiple(tickers, period="2y"):
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_ohlcv, tk, period): tk for tk in tickers}
        for future in as_completed(futures): results[futures[future]] = future.result()
    return results

@st.cache_data(ttl=600)
def get_stock_info(ticker):
    name, pe, pb = ticker, None, None
    if quote_ctx:
        try:
            ret, data = quote_ctx.get_market_snapshot([to_futu(ticker)])
            if ret == RET_OK and not data.empty:
                row = data.iloc[0]
                pe = row.get('pe_ratio'); pb = row.get('pb_ratio'); name = row.get('stock_name', ticker)
                if pd.isna(pe): pe = None
        except: pass
    if name == ticker or pe is None:
        try:
            stock = yf.Ticker(ticker); info = stock.info
            name = info.get("shortName") or info.get("longName") or ticker
            if pe is None:
                pe = info.get("trailingPE") or info.get("forwardPE")
                if pe is None:
                    eps = info.get("trailingEps"); price = info.get("currentPrice")
                    if eps and price and eps>0: pe = price/eps
            pb = info.get("priceToBook")
        except: pass
    return name, pe, pb

# ── 技术指标（保持不变）──
def calc_rsi(series, period=14):
    delta = series.diff(); gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1+gain/loss.replace(0,np.nan)))

def calc_kdj(df, n=9):
    low_n=df["low"].rolling(n).min(); high_n=df["high"].rolling(n).max()
    rsv=(df["close"]-low_n)/(high_n-low_n).replace(0,np.nan)*100
    K=rsv.ewm(alpha=1/3, adjust=False).mean(); D=K.ewm(alpha=1/3, adjust=False).mean()
    return K, D, 3*K-2*D

def calc_macd(series, fast=12, slow=26, signal=9):
    ef=series.ewm(span=fast, adjust=False).mean(); es=series.ewm(span=slow, adjust=False).mean()
    m=ef-es; s=m.ewm(span=signal, adjust=False).mean(); return m, s, m-s

def calc_cci(df, period=20):
    tp=(df["high"]+df["low"]+df["close"])/3; sma=tp.rolling(period).mean()
    mad=tp.rolling(period).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-sma)/(0.015*mad.replace(0,np.nan))

def calc_obv(df): return (np.sign(df["close"].diff()).fillna(0)*df["volume"]).cumsum()

def calc_wr(df, period=14):
    hh=df["high"].rolling(period).max(); ll=df["low"].rolling(period).min()
    return -100*(hh-df["close"])/(hh-ll).replace(0,np.nan)

def calc_mfi(df, period=14):
    tp=(df["high"]+df["low"]+df["close"])/3; mf=tp*df["volume"]
    pos=mf.where(tp>tp.shift(1),0).rolling(period).sum(); neg=mf.where(tp<tp.shift(1),0).rolling(period).sum()
    return 100-(100/(1+pos/neg.replace(0,np.nan)))

def calc_cmf(df, period=20):
    mfv = ((2*df["close"]-df["low"]-df["high"])/(df["high"]-df["low"]).replace(0, np.nan)) * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)

def calc_vwap(df):
    tp = (df["high"]+df["low"]+df["close"])/3
    return (tp*df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

def get_52w_high(df): return float(df["high"].iloc[-252:].max()) if len(df)>=252 else float(df["high"].max())
def fib_levels(swing_low, swing_high): d=swing_high-swing_low; return {f"{int(f*100)}%": round(swing_high-d*f,3) for f in FIB_LEVELS}
def drop_levels(high_price): return {f"-{int(d*100)}%": round(high_price*(1-d),3) for d in DROP_LEVELS}
def volume_zscore(df, period=20):
    vol = df["volume"]; mean = vol.rolling(period).mean().iloc[-1]; std = vol.rolling(period).std().iloc[-1]
    return (vol.iloc[-1] - mean) / std if std!=0 else 0

# ── 技术形态识别 ──
def detect_double_bottom(df, lookback=60, tolerance=0.03):
    """简单双底检测，返回 True/False"""
    if len(df) < lookback: return False
    recent = df.iloc[-lookback:]
    lows = recent['low']
    # 找两个显著低点
    min_idx = lows.idxmin()
    min_val = lows.min()
    # 找第一个低点（排除最近10天）
    first_low = lows.iloc[:-10].min()
    if first_low > 0 and abs(first_low - min_val) / first_low < tolerance:
        return True
    return False

def detect_macd_bullish_divergence(df):
    """MACD柱背离：价格新低但MACD柱未新低"""
    close = df['close']
    macd, signal, hist = calc_macd(close)
    # 找最近两个低点
    recent = df.iloc[-60:]
    lows = recent['low']
    price_lows = lows.nsmallest(2)
    if len(price_lows) < 2: return False
    # 比较对应MACD柱
    idx1, idx2 = price_lows.index[0], price_lows.index[1]
    if idx1 not in hist.index or idx2 not in hist.index: return False
    if hist.loc[idx2] > hist.loc[idx1] and close.iloc[-1] < close.loc[idx1]:
        return True
    return False

# ── 时间衰减函数 ──
def time_decay(df, indicator_series, days_back=5):
    """返回过去days_back天内指标信号的衰减加权和"""
    weight_sum = 0
    for i in range(days_back):
        idx = -1 - i
        if abs(idx) > len(indicator_series): break
        val = indicator_series.iloc[idx] if not pd.isna(indicator_series.iloc[idx]) else 50
        # 简单线性衰减，越近权重越大
        weight = 1 - (i / (days_back + 1))
        if val < 30:  # 超卖状态
            weight_sum += weight
    return weight_sum

# ── 改进的评分函数（加入时间衰减、形态加分、市场状态）──
def score_stock(df, market_state="neutral"):
    if df is None or len(df)<60: return 0,0,[],0,"無",0,0,0
    close=df["close"]; volume=df["volume"]
    rsi_d=calc_rsi(close,14); rsi_w=calc_rsi(close,70)
    K,D,_=calc_kdj(df); macd,sig,_=calc_macd(close)
    cci=calc_cci(df); obv=calc_obv(df); wr=calc_wr(df); mfi=calc_mfi(df)
    cmf=calc_cmf(df); vwap=calc_vwap(df)
    sma200=close.rolling(200).mean(); sma20=close.rolling(20).mean(); vol_ma20=volume.rolling(20).mean()
    def r(s): return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 50
    rsi_val=r(rsi_d); rsi_w_val=r(rsi_w); k_val=r(K); d_val=r(D)
    cci_val=r(cci); wr_val=r(wr); macd_val=r(macd); sig_val=r(sig)
    cmf_val=r(cmf); vwap_val=r(vwap); obv_now=r(obv)
    obv_prev=float(obv.iloc[-6]) if len(obv)>=6 else obv_now
    close_v=float(close.iloc[-1]); vol_z=volume_zscore(df)

    # 时间衰减因子（过去5天）
    decay_rsi = time_decay(df, rsi_d, 5)
    decay_kdj = time_decay(df, K, 5)  # 使用K值
    decay_cci = time_decay(df, cci, 5)
    decay_wr = time_decay(df, wr, 5)

    # 共振判断
    triggers=[]
    if rsi_val<30 or decay_rsi>1.5: triggers.append("RSI")
    if (k_val<20 and d_val<20) or decay_kdj>1.5: triggers.append("KDJ")
    if cci_val<-100 or decay_cci>1.5: triggers.append("CCI")
    if wr_val<-85 or decay_wr>1.5: triggers.append("W%R")
    oversold_count=len(triggers)
    if oversold_count>=3: resonance="強"; mult=1.5
    elif oversold_count==2: resonance="中"; mult=1.2
    elif oversold_count==1: resonance="弱"; mult=1.0
    else: resonance="無"; mult=0.8

    vol_confirm=1.0
    if vol_z>2.0 and close_v>vwap_val: vol_confirm=1.4
    elif vol_z>2.0 and close_v>float(df["open"].iloc[-1]): vol_confirm=1.2
    elif vol_z<-1.5 and close_v<sma20.iloc[-1]: vol_confirm=0.7

    cmf_bonus=5 if cmf_val>0.1 else (2 if cmf_val>0 else 0)
    vwap_bonus=3 if (close_v>vwap_val and vol_z>1.5) else 0

    short_score=0; short_sig=[]
    if "RSI" in triggers: short_score+=int(15*mult*vol_confirm)
    if "KDJ" in triggers: short_score+=int(15*mult*vol_confirm)
    if "CCI" in triggers: short_score+=int(10*mult*vol_confirm)
    if "W%R" in triggers: short_score+=int(8*mult*vol_confirm)
    if macd_val>sig_val and macd_val<0: short_score+=int(10*vol_confirm); short_sig.append("MACD低位金叉")
    if obv_now>obv_prev and close_v<=float(close.iloc[-6]): short_score+=int(10*vol_confirm); short_sig.append("OBV底背離")
    short_score+=cmf_bonus+vwap_bonus
    if vol_z>2.5: short_sig.append(f"🔥爆量(Z={vol_z:.1f})")
    elif vol_z>1.5: short_sig.append(f"📈放量(Z={vol_z:.1f})")
    if cmf_val>0.2: short_sig.append("💰CMF吸籌")
    elif cmf_val<-0.2: short_sig.append("⚠️CMF派發")

    # 形态加分
    if detect_double_bottom(df):
        short_score += 10
        short_sig.append("🕳️雙底形態")
    if detect_macd_bullish_divergence(df):
        short_score += 15
        short_sig.append("📉MACD底背離(強)")

    mid_score=0; mid_sig=[]; bias200=(close_v-sma200.iloc[-1])/sma200.iloc[-1]*100 if sma200.iloc[-1] else 0
    weekly_warning = rsi_w_val>60
    if rsi_w_val<35: mid_score+=20; mid_sig.append("周RSI超賣")
    elif rsi_w_val<45: mid_score+=10
    if bias200<-20: mid_score+=20; mid_sig.append("年線乖離>20%")
    elif bias200<-10: mid_score+=10
    if cci_val<-150: mid_score+=10; mid_sig.append("CCI極度超賣")
    if weekly_warning: mid_score=int(mid_score*0.6); mid_sig.append("⚠️周線仍強(小心假底)")

    # 市场状态调整
    if market_state == "bear_high_vol":
        short_score = int(short_score * 1.1)  # 熊市高波动，超卖更有效
        mid_score = int(mid_score * 1.1)
    elif market_state == "bull_low_vol":
        short_score = int(short_score * 0.9)  # 牛市低波动，超卖信号减弱

    signals=list(dict.fromkeys(short_sig+mid_sig))
    return min(short_score,100), min(mid_score,100), signals, oversold_count, resonance, round(cmf_val,3), round(vwap_val,2), round(vol_z,2)

def signal_label(short, mid):
    if short>=70 or mid>=70: return "🔥 強烈撈底","buy"
    if short>=50 or mid>=50: return "⭐️ 值得關注","watch"
    if short>=35 or mid>=35: return "👁️ 觀察中","observe"
    return "—","none"

def signal_badge(label):
    if label.startswith("🔥"): return "badge-buy"
    if label.startswith("⭐️"): return "badge-watch"
    if label.startswith("👁️"): return "badge-observe"
    return "badge-none"

@st.cache_data(ttl=1800)
def fetch_macro():
    result={}
    for name,tk in MACRO_TICKERS.items():
        try:
            df=fetch_ohlcv(tk, period="1y")
            if df is None or len(df)<5: continue
            c=float(df["close"].iloc[-1]); p=float(df["close"].iloc[-2])
            chg=(c-p)/p*100; hi=float(df["high"].max()); lo=float(df["low"].min())
            pct=(c-lo)/(hi-lo)*100 if hi!=lo else 50
            vol_ratio=1.0
            if "volume" in df.columns and len(df)>=20:
                vol_now=float(df["volume"].iloc[-1])
                vol_ma=df["volume"].rolling(20).mean().iloc[-1]
                if pd.notna(vol_ma) and vol_ma>0: vol_ratio=vol_now/vol_ma
            result[name]={"val":c,"chg":chg,"pct":pct,"hi":hi,"lo":lo,"rsi":float(calc_rsi(df["close"]).iloc[-1]),"close_series":df["close"].tolist()[-60:],"vol_ratio":vol_ratio}
        except: continue
    return result

# ── 市场状态分类器 ──
def classify_market_state():
    """根据SPY或HSI的走势和VIX分类"""
    try:
        spy = fetch_ohlcv("SPY", period="6mo")
        if spy is None: return "unknown", 0, 0
        close = spy['close']
        ret_60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100
        volatility = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100  # 年化波动率
        vix = fetch_macro().get("VIX", {}).get("val", 20)

        if ret_60 < -5 and vix > 25:
            return "bear_high_vol", ret_60, volatility
        elif ret_60 < -5 and vix <= 25:
            return "bear_low_vol", ret_60, volatility
        elif ret_60 > 5 and vix > 25:
            return "bull_high_vol", ret_60, volatility
        elif ret_60 > 5 and vix <= 25:
            return "bull_low_vol", ret_60, volatility
        else:
            return "neutral", ret_60, volatility
    except:
        return "unknown", 0, 0

def get_dynamic_weights(vix_val):
    if vix_val>=30: return {"tech":0.50,"val":0.25,"dd":0.10,"fund":0.15}
    elif vix_val>=25: return {"tech":0.45,"val":0.25,"dd":0.15,"fund":0.15}
    elif vix_val<=15: return {"tech":0.30,"val":0.40,"dd":0.15,"fund":0.15}
    return {"tech":0.40,"val":0.30,"dd":0.15,"fund":0.15}

SIGNAL_LOG_FILE = "signal_log.csv"
def log_signal(ticker, total_score, label, price, date):
    try: df_log = pd.read_csv(SIGNAL_LOG_FILE)
    except: df_log = pd.DataFrame(columns=["date","ticker","total_score","label","price"])
    df_log = pd.concat([df_log, pd.DataFrame([{"date":date,"ticker":ticker,"total_score":total_score,"label":label,"price":price}])], ignore_index=True)
    df_log.to_csv(SIGNAL_LOG_FILE, index=False)

# ── 背景数据引擎（简化版，使用Streamlit的定时刷新）──
def background_scanner():
    """可以在独立线程中定时运行，但这里我们只是封装以备后用"""
    # 实际使用时，可以用schedule库定时调用
    pass

# ── 风险管理计算 ──
def calculate_position(price, stop_loss, account_size=100000, risk_pct=0.02):
    """计算建议股数"""
    risk_amount = account_size * risk_pct
    per_share_risk = abs(price - stop_loss)
    if per_share_risk <= 0: return 0
    shares = int(risk_amount / per_share_risk)
    return shares, risk_amount

# ── 生成PDF报告 ──
def generate_pdf_report(results, market_state, vix):
    if not PDF_AVAILABLE: return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="每日撈底報告", ln=1, align='C')
    pdf.cell(200, 10, txt=f"市場狀態: {market_state}  VIX: {vix:.1f}", ln=1)
    pdf.ln(10)
    for r in results[:10]:
        line = f"{r['ticker']} 價格{r['price']} 總分{r['total_score']}"
        pdf.cell(200, 10, txt=line, ln=1)
    return pdf.output(dest='S').encode('latin-1')

# ═══════════ HEADER ═════════════════════════════════════════
st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統 Pro+</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據：富途 + Yahoo Finance</p>", unsafe_allow_html=True)
st.divider()

# ═══════════ SIDEBAR ═══════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    market = st.radio("市場", ["🇭🇰 港股","🇺🇸 美股","📋 自選"], index=1)
    custom_input = ""
    if market == "📋 自選": custom_input = st.text_area("輸入代碼（每行一個）","AAPL\nNVDA\n0700.HK\n9988.HK")
    st.divider()
    filter_sig = st.multiselect("篩選信號", ["🔥 強烈撈底","⭐️ 值得關注","👁️ 觀察中","—"], default=["🔥 強烈撈底","⭐️ 值得關注"])
    min_short = st.slider("最低短線分",0,100,0)
    min_mid   = st.slider("最低中線分",0,100,0)
    resonance_filter = st.selectbox("🔍 共振強度篩選", ["全部","強共振","中共振","弱共振"], index=0)
    if quote_ctx: st.success("✅ 富途API 已連線")
    else: st.warning("⚠️ 富途API 未連線，使用 Yahoo Finance 數據")
    st.divider()
    st.markdown("### 📌 評分說明")
    st.markdown("""
    **短線分（0-100）** 5-15日操作
    - 雙周期RSI共振（日<35+周28-50）
    - KDJ/CCI/WR超賣（強共振加分）
    - MACD低位金叉
    - CMF/VWAP資金流向確認
    - 技術形態（雙底、MACD背離）
    **中線分（0-100）** 1-3個月操作
    - 周RSI < 35
    - 200日均線乖離 < -20%
    - OBV底背離吸籌
    """)
    # ═══════════ 第二部分：所有 Tab 頁面與進階功能 ═══════════════════════════════

# 獲取市場狀態
market_state, market_ret, market_vol = classify_market_state()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 市場氣氛", "📊 個股掃描", "📐 回撤計算",
    "📈 技術圖表", "🎯 四維撈底評分", "📋 信號追蹤與績效",
    "⚖️ 風險管理"
])

# ═══════════ TAB 1: 市場氣氛 ═══════════════════════════════
with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")
    macro_now = fetch_macro()
    vix_now = macro_now.get("VIX",{}).get("val",20) if isinstance(macro_now.get("VIX"),dict) else 20

    # 顯示市場狀態分類
    state_map = {
        "bear_high_vol": "🐻 熊市高波動",
        "bear_low_vol": "🐻 熊市低波動",
        "bull_high_vol": "🐂 牛市高波動",
        "bull_low_vol": "🐂 牛市低波動",
        "neutral": "😐 中性",
        "unknown": "❓ 無法判斷"
    }
    st.markdown(f"### 當前市場狀態：{state_map.get(market_state, market_state)}")
    st.caption(f"SPY 60日回報：{market_ret:.1f}% | 年化波動率：{market_vol:.1f}% | VIX：{vix_now:.1f}")
    if market_state == "bear_high_vol":
        st.info("📌 策略建議：熊市高波動下，超賣信號可信度較高，可分批撈底，嚴格止損。")
    elif market_state == "bull_low_vol":
        st.info("📌 策略建議：牛市低波動下，超賣信號可能只是短暫回調，降低撈底權重，以持倉為主。")

    st.divider()

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
            v = macro_now.get(key, {})
            val = v.get("val", 0) if isinstance(v, dict) else 0
            chg = v.get("chg", 0) if isinstance(v, dict) else 0
            pct = v.get("pct", 0) if isinstance(v, dict) else 0
            color = C_GREEN if chg>=0 else C_RED
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
    st.markdown("### 🎯 撈底機會總評")
    g_col1, g_col2, g_col3 = st.columns([1,1,1])

    vix_score = 0
    if vix_now >= 30: vix_score = 80
    elif vix_now >= 25: vix_score = 60
    elif vix_now <= 15: vix_score = 20
    else: vix_score = 40

    def make_gauge(score, title):
        if score>=70:   gc=C_GREEN; gt="🔥 極佳撈底視窗"
        elif score>=55: gc=C_ORANGE; gt="⚠️ 謹慎撈底機會"
        elif score>=40: gc=C_GREY; gt="😐 市場中性"
        else:           gc=C_RED; gt="😎 市場貪婪風險"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text":f"{title}<br><span style='font-size:0.7em;color:{gc}'>{gt}</span>"},
            number={"font":{"color":gc,"size":40},"suffix":"/100"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gc,"thickness":0.25},
                "bgcolor":"#161b22","bordercolor":"#30363d",
                "steps":[
                    {"range":[0,25], "color":"#1a1a2e"},
                    {"range":[25,45],"color":"#1c1a00"},
                    {"range":[45,65],"color":"#161b22"},
                    {"range":[65,80],"color":"#0d2818"},
                    {"range":[80,100],"color":"#0d3318"},
                ],
                "threshold":{"line":{"color":"#ffffff","width":3},
                             "thickness":0.8,"value":score}
            }
        ))
        fig.update_layout(height=260, paper_bgcolor=C_BG, font=dict(color="#e6edf3"), margin=dict(l=20,r=20,t=60,b=20))
        return fig

    with g_col1: st.plotly_chart(make_gauge(vix_score, "🇺🇸 美股撈底機會"), use_container_width=True)
    with g_col2: st.plotly_chart(make_gauge(vix_score, "🇭🇰 港股撈底機會"), use_container_width=True)
    with g_col3: st.plotly_chart(make_gauge(vix_score, "🌍 綜合評分"), use_container_width=True)

# ═══════════ TAB 2: 個股掃描（含時間衰減、形態信號、指標說明）══════════
with tab2:
    if market=="🇭🇰 港股":   tickers = HK_WATCHLIST
    elif market=="🇺🇸 美股": tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        tickers = raw if raw else US_WATCHLIST

    macro_now = fetch_macro()
    vix_now = macro_now.get("VIX",{}).get("val",20) if isinstance(macro_now.get("VIX"),dict) else 20

    if vix_now>=30:
        fc=C_GREEN; fi="🔥"
        fl=f"VIX {vix_now:.1f} 極度恐慌 — 只顯示最強信號（建議中線分≥60）"
        auto_min_mid=60
    elif vix_now>=25:
        fc=C_ORANGE; fi="⚠️"
        fl=f"VIX {vix_now:.1f} 高波動市場 — 建議只操作中線分≥50的股票"
        auto_min_mid=50
    elif vix_now<=15:
        fc=C_RED; fi="😎"
        fl=f"VIX {vix_now:.1f} 市場過度貪婪 — 注意追高風險，降低倉位"
        auto_min_mid=0
    else:
        fc=C_GREY; fi="😐"
        fl=f"VIX {vix_now:.1f} 市場中性 — 正常操作"
        auto_min_mid=0

    st.markdown(
        f"<div style='background:#161b22;border-left:4px solid {fc};border-radius:8px;padding:12px 16px;margin-bottom:12px'>"
        f"<span style='font-size:1.1em'>{fi}</span> "
        f"<span style='color:{fc};font-weight:bold'>市場氣氛濾網</span>: "
        f"<span style='color:#e6edf3'>{fl}</span></div>",
        unsafe_allow_html=True)

    effective_min_mid = max(min_mid, auto_min_mid)
    if auto_min_mid>0 and auto_min_mid>min_mid:
        st.caption(f"💡 VIX濾網已自動將最低中線分提升至 {auto_min_mid}（可在側邊欄手動覆蓋）")

    st.subheader(f"📊 個股掃描 — {market} ({len(tickers)} 隻)")

    with st.spinner(f"正在並行下載 {len(tickers)} 隻股票數據..."):
        data_map = fetch_multiple(tickers, period="2y")

    rows = []
    for tk in tickers:
        df = data_map.get(tk)
        if df is None or len(df)<60: continue
        # 傳入市場狀態
        short_s,mid_s,sigs,oversold_count,resonance,cmf_val,vwap_val,vol_z = score_stock(df, market_state)
        label,stype = signal_label(short_s,mid_s)
        close_v = float(df["close"].iloc[-1])
        hi52 = get_52w_high(df)
        chg1d = (close_v-float(df["close"].iloc[-2]))/float(df["close"].iloc[-2])*100
        vol_ma = float(df["volume"].rolling(20).mean().iloc[-1]) or 1
        vol_rat = float(df["volume"].iloc[-1])/vol_ma
        swing_lo = float(df["low"].iloc[-126:].min())
        rsi_w_v = float(calc_rsi(df["close"],70).iloc[-1])

        vix_env = "🔥 極度恐慌" if vix_now>=30 else ("⚠️ 高波動" if vix_now>=25 else ("😎 市場貪婪" if vix_now<=15 else "😐 中性"))

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
            "_df":df, "cmf":cmf_val, "vwap":vwap_val, "vol_z":vol_z,
            "resonance":resonance, "oversold_count":oversold_count
        })

    # ─── 低位大成交量提示 ──
    st.markdown("### 🚨 低位大成交量提示（距20日低點≤5% 且 Z-score≥2.0 且 收陽線）")
    volume_alerts = []
    for r in rows:
        df = r["_df"]
        if df is None or len(df) < 20: continue
        close_now = r["現價"]
        recent_low = float(df["low"].iloc[-20:].min())
        pct_above_low = (close_now - recent_low) / recent_low * 100
        is_positive = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
        if pct_above_low <= 5 and r["vol_z"] >= 2.0 and is_positive:
            volume_alerts.append({
                "代碼": r["代碼"], "現價": close_now,
                "近期低點": round(recent_low,3),
                "距低點%": round(pct_above_low,1),
                "量比": r["量比"], "信號": r["信號"],
                "Z-score": r["vol_z"]
            })
    if volume_alerts:
        st.success(f"🔥 發現 **{len(volume_alerts)}** 隻股票低位放量！")
        alert_cols = st.columns(min(len(volume_alerts), 3))
        for i, alert in enumerate(volume_alerts):
            with alert_cols[i % 3]:
                st.markdown(f"""
                <div class="volume-alert">
                    <b style="font-size:1.2em;">{alert['代碼']}</b><br>
                    現價：<b>{alert['現價']}</b><br>
                    近期低點：{alert['近期低點']}<br>
                    距低點：<b>{alert['距低點%']}%</b><br>
                    量比：<b style="color:#3fb950;">{alert['量比']}x</b><br>
                    Z-score：<b>{alert['Z-score']}</b><br>
                    信號：{alert['信號']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("目前沒有股票符合「低位 + 大成交量 + 收陽線」的條件。")
    st.divider()

    # ─── 共振篩選 ───
    if resonance_filter != "全部":
        rows = [r for r in rows if r["resonance"] == resonance_filter]
        st.caption(f"🔍 已篩選：**{resonance_filter}**（{len(rows)} 隻）")

    filtered = [r for r in rows
                if r["信號"] in filter_sig
                and r["短線分"]>=min_short
                and r["中線分"]>=effective_min_mid]
    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 🔥 強烈撈底：{sum(1 for r in filtered if r['_type']=='buy')} 隻**")

    if filtered:
        df_plot = pd.DataFrame([{"代碼":r["代碼"],"短線分":r["短線分"],"中線分":r["中線分"]} for r in filtered])
        fig_bar = px.bar(df_plot.melt(id_vars="代碼",value_vars=["短線分","中線分"]),
                         x="代碼",y="value",color="variable",barmode="group",
                         color_discrete_map={"短線分":"#388bfd","中線分":C_GREEN},height=260)
        fig_bar.update_layout(paper_bgcolor=C_BG,plot_bgcolor=C_BG,
                               font=dict(color="#e6edf3"),legend_title="",
                               margin=dict(l=5,r=5,t=10,b=5))
        fig_bar.update_xaxes(gridcolor="#21262d"); fig_bar.update_yaxes(gridcolor="#21262d",range=[0,105])
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
        sorted_filtered = sorted(filtered, key=sort_key_map[sort_by], reverse=sort_by not in ["距高位%"])
        for r in sorted_filtered:
            badge_class = signal_badge(r['信號'])
            weekly_warn_str = " ⚠️周線仍強" if r.get("周線RSI",50)>60 else ""

            resonance_class = ""
            if r['resonance'] == "強": resonance_class = "resonance-strong"
            elif r['resonance'] == "中": resonance_class = "resonance-medium"
            elif r['resonance'] == "弱": resonance_class = "resonance-weak"

            with st.expander(
                f"<span class='signal-badge {badge_class}'>{r['信號']}</span>  {r['代碼']}  現價 {r['現價']}  "
                f"({r['1日漲跌%']:+.1f}%) ｜ 短線:{r['短線分']} 中線:{r['中線分']} "
                f"<span class='{resonance_class}'>{r['resonance']}共振</span>{weekly_warn_str}"
            ):
                # ─────── 四大關鍵指標 + 詳細說明 ───────
                col_a, col_b, col_c, col_d = st.columns(4)

                # CMF
                with col_a:
                    st.metric("CMF (資金流)", f"{r['cmf']:.3f}")
                    if r['cmf'] > 0.2:
                        st.caption("💰 資金明顯流入 → 主力吸籌，可跟進")
                    elif r['cmf'] > 0:
                        st.caption("✅ 資金輕微流入 → 等待進一步確認")
                    elif r['cmf'] > -0.2:
                        st.caption("⚠️ 資金輕微流出 → 小心再跌一段")
                    else:
                        st.caption("❌ 資金明顯流出 (派發) → 暫不進場")

                # VWAP
                with col_b:
                    st.metric("VWAP (均價)", f"{r['vwap']:.2f}")
                    if r['現價'] > r['vwap']:
                        st.caption("📈 股價在均價之上 (強勢) → 買盤願追價")
                    else:
                        st.caption("📉 股價在均價之下 (弱勢) → 等站回 VWAP")

                # Z-score
                with col_c:
                    st.metric("成交量 Z-score", f"{r['vol_z']:.2f}")
                    if r['vol_z'] > 2.0:
                        st.caption("🔥 極度放量 → 恐慌拋售或主力吸籌")
                    elif r['vol_z'] > 1.0:
                        st.caption("📈 明顯放量 → 有資金關注")
                    elif r['vol_z'] < -1.5:
                        st.caption("❄️ 極度縮量 → 等待放量信號")
                    else:
                        st.caption("正常量")

                # 共振
                with col_d:
                    st.metric("共振指標數", f"{r['oversold_count']}/4")
                    triggers_desc = "RSI, KDJ, CCI, W%R"
                    if r['resonance'] == "強":
                        st.caption(f"⚡ 強共振 ({triggers_desc}) → 進場信心高")
                    elif r['resonance'] == "中":
                        st.caption(f"🔹 中共振 ({triggers_desc}) → 可小量試單")
                    elif r['resonance'] == "弱":
                        st.caption(f"▪️ 弱共振 ({triggers_desc}) → 等待更多確認")
                    else:
                        st.caption(f"— 無共振 ({triggers_desc}) → 暫觀望")

                st.markdown("---")
                st.markdown(f"**觸發指標：** {r['觸發指標']}")
                st.markdown(f"**VIX環境：** {r['VIX環境']}")

                # 量能形態
                df_vol = r["_df"]
                if df_vol is not None and len(df_vol)>=20:
                    vol_s    = df_vol["volume"]
                    vol_m20  = vol_s.rolling(20).mean()
                    vol_ma20v = float(vol_m20.iloc[-1])
                    is_shrinking = all(float(vol_s.iloc[-i])<vol_ma20v*0.85 for i in range(1,4))
                    is_expanding = float(vol_s.iloc[-1])>vol_ma20v*1.5
                    is_diverging = (float(vol_s.iloc[-1])<vol_ma20v*0.7 and
                                    float(df_vol["close"].iloc[-1])<float(df_vol["close"].iloc[-2]))
                    vol_tags = []
                    if is_shrinking: vol_tags.append("📉 縮量整理 (賣壓減輕)")
                    if is_expanding: vol_tags.append("📈 放量介入 (有資金進場)")
                    if is_diverging:  vol_tags.append("🔵 量縮價跌 (動能衰竭，可能見底)")
                    st.markdown("**量能形態：** " + (" ".join(vol_tags) if vol_tags else "正常"), unsafe_allow_html=True)

                    # 近5日量比圖
                    dates_5 = [str(d)[:10] for d in df_vol.index[-5:]]
                    vr_list = (vol_s.iloc[-5:]/vol_m20.iloc[-5:].replace(0,np.nan)).round(2).fillna(1).tolist()
                    bar_c5  = [C_GREEN if v>=1 else C_RED for v in vr_list]
                    fig_mini = go.Figure(go.Bar(
                        x=dates_5,y=vr_list,marker_color=bar_c5,
                        text=[f"{v:.1f}x" for v in vr_list],
                        textposition="outside",textfont=dict(color="#e6edf3",size=10)
                    ))
                    fig_mini.add_hline(y=1.0,line_dash="dash",line_color=C_GREY,annotation_text="均量")
                    fig_mini.add_hline(y=1.5,line_dash="dot",line_color=C_ORANGE,annotation_text="1.5x")
                    fig_mini.update_layout(title="近5日量比",height=200,
                        paper_bgcolor=C_BG,plot_bgcolor=C_BG,
                        font=dict(color="#e6edf3"),margin=dict(l=5,r=5,t=35,b=5),showlegend=False)
                    fig_mini.update_xaxes(gridcolor="#21262d"); fig_mini.update_yaxes(gridcolor="#21262d")
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

    # ─────── 指標說明（可折疊）───────
    with st.expander("📖 指標說明（點擊展開）"):
        st.markdown("""
        | 指標 | 說明 | 如何判斷 |
        |------|------|----------|
        | **量比** | 今日成交量 ÷ 過去20日均量 | >1.5 放量，>2.0 爆量，<0.7 縮量 |
        | **CMF** | 資金流向（-1 ~ 1） | >0.1 資金流入，< -0.1 資金流出（派發） |
        | **VWAP** | 成交量加權平均價（當日公平價） | 股價 > VWAP 強勢，< VWAP 弱勢 |
        | **Z-score** | 成交量異常程度（標準差倍數） | >2.0 極度放量，< -1.5 極度縮量 |
        | **共振** | RSI、KDJ、CCI、W%R 中幾個同時超賣 | ≥3 強共振，2 中共振，1 弱共振 |
        | **短線分** | 綜合超賣、量能、資金流、形態的短期評分 | >70 強烈撈底，>50 值得關注 |
        | **中線分** | 綜合中線超賣、輔助因子的中期評分 | >50 中期底部機會，>30 可追蹤 |
        """)

    # ─────── 總表（新增 CMF、VWAP、Z-score、共振）───────
    if rows:
        table_data = []
        for r in rows:
            table_data.append({
                "代碼": r["代碼"],
                "現價": r["現價"],
                "漲跌%": r["1日漲跌%"],
                "量比": r["量比"],
                "CMF": r["cmf"],
                "VWAP": r["vwap"],
                "Z-score": r["vol_z"],
                "共振": r["resonance"],
                "短線分": r["短線分"],
                "中線分": r["中線分"],
                "信號": r["信號"],
            })
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df.sort_values("短線分", ascending=False), use_container_width=True, hide_index=True)

# ═══════════ TAB 3: 回撤計算（維持不變）══════════════
with tab3:
    st.subheader("📐 回撤 & 斐波那契計算器")
    col_in1,col_in2,col_in3 = st.columns(3)
    with col_in1: tk_input = st.text_input("股票代碼","NVDA").upper()
    with col_in2: manual_high = st.number_input("手動輸入高位（0=自動）",min_value=0.0,value=0.0)
    with col_in3: manual_low = st.number_input("手動輸入低位（0=自動）",min_value=0.0,value=0.0)

    if st.button("🔍 計算",type="primary"):
        df_c = fetch_ohlcv(tk_input, period="2y")
        if df_c is not None:
            close_now = float(df_c["close"].iloc[-1])
            hi = manual_high if manual_high>0 else get_52w_high(df_c)
            lo = manual_low if manual_low>0 else float(df_c["low"].iloc[-252:].min())
            st.markdown(f"### {tk_input}  現價：**{close_now:.3f}**  ｜  52周高：**{hi:.3f}**  ｜  52周低：**{lo:.3f}**")
            ca,cb = st.columns(2)
            with ca:
                rows_d = [{"回撤幅度":pct,"目標價位":price,"現價距離":f"{close_now-price:+.2f}","狀態":"✅ 已達到" if close_now<=price*1.02 else f"還需跌 {abs(close_now-price):.2f}"} for pct,price in drop_levels(hi).items()]
                st.dataframe(pd.DataFrame(rows_d),use_container_width=True,hide_index=True)
            with cb:
                fibs_v = fib_levels(lo,hi)
                rows_f = [{"比率":ratio,"支撐價位":price,"現價距離":f"{close_now-price:+.2f}","狀態":"◀ 當前附近" if abs(close_now-price)/close_now<0.03 else ("✅ 已跌穿" if close_now<price else f"距離 {close_now-price:+.2f}")} for ratio,price in fibs_v.items()]
                st.dataframe(pd.DataFrame(rows_f),use_container_width=True,hide_index=True)
        else: st.error("找不到數據，港股請用 0700.HK 格式。")

# ═══════════ TAB 4: 技術圖表（維持不變）══════════════
with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart = st.text_input("輸入股票代碼","AAPL",key="chart_tk").upper()
    period_map = {"1個月":"1mo","3個月":"3mo","6個月":"6mo","1年":"1y","2年":"2y"}
    period_sel = st.radio("時間範圍",list(period_map.keys()),index=3,horizontal=True)
    df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])
    if df_ch is not None and len(df_ch)>30:
        close_ch = df_ch["close"]
        rsi_ch = calc_rsi(close_ch); rsi_w_ch = calc_rsi(close_ch,70)
        macd_ch,sig_ch,hist_ch = calc_macd(close_ch)
        sma20 = close_ch.rolling(20).mean(); sma60 = close_ch.rolling(60).mean(); sma200 = close_ch.rolling(200).mean()
        bb_up = sma20+2*close_ch.rolling(20).std(); bb_dn = sma20-2*close_ch.rolling(20).std()

        fig_tech = make_subplots(rows=7,cols=1,shared_xaxes=True,row_heights=[0.34,0.10,0.13,0.11,0.11,0.11,0.10],vertical_spacing=0.02)
        fig_tech.add_trace(go.Candlestick(x=df_ch.index,open=df_ch["open"],high=df_ch["high"],low=df_ch["low"],close=df_ch["close"],increasing_line_color=C_GREEN,decreasing_line_color=C_RED,name="K線"),row=1,col=1)
        for ma,mc,nm in [(sma20,"#f0883e","MA20"),(sma60,C_BLUE,"MA60"),(sma200,C_PURPLE,"MA200")]:
            fig_tech.add_trace(go.Scatter(x=df_ch.index,y=ma,mode="lines",line=dict(color=mc,width=1.2),name=nm),row=1,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=bb_up,mode="lines",line=dict(color=C_GREY,dash="dot",width=1),showlegend=False),row=1,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=bb_dn,mode="lines",line=dict(color=C_GREY,dash="dot",width=1),fill="tonexty",fillcolor="rgba(139,148,158,0.05)",showlegend=False),row=1,col=1)
        vol_colors = [C_GREEN if df_ch["close"].iloc[i]>=df_ch["open"].iloc[i] else C_RED for i in range(len(df_ch))]
        fig_tech.add_trace(go.Bar(x=df_ch.index,y=df_ch["volume"],marker_color=vol_colors,name="成交量",showlegend=False),row=2,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=rsi_ch,mode="lines",line=dict(color=C_ORANGE,width=1.5),name="RSI日"),row=3,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=rsi_w_ch,mode="lines",line=dict(color=C_PURPLE,width=1.5,dash="dot"),name="RSI周"),row=3,col=1)
        for y,c in [(70,C_RED),(50,C_GREY),(30,C_GREEN)]: fig_tech.add_hline(y=y,line_dash="dash",line_color=c,row=3,col=1)
        hist_colors = [C_GREEN if v>=0 else C_RED for v in hist_ch.fillna(0)]
        fig_tech.add_trace(go.Bar(x=df_ch.index,y=hist_ch,marker_color=hist_colors,name="MACD Hist",showlegend=False),row=4,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=macd_ch,mode="lines",line=dict(color=C_BLUE,width=1.2),name="MACD"),row=4,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=sig_ch,mode="lines",line=dict(color="#f0883e",width=1.2),name="Signal"),row=4,col=1)
        fig_tech.add_hline(y=0,line_dash="dash",line_color=C_GREY,row=4,col=1)
        K_ch,D_ch,J_ch = calc_kdj(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=K_ch,mode="lines",line=dict(color=C_GREEN,width=1.2),name="K"),row=5,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=D_ch,mode="lines",line=dict(color=C_RED,width=1.2),name="D"),row=5,col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=J_ch,mode="lines",line=dict(color=C_ORANGE,width=1.2),name="J"),row=5,col=1)
        for y in [20,80]: fig_tech.add_hline(y=y,line_dash="dash",line_color=C_GREY,row=5,col=1)
        cci_ch = calc_cci(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=cci_ch,mode="lines",line=dict(color="#79c0ff",width=1.2),name="CCI"),row=6,col=1)
        for y in [100,-100,0]: fig_tech.add_hline(y=y,line_dash="dash",line_color=C_GREY,row=6,col=1)
        wr_ch = calc_wr(df_ch)
        fig_tech.add_trace(go.Scatter(x=df_ch.index,y=wr_ch,mode="lines",line=dict(color="#ffa657",width=1.2),name="W%R"),row=7,col=1)
        fig_tech.add_hline(y=-20,line_dash="dash",line_color=C_RED,row=7,col=1)
        fig_tech.add_hline(y=-80,line_dash="dash",line_color=C_GREEN,row=7,col=1)
        fig_tech.update_layout(title=f"{tk_chart} 技術分析",height=1050,paper_bgcolor=C_BG,plot_bgcolor=C_BG,font=dict(color="#e6edf3"),xaxis_rangeslider_visible=False,legend=dict(bgcolor="#161b22"),margin=dict(l=10,r=10,t=50,b=10))
        for i in range(1,8): fig_tech.update_xaxes(gridcolor="#21262d",row=i,col=1); fig_tech.update_yaxes(gridcolor="#21262d",row=i,col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        short_s,mid_s,sigs,_,_,_,_,_ = score_stock(df_ch, market_state)
        label,_ = signal_label(short_s,mid_s)
        close_v = float(df_ch["close"].iloc[-1])
        rsi_now = float(calc_rsi(df_ch["close"]).iloc[-1]); rsi_w_now = float(calc_rsi(df_ch["close"],70).iloc[-1])
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("短線評分",f"{short_s}/100"); c2.metric("中線評分",f"{mid_s}/100")
        c3.metric("信號",label); c4.metric("日線RSI",f"{rsi_now:.1f}")
        c5.metric("周線RSI",f"{rsi_w_now:.1f}"); c6.metric("目標+20% / 止損-8%",f"{round(close_v*1.2,3)} / {round(close_v*0.92,3)}")
        if sigs: st.markdown("**觸發指標：** " + " ｜ ".join(sigs))
        if rsi_now<35 and 28<=rsi_w_now<=50: st.markdown(f"<div style='background:#0d2818;border:1px solid #238636;border-radius:8px;padding:12px;margin-top:8px'><b style='color:{C_GREEN}'>⭐ 雙周期RSI共振信號</b><br>代表中線底部支撐的真實超賣機會。</div>", unsafe_allow_html=True)
        elif rsi_w_now>60 and rsi_now<35: st.markdown(f"<div style='background:#1c1a00;border:1px solid #9e6a03;border-radius:8px;padding:12px;margin-top:8px'><b style='color:{C_ORANGE}'>⚠️ 注意：周線RSI仍強</b><br>建議等周線RSI回落至50以下再操作。</div>", unsafe_allow_html=True)
    else: st.warning("找不到足夠數據。")

# ═══════════ TAB 5: 四維撈底評分（含 PE 百分位、資金流）══════════
with tab5:
    st.subheader("🎯 四維撈底評分模型（附 PE 百分位 & 資金流向）")
    st.caption("技術超賣 40% + 估值低位 30% + 股價回調 15% + 資金訊號 15%（動態調整）")

    if market == "🇭🇰 港股": auto_tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股": auto_tickers = US_WATCHLIST
    else: auto_tickers = [x.strip().upper() for x in custom_input.split("\n") if x.strip()] or US_WATCHLIST

    st.info(f"📌 當前觀察市場：**{market}**（共 {len(auto_tickers)} 隻股票）")
    col_btn1, col_btn2 = st.columns([1,2])
    with col_btn1: scan_auto = st.button("🔄 掃描當前觀察名單", type="primary")
    with col_btn2:
        with st.expander("✏️ 或手動輸入代碼"):
            custom_input_5 = st.text_area("每行一個代碼","AAPL\nNVDA\n0700.HK",height=100,key="tab5_custom")
            scan_manual = st.button("掃描手動清單")
    scan_list = auto_tickers if scan_auto else ([x.strip().upper() for x in custom_input_5.split("\n") if x.strip()] if scan_manual else None)

    def technical_detail_score(df):
        if df is None or len(df)<60: return 0, {}
        close = df["close"]
        rsi = calc_rsi(close).iloc[-1]; K,D,_ = calc_kdj(df); k,d = K.iloc[-1], D.iloc[-1]
        cci = calc_cci(df).iloc[-1]; wr = calc_wr(df).iloc[-1]
        detail = {}
        if rsi < 30: detail["RSI(14)<30"] = (25, f"{rsi:.1f}")
        elif rsi < 40: detail["RSI(14)<40"] = (10, f"{rsi:.1f}")
        else: detail["RSI(14)"] = (0, f"{rsi:.1f}")
        if k < 20 and d < 20: detail["KDJ超賣"] = (25, f"K={k:.1f}, D={d:.1f}")
        elif k < 30: detail["KDJ偏低"] = (10, f"K={k:.1f}, D={d:.1f}")
        else: detail["KDJ"] = (0, f"K={k:.1f}, D={d:.1f}")
        if cci < -100: detail["CCI<-100"] = (25, f"{cci:.1f}")
        else: detail["CCI"] = (0, f"{cci:.1f}")
        if wr < -85: detail["W%R<-85"] = (25, f"{wr:.1f}")
        else: detail["W%R"] = (0, f"{wr:.1f}")
        total = sum(v[0] for v in detail.values())
        return min(total,100), detail

    def fund_flow_detail(df):
        if df is None or len(df)<20: return 0, {}
        close, volume = df["close"], df["volume"]
        mfi_series = calc_mfi(df)
        mfi_now = float(mfi_series.iloc[-1]) if not pd.isna(mfi_series.iloc[-1]) else 50
        ret = close.pct_change(); big_down = ret < -0.02
        if big_down.sum()==0: down_ratio = 0
        else:
            down_vol = volume[big_down].tail(5); avg_vol = volume.rolling(20).mean()
            valid = avg_vol[big_down] > 0
            down_ratio = (down_vol[valid]/avg_vol[big_down][valid]).mean() if valid.sum()>0 else 1.0
        detail = {}
        if down_ratio>0 and down_ratio<0.8: detail["大跌日縮量"] = (30, f"量比{down_ratio:.2f}")
        elif down_ratio>0 and down_ratio<1.1: detail["大跌日量比正常"] = (15, f"量比{down_ratio:.2f}")
        else: detail["大跌日放量"] = (0, f"量比{down_ratio:.2f}")
        if mfi_now<25: detail["MFI極低"] = (40, f"{mfi_now:.1f}")
        elif mfi_now<35: detail["MFI偏低"] = (20, f"{mfi_now:.1f}")
        else: detail["MFI中性"] = (0, f"{mfi_now:.1f}")
        if len(mfi_series)>=10:
            start = float(mfi_series.iloc[-10]); end = float(mfi_series.iloc[-1])
            if end>start+5: detail["MFI近期回升"] = (10, f"+{end-start:.1f}")
            elif end<start-5: detail["MFI近期下降"] = (0, f"{end-start:.1f}")
        total = sum(v[0] for v in detail.values())
        return min(total,100), detail

    def four_dimension_score(ticker):
        df = fetch_ohlcv(ticker, period="2y")
        if df is None or len(df)<60: return None
        name, pe, pb = get_stock_info(ticker)

        short_s, mid_s, sigs, _, _, _, _, _ = score_stock(df, market_state)

        tech_total, tech_detail = technical_detail_score(df)
        rsi_score = kdj_score = cci_score = wr_score = 0
        for k,v in tech_detail.items():
            if "RSI" in k: rsi_score = v[0]
            elif "KDJ" in k: kdj_score = v[0]
            elif "CCI" in k: cci_score = v[0]
            elif "W%R" in k: wr_score = v[0]

        val_score = 50; val_detail = "無數據(預設50分)"
        pe_percentile = None
        if pe is not None and pe > 0:
            hist_pe, perc, _ = get_pe_percentile(ticker) if 'get_pe_percentile' in globals() else (None,None,None)
            if perc is not None:
                pe_percentile = perc
                if perc < 0.1: val_score = 90; val_detail = f"PE {pe:.1f}，歷史百分位 {perc*100:.0f}% (極低)"
                elif perc < 0.25: val_score = 70; val_detail = f"PE {pe:.1f}，歷史百分位 {perc*100:.0f}% (偏低)"
                elif perc < 0.5: val_score = 40; val_detail = f"PE {pe:.1f}，歷史百分位 {perc*100:.0f}% (中等)"
                else: val_score = 10; val_detail = f"PE {pe:.1f}，歷史百分位 {perc*100:.0f}% (偏高)"
            else:
                if pe < 10: val_score=90; val_detail=f"PE {pe:.1f} (<10)"
                elif pe < 15: val_score=70; val_detail=f"PE {pe:.1f} (10~15)"
                elif pe < 20: val_score=40; val_detail=f"PE {pe:.1f} (15~20)"
                else: val_score=10; val_detail=f"PE {pe:.1f} (>20)"

        hi52 = get_52w_high(df)
        current_price = float(df["close"].iloc[-1])
        drawdown = (current_price-hi52)/hi52*100
        if drawdown<=-40: dd_score=90
        elif drawdown<=-30: dd_score=70
        elif drawdown<=-20: dd_score=50
        elif drawdown<=-10: dd_score=20
        else: dd_score=0

        fund_total, fund_detail = fund_flow_detail(df)
        downvol_score = mfi_score = mfi_trend_score = 0
        for k,v in fund_detail.items():
            if "跌" in k or "量比" in k: downvol_score = v[0]
            elif "MFI" in k and ("極低" in k or "偏低" in k or "中性" in k): mfi_score = v[0]
            elif "MFI近期" in k: mfi_trend_score = v[0]

        capital_flow = get_futu_capital_flow(ticker) if 'get_futu_capital_flow' in globals() else None
        capital_bonus = 0; capital_detail = ""
        if capital_flow is not None and not capital_flow.empty:
            latest = capital_flow.iloc[-1]
            inflow = latest.get('in_flow', 0) if 'in_flow' in latest else 0
            if inflow > 0:
                capital_bonus = 10; capital_detail = f"主力流入 {inflow:.0f}萬"
            else: capital_detail = "主力流出"
        fund_total = min(fund_total + capital_bonus, 100)

        macro = fetch_macro()
        vix_v = macro.get("VIX",{}).get("val",20) if isinstance(macro.get("VIX"),dict) else 20
        w = get_dynamic_weights(vix_v)
        total = w["tech"]*tech_total + w["val"]*val_score + w["dd"]*dd_score + w["fund"]*fund_total
        total = round(total,1)
        confidence = "高信心" if total>=80 else ("中等信心" if total>=60 else "低信心")

        return {
            "ticker":ticker,"name":name,"price":round(current_price,2),
            "total_score":total,"confidence":confidence,
            "tech_total":tech_total,"rsi_score":rsi_score,"kdj_score":kdj_score,
            "cci_score":cci_score,"wr_score":wr_score,
            "val_score":val_score,"val_detail":val_detail,
            "pe_percentile":pe_percentile,
            "drawdown":drawdown,"dd_score":dd_score,
            "fund_total":fund_total,"downvol_score":downvol_score,
            "mfi_score":mfi_score,"mfi_trend_score":mfi_trend_score,
            "capital_detail":capital_detail,
            "hi52":hi52,"weights":w,"vix":vix_v
        }

    if scan_list:
        results = []
        with st.spinner(f"正在掃描 {len(scan_list)} 隻股票（含 PE 百分位計算）..."):
            for tk in scan_list:
                res = four_dimension_score(tk)
                if res: results.append(res)
        if results:
            results.sort(key=lambda x: x["total_score"], reverse=True)

            for r in results:
                if r["total_score"] >= 70:
                    log_signal(r["ticker"], r["total_score"], r["confidence"], r["price"], datetime.now().strftime("%Y-%m-%d"))

            st.markdown("---")
            st.markdown("### 📋 四維評分詳細細項表（動態權重 + PE 百分位 + 資金流向）")
            w_disp = results[0]["weights"]
            st.caption(f"目前 VIX = {results[0]['vix']:.1f}，權重：技術 {w_disp['tech']:.0%} / 估值 {w_disp['val']:.0%} / 回調 {w_disp['dd']:.0%} / 資金 {w_disp['fund']:.0%}")
            detailed_data = []
            for r in results:
                detailed_data.append({
                    "代碼":r["ticker"],"名稱":r["name"],"現價":r["price"],
                    "總分":r["total_score"],"信心":r["confidence"],
                    "技術總分":r["tech_total"],"RSI得分":r["rsi_score"],
                    "KDJ得分":r["kdj_score"],"CCI得分":r["cci_score"],
                    "WR得分":r["wr_score"],
                    "估值得分":r["val_score"],
                    "PE百分位":f"{r['pe_percentile']*100:.0f}%" if r['pe_percentile'] is not None else "N/A",
                    "回撤%":r["drawdown"],"回撤得分":r["dd_score"],
                    "資金總分":r["fund_total"],"大跌量得分":r["downvol_score"],
                    "MFI得分":r["mfi_score"],"MFI趨勢得分":r["mfi_trend_score"],
                    "富途資金":r["capital_detail"] if r["capital_detail"] else "未取得"
                })
            df_detailed = pd.DataFrame(detailed_data)
            st.dataframe(df_detailed, use_container_width=True, hide_index=True)
            csv = df_detailed.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ 下載詳細評分 CSV", data=csv,
                file_name=f"四維撈底詳細評分_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

            cols = st.columns(4)
            avg_score = np.mean([r["total_score"] for r in results])
            best, worst = results[0], results[-1]
            cols[0].metric("📊 監察數量", f"{len(results)} 隻")
            cols[1].metric("🎯 平均總分", f"{avg_score:.1f}")
            cols[2].metric("🏆 最高分", f"{best['ticker']} {best['total_score']}")
            cols[3].metric("📉 最低分", f"{worst['ticker']} {worst['total_score']}")

            df_plot = pd.DataFrame({"代碼":[r["ticker"] for r in results],"總分":[r["total_score"] for r in results]})
            colors = [C_GREEN if s>=80 else (C_ORANGE if s>=60 else C_RED) for s in df_plot["總分"]]
            fig = go.Figure(go.Bar(x=df_plot["總分"], y=df_plot["代碼"], orientation="h", marker_color=colors, text=[f"{s:.1f}" for s in df_plot["總分"]], textposition="outside"))
            fig.update_layout(height=100+len(results)*35, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), margin=dict(l=10,r=50,t=10,b=10), xaxis=dict(range=[0,100], gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True)

            # PDF 報告按鈕
            if PDF_AVAILABLE:
                pdf_data = generate_pdf_report(results, market_state, vix_now)
                if pdf_data:
                    st.download_button("📄 下載今日撈底報告 (PDF)", data=pdf_data,
                        file_name=f"撈底報告_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
        else:
            st.warning("沒有找到有效數據。")
    else:
        st.info("👆 點擊「掃描當前觀察名單」即自動分析已選市場的所有股票。")
    st.divider()
    st.caption("估值百分位基於 Yahoo Finance 財務數據估算，僅供參考。")

# ═══════════ TAB 6: 信號追蹤與回測績效 ═══════════════════════════════
with tab6:
    st.subheader("📋 信號追蹤與績效回測")
    st.markdown("每當「四維撈底評分」總分 ≥ 70 時，系統會自動記錄信號，並在此追蹤後續績效。")
    try:
        df_log = pd.read_csv(SIGNAL_LOG_FILE)
        if not df_log.empty:
            df_log["date"] = pd.to_datetime(df_log["date"])
            df_log = df_log.sort_values("date", ascending=False)
            st.dataframe(df_log, use_container_width=True, hide_index=True)

            # 績效回測：計算每筆信號發出後 N 日的回報
            st.markdown("---")
            st.markdown("### 📊 信號績效回測")
            hold_days = st.selectbox("持有天數", [5, 10, 20, 30], index=1)

            backtest_results = []
            for _, row in df_log.iterrows():
                ticker = row['ticker']
                entry_date = row['date']
                entry_price = row['price']
                # 抓取該日期後的價格
                try:
                    df_bt = fetch_ohlcv(ticker, period="3mo")  # 快取可能已有
                    if df_bt is not None and len(df_bt) > hold_days:
                        # 找到進場日期後的價格
                        future_dates = df_bt.index[df_bt.index >= entry_date]
                        if len(future_dates) > hold_days:
                            exit_price = df_bt.loc[future_dates[hold_days], 'close']
                            ret = (exit_price - entry_price) / entry_price * 100
                            backtest_results.append({
                                "ticker": ticker,
                                "進場日": entry_date.strftime("%Y-%m-%d"),
                                "進場價": entry_price,
                                "出場價": round(exit_price, 2),
                                "回報%": round(ret, 2)
                            })
                except Exception as e:
                    continue

            if backtest_results:
                df_bt = pd.DataFrame(backtest_results)
                st.dataframe(df_bt, use_container_width=True, hide_index=True)

                win_rate = (df_bt["回報%"] > 0).mean() * 100
                avg_return = df_bt["回報%"].mean()
                st.metric("信號勝率", f"{win_rate:.1f}%")
                st.metric("平均回報", f"{avg_return:.2f}%")

                # 繪製累積回報曲線
                df_bt["累積回報"] = (1 + df_bt["回報%"]/100).cumprod() - 1
                fig_bt = go.Figure(go.Scatter(
                    x=df_bt.index, y=df_bt["累積回報"]*100,
                    mode='lines+markers', name='累積回報',
                    line=dict(color=C_GREEN, width=2)
                ))
                fig_bt.update_layout(title=f"信號累積回報 (持有{hold_days}天)", height=400,
                    paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"))
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("尚無足夠數據進行回測，請等待更多信號。")
        else:
            st.info("尚無信號記錄。")
    except FileNotFoundError:
        st.info("信號記錄檔案不存在，將在第一次掃描時建立。")

# ═══════════ TAB 7: 風險管理 ═══════════════════════════════
with tab7:
    st.subheader("⚖️ 風險管理與部位計算")
    st.markdown("輸入你的帳戶規模，選擇要計算的股票，系統會根據止損距離自動建議買入股數。")

    account_size = st.number_input("帳戶總值 (USD)", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆風險 (%)", 0.5, 5.0, 2.0) / 100

    # 讓用戶選擇股票
    ticker_input = st.text_input("輸入股票代碼", "AAPL").upper()
    if st.button("計算部位"):
        df = fetch_ohlcv(ticker_input, period="2y")
        if df is not None:
            close_v = float(df["close"].iloc[-1])
            # 使用斐波那契支撐作為建議止損
            hi52 = get_52w_high(df)
            lo52 = float(df["low"].rolling(252).min().iloc[-1]) if len(df) >= 252 else float(df["low"].min())
            fibs = fib_levels(lo52, hi52)
            # 找一個接近的支撐位作為止損
            support_level = None
            for ratio, price in sorted(fibs.items(), key=lambda x: float(x[0].replace("%", ""))):
                if price < close_v:
                    support_level = price
                    break
            if support_level is None:
                support_level = close_v * 0.92  # 預設 -8%

            shares, risk_amount = calculate_position(close_v, support_level, account_size, risk_pct)
            st.markdown(f"**{ticker_input}** 現價：{close_v:.2f}")
            st.markdown(f"建議止損：{support_level:.2f} (斐波那契支撐)")
            st.markdown(f"每筆風險金額：${risk_amount:.2f}")
            st.markdown(f"建議買入股數：**{shares}** 股")
            st.caption("以上計算僅供參考，請自行調整止損位。")
        else:
            st.error("找不到數據。")

st.divider()
st.caption("⚠️ 本系統僅供技術分析參考，不構成投資建議。數據來自富途 API 及 Yahoo Finance，可能存在延遲。")
