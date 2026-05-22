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

st.markdown("""
<style>
  [data-testid="stAppViewContainer"]{background:#0d1117;}
  [data-testid="stSidebar"]{background:#161b22;}
  h1,h2,h3,h4,h5,h6,p,label,.stMarkdown{color:#e6edf3!important;}
  .metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
    padding:16px;text-align:center;margin:4px;}
  .signal-buy{background:#0d2818;border:1px solid #238636;border-radius:8px;padding:12px;margin:4px;}
  .signal-watch{background:#1c1a00;border:1px solid #9e6a03;border-radius:8px;padding:12px;margin:4px;}
  .signal-none{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin:4px;}
  .stDataFrame{background:#161b22;}
  div[data-testid="metric-container"]{background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:10px;}
</style>
""", unsafe_allow_html=True)

HK_WATCHLIST = [
    # 藍籌 + 恒指權重
    "0700.HK",  # 騰訊
    "0005.HK",  # 匯豐控股
    "0939.HK",  # 建設銀行
    "1398.HK",  # 工商銀行
    "3988.HK",  # 中國銀行
    "0388.HK",  # 香港交易所
    "0066.HK",  # 港鐵
    "0003.HK",  # 香港中華煤氣
    "0002.HK",  # 中電控股
    "0016.HK",  # 新鴻基地產
    "0883.HK",  # 中國海洋石油
    "2318.HK",  # 中國平安
    "1299.HK",  # 友邦保險
    "0001.HK",  # 長和
    "9988.HK",  # 阿里巴巴
    "0175.HK",  # 吉利汽車
    "0027.HK",  # 銀河娛樂
    "2628.HK",  # 中國人壽
    "0011.HK",  # 恒生銀行
    "0688.HK",  # 中國海外發展
    # 科技 + 新經濟
    "3690.HK",  # 美團
    "9618.HK",  # 京東
    "0981.HK",  # 中芯國際
    "9999.HK",  # 網易
    "2382.HK",  # 舜宇光學
    "0291.HK",  # 華潤啤酒
    "1211.HK",  # 比亞迪
    "0267.HK",  # 中信股份
    "2688.HK",  # 新奧能源
    "0762.HK",  # 中國聯通
    "6862.HK",  # 海底撈
    "0960.HK",  # 龍湖集團
    "2020.HK",  # 安踏體育
    "1810.HK",  # 小米集團
    "1024.HK",  # 快手
]

US_WATCHLIST = [
    # 科技巨頭
    "AAPL",   # 蘋果
    "MSFT",   # 微軟
    "NVDA",   # 輝達
    "AMZN",   # 亞馬遜
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # 特斯拉
    "AVGO",   # 博通
    "ORCL",   # 甲骨文
    "ASML",   # 艾司摩爾
    # 半導體
    "AMD",    # 超微
    "QCOM",   # 高通
    "INTC",   # 英特爾
    "AMAT",   # 應用材料
    "LRCX",   # 泛林集團
    # 金融
    "JPM",    # 摩根大通
    "BAC",    # 美國銀行
    "GS",     # 高盛
    "MS",     # 摩根士丹利
    "BRK-B",  # 波克夏
    # 消費/零售
    "COST",   # Costco
    "WMT",    # 沃爾瑪
    "HD",     # 家得寶
    # 醫療/能源
    "JNJ",    # 強生
    "UNH",    # 聯合健康
    "PFE",    # 輝瑞
    "XOM",    # 埃克森美孚
    "NEE",    # NextEra能源
    # 你的指定觀察股
    "UBER",   # Uber
    "LITE",   # Lumentum
    "CLX",    # Clorox
    # 大市ETF
    "SPY",    # 標普ETF
    "QQQ",    # 納指ETF
    "SOXL",   # 半導體3xETF
    "IWM",    # 羅素2000ETF
]

MACRO_TICKERS = {
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "SPX": "^GSPC",
    "HSI": "^HSI",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "VHSI": "^VHSI",
    "HYG": "HYG",
    "USDHKD": "USDHKD=X",
}

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]
DROP_LEVELS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except Exception:
        return None


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_kdj(df, n=9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def calc_obv(df):
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def calc_wr(df, period=14):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def calc_mfi(df, period=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))


def get_52w_high(df):
    return float(df["high"].iloc[-252:].max()) if len(df) >= 252 else float(df["high"].max())


def fib_levels(swing_low, swing_high):
    diff = swing_high - swing_low
    return {f"{int(f*100)}%": round(swing_high - diff*f, 3) for f in FIB_LEVELS}


def drop_levels(high_price):
    return {f"-{int(d*100)}%": round(high_price * (1-d), 3) for d in DROP_LEVELS}


def score_stock(df):
    if df is None or len(df) < 60:
        return 0, 0, []
    close = df["close"]
    rsi_d = calc_rsi(close, 14)
    rsi_w = calc_rsi(close, 14 * 5)
    K, D, J = calc_kdj(df)
    macd, sig, hist = calc_macd(close)
    cci = calc_cci(df)
    obv = calc_obv(df)
    wr = calc_wr(df)
    mfi = calc_mfi(df)
    sma200 = close.rolling(200).mean()
    vol_ma = df["volume"].rolling(20).mean()

    def r(s):
        return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 50

    def rv(s):
        return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0

    rsi_val = r(rsi_d)
    rsi_w_val = r(rsi_w)
    k_val = r(K)
    d_val = r(D)
    cci_val = rv(cci)
    wr_val = rv(wr)
    mfi_val = r(mfi)
    macd_val = rv(macd)
    sig_val = rv(sig)
    obv_now = rv(obv)
    obv_prev = float(obv.iloc[-6]) if len(obv) >= 6 else obv_now
    sma200_v = rv(sma200)
    vol_now = float(df["volume"].iloc[-1])
    vol_ma_v = rv(vol_ma)
    close_v = float(close.iloc[-1])
    bias200 = (close_v - sma200_v) / sma200_v * 100 if sma200_v else 0

    short_score = 0
    short_sig = []
    mid_score = 0
    mid_sig = []

    if rsi_val < 30:
        short_score += 20
        short_sig.append(f"日RSI超賣({rsi_val:.0f})")
    elif rsi_val < 40:
        short_score += 10

    if k_val < 20 and d_val < 20:
        short_score += 20
        short_sig.append(f"KDJ超賣K({k_val:.0f})")
    elif k_val < 30:
        short_score += 10

    if macd_val > sig_val and macd_val < 0:
        short_score += 15
        short_sig.append("MACD低位金叉")

    if cci_val < -100:
        short_score += 15
        short_sig.append(f"CCI超賣({cci_val:.0f})")

    if wr_val < -85:
        short_score += 10
        short_sig.append(f"WR超賣({wr_val:.0f})")

    if mfi_val < 30:
        short_score += 10
        short_sig.append(f"MFI超賣({mfi_val:.0f})")

    if vol_ma_v and vol_now > vol_ma_v * 1.5:
        short_score += 10
        short_sig.append("量比爆量1.5x")

    if rsi_w_val < 35:
        mid_score += 25
        mid_sig.append(f"周RSI超賣({rsi_w_val:.0f})")
    elif rsi_w_val < 45:
        mid_score += 12

    if k_val < 25 and d_val < 25:
        mid_score += 20
        mid_sig.append("周KDJ低位")

    if bias200 < -20:
        mid_score += 20
        mid_sig.append(f"年線乖離{bias200:.1f}%")
    elif bias200 < -10:
        mid_score += 10

    if obv_now > obv_prev and close_v <= float(close.iloc[-6]):
        mid_score += 20
        mid_sig.append("OBV底背離吸籌")

    if cci_val < -150:
        mid_score += 15
        mid_sig.append(f"CCI極度超賣({cci_val:.0f})")

    signals = list(set(short_sig + mid_sig))
    return min(short_score, 100), min(mid_score, 100), signals


def signal_label(short, mid):
    if short >= 70 or mid >= 70:
        return "🔥 強烈撈底", "buy"
    if short >= 50 or mid >= 50:
        return "⭐️ 值得關注", "watch"
    if short >= 35 or mid >= 35:
        return "👁️ 觀察中", "observe"
    return "—", "none"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro():
    result = {}
    for name, tk in MACRO_TICKERS.items():
        df = fetch_ohlcv(tk, period="1y")
        if df is not None and len(df) > 1:
            c = float(df["close"].iloc[-1])
            p = float(df["close"].iloc[-2])
            chg = (c - p) / p * 100
            hi = float(df["high"].iloc[-252:].max()) if len(df) >= 252 else float(df["high"].max())
            lo = float(df["low"].iloc[-252:].min()) if len(df) >= 252 else float(df["low"].min())
            pct = (c - lo) / (hi - lo) * 100 if hi != lo else 50
            result[name] = {
                "val": c,
                "chg": chg,
                "pct": pct,
                "hi": hi,
                "lo": lo,
                "rsi": float(calc_rsi(df["close"]).iloc[-1]),
                "close_series": df["close"].tolist()[-60:],
                "vol_ratio": float(df["volume"].iloc[-1]) / float(df["volume"].rolling(20).mean().iloc[-1])
                if "volume" in df.columns and len(df) >= 20 else 1
            }
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def scan_stocks(tickers):
    rows = []
    for tk in tickers:
        df = fetch_ohlcv(tk, period="2y")
        if df is None or len(df) < 60:
            continue
        close_v = float(df["close"].iloc[-1])
        hi52 = get_52w_high(df)
        chg1d = (close_v - float(df["close"].iloc[-2])) / float(df["close"].iloc[-2]) * 100
        vol_ma = float(df["volume"].rolling(20).mean().iloc[-1]) or 1
        vol_rat = float(df["volume"].iloc[-1]) / vol_ma
        short_s, mid_s, sigs = score_stock(df)
        label, stype = signal_label(short_s, mid_s)
        swing_lo = float(df["low"].iloc[-126:].min())
        rows.append({
            "代碼": tk, "現價": round(close_v, 3),
            "1日漲跌%": round(chg1d, 2),
            "52周高": round(hi52, 3),
            "距高位%": round((close_v - hi52) / hi52 * 100, 1),
            "量比": round(vol_rat, 2),
            "短線分": short_s, "中線分": mid_s,
            "信號": label, "_type": stype,
            "觸發指標": "、".join(sigs) if sigs else "—",
            "_drop": drop_levels(hi52),
            "_fib": fib_levels(swing_lo, hi52),
            "_df": df
        })
    return rows


st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據：Yahoo Finance</p>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    market = st.radio("市場", ["🇭🇰 港股", "🇺🇸 美股", "📋 自選"], index=1)
    custom_input = ""
    if market == "📋 自選":
        custom_input = st.text_area("輸入代碼（每行一個）", "AAPL\nNVDA\n0700.HK\n9988.HK")
    st.divider()
    filter_sig = st.multiselect(
        "篩選信號",
        ["🔥 強烈撈底", "⭐️ 值得關注", "👁️ 觀察中", "—"],
        default=["🔥 強烈撈底", "⭐️ 值得關注"]
    )
    min_short = st.slider("最低短線分", 0, 100, 0)
    min_mid = st.slider("最低中線分", 0, 100, 0)
    st.divider()
    st.markdown("### 📌 評分說明")
    st.markdown("""
**短線分（0-100）**
適合 5-15 日操作。
- RSI / KDJ / CCI / WR / MFI 超賣。
- MACD 低位金叉。
- 量比爆量 1.5x。

**中線分（0-100）**
適合 1-3 個月操作。
- 周線 RSI < 35。
- 200日均線乖離 < -20%。
- OBV 底背離吸籌。
- 周線 CCI < -150。
""")

tab1, tab2, tab3, tab4 = st.tabs(["🌍 市場氣氛", "📊 個股掃描", "📐 回撤計算", "📈 技術圖表"])

with tab1:
    st.subheader("🌍 宏觀市場氣氛儀表板")

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_market_sentiment():
        result = {}
        macro_map = {
            "VIX": "^VIX",
            "VVIX": "^VVIX",
            "SPX": "^GSPC",
            "HSI": "^HSI",
            "DXY": "DX-Y.NYB",
            "US10Y": "^TNX",
            "VHSI": "^VHSI",
            "HYG": "HYG",
            "USDHKD": "USDHKD=X",
        }
        for name, tk in macro_map.items():
            try:
                df = yf.download(tk, period="1y", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if df.empty or len(df) < 5:
                    continue
                c = float(df["close"].iloc[-1])
                p = float(df["close"].iloc[-2])
                hi = float(df["high"].max())
                lo = float(df["low"].min())
                chg = (c - p) / p * 100
                pct = (c - lo) / (hi - lo) * 100 if hi != lo else 50
                vol = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0
                vol_ma = float(df["volume"].rolling(20).mean().iloc[-1]) if "volume" in df.columns else 1
                result[name] = {
                    "val": c,
                    "chg": chg,
                    "pct": pct,
                    "hi": hi,
                    "lo": lo,
                    "vol_ratio": vol / vol_ma if vol_ma else 1,
                    "close_series": df["close"].tolist()[-60:],
                    "rsi": float(calc_rsi(df["close"]).iloc[-1])
                }
            except Exception:
                pass

        sp500_sample = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
                        "JPM","BAC","XOM","JNJ","UNH","PG","AVGO","ORCL",
                        "HD","MA","V","COST","MRK"]
        oversold_count = 0
        overbought_count = 0
        for tk in sp500_sample:
            try:
                df = yf.download(tk, period="3mo", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if df.empty or len(df) < 20:
                    continue
                rsi_v = float(calc_rsi(df["close"]).iloc[-1])
                if rsi_v < 30:
                    oversold_count += 1
                if rsi_v > 70:
                    overbought_count += 1
            except Exception:
                pass
        result["breadth_oversold"] = oversold_count / len(sp500_sample) * 100
        result["breadth_overbought"] = overbought_count / len(sp500_sample) * 100
        return result

    with st.spinner("載入市場氣氛數據（約15秒）..."):
        mkt = fetch_market_sentiment()

    def safe_get(key, subkey="val", default=0):
        v = mkt.get(key, {})
        return v.get(subkey, default) if isinstance(v, dict) else default

    def calc_fear_score():
        score = 50
        vix_v = safe_get("VIX")
        vhsi_v = safe_get("VHSI")
        hyg_chg = safe_get("HYG", "chg")
        dxy_pct = safe_get("DXY", "pct")
        breadth = mkt.get("breadth_oversold", 0)
        hsi_rsi = safe_get("HSI", "rsi")
        spx_pct = safe_get("SPX", "pct")
        us10y_v = safe_get("US10Y")

        if vix_v >= 40:
            score -= 30
        elif vix_v >= 30:
            score -= 20
        elif vix_v >= 22:
            score -= 8
        elif vix_v <= 15:
            score += 15
        elif vix_v <= 18:
            score += 8

        if vhsi_v >= 35:
            score -= 20
        elif vhsi_v >= 25:
            score -= 10
        elif vhsi_v <= 18:
            score += 10

        if breadth >= 40:
            score -= 25
        elif breadth >= 25:
            score -= 12
        elif breadth >= 10:
            score -= 5
        elif breadth <= 5:
            score += 10

        if hyg_chg <= -1.5:
            score -= 10
        elif hyg_chg >= 0.5:
            score += 5

        if dxy_pct >= 80:
            score -= 8
        elif dxy_pct <= 30:
            score += 5

        if hsi_rsi <= 30:
            score -= 15
        elif hsi_rsi <= 40:
            score -= 5
        elif hsi_rsi >= 65:
            score += 10

        return max(0, min(100, score))

    fear_score = calc_fear_score()
    opportunity_score = 100 - fear_score

    st.markdown("### 📊 全球宏觀指標")
    kpi_items = [
        ("VIX", "😱 恐慌指數", "^VIX"),
        ("VVIX", "🌊 波動之波動", "^VVIX"),
        ("SPX", "🇺🇸 標普500", "^GSPC"),
        ("HSI", "🇭🇰 恒生指數", "^HSI"),
        ("US10Y", "🏦 美債10年息", "^TNX"),
        ("DXY", "💵 美元指數", "DX-Y.NYB"),
        ("HYG", "📉 高收益債", "HYG"),
        ("VHSI", "🇭🇰 港股波幅", "^VHSI"),
    ]
    cols_kpi = st.columns(8)
    for i, (key, label, _) in enumerate(kpi_items):
        val = safe_get(key)
        chg = safe_get(key, "chg")
        pct = safe_get(key, "pct")
        color = "#3fb950" if chg >= 0 else "#f85149"
        arrow = "▲" if chg >= 0 else "▼"
        with cols_kpi[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.1em">{label.split()[0]}</div>
              <div style="color:#8b949e;font-size:0.72em;line-height:1.2">{' '.join(label.split()[1:])}</div>
              <div style="font-size:1.15em;font-weight:bold;color:#e6edf3;margin:4px 0">{val:.2f}</div>
              <div style="color:{color};font-size:0.85em">{arrow} {chg:+.2f}%</div>
              <div style="color:#8b949e;font-size:0.7em">52W: {pct:.0f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎯 撈底機會總評")
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        if opportunity_score >= 70:
            gauge_color = "#3fb950"
            gauge_text = "🔥 極佳撈底視窗"
        elif opportunity_score >= 55:
            gauge_color = "#d29922"
            gauge_text = "⚠️ 謹慎撈底機會"
        elif opportunity_score >= 40:
            gauge_color = "#8b949e"
            gauge_text = "😐 市場中性"
        else:
            gauge_color = "#f85149"
            gauge_text = "😎 市場貪婪風險"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=opportunity_score,
            title={"text": f"撈底機會分<br><span style='font-size:0.7em;color:{gauge_color}'>{gauge_text}</span>"},
            number={"font": {"color": gauge_color, "size": 48}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 25], "color": "#1a1a2e"},
                    {"range": [25, 45], "color": "#1c1a00"},
                    {"range": [45, 65], "color": "#161b22"},
                    {"range": [65, 80], "color": "#0d2818"},
                    {"range": [80, 100], "color": "#0d3318"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": opportunity_score
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g_col2:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        levels = [
            ("80-100", "🟢 極佳撈底視窗", "市場極度恐慌，VIX高企，大量股票超賣，歷史上最強分批建倉時機"),
            ("60-79", "🟢 良好撈底機會", "市場明顯悲觀，技術面超賣，可開始輕倉試探"),
            ("40-59", "⚪ 市場中性", "正常波動範圍，選擇個股技術信號操作"),
            ("20-39", "🟠 市場樂觀", "情緒偏熱，不宜追高，等待回調機會"),
            ("0-19", "🔴 極度貪婪", "市場過熱，VIX極低，高位風險大，宜減倉觀望"),
        ]
        for score_range, label, desc in levels:
            low, high = map(int, score_range.split("-"))
            is_current = low <= opportunity_score <= high
            border = "2px solid #58a6ff" if is_current else "1px solid #30363d"
            bg = "#0d1f3c" if is_current else "#161b22"
            st.markdown(
                f"<div style='background:{bg};border:{border};border-radius:8px;padding:8px 12px;margin:4px 0'>"
                f"<span style='color:#e6edf3;font-weight:bold'>{label}</span> "
                f"<span style='color:#8b949e;font-size:0.75em'>({score_range}分)</span><br>"
                f"<span style='color:#8b949e;font-size:0.8em'>{desc}</span></div>",
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📡 美股 vs 港股 詳細氣氛")
    us_col, hk_col = st.columns(2)

    vix_v = safe_get("VIX")
    vvix_v = safe_get("VVIX")
    breadth = mkt.get("breadth_oversold", 0)
    dxy_v = safe_get("DXY")
    dxy_pct = safe_get("DXY", "pct")
    us10y_v = safe_get("US10Y")
    hyg_chg = safe_get("HYG", "chg")
    spx_rsi = safe_get("SPX", "rsi")

    vhsi_v = safe_get("VHSI")
    hsi_rsi_v = safe_get("HSI", "rsi")
    hsi_pct = safe_get("HSI", "pct")
    hsi_vol_r = safe_get("HSI", "vol_ratio")
    usdhkd_v = safe_get("USDHKD")

    hsi_close = mkt.get("HSI", {}).get("close_series", []) if isinstance(mkt.get("HSI"), dict) else []
    hsi_20bias = 0
    if hsi_close and len(hsi_close) >= 20:
        ma20 = sum(hsi_close[-20:]) / 20
        hsi_20bias = (hsi_close[-1] - ma20) / ma20 * 100 if ma20 else 0

    def indicator_row(label, value, desc, status_color, status_text, fmt=".2f"):
        return (
            f"<div style='background:#161b22;border-radius:8px;padding:10px 14px;"
            f"margin:5px 0;border-left:3px solid {status_color}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='color:#e6edf3;font-weight:bold'>{label}</span>"
            f"<span style='color:{status_color};font-weight:bold'>{format(value, fmt)}</span></div>"
            f"<div style='color:#8b949e;font-size:0.78em;margin-top:3px'>{desc}</div>"
            f"<div style='color:{status_color};font-size:0.78em'>{status_text}</div>"
            f"</div>"
        )

    with us_col:
        st.markdown("#### 🇺🇸 美股市場氣氛")
        vc = "#3fb950" if vix_v >= 30 else ("#d29922" if vix_v >= 20 else "#f85149")
        vt = "🔥 恐慌區，撈底機會" if vix_v >= 30 else ("⚠️ 警戒區，謹慎操作" if vix_v >= 20 else "😎 低波動，市場貪婪")
        st.markdown(indicator_row("😱 VIX 恐慌指數", vix_v,
                                  "市場預期30日波動率。> 30 = 恐慌撈底區；< 15 = 市場過度貪婪", vc, vt), unsafe_allow_html=True)

        vc2 = "#3fb950" if vvix_v >= 120 else ("#d29922" if vvix_v >= 100 else "#8b949e")
        vt2 = "🌊 極端不穩，恐慌頂部" if vvix_v >= 120 else ("⚠️ 波動加劇" if vvix_v >= 100 else "—正常")
        st.markdown(indicator_row("🌊 VVIX 波動率的波動率", vvix_v,
                                  "VIX本身的波動程度。> 120 代表恐慌極端、市場不確定性達頂峰", vc2, vt2), unsafe_allow_html=True)

        bc = "#3fb950" if breadth >= 40 else ("#d29922" if breadth >= 20 else "#8b949e")
        bt = f"🔥 {breadth:.0f}% 股票超賣，系統性拋售" if breadth >= 40 else (f"⚠️ {breadth:.0f}% 超賣" if breadth >= 20 else f"—{breadth:.0f}% 超賣，正常")
        st.markdown(indicator_row("📊 市場寬度（超賣佔比）", breadth,
                                  "S&P500抽樣中RSI<30的股票佔比。> 40% = 系統性拋售，歷史底部區域", bc, bt, ".0f"), unsafe_allow_html=True)

        dc = "#f85149" if dxy_pct >= 75 else ("#d29922" if dxy_pct >= 55 else "#3fb950")
        dt = "⚠️ 美元強勢，壓制股市" if dxy_pct >= 75 else ("中性" if dxy_pct >= 45 else "✅ 美元偏弱，利好股市")
        st.markdown(indicator_row("💵 美元指數 DXY", dxy_v,
                                  "美元強弱影響資金流向。美元走強通常壓制美股及新興市場；走弱則利好", dc, dt), unsafe_allow_html=True)

        tc = "#f85149" if us10y_v >= 4.5 else ("#d29922" if us10y_v >= 4.0 else "#3fb950")
        tt = "⚠️ 高息環境，估值承壓" if us10y_v >= 4.5 else ("中性" if us10y_v >= 3.5 else "✅ 低息，資金利好股市")
        st.markdown(indicator_row("🏦 美債10年息", us10y_v,
                                  "無風險回報率基準。快速上升壓制股票估值；下行代表資金回流股市", tc, tt), unsafe_allow_html=True)

        hc = "#3fb950" if hyg_chg <= -1.5 else ("#d29922" if hyg_chg <= -0.5 else "#8b949e")
        ht = "🔥 信用市場恐慌" if hyg_chg <= -1.5 else ("⚠️ 信用輕微壓力" if hyg_chg <= -0.5 else "—信用市場穩定")
        st.markdown(indicator_row("📉 高收益債ETF (HYG) 日變化", hyg_chg,
                                  "高收益債與國債的息差。HYG下跌=信用風險上升，通常先於股市反映恐慌", hc, ht, "+.2f"), unsafe_allow_html=True)

        sr_c = "#3fb950" if spx_rsi < 35 else ("#f85149" if spx_rsi > 65 else "#8b949e")
        sr_t = "🔥 大盤超賣" if spx_rsi < 35 else ("⚠️ 大盤超買" if spx_rsi > 65 else "—大盤RSI中性")
        st.markdown(indicator_row("📈 標普500 RSI", spx_rsi,
                                  "標普500指數本身的14日RSI。< 30 = 指數超賣；> 70 = 指數超買", sr_c, sr_t), unsafe_allow_html=True)

    with hk_col:
        st.markdown("#### 🇭🇰 港股市場氣氛")
        vh_c = "#3fb950" if vhsi_v >= 30 else ("#d29922" if vhsi_v >= 22 else "#f85149")
        vh_t = "🔥 港股恐慌，撈底機會" if vhsi_v >= 30 else ("⚠️ 港股波動加劇" if vhsi_v >= 22 else "😎 港股波動低")
        st.markdown(indicator_row("😱 VHSI 港股波幅指數", vhsi_v,
                                  "港版VIX，反映恒指期權隱含波動率。> 30 = 市場恐慌；底部常出現VHSI尖頂", vh_c, vh_t), unsafe_allow_html=True)

        hr_c = "#3fb950" if hsi_rsi_v < 30 else ("#f85149" if hsi_rsi_v > 70 else "#8b949e")
        hr_t = "🔥 日線超賣，短線撈底" if hsi_rsi_v < 30 else ("⚠️ 日線超買" if hsi_rsi_v > 70 else "—日線RSI中性")
        st.markdown(indicator_row("📊 恒指 RSI（日線）", hsi_rsi_v,
                                  "恒生指數14日RSI。< 30 = 短線超賣，可考慮短線撈底；> 70 = 超買不宜追", hr_c, hr_t), unsafe_allow_html=True)

        hp_c = "#3fb950" if hsi_pct <= 25 else ("#d29922" if hsi_pct <= 45 else "#f85149")
        hp_t = "🔥 52周低位區，底部機會" if hsi_pct <= 25 else ("⚠️ 中等水位" if hsi_pct <= 55 else "😎 高位區，注意風險")
        st.markdown(indicator_row("📍 恒指52周水位", hsi_pct,
                                  "恒指現價在過去52周高低點中的位置百分比。< 25% = 處於年度低位區", hp_c, hp_t, ".1f"), unsafe_allow_html=True)

        hv_c = "#3fb950" if hsi_vol_r >= 1.5 else ("#8b949e" if hsi_vol_r >= 0.8 else "#d29922")
        hv_t = "🔥 放量，資金介入" if hsi_vol_r >= 1.5 else ("—量能正常" if hsi_vol_r >= 0.8 else "⚠️ 縮量，觀望氣氛重")
        st.markdown(indicator_row("📦 恒指成交量比（vs20日均）", hsi_vol_r,
                                  "今日成交量與20日均量比值。> 1.5 = 放量，資金介入；底部爆量止跌是真底部信號", hv_c, hv_t, ".2f"), unsafe_allow_html=True)

        hb_c = "#3fb950" if hsi_20bias <= -8 else ("#f85149" if hsi_20bias >= 8 else "#8b949e")
        hb_t = "🔥 嚴重超跌，均值回歸機率高" if hsi_20bias <= -8 else ("⚠️ 偏高，注意回調" if hsi_20bias >= 8 else "—在均線附近")
        st.markdown(indicator_row("📐 恒指20日均線乖離率", hsi_20bias,
                                  "現價與20日均線偏離程度。< -8% = 嚴重超跌，歷史上均值回歸拉力極強", hb_c, hb_t, "+.1f"), unsafe_allow_html=True)

        hd_c = "#f85149" if usdhkd_v >= 7.83 else ("#d29922" if usdhkd_v >= 7.80 else "#3fb950")
        hd_t = "⚠️ 接近弱方兌換保證，資金外流" if usdhkd_v >= 7.83 else ("注意港元走弱" if usdhkd_v >= 7.80 else "✅ 港元穩定")
        st.markdown(indicator_row("💱 港元匯率 USDHKD", usdhkd_v,
                                  "港元兌美元。接近7.85弱方兌換保證 = 資金外流壓力大，不利港股；越低越穩定", hd_c, hd_t, ".4f"), unsafe_allow_html=True)

        try:
            df_3032 = yf.download("3032.HK", period="5d", auto_adjust=True, progress=False)
            if isinstance(df_3032.columns, pd.MultiIndex):
                df_3032.columns = df_3032.columns.get_level_values(0)
            df_3032.columns = [c.lower() for c in df_3032.columns]
            if not df_3032.empty and len(df_3032) >= 2:
                nw_chg = (float(df_3032["close"].iloc[-1]) - float(df_3032["close"].iloc[-2])) / float(df_3032["close"].iloc[-2]) * 100
                nw_c = "#3fb950" if nw_chg >= 0.3 else ("#f85149" if nw_chg <= -0.3 else "#8b949e")
                nw_t = "✅ 北水淨流入（看多港股）" if nw_chg >= 0.3 else ("⚠️ 北水淨流出" if nw_chg <= -0.3 else "—北水小幅變動")
                st.markdown(indicator_row("🌊 北水資金方向（港股通ETF替代）", nw_chg,
                                          "以港股通ETF(3032.HK)升跌估算北水動向。連續流入=內地資金撐盤，底部支撐強", nw_c, nw_t, "+.2f"), unsafe_allow_html=True)
        except Exception:
            pass

    st.divider()
    st.markdown("### 📈 指數走勢圖（近60日）")
    col_a, col_b = st.columns(2)
    for col, tk, title in [(col_a, "^GSPC", "🇺🇸 標普500"), (col_b, "^HSI", "🇭🇰 恒生指數")]:
        with col:
            df_idx = fetch_ohlcv(tk, period="3mo")
            if df_idx is not None and len(df_idx) > 5:
                sma20_idx = df_idx["close"].rolling(20).mean()
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_idx.index,
                    open=df_idx["open"], high=df_idx["high"],
                    low=df_idx["low"], close=df_idx["close"],
                    increasing_line_color="#3fb950",
                    decreasing_line_color="#f85149",
                    name="K線"
                ))
                fig.add_trace(go.Scatter(
                    x=df_idx.index, y=sma20_idx, mode="lines",
                    line=dict(color="#f0883e", width=1.5), name="MA20"
                ))
                fig.update_layout(
                    title=title, height=320,
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"),
                    xaxis_rangeslider_visible=False,
                    legend=dict(bgcolor="#161b22"),
                    margin=dict(l=5, r=5, t=40, b=5)
                )
                fig.update_xaxes(gridcolor="#21262d")
                fig.update_yaxes(gridcolor="#21262d")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    if market == "🇭🇰 港股":
        tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股":
        tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        tickers = raw if raw else US_WATCHLIST

    st.subheader(f"📊 個股掃描 — {market} ({len(tickers)} 隻)")
    with st.spinner(f"正在掃描 {len(tickers)} 隻股票..."):
        rows = scan_stocks(tuple(tickers))

    filtered = [r for r in rows if r["信號"] in filter_sig and r["短線分"] >= min_short and r["中線分"] >= min_mid]
    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 🔥 強烈撈底：{sum(1 for r in filtered if r['_type']=='buy')} 隻**")

    if filtered:
        df_plot = pd.DataFrame([{"代碼": r["代碼"], "短線分": r["短線分"], "中線分": r["中線分"]} for r in filtered])
        fig_bar = px.bar(
            df_plot.melt(id_vars="代碼", value_vars=["短線分", "中線分"]),
            x="代碼", y="value", color="variable", barmode="group",
            color_discrete_map={"短線分": "#388bfd", "中線分": "#3fb950"},
            height=260
        )
        fig_bar.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), legend_title="",
            margin=dict(l=5, r=5, t=10, b=5)
        )
        fig_bar.update_xaxes(gridcolor="#21262d")
        fig_bar.update_yaxes(gridcolor="#21262d", range=[0, 105])
        st.plotly_chart(fig_bar, use_container_width=True)

        for r in sorted(filtered, key=lambda x: x["短線分"] + x["中線分"], reverse=True):
            with st.expander(
                f"{r['信號']}  {r['代碼']}  現價 {r['現價']}  "
                f"({r['1日漲跌%']:+.1f}%)  ｜ 短線:{r['短線分']}  中線:{r['中線分']}"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", r["現價"])
                c2.metric("52周高", r["52周高"])
                c3.metric("距高位", f"{r['距高位%']}%")
                c4.metric("量比", r["量比"])
                st.markdown(f"**觸發指標：** {r['觸發指標']}")

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**📉 從52周高回撤位**")
                    drop_df = pd.DataFrame(list(r["_drop"].items()), columns=["回撤", "價位"])
                    drop_df["狀態"] = drop_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x - r["現價"]) / r["現價"] < 0.03 else ""
                    )
                    st.dataframe(drop_df, use_container_width=True, hide_index=True)
                with d2:
                    st.markdown("**🌀 斐波那契黃金回調位**")
                    fib_df = pd.DataFrame(list(r["_fib"].items()), columns=["比率", "價位"])
                    fib_df["狀態"] = fib_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x - r["現價"]) / r["現價"] < 0.03 else ""
                    )
                    st.dataframe(fib_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"**🎯 目標價 (+20%)：`{round(r['現價']*1.2, 3)}`  ｜  止損位 (-8%)：`{round(r['現價']*0.92, 3)}`**"
                )

    st.divider()
    st.subheader("📋 全部股票列表")
    if rows:
        table_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        st.dataframe(table_df.sort_values("短線分", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📐 回撤 & 斐波那契計算器")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        tk_input = st.text_input("股票代碼", "NVDA").upper()
    with col_in2:
        manual_high = st.number_input("手動輸入高位（0=自動抓取）", min_value=0.0, value=0.0)
    with col_in3:
        manual_low = st.number_input("手動輸入低位（0=自動抓取）", min_value=0.0, value=0.0)

    if st.button("🔍 計算", type="primary"):
        with st.spinner("抓取數據..."):
            df_c = fetch_ohlcv(tk_input, period="2y")
        if df_c is not None:
            close_now = float(df_c["close"].iloc[-1])
            hi = manual_high if manual_high > 0 else get_52w_high(df_c)
            lo = manual_low if manual_low > 0 else float(df_c["low"].iloc[-252:].min())

            st.markdown(
                f"### {tk_input}  現價：**{close_now:.3f}**  "
                f"｜  52周高：**{hi:.3f}**  ｜  52周低：**{lo:.3f}**"
            )

            ca, cb = st.columns(2)
            with ca:
                st.markdown("#### 📉 從高位回撤價位")
                rows_d = []
                for pct, price in drop_levels(hi).items():
                    diff = close_now - price
                    status = "✅ 已達到" if close_now <= price * 1.02 else f"還需跌 {abs(diff):.2f}"
                    rows_d.append({
                        "回撤幅度": pct,
                        "目標價位": price,
                        "現價距離": f"{diff:+.2f}",
                        "狀態": status
                    })
                st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

            with cb:
                st.markdown("#### 🌀 斐波那契黃金分割")
                rows_f = []
                fibs_v = fib_levels(lo, hi)
                for ratio, price in fibs_v.items():
                    diff = close_now - price
                    status = ("◀ 當前附近" if abs(diff) / close_now < 0.03
                              else ("✅ 已跌穿" if close_now < price
                              else f"距離 {diff:+.2f}"))
                    rows_f.append({
                        "比率": ratio,
                        "支撐價位": price,
                        "現價距離": f"{diff:+.2f}",
                        "狀態": status
                    })
                st.dataframe(pd.DataFrame(rows_f), use_container_width=True, hide_index=True)

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=df_c.index[-120:], y=df_c["close"].iloc[-120:],
                mode="lines", name="收盤價",
                line=dict(color="#58a6ff", width=2)
            ))
            line_styles = ["#3fb950", "#d29922", "#f85149", "#8957e5", "#79c0ff"]
            fibs_v = fib_levels(lo, hi)
            for i, (ratio, price) in enumerate(fibs_v.items()):
                fig_ret.add_hline(
                    y=price, line_dash="dash",
                    line_color=line_styles[i % len(line_styles)],
                    annotation_text=f"Fib {ratio} ({price:.2f})",
                    annotation_position="right"
                )
            fig_ret.add_hline(
                y=hi, line_color="#ff7b72", line_width=2,
                annotation_text=f"52W High {hi:.2f}",
                annotation_position="right"
            )
            fig_ret.update_layout(
                title=f"{tk_input} 斐波那契支撐圖", height=450,
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                margin=dict(l=10, r=130, t=40, b=10)
            )
            fig_ret.update_xaxes(gridcolor="#21262d")
            fig_ret.update_yaxes(gridcolor="#21262d")
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.error("找不到數據，港股請用 0700.HK 格式。")

    st.divider()

    st.subheader("📡 全觀察名單 回撤深度儀表板")
    st.markdown("自動掃描所有觀察股，計算每隻股票從52周高位的回撤深度，及**即將到達的下一個斐波那契支撐位**。")

    if market == "🇭🇰 港股":
        retrace_tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股":
        retrace_tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        retrace_tickers = raw if raw else US_WATCHLIST

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_retrace_data(tickers):
        results = []
        for tk in tickers:
            df = fetch_ohlcv(tk, period="2y")
            if df is None or len(df) < 30:
                continue
            try:
                close_v = float(df["close"].iloc[-1])
                hi52 = get_52w_high(df)
                lo52 = float(df["low"].iloc[-252:].min()) if len(df) >= 252 else float(df["low"].min())
                drop_pct = (close_v - hi52) / hi52 * 100

                fibs = fib_levels(lo52, hi52)
                fib_values = sorted(fibs.items(), key=lambda x: float(x[0].replace("%", "")), reverse=True)

                next_fib_label = "—"
                next_fib_price = None
                next_fib_pct_away = None
                passed_fibs = []

                for label, price in fib_values:
                    if close_v > price:
                        if next_fib_price is None:
                            next_fib_label = label
                            next_fib_price = price
                            next_fib_pct_away = (price - close_v) / close_v * 100
                    else:
                        passed_fibs.append(label)

                next_drop_label = "—"
                next_drop_price = None
                next_drop_pct_away = None
                for drop_pct_key, drop_price in drop_levels(hi52).items():
                    if close_v > drop_price:
                        if next_drop_price is None:
                            next_drop_label = drop_pct_key
                            next_drop_price = drop_price
                            next_drop_pct_away = (drop_price - close_v) / close_v * 100

                results.append({
                    "代碼": tk,
                    "現價": round(close_v, 3),
                    "52W高": round(hi52, 3),
                    "回撤深度%": round(drop_pct, 1),
                    "下一回撤目標": next_drop_label,
                    "目標價": round(next_drop_price, 3) if next_drop_price else "—",
                    "距目標%": round(next_drop_pct_away, 1) if next_drop_pct_away else 0,
                    "下一Fib支撐": next_fib_label,
                    "Fib支撐價": round(next_fib_price, 3) if next_fib_price else "—",
                    "距Fib%": round(next_fib_pct_away, 1) if next_fib_pct_away else 0,
                    "已穿Fib": ", ".join(passed_fibs) if passed_fibs else "—",
                    "_hi": hi52,
                    "_lo": lo52,
                    "_close": close_v,
                })
            except Exception:
                continue
        return results

    with st.spinner("掃描回撤數據..."):
        retrace_data = fetch_retrace_data(tuple(retrace_tickers))

    if retrace_data:
        df_retrace = pd.DataFrame(retrace_data)

        # ── 圖表一：回撤深度橫向條形圖 ──────────────────────────────────────
        st.markdown("#### 📉 各股從52周高位回撤深度")
        st.caption("橫條越長代表從高位跌得越深。🟢 綠色 ≥30% 超跌，🟡 黃色 15-30%，🟠 橙色 10-15%。")

        df_sorted = df_retrace.sort_values("回撤深度%")
        bar_colors = []
        for v in df_sorted["回撤深度%"]:
            if v <= -30:
                bar_colors.append("#3fb950")
            elif v <= -15:
                bar_colors.append("#d29922")
            elif v <= -10:
                bar_colors.append("#f0883e")
            else:
                bar_colors.append("#8b949e")

        fig_bar_ret = go.Figure(go.Bar(
            x=df_sorted["回撤深度%"],
            y=df_sorted["代碼"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in df_sorted["回撤深度%"]],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
            hovertemplate="<b>%{y}</b><br>回撤: %{x:.1f}%<br>現價: %{customdata}<extra></extra>",
            customdata=df_sorted["現價"]
        ))
        fig_bar_ret.add_vline(x=-10, line_dash="dash", line_color="#f0883e",
                               annotation_text="-10%", annotation_font_color="#f0883e")
        fig_bar_ret.add_vline(x=-20, line_dash="dash", line_color="#d29922",
                               annotation_text="-20%", annotation_font_color="#d29922")
        fig_bar_ret.add_vline(x=-30, line_dash="dash", line_color="#3fb950",
                               annotation_text="-30%", annotation_font_color="#3fb950")
        fig_bar_ret.update_layout(
            height=max(400, len(df_sorted) * 22),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d", zeroline=True,
                       zerolinecolor="#30363d", range=[-65, 5]),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=10, r=80, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar_ret, use_container_width=True)

        st.divider()

        # ── 圖表二：泡泡圖（移除 colorbar，改用 text label）────────────────
        st.markdown("#### 🌀 距離下一個斐波那契支撐位的距離")
        st.caption("X軸=回撤深度，Y軸=距下一個Fib支撐位的百分比。接近0%代表即將到達支撐位。")

        df_bubble = df_retrace[df_retrace["距Fib%"] != 0].copy()
        if not df_bubble.empty:
            bubble_colors = []
            for v in df_bubble["回撤深度%"].tolist():
                if v <= -30:
                    bubble_colors.append("#3fb950")
                elif v <= -15:
                    bubble_colors.append("#d29922")
                elif v <= -10:
                    bubble_colors.append("#f0883e")
                else:
                    bubble_colors.append("#8b949e")

            fig_bubble = go.Figure(go.Scatter(
                x=df_bubble["回撤深度%"],
                y=df_bubble["距Fib%"],
                mode="markers+text",
                text=df_bubble["代碼"],
                textposition="top center",
                textfont=dict(color="#e6edf3", size=10),
                marker=dict(
                    size=14,
                    color=bubble_colors,
                    line=dict(color="#30363d", width=1)
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "回撤深度: %{x:.1f}%<br>"
                    "距Fib支撐: %{y:.1f}%<br>"
                    "<extra></extra>"
                )
            ))
            fig_bubble.add_hline(
                y=-3, line_dash="dot", line_color="#3fb950",
                annotation_text="⚡ 即將到達支撐位(3%內)",
                annotation_font_color="#3fb950"
            )
            fig_bubble.add_vline(
                x=-30, line_dash="dash", line_color="#3fb950",
                annotation_text="深度超跌區",
                annotation_font_color="#3fb950"
            )
            fig_bubble.update_layout(
                title="回撤深度 vs 距斐波那契支撐位距離",
                height=480,
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                xaxis=dict(title="從52W高位回撤 (%)", gridcolor="#21262d"),
                yaxis=dict(title="距下一個Fib支撐 (%)", gridcolor="#21262d"),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

        st.divider()

        # ── 圖表三：熱力圖 ────────────────────────────────────────────────────
        st.markdown("#### 🗺️ 回撤里程碑熱力圖")
        st.caption("每格代表該股票是否已達到對應的回撤/Fib里程碑。✅ 已達到 ／ · 未達到")

        milestones = ["-10%", "-20%", "-25%", "-30%", "Fib23.6%", "Fib38.2%", "Fib50%", "Fib61.8%", "Fib78.6%"]
        heat_matrix = []
        for row in retrace_data:
            hi_v = row["_hi"]
            lo_v = row["_lo"]
            cv = row["_close"]
            fibs_heat = fib_levels(lo_v, hi_v)
            heat_row = []
            for m in milestones:
                if m.startswith("-"):
                    pct = float(m.replace("%", "")) / 100
                    target = hi_v * (1 + pct)
                    heat_row.append(1 if cv <= target else 0)
                else:
                    fib_key = m.replace("Fib", "").replace("%", "") + "%"
                    fib_price = fibs_heat.get(fib_key, hi_v)
                    heat_row.append(1 if cv <= fib_price else 0)
            heat_matrix.append(heat_row)

        df_heat = pd.DataFrame(
            heat_matrix,
            index=[r["代碼"] for r in retrace_data],
            columns=milestones
        )

        fig_heat = go.Figure(go.Heatmap(
            z=df_heat.values,
            x=df_heat.columns.tolist(),
            y=df_heat.index.tolist(),
            colorscale=[[0, "#161b22"], [1, "#238636"]],
            showscale=False,
            text=[["✅" if v else "·" for v in row] for row in df_heat.values],
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>"
        ))
        fig_heat.update_layout(
            height=max(350, len(retrace_data) * 22),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            xaxis=dict(side="top", gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # ── 詳細數據表 ────────────────────────────────────────────────────────
        st.markdown("#### 📋 回撤詳細數據表")
        display_cols = ["代碼", "現價", "52W高", "回撤深度%",
                        "下一回撤目標", "目標價", "距目標%",
                        "下一Fib支撐", "Fib支撐價", "距Fib%", "已穿Fib"]
        df_display = df_retrace[display_cols].sort_values("回撤深度%")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()

        # ── 即將到達Fib支撐警示 ───────────────────────────────────────────────
        st.markdown("#### ⚡ 即將到達斐波那契支撐位（距離 ≤ 5%）")
        alert_stocks = [
            r for r in retrace_data
            if isinstance(r["距Fib%"], (int, float)) and -5 <= r["距Fib%"] <= 0
        ]
        if alert_stocks:
            a_cols = st.columns(min(len(alert_stocks), 4))
            for i, r in enumerate(alert_stocks):
                with a_cols[i % 4]:
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
            st.info("目前沒有股票即將到達斐波那契支撐位（5% 範圍內）。")

with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart = st.text_input("輸入股票代碼", "AAPL", key="chart_tk").upper()
    period_map = {"1個月": "1mo", "3個月": "3mo", "6個月": "6mo", "1年": "1y", "2年": "2y"}
    period_sel = st.radio("時間範圍", list(period_map.keys()), index=3, horizontal=True)

    with st.spinner("載入圖表..."):
        df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])

    if df_ch is not None and len(df_ch) > 30:
        close_ch = df_ch["close"]
        rsi_ch = calc_rsi(close_ch)
        macd_ch, sig_ch, hist_ch = calc_macd(close_ch)
        sma20 = close_ch.rolling(20).mean()
        sma60 = close_ch.rolling(60).mean()
        sma200 = close_ch.rolling(200).mean()
        bb_mid = sma20
        bb_std = close_ch.rolling(20).std()
        bb_up = bb_mid + 2 * bb_std
        bb_dn = bb_mid - 2 * bb_std

        fig_tech = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.18, 0.18, 0.14],
            vertical_spacing=0.03
        )

        fig_tech.add_trace(go.Candlestick(
            x=df_ch.index, open=df_ch["open"], high=df_ch["high"],
            low=df_ch["low"], close=df_ch["close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            name="K線"), row=1, col=1)

        for ma, col_c, nm in [(sma20, "#f0883e", "MA20"), (sma60, "#58a6ff", "MA60"), (sma200, "#bc8cff", "MA200")]:
            fig_tech.add_trace(go.Scatter(
                x=df_ch.index, y=ma, mode="lines",
                line=dict(color=col_c, width=1.2), name=nm), row=1, col=1)

        fig_tech.add_trace(go.Scatter(
            x=df_ch.index, y=bb_up, mode="lines",
            line=dict(color="#8b949e", dash="dot", width=1), name="BB Upper",
            showlegend=False), row=1, col=1)

        fig_tech.add_trace(go.Scatter(
            x=df_ch.index, y=bb_dn, mode="lines",
            line=dict(color="#8b949e", dash="dot", width=1), name="BB Lower",
            fill="tonexty", fillcolor="rgba(139,148,158,0.05)",
            showlegend=False), row=1, col=1)

        vol_colors = ["#3fb950" if df_ch["close"].iloc[i] >= df_ch["open"].iloc[i] else "#f85149"
                      for i in range(len(df_ch))]
        fig_tech.add_trace(go.Bar(
            x=df_ch.index, y=df_ch["volume"],
            marker_color=vol_colors, name="成交量", showlegend=False), row=2, col=1)

        fig_tech.add_trace(go.Scatter(
            x=df_ch.index, y=rsi_ch, mode="lines",
            line=dict(color="#d29922", width=1.5), name="RSI"), row=3, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#f85149", row=3, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#3fb950", row=3, col=1)

        hist_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist_ch]
        fig_tech.add_trace(go.Bar(
            x=df_ch.index, y=hist_ch, marker_color=hist_colors,
            name="MACD Hist", showlegend=False), row=4, col=1)
        fig_tech.add_trace(go.Scatter(
            x=df_ch.index, y=macd_ch, mode="lines",
            line=dict(color="#58a6ff", width=1.2), name="MACD"), row=4, col=1)
        fig_tech.add_trace(go.Scatter(
            x=df_ch.index, y=sig_ch, mode="lines",
            line=dict(color="#f0883e", width=1.2), name="Signal"), row=4, col=1)

        fig_tech.update_layout(
            title=f"{tk_chart} 技術分析",
            height=750, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        for i in range(1, 5):
            fig_tech.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig_tech.update_yaxes(gridcolor="#21262d", row=i, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        short_s, mid_s, sigs = score_stock(df_ch)
        label, _ = signal_label(short_s, mid_s)
        close_v = float(df_ch["close"].iloc[-1])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("短線評分", f"{short_s}/100")
        c2.metric("中線評分", f"{mid_s}/100")
        c3.metric("信號", label)
        c4.metric("目標+20%", round(close_v * 1.2, 3))
        c5.metric("止損-8%", round(close_v * 0.92, 3))
        if sigs:
            st.markdown("**觸發指標：** " + " ｜ ".join(sigs))
    else:
        st.warning("找不到足夠數據，請確認代碼（港股用 0700.HK 格式）。")

st.divider()
st.caption("⚠️ 本系統僅供技術分析參考，不構成投資建議。數據來自 Yahoo Finance，存在延遲。")
