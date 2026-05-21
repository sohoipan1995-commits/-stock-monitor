import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="📈 撈底監察系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ──────────────────────────────────────────────────────────
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

# ── Constants ────────────────────────────────────────────────────────────────
HK_WATCHLIST = [
    "0700.HK","0005.HK","0939.HK","1398.HK","3988.HK",
    "0388.HK","0066.HK","0003.HK","0002.HK","0016.HK",
    "0883.HK","2318.HK","1299.HK","0001.HK","9988.HK",
    "0175.HK","0027.HK","2628.HK","0011.HK","0688.HK"
]
US_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVBO",
    "ORCL","ASML","UBER","NEE","LITE","CLX","JPM","BAC",
    "SPY","QQQ","SOXL","AMD"
]
MACRO_TICKERS = {"VIX":"^VIX","SPX":"^GSPC","HSI":"^HSI","DXY":"DX-Y.NYB","US10Y":"^TNX"}
FIB_LEVELS    = [0.236, 0.382, 0.500, 0.618, 0.786]
DROP_LEVELS   = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]

# ── Technical Indicators ─────────────────────────────────────────────────────
def safe_float(v):
    try: return float(v)
    except: return None

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

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_kdj(df, n=9):
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv    = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3*K - 2*D
    return K, D, J

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_cci(df, period=20):
    tp  = (df["high"] + df["low"] + df["close"]) / 3
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
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    mf  = tp * df["volume"]
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
    if df is None or len(df) < 60: return 0, 0, []
    close = df["close"]
    rsi_d = calc_rsi(close, 14)
    rsi_w = calc_rsi(close, 14*5)
    K, D, J = calc_kdj(df)
    macd, sig, hist = calc_macd(close)
    cci    = calc_cci(df)
    obv    = calc_obv(df)
    wr     = calc_wr(df)
    mfi    = calc_mfi(df)
    sma200 = close.rolling(200).mean()
    vol_ma = df["volume"].rolling(20).mean()

    def r(s):
        return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 50
    def rv(s):
        return float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else 0

    rsi_val   = r(rsi_d);    rsi_w_val = r(rsi_w)
    k_val     = r(K);        d_val     = r(D)
    cci_val   = rv(cci);     wr_val    = rv(wr)
    mfi_val   = r(mfi)
    macd_val  = rv(macd);    sig_val   = rv(sig)
    obv_now   = rv(obv);     obv_prev  = float(obv.iloc[-6]) if len(obv) >= 6 else obv_now
    sma200_v  = rv(sma200)
    vol_now   = float(df["volume"].iloc[-1])
    vol_ma_v  = rv(vol_ma)
    close_v   = float(close.iloc[-1])
    bias200   = (close_v - sma200_v) / sma200_v * 100 if sma200_v else 0

    short_score = 0; short_sig = []
    mid_score   = 0; mid_sig   = []

    # Short-term scoring
    if rsi_val < 30:
        short_score += 20; short_sig.append(f"日RSI超賣({rsi_val:.0f})")
    elif rsi_val < 40:
        short_score += 10
    if k_val < 20 and d_val < 20:
        short_score += 20; short_sig.append(f"KDJ超賣K({k_val:.0f})")
    elif k_val < 30:
        short_score += 10
    if macd_val > sig_val and macd_val < 0:
        short_score += 15; short_sig.append("MACD低位金叉")
    if cci_val < -100:
        short_score += 15; short_sig.append(f"CCI超賣({cci_val:.0f})")
    if wr_val < -85:
        short_score += 10; short_sig.append(f"WR超賣({wr_val:.0f})")
    if mfi_val < 30:
        short_score += 10; short_sig.append(f"MFI超賣({mfi_val:.0f})")
    if vol_ma_v and vol_now > vol_ma_v * 1.5:
        short_score += 10; short_sig.append("量比爆量1.5x")

    # Mid-term scoring
    if rsi_w_val < 35:
        mid_score += 25; mid_sig.append(f"周RSI超賣({rsi_w_val:.0f})")
    elif rsi_w_val < 45:
        mid_score += 12
    if k_val < 25 and d_val < 25:
        mid_score += 20; mid_sig.append("周KDJ低位")
    if bias200 < -20:
        mid_score += 20; mid_sig.append(f"年線乖離{bias200:.1f}%")
    elif bias200 < -10:
        mid_score += 10
    if obv_now > obv_prev and close_v <= float(close.iloc[-6]):
        mid_score += 20; mid_sig.append("OBV底背離吸籌")
    if cci_val < -150:
        mid_score += 15; mid_sig.append(f"CCI極度超賣({cci_val:.0f})")

    signals = list(set(short_sig + mid_sig))
    return min(short_score, 100), min(mid_score, 100), signals

def signal_label(short, mid):
    if short >= 70 or mid >= 70:  return "🔥 強烈撈底", "buy"
    if short >= 50 or mid >= 50:  return "⭐️ 值得關注", "watch"
    if short >= 35 or mid >= 35:  return "👁️ 觀察中",   "observe"
    return "—", "none"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro():
    result = {}
    for name, tk in MACRO_TICKERS.items():
        df = fetch_ohlcv(tk, period="1y")
        if df is not None and len(df) > 1:
            c   = float(df["close"].iloc[-1])
            p   = float(df["close"].iloc[-2])
            chg = (c - p) / p * 100
            hi  = float(df["high"].iloc[-252:].max())
            lo  = float(df["low"].iloc[-252:].min())
            pct = (c - lo) / (hi - lo) * 100 if hi != lo else 50
            result[name] = {"val": c, "chg": chg, "pct": pct, "hi": hi, "lo": lo}
    return result

@st.cache_data(ttl=3600, show_spinner=False)
def scan_stocks(tickers):
    rows = []
    for tk in tickers:
        df = fetch_ohlcv(tk, period="2y")
        if df is None or len(df) < 60: continue
        close_v = float(df["close"].iloc[-1])
        hi52    = get_52w_high(df)
        chg1d   = (close_v - float(df["close"].iloc[-2])) / float(df["close"].iloc[-2]) * 100
        vol_ma  = float(df["volume"].rolling(20).mean().iloc[-1]) or 1
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
            "_fib":  fib_levels(swing_lo, hi52),
            "_df": df
        })
    return rows

# ════════════════════════ UI ════════════════════════════════════════════════
st.markdown("<h1 style='color:#58a6ff;margin-bottom:0'>📈 撈底監察系統</h1>",
            unsafe_allow_html=True)
st.markdown(f"<p style='color:#8b949e'>最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} HKT ｜ 數據：Yahoo Finance</p>",
            unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    market = st.radio("市場", ["🇭🇰 港股", "🇺🇸 美股", "📋 自選"], index=1)
    custom_input = ""
    if market == "📋 自選":
        custom_input = st.text_area("輸入代碼（每行一個）",
                                    "AAPL\nNVDA\n0700.HK\n9988.HK")
    st.divider()
    filter_sig = st.multiselect("篩選信號",
        ["🔥 強烈撈底","⭐️ 值得關注","👁️ 觀察中","—"],
        default=["🔥 強烈撈底","⭐️ 值得關注"])
    min_short = st.slider("最低短線分", 0, 100, 0)
    min_mid   = st.slider("最低中線分", 0, 100, 0)
    st.divider()
    st.markdown("### 📌 評分說明")
    st.markdown("""
**短線分（0-100）**
適合 5-15 日操作
- RSI/KDJ/CCI/WR/MFI 超賣
- MACD 低位金叉
- 量比爆量 1.5x

**中線分（0-100）**
適合 1-3 個月操作
- 周線 RSI < 35
- 200日均線乖離 < -20%
- OBV 底背離吸籌
- 周線 CCI < -150
    """)

tab1, tab2, tab3, tab4 = st.tabs(["🌍 市場氣氛", "📊 個股掃描", "📐 回撤計算", "📈 技術圖表"])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🌍 宏觀市場氣氛")
    with st.spinner("載入宏觀數據..."):
        macro = fetch_macro()

    cols = st.columns(5)
    icons = {"VIX":"😱","SPX":"🇺🇸","HSI":"🇭🇰","DXY":"💵","US10Y":"🏦"}
    names = {"VIX":"恐慌指數 VIX","SPX":"標普500","HSI":"恒生指數",
             "DXY":"美元指數","US10Y":"美10年債息"}
    for i, (k, v) in enumerate(macro.items()):
        with cols[i]:
            color = "#3fb950" if v["chg"] >= 0 else "#f85149"
            arrow = "▲" if v["chg"] >= 0 else "▼"
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.6em">{icons.get(k,'📊')}</div>
              <div style="color:#8b949e;font-size:0.8em">{names.get(k,k)}</div>
              <div style="font-size:1.3em;font-weight:bold;color:#e6edf3">{v['val']:.2f}</div>
              <div style="color:{color}">{arrow} {v['chg']:+.2f}%</div>
              <div style="color:#8b949e;font-size:0.75em">52W水位 {v['pct']:.0f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    vix_val = macro.get("VIX", {}).get("val", 20)
    if   vix_val >= 35: sentiment = "🔥 極度恐慌 — 歷史上最佳撈底時機！可積極分批建倉"; s_col = "#3fb950"
    elif vix_val >= 25: sentiment = "⚠️ 市場恐慌 — 謹慎撈底，等待技術確認信號";         s_col = "#d29922"
    elif vix_val >= 18: sentiment = "😐 市場中性 — 正常波動，選擇性買入高分個股";         s_col = "#8b949e"
    else:               sentiment = "😎 市場貪婪 — 注意風險，等待回調機會";               s_col = "#f85149"

    st.markdown(
        f"<div style='background:#161b22;border-radius:10px;padding:20px;border-left:4px solid {s_col}'>"
        f"<b style='color:{s_col}'>VIX {vix_val:.1f}</b><br>"
        f"<span style='color:#e6edf3'>{sentiment}</span></div>",
        unsafe_allow_html=True)

    st.divider()
    col_a, col_b = st.columns(2)
    for col, tk, title in [(col_a, "^GSPC", "🇺🇸 標普500 (60日)"),
                            (col_b, "^HSI",  "🇭🇰 恒生指數 (60日)")]:
        with col:
            df_idx = fetch_ohlcv(tk, period="3mo")
            if df_idx is not None:
                fig = go.Figure(go.Candlestick(
                    x=df_idx.index,
                    open=df_idx["open"], high=df_idx["high"],
                    low=df_idx["low"],  close=df_idx["close"],
                    increasing_line_color="#3fb950",
                    decreasing_line_color="#f85149"))
                fig.update_layout(title=title, height=300,
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"),
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=5,r=5,t=40,b=5))
                fig.update_xaxes(gridcolor="#21262d")
                fig.update_yaxes(gridcolor="#21262d")
                st.plotly_chart(fig, use_container_width=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    if   market == "🇭🇰 港股": tickers = HK_WATCHLIST
    elif market == "🇺🇸 美股": tickers = US_WATCHLIST
    else:
        raw = [x.strip().upper() for x in custom_input.split("\n") if x.strip()]
        tickers = raw if raw else US_WATCHLIST

    st.subheader(f"📊 個股掃描 — {market} ({len(tickers)} 隻)")
    with st.spinner(f"正在掃描 {len(tickers)} 隻股票..."):
        rows = scan_stocks(tuple(tickers))

    filtered = [r for r in rows
                if r["信號"] in filter_sig
                and r["短線分"] >= min_short
                and r["中線分"] >= min_mid]

    st.markdown(f"**篩選後：{len(filtered)} 隻 ｜ 🔥 強烈撈底：{sum(1 for r in filtered if r['_type']=='buy')} 隻**")

    if filtered:
        df_plot = pd.DataFrame([{"代碼":r["代碼"],"短線分":r["短線分"],"中線分":r["中線分"]}
                                 for r in filtered])
        fig_bar = px.bar(df_plot.melt(id_vars="代碼", value_vars=["短線分","中線分"]),
                         x="代碼", y="value", color="variable", barmode="group",
                         color_discrete_map={"短線分":"#388bfd","中線分":"#3fb950"},
                         height=260)
        fig_bar.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                               font=dict(color="#e6edf3"), legend_title="",
                               margin=dict(l=5,r=5,t=10,b=5))
        fig_bar.update_xaxis(gridcolor="#21262d")
        fig_bar.update_yaxis(gridcolor="#21262d", range=[0,105])
        st.plotly_chart(fig_bar, use_container_width=True)

        for r in sorted(filtered, key=lambda x: x["短線分"]+x["中線分"], reverse=True):
            with st.expander(
                f"{r['信號']}  {r['代碼']}  現價 {r['現價']}  "
                f"({r['1日漲跌%']:+.1f}%)  ｜ 短線:{r['短線分']}  中線:{r['中線分']}"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("現價",    r["現價"])
                c2.metric("52周高",  r["52周高"])
                c3.metric("距高位",  f"{r['距高位%']}%")
                c4.metric("量比",    r["量比"])
                st.markdown(f"**觸發指標：** {r['觸發指標']}")

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**📉 從52周高回撤位**")
                    drop_df = pd.DataFrame(list(r["_drop"].items()), columns=["回撤","價位"])
                    drop_df["狀態"] = drop_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x - r["現價"]) / r["現價"] < 0.03 else "")
                    st.dataframe(drop_df, use_container_width=True, hide_index=True)
                with d2:
                    st.markdown("**🌀 斐波那契黃金回調位**")
                    fib_df = pd.DataFrame(list(r["_fib"].items()), columns=["比率","價位"])
                    fib_df["狀態"] = fib_df["價位"].apply(
                        lambda x: "◀ 當前附近" if abs(x - r["現價"]) / r["現價"] < 0.03 else "")
                    st.dataframe(fib_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"**🎯 目標價 (+20%)：`{round(r['現價']*1.2, 3)}`  "
                    f"｜  止損位 (-8%)：`{round(r['現價']*0.92, 3)}`**")

    st.divider()
    st.subheader("📋 全部股票列表")
    if rows:
        table_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
        st.dataframe(table_df.sort_values("短線分", ascending=False),
                     use_container_width=True, hide_index=True)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📐 回撤 & 斐波那契計算器")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1: tk_input    = st.text_input("股票代碼", "NVDA").upper()
    with col_in2: manual_high = st.number_input("手動輸入高位（0=自動）", min_value=0.0, value=0.0)
    with col_in3: manual_low  = st.number_input("手動輸入低位（0=自動）", min_value=0.0, value=0.0)

    if st.button("🔍 計算", type="primary"):
        with st.spinner("抓取數據..."):
            df_c = fetch_ohlcv(tk_input, period="2y")
        if df_c is not None:
            close_now = float(df_c["close"].iloc[-1])
            hi = manual_high if manual_high > 0 else get_52w_high(df_c)
            lo = manual_low  if manual_low  > 0 else float(df_c["low"].iloc[-252:].min())

            st.markdown(
                f"### {tk_input}  現價：**{close_now:.3f}**  "
                f"｜  52周高：**{hi:.3f}**  ｜  52周低：**{lo:.3f}**")

            ca, cb = st.columns(2)
            with ca:
                st.markdown("#### 📉 從高位回撤價位")
                rows_d = []
                for pct, price in drop_levels(hi).items():
                    diff   = close_now - price
                    status = "✅ 已達到" if close_now <= price*1.02 else f"還需跌 {abs(diff):.2f}"
                    rows_d.append({"回撤幅度":pct,"目標價位":price,
                                   "現價距離":f"{diff:+.2f}","狀態":status})
                st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

            with cb:
                st.markdown("#### 🌀 斐波那契黃金分割")
                rows_f = []
                fibs_v = fib_levels(lo, hi)
                for ratio, price in fibs_v.items():
                    diff   = close_now - price
                    status = ("◀ 當前附近" if abs(diff)/close_now < 0.03
                              else ("✅ 已跌穿" if close_now < price
                              else f"距離 {diff:+.2f}"))
                    rows_f.append({"比率":ratio,"支撐價位":price,
                                   "現價距離":f"{diff:+.2f}","狀態":status})
                st.dataframe(pd.DataFrame(rows_f), use_container_width=True, hide_index=True)

            # Fibonacci chart
            from plotly.subplots import make_subplots
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=df_c.index[-120:], y=df_c["close"].iloc[-120:],
                mode="lines", name="收盤價",
                line=dict(color="#58a6ff", width=2)))
            line_styles = ["#3fb950","#d29922","#f85149","#8957e5","#79c0ff"]
            for i, (ratio, price) in enumerate(fibs_v.items()):
                fig_ret.add_hline(y=price, line_dash="dash",
                    line_color=line_styles[i % len(line_styles)],
                    annotation_text=f"Fib {ratio} ({price:.2f})",
                    annotation_position="right")
            fig_ret.add_hline(y=hi, line_color="#ff7b72", line_width=2,
                annotation_text=f"52W High {hi:.2f}", annotation_position="right")
            fig_ret.update_layout(
                title=f"{tk_input} 斐波那契支撐圖", height=450,
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                margin=dict(l=10, r=120, t=40, b=10))
            fig_ret.update_xaxis(gridcolor="#21262d")
            fig_ret.update_yaxis(gridcolor="#21262d")
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.error("找不到數據，港股請用 0700.HK 格式。")

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 個股技術分析圖表")
    tk_chart   = st.text_input("輸入股票代碼", "AAPL", key="chart_tk").upper()
    period_map = {"1個月":"1mo","3個月":"3mo","6個月":"6mo","1年":"1y","2年":"2y"}
    period_sel = st.radio("時間範圍", list(period_map.keys()), index=3, horizontal=True)

    with st.spinner("載入圖表..."):
        df_ch = fetch_ohlcv(tk_chart, period=period_map[period_sel])

    if df_ch is not None and len(df_ch) > 30:
        close_ch = df_ch["close"]
        rsi_ch   = calc_rsi(close_ch)
        macd_ch, sig_ch, hist_ch = calc_macd(close_ch)
        sma20    = close_ch.rolling(20).mean()
        sma60    = close_ch.rolling(60).mean()
        sma200   = close_ch.rolling(200).mean()
        bb_mid   = sma20
        bb_std   = close_ch.rolling(20).std()
        bb_up    = bb_mid + 2*bb_std
        bb_dn    = bb_mid - 2*bb_std

        from plotly.subplots import make_subplots
        fig_tech = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                 row_heights=[0.5,0.18,0.18,0.14],
                                 vertical_spacing=0.03)
        fig_tech.add_trace(go.Candlestick(
            x=df_ch.index, open=df_ch["open"], high=df_ch["high"],
            low=df_ch["low"], close=df_ch["close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            name="K線"), row=1, col=1)
        for ma, col_c, nm in [(sma20,"#f0883e","MA20"),(sma60,"#58a6ff","MA60"),(sma200,"#bc8cff","MA200")]:
            fig_tech.add_trace(go.Scatter(x=df_ch.index, y=ma, mode="lines",
                line=dict(color=col_c, width=1.2), name=nm), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index, y=bb_up, mode="lines",
            line=dict(color="#8b949e", dash="dot", width=1), name="BB Upper",
            showlegend=False), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index, y=bb_dn, mode="lines",
            line=dict(color="#8b949e", dash="dot", width=1), name="BB Lower",
            fill="tonexty", fillcolor="rgba(139,148,158,0.05)",
            showlegend=False), row=1, col=1)
        vol_colors = ["#3fb950" if df_ch["close"].iloc[i] >= df_ch["open"].iloc[i]
                      else "#f85149" for i in range(len(df_ch))]
        fig_tech.add_trace(go.Bar(x=df_ch.index, y=df_ch["volume"],
            marker_color=vol_colors, name="成交量", showlegend=False), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index, y=rsi_ch, mode="lines",
            line=dict(color="#d29922", width=1.5), name="RSI"), row=3, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#f85149", row=3, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#3fb950", row=3, col=1)
        hist_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist_ch]
        fig_tech.add_trace(go.Bar(x=df_ch.index, y=hist_ch,
            marker_color=hist_colors, name="MACD Hist", showlegend=False), row=4, col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index, y=macd_ch, mode="lines",
            line=dict(color="#58a6ff", width=1.2), name="MACD"), row=4, col=1)
        fig_tech.add_trace(go.Scatter(x=df_ch.index, y=sig_ch, mode="lines",
            line=dict(color="#f0883e", width=1.2), name="Signal"), row=4, col=1)

        fig_tech.update_layout(
            title=f"{tk_chart} 技術分析",
            height=750, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            margin=dict(l=10,r=10,t=50,b=10))
        for i in range(1, 5):
            fig_tech.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig_tech.update_yaxes(gridcolor="#21262d", row=i, col=1)
        st.plotly_chart(fig_tech, use_container_width=True)

        short_s, mid_s, sigs = score_stock(df_ch)
        label, _ = signal_label(short_s, mid_s)
        close_v  = float(df_ch["close"].iloc[-1])
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("短線評分", f"{short_s}/100")
        c2.metric("中線評分", f"{mid_s}/100")
        c3.metric("信號", label)
        c4.metric("目標+20%", round(close_v*1.2, 3))
        c5.metric("止損-8%",  round(close_v*0.92, 3))
        if sigs:
            st.markdown("**觸發指標：** " + " ｜ ".join(sigs))
    else:
        st.warning("找不到足夠數據，請確認代碼（港股用 0700.HK 格式）。")

st.divider()
st.caption("⚠️ 本系統僅供技術分析參考，不構成投資建議。數據來自 Yahoo Finance，存在延遲。")
