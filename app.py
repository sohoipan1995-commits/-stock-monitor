import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sqlite3
from pathlib import Path

st.set_page_config(page_title="撈底監察系統 V4", page_icon="📈", layout="wide")

HK_WATCHLIST = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","0066.HK","0003.HK","0002.HK","0016.HK","0883.HK","2318.HK","1299.HK","0001.HK","9988.HK","0175.HK","3690.HK","9618.HK","0981.HK","9999.HK","1211.HK","2688.HK","0762.HK"]
US_WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","AMD","QCOM","INTC","AMAT","JPM","BAC","GS","BRK-B","COST","WMT","JNJ","XOM","NEE","UBER","NFLX","SPY","QQQ"]
DB_FILE = Path("signal_events_v4.sqlite")
C_GREEN, C_RED, C_ORANGE, C_BLUE, C_GREY, C_BG = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#8b949e", "#0d1117"

st.markdown("""<style>
[data-testid='stAppViewContainer']{background:#0d1117}.metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}.good{color:#3fb950}.bad{color:#f85149}.warn{color:#d29922}
</style>""", unsafe_allow_html=True)


def db():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.execute('''CREATE TABLE IF NOT EXISTS signal_events (
        event_id TEXT PRIMARY KEY, event_time TEXT, ticker TEXT, market TEXT, action TEXT,
        score REAL, price REAL, stop_price REAL, target_price REAL, risk_reward REAL,
        valuation REAL, quality REAL, catalyst REAL, technical REAL, risk REAL,
        snapshot TEXT)''')
    return con


def save_event(row):
    event_id = f"{row['ticker']}_{row['event_time'][:10]}_{row['action']}"
    payload = (event_id, row['event_time'], row['ticker'], row['market'], row['action'], row['score'], row['price'], row['stop'], row['target'], row['rr'], row['valuation'], row['quality'], row['catalyst'], row['technical'], row['risk'], row['snapshot'])
    with db() as con:
        con.execute('''INSERT OR REPLACE INTO signal_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', payload)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(x).lower() for x in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    default = {"name": ticker, "forward_pe": np.nan, "pb": np.nan, "ev_ebitda": np.nan, "fcf_yield": np.nan, "roe": np.nan, "debt_equity": np.nan, "earnings_growth": np.nan, "market_cap": np.nan}
    try:
        info = yf.Ticker(ticker).info
        fcf = info.get("freeCashflow")
        cap = info.get("marketCap")
        default.update({
            "name": info.get("shortName") or info.get("longName") or ticker,
            "forward_pe": info.get("forwardPE", np.nan),
            "pb": info.get("priceToBook", np.nan),
            "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
            "fcf_yield": (fcf / cap * 100) if fcf and cap else np.nan,
            "roe": info.get("returnOnEquity", np.nan),
            "debt_equity": info.get("debtToEquity", np.nan),
            "earnings_growth": info.get("earningsGrowth", np.nan),
            "market_cap": cap or np.nan,
        })
    except Exception:
        pass
    return default


def fetch_many(tickers):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_ohlcv, ticker): ticker for ticker in tickers}
        for f in as_completed(futures):
            out[futures[f]] = f.result()
    return out


def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def atr(df, period=14):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def cmf(df, period=20):
    multiplier = (2 * df.close - df.low - df.high) / (df.high - df.low).replace(0, np.nan)
    return (multiplier * df.volume).rolling(period).sum() / df.volume.rolling(period).sum().replace(0, np.nan)


def weekly_rsi(df):
    weekly_close = df.close.resample("W-FRI").last().dropna()
    return rsi(weekly_close, 14)


def anchored_vwap(df, lookback=60):
    recent = df.iloc[-lookback:].copy()
    anchor = recent.low.idxmin()
    part = df.loc[anchor:]
    typical = (part.high + part.low + part.close) / 3
    return float((typical * part.volume).sum() / part.volume.sum()) if part.volume.sum() else np.nan


def pivots(series, order=4):
    lows, highs = [], []
    for i in range(order, len(series) - order):
        w = series.iloc[i-order:i+order+1]
        if series.iloc[i] == w.min(): lows.append(i)
        if series.iloc[i] == w.max(): highs.append(i)
    return lows, highs


def bottom_structure(df):
    lows, highs = pivots(df.low.iloc[-100:].reset_index(drop=True), 4)
    if len(lows) < 2:
        return False, np.nan, "未形成清晰結構"
    a, b = lows[-2], lows[-1]
    segment = df.iloc[-100:].reset_index(drop=True)
    low_a, low_b = float(segment.low.iloc[a]), float(segment.low.iloc[b])
    neckline = float(segment.high.iloc[a:b+1].max()) if b > a else np.nan
    tolerance = abs(low_b - low_a) / low_a
    close = float(segment.close.iloc[-1])
    if tolerance <= 0.04 and close > neckline:
        return True, neckline, "雙底已突破頸線"
    if low_b >= low_a * 0.97:
        return False, neckline, "有雙底雛形，等待突破頸線"
    return False, neckline, "低點結構未確認"


def regime(market):
    benchmark = "SPY" if market == "US" else "^HSI"
    vol_ticker = "^VIX" if market == "US" else "^VHSI"
    b, v = fetch_ohlcv(benchmark, "6mo"), fetch_ohlcv(vol_ticker, "6mo")
    if b is None or len(b) < 60:
        return "unknown", 0.0, 20.0
    ret60 = (b.close.iloc[-1] / b.close.iloc[-60] - 1) * 100
    vol = float(v.close.iloc[-1]) if v is not None and not v.empty else 20.0
    if ret60 < -5 and vol >= 25: return "bear_high_vol", ret60, vol
    if ret60 > 5 and vol < 20: return "bull_low_vol", ret60, vol
    return "neutral", ret60, vol


def fundamentals_score(info, market):
    vals, flags = [], []
    pe, pb, ev, fcf = info['forward_pe'], info['pb'], info['ev_ebitda'], info['fcf_yield']
    if pd.notna(pe):
        vals.append(90 if pe < 10 else 75 if pe < 15 else 55 if pe < 22 else 25)
        flags.append(f"Forward PE {pe:.1f}")
    if pd.notna(pb):
        vals.append(80 if pb < 1 else 65 if pb < 2 else 35)
        flags.append(f"PB {pb:.2f}")
    if pd.notna(ev): vals.append(85 if ev < 8 else 70 if ev < 12 else 45 if ev < 20 else 20)
    if pd.notna(fcf): vals.append(90 if fcf > 8 else 70 if fcf > 4 else 40 if fcf > 0 else 10)
    return (float(np.mean(vals)) if vals else np.nan), " / ".join(flags) or "估值資料不足"


def quality_score(info):
    score, flags = 50, []
    roe, debt, growth = info['roe'], info['debt_equity'], info['earnings_growth']
    if pd.notna(roe):
        score += 20 if roe > .15 else 10 if roe > .08 else -15
        flags.append(f"ROE {roe*100:.1f}%")
    if pd.notna(debt):
        score += 10 if debt < 80 else 0 if debt < 180 else -15
        flags.append(f"D/E {debt:.0f}")
    if pd.notna(growth):
        score += 15 if growth > .10 else 5 if growth > 0 else -15
        flags.append(f"盈利增長 {growth*100:.1f}%")
    return float(np.clip(score, 0, 100)), " / ".join(flags) or "質素資料不足"


def analyse(ticker, df, market, regime_name):
    if df is None or len(df) < 210:
        return None
    info = fetch_info(ticker)
    close = float(df.close.iloc[-1])
    ma20, ma60, ma200 = [float(df.close.rolling(n).mean().iloc[-1]) for n in (20, 60, 200)]
    daily_rsi = float(rsi(df.close).iloc[-1])
    w_rsi = float(weekly_rsi(df).iloc[-1])
    m_line, signal, hist = macd(df.close)
    avwap = anchored_vwap(df)
    cmf_now = float(cmf(df).iloc[-1]) if pd.notna(cmf(df).iloc[-1]) else 0
    vol_ratio = float(df.volume.iloc[-1] / df.volume.rolling(20).mean().iloc[-1])
    high52 = float(df.high.iloc[-252:].max())
    drawdown = (close / high52 - 1) * 100
    atr_now = float(atr(df).iloc[-1])
    structure_ok, neckline, structure_note = bottom_structure(df)
    lows, _ = pivots(df.low.iloc[-60:].reset_index(drop=True), 4)
    higher_low = len(lows) >= 2 and df.low.iloc[-60:].iloc[lows[-1]] >= df.low.iloc[-60:].iloc[lows[-2]] * .97

    valuation, valuation_note = fundamentals_score(info, market)
    quality, quality_note = quality_score(info)
    valuation_ok = pd.notna(valuation)
    catalyst = 50
    catalyst += 20 if pd.notna(info['earnings_growth']) and info['earnings_growth'] > .10 else 0
    catalyst += 15 if cmf_now > .05 else -10 if cmf_now < -.10 else 0
    catalyst += 15 if vol_ratio >= 1.5 and close > ma20 else 0
    catalyst = float(np.clip(catalyst, 0, 100))

    technical = 0
    technical += 20 if daily_rsi < 35 else 10 if daily_rsi < 45 else 0
    technical += 15 if 30 <= w_rsi <= 55 else 0
    technical += 20 if m_line.iloc[-1] > signal.iloc[-1] and hist.iloc[-1] > hist.iloc[-2] else 0
    technical += 15 if close > ma20 and close > avwap else 0
    technical += 15 if higher_low else 0
    technical += 15 if structure_ok else 0
    technical = float(np.clip(technical, 0, 100))

    risk = 70
    risk += 15 if close > ma200 else -20
    risk += 10 if vol_ratio > .7 else -25
    risk += 5 if drawdown > -55 else -15
    risk = float(np.clip(risk, 0, 100))
    stop = min(float(df.low.iloc[-20:].min()), close - 1.5 * atr_now)
    stop = max(stop, close - 3 * atr_now)
    risk_per_share = close - stop
    target = max(close + 2 * risk_per_share, neckline if pd.notna(neckline) and neckline > close else close + 2 * risk_per_share)
    rr = (target - close) / risk_per_share if risk_per_share > 0 else 0

    if not valuation_ok or risk < 40 or risk_per_share / close > .12:
        action = "不合資格"
    elif valuation >= 60 and quality >= 50 and technical < 55:
        action = "觀察"
    elif valuation >= 60 and quality >= 50 and technical >= 55 and not structure_ok:
        action = "等待突破"
    elif valuation >= 60 and quality >= 50 and technical >= 65 and rr >= 2 and (structure_ok or close > ma20):
        action = "小量試倉"
    else:
        action = "觀察"

    total = .30 * (valuation if valuation_ok else 0) + .25 * quality + .20 * catalyst + .15 * technical + .10 * risk
    if regime_name == "bear_high_vol": total *= .90
    if regime_name == "bull_low_vol" and close < ma20: total *= .90
    snapshot = f"日RSI={daily_rsi:.1f}; 周RSI={w_rsi:.1f}; CMF={cmf_now:.2f}; AVWAP={avwap:.2f}; {structure_note}"
    return {"ticker":ticker,"name":info['name'],"price":close,"score":round(total,1),"action":action,"valuation":round(valuation,1) if valuation_ok else np.nan,"quality":round(quality,1),"catalyst":round(catalyst,1),"technical":round(technical,1),"risk":round(risk,1),"daily_rsi":daily_rsi,"weekly_rsi":w_rsi,"drawdown":drawdown,"vol_ratio":vol_ratio,"cmf":cmf_now,"avwap":avwap,"structure":structure_note,"neckline":neckline,"stop":stop,"target":target,"rr":rr,"atr":atr_now,"snapshot":snapshot,"market":market,"event_time":datetime.now().isoformat(timespec='seconds'),"df":df}


def plot_chart(r):
    df = r['df'].tail(250)
    ma20 = df.close.rolling(20).mean(); ma60 = df.close.rolling(60).mean(); ma200 = df.close.rolling(200).mean()
    m, s, h = macd(df.close)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.55,.12,.17,.16], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name='K線', increasing_line_color=C_GREEN, decreasing_line_color=C_RED), 1, 1)
    for series, name, color in [(ma20,'MA20',C_ORANGE),(ma60,'MA60',C_BLUE),(ma200,'MA200','#bc8cff')]: fig.add_trace(go.Scatter(x=df.index,y=series,name=name,line=dict(color=color,width=1)),1,1)
    fig.add_hline(y=r['avwap'], line_dash='dot', line_color=C_GREEN, annotation_text='60日低點 AVWAP', row=1, col=1)
    if pd.notna(r['neckline']): fig.add_hline(y=r['neckline'], line_dash='dash', line_color=C_ORANGE, annotation_text='結構頸線', row=1, col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.volume,name='成交量',marker_color=C_BLUE),2,1)
    fig.add_trace(go.Scatter(x=df.index,y=rsi(df.close),name='RSI(14)',line=dict(color=C_ORANGE)),3,1)
    fig.add_hline(y=30,line_dash='dash',line_color=C_GREEN,row=3,col=1); fig.add_hline(y=70,line_dash='dash',line_color=C_RED,row=3,col=1)
    fig.add_trace(go.Bar(x=df.index,y=h,name='MACD Hist',marker_color=[C_GREEN if x>=0 else C_RED for x in h.fillna(0)]),4,1)
    fig.add_trace(go.Scatter(x=df.index,y=m,name='MACD',line=dict(color=C_BLUE)),4,1); fig.add_trace(go.Scatter(x=df.index,y=s,name='Signal',line=dict(color=C_ORANGE)),4,1)
    fig.update_layout(height=820, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color='#e6edf3'), xaxis_rangeslider_visible=False, legend=dict(orientation='h'))
    return fig


def render_scan(results):
    cols = ["ticker","name","price","action","score","valuation","quality","catalyst","technical","risk","daily_rsi","weekly_rsi","drawdown","vol_ratio","rr"]
    table = pd.DataFrame([{k:r[k] for k in cols} for r in results])
    table.columns = ["代碼","名稱","現價","決策","候選分","估值","質素","催化","轉勢確認","風險","日RSI","真周RSI","距52周高%","量比","R/R"]
    st.dataframe(table.sort_values(["決策","候選分"], ascending=[True,False]), use_container_width=True, hide_index=True)
    choices = [r['ticker'] for r in results]
    selected = st.selectbox("查看個股決策卡", choices)
    r = next(x for x in results if x['ticker'] == selected)
    a,b,c,d,e = st.columns(5)
    for col, label, value in [(a,"估值",r['valuation']),(b,"質素",r['quality']),(c,"催化",r['catalyst']),(d,"轉勢確認",r['technical']),(e,"風險",r['risk'])]: col.metric(label, "資料不足" if pd.isna(value) else f"{value:.1f}/100")
    st.markdown(f"### {r['ticker']} — {r['action']}")
    st.write(f"**結構：** {r['structure']}　|　**日 RSI：** {r['daily_rsi']:.1f}　|　**真周 RSI：** {r['weekly_rsi']:.1f}　|　**AVWAP：** {r['avwap']:.2f}")
    st.write(f"**入場參考：** {r['price']:.2f}　|　**結構止損：** {r['stop']:.2f}　|　**2R 目標：** {r['target']:.2f}　|　**風險回報：** {r['rr']:.2f}R")
    st.plotly_chart(plot_chart(r), use_container_width=True)
    if r['action'] == '小量試倉':
        if st.button(f"記錄 {r['ticker']} 試倉訊號"):
            save_event(r); st.success("訊號已保存到 SQLite，供之後績效追蹤。")


st.title("📈 撈底監察系統 V4")
st.caption("保留原有掃描、圖表、回撤、風控及訊號追蹤概念；新增合資格閘門、真周線 RSI、AVWAP、估值/質素/催化/轉勢分離及決策狀態機。")
with st.sidebar:
    market_label = st.radio("市場", ["🇺🇸 美股", "🇭🇰 港股", "📋 自選"])
    raw = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK") if market_label == "📋 自選" else ""
    account = st.number_input("帳戶總值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆最大風險 (%)", .25, 2.0, 1.0, .25) / 100
    st.caption("V4 不把『跌得深』直接等同於『值得買』。")

market = "US" if "美股" in market_label else "HK"
tickers = US_WATCHLIST if market_label == "🇺🇸 美股" else HK_WATCHLIST if market_label == "🇭🇰 港股" else [x.strip().upper() for x in raw.splitlines() if x.strip()]
regime_name, benchmark_ret, vol = regime(market)
state_text = {"bear_high_vol":"熊市高波動","bull_low_vol":"牛市低波動","neutral":"中性","unknown":"未知"}[regime_name]
st.info(f"市場 regime：**{state_text}** ｜ 基準 60 日回報：{benchmark_ret:.1f}% ｜ 波動指標：{vol:.1f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 V4 決策掃描", "📈 技術圖表", "📐 回撤與止損", "⚖️ 部位管理", "📋 訊號追蹤", "🧪 模型檢查"])

with tab1:
    run = st.button("🔄 掃描目前名單", type="primary")
    if run:
        with st.spinner(f"正在分析 {len(tickers)} 隻股票..."):
            data = fetch_many(tickers)
            results = [analyse(t, data[t], market, regime_name) for t in tickers]
            results = [r for r in results if r]
        st.session_state['v4_results'] = results
    if 'v4_results' in st.session_state:
        render_scan(st.session_state['v4_results'])
    else: st.info("按「掃描目前名單」開始。")

with tab2:
    if 'v4_results' in st.session_state:
        t = st.selectbox("選擇股票", [r['ticker'] for r in st.session_state['v4_results']], key='chart_select')
        st.plotly_chart(plot_chart(next(r for r in st.session_state['v4_results'] if r['ticker']==t)), use_container_width=True)
    else: st.info("先在 V4 決策掃描完成掃描。")

with tab3:
    ticker = st.text_input("股票代碼", "NVDA").upper()
    if st.button("計算結構止損"):
        d = fetch_ohlcv(ticker)
        if d is not None and len(d) >= 60:
            p = float(d.close.iloc[-1]); a = float(atr(d).iloc[-1]); low20 = float(d.low.iloc[-20:].min()); high52 = float(d.high.iloc[-252:].max())
            stop = max(min(low20, p-1.5*a), p-3*a)
            c1,c2,c3,c4 = st.columns(4); c1.metric("現價",f"{p:.2f}"); c2.metric("52周回撤",f"{(p/high52-1)*100:.1f}%"); c3.metric("ATR 結構止損",f"{stop:.2f}"); c4.metric("2R 目標",f"{p+2*(p-stop):.2f}")
        else: st.error("資料不足。")

with tab4:
    ticker = st.text_input("股票代碼", "AAPL", key='position_ticker').upper()
    if st.button("計算最大部位"):
        d = fetch_ohlcv(ticker)
        if d is not None and len(d) >= 60:
            p = float(d.close.iloc[-1]); a = float(atr(d).iloc[-1]); stop = max(min(float(d.low.iloc[-20:].min()), p-1.5*a), p-3*a); per_share = max(p-stop, .0001)
            risk_cash = account * risk_pct; shares_by_risk = int(risk_cash/per_share); shares_by_cash = int(account/p); shares = min(shares_by_risk, shares_by_cash)
            st.success(f"建議最大股數：{shares:,} 股")
            st.write(f"入場 {p:.2f} ｜ 止損 {stop:.2f} ｜ 每股風險 {per_share:.2f} ｜ 最大帳面風險 {shares*per_share:,.2f}")
            st.caption("港股下單前須再按每手股數及貨幣換算向下調整。")

with tab5:
    with db() as con: events = pd.read_sql_query("SELECT * FROM signal_events ORDER BY event_time DESC", con)
    if events.empty: st.info("尚未保存任何試倉訊號。只有『小量試倉』狀態才可記錄。")
    else:
        st.dataframe(events.drop(columns=['snapshot']), use_container_width=True, hide_index=True)
        st.caption("下一版可加入每日排程、下一日開市進場、止損/目標離場、費用及 SPY/HSI 基準的 walk-forward 回測。")

with tab6:
    st.markdown("## V4 已修正／新增項目")
    st.markdown("""
- **真周線 RSI**：先將日線資料轉成每周收市價，再算 RSI(14)，不再把日線 RSI(70) 當周線。
- **AVWAP**：從近 60 日低點起算 anchored VWAP，避免把兩年累計 VWAP 當作今日公平價。
- **指標去重**：RSI、MACD、均線、higher-low、結構突破分別反映不同資訊；不再把多個同類超賣指標無限疊加。
- **合資格閘門**：估值資料不足、風險分過低、止損距離過大，不會直接給買入結論。
- **狀態機**：不合資格 → 觀察 → 等待突破 → 小量試倉。
- **風控**：止損以 20 日結構低點及 ATR 約束，部位同時受帳戶資金與每筆風險上限限制。
- **資料庫**：以 SQLite 保存訊號事件，而非以 CSV 作為長期事件紀錄。
""")
