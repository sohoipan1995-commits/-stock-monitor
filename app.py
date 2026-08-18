import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="撈底監察系統 V4.3", page_icon="📈", layout="wide")

HK = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","0066.HK","0003.HK","0883.HK","2318.HK","1299.HK","9988.HK","0175.HK","3690.HK","9618.HK","0981.HK","9999.HK","1211.HK","2688.HK","0762.HK","1810.HK","1024.HK","2020.HK"]
US = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","AMD","QCOM","INTC","AMAT","LRCX","JPM","BAC","GS","BRK-B","COST","WMT","JNJ","UNH","XOM","NEE","UBER","NFLX","SPY","QQQ"]
DB = Path("signals_v43.sqlite")
BG, PANEL, BORDER, GREEN, RED, ORANGE, BLUE, GREY = "#0d1117", "#161b22", "#30363d", "#3fb950", "#f85149", "#d29922", "#58a6ff", "#8b949e"

st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background:{BG};}}
[data-testid="stSidebar"]{{background:{PANEL};}}
h1,h2,h3,p,label,.stMarkdown{{color:#e6edf3!important;}}
.card{{background:{PANEL};border:1px solid {BORDER};border-left:5px solid {BLUE};border-radius:10px;padding:14px;margin:8px 0;}}
.good{{border-left-color:{GREEN};}} .wait{{border-left-color:{ORANGE};}} .bad{{border-left-color:{RED};}}
</style>""", unsafe_allow_html=True)


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS signals(
            event_id TEXT PRIMARY KEY,event_time TEXT,ticker TEXT,action TEXT,
            price REAL,breakout REAL,stop REAL,target REAL,score REAL,snapshot TEXT)""")


def save_signal(r):
    event_id = f"{r['ticker']}_{r['event_time'][:10]}_{r['action']}"
    values = (event_id, r["event_time"], r["ticker"], r["action"], r["price"], r["breakout"], r["stop"], r["target"], r["score"], json.dumps(r["snapshot"], ensure_ascii=False))
    with sqlite3.connect(DB) as con:
        con.execute("INSERT OR REPLACE INTO signals VALUES(?,?,?,?,?,?,?,?,?,?)", values)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        cols = ["open", "high", "low", "close", "volume"]
        return df[cols].dropna() if all(c in df.columns for c in cols) else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker):
    out = {"name":ticker,"pe":np.nan,"pb":np.nan,"ev":np.nan,"fcf":np.nan,"roe":np.nan,"de":np.nan,"growth":np.nan}
    try:
        info = yf.Ticker(ticker).info
        cap, fcf = info.get("marketCap"), info.get("freeCashflow")
        out.update({
            "name": info.get("shortName") or info.get("longName") or ticker,
            "pe": info.get("forwardPE", np.nan),
            "pb": info.get("priceToBook", np.nan),
            "ev": info.get("enterpriseToEbitda", np.nan),
            "fcf": fcf / cap * 100 if fcf and cap else np.nan,
            "roe": info.get("returnOnEquity", np.nan),
            "de": info.get("debtToEquity", np.nan),
            "growth": info.get("earningsGrowth", np.nan),
        })
    except Exception:
        pass
    return out


def fetch_many(tickers):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        jobs = {ex.submit(fetch_prices, ticker): ticker for ticker in tickers}
        for job in as_completed(jobs):
            ticker = jobs[job]
            try:
                out[ticker] = job.result()
            except Exception:
                out[ticker] = None
    return out


def calc_rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calc_macd(s):
    line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def calc_atr(d, n=14):
    ranges = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1)
    return ranges.max(axis=1).rolling(n).mean()


def calc_cmf(d, n=20):
    multiplier = (2 * d.close - d.high - d.low) / (d.high - d.low).replace(0, np.nan)
    return (multiplier * d.volume).rolling(n).sum() / d.volume.rolling(n).sum().replace(0, np.nan)


def calc_avwap(d, lookback=60):
    anchor = d.iloc[-lookback:].low.idxmin()
    part = d.loc[anchor:]
    typical = (part.high + part.low + part.close) / 3
    return float((typical * part.volume).sum() / part.volume.sum())


def find_lows(s, order=4):
    return [i for i in range(order, len(s) - order) if s.iloc[i] == s.iloc[i-order:i+order+1].min()]


def detect_structure(d):
    part = d.iloc[-100:].reset_index(drop=True)
    lows = find_lows(part.low)
    if len(lows) < 2:
        return False, np.nan, False, "未有足夠底部結構"
    a, b = lows[-2], lows[-1]
    low_a, low_b = float(part.low.iloc[a]), float(part.low.iloc[b])
    neckline = float(part.high.iloc[a:b+1].max())
    similar = abs(low_a - low_b) / max(low_a, 0.0001) <= 0.04
    higher = low_b >= low_a * 0.97
    if similar and part.close.iloc[-1] > neckline:
        return True, neckline, higher, "雙底已突破"
    if similar or higher:
        return False, neckline, higher, "底部雛形，等待確認突破"
    return False, neckline, False, "結構仍偏弱"


def score_valuation(f):
    scores = []
    if pd.notna(f["pe"]) and f["pe"] > 0:
        scores.append(90 if f["pe"] < 10 else 75 if f["pe"] < 15 else 55 if f["pe"] < 22 else 25)
    if pd.notna(f["pb"]) and f["pb"] > 0:
        scores.append(85 if f["pb"] < 1 else 65 if f["pb"] < 2 else 35)
    if pd.notna(f["ev"]) and f["ev"] > 0:
        scores.append(85 if f["ev"] < 8 else 70 if f["ev"] < 12 else 45)
    if pd.notna(f["fcf"]):
        scores.append(90 if f["fcf"] > 8 else 70 if f["fcf"] > 4 else 45 if f["fcf"] > 0 else 10)
    return float(np.mean(scores)) if scores else np.nan


def analyse_stock(ticker, d):
    if d is None or len(d) < 210:
        return None
    f = fetch_fundamentals(ticker)
    price = float(d.close.iloc[-1])
    ma20 = float(d.close.rolling(20).mean().iloc[-1])
    ma200 = float(d.close.rolling(200).mean().iloc[-1])
    daily_rsi = float(calc_rsi(d.close).iloc[-1])
    weekly_close = d.close.resample("W-FRI").last().dropna()
    weekly_rsi = float(calc_rsi(weekly_close).iloc[-1])
    macd_line, macd_signal, macd_hist = calc_macd(d.close)
    volume_ratio = float(d.volume.iloc[-1] / d.volume.rolling(20).mean().iloc[-1])
    cmf_now = float(calc_cmf(d).iloc[-1]) if pd.notna(calc_cmf(d).iloc[-1]) else 0.0
    high52 = float(d.high.iloc[-252:].max())
    drawdown = (price / high52 - 1) * 100
    confirmed, neckline, higher_low, structure_note = detect_structure(d)
    breakout = neckline if pd.notna(neckline) else max(ma20, float(d.high.iloc[-10:].max()))
    improving = macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_hist.iloc[-1] > macd_hist.iloc[-2]
    valuation = score_valuation(f)
    quality = float(np.clip(
        50 + (20 if pd.notna(f["roe"]) and f["roe"] > .15 else 10 if pd.notna(f["roe"]) and f["roe"] > .08 else -15 if pd.notna(f["roe"]) else 0)
        + (10 if pd.notna(f["de"]) and f["de"] < 80 else -15 if pd.notna(f["de"]) and f["de"] > 180 else 0)
        + (15 if pd.notna(f["growth"]) and f["growth"] > .10 else -15 if pd.notna(f["growth"]) and f["growth"] < 0 else 0), 0, 100))
    catalyst = float(np.clip(
        50 + (20 if pd.notna(f["growth"]) and f["growth"] > .10 else -10 if pd.notna(f["growth"]) and f["growth"] < 0 else 0)
        + (15 if cmf_now > .05 else -10 if cmf_now < -.10 else 0)
        + (15 if volume_ratio >= 1.5 and price > ma20 else 0), 0, 100))
    technical = float(np.clip(
        (15 if daily_rsi < 35 else 8 if daily_rsi < 45 else 0)
        + (15 if 30 <= weekly_rsi <= 55 else 0)
        + (20 if improving else 0)
        + (15 if price > ma20 and price > calc_avwap(d) else 0)
        + (15 if higher_low else 0)
        + (20 if confirmed else 0), 0, 100))
    risk = float(np.clip(60 + (15 if price > ma200 else -20) + (10 if volume_ratio >= .7 else -25) + (5 if drawdown > -55 else -15), 0, 100))
    atr_now = float(calc_atr(d).iloc[-1])
    stop = max(min(float(d.low.iloc[-20:].min()), price - 1.5 * atr_now), price - 3 * atr_now)
    if stop >= price:
        stop = price - max(atr_now, price * .03)
    target = price + 2 * (price - stop)
    rr = (target - price) / max(price - stop, .0001)
    eligible = pd.notna(valuation) and quality >= 40 and risk >= 40 and (price - stop) / price <= .12 and volume_ratio >= .35
    action = "不合資格" if not eligible else "觀察" if technical < 55 else "等待突破" if price <= breakout else "小量試倉" if technical >= 60 and rr >= 2 else "觀察"
    total = round(float(.30 * (valuation if pd.notna(valuation) else 0) + .25 * quality + .20 * catalyst + .15 * technical + .10 * risk), 1)
    tier = "資料不足" if pd.isna(valuation) else "嚴重低估" if valuation >= 80 else "中度低估" if valuation >= 65 else "輕度低估" if valuation >= 50 else "估值不吸引"
    reasons, missing = [], []
    if pd.notna(valuation) and valuation >= 65: reasons.append(f"{tier}（估值 {valuation:.0f}）")
    if quality >= 60: reasons.append(f"質素合格（{quality:.0f}）")
    if daily_rsi < 35: reasons.append(f"日 RSI {daily_rsi:.1f} 偏低")
    if higher_low: reasons.append("形成較高低點")
    if improving: reasons.append("MACD 改善")
    if price <= breakout: missing.append(f"未突破確認價 {breakout:.2f}")
    if volume_ratio < 1.5: missing.append("量能未達 1.5 倍")
    if price < ma200: missing.append("仍低於 MA200")
    return {
        "ticker":ticker, "name":f["name"], "event_time":datetime.now().isoformat(timespec="seconds"),
        "price":price, "action":action, "score":total, "tier":tier, "valuation":valuation,
        "quality":quality, "catalyst":catalyst, "technical":technical, "risk":risk,
        "daily_rsi":daily_rsi, "weekly_rsi":weekly_rsi, "vol":volume_ratio,
        "breakout":breakout, "stop":stop, "target":target, "rr":rr, "df":d,
        "snapshot":{"reasons":reasons,"missing":missing,"structure":structure_note,"cmf":cmf_now}
    }


def stock_chart(r):
    d = r["df"].tail(250)
    macd_line, macd_signal, _ = calc_macd(d.close)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.65,.15,.20], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="K線", increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)
    for series, name, color in [(d.close.rolling(20).mean(),"MA20",ORANGE),(d.close.rolling(200).mean(),"MA200","#bc8cff")]:
        fig.add_trace(go.Scatter(x=d.index, y=series, name=name, line=dict(color=color)), 1, 1)
    for value, name, color in [(r["breakout"],"確認價",ORANGE),(r["stop"],"止損",RED),(r["target"],"2R目標",GREEN)]:
        fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=name, row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d.volume, name="成交量", marker_color=BLUE), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=calc_rsi(d.close), name="RSI", line=dict(color=ORANGE)), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=macd_line, name="MACD", line=dict(color=BLUE)), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=macd_signal, name="Signal", line=dict(color=GREY)), 3, 1)
    fig.update_layout(height=720, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig

# ------------------------- Gann stock calendar -------------------------
def atr_zigzag(d, multiplier=2.0):
    a = calc_atr(d).bfill()
    pivots, trend = [], 0
    hp, lp = float(d.high.iloc[0]), float(d.low.iloc[0])
    hd, ld = d.index[0], d.index[0]
    for i in range(1, len(d)):
        high, low, close = float(d.high.iloc[i]), float(d.low.iloc[i]), float(d.close.iloc[i])
        threshold = max(float(a.iloc[i]) * multiplier, close * .004)
        if trend >= 0:
            if high >= hp: hp, hd = high, d.index[i]
            if close <= hp - threshold:
                pivots.append((hd, hp, "HIGH", True)); trend = -1; lp, ld = low, d.index[i]
        if trend <= 0:
            if low <= lp: lp, ld = low, d.index[i]
            if close >= lp + threshold:
                pivots.append((ld, lp, "LOW", True)); trend = 1; hp, hd = high, d.index[i]
    pivots.append((hd, hp, "HIGH", False) if trend >= 0 else (ld, lp, "LOW", False))
    return pd.DataFrame(pivots, columns=["date","price","kind","confirmed"]).drop_duplicates(["date","kind"], keep="last")


def next_cycle(anchor, asof, cycles):
    output = []
    for cycle in cycles:
        date = anchor + pd.offsets.BDay(cycle)
        while date < asof - pd.offsets.BDay(3):
            date += pd.offsets.BDay(cycle)
        output.append((date, cycle))
    return min(output, key=lambda x: x[0])


def gann_calendar_row(r, multiplier=2.0):
    d = r["df"]
    z = atr_zigzag(d, multiplier)
    confirmed = z[z["confirmed"]]
    if len(confirmed) < 2:
        return None
    last, prior = confirmed.iloc[-1], confirmed.iloc[-2]
    asof, close = d.index[-1], float(d.close.iloc[-1])
    atr_now = float(calc_atr(d).iloc[-1])
    high, low = max(float(last.price), float(prior.price)), min(float(last.price), float(prior.price))
    levels = np.array([low + (high - low) * x for x in [.125,.25,.333,.5,.667,.75,.875]])
    near = float(levels[np.argmin(abs(levels - close))])
    price_ok = abs(near - close) <= max(.75 * atr_now, close * .005)
    light = next_cycle(pd.Timestamp(last.date), asof, [20,30,45])
    medium = next_cycle(pd.Timestamp(last.date), asof, [60,90])
    important = next_cycle(pd.Timestamp(last.date), asof, [120,144,180])
    macd_line, macd_signal, _ = calc_macd(d.close)
    reverse = (last.kind == "LOW" and macd_line.iloc[-1] > macd_signal.iloc[-1]) or (last.kind == "HIGH" and macd_line.iloc[-1] < macd_signal.iloc[-1])
    nearest = min([light, medium, important], key=lambda x: abs((x[0] - asof).days))
    base = "重要" if nearest[1] in [120,144,180] else "中度" if nearest[1] in [60,90] else "輕度"
    level = {"輕度":1,"中度":2,"重要":3}[base] + (1 if price_ok else 0) + (1 if reverse else 0)
    attention = "高注意" if level >= 4 else "中注意" if level >= 3 else "輕度注意"
    direction = "下行波段後，留意向上轉勢" if last.kind == "LOW" else "上行波段後，留意向下轉勢"
    fmt = lambda x: f"{x[0].date()} ±3交易日（{x[1]}日）"
    confluence = min(100, 35 + (25 if price_ok else 0) + (25 if reverse else 0) + (15 if base == "重要" else 8 if base == "中度" else 0))
    return {
        "代碼": r["ticker"],
        "現價": round(close, 2),
        "最近確認樞紐": f"{last.kind} {last.price:.2f}｜{pd.Timestamp(last.date).date()}",
        "預期方向": direction,
        "輕度轉勢日期": fmt(light),
        "中度轉勢日期": fmt(medium),
        "重要轉勢日期": fmt(important),
        "最近時間窗": f"{nearest[0].date()}（{base}）",
        "價格共振": f"{near:.2f}" if price_ok else "無",
        "技術確認": "MACD 初步確認" if reverse else "未確認",
        "注意級別": attention,
        "江恩共振分": confluence,
        "_light": light[0],
        "_medium": medium[0],
        "_important": important[0],
        "_df": d,
        "_z": z
    }


def gann_chart(row):
    d, z = row["_df"].tail(260), row["_z"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="K線", increasing_line_color=GREEN, decreasing_line_color=RED))
    p = z[z.date >= d.index[0]]
    fig.add_trace(go.Scatter(x=p.date, y=p.price, mode="lines+markers", name="ATR ZigZag", line=dict(color=BLUE), marker=dict(color=[RED if x == "HIGH" else GREEN for x in p.kind], size=9)))
    for date, color, label in [(row["_light"],GREY,"輕度"),(row["_medium"],ORANGE,"中度"),(row["_important"],RED,"重要")]:
        fig.add_vrect(x0=date-pd.offsets.BDay(3), x1=date+pd.offsets.BDay(3), fillcolor=color, opacity=.12, line_width=0, annotation_text=label, annotation_position="top left")
    fig.update_layout(height=600, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


init_db()
st.title("📈 撈底監察系統 V4.3 — 江恩個股轉勢日曆")
st.caption("先用估值、質素、催化、轉勢與風險篩選；再以江恩時間窗安排觀察與交易時機。江恩日期是預警窗口，不是必然轉勢預言。")

with st.sidebar:
    market_label = st.radio("市場", ["🇺🇸 美股","🇭🇰 港股","📋 自選"])
    custom = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK") if market_label == "📋 自選" else ""
    account = st.number_input("帳戶總值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆最大風險 (%)", .25, 2.0, 1.0, .25) / 100
    show_all = st.checkbox("候選頁顯示所有分析股", False)

tickers = US if market_label == "🇺🇸 美股" else HK if market_label == "🇭🇰 港股" else [x.strip().upper() for x in custom.splitlines() if x.strip()]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 今日候選","📊 全部掃描","📈 技術圖表","📐 交易計劃","🔮 江恩個股日曆","📋 訊號紀錄"])

with tab1:
    if st.button("🔄 掃描目前名單", type="primary", key="run_scan"):
        with st.spinner(f"正在分析 {len(tickers)} 隻股票..."):
            data = fetch_many(tickers)
            st.session_state["v43_results"] = [r for r in [analyse_stock(t, data.get(t)) for t in tickers] if r]
    results = st.session_state.get("v43_results", [])
    if not results:
        st.info("按「掃描目前名單」開始。")
    else:
        visible = results if show_all else [r for r in results if r["action"] in ["等待突破","小量試倉"]]
        if not visible: st.warning("目前沒有等待突破或小量試倉候選。")
        for r in sorted(visible, key=lambda x: (x["action"] != "小量試倉", -x["score"])):
            style = "good" if r["action"] == "小量試倉" else "wait"
            reasons = "<br>".join("✓ " + x for x in r["snapshot"]["reasons"]) or "—"
            missing = "<br>".join("• " + x for x in r["snapshot"]["missing"]) or "—"
            st.markdown(f"<div class='card {style}'><h3>{r['ticker']}　{r['action']}　|　候選分 {r['score']}</h3><p>現價 {r['price']:.2f}　確認價 {r['breakout']:.2f}　止損 {r['stop']:.2f}　2R目標 {r['target']:.2f}</p><p><b>入選原因</b><br>{reasons}</p><p><b>仍欠條件</b><br>{missing}</p></div>", unsafe_allow_html=True)
            a,b,c,d,e = st.columns(5)
            a.metric("估值", "資料不足" if pd.isna(r["valuation"]) else f"{r['valuation']:.0f}")
            b.metric("質素", f"{r['quality']:.0f}")
            c.metric("催化", f"{r['catalyst']:.0f}")
            d.metric("轉勢", f"{r['technical']:.0f}")
            e.metric("R/R", f"{r['rr']:.2f}R")
            if r["action"] == "小量試倉" and st.button(f"保存 {r['ticker']} 訊號", key=f"save_{r['ticker']}"):
                save_signal(r); st.success("已保存訊號。")

with tab2:
    results = st.session_state.get("v43_results", [])
    if results:
        rows = [{k:r[k] for k in ["ticker","name","price","action","score","tier","valuation","quality","catalyst","technical","risk","daily_rsi","weekly_rsi","vol","breakout","rr"]} for r in results]
        table = pd.DataFrame(rows)
        table.columns = ["代碼","名稱","現價","決策","候選分","低估級別","估值","質素","催化","轉勢","風險","日RSI","真周RSI","量比","確認價","R/R"]
        st.dataframe(table.sort_values("候選分", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("下載 CSV", table.to_csv(index=False).encode("utf-8-sig"), "scan_v43.csv", "text/csv", key="scan_csv")
    else: st.info("請先掃描。")

with tab3:
    results = st.session_state.get("v43_results", [])
    if results:
        ticker = st.selectbox("選擇股票", [r["ticker"] for r in results], key="chart_ticker")
        selected = next(r for r in results if r["ticker"] == ticker)
        st.plotly_chart(stock_chart(selected), use_container_width=True, key=f"stock_chart_{ticker}")
    else: st.info("請先掃描。")

with tab4:
    ticker = st.text_input("股票代碼", "AAPL", key="plan_ticker").upper()
    if st.button("建立交易計劃", key="build_plan"):
        r = analyse_stock(ticker, fetch_prices(ticker))
        if not r: st.error("資料不足。")
        else:
            per_share_risk = max(r["price"] - r["stop"], .0001)
            shares = min(int(account * risk_pct / per_share_risk), int(account / r["price"]))
            a,b,c,d,e = st.columns(5)
            a.metric("現價", f"{r['price']:.2f}"); b.metric("確認價", f"{r['breakout']:.2f}")
            c.metric("止損", f"{r['stop']:.2f}"); d.metric("2R目標", f"{r['target']:.2f}")
            e.metric("最大股數", f"{shares:,}")

with tab5:
    st.subheader("🔮 所有掃描股票：江恩轉勢日期表")
    st.caption("輕度＝20、30、45交易日；中度＝60、90交易日；重要＝120、144、180交易日。所有日期均以中心日 ±3 個交易日顯示。")
    results = st.session_state.get("v43_results", [])
    sensitivity = st.slider("ATR ZigZag 敏感度", 1.0, 4.0, 2.0, .25, key="gann_sensitivity")
    if results and st.button("生成所有股票江恩日期表", type="primary", key="make_gann_table"):
        with st.spinner("正在計算所有已掃描股票的江恩時間窗..."):
            st.session_state["gann_calendar"] = [x for x in [gann_calendar_row(r, sensitivity) for r in results] if x]
    cal = st.session_state.get("gann_calendar", [])
    if not results: st.info("請先在『今日候選』掃描股票。")
    elif not cal: st.info("按『生成所有股票江恩日期表』。")
    else:
        levels = st.multiselect("篩選注意級別", ["高注意","中注意","輕度注意"], default=["高注意","中注意","輕度注意"], key="gann_filter")
        display = [x for x in cal if x["注意級別"] in levels]
        columns = ["代碼","現價","最近確認樞紐","預期方向","輕度轉勢日期","中度轉勢日期","重要轉勢日期","最近時間窗","價格共振","技術確認","注意級別","江恩共振分"]
        table = pd.DataFrame(display)[columns].sort_values("江恩共振分", ascending=False)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("下載江恩日期表 CSV", table.to_csv(index=False).encode("utf-8-sig"), "gann_stock_calendar_v43.csv", "text/csv", key="gann_csv")
        pick = st.selectbox("查看個股江恩圖", [x["代碼"] for x in display], key="gann_pick")
        row = next(x for x in display if x["代碼"] == pick)
        st.plotly_chart(gann_chart(row), use_container_width=True, key=f"gann_chart_{pick}_{sensitivity}")
        st.info("先看未來最近的『重要』日期；只有時間窗、價格共振及 MACD 初步確認同時出現時，才視為高注意。日期本身不是買賣指令。")

with tab6:
    with sqlite3.connect(DB) as con:
        log = pd.read_sql_query("SELECT * FROM signals ORDER BY event_time DESC", con)
    if log.empty: st.info("尚未有保存訊號。")
    else: st.dataframe(log.drop(columns=["snapshot"]), use_container_width=True, hide_index=True)
