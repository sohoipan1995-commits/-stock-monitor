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

st.set_page_config(page_title="撈底監察系統 V4.1", page_icon="📈", layout="wide")

HK_WATCHLIST = [
    "0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "0066.HK",
    "0003.HK", "0002.HK", "0016.HK", "0883.HK", "2318.HK", "1299.HK", "0001.HK",
    "9988.HK", "0175.HK", "3690.HK", "9618.HK", "0981.HK", "9999.HK", "1211.HK",
    "2688.HK", "0762.HK", "1810.HK", "1024.HK", "2020.HK"
]
US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD",
    "QCOM", "INTC", "AMAT", "LRCX", "JPM", "BAC", "GS", "BRK-B", "COST", "WMT",
    "JNJ", "UNH", "XOM", "NEE", "UBER", "NFLX", "SPY", "QQQ"
]

DB_FILE = Path("signal_events_v41.sqlite")
C_BG = "#0d1117"
C_PANEL = "#161b22"
C_BORDER = "#30363d"
C_GREEN = "#3fb950"
C_RED = "#f85149"
C_ORANGE = "#d29922"
C_BLUE = "#58a6ff"
C_GREY = "#8b949e"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background: {C_BG}; }}
[data-testid="stSidebar"] {{ background: {C_PANEL}; }}
h1,h2,h3,p,label,.stMarkdown {{ color: #e6edf3 !important; }}
.decision-card {{ background:{C_PANEL}; border:1px solid {C_BORDER}; border-left:5px solid {C_BLUE}; border-radius:10px; padding:16px; margin:8px 0; }}
.good-card {{ border-left-color:{C_GREEN}; }}
.wait-card {{ border-left-color:{C_ORANGE}; }}
.bad-card {{ border-left-color:{C_RED}; }}
.small-note {{ color:{C_GREY}; font-size:0.85rem; }}
</style>
""", unsafe_allow_html=True)


def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS signal_events (
            event_id TEXT PRIMARY KEY,
            event_time TEXT NOT NULL,
            ticker TEXT NOT NULL,
            market TEXT NOT NULL,
            action TEXT NOT NULL,
            total_score REAL,
            price REAL,
            breakout_price REAL,
            stop_price REAL,
            target_price REAL,
            risk_reward REAL,
            valuation REAL,
            quality REAL,
            catalyst REAL,
            technical REAL,
            risk REAL,
            snapshot_json TEXT
        )""")


def save_signal(r):
    event_id = f"{r['ticker']}_{r['event_time'][:10]}_{r['action']}"
    values = (
        event_id, r["event_time"], r["ticker"], r["market"], r["action"], r["total_score"],
        r["price"], r["breakout_price"], r["stop"], r["target"], r["rr"], r["valuation"],
        r["quality"], r["catalyst"], r["technical"], r["risk"], json.dumps(r["snapshot"], ensure_ascii=False)
    )
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""INSERT OR REPLACE INTO signal_events VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)


def load_signals():
    with sqlite3.connect(DB_FILE) as con:
        return pd.read_sql_query("SELECT * FROM signal_events ORDER BY event_time DESC", con)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(x).lower() for x in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].dropna()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    output = {
        "name": ticker, "forward_pe": np.nan, "pb": np.nan, "ev_ebitda": np.nan,
        "fcf_yield": np.nan, "roe": np.nan, "debt_equity": np.nan,
        "earnings_growth": np.nan, "market_cap": np.nan
    }
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap")
        free_cash_flow = info.get("freeCashflow")
        output.update({
            "name": info.get("shortName") or info.get("longName") or ticker,
            "forward_pe": info.get("forwardPE", np.nan),
            "pb": info.get("priceToBook", np.nan),
            "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
            "fcf_yield": (free_cash_flow / market_cap * 100) if free_cash_flow and market_cap else np.nan,
            "roe": info.get("returnOnEquity", np.nan),
            "debt_equity": info.get("debtToEquity", np.nan),
            "earnings_growth": info.get("earningsGrowth", np.nan),
            "market_cap": market_cap if market_cap else np.nan,
        })
    except Exception:
        pass
    return output


def fetch_many(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception:
                results[ticker] = None
    return results


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calc_macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def calc_atr(df, period=14):
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1)
    return ranges.max(axis=1).rolling(period).mean()


def calc_cmf(df, period=20):
    multiplier = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def calc_weekly_rsi(df):
    weekly_close = df["close"].resample("W-FRI").last().dropna()
    return rsi(weekly_close, 14)


def anchored_vwap(df, lookback=60):
    recent = df.iloc[-lookback:]
    anchor_date = recent["low"].idxmin()
    part = df.loc[anchor_date:]
    typical = (part["high"] + part["low"] + part["close"]) / 3
    denom = part["volume"].sum()
    return float((typical * part["volume"]).sum() / denom) if denom > 0 else np.nan


def find_pivots(series, order=4):
    lows, highs = [], []
    for i in range(order, len(series) - order):
        window = series.iloc[i - order:i + order + 1]
        if series.iloc[i] == window.min():
            lows.append(i)
        if series.iloc[i] == window.max():
            highs.append(i)
    return lows, highs


def bottom_structure(df):
    part = df.iloc[-100:].reset_index(drop=True)
    lows, _ = find_pivots(part["low"], 4)
    if len(lows) < 2:
        return False, np.nan, False, "未形成可辨識底部結構"
    first, second = lows[-2], lows[-1]
    first_low, second_low = float(part["low"].iloc[first]), float(part["low"].iloc[second])
    neckline = float(part["high"].iloc[first:second + 1].max())
    close = float(part["close"].iloc[-1])
    similar_low = abs(first_low - second_low) / max(first_low, 0.00001) <= 0.04
    higher_low = second_low >= first_low * 0.97
    if similar_low and close > neckline:
        return True, neckline, higher_low, "雙底完成，已收市突破頸線"
    if similar_low or higher_low:
        return False, neckline, higher_low, "有底部雛形，仍等待收市突破確認價"
    return False, neckline, False, "低點結構仍偏弱"


def get_regime(market):
    benchmark = "SPY" if market == "US" else "^HSI"
    volatility = "^VIX" if market == "US" else "^VHSI"
    benchmark_df = fetch_ohlcv(benchmark, "6mo")
    vol_df = fetch_ohlcv(volatility, "6mo")
    if benchmark_df is None or len(benchmark_df) < 60:
        return "unknown", 0.0, np.nan, benchmark
    ret60 = (benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[-60] - 1) * 100
    vol = float(vol_df["close"].iloc[-1]) if vol_df is not None and not vol_df.empty else np.nan
    if ret60 < -5 and pd.notna(vol) and vol >= 25:
        return "bear_high_vol", ret60, vol, benchmark
    if ret60 > 5 and (pd.isna(vol) or vol < 20):
        return "bull_low_vol", ret60, vol, benchmark
    return "neutral", ret60, vol, benchmark


def valuation_score(info):
    component_scores, notes = [], []
    pe, pb, ev, fcf = info["forward_pe"], info["pb"], info["ev_ebitda"], info["fcf_yield"]
    if pd.notna(pe) and pe > 0:
        component_scores.append(90 if pe < 10 else 75 if pe < 15 else 55 if pe < 22 else 25)
        notes.append(f"Forward PE {pe:.1f}")
    if pd.notna(pb) and pb > 0:
        component_scores.append(85 if pb < 1 else 65 if pb < 2 else 35)
        notes.append(f"PB {pb:.2f}")
    if pd.notna(ev) and ev > 0:
        component_scores.append(85 if ev < 8 else 70 if ev < 12 else 45 if ev < 20 else 20)
        notes.append(f"EV/EBITDA {ev:.1f}")
    if pd.notna(fcf):
        component_scores.append(90 if fcf > 8 else 70 if fcf > 4 else 45 if fcf > 0 else 10)
        notes.append(f"FCF Yield {fcf:.1f}%")
    if not component_scores:
        return np.nan, "資料不足"
    return float(np.mean(component_scores)), " ｜ ".join(notes)


def quality_score(info):
    score, notes = 50, []
    roe, debt, growth = info["roe"], info["debt_equity"], info["earnings_growth"]
    if pd.notna(roe):
        score += 20 if roe > .15 else 10 if roe > .08 else -15
        notes.append(f"ROE {roe * 100:.1f}%")
    if pd.notna(debt):
        score += 10 if debt < 80 else 0 if debt < 180 else -15
        notes.append(f"D/E {debt:.0f}")
    if pd.notna(growth):
        score += 15 if growth > .10 else 5 if growth > 0 else -15
        notes.append(f"盈利增長 {growth * 100:.1f}%")
    return float(np.clip(score, 0, 100)), " ｜ ".join(notes) if notes else "資料不足"


def undervaluation_tier(score):
    if pd.isna(score):
        return "資料不足"
    if score >= 80:
        return "嚴重低估"
    if score >= 65:
        return "中度低估"
    if score >= 50:
        return "輕度低估"
    return "估值不吸引"


def analyse_stock(ticker, df, market, regime):
    if df is None or len(df) < 210:
        return None
    info = fetch_info(ticker)
    close = float(df["close"].iloc[-1])
    ma20 = float(df["close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["close"].rolling(60).mean().iloc[-1])
    ma200 = float(df["close"].rolling(200).mean().iloc[-1])
    daily_rsi = float(rsi(df["close"]).iloc[-1])
    weekly_series = calc_weekly_rsi(df)
    weekly_rsi = float(weekly_series.iloc[-1]) if not weekly_series.empty and pd.notna(weekly_series.iloc[-1]) else np.nan
    macd_line, macd_signal, macd_hist = calc_macd(df["close"])
    cmf_series = calc_cmf(df)
    cmf = float(cmf_series.iloc[-1]) if pd.notna(cmf_series.iloc[-1]) else 0.0
    avwap = anchored_vwap(df)
    avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
    vol_ratio = float(df["volume"].iloc[-1] / avg_vol) if avg_vol > 0 else 0.0
    high52 = float(df["high"].iloc[-252:].max())
    drawdown = (close / high52 - 1) * 100
    atr_value = float(calc_atr(df).iloc[-1])
    confirmed_bottom, neckline, higher_low, structure_note = bottom_structure(df)
    breakout_price = neckline if pd.notna(neckline) else max(ma20, float(df["high"].iloc[-10:].max()))
    macd_improving = macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_hist.iloc[-1] > macd_hist.iloc[-2]
    above_confirmation = close > breakout_price

    valuation, valuation_note = valuation_score(info)
    quality, quality_note = quality_score(info)
    catalyst = 50
    if pd.notna(info["earnings_growth"]):
        catalyst += 20 if info["earnings_growth"] > .10 else -10 if info["earnings_growth"] < 0 else 0
    catalyst += 15 if cmf > .05 else -10 if cmf < -.10 else 0
    catalyst += 15 if vol_ratio >= 1.5 and close > ma20 else 0
    catalyst = float(np.clip(catalyst, 0, 100))

    technical = 0
    technical += 15 if daily_rsi < 35 else 8 if daily_rsi < 45 else 0
    technical += 15 if pd.notna(weekly_rsi) and 30 <= weekly_rsi <= 55 else 0
    technical += 20 if macd_improving else 0
    technical += 15 if close > ma20 and close > avwap else 0
    technical += 15 if higher_low else 0
    technical += 20 if confirmed_bottom else 0
    technical = float(np.clip(technical, 0, 100))

    risk = 60
    risk += 15 if close > ma200 else -20
    risk += 10 if vol_ratio >= .7 else -25
    risk += 5 if drawdown > -55 else -15
    risk = float(np.clip(risk, 0, 100))

    recent_low = float(df["low"].iloc[-20:].min())
    stop = max(min(recent_low, close - 1.5 * atr_value), close - 3 * atr_value)
    if stop >= close:
        stop = close - max(atr_value, close * .03)
    risk_per_share = close - stop
    target = close + 2 * risk_per_share
    rr = (target - close) / risk_per_share if risk_per_share > 0 else 0.0

    eligible = pd.notna(valuation) and risk >= 40 and risk_per_share / close <= .12 and vol_ratio >= .35
    if not eligible:
        action = "不合資格"
    elif valuation >= 60 and quality >= 50 and technical < 55:
        action = "觀察"
    elif valuation >= 60 and quality >= 50 and technical >= 55 and not above_confirmation:
        action = "等待突破"
    elif valuation >= 60 and quality >= 50 and technical >= 60 and above_confirmation and rr >= 2:
        action = "小量試倉"
    else:
        action = "觀察"

    total = .30 * (valuation if pd.notna(valuation) else 0) + .25 * quality + .20 * catalyst + .15 * technical + .10 * risk
    if regime == "bear_high_vol":
        total *= .90
    if regime == "bull_low_vol" and close < ma20:
        total *= .90
    total = round(float(total), 1)

    reasons = []
    missing = []
    if pd.notna(valuation) and valuation >= 65:
        reasons.append(f"{undervaluation_tier(valuation)}：估值分 {valuation:.0f}")
    elif pd.isna(valuation):
        missing.append("估值資料不足")
    if quality >= 60:
        reasons.append(f"財務質素合格：{quality:.0f} 分")
    elif quality < 50:
        missing.append("財務質素未達標")
    if daily_rsi < 35:
        reasons.append(f"日 RSI {daily_rsi:.1f}，短期超賣")
    if higher_low:
        reasons.append("形成較高低點，賣壓可能減弱")
    if macd_improving:
        reasons.append("MACD 動能改善")
    if confirmed_bottom:
        reasons.append("底部結構已突破確認")
    if not above_confirmation:
        missing.append(f"未收市突破確認價 {breakout_price:.2f}")
    if vol_ratio < 1.5:
        missing.append("成交量未達確認級別（1.5 倍均量）")
    if close < ma200:
        missing.append("仍低於 MA200，長期趨勢偏弱")

    snapshot = {
        "valuation_note": valuation_note, "quality_note": quality_note, "structure": structure_note,
        "daily_rsi": round(daily_rsi, 2), "weekly_rsi": round(weekly_rsi, 2) if pd.notna(weekly_rsi) else None,
        "cmf": round(cmf, 3), "volume_ratio": round(vol_ratio, 2), "regime": regime,
        "reasons": reasons, "missing": missing
    }
    return {
        "ticker": ticker, "name": info["name"], "market": market, "event_time": datetime.now().isoformat(timespec="seconds"),
        "price": close, "total_score": total, "action": action, "undervaluation": undervaluation_tier(valuation),
        "valuation": round(valuation, 1) if pd.notna(valuation) else np.nan, "quality": round(quality, 1),
        "catalyst": round(catalyst, 1), "technical": round(technical, 1), "risk": round(risk, 1),
        "daily_rsi": daily_rsi, "weekly_rsi": weekly_rsi, "cmf": cmf, "avwap": avwap, "vol_ratio": vol_ratio,
        "drawdown": drawdown, "breakout_price": breakout_price, "stop": stop, "target": target, "rr": rr,
        "structure": structure_note, "snapshot": snapshot, "df": df
    }


def chart_figure(r):
    df = r["df"].tail(250)
    ma20 = df["close"].rolling(20).mean()
    ma60 = df["close"].rolling(60).mean()
    ma200 = df["close"].rolling(200).mean()
    macd_line, macd_signal, macd_hist = calc_macd(df["close"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[.56, .12, .16, .16], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K線", increasing_line_color=C_GREEN, decreasing_line_color=C_RED), row=1, col=1)
    for series, label, color in [(ma20, "MA20", C_ORANGE), (ma60, "MA60", C_BLUE), (ma200, "MA200", "#bc8cff")]:
        fig.add_trace(go.Scatter(x=df.index, y=series, name=label, line=dict(color=color, width=1.1)), row=1, col=1)
    fig.add_hline(y=r["avwap"], line_dash="dot", line_color=C_GREEN, annotation_text="AVWAP", row=1, col=1)
    fig.add_hline(y=r["breakout_price"], line_dash="dash", line_color=C_ORANGE, annotation_text="確認價", row=1, col=1)
    fig.add_hline(y=r["stop"], line_dash="dot", line_color=C_RED, annotation_text="止損", row=1, col=1)
    colors = [C_GREEN if c >= o else C_RED for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rsi(df["close"]), name="RSI(14)", line=dict(color=C_ORANGE)), row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=C_GREEN, row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=C_RED, row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="MACD Hist", marker_color=[C_GREEN if x >= 0 else C_RED for x in macd_hist.fillna(0)]), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color=C_BLUE)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_signal, name="Signal", line=dict(color=C_ORANGE)), row=4, col=1)
    fig.update_layout(height=820, paper_bgcolor=C_BG, plot_bgcolor=C_BG, font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False, legend=dict(orientation="h"), margin=dict(l=5, r=5, t=35, b=5))
    return fig


def action_class(action):
    if action == "小量試倉":
        return "good-card"
    if action in ["觀察", "等待突破"]:
        return "wait-card"
    return "bad-card"


def render_decision_card(r, key_prefix):
    reasons = "<br>".join(f"✓ {x}" for x in r["snapshot"]["reasons"]) or "—"
    missing = "<br>".join(f"• {x}" for x in r["snapshot"]["missing"]) or "—"
    st.markdown(f"""
    <div class="decision-card {action_class(r['action'])}">
    <h3>{r['ticker']}　{r['action']}　|　候選分 {r['total_score']:.1f}</h3>
    <p><b>現價：</b>{r['price']:.2f}　　<b>確認價：</b>{r['breakout_price']:.2f}　　<b>止損：</b>{r['stop']:.2f}　　<b>2R 目標：</b>{r['target']:.2f}</p>
    <p><b>入選原因</b><br>{reasons}</p>
    <p><b>仍欠條件／失效風險</b><br>{missing}</p>
    </div>
    """, unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("估值", "資料不足" if pd.isna(r["valuation"]) else f"{r['valuation']:.0f}", help="Forward PE、PB、EV/EBITDA、FCF Yield 的綜合估值分數。")
    m2.metric("質素", f"{r['quality']:.0f}", help="ROE、負債及盈利增長。高分代表基本面相對穩健。")
    m3.metric("催化", f"{r['catalyst']:.0f}", help="盈利增長、資金流及量價配合。")
    m4.metric("轉勢確認", f"{r['technical']:.0f}", help="超賣、MACD 改善、AVWAP、較高低點及結構突破。")
    m5.metric("R/R", f"{r['rr']:.2f}R", help="預期目標回報相對止損風險。系統只考慮至少 2R 的試倉。")
    st.plotly_chart(chart_figure(r), use_container_width=True, key=f"{key_prefix}_plot_{r['ticker']}")


init_db()
st.title("📈 撈底監察系統 V4.1 — 決策清晰版")
st.caption("目的：找出估值具吸引力、基本面未明顯惡化、賣壓減弱並開始確認轉勢的股票；不是單純買入跌得最多的股票。")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    market_label = st.radio("市場", ["🇺🇸 美股", "🇭🇰 港股", "📋 自選"], key="market_choice")
    custom_text = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK", key="custom_symbols") if market_label == "📋 自選" else ""
    account_size = st.number_input("帳戶總值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("每筆最大風險 (%)", min_value=.25, max_value=2.0, value=1.0, step=.25) / 100
    show_all = st.checkbox("掃描頁顯示所有分析股", value=False)
    st.divider()
    st.markdown("### 決策定義")
    st.caption("不合資格：資料、風險或流動性條件不符。\n\n觀察：便宜或超賣，但未確認。\n\n等待突破：有底部條件，未突破確認價。\n\n小量試倉：確認價已突破、止損清晰及 R/R ≥ 2。")

market = "US" if "美股" in market_label else "HK"
tickers = US_WATCHLIST if market_label == "🇺🇸 美股" else HK_WATCHLIST if market_label == "🇭🇰 港股" else [x.strip().upper() for x in custom_text.splitlines() if x.strip()]
regime, bench_return, vol_value, benchmark = get_regime(market)
regime_text = {"bear_high_vol": "熊市高波動", "bull_low_vol": "牛市低波動", "neutral": "中性", "unknown": "未知"}[regime]
vol_name = "VIX" if market == "US" else "VHSI"

r1, r2, r3, r4 = st.columns(4)
r1.metric("市場", "美股" if market == "US" else "港股")
r2.metric("市場環境", regime_text)
r3.metric(f"{benchmark} 60 日回報", f"{bench_return:.1f}%")
r4.metric(vol_name, "資料不足" if pd.isna(vol_value) else f"{vol_value:.1f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 今日候選", "📊 全部掃描", "📈 技術圖表", "📐 交易計劃", "📋 訊號紀錄", "📖 指標教學"])

with tab1:
    if st.button("🔄 掃描目前名單", type="primary", key="run_v41_scan"):
        with st.spinner(f"正在分析 {len(tickers)} 隻股票..."):
            data_map = fetch_many(tickers)
            analysed = [analyse_stock(ticker, data_map.get(ticker), market, regime) for ticker in tickers]
            st.session_state["v41_results"] = [r for r in analysed if r is not None]
            st.session_state["v41_market"] = market
    results = st.session_state.get("v41_results", [])
    if not results:
        st.info("按「掃描目前名單」開始。系統預設只展示值得處理的候選股。")
    else:
        visible = results if show_all else [r for r in results if r["action"] in ["等待突破", "小量試倉"]]
        counts = pd.Series([r["action"] for r in results]).value_counts()
        st.caption(f"已分析 {len(results)} 隻｜小量試倉 {counts.get('小量試倉', 0)}｜等待突破 {counts.get('等待突破', 0)}｜觀察 {counts.get('觀察', 0)}｜不合資格 {counts.get('不合資格', 0)}")
        if not visible:
            st.warning("目前未有『等待突破』或『小量試倉』候選。這是正常結果，代表系統沒有強行給買入訊號。")
        for r in sorted(visible, key=lambda x: (x["action"] != "小量試倉", -x["total_score"])):
            render_decision_card(r, "candidate")
            if r["action"] == "小量試倉":
                if st.button(f"保存 {r['ticker']} 試倉訊號", key=f"save_signal_{r['ticker']}"):
                    save_signal(r)
                    st.success(f"已保存 {r['ticker']} 的 point-in-time 訊號快照。")

with tab2:
    results = st.session_state.get("v41_results", [])
    if results:
        table_rows = []
        for r in results:
            table_rows.append({
                "代碼": r["ticker"], "名稱": r["name"], "現價": round(r["price"], 2), "決策": r["action"],
                "候選分": r["total_score"], "低估級別": r["undervaluation"], "估值": r["valuation"],
                "質素": r["quality"], "催化": r["catalyst"], "轉勢": r["technical"], "風險": r["risk"],
                "日RSI": round(r["daily_rsi"], 1), "真周RSI": round(r["weekly_rsi"], 1) if pd.notna(r["weekly_rsi"]) else np.nan,
                "量比": round(r["vol_ratio"], 2), "距52周高%": round(r["drawdown"], 1), "確認價": round(r["breakout_price"], 2), "R/R": round(r["rr"], 2)
            })
        table = pd.DataFrame(table_rows)
        st.dataframe(table.sort_values("候選分", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("下載 CSV", table.to_csv(index=False).encode("utf-8-sig"), "v41_scan_results.csv", "text/csv", key="download_v41_results")
    else:
        st.info("請先完成掃描。")

with tab3:
    results = st.session_state.get("v41_results", [])
    if results:
        chosen = st.selectbox("選擇股票", [r["ticker"] for r in results], key="technical_chart_choice")
        selected = next(r for r in results if r["ticker"] == chosen)
        render_decision_card(selected, "technical_tab")
    else:
        st.info("請先完成掃描。")

with tab4:
    ticker_input = st.text_input("股票代碼", "AAPL", key="trade_plan_symbol").upper()
    if st.button("建立交易計劃", key="build_trade_plan"):
        df_input = fetch_ohlcv(ticker_input)
        plan = analyse_stock(ticker_input, df_input, "HK" if ticker_input.endswith(".HK") else "US", regime)
        if plan is None:
            st.error("資料不足，無法建立交易計劃。")
        else:
            cash_risk = account_size * risk_pct
            per_share_risk = max(plan["price"] - plan["stop"], .0001)
            shares_by_risk = int(cash_risk / per_share_risk)
            shares_by_cash = int(account_size / plan["price"])
            shares = min(shares_by_risk, shares_by_cash)
            st.markdown(f"## {ticker_input} — {plan['action']}")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("參考現價", f"{plan['price']:.2f}")
            p2.metric("確認買入價", f"{plan['breakout_price']:.2f}")
            p3.metric("結構止損", f"{plan['stop']:.2f}")
            p4.metric("2R 目標", f"{plan['target']:.2f}")
            p5.metric("最大股數", f"{shares:,}")
            st.write(f"每股風險：{per_share_risk:.2f} ｜ 單筆最大風險：{cash_risk:,.2f} ｜ 最大資金使用：約 {shares * plan['price']:,.2f}")
            st.warning("港股請按每手股數、港元／美元匯率及實際交易成本向下調整；本頁只提供風險框架，不構成投資建議。")

with tab5:
    events = load_signals()
    if events.empty:
        st.info("尚未有保存的試倉訊號。只有「小量試倉」才可由今日候選頁保存。")
    else:
        show_events = events.drop(columns=["snapshot_json"])
        st.dataframe(show_events, use_container_width=True, hide_index=True)
        st.caption("這是 point-in-time 訊號資料庫。完成一段時間的日常掃描後，可擴展成含下一日入場、止損、目標、成本與基準比較的 walk-forward 回測。")

with tab6:
    st.markdown("## 先掌握五個核心概念")
    teach = pd.DataFrame([
        ["RSI", "近期升跌速度是否過急", "低於 30 代表短期跌得急；只用作尋找觀察名單，不能單獨當買入理由。"],
        ["真周 RSI", "中期強弱", "以每周收市價計算，比日 RSI 慢；用來避免在長期弱勢中過早撈底。"],
        ["成交量／量比", "今日參與者是否比平日多", "突破確認價時最好見到約 1.5 倍或以上均量。放量下跌則可能是恐慌拋售。"],
        ["MA20／MA60／MA200", "短、中、長期平均成本", "站回 MA20 是初步改善；低於 MA200 代表長期趨勢仍要保守。"],
        ["AVWAP", "由重要低點起計的平均持貨成本", "股價站上 AVWAP，代表自該低點買入的大部分資金可能不再虧損。"],
        ["MACD", "短期動能相對中期動能的改變", "低位改善可支持『跌勢減弱』判斷，但仍需價格突破確認。"],
        ["CMF", "收市位置與成交量推算的買賣壓力", "正數偏向買方主動；負數偏向沽壓。只作資金流輔助。"],
        ["Higher Low", "第二個低點未明顯低於第一個低點", "賣方未能再壓低價格，是比單純超賣更有用的結構訊號。"],
        ["確認價／頸線", "底部結構中間反彈高點", "收市突破才代表底部可能確認；未突破前只屬觀察。"],
        ["止損與 2R", "錯了在哪裡退出，以及對了可賺多少", "只考慮預期回報至少是風險兩倍的交易；R/R 是風控，不是預測。"],
        ["Forward PE", "股價相對預測盈利", "低 PE 未必便宜；要配合盈利是否可維持、行業比較及財務質素。"],
        ["PB／ROE／負債", "資產估值、資本效率與財務壓力", "PB 對銀行保險較有用；高 ROE 但負債很高時要特別小心。"],
    ], columns=["指標", "白話意思", "正確使用方法"])
    st.dataframe(teach, use_container_width=True, hide_index=True)
    st.info("使用順序：先看『是否合資格』→ 再看『估值與質素』→ 最後才以 RSI、MACD、量能和突破決定入場時機。不要因為一個指標超賣而直接買入。")
