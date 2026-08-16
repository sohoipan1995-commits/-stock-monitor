from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from supabase import Client, create_client

st.set_page_config(page_title="股票監察系統 Pro · 雲端版", page_icon="📈", layout="wide")

MODEL_VERSION = "2.1.2"
WEB_SCAN_LIMIT = 200
DEFAULT_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM", "SPY", "QQQ"]
DEFAULT_HK = ["0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "2318.HK", "9988.HK", "3690.HK", "9618.HK", "1211.HK"]
SOURCE_LABELS = {
    "manual": "手動永久", "portfolio": "持倉", "signal_high_score": "高分訊號",
    "sp500_constituent": "S&P 500", "hsi_constituent": "恒指成分",
    "hsi_top30_turnover": "恒指20日成交額 Top 30",
}


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(" ", "")


def ticker_market(ticker: str) -> str:
    return "HK" if normalize_ticker(ticker).endswith(".HK") else "US"


@st.cache_resource
def get_supabase() -> Client | None:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SECRET_KEY"])
    except Exception:
        return None


def set_db_error(where: str, exc: Exception) -> None:
    st.session_state["db_error"] = f"{where}: {type(exc).__name__}: {exc}"


def db_upsert(table: str, data: dict | list[dict], conflict: str) -> bool:
    client = get_supabase()
    if client is None:
        return False
    try:
        client.table(table).upsert(data, on_conflict=conflict).execute()
        return True
    except Exception as exc:
        set_db_error(f"寫入 {table}", exc)
        return False


def upsert_instruments(rows: list[dict]) -> bool:
    now = datetime.utcnow().isoformat()
    payload = []
    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        market = row.get("market") or ticker_market(ticker)
        payload.append({
            "ticker": ticker, "market": market, "name": row.get("name"),
            "sector": row.get("sector"), "industry": row.get("industry"),
            "currency": "HKD" if market == "HK" else "USD", "is_active": True,
            "updated_at": now,
        })
    return db_upsert("instruments", payload, "ticker") if payload else True


def add_membership(ticker: str, source: str, permanent: bool, priority: int, notes: str | None = None) -> bool:
    ticker = normalize_ticker(ticker)
    if not upsert_instruments([{"ticker": ticker, "market": ticker_market(ticker)}]):
        return False
    return db_upsert("watchlist_memberships", {
        "ticker": ticker, "source": source, "is_permanent": permanent, "is_active": True,
        "priority": priority, "notes": notes, "last_confirmed_at": datetime.utcnow().isoformat(),
        "removed_at": None,
    }, "ticker,source")


def deactivate_missing(source: str, active: set[str]) -> None:
    client = get_supabase()
    if client is None:
        return
    try:
        existing = client.table("watchlist_memberships").select("id,ticker").eq("source", source).eq("is_active", True).execute().data or []
        for row in existing:
            if row["ticker"] not in active:
                client.table("watchlist_memberships").update({"is_active": False, "removed_at": datetime.utcnow().isoformat()}).eq("id", row["id"]).execute()
    except Exception as exc:
        set_db_error("更新候選狀態", exc)


def get_watchlist(market: str | None = None) -> pd.DataFrame:
    client = get_supabase()
    if client is None:
        return pd.DataFrame()
    try:
        memberships = client.table("watchlist_memberships").select("ticker,source,is_permanent,priority,notes,added_at,last_confirmed_at").eq("is_active", True).execute().data or []
        instruments = client.table("instruments").select("ticker,market,name,sector").eq("is_active", True).execute().data or []
        if not memberships:
            return pd.DataFrame()
        result = pd.DataFrame(memberships).merge(pd.DataFrame(instruments), on="ticker", how="left")
        if market:
            result = result[result["market"] == market]
        return result.sort_values(["is_permanent", "priority", "ticker"], ascending=[False, False, True])
    except Exception as exc:
        set_db_error("讀取觀察名單", exc)
        return pd.DataFrame()


def validate_ohlcv(df: pd.DataFrame | None) -> tuple[bool, str]:
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        return False, "沒有取得價格資料"
    if required - set(df.columns) or len(df) < 80:
        return False, "OHLCV 欄位不完整或資料不足 80 日"
    if df[list(required)].isna().any().any() or (df["close"] <= 0).any() or (df["volume"] < 0).any():
        return False, "OHLCV 有遺漏或不合理值"
    return True, "OK"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "3y") -> tuple[pd.DataFrame | None, dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    errors = []
    for attempt in range(3):
        try:
            raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(col).lower() for col in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            valid, message = validate_ohlcv(df)
            return (df if valid else None), {"ticker": ticker, "source": "Yahoo Finance", "adjusted": True, "last_bar": str(df.index[-1].date()) if not df.empty else None, "rows": len(df), "status": message}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(.8 * (2 ** attempt))
    return None, {"ticker": ticker, "source": "Yahoo Finance", "last_bar": None, "rows": 0, "status": "；".join(errors)}


def clean_us_ticker(value: Any) -> str:
    return normalize_ticker(value).replace(".", "-")


def clean_hk_ticker(value: Any) -> str | None:
    digits = "".join(char for char in str(value) if char.isdigit())
    return f"{digits.zfill(4)}.HK" if digits else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "StockMonitorPro/2.1.2 research application", "Accept-Language": "en-US,en;q=0.9"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
    if not tables:
        raise RuntimeError("找不到 S&P 500 成分表")
    table = tables[0]
    if not {"Symbol", "Security"}.issubset(table.columns):
        raise RuntimeError("S&P 500 成分表格式已改變")
    return pd.DataFrame({
        "ticker": table["Symbol"].map(clean_us_ticker), "name": table["Security"].astype(str),
        "sector": table["GICS Sector"].astype(str) if "GICS Sector" in table.columns else None,
        "industry": table["GICS Sub-Industry"].astype(str) if "GICS Sub-Industry" in table.columns else None,
        "market": "US",
    }).drop_duplicates("ticker")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hsi_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "StockMonitorPro/2.1.2 research application", "Accept-Language": "en-US,en;q=0.9"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    candidate = None
    for table in tables:
        cols = [str(col).lower() for col in table.columns]
        if len(table) >= 40 and any("code" in col or "ticker" in col for col in cols):
            candidate = table
            break
    if candidate is None:
        raise RuntimeError("找不到恒指成分表")
    code_col = next(col for col in candidate.columns if "code" in str(col).lower() or "ticker" in str(col).lower())
    name_col = next((col for col in candidate.columns if "company" in str(col).lower() or "name" in str(col).lower()), None)
    sector_col = next((col for col in candidate.columns if "sector" in str(col).lower() or "industry" in str(col).lower()), None)
    output = pd.DataFrame({
        "ticker": candidate[code_col].map(clean_hk_ticker),
        "name": candidate[name_col].astype(str) if name_col else None,
        "sector": candidate[sector_col].astype(str) if sector_col else None,
        "industry": None, "market": "HK",
    }).dropna(subset=["ticker"]).drop_duplicates("ticker")
    if len(output) < 40:
        raise RuntimeError("恒指成分資料不完整")
    return output


def sync_sp500() -> tuple[bool, str]:
    try:
        universe = fetch_sp500_constituents()
        if not upsert_instruments(universe.to_dict("records")):
            return False, "無法寫入 instruments"
        today, now = date.today().isoformat(), datetime.utcnow().isoformat()
        memberships = [{"ticker": ticker, "source": "sp500_constituent", "is_permanent": False, "is_active": True, "priority": 30, "last_confirmed_at": now, "removed_at": None} for ticker in universe["ticker"]]
        snapshots = [{"snapshot_date": today, "ticker": ticker, "source": "sp500_constituent", "index_member": True, "is_selected": True} for ticker in universe["ticker"]]
        if not db_upsert("watchlist_memberships", memberships, "ticker,source"):
            return False, st.session_state.get("db_error", "寫入觀察名單失敗")
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        deactivate_missing("sp500_constituent", set(universe["ticker"]))
        return True, f"已同步 {len(universe)} 隻 S&P 500 成分股。"
    except Exception as exc:
        return False, f"S&P 500 同步失敗：{type(exc).__name__}: {exc}"


def sync_hsi_top30() -> tuple[bool, str]:
    try:
        universe = fetch_hsi_constituents()
        if not upsert_instruments(universe.to_dict("records")):
            return False, "無法寫入 instruments"
        today, now = date.today().isoformat(), datetime.utcnow().isoformat()
        members = [{"ticker": ticker, "source": "hsi_constituent", "is_permanent": False, "is_active": True, "priority": 40, "last_confirmed_at": now, "removed_at": None} for ticker in universe["ticker"]]
        snapshots = [{"snapshot_date": today, "ticker": ticker, "source": "hsi_constituent", "index_member": True, "is_selected": True} for ticker in universe["ticker"]]
        db_upsert("watchlist_memberships", members, "ticker,source")
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_constituent", set(universe["ticker"]))
        turnover = []
        for ticker in universe["ticker"]:
            df, _ = fetch_ohlcv(ticker, "6mo")
            if df is not None and len(df) >= 20:
                turnover.append({"ticker": ticker, "turnover_20d": float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean()), "average_volume_20d": float(df["volume"].iloc[-20:].mean()), "close_price": float(df["close"].iloc[-1])})
        ranking = pd.DataFrame(turnover).sort_values("turnover_20d", ascending=False).head(30).reset_index(drop=True)
        if ranking.empty:
            return False, "未取得足夠恒指成分股價格資料"
        ranking["turnover_rank"] = ranking.index + 1
        top_members = [{"ticker": row.ticker, "source": "hsi_top30_turnover", "is_permanent": False, "is_active": True, "priority": 90, "last_confirmed_at": now, "removed_at": None} for row in ranking.itertuples()]
        top_snapshots = [{"snapshot_date": today, "ticker": row.ticker, "source": "hsi_top30_turnover", "index_member": True, "turnover_20d": row.turnover_20d, "turnover_rank": int(row.turnover_rank), "average_volume_20d": row.average_volume_20d, "close_price": row.close_price, "is_selected": True} for row in ranking.itertuples()]
        db_upsert("watchlist_memberships", top_members, "ticker,source")
        db_upsert("watchlist_snapshots", top_snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_top30_turnover", set(ranking["ticker"]))
        return True, f"已同步 {len(universe)} 隻恒指成分股，並選出 20 日平均成交額 Top {len(ranking)}。"
    except Exception as exc:
        return False, f"恒指／Top 30 同步失敗：{type(exc).__name__}: {exc}"


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - previous).abs(), (df["low"] - previous).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((2 * df["close"] - df["high"] - df["low"]) / spread).fillna(0)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def weekly_rsi(close: pd.Series) -> pd.Series:
    return rsi(close.resample("W-FRI").last().dropna()).reindex(close.index, method="ffill")


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    return (series - series.rolling(period).mean()) / series.rolling(period).std(ddof=0).replace(0, np.nan)


def market_regime(market: str) -> tuple[str, dict[str, float]]:
    benchmark, vol_ticker = ("SPY", "^VIX") if market == "US" else ("^HSI", "^VHSI")
    px, _ = fetch_ohlcv(benchmark, "1y")
    vol, _ = fetch_ohlcv(vol_ticker, "1y")
    if px is None or len(px) < 70:
        return "unknown", {"return_60d": np.nan, "volatility": np.nan, "vol_index": np.nan}
    ret60 = 100 * (px["close"].iloc[-1] / px["close"].iloc[-61] - 1)
    annual_vol = 100 * px["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_index = float(vol["close"].iloc[-1]) if vol is not None else np.nan
    high_vol = vol_index >= (25 if market == "US" else 30)
    if ret60 >= 5:
        state = "bull_high_vol" if high_vol else "bull_low_vol"
    elif ret60 <= -5:
        state = "bear_high_vol" if high_vol else "bear_low_vol"
    else:
        state = "neutral_high_vol" if high_vol else "neutral"
    return state, {"return_60d": ret60, "volatility": annual_vol, "vol_index": vol_index}


@dataclass
class ScoreResult:
    score: float
    label: str
    regime: str
    price: float
    stop: float
    target: float
    factors: dict[str, float]
    explanations: list[str]


def score_stock(df: pd.DataFrame, regime: str) -> ScoreResult:
    close, volume = df["close"], df["volume"]
    price = float(close.iloc[-1])
    rd, rw = rsi(close), weekly_rsi(close)
    ma20, ma60, ma200 = close.rolling(20).mean(), close.rolling(60).mean(), close.rolling(200).mean()
    ml, ms, _ = macd(close)
    a, vz, flow, vw = atr(df), zscore(volume), cmf(df), rolling_vwap(df)
    if any(pd.isna(x.iloc[-1]) for x in [rd, rw, ma60, ma200, a, vz, flow, vw]):
        raise ValueError("指標暖機資料不足")
    factors = {"reversal": 0.0, "trend": 0.0, "flow": 0.0, "risk": 0.0}
    notes = []
    if rd.iloc[-1] <= 30:
        factors["reversal"] += 15; notes.append(f"日 RSI {rd.iloc[-1]:.1f} 超賣")
    elif rd.iloc[-1] <= 38:
        factors["reversal"] += 8; notes.append(f"日 RSI {rd.iloc[-1]:.1f} 偏低")
    if rw.iloc[-1] <= 42:
        factors["reversal"] += 8; notes.append(f"真週 RSI {rw.iloc[-1]:.1f} 偏低")
    if ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-1] < 0:
        factors["reversal"] += 7; notes.append("MACD 零軸下金叉")
    factors["reversal"] = min(factors["reversal"], 30)
    if price > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-6]:
        factors["trend"] += 10; notes.append("站上 MA20 且上彎")
    if price > ma60.iloc[-1]:
        factors["trend"] += 8; notes.append("位於 MA60 之上")
    if ma200.iloc[-1] > ma200.iloc[-21]:
        factors["trend"] += 7; notes.append("MA200 上升")
    elif price < ma200.iloc[-1]:
        notes.append("仍低於 MA200")
    factors["trend"] = min(factors["trend"], 25)
    if flow.iloc[-1] > .08:
        factors["flow"] += 10; notes.append(f"CMF {flow.iloc[-1]:.2f} 流入")
    elif flow.iloc[-1] < -.08:
        notes.append(f"CMF {flow.iloc[-1]:.2f} 流出")
    if vz.iloc[-1] >= 1.5 and price > vw.iloc[-1] and close.iloc[-1] > df["open"].iloc[-1]:
        factors["flow"] += 10; notes.append(f"放量收陽 Z={vz.iloc[-1]:.1f}")
    factors["flow"] = min(factors["flow"], 20)
    atr_pct = float(a.iloc[-1] / price * 100)
    if atr_pct <= 4:
        factors["risk"] += 10
    elif atr_pct <= 7:
        factors["risk"] += 5
    else:
        notes.append(f"ATR 高波動 {atr_pct:.1f}%")
    if regime.startswith("bear"):
        notes.append("熊市 regime：不放大撈底分數")
    elif regime.startswith("bull"):
        factors["risk"] += 3
    factors["risk"] = min(factors["risk"], 15)
    stop = round(min(float(df["low"].iloc[-10:].min()), price - .5 * float(a.iloc[-1])), 3)
    target = round(price + 1.8 * (price - stop), 3)
    total = round(sum(factors.values()), 1)
    label = "強烈關注" if total >= 75 else "值得關注" if total >= 60 else "觀察中" if total >= 45 else "未觸發"
    return ScoreResult(total, label, regime, round(price, 3), stop, target, factors, notes)


def create_scan_run(market: str, count: int) -> int | None:
    client = get_supabase()
    if client is None:
        return None
    try:
        response = client.table("scan_runs").insert({"run_type": "manual", "market": market, "universe_size": count, "model_version": MODEL_VERSION}).execute()
        return response.data[0]["id"]
    except Exception as exc:
        set_db_error("建立掃描紀錄", exc)
        return None


def finish_scan_run(run_id: int | None, success: int, failed: int, errors: list[str]) -> None:
    client = get_supabase()
    if client is None or run_id is None:
        return
    try:
        client.table("scan_runs").update({"completed_at": datetime.utcnow().isoformat(), "status": "completed" if failed == 0 else "completed_with_errors", "success_count": success, "failed_count": failed, "error_summary": " | ".join(errors[:10]) or None}).eq("id", run_id).execute()
    except Exception as exc:
        set_db_error("完成掃描紀錄", exc)


def persist_scan(run_id: int | None, ticker: str, result: ScoreResult, meta: dict) -> None:
    if run_id is None:
        return
    db_upsert("scan_results", {"scan_run_id": run_id, "ticker": ticker, "price": result.price, "score": result.score, "label": result.label, "regime": result.regime, "stop_price": result.stop, "target_price": result.target, "factors": result.factors, "explanations": result.explanations, "data_source": meta["source"], "last_bar_date": meta["last_bar"]}, "scan_run_id,ticker")


def persist_signal(ticker: str, signal_date: str, result: ScoreResult, meta: dict) -> None:
    add_membership(ticker, "signal_high_score", False, 80, "由高分訊號自動加入")
    db_upsert("signals", {"signal_date": signal_date, "ticker": ticker, "model_version": MODEL_VERSION, "price": result.price, "score": result.score, "label": result.label, "regime": result.regime, "stop_price": result.stop, "target_price": result.target, "factors": result.factors, "explanations": result.explanations, "data_source": meta["source"], "last_bar_date": meta["last_bar"]}, "signal_date,ticker,model_version")


def make_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="價格"))
    for n, color in [(20, "#f59e0b"), (60, "#3b82f6"), (200, "#a855f7")]:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"].rolling(n).mean(), name=f"MA{n}", line={"width": 1.2, "color": color}))
    fig.update_layout(template="plotly_dark", height=560, title=f"{ticker}｜調整後日線", xaxis_rangeslider_visible=False)
    return fig


st.title("📈 股票監察系統 Pro · 雲端觀察名單版")
st.caption("模型 2.1.2｜S&P 500 全成分股＋恒指成分股 20 日平均成交額 Top 30｜評分只供研究與篩選，不構成投資建議。")
client = get_supabase()
if client is None:
    st.error("未偵測到 Supabase Secrets。請檢查 Streamlit Cloud 的 SUPABASE_URL 與 SUPABASE_SECRET_KEY。")
else:
    st.success("Supabase 已連線：觀察名單、掃描及訊號將保存至雲端資料庫。")

with st.sidebar:
    st.header("控制面板")
    market_label = st.radio("市場", ["美股", "港股", "自選"], index=0)
    market = "US" if market_label == "美股" else "HK"
    custom = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK") if market_label == "自選" else ""
    minimum_score = st.slider("最低訊號分數", 0, 90, 60)

if market_label == "自選":
    tickers = list(dict.fromkeys(normalize_ticker(x) for x in custom.splitlines() if x.strip()))
else:
    saved = get_watchlist(market)
    tickers = saved["ticker"].drop_duplicates().tolist() if not saved.empty else (DEFAULT_US if market == "US" else DEFAULT_HK)
regime, stats = market_regime(market)
regime_names = {"bull_low_vol": "牛市／低波動", "bull_high_vol": "牛市／高波動", "bear_low_vol": "熊市／低波動", "bear_high_vol": "熊市／高波動", "neutral": "中性", "neutral_high_vol": "中性／高波動", "unknown": "未能判定"}
a, b, c, d = st.columns(4)
a.metric("市場 regime", regime_names.get(regime, regime))
b.metric("60 日基準回報", "—" if pd.isna(stats["return_60d"]) else f"{stats['return_60d']:.1f}%")
c.metric("20 日年化波動", "—" if pd.isna(stats["volatility"]) else f"{stats['volatility']:.1f}%")
d.metric("波動率指數", "—" if pd.isna(stats["vol_index"]) else f"{stats['vol_index']:.1f}")

watch_tab, scan_tab, detail_tab, logs_tab, risk_tab = st.tabs(["📌 觀察名單管理", "📊 掃描", "📈 個股詳情", "🗂️ 訊號紀錄", "⚖️ 風控"])

with watch_tab:
    st.subheader("永久觀察名單與自動候選池")
    x1, x2, x3 = st.columns(3)
    if x1.button("同步 S&P 500 全成分股", type="primary", disabled=client is None):
        with st.spinner("正在同步 S&P 500 成分表…"):
            ok, message = sync_sp500()
        st.success(message) if ok else st.error(message)
    if x2.button("同步恒指及成交額 Top 30", type="primary", disabled=client is None):
        with st.spinner("正在同步恒指成分並計算 20 日平均成交額；首次可能需要數分鐘…"):
            ok, message = sync_hsi_top30()
        st.success(message) if ok else st.error(message)
    x3.info("手動與持倉名單永久保留；自動候選跌出指數或 Top 30 時只會停用來源，不會刪除歷史。")
    st.divider()
    m1, m2, m3, m4 = st.columns([2, 1, 1, 3])
    manual_ticker = m1.text_input("代碼", placeholder="AAPL 或 0700.HK")
    manual_source = m2.selectbox("類別", ["manual", "portfolio"], format_func=lambda v: SOURCE_LABELS[v])
    priority = m3.slider("優先級", 1, 100, 100)
    notes = m4.text_input("備註", placeholder="已持有／等業績")
    if st.button("加入並永久保存", disabled=client is None):
        if not manual_ticker.strip():
            st.warning("請輸入股票代碼。")
        elif add_membership(manual_ticker, manual_source, True, priority, notes):
            st.success(f"已永久加入 {normalize_ticker(manual_ticker)}。")
            st.rerun()
        else:
            st.error(st.session_state.get("db_error", "寫入失敗"))
    members = get_watchlist()
    if members.empty:
        st.info("尚未有雲端觀察名單。請同步指數成分或手動加入。")
    else:
        display = members.copy()
        display["來源"] = display["source"].map(SOURCE_LABELS).fillna(display["source"])
        display["永久"] = display["is_permanent"].map({True: "是", False: "否"})
        st.dataframe(display[["ticker", "market", "name", "sector", "來源", "永久", "priority", "notes", "added_at", "last_confirmed_at"]], use_container_width=True, hide_index=True)

with scan_tab:
    st.subheader(f"{market_label} 觀察名單掃描")
    if not tickers:
        st.warning("目前沒有可掃描的股票。請先同步或手動加入。")
    else:
        if len(tickers) > WEB_SCAN_LIMIT:
            st.warning(f"目前有 {len(tickers)} 隻股票；網頁手動掃描每次最多 {WEB_SCAN_LIMIT} 隻，以降低限流及 Cloud 超時風險。")
        max_limit = min(len(tickers), WEB_SCAN_LIMIT)
        scan_count = st.number_input("本次掃描數量", min_value=1, max_value=max_limit, value=max_limit, step=1)
        selected = tickers[:int(scan_count)]
        st.caption(f"目前可掃描：{len(tickers)} 隻｜本次掃描：{len(selected)} 隻｜網頁上限：{WEB_SCAN_LIMIT} 隻")
        if st.button("開始掃描", type="primary"):
            run_id = create_scan_run(market if market_label != "自選" else "BOTH", len(selected))
            rows, failures, errors = [], [], []
            progress, status = st.progress(0), st.empty()
            for index, ticker in enumerate(selected, 1):
                status.caption(f"正在掃描 {index}/{len(selected)}：{ticker}")
                df, meta = fetch_ohlcv(ticker)
                if df is None:
                    failures.append({"代碼": ticker, "原因": meta["status"]}); errors.append(f"{ticker}: {meta['status']}")
                else:
                    try:
                        result = score_stock(df, regime)
                        persist_scan(run_id, ticker, result, meta)
                        rr = round((result.target - result.price) / (result.price - result.stop), 2) if result.price > result.stop else None
                        rows.append({"代碼": ticker, "現價": result.price, "總分": result.score, "標籤": result.label, "止損": result.stop, "目標": result.target, "R/R": rr, "資料最後日": meta["last_bar"], "因子": " / ".join(f"{k}:{v:.0f}" for k, v in result.factors.items()), "說明": "；".join(result.explanations)})
                        if result.score >= minimum_score:
                            persist_signal(ticker, str(df.index[-1].date()), result, meta)
                    except Exception as exc:
                        failures.append({"代碼": ticker, "原因": f"評分失敗：{type(exc).__name__}: {exc}"}); errors.append(f"{ticker}: {exc}")
                progress.progress(index / len(selected))
            status.empty()
            finish_scan_run(run_id, len(rows), len(failures), errors)
            if rows:
                results = pd.DataFrame(rows).sort_values(["總分", "代碼"], ascending=[False, True])
                st.success(f"掃描完成：成功 {len(rows)} 隻｜失敗 {len(failures)} 隻")
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button("下載掃描 CSV", results.to_csv(index=False).encode("utf-8-sig"), "scan_results.csv", "text/csv")
            if failures:
                st.warning(f"{len(failures)} 隻未完成掃描。")
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

with detail_tab:
    ticker = st.text_input("代碼", tickers[0] if tickers else "AAPL")
    period = st.selectbox("圖表範圍", ["1y", "2y", "3y", "5y"], index=1)
    if st.button("載入個股"):
        df, meta = fetch_ohlcv(ticker, period)
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("總分", f"{result.score:.1f}/90", result.label); q2.metric("現價", result.price)
            q3.metric("結構止損", result.stop); q4.metric("最低 1.8R 目標", result.target)
            st.caption(f"資料：{meta['source']}｜最後 bar：{meta['last_bar']}｜調整後價格：{meta['adjusted']}")
            st.plotly_chart(make_chart(df, normalize_ticker(ticker)), use_container_width=True)
            st.dataframe(pd.DataFrame([result.factors]), use_container_width=True, hide_index=True)
            st.write("；".join(result.explanations) if result.explanations else "目前沒有額外確認條件。")

with logs_tab:
    st.subheader("雲端訊號紀錄")
    if client is None:
        st.error("Supabase 未連線。")
    else:
        try:
            signals = client.table("signals").select("signal_date,ticker,price,score,label,regime,stop_price,target_price,data_source,last_bar_date,created_at").order("signal_date", desc=True).order("score", desc=True).limit(1000).execute().data or []
            if signals:
                signal_df = pd.DataFrame(signals)
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
                st.download_button("下載訊號 CSV", signal_df.to_csv(index=False).encode("utf-8-sig"), "cloud_signals.csv", "text/csv")
            else:
                st.info("尚未有保存的訊號，請先掃描。")
        except Exception as exc:
            st.error(f"讀取訊號失敗：{type(exc).__name__}: {exc}")

with risk_tab:
    st.subheader("風險為本的部位計算")
    r1, r2, r3, r4 = st.columns(4)
    account = r1.number_input("帳戶淨值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = r2.slider("每筆最大帳戶風險 (%)", .25, 2.0, .75, .25)
    allocation = r3.slider("單一持倉最大名義比例 (%)", 1.0, 30.0, 10.0, 1.0)
    risk_ticker = r4.text_input("代碼", tickers[0] if tickers else "AAPL", key="risk_ticker")
    if st.button("計算可承受部位"):
        df, meta = fetch_ohlcv(risk_ticker, "2y")
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime)
            per_share_risk = result.price - result.stop
            budget = account * risk_pct / 100
            by_risk = int(budget / per_share_risk) if per_share_risk > 0 else 0
            by_cap = int((account * allocation / 100) / result.price)
            shares = max(0, min(by_risk, by_cap))
            z1, z2, z3, z4 = st.columns(4)
            z1.metric("入場參考價", result.price); z2.metric("結構止損", result.stop)
            z3.metric("每股風險", f"{per_share_risk:.3f}"); z4.metric("建議上限股數", shares)
            st.write(f"風險預算：{budget:,.2f}｜名義金額：約 {shares * result.price:,.2f}｜模型目標：{result.target}")
            st.warning("下單前請按整手規則向下調整，並自行考慮匯率、手續費、價差、稅項與停損成交風險。")
