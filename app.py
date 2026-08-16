from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from supabase import Client, create_client

st.set_page_config(page_title="股票監察系統 Pro · 雲端版", page_icon="📈", layout="wide")

MODEL_VERSION = "2.1.1"
DEFAULT_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM", "SPY", "QQQ"]
DEFAULT_HK = ["0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "2318.HK", "9988.HK", "3690.HK", "9618.HK", "1211.HK"]
SOURCE_LABELS = {
    "manual": "手動永久",
    "portfolio": "持倉",
    "signal_high_score": "高分訊號",
    "sp500_constituent": "S&P 500",
    "hsi_constituent": "恒指成分",
    "hsi_top30_turnover": "恒指20日成交額 Top 30",
}


# ==================== Supabase ====================
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
    if not rows:
        return True
    now = datetime.utcnow().isoformat()
    payload = []
    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        market = row.get("market") or ticker_market(ticker)
        payload.append({
            "ticker": ticker,
            "market": market,
            "name": row.get("name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "currency": "HKD" if market == "HK" else "USD",
            "is_active": True,
            "updated_at": now,
        })
    return db_upsert("instruments", payload, "ticker")


def add_membership(ticker: str, source: str, permanent: bool, priority: int, notes: str | None = None) -> bool:
    ticker = normalize_ticker(ticker)
    if not upsert_instruments([{"ticker": ticker, "market": ticker_market(ticker)}]):
        return False
    return db_upsert(
        "watchlist_memberships",
        {
            "ticker": ticker,
            "source": source,
            "is_permanent": permanent,
            "is_active": True,
            "priority": priority,
            "notes": notes,
            "last_confirmed_at": datetime.utcnow().isoformat(),
            "removed_at": None,
        },
        "ticker,source",
    )


def deactivate_missing(source: str, active_tickers: set[str]) -> None:
    client = get_supabase()
    if client is None:
        return
    try:
        existing = client.table("watchlist_memberships").select("id,ticker").eq("source", source).eq("is_active", True).execute().data or []
        for row in existing:
            if row["ticker"] not in active_tickers:
                client.table("watchlist_memberships").update({
                    "is_active": False,
                    "removed_at": datetime.utcnow().isoformat(),
                }).eq("id", row["id"]).execute()
    except Exception as exc:
        set_db_error("更新非現役候選", exc)


def get_watchlist(market: str | None = None) -> pd.DataFrame:
    client = get_supabase()
    if client is None:
        return pd.DataFrame()
    try:
        memberships = client.table("watchlist_memberships").select(
            "ticker,source,is_permanent,is_active,priority,notes,added_at,last_confirmed_at"
        ).eq("is_active", True).execute().data or []
        instruments = client.table("instruments").select("ticker,market,name,sector").eq("is_active", True).execute().data or []
        if not memberships:
            return pd.DataFrame()
        output = pd.DataFrame(memberships).merge(pd.DataFrame(instruments), on="ticker", how="left")
        if market in {"US", "HK"}:
            output = output[output["market"] == market]
        return output.sort_values(["is_permanent", "priority", "ticker"], ascending=[False, False, True])
    except Exception as exc:
        set_db_error("讀取觀察名單", exc)
        return pd.DataFrame()


# ==================== Price data ====================
def validate_ohlcv(df: pd.DataFrame | None) -> tuple[bool, str]:
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        return False, "沒有取得價格資料"
    if required - set(df.columns):
        return False, "OHLCV 欄位不完整"
    if len(df) < 80:
        return False, "歷史資料少於 80 個交易日"
    if df[list(required)].isna().any().any():
        return False, "OHLCV 有遺漏值"
    if (df["close"] <= 0).any() or (df["volume"] < 0).any():
        return False, "價格或成交量不合理"
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
            meta = {
                "ticker": ticker,
                "source": "Yahoo Finance",
                "adjusted": True,
                "last_bar": str(df.index[-1].date()) if not df.empty else None,
                "rows": len(df),
                "status": message,
            }
            return (df if valid else None), meta
        except Exception as exc:
            errors.append(f"第{attempt + 1}次 {type(exc).__name__}: {exc}")
            time.sleep(0.8 * (2 ** attempt))
    return None, {"ticker": ticker, "source": "Yahoo Finance", "status": "；".join(errors), "rows": 0}


# ==================== Constituents and watchlist sync ====================
def clean_us_ticker(value: Any) -> str:
    return normalize_ticker(value).replace(".", "-")


def clean_hk_ticker(value: Any) -> str | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return f"{digits.zfill(4)}.HK" if digits else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    table = next((item for item in tables if "Symbol" in item.columns and "Security" in item.columns), None)
    if table is None:
        raise RuntimeError("找不到 S&P 500 成分表")
    return pd.DataFrame({
        "ticker": table["Symbol"].map(clean_us_ticker),
        "name": table["Security"].astype(str),
        "sector": table["GICS Sector"].astype(str) if "GICS Sector" in table.columns else None,
        "industry": table["GICS Sub-Industry"].astype(str) if "GICS Sub-Industry" in table.columns else None,
        "market": "US",
    }).drop_duplicates("ticker")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hsi_constituents() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Hang_Seng_Index")
    candidate = None
    for table in tables:
        columns = [str(col).lower() for col in table.columns]
        if len(table) >= 40 and any("code" in col or "ticker" in col for col in columns):
            candidate = table
            break
    if candidate is None:
        raise RuntimeError("找不到恒指成分表，請稍後重試")
    code_col = next(col for col in candidate.columns if "code" in str(col).lower() or "ticker" in str(col).lower())
    name_col = next((col for col in candidate.columns if "company" in str(col).lower() or "name" in str(col).lower()), None)
    sector_col = next((col for col in candidate.columns if "sector" in str(col).lower() or "industry" in str(col).lower()), None)
    output = pd.DataFrame({
        "ticker": candidate[code_col].map(clean_hk_ticker),
        "name": candidate[name_col].astype(str) if name_col else None,
        "sector": candidate[sector_col].astype(str) if sector_col else None,
        "industry": None,
        "market": "HK",
    }).dropna(subset=["ticker"]).drop_duplicates("ticker")
    if len(output) < 40:
        raise RuntimeError("取得的恒指成分資料不完整")
    return output


def sync_sp500() -> tuple[bool, str]:
    try:
        universe = fetch_sp500_constituents()
        if not upsert_instruments(universe.to_dict("records")):
            return False, "無法寫入 instruments 資料表"
        today = date.today().isoformat()
        memberships = []
        snapshots = []
        for row in universe.to_dict("records"):
            memberships.append({
                "ticker": row["ticker"], "source": "sp500_constituent", "is_permanent": False,
                "is_active": True, "priority": 30, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None,
            })
            snapshots.append({
                "snapshot_date": today, "ticker": row["ticker"], "source": "sp500_constituent",
                "index_member": True, "is_selected": True,
            })
        db_upsert("watchlist_memberships", memberships, "ticker,source")
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        deactivate_missing("sp500_constituent", set(universe["ticker"]))
        return True, f"已同步 {len(universe)} 隻 S&P 500 成分股。"
    except Exception as exc:
        return False, f"S&P 500 同步失敗：{type(exc).__name__}: {exc}"


def sync_hsi_top30() -> tuple[bool, str]:
    try:
        universe = fetch_hsi_constituents()
        if not upsert_instruments(universe.to_dict("records")):
            return False, "無法寫入 instruments 資料表"
        today = date.today().isoformat()
        members = []
        snapshots = []
        for row in universe.to_dict("records"):
            members.append({
                "ticker": row["ticker"], "source": "hsi_constituent", "is_permanent": False,
                "is_active": True, "priority": 40, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None,
            })
            snapshots.append({
                "snapshot_date": today, "ticker": row["ticker"], "source": "hsi_constituent",
                "index_member": True, "is_selected": True,
            })
        db_upsert("watchlist_memberships", members, "ticker,source")
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_constituent", set(universe["ticker"]))

        turnover_rows = []
        for ticker in universe["ticker"].tolist():
            df, _ = fetch_ohlcv(ticker, "6mo")
            if df is None or len(df) < 20:
                continue
            turnover_rows.append({
                "ticker": ticker,
                "turnover_20d": float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean()),
                "average_volume_20d": float(df["volume"].iloc[-20:].mean()),
                "close_price": float(df["close"].iloc[-1]),
            })
        ranking = pd.DataFrame(turnover_rows).sort_values("turnover_20d", ascending=False).head(30).reset_index(drop=True)
        if ranking.empty:
            return False, "未能取得足夠恒指成分股價格資料"
        ranking["turnover_rank"] = ranking.index + 1
        top_members = []
        top_snapshots = []
        for row in ranking.to_dict("records"):
            top_members.append({
                "ticker": row["ticker"], "source": "hsi_top30_turnover", "is_permanent": False,
                "is_active": True, "priority": 90, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None,
            })
            top_snapshots.append({
                "snapshot_date": today, "ticker": row["ticker"], "source": "hsi_top30_turnover",
                "index_member": True, "turnover_20d": row["turnover_20d"],
                "turnover_rank": int(row["turnover_rank"]), "average_volume_20d": row["average_volume_20d"],
                "close_price": row["close_price"], "is_selected": True,
            })
        db_upsert("watchlist_memberships", top_members, "ticker,source")
        db_upsert("watchlist_snapshots", top_snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_top30_turnover", set(ranking["ticker"]))
        return True, f"已同步 {len(universe)} 隻恒指成分股，並按 20 日平均成交額選出 Top {len(ranking)}。"
    except Exception as exc:
        return False, f"恒指／Top 30 同步失敗：{type(exc).__name__}: {exc}"


# ==================== Indicators, regime and score ====================
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
    previous = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((2 * df["close"] - df["high"] - df["low"]) / spread).fillna(0)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def weekly_rsi(close: pd.Series) -> pd.Series:
    weekly_close = close.resample("W-FRI").last().dropna()
    return rsi(weekly_close).reindex(close.index, method="ffill")


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    return (series - series.rolling(period).mean()) / series.rolling(period).std(ddof=0).replace(0, np.nan)


def bullish_divergence(df: pd.DataFrame) -> bool:
    hist = macd(df["close"])[2]
    lows = df["low"].iloc[:-3]
    pivots = lows[lows == lows.rolling(7, center=True).min()].dropna()
    if len(pivots) < 2:
        return False
    first, second = pivots.index[-2], pivots.index[-1]
    return bool(pivots.loc[second] < pivots.loc[first] and hist.loc[second] > hist.loc[first])


def market_regime(market: str) -> tuple[str, dict[str, float]]:
    benchmark, vol_ticker = ("SPY", "^VIX") if market == "US" else ("^HSI", "^VHSI")
    prices, _ = fetch_ohlcv(benchmark, "1y")
    vol, _ = fetch_ohlcv(vol_ticker, "1y")
    if prices is None or len(prices) < 70:
        return "unknown", {"return_60d": np.nan, "volatility": np.nan, "vol_index": np.nan}
    ret60 = 100 * (prices["close"].iloc[-1] / prices["close"].iloc[-61] - 1)
    annual_vol = 100 * prices["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_index = float(vol["close"].iloc[-1]) if vol is not None else np.nan
    high_vol = vol_index >= (25 if market == "US" else 30)
    if ret60 >= 5:
        regime = "bull_high_vol" if high_vol else "bull_low_vol"
    elif ret60 <= -5:
        regime = "bear_high_vol" if high_vol else "bear_low_vol"
    else:
        regime = "neutral_high_vol" if high_vol else "neutral"
    return regime, {"return_60d": ret60, "volatility": annual_vol, "vol_index": vol_index}


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


def score_label(score: float) -> str:
    if score >= 75:
        return "強烈關注"
    if score >= 60:
        return "值得關注"
    if score >= 45:
        return "觀察中"
    return "未觸發"


def score_stock(df: pd.DataFrame, regime: str) -> ScoreResult:
    close = df["close"]
    volume = df["volume"]
    price = float(close.iloc[-1])
    daily_rsi = rsi(close)
    week_rsi = weekly_rsi(close)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()
    macd_line, macd_signal, _ = macd(close)
    atr14 = atr(df)
    volume_z = zscore(volume)
    cmf20 = cmf(df)
    vwap20 = rolling_vwap(df)
    required = [daily_rsi, week_rsi, ma60, ma200, atr14, volume_z, cmf20, vwap20]
    if any(pd.isna(series.iloc[-1]) for series in required):
        raise ValueError("指標暖機資料不足")

    factors = {"reversal": 0.0, "trend": 0.0, "flow": 0.0, "risk": 0.0}
    notes: list[str] = []
    if daily_rsi.iloc[-1] <= 30:
        factors["reversal"] += 15; notes.append(f"日 RSI {daily_rsi.iloc[-1]:.1f} 超賣")
    elif daily_rsi.iloc[-1] <= 38:
        factors["reversal"] += 8; notes.append(f"日 RSI {daily_rsi.iloc[-1]:.1f} 偏低")
    if week_rsi.iloc[-1] <= 42:
        factors["reversal"] += 8; notes.append(f"真週 RSI {week_rsi.iloc[-1]:.1f} 偏低")
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-1] < 0:
        factors["reversal"] += 7; notes.append("MACD 零軸下金叉")
    if bullish_divergence(df):
        factors["reversal"] += 8; notes.append("確認的 MACD 底背離")
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

    if cmf20.iloc[-1] > 0.08:
        factors["flow"] += 10; notes.append(f"CMF {cmf20.iloc[-1]:.2f} 流入")
    elif cmf20.iloc[-1] < -0.08:
        notes.append(f"CMF {cmf20.iloc[-1]:.2f} 流出")
    if volume_z.iloc[-1] >= 1.5 and price > vwap20.iloc[-1] and close.iloc[-1] > df["open"].iloc[-1]:
        factors["flow"] += 10; notes.append(f"放量收陽 Z={volume_z.iloc[-1]:.1f}")
    factors["flow"] = min(factors["flow"], 20)

    atr_pct = float(atr14.iloc[-1] / price * 100)
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

    stop = min(float(df["low"].iloc[-10:].min()), price - 0.5 * float(atr14.iloc[-1]))
    stop = round(stop, 3)
    target = round(price + 1.8 * (price - stop), 3)
    total = round(sum(factors.values()), 1)
    return ScoreResult(total, score_label(total), regime, round(price, 3), stop, target, factors, notes)


# ==================== Scan persistence ====================
def create_scan_run(market: str, count: int) -> int | None:
    client = get_supabase()
    if client is None:
        return None
    try:
        response = client.table("scan_runs").insert({
            "run_type": "manual", "market": market, "universe_size": count, "model_version": MODEL_VERSION,
        }).execute()
        return response.data[0]["id"]
    except Exception as exc:
        set_db_error("建立掃描紀錄", exc)
        return None


def finish_scan_run(run_id: int | None, success: int, failed: int, errors: list[str]) -> None:
    client = get_supabase()
    if client is None or run_id is None:
        return
    try:
        client.table("scan_runs").update({
            "completed_at": datetime.utcnow().isoformat(),
            "status": "completed" if failed == 0 else "completed_with_errors",
            "success_count": success,
            "failed_count": failed,
            "error_summary": " | ".join(errors[:10]) or None,
        }).eq("id", run_id).execute()
    except Exception as exc:
        set_db_error("完成掃描紀錄", exc)


def persist_result(run_id: int | None, ticker: str, result: ScoreResult, meta: dict[str, Any]) -> None:
    if run_id is None:
        return
    db_upsert("scan_results", {
        "scan_run_id": run_id, "ticker": ticker, "price": result.price, "score": result.score,
        "label": result.label, "regime": result.regime, "stop_price": result.stop,
        "target_price": result.target, "factors": result.factors, "explanations": result.explanations,
        "data_source": meta["source"], "last_bar_date": meta["last_bar"],
    }, "scan_run_id,ticker")


def persist_signal(ticker: str, signal_date: str, result: ScoreResult, meta: dict[str, Any]) -> None:
    add_membership(ticker, "signal_high_score", False, 80, "由高分訊號自動加入")
    db_upsert("signals", {
        "signal_date": signal_date, "ticker": ticker, "model_version": MODEL_VERSION,
        "price": result.price, "score": result.score, "label": result.label, "regime": result.regime,
        "stop_price": result.stop, "target_price": result.target, "factors": result.factors,
        "explanations": result.explanations, "data_source": meta["source"], "last_bar_date": meta["last_bar"],
    }, "signal_date,ticker,model_version")


def make_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="價格"))
    for days, color in [(20, "#f59e0b"), (60, "#3b82f6"), (200, "#a855f7")]:
        figure.add_trace(go.Scatter(x=df.index, y=df["close"].rolling(days).mean(), name=f"MA{days}", line={"width": 1.2, "color": color}))
    figure.update_layout(template="plotly_dark", height=560, title=f"{ticker}｜調整後日線", xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 45, "b": 10})
    return figure


# ==================== Interface ====================
st.title("📈 股票監察系統 Pro · 雲端觀察名單版")
st.caption("模型 2.1.1｜S&P 500 全成分股＋恒指成分股 20 日平均成交額 Top 30｜評分只供篩選和研究，不構成投資建議。")

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
    st.caption("日線評分使用真週 RSI、趨勢、資金流及 ATR 風險。高分不等於應立即買入。")

if market_label == "自選":
    tickers = list(dict.fromkeys(normalize_ticker(item) for item in custom.splitlines() if item.strip()))
else:
    stored = get_watchlist(market)
    fallback = DEFAULT_US if market == "US" else DEFAULT_HK
    tickers = stored["ticker"].drop_duplicates().tolist() if not stored.empty else fallback

regime, regime_stats = market_regime(market)
regime_names = {
    "bull_low_vol": "牛市／低波動", "bull_high_vol": "牛市／高波動",
    "bear_low_vol": "熊市／低波動", "bear_high_vol": "熊市／高波動",
    "neutral": "中性", "neutral_high_vol": "中性／高波動", "unknown": "未能判定",
}
col1, col2, col3, col4 = st.columns(4)
col1.metric("市場 regime", regime_names.get(regime, regime))
col2.metric("60 日基準回報", "—" if pd.isna(regime_stats["return_60d"]) else f"{regime_stats['return_60d']:.1f}%")
col3.metric("20 日年化波動", "—" if pd.isna(regime_stats["volatility"]) else f"{regime_stats['volatility']:.1f}%")
col4.metric("波動率指數", "—" if pd.isna(regime_stats["vol_index"]) else f"{regime_stats['vol_index']:.1f}")

watch_tab, scan_tab, detail_tab, log_tab, risk_tab = st.tabs([
    "📌 觀察名單管理", "📊 掃描", "📈 個股詳情", "🗂️ 訊號紀錄", "⚖️ 風控"
])

with watch_tab:
    st.subheader("永久觀察名單與自動候選池")
    st.caption("手動及持倉名單永久保存；S&P 500、恒指成分與恒指成交額 Top 30 可更新為現役候選。跌出指數或 Top 30 只會停用該候選來源，不會刪除你的手動名單或歷史紀錄。")
    sync1, sync2, sync3 = st.columns(3)
    if sync1.button("同步 S&P 500 全成分股", type="primary", disabled=client is None):
        with st.spinner("正在同步 S&P 500 成分表…"):
            ok, message = sync_sp500()
        st.success(message) if ok else st.error(message)
    if sync2.button("同步恒指及成交額 Top 30", type="primary", disabled=client is None):
        with st.spinner("正在下載恒指成分日線並計算 20 日平均成交額；首次可能需要數分鐘…"):
            ok, message = sync_hsi_top30()
        st.success(message) if ok else st.error(message)
    sync3.info("第一版以公開成分表匯入。請定期按官方指數資料核對，尤其不可將成分同步本身視為交易訊號。")

    st.divider()
    st.markdown("### 手動永久加入")
    add1, add2, add3, add4 = st.columns([2, 1, 1, 3])
    manual_ticker = add1.text_input("代碼", placeholder="AAPL 或 0700.HK")
    manual_source = add2.selectbox("類別", ["manual", "portfolio"], format_func=lambda value: SOURCE_LABELS[value])
    manual_priority = add3.slider("優先級", 1, 100, 100)
    manual_notes = add4.text_input("備註", placeholder="已持有／等業績／只觀察")
    if st.button("加入並永久保存", disabled=client is None):
        if not manual_ticker.strip():
            st.warning("請先輸入股票代碼。")
        elif add_membership(manual_ticker, manual_source, True, manual_priority, manual_notes):
            st.success(f"已永久加入 {normalize_ticker(manual_ticker)}。")
            st.rerun()
        else:
            st.error(st.session_state.get("db_error", "寫入失敗"))

    members = get_watchlist()
    if members.empty:
        st.info("尚未有雲端觀察名單。請同步指數成分，或手動加入股票。")
    else:
        display = members.copy()
        display["來源"] = display["source"].map(SOURCE_LABELS).fillna(display["source"])
        display["永久"] = display["is_permanent"].map({True: "是", False: "否"})
        st.dataframe(display[["ticker", "market", "name", "sector", "來源", "永久", "priority", "notes", "added_at", "last_confirmed_at"]], use_container_width=True, hide_index=True)
    if st.session_state.get("db_error"):
        st.caption(f"最近資料庫錯誤：{st.session_state['db_error']}")

with scan_tab:
    st.subheader(f"{market_label} 觀察名單掃描")
    if len(tickers) > 100:
        st.warning(f"現有 {len(tickers)} 隻股票。為降低免費資料源限流風險，本頁每次最多掃描 100 隻；完整每日掃描應在下一階段使用 GitHub Actions。")
    max_limit = max(1, min(len(tickers), 100))
    scan_count = st.number_input("本次掃描數量", min_value=1, max_value=max_limit, value=max_limit, step=1)
    selected_tickers = tickers[:int(scan_count)]
    if st.button("開始掃描", type="primary"):
        run_id = create_scan_run(market if market_label != "自選" else "BOTH", len(selected_tickers))
        rows: list[dict] = []
        failures: list[dict] = []
        errors: list[str] = []
        progress = st.progress(0)
        for number, ticker in enumerate(selected_tickers, start=1):
            df, meta = fetch_ohlcv(ticker)
            if df is None:
                failures.append({"代碼": ticker, "原因": meta["status"]})
                errors.append(f"{ticker}: {meta['status']}")
            else:
                try:
                    result = score_stock(df, regime)
                    persist_result(run_id, ticker, result, meta)
                    rows.append({
                        "代碼": ticker, "現價": result.price, "總分": result.score, "標籤": result.label,
                        "止損": result.stop, "目標": result.target,
                        "R/R": round((result.target - result.price) / (result.price - result.stop), 2),
                        "資料最後日": meta["last_bar"],
                        "因子": " / ".join(f"{key}:{value:.0f}" for key, value in result.factors.items()),
                        "說明": "；".join(result.explanations),
                    })
                    if result.score >= minimum_score:
                        persist_signal(ticker, str(df.index[-1].date()), result, meta)
                except Exception as exc:
                    failures.append({"代碼": ticker, "原因": f"評分失敗：{type(exc).__name__}: {exc}"})
                    errors.append(f"{ticker}: {exc}")
            progress.progress(number / len(selected_tickers))
        finish_scan_run(run_id, len(rows), len(failures), errors)
        if rows:
            result_table = pd.DataFrame(rows).sort_values(["總分", "代碼"], ascending=[False, True])
            st.dataframe(result_table, use_container_width=True, hide_index=True)
            st.download_button("下載掃描 CSV", result_table.to_csv(index=False).encode("utf-8-sig"), "scan_results.csv", "text/csv")
            st.info("掃描結果已儲存於 Supabase；達到門檻的訊號會寫入雲端訊號紀錄，並新增「高分訊號」候選來源。")
        if failures:
            st.warning("部分股票未完成掃描，請檢查以下原因。")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

with detail_tab:
    ticker = st.text_input("代碼", tickers[0] if tickers else "AAPL", key="detail_ticker")
    period = st.selectbox("圖表範圍", ["1y", "2y", "3y", "5y"], index=1)
    if st.button("載入個股"):
        df, meta = fetch_ohlcv(ticker, period)
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime)
            item1, item2, item3, item4 = st.columns(4)
            item1.metric("總分", f"{result.score:.1f}/90", result.label)
            item2.metric("現價", result.price)
            item3.metric("結構止損", result.stop)
            item4.metric("最低 1.8R 目標", result.target)
            st.caption(f"來源：{meta['source']}｜調整後價格：{meta['adjusted']}｜最後 bar：{meta['last_bar']}｜資料行數：{meta['rows']}")
            st.plotly_chart(make_chart(df, normalize_ticker(ticker)), use_container_width=True)
            st.markdown("### 分數拆解")
            st.dataframe(pd.DataFrame([result.factors]), use_container_width=True, hide_index=True)
            st.write("；".join(result.explanations) if result.explanations else "目前沒有額外確認條件。")

with log_tab:
    st.subheader("雲端訊號紀錄")
    if client is None:
        st.error("Supabase 未連線。")
    else:
        try:
            signals = client.table("signals").select(
                "signal_date,ticker,price,score,label,regime,stop_price,target_price,data_source,last_bar_date,created_at"
            ).order("signal_date", desc=True).order("score", desc=True).limit(1000).execute().data or []
            if signals:
                signal_df = pd.DataFrame(signals)
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
                st.download_button("下載訊號 CSV", signal_df.to_csv(index=False).encode("utf-8-sig"), "cloud_signals.csv", "text/csv")
            else:
                st.info("尚未有保存的訊號；請先完成掃描。")
        except Exception as exc:
            st.error(f"讀取訊號失敗：{type(exc).__name__}: {exc}")

with risk_tab:
    st.subheader("風險為本的部位計算")
    risk1, risk2, risk3, risk4 = st.columns(4)
    account_value = risk1.number_input("帳戶淨值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_percent = risk2.slider("每筆最大帳戶風險 (%)", 0.25, 2.0, 0.75, 0.25)
    allocation_limit = risk3.slider("單一持倉最大名義比例 (%)", 1.0, 30.0, 10.0, 1.0)
    risk_ticker = risk4.text_input("代碼", tickers[0] if tickers else "AAPL", key="risk_ticker")
    if st.button("計算可承受部位"):
        df, meta = fetch_ohlcv(risk_ticker, "2y")
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime)
            per_share_risk = result.price - result.stop
            risk_budget = account_value * risk_percent / 100
            risk_shares = int(risk_budget / per_share_risk) if per_share_risk > 0 else 0
            allocation_shares = int((account_value * allocation_limit / 100) / result.price)
            shares = max(0, min(risk_shares, allocation_shares))
            out1, out2, out3, out4 = st.columns(4)
            out1.metric("入場參考價", result.price)
            out2.metric("結構止損", result.stop)
            out3.metric("每股風險", f"{per_share_risk:.3f}")
            out4.metric("建議上限股數", shares)
            st.write(f"風險預算：{risk_budget:,.2f}｜名義金額：約 {shares * result.price:,.2f}｜模型目標：{result.target}")
            st.warning("請按交易所整手規則向下調整股數，並自行考慮匯率、稅項、手續費、買賣價差與停損成交風險。")
