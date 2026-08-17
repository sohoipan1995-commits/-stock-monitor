from __future__ import annotations

import io
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from supabase import Client, create_client
from scoring_v3 import V3ScoreResult, score_stock_v3

st.set_page_config(page_title="股票監察系統 Pro · V3", page_icon="📈", layout="wide")

MODEL_VERSION = "3.0.0"
WEB_SCAN_LIMIT = 200
DEFAULT_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM", "SPY", "QQQ"]
DEFAULT_HK = ["0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "2318.HK", "9988.HK", "3690.HK", "9618.HK", "1211.HK"]
SOURCE_LABELS = {
    "manual": "手動永久",
    "portfolio": "持倉",
    "signal_high_score": "V3試倉候選",
    "sp500_constituent": "S&P 500",
    "hsi_constituent": "恒指成分",
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
    payload = []
    now = datetime.utcnow().isoformat()
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
    return db_upsert("instruments", payload, "ticker") if payload else True


def add_membership(ticker: str, source: str, permanent: bool, priority: int, notes: str | None = None) -> bool:
    ticker = normalize_ticker(ticker)
    if not upsert_instruments([{"ticker": ticker, "market": ticker_market(ticker)}]):
        return False
    return db_upsert("watchlist_memberships", {
        "ticker": ticker,
        "source": source,
        "is_permanent": permanent,
        "is_active": True,
        "priority": priority,
        "notes": notes,
        "last_confirmed_at": datetime.utcnow().isoformat(),
        "removed_at": None,
    }, "ticker,source")


def deactivate_missing(source: str, active_tickers: set[str]) -> None:
    client = get_supabase()
    if client is None:
        return
    try:
        existing = client.table("watchlist_memberships").select("id,ticker").eq("source", source).eq("is_active", True).execute().data or []
        for item in existing:
            if item["ticker"] not in active_tickers:
                client.table("watchlist_memberships").update({
                    "is_active": False,
                    "removed_at": datetime.utcnow().isoformat(),
                }).eq("id", item["id"]).execute()
    except Exception as exc:
        set_db_error("更新候選狀態", exc)


def get_watchlist(market: str | None = None) -> pd.DataFrame:
    client = get_supabase()
    if client is None:
        return pd.DataFrame()
    try:
        memberships = client.table("watchlist_memberships").select(
            "ticker,source,is_permanent,priority,notes,added_at,last_confirmed_at"
        ).eq("is_active", True).execute().data or []
        instruments = client.table("instruments").select(
            "ticker,market,name,sector"
        ).eq("is_active", True).execute().data or []
        if not memberships:
            return pd.DataFrame()
        output = pd.DataFrame(memberships).merge(pd.DataFrame(instruments), on="ticker", how="left")
        if market in {"US", "HK"}:
            output = output[output["market"] == market]
        return output.sort_values(["is_permanent", "priority", "ticker"], ascending=[False, False, True])
    except Exception as exc:
        set_db_error("讀取觀察名單", exc)
        return pd.DataFrame()


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
            raw.columns = [str(column).lower() for column in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            valid, message = validate_ohlcv(df)
            return (df if valid else None), {
                "ticker": ticker,
                "source": "Yahoo Finance",
                "adjusted": True,
                "last_bar": str(df.index[-1].date()) if not df.empty else None,
                "rows": len(df),
                "status": message,
            }
        except Exception as exc:
            errors.append(f"第 {attempt + 1} 次：{type(exc).__name__}: {exc}")
            time.sleep(0.8 * (2 ** attempt))
    return None, {"ticker": ticker, "source": "Yahoo Finance", "last_bar": None, "rows": 0, "status": "；".join(errors)}


def clean_us_ticker(value: Any) -> str:
    return normalize_ticker(value).replace(".", "-")


def clean_hk_ticker(value: Any) -> str | None:
    digits = "".join(char for char in str(value) if char.isdigit())
    return f"{digits.zfill(4)}.HK" if digits else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={
            "User-Agent": "StockMonitorPro/3.0 research application",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
    if not tables:
        raise RuntimeError("找不到 S&P 500 成分表")
    table = tables[0]
    if not {"Symbol", "Security"}.issubset(table.columns):
        raise RuntimeError("S&P 500 成分表格式已改變")
    return pd.DataFrame({
        "ticker": table["Symbol"].map(clean_us_ticker),
        "name": table["Security"].astype(str),
        "sector": table["GICS Sector"].astype(str) if "GICS Sector" in table.columns else None,
        "industry": table["GICS Sub-Industry"].astype(str) if "GICS Sub-Industry" in table.columns else None,
        "market": "US",
    }).drop_duplicates("ticker")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hsi_constituents() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        headers={
            "User-Agent": "StockMonitorPro/3.0 research application",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    candidate = None
    for table in tables:
        columns = [str(column).lower() for column in table.columns]
        if len(table) >= 40 and any("code" in column or "ticker" in column for column in columns):
            candidate = table
            break
    if candidate is None:
        raise RuntimeError("找不到恒指成分表")
    code_column = next(column for column in candidate.columns if "code" in str(column).lower() or "ticker" in str(column).lower())
    name_column = next((column for column in candidate.columns if "company" in str(column).lower() or "name" in str(column).lower()), None)
    sector_column = next((column for column in candidate.columns if "sector" in str(column).lower() or "industry" in str(column).lower()), None)
    output = pd.DataFrame({
        "ticker": candidate[code_column].map(clean_hk_ticker),
        "name": candidate[name_column].astype(str) if name_column else None,
        "sector": candidate[sector_column].astype(str) if sector_column else None,
        "industry": None,
        "market": "HK",
    }).dropna(subset=["ticker"]).drop_duplicates("ticker")
    if len(output) < 40:
        raise RuntimeError("恒指成分資料不完整")
    return output


def sync_sp500() -> tuple[bool, str]:
    try:
        universe = fetch_sp500_constituents()
        if not upsert_instruments(universe.to_dict("records")):
            return False, "無法寫入 instruments"
        today = date.today().isoformat()
        now = datetime.utcnow().isoformat()
        memberships = [{
            "ticker": ticker, "source": "sp500_constituent", "is_permanent": False,
            "is_active": True, "priority": 30, "last_confirmed_at": now, "removed_at": None,
        } for ticker in universe["ticker"]]
        snapshots = [{
            "snapshot_date": today, "ticker": ticker, "source": "sp500_constituent",
            "index_member": True, "is_selected": True,
        } for ticker in universe["ticker"]]
        if not db_upsert("watchlist_memberships", memberships, "ticker,source"):
            return False, st.session_state.get("db_error", "寫入 S&P 500 觀察名單失敗")
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
        today = date.today().isoformat()
        now = datetime.utcnow().isoformat()
        memberships = [{
            "ticker": ticker, "source": "hsi_constituent", "is_permanent": False,
            "is_active": True, "priority": 40, "last_confirmed_at": now, "removed_at": None,
        } for ticker in universe["ticker"]]
        snapshots = [{
            "snapshot_date": today, "ticker": ticker, "source": "hsi_constituent",
            "index_member": True, "is_selected": True,
        } for ticker in universe["ticker"]]
        db_upsert("watchlist_memberships", memberships, "ticker,source")
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_constituent", set(universe["ticker"]))

        turnover_rows = []
        for ticker in universe["ticker"].tolist():
            df, _ = fetch_ohlcv(ticker, "6mo")
            if df is not None and len(df) >= 20:
                turnover_rows.append({
                    "ticker": ticker,
                    "turnover_20d": float((df["close"].iloc[-20:] * df["volume"].iloc[-20:]).mean()),
                    "average_volume_20d": float(df["volume"].iloc[-20:].mean()),
                    "close_price": float(df["close"].iloc[-1]),
                })
        ranking = pd.DataFrame(turnover_rows).sort_values("turnover_20d", ascending=False).head(30).reset_index(drop=True)
        if ranking.empty:
            return False, "未取得足夠恒指成分股價格資料"
        ranking["turnover_rank"] = ranking.index + 1
        top_members = [{
            "ticker": row.ticker, "source": "hsi_top30_turnover", "is_permanent": False,
            "is_active": True, "priority": 90, "last_confirmed_at": now, "removed_at": None,
        } for row in ranking.itertuples()]
        top_snapshots = [{
            "snapshot_date": today, "ticker": row.ticker, "source": "hsi_top30_turnover",
            "index_member": True, "turnover_20d": row.turnover_20d,
            "turnover_rank": int(row.turnover_rank), "average_volume_20d": row.average_volume_20d,
            "close_price": row.close_price, "is_selected": True,
        } for row in ranking.itertuples()]
        db_upsert("watchlist_memberships", top_members, "ticker,source")
        db_upsert("watchlist_snapshots", top_snapshots, "snapshot_date,ticker,source")
        deactivate_missing("hsi_top30_turnover", set(ranking["ticker"]))
        return True, f"已同步 {len(universe)} 隻恒指成分股，並選出 20 日平均成交額 Top {len(ranking)}。"
    except Exception as exc:
        return False, f"恒指／Top 30 同步失敗：{type(exc).__name__}: {exc}"


def market_regime(market: str) -> tuple[str, dict[str, float]]:
    benchmark_ticker, volatility_ticker = ("SPY", "^VIX") if market == "US" else ("^HSI", "^VHSI")
    prices, _ = fetch_ohlcv(benchmark_ticker, "1y")
    volatility, _ = fetch_ohlcv(volatility_ticker, "1y")
    if prices is None or len(prices) < 70:
        return "unknown", {"return_60d": float("nan"), "volatility": float("nan"), "vol_index": float("nan")}
    ret60 = 100 * (prices["close"].iloc[-1] / prices["close"].iloc[-61] - 1)
    annual_volatility = 100 * prices["close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)
    vol_index = float(volatility["close"].iloc[-1]) if volatility is not None else float("nan")
    high_volatility = vol_index >= (25 if market == "US" else 30)
    if ret60 >= 5:
        state = "bull_high_vol" if high_volatility else "bull_low_vol"
    elif ret60 <= -5:
        state = "bear_high_vol" if high_volatility else "bear_low_vol"
    else:
        state = "neutral_high_vol" if high_volatility else "neutral"
    return state, {"return_60d": ret60, "volatility": annual_volatility, "vol_index": vol_index}


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


def persist_scan(run_id: int | None, ticker: str, result: V3ScoreResult, meta: dict[str, Any], regime: str) -> None:
    if run_id is None:
        return
    factors = {
        "bottom_structure": result.bottom_score,
        "reversal_confirmation": result.confirmation_score,
        "quality": result.quality_score,
        "risk_deduction": result.risk_deduction,
        "eligible": result.eligible,
        "action": result.action,
        "metrics": result.metrics,
        "trigger_price": result.trigger_price,
        "risk_reward": result.risk_reward,
    }
    db_upsert("scan_results", {
        "scan_run_id": run_id, "ticker": ticker, "price": result.price,
        "score": result.total_score, "label": result.action, "regime": regime,
        "stop_price": result.stop_price, "target_price": result.target_price,
        "factors": factors, "explanations": result.reasons + result.blockers,
        "data_source": meta["source"], "last_bar_date": meta["last_bar"],
    }, "scan_run_id,ticker")


def persist_signal(ticker: str, signal_date: str, result: V3ScoreResult, meta: dict[str, Any], regime: str) -> None:
    if result.action != "可小量試倉":
        return
    add_membership(ticker, "signal_high_score", False, 80, "V3 可小量試倉候選")
    db_upsert("signals", {
        "signal_date": signal_date, "ticker": ticker, "model_version": MODEL_VERSION,
        "price": result.price, "score": result.total_score, "label": result.action,
        "regime": regime, "stop_price": result.stop_price, "target_price": result.target_price,
        "factors": {"bottom_structure": result.bottom_score, "reversal_confirmation": result.confirmation_score, "quality": result.quality_score, "risk_deduction": result.risk_deduction, "metrics": result.metrics, "trigger_price": result.trigger_price, "risk_reward": result.risk_reward},
        "explanations": result.reasons + result.blockers,
        "data_source": meta["source"], "last_bar_date": meta["last_bar"],
    }, "signal_date,ticker,model_version")


def make_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="價格"))
    for days, color in [(20, "#f59e0b"), (60, "#3b82f6"), (200, "#a855f7")]:
        figure.add_trace(go.Scatter(x=df.index, y=df["close"].rolling(days).mean(), name=f"MA{days}", line={"width": 1.2, "color": color}))
    figure.update_layout(template="plotly_dark", height=560, title=f"{ticker}｜調整後日線", xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 45, "b": 10})
    return figure


st.title("📈 股票監察系統 Pro · V3 反轉候選版")
st.caption("V3 將『超賣』與『反轉確認』分開：只在資格、確認價、止損及 2R 目標均合理時標示為可小量試倉。這不是見底或獲利保證。")

client = get_supabase()
if client is None:
    st.error("未偵測到 Supabase Secrets。請檢查 Streamlit Cloud 的 SUPABASE_URL 與 SUPABASE_SECRET_KEY。")
else:
    st.success("Supabase 已連線：觀察名單、掃描及 V3 訊號會保存至雲端資料庫。")

with st.sidebar:
    st.header("控制面板")
    market_label = st.radio("市場", ["美股", "港股", "自選"], index=0)
    market = "US" if market_label == "美股" else "HK"
    custom = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK") if market_label == "自選" else ""
    st.caption("V3 行動規則：資格合格＋反轉確認 ≥ 20＋總分 ≥ 65＋止損距離合理，才標示『可小量試倉』。")

if market_label == "自選":
    tickers = list(dict.fromkeys(normalize_ticker(item) for item in custom.splitlines() if item.strip()))
else:
    stored = get_watchlist(market)
    tickers = stored["ticker"].drop_duplicates().tolist() if not stored.empty else (DEFAULT_US if market == "US" else DEFAULT_HK)

regime, regime_stats = market_regime(market)
regime_names = {
    "bull_low_vol": "牛市／低波動", "bull_high_vol": "牛市／高波動",
    "bear_low_vol": "熊市／低波動", "bear_high_vol": "熊市／高波動",
    "neutral": "中性", "neutral_high_vol": "中性／高波動", "unknown": "未能判定",
}
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("市場 regime", regime_names.get(regime, regime))
metric2.metric("60 日基準回報", "—" if pd.isna(regime_stats["return_60d"]) else f"{regime_stats['return_60d']:.1f}%")
metric3.metric("20 日年化波動", "—" if pd.isna(regime_stats["volatility"]) else f"{regime_stats['volatility']:.1f}%")
metric4.metric("波動率指數", "—" if pd.isna(regime_stats["vol_index"]) else f"{regime_stats['vol_index']:.1f}")

watch_tab, scan_tab, detail_tab, log_tab, risk_tab = st.tabs(["📌 觀察名單管理", "📊 V3掃描", "📈 V3個股詳情", "🗂️ 雲端訊號", "⚖️ 風控"])

with watch_tab:
    st.subheader("永久觀察名單與自動候選池")
    sync1, sync2, sync3 = st.columns(3)
    if sync1.button("同步 S&P 500 全成分股", type="primary", disabled=client is None):
        with st.spinner("正在同步 S&P 500…"):
            ok, message = sync_sp500()
        st.success(message) if ok else st.error(message)
    if sync2.button("同步恒指及成交額 Top 30", type="primary", disabled=client is None):
        with st.spinner("正在同步恒指並計算 20 日平均成交額；首次可能需數分鐘…"):
            ok, message = sync_hsi_top30()
        st.success(message) if ok else st.error(message)
    sync3.info("手動及持倉名單永久保存。自動候選跌出指數或 Top 30 時只會停用該來源，不會刪除歷史。")
    st.divider()
    add1, add2, add3, add4 = st.columns([2, 1, 1, 3])
    manual_ticker = add1.text_input("代碼", placeholder="AAPL 或 0700.HK")
    manual_source = add2.selectbox("類別", ["manual", "portfolio"], format_func=lambda value: SOURCE_LABELS[value])
    manual_priority = add3.slider("優先級", 1, 100, 100)
    manual_notes = add4.text_input("備註", placeholder="已持有／等業績／只觀察")
    if st.button("加入並永久保存", disabled=client is None):
        if not manual_ticker.strip():
            st.warning("請輸入股票代碼。")
        elif add_membership(manual_ticker, manual_source, True, manual_priority, manual_notes):
            st.success(f"已永久加入 {normalize_ticker(manual_ticker)}。")
            st.rerun()
        else:
            st.error(st.session_state.get("db_error", "寫入失敗"))
    members = get_watchlist()
    if members.empty:
        st.info("尚未有雲端觀察名單。請同步指數成分或手動加入股票。")
    else:
        display = members.copy()
        display["來源"] = display["source"].map(SOURCE_LABELS).fillna(display["source"])
        display["永久"] = display["is_permanent"].map({True: "是", False: "否"})
        st.dataframe(display[["ticker", "market", "name", "sector", "來源", "永久", "priority", "notes", "added_at", "last_confirmed_at"]], use_container_width=True, hide_index=True)

with scan_tab:
    st.subheader(f"{market_label} V3 反轉候選掃描")
    st.caption("輸出欄位分開顯示資格、底部結構、反轉確認、品質與風險。『可小量試倉』仍需自行考慮業績、公告和個人風險承受能力。")
    if not tickers:
        st.warning("目前沒有可掃描的股票。請先同步指數成分或手動加入。")
    else:
        if len(tickers) > WEB_SCAN_LIMIT:
            st.warning(f"目前有 {len(tickers)} 隻股票；網頁單次最多掃描 {WEB_SCAN_LIMIT} 隻，以避免資料源限流或 Cloud 超時。")
        max_limit = min(len(tickers), WEB_SCAN_LIMIT)
        scan_count = st.number_input("本次掃描數量", min_value=1, max_value=max_limit, value=max_limit, step=1)
        selected = tickers[:int(scan_count)]
        st.caption(f"候選池：{len(tickers)} 隻｜本次掃描：{len(selected)} 隻｜網頁上限：{WEB_SCAN_LIMIT} 隻")
        if st.button("開始 V3 掃描", type="primary"):
            benchmark_ticker = "SPY" if market == "US" else "^HSI"
            benchmark_df, _ = fetch_ohlcv(benchmark_ticker, "3y")
            benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
            if benchmark_close is None:
                st.warning(f"未能取得 {benchmark_ticker}；品質分不會包含相對強弱。")
            run_id = create_scan_run(market if market_label != "自選" else "BOTH", len(selected))
            rows, failures, errors = [], [], []
            progress = st.progress(0)
            status = st.empty()
            for index, ticker in enumerate(selected, start=1):
                status.caption(f"正在掃描 {index}/{len(selected)}：{ticker}")
                df, meta = fetch_ohlcv(ticker)
                if df is None:
                    failures.append({"代碼": ticker, "原因": meta["status"]})
                    errors.append(f"{ticker}: {meta['status']}")
                else:
                    try:
                        result = score_stock_v3(df, regime, benchmark_close, ticker_market(ticker))
                        persist_scan(run_id, ticker, result, meta, regime)
                        rows.append({
                            "代碼": ticker,
                            "資格狀態": "通過" if result.eligible else "不通過",
                            "底部結構": result.bottom_score,
                            "反轉確認": result.confirmation_score,
                            "品質": result.quality_score,
                            "風險扣減": result.risk_deduction,
                            "總分": result.total_score,
                            "現價": result.price,
                            "確認買入價": result.trigger_price,
                            "止損": result.stop_price,
                            "2R目標": result.target_price,
                            "R/R": result.risk_reward,
                            "行動結論": result.action,
                            "不合格原因": "；".join(result.blockers) if result.blockers else "—",
                            "成立原因": "；".join(result.reasons),
                        })
                        if result.action == "可小量試倉":
                            persist_signal(ticker, str(df.index[-1].date()), result, meta, regime)
                    except Exception as exc:
                        message = f"V3 評分失敗：{type(exc).__name__}: {exc}"
                        failures.append({"代碼": ticker, "原因": message})
                        errors.append(f"{ticker}: {message}")
                progress.progress(index / len(selected))
            status.empty()
            finish_scan_run(run_id, len(rows), len(failures), errors)
            if rows:
                results = pd.DataFrame(rows)
                order = {"可小量試倉": 0, "等待突破": 1, "觀察": 2, "不合格": 3}
                results["_sort"] = results["行動結論"].map(order).fillna(9)
                results = results.sort_values(["_sort", "總分"], ascending=[True, False]).drop(columns=["_sort"])
                trial_count = int((results["行動結論"] == "可小量試倉").sum())
                waiting_count = int((results["行動結論"] == "等待突破").sum())
                st.success(f"掃描完成：可小量試倉 {trial_count} 隻｜等待突破 {waiting_count} 隻｜完成 {len(rows)} 隻｜失敗 {len(failures)} 隻")
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button("下載 V3 掃描 CSV", results.to_csv(index=False).encode("utf-8-sig"), "v3_scan_results.csv", "text/csv")
            if failures:
                st.warning(f"有 {len(failures)} 隻未完成掃描。")
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

with detail_tab:
    st.subheader("V3 個股反轉候選詳情")
    ticker = st.text_input("代碼", tickers[0] if tickers else "AAPL", key="detail_ticker")
    period = st.selectbox("圖表範圍", ["1y", "2y", "3y", "5y"], index=1)
    if st.button("載入 V3 個股詳情"):
        df, meta = fetch_ohlcv(ticker, period)
        benchmark_ticker = "SPY" if ticker_market(ticker) == "US" else "^HSI"
        benchmark_df, _ = fetch_ohlcv(benchmark_ticker, period)
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock_v3(df, regime, benchmark_close, ticker_market(ticker))
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("資格", "通過" if result.eligible else "不通過")
            p2.metric("總分", f"{result.total_score:.1f}/90")
            p3.metric("行動結論", result.action)
            p4.metric("確認買入價", result.trigger_price)
            p5.metric("2R目標", result.target_price)
            p6, p7, p8, p9 = st.columns(4)
            p6.metric("底部結構", f"{result.bottom_score:.1f}/35")
            p7.metric("反轉確認", f"{result.confirmation_score:.1f}/35")
            p8.metric("品質", f"{result.quality_score:.1f}/20")
            p9.metric("風險扣減", f"{result.risk_deduction:.1f}")
            st.caption(f"現價：{result.price}｜結構止損：{result.stop_price}｜預設 R/R：{result.risk_reward}｜資料最後日：{meta['last_bar']}")
            st.plotly_chart(make_chart(df, normalize_ticker(ticker)), use_container_width=True)
            if result.blockers:
                st.error("不合格原因：" + "；".join(result.blockers))
            st.markdown("### 已成立條件")
            if result.reasons:
                for reason in result.reasons:
                    st.write(f"• {reason}")
            else:
                st.write("目前未出現足夠的底部或反轉確認。")
            st.markdown("### 量化指標")
            st.dataframe(pd.DataFrame([result.metrics]), use_container_width=True, hide_index=True)
            st.warning("確認買入價是近 5 日高位突破價，不等於要求立即以現價買入。歷史直駁率尚未校準，因此分數不能解讀為見底機率。")

with log_tab:
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
                st.info("尚未有保存的 V3 試倉候選訊號。")
        except Exception as exc:
            st.error(f"讀取訊號失敗：{type(exc).__name__}: {exc}")

with risk_tab:
    st.subheader("風險為本的部位計算")
    r1, r2, r3, r4 = st.columns(4)
    account = r1.number_input("帳戶淨值", min_value=1000.0, value=100000.0, step=1000.0)
    risk_percent = r2.slider("每筆最大帳戶風險 (%)", 0.25, 2.0, 0.75, 0.25)
    allocation_limit = r3.slider("單一持倉最大名義比例 (%)", 1.0, 30.0, 10.0, 1.0)
    risk_ticker = r4.text_input("代碼", tickers[0] if tickers else "AAPL", key="risk_ticker")
    if st.button("計算可承受部位"):
        df, meta = fetch_ohlcv(risk_ticker, "2y")
        benchmark_ticker = "SPY" if ticker_market(risk_ticker) == "US" else "^HSI"
        benchmark_df, _ = fetch_ohlcv(benchmark_ticker, "2y")
        benchmark_close = benchmark_df["close"] if benchmark_df is not None else None
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock_v3(df, regime, benchmark_close, ticker_market(risk_ticker))
            entry = result.trigger_price
            per_share_risk = entry - result.stop_price
            budget = account * risk_percent / 100
            shares_by_risk = int(budget / per_share_risk) if per_share_risk > 0 else 0
            shares_by_allocation = int((account * allocation_limit / 100) / entry) if entry > 0 else 0
            shares = max(0, min(shares_by_risk, shares_by_allocation))
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("確認買入價", entry)
            q2.metric("結構止損", result.stop_price)
            q3.metric("每股風險", f"{per_share_risk:.3f}")
            q4.metric("建議上限股數", shares)
            st.write(f"風險預算：{budget:,.2f}｜名義金額：約 {shares * entry:,.2f}｜2R目標：{result.target_price}")
            st.warning("下單前請按整手規則向下調整，並自行考慮匯率、手續費、買賣價差、稅項及止損成交風險。")
