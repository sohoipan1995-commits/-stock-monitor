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

st.set_page_config(page_title="股票監察系統 Pro · 雲端觀察名單版", page_icon="📈", layout="wide")

MODEL_VERSION = "2.1.0"
DEFAULT_HK = ["0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "2318.HK", "9988.HK", "3690.HK", "9618.HK", "1211.HK"]
DEFAULT_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM", "SPY", "QQQ"]
SOURCE_LABELS = {
    "manual": "手動永久", "portfolio": "持倉", "signal_high_score": "高分訊號",
    "sp500_constituent": "S&P 500", "hsi_constituent": "恒指成分", "hsi_top30_turnover": "恒指20日成交 Top 30",
}


# ---------- Connection, data quality and database ----------
def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(" ", "")


def ticker_market(ticker: str) -> str:
    return "HK" if ticker.endswith(".HK") else "US"


@st.cache_resource
def supabase_client() -> Client | None:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_SECRET_KEY"]
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def db_error(where: str, exc: Exception) -> None:
    st.session_state["db_last_error"] = f"{where}: {type(exc).__name__}: {exc}"


def db_upsert(table: str, rows: dict | list[dict], on_conflict: str) -> bool:
    client = supabase_client()
    if client is None:
        return False
    try:
        client.table(table).upsert(rows, on_conflict=on_conflict).execute()
        return True
    except Exception as exc:
        db_error(f"寫入 {table}", exc)
        return False


def db_select(table: str, columns: str = "*", **eq: Any) -> list[dict]:
    client = supabase_client()
    if client is None:
        return []
    try:
        query = client.table(table).select(columns)
        for key, value in eq.items():
            query = query.eq(key, value)
        return query.execute().data or []
    except Exception as exc:
        db_error(f"讀取 {table}", exc)
        return []


def upsert_instruments(rows: list[dict]) -> bool:
    if not rows:
        return True
    now = datetime.utcnow().isoformat()
    payload = [{
        "ticker": normalize_ticker(r["ticker"]), "market": r.get("market", ticker_market(r["ticker"])),
        "name": r.get("name"), "sector": r.get("sector"), "industry": r.get("industry"),
        "currency": "HKD" if r.get("market", ticker_market(r["ticker"])) == "HK" else "USD",
        "is_active": True, "updated_at": now,
    } for r in rows]
    return db_upsert("instruments", payload, "ticker")


def add_membership(ticker: str, source: str, permanent: bool = False, priority: int = 50, notes: str | None = None) -> bool:
    ticker = normalize_ticker(ticker)
    upsert_instruments([{"ticker": ticker, "market": ticker_market(ticker)}])
    payload = {
        "ticker": ticker, "source": source, "is_permanent": permanent, "is_active": True,
        "priority": priority, "notes": notes, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None,
    }
    return db_upsert("watchlist_memberships", payload, "ticker,source")


def deactivate_source_not_in(source: str, active_tickers: set[str]) -> None:
    client = supabase_client()
    if client is None:
        return
    try:
        existing = client.table("watchlist_memberships").select("id,ticker").eq("source", source).eq("is_active", True).execute().data or []
        to_disable = [r["id"] for r in existing if r["ticker"] not in active_tickers]
        for item_id in to_disable:
            client.table("watchlist_memberships").update({"is_active": False, "removed_at": datetime.utcnow().isoformat()}).eq("id", item_id).execute()
    except Exception as exc:
        db_error("更新已退出候選", exc)


def validate_ohlcv(df: pd.DataFrame | None) -> tuple[bool, str]:
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        return False, "沒有取得價格資料"
    if required - set(df.columns):
        return False, "資料欄位不完整"
    if len(df) < 80:
        return False, "歷史資料少於 80 個交易日"
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
            raw.columns = [str(x).lower() for x in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            valid, status = validate_ohlcv(df)
            return (df if valid else None), {"ticker": ticker, "source": "Yahoo Finance", "adjusted": True, "last_bar": str(df.index[-1].date()) if len(df) else None, "rows": len(df), "status": status}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.7 * (2 ** attempt))
    return None, {"ticker": ticker, "source": "Yahoo Finance", "rows": 0, "status": "；".join(errors)}


# ---------- Index constituent imports ----------
def _clean_us_ticker(value: Any) -> str:
    return normalize_ticker(str(value)).replace(".", "-")


def _clean_hk_code(value: Any) -> str | None:
    digits = "".join(char for char in str(value) if char.isdigit())
    if not digits:
        return None
    return f"{digits.zfill(4)}.HK"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents() -> tuple[pd.DataFrame, str]:
    """Imports the public S&P 500 table. Review the source/status before relying on it for trading."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    table = next((x for x in tables if "Symbol" in x.columns and "Security" in x.columns), None)
    if table is None:
        raise RuntimeError("找不到 S&P 500 成分表")
    result = pd.DataFrame({
        "ticker": table["Symbol"].map(_clean_us_ticker), "name": table["Security"].astype(str),
        "sector": table["GICS Sector"].astype(str) if "GICS Sector" in table.columns else None,
        "industry": table["GICS Sub-Industry"].astype(str) if "GICS Sub-Industry" in table.columns else None,
        "market": "US",
    }).drop_duplicates("ticker")
    return result, "Wikipedia S&P 500 constituent table (verify against official S&P source before production trading)"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hsi_constituents() -> tuple[pd.DataFrame, str]:
    """Imports a public HSI constituent table and normalizes HKEX/Yahoo tickers."""
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    tables = pd.read_html(url)
    best = None
    for table in tables:
        cols = [str(x).lower() for x in table.columns]
        has_code = any("code" in x or "ticker" in x for x in cols)
        if has_code and len(table) >= 40:
            best = table
            break
    if best is None:
        raise RuntimeError("找不到恒生指數成分表；請稍後重試或改用已驗證的資料源")
    code_col = next(col for col in best.columns if "code" in str(col).lower() or "ticker" in str(col).lower())
    name_col = next((col for col in best.columns if any(x in str(col).lower() for x in ["company", "name", "constituent"])), None)
    sector_col = next((col for col in best.columns if "sector" in str(col).lower() or "industry" in str(col).lower()), None)
    result = pd.DataFrame({
        "ticker": best[code_col].map(_clean_hk_code), "name": best[name_col].astype(str) if name_col else None,
        "sector": best[sector_col].astype(str) if sector_col else None, "industry": None, "market": "HK",
    }).dropna(subset=["ticker"]).drop_duplicates("ticker")
    if len(result) < 40:
        raise RuntimeError("恒指成分表資料不完整")
    return result, "Wikipedia Hang Seng Index constituent table (verify against Hang Seng Indexes official constituent file before production trading)"


def sync_sp500() -> tuple[bool, str, int]:
    try:
        universe, source_note = fetch_sp500_constituents()
        rows = universe.to_dict("records")
        if not upsert_instruments(rows):
            return False, "寫入 instruments 失敗", 0
        today = date.today().isoformat()
        snapshots, memberships = [], []
        for row in rows:
            ticker = row["ticker"]
            snapshots.append({"snapshot_date": today, "ticker": ticker, "source": "sp500_constituent", "index_member": True, "is_selected": True})
            memberships.append({"ticker": ticker, "source": "sp500_constituent", "is_permanent": False, "is_active": True, "priority": 30, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None})
        db_upsert("watchlist_snapshots", snapshots, "snapshot_date,ticker,source")
        db_upsert("watchlist_memberships", memberships, "ticker,source")
        deactivate_source_not_in("sp500_constituent", set(universe["ticker"]))
        return True, source_note, len(universe)
    except Exception as exc:
        return False, f"S&P 500 同步失敗：{type(exc).__name__}: {exc}", 0


def sync_hsi_and_top30() -> tuple[bool, str, int, int]:
    try:
        universe, source_note = fetch_hsi_constituents()
        rows = universe.to_dict("records")
        if not upsert_instruments(rows):
            return False, "寫入 instruments 失敗", 0, 0
        today = date.today().isoformat()
        hsi_memberships = [{"ticker": r["ticker"], "source": "hsi_constituent", "is_permanent": False, "is_active": True, "priority": 40, "last_confirmed_at": datetime.utcnow().isoformat(), "removed_at": None} for r in rows]
        hsi_snapshots = [{"snapshot_date": today, "ticker": r["ticker"], "source": "hsi_constitu
