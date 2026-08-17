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

st.set_page_config(
    page_title="股票監察系統 Pro · V3.2",
    page_icon="📈",
    layout="wide",
)

MODEL_VERSION = "3.2.0"
WEB_SCAN_LIMIT = 80

MAG7 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
]

DEFAULT_US = MAG7 + [
    "AVGO",
    "AMD",
    "JPM",
    "SPY",
    "QQQ",
]

DEFAULT_HK = [
    "0700.HK",
    "0005.HK",
    "0939.HK",
    "1398.HK",
    "3988.HK",
    "0388.HK",
    "2318.HK",
    "9988.HK",
    "3690.HK",
    "9618.HK",
    "1211.HK",
]

SOURCE_LABELS = {
    "manual": "手動永久",
    "portfolio": "持倉",
    "signal_high_score": "V3 試倉候選",
    "sp500_constituent": "S&P 500 成分",
    "sp500_top30_turnover": "S&P 500 20日成交額 Top 30",
    "mag7_priority": "Mag 7 優先",
    "hsi_constituent": "恒指成分",
    "hsi_top30_turnover": "恒指20日成交額 Top 30",
}


# ============================================================
# 基本工具及 Supabase 資料庫連線
# ============================================================

def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(" ", "")


def ticker_market(ticker: str) -> str:
    if normalize_ticker(ticker).endswith(".HK"):
        return "HK"
    return "US"


@st.cache_resource
def get_supabase() -> Client | None:
    try:
        return create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_SECRET_KEY"],
        )
    except Exception:
        return None


def set_db_error(where: str, exc: Exception) -> None:
    st.session_state["db_error"] = (
        f"{where}: {type(exc).__name__}: {exc}"
    )


def db_upsert(
    table: str,
    data: dict | list[dict],
    conflict: str,
) -> bool:
    client = get_supabase()

    if client is None:
        return False

    try:
        client.table(table).upsert(
            data,
            on_conflict=conflict,
        ).execute()
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

        payload.append(
            {
                "ticker": ticker,
                "market": market,
                "name": row.get("name"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "currency": "HKD" if market == "HK" else "USD",
                "is_active": True,
                "updated_at": now,
            }
        )

    if not payload:
        return True

    return db_upsert(
        "instruments",
        payload,
        "ticker",
    )


def add_membership(
    ticker: str,
    source: str,
    permanent: bool,
    priority: int,
    notes: str | None = None,
) -> bool:
    ticker = normalize_ticker(ticker)

    if not upsert_instruments(
        [
            {
                "ticker": ticker,
                "market": ticker_market(ticker),
            }
        ]
    ):
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


def deactivate_missing(
    source: str,
    active_tickers: set[str],
) -> None:
    client = get_supabase()

    if client is None:
        return

    try:
        existing = (
            client.table("watchlist_memberships")
            .select("id,ticker")
            .eq("source", source)
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        for row in existing:
            if row["ticker"] not in active_tickers:
                (
                    client.table("watchlist_memberships")
                    .update(
                        {
                            "is_active": False,
                            "removed_at": datetime.utcnow().isoformat(),
                        }
                    )
                    .eq("id", row["id"])
                    .execute()
                )

    except Exception as exc:
        set_db_error("更新候選狀態", exc)


def get_watchlist(market: str | None = None) -> pd.DataFrame:
    client = get_supabase()

    if client is None:
        return pd.DataFrame()

    try:
        memberships = (
            client.table("watchlist_memberships")
            .select(
                "ticker,source,is_permanent,priority,"
                "notes,added_at,last_confirmed_at"
            )
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        instruments = (
            client.table("instruments")
            .select("ticker,market,name,sector")
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        if not memberships:
            return pd.DataFrame()

        output = pd.DataFrame(memberships).merge(
            pd.DataFrame(instruments),
            on="ticker",
            how="left",
        )

        if market in {"US", "HK"}:
            output = output[output["market"] == market]

        return output.sort_values(
            ["is_permanent", "priority", "ticker"],
            ascending=[False, False, True],
        )

    except Exception as exc:
        set_db_error("讀取觀察名單", exc)
        return pd.DataFrame()


# ============================================================
# Yahoo Finance 價格資料
# ============================================================

def validate_ohlcv(
    df: pd.DataFrame | None,
) -> tuple[bool, str]:
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if df is None or df.empty:
        return False, "沒有取得價格資料"

    if required - set(df.columns):
        return False, "OHLCV 欄位不完整"

    if len(df) < 80:
        return False, "歷史資料少於 80 個交易日"

    if df[list(required)].isna().any().any():
        return False, "OHLCV 有遺漏值"

    if (df["close"] <= 0).any():
        return False, "價格資料不合理"

    if (df["volume"] < 0).any():
        return False, "成交量資料不合理"

    return True, "OK"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(
    ticker: str,
    period: str = "3y",
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    errors = []

    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [
                str(column).lower()
                for column in raw.columns
            ]

            df = raw[
                ["open", "high", "low", "close", "volume"]
            ].dropna().copy()

            df.index = pd.to_datetime(
                df.index
            ).tz_localize(None)

            valid, status = validate_ohlcv(df)

            meta = {
                "ticker": ticker,
                "source": "Yahoo Finance",
                "adjusted": True,
                "last_bar": (
                    str(df.index[-1].date())
                    if not df.empty
                    else None
                ),
                "rows": len(df),
                "status": status,
            }

            return (df if valid else None), meta

        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(0.8 * (2 ** attempt))

    return None, {
        "ticker": ticker,
        "source": "Yahoo Finance",
        "last_bar": None,
        "rows": 0,
        "status": "；".join(errors),
    }


# ============================================================
# S&P 500、Mag 7、恒指成分股及高成交額核心池
# ============================================================

def clean_us_ticker(value: Any) -> str:
    return normalize_ticker(value).replace(".", "-")


def clean_hk_ticker(value: Any) -> str | None:
    digits = "".join(
        char
        for char in str(value)
        if char.isdigit()
    )

    if not digits:
        return None

    return f"{digits.zfill(4)}.HK"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies",
        headers={
            "User-Agent": (
                "StockMonitorPro/3.2 "
                "research application"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(response.text),
        attrs={"id": "constituents"},
    )

    if not tables:
        raise RuntimeError("找不到 S&P 500 成分表")

    table = tables[0]

    if not {"Symbol", "Security"}.issubset(
        table.columns
    ):
        raise RuntimeError("S&P 500 成分表格式已改變")

    return pd.DataFrame(
        {
            "ticker": table["Symbol"].map(
                clean_us_ticker
            ),
            "name": table["Security"].astype(str),
            "sector": (
                table["GICS Sector"].astype(str)
                if "GICS Sector" in table.columns
                else None
            ),
            "industry": (
                table["GICS Sub-Industry"].astype(str)
                if "GICS Sub-Industry" in table.columns
                else None
            ),
            "market": "US",
        }
    ).drop_duplicates("ticker")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hsi_constituents() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        headers={
            "User-Agent": (
                "StockMonitorPro/3.2 "
                "research application"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(response.text)
    )

    candidate = None

    for table in tables:
        columns = [
            str(column).lower()
            for column in table.columns
        ]

        has_code = any(
            "code" in column or "ticker" in column
            for column in columns
        )

        if len(table) >= 40 and has_code:
            candidate = table
            break

    if candidate is None:
        raise RuntimeError("找不到恒指成分表")

    code_column = next(
        column
        for column in candidate.columns
        if (
            "code" in str(column).lower()
            or "ticker" in str(column).lower()
        )
    )

    name_column = next(
        (
            column
            for column in candidate.columns
            if (
                "company" in str(column).lower()
                or "name" in str(column).lower()
            )
        ),
        None,
    )

    sector_column = next(
        (
            column
            for column in candidate.columns
            if (
                "sector" in str(column).lower()
                or "industry" in str(column).lower()
            )
        ),
        None,
    )

    result = pd.DataFrame(
        {
            "ticker": candidate[code_column].map(
                clean_hk_ticker
            ),
            "name": (
                candidate[name_column].astype(str)
                if name_column
                else None
            ),
            "sector": (
                candidate[sector_column].astype(str)
                if sector_column
                else None
            ),
            "industry": None,
            "market": "HK",
        }
    ).dropna(subset=["ticker"]).drop_duplicates(
        "ticker"
    )

    if len(result) < 40:
        raise RuntimeError("恒指成分資料不完整")

    return result
    def sync_sp500_and_us_top30() -> tuple[bool, str]:
    """
    同步全部 S&P 500 成分股至資料庫，
    再計算最近 20 日平均成交額 Top 30，
    並固定把 Mag 7 加入日常美股核心掃描池。
    """
    try:
        universe = fetch_sp500_constituents()

        if not upsert_instruments(
            universe.to_dict("records")
        ):
            return False, "無法寫入 instruments"

        now = datetime.utcnow().isoformat()
        today = date.today().isoformat()

        sp_members = []

        for ticker in universe["ticker"]:
            sp_members.append(
                {
                    "ticker": ticker,
                    "source": "sp500_constituent",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 20,
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        sp_snapshots = []

        for ticker in universe["ticker"]:
            sp_snapshots.append(
                {
                    "snapshot_date": today,
                    "ticker": ticker,
                    "source": "sp500_constituent",
                    "index_member": True,
                    "is_selected": True,
                }
            )

        if not db_upsert(
            "watchlist_memberships",
            sp_members,
            "ticker,source",
        ):
            return False, st.session_state.get(
                "db_error",
                "S&P 500 寫入失敗",
            )

        db_upsert(
            "watchlist_snapshots",
            sp_snapshots,
            "snapshot_date,ticker,source",
        )

        deactivate_missing(
            "sp500_constituent",
            set(universe["ticker"]),
        )

        # Mag 7 永久優先加入美股核心掃描池。
        valid_mag7 = [
            ticker
            for ticker in MAG7
            if ticker in set(universe["ticker"])
        ]

        mag7_members = []

        for ticker in valid_mag7:
            mag7_members.append(
                {
                    "ticker": ticker,
                    "source": "mag7_priority",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 100,
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        db_upsert(
            "watchlist_memberships",
            mag7_members,
            "ticker,source",
        )

        deactivate_missing(
            "mag7_priority",
            set(valid_mag7),
        )

        # 計算 S&P 500 中最近 20 日平均成交額最高的 30 隻。
        ranking_rows = []

        for ticker in universe["ticker"].tolist():
            df, _ = fetch_ohlcv(ticker, "6mo")

            if df is None or len(df) < 20:
                continue

            average_turnover = float(
                (
                    df["close"].iloc[-20:]
                    * df["volume"].iloc[-20:]
                ).mean()
            )

            ranking_rows.append(
                {
                    "ticker": ticker,
                    "turnover_20d": average_turnover,
                    "average_volume_20d": float(
                        df["volume"].iloc[-20:].mean()
                    ),
                    "close_price": float(
                        df["close"].iloc[-1]
                    ),
                }
            )

        ranking = (
            pd.DataFrame(ranking_rows)
            .sort_values(
                "turnover_20d",
                ascending=False,
            )
            .head(30)
            .reset_index(drop=True)
        )

        if ranking.empty:
            return False, (
                "未能取得足夠 S&P 500 成分股價格資料"
            )

        ranking["turnover_rank"] = ranking.index + 1

        top_members = []
        top_snapshots = []

        for row in ranking.itertuples():
            top_members.append(
                {
                    "ticker": row.ticker,
                    "source": "sp500_top30_turnover",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 90,
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

            top_snapshots.append(
                {
                    "snapshot_date": today,
                    "ticker": row.ticker,
                    "source": "sp500_top30_turnover",
                    "index_member": True,
                    "turnover_20d": row.turnover_20d,
                    "turnover_rank": int(
                        row.turnover_rank
                    ),
                    "average_volume_20d": (
                        row.average_volume_20d
                    ),
                    "close_price": row.close_price,
                    "is_selected": True,
                }
            )

        db_upsert(
            "watchlist_memberships",
            top_members,
            "ticker,source",
        )

        db_upsert(
            "watchlist_snapshots",
            top_snapshots,
            "snapshot_date,ticker,source",
        )

        deactivate_missing(
            "sp500_top30_turnover",
            set(ranking["ticker"]),
        )

        return True, (
            f"已同步 {len(universe)} 隻 S&P 500；"
            f"已加入 Mag 7；並選出 "
            f"20 日平均成交額 Top {len(ranking)}。"
        )

    except Exception as exc:
        return False, (
            "美股核心池同步失敗："
            f"{type(exc).__name__}: {exc}"
        )


def sync_hsi_top30() -> tuple[bool, str]:
    """
    同步恒指成分股，再以最近 20 日平均成交額
    選出恒指成分內成交額最高的 Top 30。
    """
    try:
        universe = fetch_hsi_constituents()

        if not upsert_instruments(
            universe.to_dict("records")
        ):
            return False, "無法寫入 instruments"

        now = datetime.utcnow().isoformat()
        today = date.today().isoformat()

        hsi_members = []

        for ticker in universe["ticker"]:
            hsi_members.append(
                {
                    "ticker": ticker,
                    "source": "hsi_constituent",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 40,
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        hsi_snapshots = []

        for ticker in universe["ticker"]:
            hsi_snapshots.append(
                {
                    "snapshot_date": today,
                    "ticker": ticker,
                    "source": "hsi_constituent",
                    "index_member": True,
                    "is_selected": True,
                }
            )

        db_upsert(
            "watchlist_memberships",
            hsi_members,
            "ticker,source",
        )

        db_upsert(
            "watchlist_snapshots",
            hsi_snapshots,
            "snapshot_date,ticker,source",
        )

        deactivate_missing(
            "hsi_constituent",
            set(universe["ticker"]),
        )

        ranking_rows = []

        for ticker in universe["ticker"].tolist():
            df, _ = fetch_ohlcv(ticker, "6mo")

            if df is None or len(df) < 20:
                continue

            average_turnover = float(
                (
                    df["close"].iloc[-20:]
                    * df["volume"].iloc[-20:]
                ).mean()
            )

            ranking_rows.append(
                {
                    "ticker": ticker,
                    "turnover_20d": average_turnover,
                    "average_volume_20d": float(
                        df["volume"].iloc[-20:].mean()
                    ),
                    "close_price": float(
                        df["close"].iloc[-1]
                    ),
                }
            )

        ranking = (
            pd.DataFrame(ranking_rows)
            .sort_values(
                "turnover_20d",
                ascending=False,
            )
            .head(30)
            .reset_index(drop=True)
        )

        if ranking.empty:
            return False, (
                "未取得足夠恒指成分股價格資料"
            )

        ranking["turnover_rank"] = ranking.index + 1

        top_members = []
        top_snapshots = []

        for row in ranking.itertuples():
            top_members.append(
                {
                    "ticker": row.ticker,
                    "source": "hsi_top30_turnover",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 90,
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

            top_snapshots.append(
                {
                    "snapshot_date": today,
                    "ticker": row.ticker,
                    "source": "hsi_top30_turnover",
                    "index_member": True,
                    "turnover_20d": row.turnover_20d,
                    "turnover_rank": int(
                        row.turnover_rank
                    ),
                    "average_volume_20d": (
                        row.average_volume_20d
                    ),
                    "close_price": row.close_price,
                    "is_selected": True,
                }
            )

        db_upsert(
            "watchlist_memberships",
            top_members,
            "ticker,source",
        )

        db_upsert(
            "watchlist_snapshots",
            top_snapshots,
            "snapshot_date,ticker,source",
        )

        deactivate_missing(
            "hsi_top30_turnover",
            set(ranking["ticker"]),
        )

        return True, (
            f"已同步 {len(universe)} 隻恒指成分股，"
            f"並選出 20 日平均成交額 "
            f"Top {len(ranking)}。"
        )

    except Exception as exc:
        return False, (
            "恒指／Top 30 同步失敗："
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# V3 技術反轉評分：指標、資格、反轉確認、風險
# ============================================================

def rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
    ).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return 100 - 100 / (
        1 + gain / loss.replace(0, np.nan)
    )


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    previous_close = df["close"].shift()

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def cmf(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    spread = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    multiplier = (
        (
            2 * df["close"]
            - df["high"]
            - df["low"]
        )
        / spread
    ).fillna(0)

    return (
        (multiplier * df["volume"])
        .rolling(period)
        .sum()
        / df["volume"]
        .rolling(period)
        .sum()
        .replace(0, np.nan)
    )


def weekly_rsi(close: pd.Series) -> pd.Series:
    weekly_close = (
        close.resample("W-FRI")
        .last()
        .dropna()
    )

    return rsi(weekly_close).reindex(
        close.index,
        method="ffill",
    )


def relative_strength_20d(
    close: pd.Series,
    benchmark_close: pd.Series | None,
) -> float:
    if benchmark_close is None or len(close) < 21:
        return float("nan")

    aligned = benchmark_close.reindex(
        close.index,
        method="ffill",
    ).dropna()

    if len(aligned) < 21:
        return float("nan")

    stock_return = (
        close.iloc[-1] / close.iloc[-21] - 1
    )

    benchmark_return = (
        aligned.iloc[-1] / aligned.iloc[-21] - 1
    )

    return float(
        100 * (stock_return - benchmark_return)
    )


@dataclass
class V3ScoreResult:
    eligible: bool
    action: str
    total_score: float
    bottom_score: float
    confirmation_score: float
    quality_score: float
    risk_deduction: float
    price: float
    trigger_price: float
    stop_price: float
    target_price: float
    risk_reward: float | None
    reasons: list[str]
    blockers: list[str]
    metrics: dict[str, float]


def score_stock_v3(
    df: pd.DataFrame,
    regime: str,
    benchmark_close: pd.Series | None,
    market: str,
) -> V3ScoreResult:
    if df is None or len(df) < 252:
        return V3ScoreResult(
            eligible=False,
            action="不合格",
            total_score=0,
            bottom_score=0,
            confirmation_score=0,
            quality_score=0,
            risk_deduction=0,
            price=np.nan,
            trigger_price=np.nan,
            stop_price=np.nan,
            target_price=np.nan,
            risk_reward=None,
            reasons=[],
            blockers=["歷史資料少於 252 個交易日"],
            metrics={},
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    price = float(close.iloc[-1])

    daily_rsi = rsi(close)
    week_rsi = weekly_rsi(close)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()

    atr14 = atr(df)
    cmf20 = cmf(df)
    volume20 = volume.rolling(20).mean()

    dollar_volume20 = float(
        (
            close.iloc[-20:]
            * volume.iloc[-20:]
        ).mean()
    )

    volume_ratio = (
        float(volume.iloc[-1] / volume20.iloc[-1])
        if volume20.iloc[-1] > 0
        else np.nan
    )

    atr_pct = float(
        100 * atr14.iloc[-1] / price
    )

    rs20 = relative_strength_20d(
        close,
        benchmark_close,
    )

    required_series = [
        daily_rsi,
        week_rsi,
        ma20,
        ma60,
        ma200,
        atr14,
        cmf20,
        volume20,
    ]

    if any(
        pd.isna(series.iloc[-1])
        for series in required_series
    ):
        return V3ScoreResult(
            eligible=False,
            action="不合格",
            total_score=0,
            bottom_score=0,
            confirmation_score=0,
            quality_score=0,
            risk_deduction=0,
            price=price,
            trigger_price=np.nan,
            stop_price=np.nan,
            target_price=np.nan,
            risk_reward=None,
            reasons=[],
            blockers=["指標暖機資料不足"],
            metrics={},
        )

    blockers = []
    reasons = []

    minimum_turnover = (
        20_000_000
        if market == "US"
        else 50_000_000
    )

    if market == "US" and price < 5:
        blockers.append("美股股價低於 5 美元")

    if dollar_volume20 < minimum_turnover:
        blockers.append(
            f"20 日平均成交額不足 "
            f"{minimum_turnover:,.0f}"
        )

    # 底部結構：0 至 35 分。
    bottom_score = 0.0

    low60 = float(low.iloc[-60:].min())

    distance_from_low = (
        100 * (price / low60 - 1)
        if low60 > 0
        else np.nan
    )

    drawdown_252 = 100 * (
        price / float(high.iloc[-252:].max()) - 1
    )

    if daily_rsi.iloc[-1] <= 32:
        bottom_score += 8

        reasons.append(
            f"日 RSI {daily_rsi.iloc[-1]:.1f} 偏低，"
            "代表近期跌勢較急"
        )

    if week_rsi.iloc[-1] <= 45:
        bottom_score += 6

        reasons.append(
            f"週 RSI {week_rsi.iloc[-1]:.1f} 偏低"
        )

    if distance_from_low <= 6:
        bottom_score += 8

        reasons.append(
            f"現價距 60 日低位僅 "
            f"{distance_from_low:.1f}%"
        )

    if -45 <= drawdown_252 <= -12:
        bottom_score += 5

        reasons.append(
            f"距 52 周高回撤 "
            f"{drawdown_252:.1f}%"
        )

    if (
        cmf20.iloc[-1] > cmf20.iloc[-6]
        and cmf20.iloc[-1] > -0.08
    ):
        bottom_score += 8

        reasons.append(
            "資金流指標 CMF 正在改善，"
            "賣壓可能減弱"
        )

    bottom_score = min(bottom_score, 35)

    # 第 3 段會繼續：反轉確認、品質、風險扣減、Regime 分析及全部 UI。
        # 反轉確認：0 至 35 分。
    # 重點不是「跌得夠多」，而是有沒有開始由跌轉升。
    confirmation_score = 0.0

    trigger_price = float(
        high.iloc[-6:-1].max()
    )

    if price > trigger_price:
        confirmation_score += 12

        reasons.append(
            f"收市突破近 5 日確認買入價 "
            f"{trigger_price:.2f}"
        )

    if price > ma20.iloc[-1]:
        confirmation_score += 8

        reasons.append(
            "收市重回 20 日平均線上方"
        )

    if ma20.iloc[-1] > ma20.iloc[-6]:
        confirmation_score += 5

        reasons.append(
            "20 日平均線開始上彎"
        )

    if (
        volume_ratio >= 1.3
        and close.iloc[-1] > df["open"].iloc[-1]
    ):
        confirmation_score += 5

        reasons.append(
            f"放量收陽，成交量約為 "
            f"20 日均量的 {volume_ratio:.2f} 倍"
        )

    if cmf20.iloc[-1] > 0:
        confirmation_score += 5

        reasons.append(
            "CMF 為正，近期資金流偏向流入"
        )

    confirmation_score = min(
        confirmation_score,
        35,
    )

    # 品質分：0 至 20 分。
    # 優先挑選長期趨勢較好、較能跑贏大市的股票。
    quality_score = 0.0

    if price > ma60.iloc[-1]:
        quality_score += 6

    if ma200.iloc[-1] > ma200.iloc[-21]:
        quality_score += 6

        reasons.append(
            "200 日長期趨勢仍向上"
        )

    if not np.isnan(rs20) and rs20 >= 0:
        quality_score += 8

        reasons.append(
            f"20 日相對大市強弱 "
            f"{rs20:+.1f}%"
        )

    elif not np.isnan(rs20) and rs20 < -8:
        reasons.append(
            f"20 日明顯跑輸大市 "
            f"{rs20:.1f}%"
        )

    quality_score = min(quality_score, 20)

    # 風險扣減：0 至 -20 分。
    risk_deduction = 0.0

    if "熊市" in regime:
        risk_deduction -= 7

        reasons.append(
            "熊市環境：逆勢反彈的失敗風險較高"
        )

    if atr_pct > 8:
        risk_deduction -= 7

        reasons.append(
            f"個股日常波動偏高 "
            f"（ATR {atr_pct:.1f}%）"
        )

    elif atr_pct > 5:
        risk_deduction -= 3

        reasons.append(
            f"個股波動偏高 "
            f"（ATR {atr_pct:.1f}%）"
        )

    if not np.isnan(rs20) and rs20 < -12:
        risk_deduction -= 4

    # 入場價使用「現價」或「確認買入價」中較高者。
    # 這表示未突破前不應把超賣本身當作立即買入訊號。
    entry_price = max(price, trigger_price)

    structure_stop = float(
        low.iloc[-10:].min()
        - 0.30 * atr14.iloc[-1]
    )

    volatility_stop = float(
        entry_price
        - 1.50 * atr14.iloc[-1]
    )

    stop_price = round(
        min(structure_stop, volatility_stop),
        3,
    )

    risk_per_share = entry_price - stop_price

    target_price = (
        round(
            entry_price + 2.0 * risk_per_share,
            3,
        )
        if risk_per_share > 0
        else np.nan
    )

    risk_reward = (
        round(
            (target_price - entry_price)
            / risk_per_share,
            2,
        )
        if risk_per_share > 0
        else None
    )

    if risk_per_share <= 0:
        blockers.append("無法建立有效止損")

    elif risk_per_share / entry_price > 0.10:
        blockers.append(
            "止損距離超過入場價 10%，風險過寬"
        )

    total_score = round(
        max(
            0,
            bottom_score
            + confirmation_score
            + quality_score
            + risk_deduction,
        ),
        1,
    )

    eligible = len(blockers) == 0

    if not eligible:
        action = "不合格"

    elif confirmation_score < 12:
        action = "等待突破"

    elif total_score < 65:
        action = "觀察"

    else:
        action = "可小量試倉"

    metrics = {
        "日 RSI": round(
            float(daily_rsi.iloc[-1]),
            1,
        ),
        "週 RSI": round(
            float(week_rsi.iloc[-1]),
            1,
        ),
        "20日平均成交額": round(
            dollar_volume20,
            0,
        ),
        "量比": round(
            volume_ratio,
            2,
        ),
        "ATR波動%": round(
            atr_pct,
            2,
        ),
        "距60日低位%": round(
            float(distance_from_low),
            2,
        ),
        "52周回撤%": round(
            float(drawdown_252),
            2,
        ),
        "20日相對強弱%": (
            round(float(rs20), 2)
            if not np.isnan(rs20)
            else np.nan
        ),
    }

    return V3ScoreResult(
        eligible=eligible,
        action=action,
        total_score=total_score,
        bottom_score=bottom_score,
        confirmation_score=confirmation_score,
        quality_score=quality_score,
        risk_deduction=risk_deduction,
        price=round(price, 3),
        trigger_price=round(trigger_price, 3),
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=risk_reward,
        reasons=reasons,
        blockers=blockers,
        metrics=metrics,
    )


# ============================================================
# 市場 Regime V2：趨勢、波動、風險偏好
# ============================================================

def percentage_change(
    series: pd.Series,
    days: int,
) -> float:
    if len(series) <= days:
        return float("nan")

    return float(
        100
        * (
            series.iloc[-1]
            / series.iloc[-days - 1]
            - 1
        )
    )


def market_regime_v2(
    market: str,
) -> tuple[str, dict[str, Any], list[str]]:
    if market == "US":
        benchmark = "SPY"
        volatility_ticker = "^VIX"
        risk_ticker = "QQQ"
    else:
        benchmark = "^HSI"
        volatility_ticker = "^VHSI"
        risk_ticker = "^HSTECH"

    index_df, _ = fetch_ohlcv(
        benchmark,
        "3y",
    )

    volatility_df, _ = fetch_ohlcv(
        volatility_ticker,
        "1y",
    )

    risk_df, _ = fetch_ohlcv(
        risk_ticker,
        "1y",
    )

    if index_df is None or len(index_df) < 252:
        return (
            "中性整理",
            {},
            ["基準指數資料不足，採用保守中性設定。"],
        )

    close = index_df["close"]

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    price = float(close.iloc[-1])

    above_ma50 = bool(price > ma50.iloc[-1])
    above_ma200 = bool(price > ma200.iloc[-1])

    ma200_up = bool(
        ma200.iloc[-1] > ma200.iloc[-21]
    )

    return20 = percentage_change(close, 20)
    return60 = percentage_change(close, 60)

    drawdown = float(
        100
        * (
            price
            / close.iloc[-252:].max()
            - 1
        )
    )

    realized_volatility = float(
        close.pct_change()
        .rolling(20)
        .std()
        .iloc[-1]
        * np.sqrt(252)
        * 100
    )

    volatility_index = (
        float(volatility_df["close"].iloc[-1])
        if volatility_df is not None
        else np.nan
    )

    volatility_change_5d = (
        percentage_change(
            volatility_df["close"],
            5,
        )
        if volatility_df is not None
        else np.nan
    )

    risk_relative = np.nan

    if risk_df is not None and len(risk_df) >= 21:
        risk_relative = (
            percentage_change(
                risk_df["close"],
                20,
            )
            - return20
        )

    high_volatility = (
        (
            not np.isnan(volatility_index)
            and volatility_index >= (
                30 if market == "US" else 35
            )
        )
        or realized_volatility >= 32
    )

    rising_volatility = (
        not np.isnan(volatility_change_5d)
        and volatility_change_5d >= 15
    )

    risk_off = (
        not np.isnan(risk_relative)
        and risk_relative <= -3
    )

    if (
        high_volatility
        and (
            return20 <= -6
            or drawdown <= -12
        )
    ):
        regime = "恐慌／高波動"

    elif not above_ma200 and return20 > 3:
        regime = "熊市反彈"

    elif (
        not above_ma200
        and (return60 < -5 or risk_off)
    ):
        regime = "熊市／風險趨避"

    elif (
        above_ma50
        and above_ma200
        and ma200_up
        and return60 >= 5
        and not rising_volatility
        and not risk_off
    ):
        regime = "健康牛市"

    elif (
        above_ma50
        and above_ma200
        and (rising_volatility or risk_off)
    ):
        regime = "牛市壓力上升"

    elif (
        above_ma50
        and not above_ma200
        and return20 > 2
        and not high_volatility
    ):
        regime = "潛在復甦"

    else:
        regime = "中性整理"

    metrics = {
        "benchmark": benchmark,
        "price": price,
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "ma200_up": ma200_up,
        "return20": return20,
        "return60": return60,
        "drawdown": drawdown,
        "realized_volatility": realized_volatility,
        "volatility_ticker": volatility_ticker,
        "volatility_index": volatility_index,
        "volatility_change_5d": volatility_change_5d,
        "risk_ticker": risk_ticker,
        "risk_relative": risk_relative,
    }

    notes = [
        (
            f"趨勢：{benchmark} "
            f"{'高於' if above_ma50 else '低於'} MA50、"
            f"{'高於' if above_ma200 else '低於'} MA200，"
            f"MA200 {'向上' if ma200_up else '未向上'}。"
        ),
        (
            f"動能：20 日回報 {return20:+.1f}%，"
            f"60 日回報 {return60:+.1f}%，"
            f"距 52 周高 {drawdown:.1f}%。"
        ),
        (
            f"波動：{volatility_ticker} "
            f"{volatility_index:.1f}，"
            f"5 日變化 {volatility_change_5d:+.1f}%，"
            f"20 日年化實現波動 "
            f"{realized_volatility:.1f}%。"
        ),
        (
            f"風險偏好：{risk_ticker} 相對 "
            f"{benchmark} 的 20 日表現 "
            f"{risk_relative:+.1f}% "
            "（正值偏風險偏好；負值偏防守）。"
        ),
    ]

    return regime, metrics, notes


def regime_rules(regime: str) -> dict[str, Any]:
    mapping = {
        "健康牛市": (
            60,
            18,
            0.75,
            "可按規則試倉；優先強勢股健康回調。",
        ),
        "牛市壓力上升": (
            65,
            20,
            0.50,
            "降低倉位；只選量價確認及相對強勢候選。",
        ),
        "中性整理": (
            65,
            20,
            0.50,
            "選擇性等待突破，不提前買入。",
        ),
        "熊市反彈": (
            70,
            25,
            0.25,
            "只做短線；快進快出，不可攤平。",
        ),
        "熊市／風險趨避": (
            75,
            28,
            0.25,
            "原則上暫停新倉；只研究極少數高品質候選。",
        ),
        "恐慌／高波動": (
            75,
            28,
            0.25,
            "等待恐慌消退與結構確認；不可因超賣直接買入。",
        ),
        "潛在復甦": (
            65,
            20,
            0.50,
            "優先尋找相對強勢、突破確認的領先股。",
        ),
    }

    minimum_total, minimum_confirmation, risk_pct, advice = (
        mapping.get(
            regime,
            mapping["中性整理"],
        )
    )

    return {
        "min_total": minimum_total,
        "min_confirmation": minimum_confirmation,
        "risk_pct": risk_pct,
        "advice": advice,
    }


# ============================================================
# 掃描結果儲存及畫面工具
# ============================================================

def create_scan_run(
    market: str,
    count: int,
) -> int | None:
    client = get_supabase()

    if client is None:
        return None

    try:
        response = client.table("scan_runs").insert(
            {
                "run_type": "manual",
                "market": market,
                "universe_size": count,
                "model_version": MODEL_VERSION,
            }
        ).execute()

        return response.data[0]["id"]

    except Exception as exc:
        set_db_error("建立掃描紀錄", exc)
        return None


def finish_scan_run(
    run_id: int | None,
    success_count: int,
    failed_count: int,
    errors: list[str],
) -> None:
    client = get_supabase()

    if client is None or run_id is None:
        return

    try:
        (
            client.table("scan_runs")
            .update(
                {
                    "completed_at": (
                        datetime.utcnow().isoformat()
                    ),
                    "status": (
                        "completed"
                        if failed_count == 0
                        else "completed_with_errors"
                    ),
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "error_summary": (
                        " | ".join(errors[:10])
                        if errors
                        else None
                    ),
                }
            )
            .eq("id", run_id)
            .execute()
        )

    except Exception as exc:
        set_db_error("完成掃描紀錄", exc)


def persist_scan(
    run_id: int | None,
    ticker: str,
    result: V3ScoreResult,
    meta: dict,
    regime: str,
) -> None:
    if run_id is None:
        return

    factors = {
        "bottom_structure": result.bottom_score,
        "reversal_confirmation": (
            result.confirmation_score
        ),
        "quality": result.quality_score,
        "risk_deduction": result.risk_deduction,
        "eligible": result.eligible,
        "action": result.action,
        "metrics": result.metrics,
        "trigger_price": result.trigger_price,
        "risk_reward": result.risk_reward,
    }

    db_upsert(
        "scan_results",
        {
            "scan_run_id": run_id,
            "ticker": ticker,
            "price": result.price,
            "score": result.total_score,
            "label": result.action,
            "regime": regime,
            "stop_price": result.stop_price,
            "target_price": result.target_price,
            "factors": factors,
            "explanations": (
                result.reasons + result.blockers
            ),
            "data_source": meta["source"],
            "last_bar_date": meta["last_bar"],
        },
        "scan_run_id,ticker",
    )


def persist_signal(
    ticker: str,
    signal_date: str,
    result: V3ScoreResult,
    meta: dict,
    regime: str,
) -> None:
    if result.action != "可小量試倉":
        return

    add_membership(
        ticker=ticker,
        source="signal_high_score",
        permanent=False,
        priority=80,
        notes="V3.2 可小量試倉候選",
    )

    factors = {
        "bottom_structure": result.bottom_score,
        "reversal_confirmation": (
            result.confirmation_score
        ),
        "quality": result.quality_score,
        "risk_deduction": result.risk_deduction,
        "metrics": result.metrics,
        "trigger_price": result.trigger_price,
        "risk_reward": result.risk_reward,
    }

    db_upsert(
        "signals",
        {
            "signal_date": signal_date,
            "ticker": ticker,
            "model_version": MODEL_VERSION,
            "price": result.price,
            "score": result.total_score,
            "label": result.action,
            "regime": regime,
            "stop_price": result.stop_price,
            "target_price": result.target_price,
            "factors": factors,
            "explanations": (
                result.reasons + result.blockers
            ),
            "data_source": meta["source"],
            "last_bar_date": meta["last_bar"],
        },
        "signal_date,ticker,model_version",
    )


def make_chart(
    df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="價格",
        )
    )

    for days, color in [
        (20, "#f59e0b"),
        (50, "#22c55e"),
        (60, "#3b82f6"),
        (200, "#a855f7"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"].rolling(days).mean(),
                name=f"MA{days}",
                line={
                    "width": 1.2,
                    "color": color,
                },
            )
        )

    figure.update_layout(
        template="plotly_dark",
        height=560,
        title=f"{ticker}｜調整後日線",
        xaxis_rangeslider_visible=False,
        margin={
            "l": 10,
            "r": 10,
            "t": 45,
            "b": 10,
        },
    )

    return figure


def source_text(
    ticker: str,
    sources: dict[str, list[str]],
) -> str:
    labels = [
        SOURCE_LABELS.get(source, source)
        for source in sources.get(ticker, [])
    ]

    return " + ".join(labels) if labels else "—"


def core_scan_universe(
    market: str,
    watchlist: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    日常掃描只選高流動性核心池、Mag 7、
    手動名單、持倉及 V3 試倉候選。
    """
    if watchlist.empty:
        fallback = (
            DEFAULT_US
            if market == "US"
            else DEFAULT_HK
        )

        return (
            fallback,
            {
                ticker: ["manual"]
                for ticker in fallback
            },
        )

    grouped_sources = (
        watchlist.groupby("ticker")["source"]
        .agg(list)
        .to_dict()
    )

    if market == "US":
        preferred_sources = {
            "mag7_priority",
            "sp500_top30_turnover",
            "manual",
            "portfolio",
            "signal_high_score",
        }
    else:
        preferred_sources = {
            "hsi_top30_turnover",
            "manual",
            "portfolio",
            "signal_high_score",
        }

    selected = []

    for ticker, sources in grouped_sources.items():
        if any(
            source in preferred_sources
            for source in sources
        ):
            selected.append(ticker)

    # 保留所有永久觀察／持倉股票。
    permanent_tickers = watchlist.loc[
        watchlist["is_permanent"] == True,
        "ticker",
    ].tolist()

    selected.extend(permanent_tickers)

    selected = list(dict.fromkeys(selected))

    return selected, grouped_sources
    # ============================================================
# Streamlit 使用者介面
# ============================================================

st.title("📈 股票監察系統 Pro · V3.2 高流動性候選版")

st.caption(
    "日常美股只掃描 S&P 500 20 日成交額 Top 30、"
    "Mag 7、你的手動／持倉股票及 V3 試倉候選。"
    "分析只供研究，不構成投資建議。"
)

client = get_supabase()

if client is None:
    st.error(
        "未偵測到 Supabase Secrets。請檢查 "
        "Streamlit Cloud 的 SUPABASE_URL 與 "
        "SUPABASE_SECRET_KEY。"
    )
else:
    st.success(
        "Supabase 已連線：觀察名單、掃描及訊號"
        "會保存至雲端資料庫。"
    )


with st.sidebar:
    st.header("控制面板")

    market_label = st.radio(
        "市場",
        ["美股", "港股", "自選"],
        index=0,
    )

    market = "US" if market_label == "美股" else "HK"

    custom_input = ""

    if market_label == "自選":
        custom_input = st.text_area(
            "自選代碼（每行一個）",
            "AAPL\nNVDA\n0700.HK",
        )


if market_label == "自選":
    tickers = list(
        dict.fromkeys(
            normalize_ticker(item)
            for item in custom_input.splitlines()
            if item.strip()
        )
    )

    source_map = {
        ticker: ["manual"]
        for ticker in tickers
    }

else:
    stored_watchlist = get_watchlist(market)

    tickers, source_map = core_scan_universe(
        market,
        stored_watchlist,
    )


regime, regime_metrics, regime_notes = (
    market_regime_v2(market)
)

rules = regime_rules(regime)


headline1, headline2, headline3, headline4 = (
    st.columns(4)
)

headline1.metric(
    "市場 Regime",
    regime,
)

headline2.metric(
    "最低總分",
    rules["min_total"],
)

headline3.metric(
    "最低反轉確認",
    rules["min_confirmation"],
)

headline4.metric(
    "建議每筆帳戶風險",
    f"{rules['risk_pct']:.2f}%",
)


(
    regime_tab,
    watch_tab,
    scan_tab,
    detail_tab,
    log_tab,
    risk_tab,
) = st.tabs(
    [
        "🌍 市場分析",
        "📌 觀察名單管理",
        "📊 易讀掃描",
        "📈 個股詳情",
        "🗂️ 雲端訊號",
        "⚖️ 風控",
    ]
)


# ============================================================
# 市場 Regime 分析
# ============================================================

with regime_tab:
    st.subheader(f"市場 Regime 分析：{regime}")

    st.info(
        f"系統操作建議：{rules['advice']}"
    )

    trend_col, volatility_col, risk_col = st.columns(3)

    with trend_col:
        st.markdown("### 趨勢結構是甚麼？")

        st.write(
            "它用來看大市整體是在上升、"
            "下降還是整理。"
        )

        st.write(
            "指數高於長期平均線，"
            "通常代表長期趨勢較健康。"
        )

        st.write(
            f"• {regime_metrics.get('benchmark', '—')} "
            f"{'高於' if regime_metrics.get('above_ma50') else '低於'} "
            "MA50"
        )

        st.write(
            f"• {regime_metrics.get('benchmark', '—')} "
            f"{'高於' if regime_metrics.get('above_ma200') else '低於'} "
            "MA200"
        )

        st.write(
            f"• 20 日回報："
            f"{regime_metrics.get('return20', np.nan):+.1f}%"
        )

        st.write(
            f"• 60 日回報："
            f"{regime_metrics.get('return60', np.nan):+.1f}%"
        )

    with volatility_col:
        st.markdown("### 波動／市場壓力是甚麼？")

        st.write(
            "波動愈高，股價日內和日間起伏愈大。"
        )

        st.write(
            "同樣的交易規則應使用較小部位，"
            "並要求更嚴格確認。"
        )

        st.write(
            f"• {regime_metrics.get('volatility_ticker', 'Vol')}："
            f"{regime_metrics.get('volatility_index', np.nan):.1f}"
        )

        st.write(
            f"• 5 日變化："
            f"{regime_metrics.get('volatility_change_5d', np.nan):+.1f}%"
        )

        st.write(
            f"• 20 日年化實現波動："
            f"{regime_metrics.get('realized_volatility', np.nan):.1f}%"
        )

    with risk_col:
        st.markdown("### 風險偏好是甚麼？")

        st.write(
            "比較較高風險指數與大市的表現。"
        )

        st.write(
            "正值表示資金較願承擔風險；"
            "負值表示資金偏向防守。"
        )

        st.write(
            f"• {regime_metrics.get('risk_ticker', 'Risk')} "
            f"相對大市："
            f"{regime_metrics.get('risk_relative', np.nan):+.1f}%"
        )

        st.write(
            f"• 本 Regime 建議每筆帳戶風險："
            f"{rules['risk_pct']:.2f}%"
        )

    st.markdown("### 數據解讀")

    for note in regime_notes:
        st.write(f"• {note}")

    st.caption(
        "市場廣度，即有多少成分股高於 MA50／MA200，"
        "適合以每日背景批次掃描計算。"
        "為避免每次開頁都下載數百隻股票，"
        "目前版本未即時計算市場廣度。"
    )

    st.markdown("### 📖 V3 行動結論速查")

    st.markdown(
        """
| 行動結論 | 系統意思 | 建議做法 |
|---|---|---|
| ⛔ 不合格 | 流動性、資料、股價、止損距離或最低條件不合格 | 不因為跌得多而撈底 |
| 👀 觀察 | 有部分底部跡象，但反轉證據不足 | 追蹤並等待量價或趨勢確認 |
| ⏳ 等待突破 | 底部結構較完整，但尚未突破近 5 日確認買入價 | 不提前買；等收市突破或回踩確認 |
| ✅ 可小量試倉 | 資格通過、反轉確認及 Regime 門檻達標，止損與 2R 可計算 | 只按風險預算小倉位嘗試，嚴守止損 |
        """
    )


# ============================================================
# 觀察名單管理
# ============================================================

with watch_tab:
    st.subheader("觀察名單與高流動性核心池")

    st.markdown("### 日常掃描範圍")

    st.markdown(
        """
| 市場 | 日常主動掃描股票 |
|---|---|
| 美股 | S&P 500 20 日平均成交額 Top 30 + Mag 7 + 手動名單／持倉／V3 試倉候選 |
| 港股 | 恒指成分內 20 日平均成交額 Top 30 + 手動名單／持倉／V3 試倉候選 |

完整成分股仍會保留在資料庫，但不會令日常掃描表格混亂。
        """
    )

    sync_col1, sync_col2 = st.columns(2)

    if sync_col1.button(
        "同步美股：S&P 500 + Top 30 + Mag 7",
        type="primary",
        disabled=client is None,
    ):
        with st.spinner(
            "首次會下載 S&P 500 各成分近 6 個月資料，"
            "並計算成交額，可能需要數分鐘…"
        ):
            success, message = (
                sync_sp500_and_us_top30()
            )

        if success:
            st.success(message)
        else:
            st.error(message)

    if sync_col2.button(
        "同步港股：恒指 + 成交額 Top 30",
        type="primary",
        disabled=client is None,
    ):
        with st.spinner(
            "首次會下載恒指成分資料並計算成交額，"
            "可能需要數分鐘…"
        ):
            success, message = sync_hsi_top30()

        if success:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    st.markdown("### 手動永久加入")

    add_col1, add_col2, add_col3, add_col4 = st.columns(
        [2, 1, 1, 3]
    )

    manual_ticker = add_col1.text_input(
        "代碼",
        placeholder="例如 AAPL 或 0700.HK",
    )

    manual_source = add_col2.selectbox(
        "類別",
        ["manual", "portfolio"],
        format_func=lambda item: SOURCE_LABELS[item],
    )

    manual_priority = add_col3.slider(
        "優先級",
        1,
        100,
        100,
    )

    manual_note = add_col4.text_input(
        "備註",
        placeholder="例如：已持有／只觀察／等業績",
    )

    if st.button(
        "加入並永久保存",
        disabled=client is None,
    ):
        if not manual_ticker.strip():
            st.warning("請輸入股票代碼。")

        elif add_membership(
            ticker=manual_ticker,
            source=manual_source,
            permanent=True,
            priority=manual_priority,
            notes=manual_note,
        ):
            st.success(
                f"已永久加入 "
                f"{normalize_ticker(manual_ticker)}。"
            )
            st.rerun()

        else:
            st.error(
                st.session_state.get(
                    "db_error",
                    "寫入失敗",
                )
            )

    all_members = get_watchlist()

    if all_members.empty:
        st.info(
            "尚未有雲端觀察名單。"
            "請同步核心池或手動加入。"
        )

    else:
        display = all_members.copy()

        display["來源"] = (
            display["source"]
            .map(SOURCE_LABELS)
            .fillna(display["source"])
        )

        display["永久"] = display["is_permanent"].map(
            {
                True: "是",
                False: "否",
            }
        )

        st.dataframe(
            display[
                [
                    "ticker",
                    "market",
                    "name",
                    "sector",
                    "來源",
                    "永久",
                    "priority",
                    "notes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# V3.2 易讀掃描
# ============================================================

with scan_tab:
    st.subheader(
        f"{market_label} V3.2 易讀反轉候選掃描"
    )

    st.info(
        f"目前市場：{regime}｜"
        f"最低總分 {rules['min_total']}｜"
        f"最低反轉確認 {rules['min_confirmation']}｜"
        f"建議每筆帳戶風險 "
        f"{rules['risk_pct']:.2f}%"
    )

    st.markdown("### 📖 不知道怎樣看表格？")

    with st.expander(
        "點擊閱讀：每一欄代表甚麼、應怎樣用",
        expanded=True,
    ):
        st.markdown(
            """
| 表格項目 | 白話意思 | 怎樣用 |
|---|---|---|
| 基本資格是否通過 | 股票是否有足夠成交額、足夠歷史資料、股價和止損距離是否適合交易 | 不通過就不要因跌得多而撈底 |
| 跌勢減弱／接近底部跡象 | 超賣、接近近期低位、資金流改善等跡象 | 只代表可能接近底部，不代表已見底 |
| 由跌轉升的確認程度 | 有沒有突破近 5 日高、重上 MA20、放量收陽等 | 這是最重要的入場確認；不足時請等待 |
| 趨勢與相對大市表現 | 長期趨勢是否健康、是否跑贏 SPY／HSI | 優先選相對強勢，不只選跌得最多的股票 |
| 額外風險扣減 | 熊市、高波動、嚴重跑輸大市等風險 | 扣分愈多，越應降低倉位或不做 |
| 確認買入價 | 系統希望股價先突破的價位，通常是近 5 日高位 | 未突破前不應提早買入 |
| 止損 | 若跌穿，代表原本「可能反轉」的判斷失效 | 進場前先知道最多能承受多少風險 |
| 2R 目標 | 目標回報約為預定風險的兩倍 | 例如最多容許輸 1 元，初步目標至少賺 2 元 |
| 行動結論 | 系統此刻建議你做甚麼 | 先看這一欄，再研究細節 |
            """
        )

        st.markdown(
            """
| 行動結論 | 系統意思 | 建議做法 |
|---|---|---|
| ⛔ 不合格 | 最低條件不符合 | 不撈底 |
| 👀 觀察 | 有部分底部跡象，但不夠確認 | 加入觀察，等待更多證據 |
| ⏳ 等待突破 | 底部條件較完整，但未突破確認買入價 | 等收市突破或突破後回踩確認 |
| ✅ 可小量試倉 | 資格、反轉確認、Regime 門檻、止損與 2R 均符合 | 只按建議風險小倉位嘗試並守止損 |
            """
        )

    if not tickers:
        st.warning(
            "目前沒有可掃描股票。"
            "請先同步核心池或手動加入。"
        )

    else:
        if len(tickers) > WEB_SCAN_LIMIT:
            st.warning(
                f"核心池有 {len(tickers)} 隻；"
                f"網頁每次最多掃描 "
                f"{WEB_SCAN_LIMIT} 隻。"
            )

        max_limit = min(
            len(tickers),
            WEB_SCAN_LIMIT,
        )

        scan_count = st.number_input(
            "本次掃描數量",
            min_value=1,
            max_value=max_limit,
            value=max_limit,
            step=1,
        )

        selected_tickers = tickers[:int(scan_count)]

        st.caption(
            f"日常核心掃描池：{len(tickers)} 隻｜"
            f"本次掃描：{len(selected_tickers)} 隻"
        )

        if st.button(
            "開始 V3.2 掃描",
            type="primary",
        ):
            benchmark_ticker = (
                "SPY"
                if market == "US"
                else "^HSI"
            )

            benchmark_df, _ = fetch_ohlcv(
                benchmark_ticker,
                "3y",
            )

            benchmark_close = (
                benchmark_df["close"]
                if benchmark_df is not None
                else None
            )

            if benchmark_close is None:
                st.warning(
                    f"未能取得 {benchmark_ticker}，"
                    "品質分不會包含相對強弱。"
                )

            run_id = create_scan_run(
                (
                    market
                    if market_label != "自選"
                    else "BOTH"
                ),
                len(selected_tickers),
            )

            summary_rows = []
            detail_rows = []
            failures = []
            errors = []

            progress = st.progress(0)
            status = st.empty()

            for index, ticker in enumerate(
                selected_tickers,
                start=1,
            ):
                status.caption(
                    f"正在掃描 "
                    f"{index}/{len(selected_tickers)}："
                    f"{ticker}"
                )

                df, meta = fetch_ohlcv(ticker)

                if df is None:
                    failures.append(
                        {
                            "代碼": ticker,
                            "原因": meta["status"],
                        }
                    )

                    errors.append(
                        f"{ticker}: {meta['status']}"
                    )

                else:
                    try:
                        result = score_stock_v3(
                            df=df,
                            regime=regime,
                            benchmark_close=benchmark_close,
                            market=ticker_market(ticker),
                        )

                        # Regime V2 會提高或降低試倉門檻。
                        if (
                            result.eligible
                            and result.total_score
                            >= rules["min_total"]
                            and result.confirmation_score
                            >= rules["min_confirmation"]
                            and result.risk_reward is not None
                            and result.risk_reward >= 1.8
                        ):
                            result.action = "可小量試倉"

                        elif (
                            result.eligible
                            and result.confirmation_score
                            < rules["min_confirmation"]
                        ):
                            result.action = "等待突破"

                        elif (
                            result.eligible
                            and result.total_score >= 45
                        ):
                            result.action = "觀察"

                        else:
                            result.action = "不合格"

                        persist_scan(
                            run_id=run_id,
                            ticker=ticker,
                            result=result,
                            meta=meta,
                            regime=regime,
                        )

                        one_reason = (
                            result.reasons[0]
                            if result.reasons
                            else (
                                result.blockers[0]
                                if result.blockers
                                else "暫未有足夠確認"
                            )
                        )

                        summary_rows.append(
                            {
                                "代碼": ticker,
                                "來源": source_text(
                                    ticker,
                                    source_map,
                                ),
                                "現價": result.price,
                                "現在應怎樣做": (
                                    result.action
                                ),
                                "確認買入價（突破才考慮）": (
                                    result.trigger_price
                                ),
                                "止損（跌穿即失效）": (
                                    result.stop_price
                                ),
                                "2R初步目標": (
                                    result.target_price
                                ),
                                "建議帳戶風險": (
                                    f"{rules['risk_pct']:.2f}%"
                                ),
                                "一句原因": one_reason,
                            }
                        )

                        detail_rows.append(
                            {
                                "代碼": ticker,
                                "基本資格是否通過": (
                                    "通過"
                                    if result.eligible
                                    else "不通過"
                                ),
                                "跌勢減弱／接近底部跡象": (
                                    result.bottom_score
                                ),
                                "由跌轉升的確認程度": (
                                    result.confirmation_score
                                ),
                                "趨勢與相對大市表現": (
                                    result.quality_score
                                ),
                                "額外風險扣減": (
                                    result.risk_deduction
                                ),
                                "V3總分": (
                                    result.total_score
                                ),
                                "日RSI": (
                                    result.metrics.get("日 RSI")
                                ),
                                "週RSI": (
                                    result.metrics.get("週 RSI")
                                ),
                                "20日平均成交額": (
                                    result.metrics.get(
                                        "20日平均成交額"
                                    )
                                ),
                                "量比": (
                                    result.metrics.get("量比")
                                ),
                                "ATR波動%": (
                                    result.metrics.get(
                                        "ATR波動%"
                                    )
                                ),
                                "20日相對強弱%": (
                                    result.metrics.get(
                                        "20日相對強弱%"
                                    )
                                ),
                                "完整成立原因": (
                                    "；".join(result.reasons)
                                ),
                                "不合格／風險原因": (
                                    "；".join(result.blockers)
                                    if result.blockers
                                    else "—"
                                ),
                            }
                        )

                        if result.action == "可小量試倉":
                            persist_signal(
                                ticker=ticker,
                                signal_date=str(
                                    df.index[-1].date()
                                ),
                                result=result,
                                meta=meta,
                                regime=regime,
                            )

                    except Exception as exc:
                        message = (
                            "V3.2 評分失敗："
                            f"{type(exc).__name__}: {exc}"
                        )

                        failures.append(
                            {
                                "代碼": ticker,
                                "原因": message,
                            }
                        )

                        errors.append(
                            f"{ticker}: {message}"
                        )

                progress.progress(
                    index / len(selected_tickers)
                )

            status.empty()

            finish_scan_run(
                run_id=run_id,
                success_count=len(summary_rows),
                failed_count=len(failures),
                errors=errors,
            )

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)

                action_order = {
                    "可小量試倉": 0,
                    "等待突破": 1,
                    "觀察": 2,
                    "不合格": 3,
                }

                summary_df["_sort"] = (
                    summary_df["現在應怎樣做"]
                    .map(action_order)
                    .fillna(9)
                )

                summary_df = (
                    summary_df.sort_values(
                        ["_sort", "現價"],
                        ascending=[True, False],
                    )
                    .drop(columns=["_sort"])
                )

                trial_count = int(
                    (
                        summary_df["現在應怎樣做"]
                        == "可小量試倉"
                    ).sum()
                )

                wait_count = int(
                    (
                        summary_df["現在應怎樣做"]
                        == "等待突破"
                    ).sum()
                )

                st.success(
                    f"掃描完成：可小量試倉 "
                    f"{trial_count} 隻｜"
                    f"等待突破 {wait_count} 隻｜"
                    f"完成 {len(summary_rows)} 隻｜"
                    f"失敗 {len(failures)} 隻"
                )

                st.markdown(
                    "### 先看這張表：現在可以怎樣做"
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander(
                    "進階評分與完整原因"
                    "（不熟悉時可先不看）"
                ):
                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.download_button(
                    "下載 V3.2 掃描 CSV",
                    summary_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    "v3_2_scan_summary.csv",
                    "text/csv",
                )

            if failures:
                st.warning(
                    f"有 {len(failures)} 隻"
                    "未完成掃描。"
                )

                st.dataframe(
                    pd.DataFrame(failures),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# 個股詳情
# ============================================================

with detail_tab:
    st.subheader("V3.2 個股詳情")

    ticker = st.text_input(
        "代碼",
        tickers[0] if tickers else "AAPL",
        key="detail_ticker",
    )

    period = st.selectbox(
        "圖表範圍",
        ["1y", "2y", "3y", "5y"],
        index=1,
    )

    if st.button("載入個股詳情"):
        df, meta = fetch_ohlcv(ticker, period)

        local_market = ticker_market(ticker)

        benchmark_ticker = (
            "SPY"
            if local_market == "US"
            else "^HSI"
        )

        benchmark_df, _ = fetch_ohlcv(
            benchmark_ticker,
            period,
        )

        benchmark_close = (
            benchmark_df["close"]
            if benchmark_df is not None
            else None
        )

        if df is None:
            st.error(meta["status"])

        else:
            result = score_stock_v3(
                df=df,
                regime=regime,
                benchmark_close=benchmark_close,
                market=local_market,
            )

            if (
                result.eligible
                and result.total_score
                >= rules["min_total"]
                and result.confirmation_score
                >= rules["min_confirmation"]
                and result.risk_reward is not None
                and result.risk_reward >= 1.8
            ):
                result.action = "可小量試倉"

            elif (
                result.eligible
                and result.confirmation_score
                < rules["min_confirmation"]
            ):
                result.action = "等待突破"

            elif (
                result.eligible
                and result.total_score >= 45
            ):
                result.action = "觀察"

            else:
                result.action = "不合格"

            item1, item2, item3, item4 = st.columns(4)

            item1.metric(
                "現在應怎樣做",
                result.action,
            )

            item2.metric(
                "確認買入價（突破才考慮）",
                result.trigger_price,
            )

            item3.metric(
                "止損（跌穿即失效）",
                result.stop_price,
            )

            item4.metric(
                "2R 初步目標",
                result.target_price,
            )

            st.info(
                f"目前 Regime：{regime}｜"
                f"建議單筆帳戶風險："
                f"{rules['risk_pct']:.2f}%｜"
                f"{rules['advice']}"
            )

            st.caption(
                f"現價：{result.price}｜"
                f"R/R：{result.risk_reward}｜"
                f"資料最後日：{meta['last_bar']}"
            )

            st.plotly_chart(
                make_chart(
                    df,
                    normalize_ticker(ticker),
                ),
                use_container_width=True,
            )

            st.markdown(
                "### 這隻股票為何得到這個結論？"
            )

            if result.blockers:
                st.error(
                    "不合格／風險原因："
                    + "；".join(result.blockers)
                )

            if result.reasons:
                for reason in result.reasons:
                    st.write(f"• {reason}")
            else:
                st.write(
                    "目前未出現足夠的底部或反轉確認。"
                )

            with st.expander(
                "進階分數與術語說明"
            ):
                st.write(
                    f"• 跌勢減弱／接近底部跡象："
                    f"{result.bottom_score:.1f}/35"
                )

                st.write(
                    f"• 由跌轉升的確認程度："
                    f"{result.confirmation_score:.1f}/35"
                )

                st.write(
                    f"• 趨勢與相對大市表現："
                    f"{result.quality_score:.1f}/20"
                )

                st.write(
                    f"• 額外風險扣減："
                    f"{result.risk_deduction:.1f}"
                )

                st.dataframe(
                    pd.DataFrame([result.metrics]),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "底部跡象不等於見底；"
                    "反轉確認才是等待或試倉的核心。"
                    "確認買入價是突破價，"
                    "不是即時買入指令。"
                )


# ============================================================
# 雲端訊號紀錄
# ============================================================

with log_tab:
    st.subheader("雲端 V3.2 試倉候選紀錄")

    if client is None:
        st.error("Supabase 未連線。")

    else:
        try:
            signals = (
                client.table("signals")
                .select(
                    "signal_date,ticker,price,score,"
                    "label,regime,stop_price,target_price,"
                    "data_source,last_bar_date,created_at"
                )
                .order(
                    "signal_date",
                    desc=True,
                )
                .order(
                    "score",
                    desc=True,
                )
                .limit(1000)
                .execute()
                .data
                or []
            )

            if signals:
                signal_df = pd.DataFrame(signals)

                st.dataframe(
                    signal_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "下載訊號 CSV",
                    signal_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    "v3_2_signals.csv",
                    "text/csv",
                )

            else:
                st.info(
                    "尚未有保存的試倉候選。"
                )

        except Exception as exc:
            st.error(
                f"讀取訊號失敗："
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# 風控及部位計算
# ============================================================

with risk_tab:
    st.subheader("風險為本的部位計算")

    risk1, risk2, risk3, risk4 = st.columns(4)

    account_value = risk1.number_input(
        "帳戶淨值",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
    )

    risk_percent = risk2.slider(
        "每筆帳戶風險 (%)",
        0.10,
        2.0,
        float(rules["risk_pct"]),
        0.05,
    )

    allocation_limit = risk3.slider(
        "單一持倉最大名義比例 (%)",
        1.0,
        30.0,
        10.0,
        1.0,
    )

    risk_ticker = risk4.text_input(
        "代碼",
        tickers[0] if tickers else "AAPL",
        key="risk_ticker",
    )

    st.caption(
        f"目前 Regime 建議："
        f"每筆帳戶風險約 "
        f"{rules['risk_pct']:.2f}%。"
        "高波動或熊市時不建議自行大幅提高。"
    )

    if st.button("計算建議上限股數"):
        df, meta = fetch_ohlcv(
            risk_ticker,
            "2y",
        )

        local_market = ticker_market(risk_ticker)

        benchmark_ticker = (
            "SPY"
            if local_market == "US"
            else "^HSI"
        )

        benchmark_df, _ = fetch_ohlcv(
            benchmark_ticker,
            "2y",
        )

        benchmark_close = (
            benchmark_df["close"]
            if benchmark_df is not None
            else None
        )

        if df is None:
            st.error(meta["status"])

        else:
            result = score_stock_v3(
                df=df,
                regime=regime,
                benchmark_close=benchmark_close,
                market=local_market,
            )

            entry_price = result.trigger_price

            risk_per_share = (
                result.trigger_price
                - result.stop_price
            )

            risk_budget = (
                account_value
                * risk_percent
                / 100
            )

            shares_by_risk = (
                int(risk_budget / risk_per_share)
                if risk_per_share > 0
                else 0
            )

            shares_by_allocation = (
                int(
                    (
                        account_value
                        * allocation_limit
                        / 100
                    )
                    / entry_price
                )
                if entry_price > 0
                else 0
            )

            shares = max(
                0,
                min(
                    shares_by_risk,
                    shares_by_allocation,
                ),
            )

            output1, output2, output3, output4 = (
                st.columns(4)
            )

            output1.metric(
                "確認買入價",
                entry_price,
            )

            output2.metric(
                "止損",
                result.stop_price,
            )

            output3.metric(
                "每股可能風險",
                f"{risk_per_share:.3f}",
            )

            output4.metric(
                "建議上限股數",
                shares,
            )

            st.write(
                f"風險預算：{risk_budget:,.2f}｜"
                f"名義金額：約 "
                f"{shares * entry_price:,.2f}｜"
                f"2R 初步目標："
                f"{result.target_price}"
            )

            st.warning(
                "請按交易所整手規則向下調整，"
                "並自行考慮匯率、手續費、"
                "買賣價差、稅項和"
                "實際止損成交風險。"
            )
