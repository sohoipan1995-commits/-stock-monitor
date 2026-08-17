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
# 基本工具與 Supabase
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


def set_db_error(
    where: str,
    exc: Exception,
) -> None:
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
        set_db_error(
            f"寫入 {table}",
            exc,
        )
        return False


def upsert_instruments(
    rows: list[dict],
) -> bool:
    now = datetime.utcnow().isoformat()
    payload = []

    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        market = row.get("market") or ticker_market(
            ticker
        )

        payload.append(
            {
                "ticker": ticker,
                "market": market,
                "name": row.get("name"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
                "currency": (
                    "HKD"
                    if market == "HK"
                    else "USD"
                ),
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
            "last_confirmed_at": (
                datetime.utcnow().isoformat()
            ),
            "removed_at": None,
        },
        "ticker,source",
    )


def get_watchlist(
    market: str | None = None,
) -> pd.DataFrame:
    client = get_supabase()

    if client is None:
        return pd.DataFrame()

    try:
        memberships = (
            client.table("watchlist_memberships")
            .select(
                "ticker,source,is_permanent,"
                "priority,notes,added_at,last_confirmed_at"
            )
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        instruments = (
            client.table("instruments")
            .select(
                "ticker,market,name,sector"
            )
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        if not memberships:
            return pd.DataFrame()

        result = pd.DataFrame(memberships).merge(
            pd.DataFrame(instruments),
            on="ticker",
            how="left",
        )

        if market in {"US", "HK"}:
            result = result[
                result["market"] == market
            ]

        return result.sort_values(
            [
                "is_permanent",
                "priority",
                "ticker",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

    except Exception as exc:
        set_db_error(
            "讀取觀察名單",
            exc,
        )
        return pd.DataFrame()


# ============================================================
# Yahoo Finance 日線資料
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


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
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

            if isinstance(
                raw.columns,
                pd.MultiIndex,
            ):
                raw.columns = (
                    raw.columns.get_level_values(0)
                )

            raw.columns = [
                str(column).lower()
                for column in raw.columns
            ]

            df = raw[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ].dropna().copy()

            df.index = pd.to_datetime(
                df.index
            ).tz_localize(None)

            valid, status = validate_ohlcv(df)

            return (
                df if valid else None,
                {
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
                },
            )

        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )

            time.sleep(
                0.8 * (2 ** attempt)
            )

    return (
        None,
        {
            "ticker": ticker,
            "source": "Yahoo Finance",
            "last_bar": None,
            "rows": 0,
            "status": "；".join(errors),
        },
    )


# ============================================================
# 技術指標與 V3 評分資料結構
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
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
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


def weekly_rsi(
    close: pd.Series,
) -> pd.Series:
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
        100 * (
            stock_return
            - benchmark_return
        )
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
    # ============================================================
# V3 評分：底部跡象、反轉確認、品質及風險
# ============================================================

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
        atr14.iloc[-1] / price * 100
    )

    rs20 = relative_strength_20d(
        close,
        benchmark_close,
    )

    required = [
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
        pd.isna(item.iloc[-1])
        for item in required
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

    # 1. 底部結構：只表示跌勢可能減弱，不等於已見底。
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
            "近期跌勢較急"
        )

    if week_rsi.iloc[-1] <= 45:
        bottom_score += 6

        reasons.append(
            f"週 RSI {week_rsi.iloc[-1]:.1f} 偏低"
        )

    if distance_from_low <= 6:
        bottom_score += 8

        reasons.append(
            f"現價距 60 日低位只有 "
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
            "資金流 CMF 正在改善，"
            "賣壓可能減弱"
        )

    bottom_score = min(bottom_score, 35)

    # 2. 反轉確認：最重要的「由跌轉升」證據。
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
            f"20 日均量 {volume_ratio:.2f} 倍"
        )

    if cmf20.iloc[-1] > 0:
        confirmation_score += 5

        reasons.append(
            "CMF 為正，資金流偏向流入"
        )

    confirmation_score = min(
        confirmation_score,
        35,
    )

    # 3. 品質分：優先選長期趨勢和相對大市較好的股票。
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

    # 4. 風險扣減：市場或個股危險程度。
    risk_deduction = 0.0

    if "熊市" in regime:
        risk_deduction -= 7

        reasons.append(
            "熊市環境：逆勢反彈失敗風險較高"
        )

    if atr_pct > 8:
        risk_deduction -= 7

        reasons.append(
            f"個股波動很高（ATR {atr_pct:.1f}%）"
        )

    elif atr_pct > 5:
        risk_deduction -= 3

        reasons.append(
            f"個股波動偏高（ATR {atr_pct:.1f}%）"
        )

    if not np.isnan(rs20) and rs20 < -12:
        risk_deduction -= 4

    # 確認買入價不是立即買入價。
    # 未突破前，系統只會提示等待突破。
    entry_price = max(
        price,
        trigger_price,
    )

    structure_stop = float(
        low.iloc[-10:].min()
        - 0.30 * atr14.iloc[-1]
    )

    atr_stop = float(
        entry_price
        - 1.50 * atr14.iloc[-1]
    )

    stop_price = round(
        min(structure_stop, atr_stop),
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
# 市場 Regime：趨勢、波動及風險偏好
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


def regime_rules(
    regime: str,
) -> dict[str, Any]:
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

    total, confirmation, risk_pct, advice = (
        mapping.get(
            regime,
            mapping["中性整理"],
        )
    )

    return {
        "min_total": total,
        "min_confirmation": confirmation,
        "risk_pct": risk_pct,
        "advice": advice,
    }


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

    vol_df, _ = fetch_ohlcv(
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
            [
                "基準指數資料不足，"
                "系統採用保守中性設定。"
            ],
        )

    close = index_df["close"]

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    price = float(close.iloc[-1])

    above_ma50 = price > ma50.iloc[-1]
    above_ma200 = price > ma200.iloc[-1]
    ma200_up = ma200.iloc[-1] > ma200.iloc[-21]

    return20 = percentage_change(close, 20)
    return60 = percentage_change(close, 60)

    drawdown = float(
        100 * (
            price / close.iloc[-252:].max() - 1
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
        float(vol_df["close"].iloc[-1])
        if vol_df is not None
        else np.nan
    )

    volatility_change_5d = (
        percentage_change(
            vol_df["close"],
            5,
        )
        if vol_df is not None
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
            and volatility_index
            >= (30 if market == "US" else 35)
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
        and (
            rising_volatility
            or risk_off
        )
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
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "ma200_up": ma200_up,
        "return20": return20,
        "return60": return60,
        "drawdown": drawdown,
        "realized_volatility": realized_volatility,
        "volatility_ticker": volatility_ticker,
        "volatility_index": volatility_index,
        "volatility_change_5d": (
            volatility_change_5d
        ),
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
            f"5 日變化 "
            f"{volatility_change_5d:+.1f}%。"
        ),
        (
            f"風險偏好：{risk_ticker} 相對 "
            f"{benchmark} 的 20 日表現 "
            f"{risk_relative:+.1f}% "
            "（正值偏風險偏好；負值偏防守）。"
        ),
    ]

    return regime, metrics, notes


# ============================================================
# 核心掃描池：Mag 7 + 高成交額股票 + 個人名單
# ============================================================

def source_text(
    ticker: str,
    source_map: dict[str, list[str]],
) -> str:
    labels = [
        SOURCE_LABELS.get(source, source)
        for source in source_map.get(ticker, [])
    ]

    return " + ".join(labels) if labels else "—"


def build_core_universe(
    market: str,
    watchlist: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    這個簡化版不會每次開頁下載 S&P 500 全部資料。
    美股日常核心池使用：
    Mag 7 + 手動名單 + 持倉 + 試倉候選 + 已保存高成交額候選。
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

    source_map = (
        watchlist.groupby("ticker")["source"]
        .agg(list)
        .to_dict()
    )

    if market == "US":
        preferred = {
            "manual",
            "portfolio",
            "signal_high_score",
            "mag7_priority",
            "sp500_top30_turnover",
        }
    else:
        preferred = {
            "manual",
            "portfolio",
            "signal_high_score",
            "hsi_top30_turnover",
        }

    selected = []

    for ticker, sources in source_map.items():
        if any(
            source in preferred
            for source in sources
        ):
            selected.append(ticker)

    permanent = watchlist.loc[
        watchlist["is_permanent"] == True,
        "ticker",
    ].tolist()

    selected.extend(permanent)

    if market == "US":
        selected.extend(MAG7)

    selected = list(
        dict.fromkeys(selected)
    )

    return selected, source_map
    # ============================================================
# S&P 500 高成交額核心池同步
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def fetch_sp500_constituents() -> pd.DataFrame:
    response = requests.get(
        (
            "https://en.wikipedia.org/wiki/"
            "List_of_S%26P_500_companies"
        ),
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
        raise RuntimeError(
            "S&P 500 成分表格式已改變"
        )

    return pd.DataFrame(
        {
            "ticker": (
                table["Symbol"]
                .astype(str)
                .str.replace(".", "-", regex=False)
                .str.upper()
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


def sync_us_core_universe() -> tuple[bool, str]:
    """
    同步 S&P 500 全部成分至資料庫，
    並計算 20 日平均成交額 Top 30。
    Mag 7 會固定加入核心掃描池。
    """
    try:
        universe = fetch_sp500_constituents()

        if not upsert_instruments(
            universe.to_dict("records")
        ):
            return False, "無法寫入股票資料表"

        now = datetime.utcnow().isoformat()

        # 固定保存 Mag 7。
        mag7_rows = []

        for ticker in MAG7:
            mag7_rows.append(
                {
                    "ticker": ticker,
                    "source": "mag7_priority",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 100,
                    "notes": (
                        "Mag 7 固定優先加入日常掃描"
                    ),
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        if not db_upsert(
            "watchlist_memberships",
            mag7_rows,
            "ticker,source",
        ):
            return False, st.session_state.get(
                "db_error",
                "無法儲存 Mag 7",
            )

        # S&P 500 所有成分都保存在資料庫，
        # 但日常掃描只會選 Top 30 + Mag 7。
        constituent_rows = []

        for ticker in universe["ticker"]:
            constituent_rows.append(
                {
                    "ticker": ticker,
                    "source": "sp500_constituent",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 20,
                    "notes": (
                        "S&P 500 成分；"
                        "不一定進入日常掃描"
                    ),
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        db_upsert(
            "watchlist_memberships",
            constituent_rows,
            "ticker,source",
        )

        ranking_rows = []

        for ticker in universe["ticker"].tolist():
            df, _ = fetch_ohlcv(
                ticker,
                "6mo",
            )

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
                "未能取得足夠 S&P 500 "
                "成交額資料"
            )

        top30_rows = []

        for row in ranking.itertuples():
            top30_rows.append(
                {
                    "ticker": row.ticker,
                    "source": "sp500_top30_turnover",
                    "is_permanent": False,
                    "is_active": True,
                    "priority": 90,
                    "notes": (
                        f"20日平均成交額："
                        f"{row.turnover_20d:,.0f}"
                    ),
                    "last_confirmed_at": now,
                    "removed_at": None,
                }
            )

        if not db_upsert(
            "watchlist_memberships",
            top30_rows,
            "ticker,source",
        ):
            return False, st.session_state.get(
                "db_error",
                "無法儲存美股成交額 Top 30",
            )

        return True, (
            f"已同步 {len(universe)} 隻 S&P 500 成分股；"
            "日常掃描將使用 Mag 7 + "
            "20 日平均成交額 Top 30。"
        )

    except Exception as exc:
        return False, (
            "美股核心池同步失敗："
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# 介面
# ============================================================

st.title("📈 股票監察系統 Pro · V3.2")

st.caption(
    "美股日常掃描：Mag 7 + S&P 500 20 日平均成交額 Top 30 "
    "+ 你的手動名單／持倉／試倉候選。"
)

client = get_supabase()

if client is None:
    st.error(
        "未偵測到 Supabase Secrets。請檢查 "
        "SUPABASE_URL 與 SUPABASE_SECRET_KEY。"
    )
else:
    st.success(
        "Supabase 已連線：觀察名單、掃描和訊號"
        "將保存至雲端資料庫。"
    )


with st.sidebar:
    st.header("控制面板")

    market_label = st.radio(
        "市場",
        ["美股", "港股", "自選"],
        index=0,
    )

    market = (
        "US"
        if market_label == "美股"
        else "HK"
    )

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

    tickers, source_map = build_core_universe(
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


regime_tab, watch_tab, scan_tab, detail_tab = st.tabs(
    [
        "🌍 市場分析",
        "📌 觀察名單",
        "📊 易讀掃描",
        "📈 個股詳情",
    ]
)


# ============================================================
# 市場分析
# ============================================================

with regime_tab:
    st.subheader(f"市場 Regime 分析：{regime}")

    st.info(
        f"系統操作建議：{rules['advice']}"
    )

    trend_col, vol_col, risk_col = st.columns(3)

    with trend_col:
        st.markdown("### 趨勢結構")

        st.write(
            "這用來看大市整體是在上升、"
            "下降還是整理。"
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

    with vol_col:
        st.markdown("### 波動／市場壓力")

        st.write(
            "波動愈高，股價起伏愈大，"
            "應降低倉位並提高確認門檻。"
        )

        st.write(
            f"• {regime_metrics.get('volatility_ticker', 'Vol')}："
            f"{regime_metrics.get('volatility_index', np.nan):.1f}"
        )

        st.write(
            f"• 5 日變化："
            f"{regime_metrics.get('volatility_change_5d', np.nan):+.1f}%"
        )

    with risk_col:
        st.markdown("### 風險偏好")

        st.write(
            "較高風險指數相對大市表現較好，"
            "代表市場較願意承擔風險。"
        )

        st.write(
            f"• {regime_metrics.get('risk_ticker', 'Risk')} "
            f"相對大市："
            f"{regime_metrics.get('risk_relative', np.nan):+.1f}%"
        )

        st.write(
            f"• 建議每筆帳戶風險："
            f"{rules['risk_pct']:.2f}%"
        )

    st.markdown("### 數據解讀")

    for note in regime_notes:
        st.write(f"• {note}")

    st.markdown("### 📖 V3 行動結論速查")

    st.markdown(
        """
| 行動結論 | 系統意思 | 建議做法 |
|---|---|---|
| ⛔ 不合格 | 流動性、資料、股價、止損距離或最低條件不合格 | 不因為跌得多而撈底 |
| 👀 觀察 | 有部分底部跡象，但反轉證據不足 | 追蹤並等待更多確認 |
| ⏳ 等待突破 | 底部條件較完整，但未突破近 5 日確認買入價 | 等收市突破或回踩確認 |
| ✅ 可小量試倉 | 資格、反轉確認、Regime 門檻、止損與 2R 均符合 | 只按風險預算小倉位嘗試並守止損 |
        """
    )


# ============================================================
# 觀察名單與美股核心池同步
# ============================================================

with watch_tab:
    st.subheader("觀察名單與高流動性核心池")

    st.markdown(
        """
| 市場 | 日常主動掃描股票 |
|---|---|
| 美股 | Mag 7 + S&P 500 20 日平均成交額 Top 30 + 手動名單／持倉／試倉候選 |
| 港股 | 手動名單／持倉／試倉候選；可自行在自選頁輸入港股代碼 |
        """
    )

    if st.button(
        "同步美股：S&P 500 + Top 30 + Mag 7",
        type="primary",
        disabled=client is None,
    ):
        with st.spinner(
            "首次需要下載 S&P 500 成分股近 6 個月資料"
            "並計算成交額，可能需要數分鐘…"
        ):
            success, message = sync_us_core_universe()

        if success:
            st.success(message)
        else:
            st.error(message)

    st.divider()

    st.markdown("### 手動永久加入")

    add1, add2, add3, add4 = st.columns(
        [2, 1, 1, 3]
    )

    manual_ticker = add1.text_input(
        "代碼",
        placeholder="例如 AAPL 或 0700.HK",
    )

    manual_source = add2.selectbox(
        "類別",
        ["manual", "portfolio"],
        format_func=lambda item: SOURCE_LABELS[item],
    )

    manual_priority = add3.slider(
        "優先級",
        1,
        100,
        100,
    )

    manual_note = add4.text_input(
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

    if not all_members.empty:
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
# 易讀掃描
# ============================================================

with scan_tab:
    st.subheader(
        f"{market_label} V3.2 易讀反轉候選掃描"
    )

    st.info(
        f"目前市場：{regime}｜"
        f"最低總分 {rules['min_total']}｜"
        f"最低反轉確認 {rules['min_confirmation']}｜"
        f"建議帳戶風險 "
        f"{rules['risk_pct']:.2f}%"
    )

    with st.expander(
        "📖 不知道怎樣看表格？",
        expanded=True,
    ):
        st.markdown(
            """
| 表格項目 | 白話意思 |
|---|---|
| 現在應怎樣做 | 先看這欄：不合格、觀察、等待突破或可小量試倉 |
| 確認買入價 | 近 5 日高位；未突破前不應提前買入 |
| 止損 | 跌穿代表原本反轉判斷可能錯誤 |
| 2R初步目標 | 預定風險的兩倍。例如最多輸 1 元，目標至少賺 2 元 |
| 一句原因 | 系統最主要的成立或不合格原因 |
            """
        )

    if not tickers:
        st.warning(
            "目前沒有可掃描股票。"
            "請先同步美股核心池或手動加入。"
        )

    else:
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

            summary_rows = []
            detail_rows = []

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
                    continue

                result = score_stock_v3(
                    df=df,
                    regime=regime,
                    benchmark_close=benchmark_close,
                    market=ticker_market(ticker),
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
                        "現在應怎樣做": result.action,
                        "確認買入價": result.trigger_price,
                        "止損": result.stop_price,
                        "2R初步目標": result.target_price,
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
                        "底部結構": result.bottom_score,
                        "反轉確認": (
                            result.confirmation_score
                        ),
                        "品質": result.quality_score,
                        "風險扣減": (
                            result.risk_deduction
                        ),
                        "V3總分": result.total_score,
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
                        "完整成立原因": (
                            "；".join(result.reasons)
                        ),
                        "不合格原因": (
                            "；".join(result.blockers)
                            if result.blockers
                            else "—"
                        ),
                    }
                )

                progress.progress(
                    index / len(selected_tickers)
                )

            status.empty()

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
                ):
                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.download_button(
                    "下載掃描 CSV",
                    summary_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    "v3_2_scan.csv",
                    "text/csv",
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
        df, meta = fetch_ohlcv(
            ticker,
            period,
        )

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

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "現在應怎樣做",
                result.action,
            )

            col2.metric(
                "確認買入價",
                result.trigger_price,
            )

            col3.metric(
                "止損",
                result.stop_price,
            )

            col4.metric(
                "2R 初步目標",
                result.target_price,
            )

            st.info(
                f"目前 Regime：{regime}｜"
                f"建議帳戶風險："
                f"{rules['risk_pct']:.2f}%｜"
                f"{rules['advice']}"
            )

            st.caption(
                f"現價：{result.price}｜"
                f"R/R：{result.risk_reward}｜"
                f"資料最後日：{meta['last_bar']}"
            )

            fig = go.Figure()

            fig.add_trace(
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
                (200, "#a855f7"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=(
                            df["close"]
                            .rolling(days)
                            .mean()
                        ),
                        name=f"MA{days}",
                        line={
                            "width": 1.2,
                            "color": color,
                        },
                    )
                )

            fig.update_layout(
                template="plotly_dark",
                height=560,
                title=(
                    f"{normalize_ticker(ticker)}"
                    "｜調整後日線"
                ),
                xaxis_rangeslider_visible=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.markdown(
                "### 為何得到這個結論？"
            )

            if result.blockers:
                st.error(
                    "不合格／風險原因："
                    + "；".join(result.blockers)
                )

            if result.reasons:
                for reason in result.reasons:
                    st.write(f"• {reason}")

            with st.expander(
                "進階分數與術語"
            ):
                st.write(
                    f"• 底部結構："
                    f"{result.bottom_score:.1f}/35"
                )

                st.write(
                    f"• 反轉確認："
                    f"{result.confirmation_score:.1f}/35"
                )

                st.write(
                    f"• 品質："
                    f"{result.quality_score:.1f}/20"
                )

                st.write(
                    f"• 風險扣減："
                    f"{result.risk_deduction:.1f}"
                )

                st.dataframe(
                    pd.DataFrame([result.metrics]),
                    use_container_width=True,
                    hide_index=True,
                )
