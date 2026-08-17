from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift()
    values = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((2 * df["close"] - df["high"] - df["low"]) / spread).fillna(0)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def weekly_rsi(close: pd.Series) -> pd.Series:
    weekly = close.resample("W-FRI").last().dropna()
    return rsi(weekly).reindex(close.index, method="ffill")


def relative_strength_20d(close: pd.Series, benchmark_close: pd.Series | None) -> float:
    if benchmark_close is None or len(close) < 21 or len(benchmark_close) < 21:
        return np.nan
    stock_return = close.iloc[-1] / close.iloc[-21] - 1
    aligned = benchmark_close.reindex(close.index, method="ffill").dropna()
    if len(aligned) < 21:
        return np.nan
    benchmark_return = aligned.iloc[-1] / aligned.iloc[-21] - 1
    return float(100 * (stock_return - benchmark_return))


def score_stock_v3(
    df: pd.DataFrame,
    regime: str,
    benchmark_close: pd.Series | None = None,
    market: str = "US",
    min_us_dollar_volume: float = 20_000_000,
    min_hk_dollar_volume: float = 50_000_000,
) -> V3ScoreResult:
    """Rule-based research score.

    It intentionally does NOT return a 'bottom probability'. That must be statistically
    calibrated from a separate walk-forward backtest before being shown to users.
    """
    if df is None or len(df) < 252:
        return V3ScoreResult(False, "不合格", 0, 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, None, [], ["歷史資料少於 252 個交易日"], {})

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    price = float(close.iloc[-1])
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()
    rsi_d = rsi(close)
    rsi_w = weekly_rsi(close)
    atr14 = atr(df)
    cmf20 = cmf(df)
    vol20 = volume.rolling(20).mean()
    dollar_volume20 = float((close.iloc[-20:] * volume.iloc[-20:]).mean())
    volume_ratio = float(volume.iloc[-1] / vol20.iloc[-1]) if vol20.iloc[-1] > 0 else np.nan
    atr_pct = float(100 * atr14.iloc[-1] / price)
    rs20 = relative_strength_20d(close, benchmark_close)

    required = [ma20, ma60, ma200, rsi_d, rsi_w, atr14, cmf20, vol20]
    if any(pd.isna(value.iloc[-1]) for value in required):
        return V3ScoreResult(False, "不合格", 0, 0, 0, 0, 0, price, np.nan, np.nan, np.nan, None, [], ["指標暖機資料不足"], {})

    blockers: list[str] = []
    reasons: list[str] = []
    min_dollar_volume = min_us_dollar_volume if market == "US" else min_hk_dollar_volume
    if market == "US" and price < 5:
        blockers.append("美股股價低於 5 美元")
    if dollar_volume20 < min_dollar_volume:
        blockers.append(f"20 日平均成交額不足 {min_dollar_volume:,.0f}")

    # Bottom structure: 0-35. It asks whether selling pressure may be exhausted.
    bottom = 0.0
    low60 = float(low.iloc[-60:].min())
    distance_from_low = 100 * (price / low60 - 1) if low60 > 0 else np.nan
    drawdown_252 = 100 * (price / float(high.iloc[-252:].max()) - 1)
    if rsi_d.iloc[-1] <= 32:
        bottom += 8
        reasons.append(f"日 RSI {rsi_d.iloc[-1]:.1f} 位於超賣／偏低區")
    if rsi_w.iloc[-1] <= 45:
        bottom += 6
        reasons.append(f"真週 RSI {rsi_w.iloc[-1]:.1f} 偏低")
    if distance_from_low <= 6:
        bottom += 8
        reasons.append(f"現價距 60 日低位 {distance_from_low:.1f}%")
    if -45 <= drawdown_252 <= -12:
        bottom += 5
        reasons.append(f"距 52 周高回撤 {drawdown_252:.1f}%")
    if cmf20.iloc[-1] > cmf20.iloc[-6] and cmf20.iloc[-1] > -0.08:
        bottom += 8
        reasons.append("CMF 改善，拋售壓力可能減弱")
    bottom = min(bottom, 35)

    # Confirmation: 0-35. This is the key gate that prevents buying merely because a stock is oversold.
    confirmation = 0.0
    trigger = float(high.iloc[-6:-1].max())
    if price > trigger:
        confirmation += 12
        reasons.append(f"收市突破近 5 日確認價 {trigger:.2f}")
    if price > ma20.iloc[-1]:
        confirmation += 8
        reasons.append("收市重回 MA20 之上")
    if ma20.iloc[-1] > ma20.iloc[-6]:
        confirmation += 5
        reasons.append("MA20 開始上彎")
    if volume_ratio >= 1.3 and close.iloc[-1] > df["open"].iloc[-1]:
        confirmation += 5
        reasons.append(f"放量收陽，量比 {volume_ratio:.2f}")
    if cmf20.iloc[-1] > 0:
        confirmation += 5
        reasons.append(f"CMF {cmf20.iloc[-1]:.2f} 回到正值")
    confirmation = min(confirmation, 35)

    # Quality: 0-20. A stronger stock and a healthier long trend receive preference.
    quality = 0.0
    if price > ma60.iloc[-1]:
        quality += 6
    if ma200.iloc[-1] > ma200.iloc[-21]:
        quality += 6
        reasons.append("MA200 長期趨勢上升")
    if not np.isnan(rs20) and rs20 >= 0:
        quality += 8
        reasons.append(f"20 日相對大市強弱 {rs20:+.1f}%")
    elif not np.isnan(rs20) and rs20 < -8:
        reasons.append(f"20 日明顯跑輸大市 {rs20:.1f}%")
    quality = min(quality, 20)

    # Risk deduction: 0 to -20. Do not let a good technical setup hide a hostile regime.
    risk_deduction = 0.0
    if regime.startswith("bear"):
        risk_deduction -= 7
        reasons.append("熊市 regime：降低逆勢撈底評級")
    if atr_pct > 8:
        risk_deduction -= 7
        reasons.append(f"ATR 波動過高 {atr_pct:.1f}%")
    elif atr_pct > 5:
        risk_deduction -= 3
        reasons.append(f"ATR 波動偏高 {atr_pct:.1f}%")
    if not np.isnan(rs20) and rs20 < -12:
        risk_deduction -= 4

    # Entry is the trigger price, not necessarily the current price.
    entry = max(price, trigger)
    structure_stop = float(low.iloc[-10:].min() - 0.30 * atr14.iloc[-1])
    volatility_stop = float(entry - 1.50 * atr14.iloc[-1])
    stop = min(structure_stop, volatility_stop)
    stop = round(stop, 3)
    risk_per_share = entry - stop
    target = round(entry + 2.0 * risk_per_share, 3) if risk_per_share > 0 else np.nan
    risk_reward = round((target - entry) / risk_per_share, 2) if risk_per_share > 0 else None

    if risk_per_share <= 0:
        blockers.append("無法建立有效止損")
    if risk_per_share / entry > 0.10:
        blockers.append("止損距離超過入場價 10%，風險過寬")

    total = round(max(0.0, bottom + confirmation + quality + risk_deduction), 1)
    eligible = len(blockers) == 0

    if not eligible:
        action = "不合格"
    elif confirmation < 12:
        action = "等待突破"
    elif total >= 65 and confirmation >= 20 and risk_reward is not None and risk_reward >= 1.8:
        action = "可小量試倉"
    elif total >= 50:
        action = "觀察"
    else:
        action = "不合格"

    metrics = {
        "日RSI": round(float(rsi_d.iloc[-1]), 1),
        "週RSI": round(float(rsi_w.iloc[-1]), 1),
        "20日平均成交額": round(dollar_volume20, 0),
        "量比": round(volume_ratio, 2),
        "ATR%": round(atr_pct, 2),
        "距60日低位%": round(float(distance_from_low), 2),
        "52周回撤%": round(float(drawdown_252), 2),
        "20日相對強弱%": round(float(rs20), 2) if not np.isnan(rs20) else np.nan,
    }
    return V3ScoreResult(eligible, action, total, bottom, confirmation, quality, risk_deduction, round(price, 3), round(trigger, 3), stop, target, risk_reward, reasons, blockers, metrics)
