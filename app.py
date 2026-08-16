import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="股票監察系統 Pro · 改善版", page_icon="📈", layout="wide")

HK_WATCHLIST = [
    "0700.HK", "0005.HK", "0939.HK", "1398.HK", "3988.HK", "0388.HK", "0016.HK",
    "0883.HK", "2318.HK", "1299.HK", "9988.HK", "0175.HK", "2628.HK", "3690.HK",
    "9618.HK", "0981.HK", "9999.HK", "1211.HK", "0762.HK", "1810.HK",
]
US_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "ASML",
    "JPM", "BRK-B", "COST", "UNH", "XOM", "SPY", "QQQ", "IWM",
]
MACRO = {"美股波動率 VIX": "^VIX", "美股基準 SPY": "SPY", "港股基準 HSI": "^HSI", "港股波動率 VHSI": "^VHSI"}
DB_PATH = Path("stock_monitor.sqlite3")
MODEL_VERSION = "2.0.0"


# -------------------- Data, persistence and validation --------------------
def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(" ", "")


def validate_ohlcv(df: pd.DataFrame | None) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "沒有取得價格資料"
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        return False, f"缺少欄位：{', '.join(sorted(missing))}"
    if len(df) < 80:
        return False, "歷史資料少於 80 個交易日"
    if df[list(required)].isna().any().any():
        return False, "OHLCV 有遺漏值"
    if (df["close"] <= 0).any() or (df["volume"] < 0).any():
        return False, "價格或成交量資料不合理"
    return True, "OK"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "3y") -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Download adjusted daily bars with retries and an explicit quality/status record."""
    ticker = normalize_ticker(ticker)
    errors = []
    for attempt in range(3):
        try:
            raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                              progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).lower() for c in raw.columns]
            df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            valid, message = validate_ohlcv(df)
            meta = {
                "ticker": ticker, "source": "Yahoo Finance", "adjusted": True,
                "last_bar": str(df.index[-1].date()) if not df.empty else None,
                "rows": len(df), "status": message,
            }
            return (df if valid else None), meta
        except Exception as exc:
            errors.append(f"第 {attempt + 1} 次：{type(exc).__name__}: {exc}")
            time.sleep(0.8 * (2 ** attempt))
    return None, {"ticker": ticker, "source": "Yahoo Finance", "status": "；".join(errors), "rows": 0}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                signal_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model_version TEXT NOT NULL,
                price REAL NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                regime TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (signal_date, ticker, model_version)
            )
        """)


def save_signal(signal: dict[str, Any]) -> None:
    import json
    init_db()
    payload = json.dumps(signal, ensure_ascii=False, default=str)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO signal_log
            (signal_date,ticker,model_version,price,score,label,regime,payload_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            signal["signal_date"], signal["ticker"], MODEL_VERSION, signal["price"],
            signal["score"], signal["label"], signal["regime"], payload,
            datetime.now().isoformat(timespec="seconds"),
        ))


def load_signals() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM signal_log ORDER BY signal_date DESC, score DESC", conn)


# -------------------- Indicators --------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((2 * df["close"] - df["high"] - df["low"]) / spread).fillna(0)
    return (multiplier * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """20-day volume-weighted average price; this is not mislabeled as intraday VWAP."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def weekly_rsi(daily_close: pd.Series, period: int = 14) -> pd.Series:
    """True weekly RSI: resample daily data before calculating the indicator."""
    weekly_close = daily_close.resample("W-FRI").last().dropna()
    return rsi(weekly_close, period).reindex(daily_close.index, method="ffill")


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def local_pivots(series: pd.Series, left: int = 3, right: int = 3, kind: str = "low") -> pd.Series:
    window = left + right + 1
    if kind == "low":
        candidate = series.rolling(window, center=True).min()
        return series[(series == candidate)].dropna()
    candidate = series.rolling(window, center=True).max()
    return series[(series == candidate)].dropna()


def bullish_divergence(df: pd.DataFrame) -> bool:
    """Confirmed pivot-based divergence; no use of future bar for current signal."""
    hist = macd(df["close"])[2]
    lows = local_pivots(df["low"].iloc[:-3], kind="low")  # omit final 3 bars to confirm pivots
    if len(lows) < 2:
        return False
    first_dt, second_dt = lows.index[-2], lows.index[-1]
    return bool(lows.loc[second_dt] < lows.loc[first_dt] and hist.loc[second_dt] > hist.loc[first_dt])


# -------------------- Regime and scoring --------------------
def market_regime(market: str) -> tuple[str, dict[str, float]]:
    benchmark = "SPY" if market == "美股" else "^HSI"
    vol_ticker = "^VIX" if market == "美股" else "^VHSI"
    px, _ = fetch_ohlcv(benchmark, "1y")
    vol, _ = fetch_ohlcv(vol_ticker, "1y")
    if px is None or len(px) < 70:
        return "unknown", {"return_60d": np.nan, "volatility": np.nan, "vol_index": np.nan}
    ret60 = 100 * (px["close"].iloc[-1] / px["close"].iloc[-61] - 1)
    annual_vol = 100 * px["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_index = float(vol["close"].iloc[-1]) if vol is not None else np.nan
    high_vol = bool(vol_index >= (25 if market == "美股" else 30))
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


def label_score(score: float) -> str:
    if score >= 75:
        return "強烈關注"
    if score >= 60:
        return "值得關注"
    if score >= 45:
        return "觀察中"
    return "未觸發"


def score_stock(df: pd.DataFrame, regime: str, regime_stats: dict[str, float]) -> ScoreResult:
    """Score independent factor groups: reversal, trend, volume/flow, and risk.
    The score is a screening rank, not a buy instruction.
    """
    close, volume = df["close"], df["volume"]
    last = df.index[-1]
    price = float(close.iloc[-1])
    rsi_d = rsi(close)
    rsi_w = weekly_rsi(close)
    ma20, ma60, ma200 = close.rolling(20).mean(), close.rolling(60).mean(), close.rolling(200).mean()
    macd_line, macd_sig, hist = macd(close)
    atr14 = atr(df)
    vol_z = zscore(volume)
    cmf20, vwap20 = cmf(df), rolling_vwap(df)

    if any(pd.isna(x.iloc[-1]) for x in [rsi_d, rsi_w, ma60, ma200, atr14, vol_z, cmf20, vwap20]):
        raise ValueError("指標暖機資料不足")

    factors: dict[str, float] = {"reversal": 0, "trend": 0, "flow": 0, "risk": 0}
    notes: list[str] = []

    # A. Reversal: one coherent group, capped at 30 points.
    if rsi_d.iloc[-1] <= 30:
        factors["reversal"] += 15; notes.append(f"日 RSI {rsi_d.iloc[-1]:.1f} 處於超賣")
    elif rsi_d.iloc[-1] <= 38:
        factors["reversal"] += 8; notes.append(f"日 RSI {rsi_d.iloc[-1]:.1f} 偏低")
    if rsi_w.iloc[-1] <= 42:
        factors["reversal"] += 8; notes.append(f"真正週 RSI {rsi_w.iloc[-1]:.1f} 偏低")
    if macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-1] < 0:
        factors["reversal"] += 7; notes.append("MACD 在零軸下方金叉")
    if bullish_divergence(df):
        factors["reversal"] += 8; notes.append("已確認 pivot MACD 底背離")
    factors["reversal"] = min(factors["reversal"], 30)

    # B. Trend context: bottoms against a falling long trend get less, not more, confidence.
    if price > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-6]:
        factors["trend"] += 10; notes.append("價格站上且 MA20 上彎")
    if price > ma60.iloc[-1]:
        factors["trend"] += 8; notes.append("價格位於 MA60 之上")
    if ma200.iloc[-1] > ma200.iloc[-21]:
        factors["trend"] += 7; notes.append("MA200 趨勢向上")
    elif price < ma200.iloc[-1]:
        factors["trend"] -= 5; notes.append("仍低於或受制於長期趨勢")
    factors["trend"] = float(np.clip(factors["trend"], 0, 25))

    # C. Flow/volume: distinguish accumulation from merely large volume.
    if cmf20.iloc[-1] > 0.08:
        factors["flow"] += 10; notes.append(f"CMF {cmf20.iloc[-1]:.2f} 顯示資金流入")
    elif cmf20.iloc[-1] < -0.08:
        factors["flow"] -= 5; notes.append(f"CMF {cmf20.iloc[-1]:.2f} 顯示資金流出")
    if vol_z.iloc[-1] >= 1.5 and price > vwap20.iloc[-1] and close.iloc[-1] > df["open"].iloc[-1]:
        factors["flow"] += 10; notes.append(f"放量收陽（成交量 Z={vol_z.iloc[-1]:.1f}）")
    elif vol_z.iloc[-1] >= 2 and close.iloc[-1] < df["open"].iloc[-1]:
        factors["flow"] -= 5; notes.append("高成交量收陰，需防止拋售壓力")
    factors["flow"] = float(np.clip(factors["flow"], 0, 20))

    # D. Risk and regime: reward acceptable volatility, but never raise a falling-market signal mechanically.
    atr_pct = float(atr14.iloc[-1] / price * 100)
    if atr_pct <= 4:
        factors["risk"] += 10; notes.append(f"ATR 波動可控（{atr_pct:.1f}%）")
    elif atr_pct <= 7:
        factors["risk"] += 5; notes.append(f"ATR 波動偏高（{atr_pct:.1f}%）")
    else:
        notes.append(f"ATR 波動高（{atr_pct:.1f}%），應縮小倉位")
    if regime.startswith("bear"):
        factors["risk"] -= 5; notes.append("熊市 regime：不自動放大撈底訊號")
    elif regime.startswith("bull"):
        factors["risk"] += 3
    factors["risk"] = float(np.clip(factors["risk"], 0, 15))

    total = round(sum(factors.values()), 1)
    stop = round(float(min(df["low"].iloc[-10:].min(), price - 1.5 * atr14.iloc[-1])), 3)
    # Stop must remain below current price; target uses a minimum 1.8R rather than an arbitrary +20%.
    stop = min(stop, round(price - 0.5 * atr14.iloc[-1], 3))
    target = round(price + 1.8 * (price - stop), 3)
    return ScoreResult(total, label_score(total), regime, round(price, 3), stop, target, factors, notes)


# -------------------- Backtesting and plotting --------------------
def backtest_ticker(df: pd.DataFrame, regime: str, regime_stats: dict[str, float], min_score: float,
                    hold_days: int, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    """Walk-forward daily simulation. Signal at close t; entry at next open t+1.
    Every decision uses only bars up to signal date t. Exit is first of stop, target or time exit.
    """
    trades: list[dict[str, Any]] = []
    cooldown_until = -1
    costs = 2 * (fee_bps + slippage_bps) / 10000
    for i in range(220, len(df) - hold_days - 1):
        if i <= cooldown_until:
            continue
        history = df.iloc[: i + 1]
        try:
            result = score_stock(history, regime, regime_stats)
        except (ValueError, IndexError):
            continue
        if result.score < min_score or result.stop >= result.price:
            continue
        entry_i = i + 1
        entry = float(df["open"].iloc[entry_i])
        stop = result.stop
        per_share_risk = entry - stop
        if per_share_risk <= 0:
            continue
        exit_i, exit_price, reason = entry_i + hold_days, float(df["close"].iloc[entry_i + hold_days]), "時間出場"
        for j in range(entry_i, min(entry_i + hold_days + 1, len(df))):
            # Conservative daily-bar assumption: if stop and target are both touched, stop is filled first.
            if float(df["low"].iloc[j]) <= stop:
                exit_i, exit_price, reason = j, stop, "止損"
                break
            if float(df["high"].iloc[j]) >= result.target:
                exit_i, exit_price, reason = j, result.target, "目標"
                break
        gross = exit_price / entry - 1
        net = gross - costs
        trades.append({
            "訊號日": df.index[i].date(), "入場日": df.index[entry_i].date(), "出場日": df.index[exit_i].date(),
            "入場": round(entry, 3), "止損": stop, "目標": result.target, "出場": round(exit_price, 3),
            "原因": reason, "評分": result.score, "淨回報%": round(100 * net, 2),
        })
        cooldown_until = exit_i
    return pd.DataFrame(trades)


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    ma20, ma60, ma200 = df["close"].rolling(20).mean(), df["close"].rolling(60).mean(), df["close"].rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="價格"))
    for values, name, color in [(ma20, "MA20", "#f59e0b"), (ma60, "MA60", "#3b82f6"), (ma200, "MA200", "#a855f7")]:
        fig.add_trace(go.Scatter(x=df.index, y=values, name=name, line={"width": 1.3, "color": color}))
    fig.update_layout(height=580, template="plotly_dark", title=f"{ticker}｜調整後日線", xaxis_rangeslider_visible=False,
                      margin={"l": 10, "r": 10, "t": 45, "b": 10})
    return fig


# -------------------- UI --------------------
st.title("📈 股票監察系統 Pro · 改善版")
st.caption("模型版本 2.0.0｜資料預設使用 Yahoo Finance 調整後日線｜評分是篩選排序，不構成買賣建議。")

with st.sidebar:
    st.header("控制面板")
    market_label = st.radio("市場", ["美股", "港股", "自選"], index=0)
    market_key = "美股" if market_label == "美股" else "港股"
    custom = st.text_area("自選代碼（每行一個）", "AAPL\nNVDA\n0700.HK") if market_label == "自選" else ""
    min_score = st.slider("最低篩選分數", 0, 100, 60)
    st.divider()
    st.caption("資料品質原則")
    st.caption("• 顯示最後 bar 和資料來源\n• 下載失敗不會靜默當作無訊號\n• 週線 RSI 由週收市價計算\n• 20 日 VWAP 僅作滾動均價，不冒充日內 VWAP")

if market_label == "美股":
    tickers = US_WATCHLIST
elif market_label == "港股":
    tickers = HK_WATCHLIST
else:
    tickers = list(dict.fromkeys(normalize_ticker(x) for x in custom.splitlines() if x.strip()))

regime, regime_stats = market_regime(market_key)
state_cn = {
    "bull_low_vol": "牛市／低波動", "bull_high_vol": "牛市／高波動", "bear_low_vol": "熊市／低波動",
    "bear_high_vol": "熊市／高波動", "neutral": "中性", "neutral_high_vol": "中性／高波動", "unknown": "未能判定",
}.get(regime, regime)

c1, c2, c3, c4 = st.columns(4)
c1.metric("市場 regime", state_cn)
c2.metric("60 日基準回報", "—" if pd.isna(regime_stats["return_60d"]) else f"{regime_stats['return_60d']:.1f}%")
c3.metric("20 日年化波動", "—" if pd.isna(regime_stats["volatility"]) else f"{regime_stats['volatility']:.1f}%")
c4.metric("波動率指數", "—" if pd.isna(regime_stats["vol_index"]) else f"{regime_stats['vol_index']:.1f}")

scan_tab, detail_tab, backtest_tab, log_tab, risk_tab = st.tabs(["📊 掃描", "📈 個股詳情", "🧪 Walk-forward 回測", "🗂️ 訊號紀錄", "⚖️ 風控"])

with scan_tab:
    st.subheader(f"{market_label} 觀察名單掃描")
    if st.button("開始掃描", type="primary"):
        rows, failures = [], []
        progress = st.progress(0)
        for n, ticker in enumerate(tickers, 1):
            df, meta = fetch_ohlcv(ticker)
            if df is None:
                failures.append({"代碼": ticker, "原因": meta["status"]})
            else:
                try:
                    result = score_stock(df, regime, regime_stats)
                    row = {
                        "代碼": ticker, "現價": result.price, "總分": result.score, "標籤": result.label,
                        "止損": result.stop, "目標": result.target, "Risk/Reward": round((result.target-result.price)/(result.price-result.stop), 2),
                        "資料最後日": meta["last_bar"], "因子": " / ".join(f"{k}:{v:.0f}" for k, v in result.factors.items()),
                        "說明": "；".join(result.explanations),
                    }
                    rows.append(row)
                    if result.score >= min_score:
                        save_signal({
                            "signal_date": str(df.index[-1].date()), "ticker": ticker, "price": result.price,
                            "score": result.score, "label": result.label, "regime": regime, "factors": result.factors,
                            "explanations": result.explanations, "stop": result.stop, "target": result.target,
                            "data_source": meta["source"], "last_bar": meta["last_bar"],
                        })
                except Exception as exc:
                    failures.append({"代碼": ticker, "原因": f"指標計算失敗：{type(exc).__name__}: {exc}"})
            progress.progress(n / max(len(tickers), 1))
        if rows:
            out = pd.DataFrame(rows).sort_values(["總分", "代碼"], ascending=[False, True])
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("下載掃描 CSV", out.to_csv(index=False).encode("utf-8-sig"), "scan_results.csv", "text/csv")
            st.info("分數達門檻的訊號已以「日期 + 代碼 + 模型版本」去重後寫入 SQLite。")
        if failures:
            st.warning("部分代碼未完成掃描；請查看原因，而非把它們視為無訊號。")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

with detail_tab:
    ticker = st.text_input("代碼", tickers[0] if tickers else "AAPL", key="detail_ticker")
    period = st.selectbox("圖表資料範圍", ["1y", "2y", "3y", "5y"], index=1)
    if st.button("載入個股", key="load_detail"):
        df, meta = fetch_ohlcv(ticker, period)
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime, regime_stats)
            a, b, c, d = st.columns(4)
            a.metric("總分", f"{result.score:.1f}/90", result.label)
            b.metric("現價", result.price)
            c.metric("結構止損", result.stop)
            d.metric("最低 1.8R 目標", result.target)
            st.caption(f"來源：{meta['source']}｜調整後價格：{meta['adjusted']}｜最後 bar：{meta['last_bar']}｜資料行數：{meta['rows']}")
            st.plotly_chart(candlestick_chart(df, normalize_ticker(ticker)), use_container_width=True)
            st.markdown("### 分數拆解")
            st.dataframe(pd.DataFrame([result.factors]), use_container_width=True, hide_index=True)
            for note in result.explanations:
                st.write(f"• {note}")
            with st.expander("指標定義與限制"):
                st.markdown("""
                - 日 RSI 使用 14 個日線 bar；週 RSI 先以每週五收市價重採樣，再計算 14 週 RSI。
                - 20 日 VWAP 是成交量加權的滾動日線均價，不是日內 session VWAP。
                - MACD 背離只使用已確認的局部低點；最新 3 個 bar 不作 pivot，避免以未完成資料確認形態。
                - 分數最高為 90，刻意保留不確定性；高分代表較符合本模型的條件，並不代表勝率或預期回報得到保證。
                """)

with backtest_tab:
    st.subheader("Walk-forward 回測")
    st.caption("訊號於 t 日收市後產生，於 t+1 開市入場；逐日檢查止損、目標或時間出場。回測包含雙邊費用與滑點，且不使用未來資料作當天訊號。")
    b1, b2, b3, b4 = st.columns(4)
    bt_ticker = b1.text_input("回測代碼", tickers[0] if tickers else "AAPL", key="bt_ticker")
    hold_days = b2.selectbox("最長持有交易日", [5, 10, 15, 20, 30], index=2)
    bt_threshold = b3.slider("訊號門檻", 35, 85, 60, key="bt_threshold")
    fee_bps = b4.number_input("單邊成本 + 滑點（bps）", 0.0, 100.0, 10.0, 1.0)
    if st.button("執行回測", key="run_backtest"):
        df, meta = fetch_ohlcv(bt_ticker, "5y")
        if df is None:
            st.error(meta["status"])
        else:
            trades = backtest_ticker(df, regime, regime_stats, bt_threshold, hold_days, fee_bps / 2, fee_bps / 2)
            if trades.empty:
                st.warning("此參數組合沒有交易。不要因而降低門檻；先檢查策略覆蓋率與市場適配性。")
            else:
                win_rate = 100 * (trades["淨回報%"] > 0).mean()
                avg_return = trades["淨回報%"].mean()
                profit_factor = trades.loc[trades["淨回報%"] > 0, "淨回報%"].sum() / abs(trades.loc[trades["淨回報%"] < 0, "淨回報%"].sum()) if (trades["淨回報%"] < 0).any() else np.nan
                equity = (1 + trades["淨回報%"] / 100).cumprod()
                max_dd = ((equity / equity.cummax()) - 1).min() * 100
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("交易數", len(trades)); x2.metric("勝率", f"{win_rate:.1f}%")
                x3.metric("平均淨回報", f"{avg_return:.2f}%"); x4.metric("最大交易序列回撤", f"{max_dd:.2f}%")
                st.metric("Profit factor", "—" if pd.isna(profit_factor) else f"{profit_factor:.2f}")
                st.dataframe(trades, use_container_width=True, hide_index=True)
                fig = go.Figure(go.Scatter(x=trades["出場日"], y=(equity - 1) * 100, mode="lines+markers", name="累積淨回報"))
                fig.update_layout(template="plotly_dark", height=350, title="按交易順序的累積淨回報", yaxis_title="%")
                st.plotly_chart(fig, use_container_width=True)
                st.warning("限制：日線 OHLC 無法知道同一日先觸及止損還是目標；本回測保守地假設同日雙觸發時先止損。結果不代表未來表現。")

with log_tab:
    st.subheader("去重後的訊號紀錄")
    logs = load_signals()
    if logs.empty:
        st.info("尚未記錄訊號。請先在掃描頁面執行掃描。")
    else:
        st.dataframe(logs.drop(columns=["payload_json"]), use_container_width=True, hide_index=True)
        st.download_button("下載訊號紀錄 CSV", logs.to_csv(index=False).encode("utf-8-sig"), "signal_log.csv", "text/csv")
        st.caption("每筆紀錄保留模型版本、regime 和完整因子快照，避免日後無法重現當時評分。")

with risk_tab:
    st.subheader("風險為本的部位計算")
    st.caption("此工具先限制每筆最大虧損，再以可承受損失反推股數；仍須自行考慮整手、稅項、貨幣、流動性及持倉相關性。")
    r1, r2, r3, r4 = st.columns(4)
    account = r1.number_input("帳戶淨值", min_value=1_000.0, value=100_000.0, step=1_000.0)
    risk_pct = r2.slider("每筆最大帳戶風險 (%)", 0.25, 2.0, 0.75, 0.25)
    max_alloc_pct = r3.slider("單一持倉最大名義比例 (%)", 1.0, 30.0, 10.0, 1.0)
    risk_ticker = r4.text_input("代碼", tickers[0] if tickers else "AAPL", key="risk_ticker")
    if st.button("計算可承受部位"):
        df, meta = fetch_ohlcv(risk_ticker, "2y")
        if df is None:
            st.error(meta["status"])
        else:
            result = score_stock(df, regime, regime_stats)
            per_share_risk = result.price - result.stop
            risk_budget = account * risk_pct / 100
            risk_shares = int(risk_budget / per_share_risk) if per_share_risk > 0 else 0
            allocation_shares = int((account * max_alloc_pct / 100) / result.price)
            shares = max(0, min(risk_shares, allocation_shares))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("入場參考價", result.price); m2.metric("結構止損", result.stop)
            m3.metric("每股風險", f"{per_share_risk:.3f}"); m4.metric("建議上限股數", shares)
            st.write(f"風險預算：{risk_budget:,.2f}｜名義金額：約 {shares * result.price:,.2f}｜模型目標：{result.target}")
            st.warning("下單前應將股數向下調整至交易所整手要求，並確認盤中價差、貨幣兌換及實際止損可成交性。")
