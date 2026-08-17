import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="V3.3 Value + Timing Scanner", page_icon="💎", layout="wide")

APP_VERSION = "V3.3"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; ValueTimingScanner/3.3)"}


# ----------------------------- Universe -----------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_sp500_universe():
    """Current S&P 500 constituents from the public constituent table."""
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0].copy()
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["market"] = "US"
    return df[["ticker", "name", "sector", "market"]].dropna().drop_duplicates("ticker")


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_hsi_universe():
    """Current Hang Seng Index constituents from the public constituent table.

    HK tickers are converted to the Yahoo Finance convention, e.g. 0005.HK.
    The table layout changes occasionally, so a clear error is better than silently
    using a stale hard-coded list.
    """
    tables = pd.read_html("https://en.wikipedia.org/wiki/Hang_Seng_Index")
    candidates = []
    for raw in tables:
        cols = {str(c).strip().lower(): c for c in raw.columns}
        code_col = next((cols[k] for k in cols if k in {"ticker", "code", "stock code", "symbol"}), None)
        name_col = next((cols[k] for k in cols if "company" in k or k == "name"), None)
        if code_col is None:
            continue
        temp = pd.DataFrame()
        temp["raw_code"] = raw[code_col].astype(str)
        temp["name"] = raw[name_col].astype(str) if name_col else temp["raw_code"]
        temp["code"] = temp["raw_code"].str.extract(r"(\d{1,5})", expand=False)
        temp = temp.dropna(subset=["code"])
        temp["ticker"] = temp["code"].str.zfill(4) + ".HK"
        temp["sector"] = "Hong Kong"
        temp["market"] = "HK"
        candidates.append(temp[["ticker", "name", "sector", "market"]])

    if not candidates:
        raise RuntimeError("Cannot locate the current HSI constituent table.")

    df = pd.concat(candidates, ignore_index=True).drop_duplicates("ticker")
    # A real HSI constituent list is much smaller than a broad index table.
    if len(df) < 50 or len(df) > 150:
        raise RuntimeError("The HSI source format changed; please refresh the constituent parser.")
    return df


def get_universe(selection):
    frames = []
    errors = []
    if selection in ("S&P 500", "Both"):
        try:
            frames.append(get_sp500_universe())
        except Exception as exc:
            errors.append(f"S&P 500 universe error: {exc}")
    if selection in ("Hang Seng Index", "Both"):
        try:
            frames.append(get_hsi_universe())
        except Exception as exc:
            errors.append(f"HSI universe error: {exc}")
    if not frames:
        raise RuntimeError(" | ".join(errors))
    return pd.concat(frames, ignore_index=True), errors


# ----------------------------- Market data -----------------------------
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cmf(frame, period=20):
    spread = (frame["High"] - frame["Low"]).replace(0, np.nan)
    multiplier = ((frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])) / spread
    return (multiplier * frame["Volume"]).rolling(period).sum() / frame["Volume"].rolling(period).sum()


def download_prices(tickers):
    raw = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result = {}
    if raw.empty:
        return result
    if len(tickers) == 1:
        result[tickers[0]] = raw.dropna(how="all")
        return result
    for ticker in tickers:
        try:
            df = raw[ticker].dropna(how="all")
            if not df.empty:
                result[ticker] = df
        except Exception:
            pass
    return result


def technical_snapshot(ticker, prices):
    if ticker not in prices:
        return None
    df = prices[ticker].copy().dropna()
    if len(df) < 205:
        return None
    close = df["Close"]
    last = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    rsi14 = float(rsi(close).iloc[-1])
    cmf20 = float(cmf(df).iloc[-1])
    vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1])
    low60 = float(close.tail(60).min())
    high5 = float(close.tail(5).max())
    return {
        "price": last,
        "rsi14": rsi14,
        "cmf20": cmf20,
        "volume_ratio": vol_ratio,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "near_60d_low": last <= low60 * 1.10,
        "breakout_5d": last >= high5 * 0.995,
        "above_ma20": last > ma20,
        "above_ma200": last > ma200,
    }


# ----------------------------- Fundamentals -----------------------------
def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fundamentals(ticker):
    info = yf.Ticker(ticker).get_info()
    market_cap = finite(info.get("marketCap"))
    free_cash_flow = finite(info.get("freeCashflow"))
    fcf_yield = free_cash_flow / market_cap if market_cap > 0 and not np.isnan(free_cash_flow) else np.nan
    return {
        "pe": finite(info.get("trailingPE")),
        "forward_pe": finite(info.get("forwardPE")),
        "pb": finite(info.get("priceToBook")),
        "ev_ebitda": finite(info.get("enterpriseToEbitda")),
        "fcf_yield": fcf_yield,
        "revenue_growth": finite(info.get("revenueGrowth")),
        "profit_margin": finite(info.get("profitMargins")),
        "debt_to_equity": finite(info.get("debtToEquity")),
        "current_ratio": finite(info.get("currentRatio")),
        "return_on_equity": finite(info.get("returnOnEquity")),
        "free_cash_flow": free_cash_flow,
        "market_cap": market_cap,
    }


def score_valuation(f):
    points = 0
    available = 0
    reasons = []
    pe = f["forward_pe"] if not np.isnan(f["forward_pe"]) else f["pe"]
    if not np.isnan(pe) and pe > 0:
        available += 1
        if pe <= 12:
            points += 22; reasons.append("低盈利倍數")
        elif pe <= 18:
            points += 15; reasons.append("合理盈利倍數")
        elif pe <= 25:
            points += 7
    if not np.isnan(f["pb"]) and f["pb"] > 0:
        available += 1
        if f["pb"] <= 1.2:
            points += 18; reasons.append("低市賬率")
        elif f["pb"] <= 2.0:
            points += 11
        elif f["pb"] <= 3.0:
            points += 5
    if not np.isnan(f["ev_ebitda"]) and f["ev_ebitda"] > 0:
        available += 1
        if f["ev_ebitda"] <= 8:
            points += 18; reasons.append("低 EV/EBITDA")
        elif f["ev_ebitda"] <= 12:
            points += 11
        elif f["ev_ebitda"] <= 16:
            points += 5
    if not np.isnan(f["fcf_yield"]):
        available += 1
        if f["fcf_yield"] >= 0.08:
            points += 22; reasons.append("高自由現金流收益率")
        elif f["fcf_yield"] >= 0.05:
            points += 15; reasons.append("正自由現金流收益率")
        elif f["fcf_yield"] >= 0.03:
            points += 7
    if not np.isnan(f["return_on_equity"]) and f["return_on_equity"] >= 0.12:
        points += 10; reasons.append("ROE 合格")
    if not np.isnan(f["profit_margin"]) and f["profit_margin"] > 0:
        points += 5
    normalizer = 95 if available >= 4 else (80 if available == 3 else 60)
    score = round(min(100, points / normalizer * 100)) if available >= 2 else np.nan
    return score, reasons


def value_traps(f):
    flags = []
    if not np.isnan(f["free_cash_flow"]) and f["free_cash_flow"] < 0:
        flags.append("自由現金流為負")
    if not np.isnan(f["revenue_growth"]) and f["revenue_growth"] < -0.10:
        flags.append("收入年增長低於 -10%")
    if not np.isnan(f["profit_margin"]) and f["profit_margin"] < 0:
        flags.append("淨利率為負")
    if not np.isnan(f["debt_to_equity"]) and f["debt_to_equity"] > 200:
        flags.append("負債權益比過高")
    if not np.isnan(f["current_ratio"]) and f["current_ratio"] < 0.8:
        flags.append("流動比率偏低")
    return flags


def score_timing(t):
    points = 0
    reasons = []
    if t["rsi14"] <= 32:
        points += 25; reasons.append("RSI 超賣")
    elif t["rsi14"] <= 40:
        points += 15; reasons.append("RSI 偏低")
    if t["near_60d_low"]:
        points += 15; reasons.append("接近 60 日低位")
    if t["above_ma20"]:
        points += 20; reasons.append("重上 MA20")
    if t["cmf20"] > 0:
        points += 15; reasons.append("資金流轉正")
    if t["volume_ratio"] >= 1.3:
        points += 10; reasons.append("成交量確認")
    if t["breakout_5d"]:
        points += 15; reasons.append("接近 5 日突破")
    return min(100, points), reasons


def labels(value_score, timing_score, flags):
    if np.isnan(value_score):
        value_label = "資料不足"
    elif value_score >= 80:
        value_label = "🟢 重度被低估"
    elif value_score >= 65:
        value_label = "🟠 中度被低估"
    elif value_score >= 50:
        value_label = "🟡 輕度被低估"
    else:
        value_label = "未達估值門檻"

    if flags:
        action = "⚠️ 價值陷阱風險"
    elif np.isnan(value_score) or value_score < 50:
        action = "—"
    elif timing_score >= 55:
        action = "⭐ 優先研究候選"
    elif timing_score >= 30:
        action = "⏳ 便宜但等待突破"
    else:
        action = "👀 估值觀察"
    return value_label, action


def scan(universe, max_stocks, workers):
    universe = universe.head(max_stocks).copy()
    prices = download_prices(universe["ticker"].tolist())
    technical = {ticker: technical_snapshot(ticker, prices) for ticker in universe["ticker"]}
    candidates = universe[universe["ticker"].map(lambda x: technical.get(x) is not None)].copy()
    rows = []
    progress = st.progress(0, text="正在取得基本面資料…")

    def work(row):
        ticker = row.ticker
        f = fundamentals(ticker)
        v_score, v_reasons = score_valuation(f)
        traps = value_traps(f)
        t = technical[ticker]
        t_score, t_reasons = score_timing(t)
        v_label, action = labels(v_score, t_score, traps)
        return {
            "市場": row.market, "代號": ticker, "公司": row.name, "行業": row.sector,
            "估值分": v_score, "估值分類": v_label, "技術分": t_score, "綜合結論": action,
            "價格": t["price"], "RSI14": t["rsi14"], "CMF20": t["cmf20"],
            "P/E": f["forward_pe"] if not np.isnan(f["forward_pe"]) else f["pe"],
            "P/B": f["pb"], "EV/EBITDA": f["ev_ebitda"],
            "FCF Yield": f["fcf_yield"], "負債權益比": f["debt_to_equity"],
            "估值理由": "、".join(v_reasons) or "可用估值資料有限",
            "技術理由": "、".join(t_reasons) or "尚未確認反轉",
            "紅旗": "、".join(traps) if traps else "無主要紅旗",
            "資料時間": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, row) for row in candidates.itertuples(index=False)]
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception:
                pass
            progress.progress(done / max(1, total), text=f"基本面掃描：{done}/{total}")
            time.sleep(0.05)
    progress.empty()
    return pd.DataFrame(rows)


# ----------------------------- UI -----------------------------
st.title("💎 V3.3 被低估＋技術時機掃描")
st.caption("S&P 500／恒生指數成分股｜估值篩選、價值陷阱紅旗、V3 技術時機")

with st.sidebar:
    st.header("掃描設定")
    market = st.selectbox("市場範圍", ["S&P 500", "Hang Seng Index", "Both"])
    max_scan = st.slider("本次最大掃描數", min_value=25, max_value=600, value=100, step=25)
    workers = st.slider("基本面並行請求數", min_value=1, max_value=8, value=3)
    run = st.button("開始 V3.3 掃描", type="primary", use_container_width=True)
    st.caption("首次全市場掃描會較慢；建議先以 100 隻測試，再安排每日背景更新。")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()

if run:
    try:
        with st.spinner("正在更新成分股與下載一年價格資料…"):
            universe, source_errors = get_universe(market)
        if source_errors:
            for message in source_errors:
                st.warning(message)
        st.info(f"已取得 {len(universe)} 隻成分股；本次掃描前 {min(max_scan, len(universe))} 隻。")
        st.session_state.scan_results = scan(universe, max_scan, workers)
    except Exception as exc:
        st.error(f"掃描未完成：{exc}")

results = st.session_state.scan_results
if results.empty:
    st.info("按左側「開始 V3.3 掃描」建立候選名單。系統只會把通過估值門檻的股票顯示為研究候選。")
else:
    valid = results[results["估值分"].notna()].copy()
    candidates = valid[valid["綜合結論"].isin(["⭐ 優先研究候選", "⏳ 便宜但等待突破", "👀 估值觀察"])].copy()

    a, b, c, d = st.columns(4)
    a.metric("已取得基本面", len(results))
    b.metric("值得研究候選", len(candidates))
    c.metric("優先研究", int((results["綜合結論"] == "⭐ 優先研究候選").sum()))
    d.metric("價值陷阱紅旗", int(results["綜合結論"].eq("⚠️ 價值陷阱風險").sum()))

    st.subheader("研究候選")
    if candidates.empty:
        st.warning("本輪尚未出現符合門檻的候選。這不是買入訊號缺失，而是篩選器沒有強行推薦股票。")
    else:
        candidates = candidates.sort_values(["估值分", "技術分"], ascending=False)
        display_cols = ["市場", "代號", "公司", "估值分類", "估值分", "技術分", "綜合結論", "價格", "RSI14", "P/E", "P/B", "EV/EBITDA", "FCF Yield", "估值理由", "技術理由", "紅旗"]
        st.dataframe(
            candidates[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "價格": st.column_config.NumberColumn(format="%.2f"),
                "估值分": st.column_config.NumberColumn(format="%d"),
                "技術分": st.column_config.NumberColumn(format="%d"),
                "RSI14": st.column_config.NumberColumn(format="%.1f"),
                "P/E": st.column_config.NumberColumn(format="%.1f"),
                "P/B": st.column_config.NumberColumn(format="%.2f"),
                "EV/EBITDA": st.column_config.NumberColumn(format="%.1f"),
                "FCF Yield": st.column_config.NumberColumn(format="%.1%%"),
            },
        )

    with st.expander("查看所有已掃描結果（包括紅旗與資料不足）"):
        st.dataframe(results.sort_values("估值分", ascending=False), use_container_width=True, hide_index=True)

    csv = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載本輪掃描 CSV", csv, "v3_3_value_timing_scan.csv", "text/csv")

st.divider()
st.subheader("如何解讀")
st.markdown("""
- **估值分類**：以 P/E、P/B、EV/EBITDA、自由現金流收益率、ROE 與盈利質素組合計分；分數僅供篩選，不是內在價值估算。
- **價值陷阱紅旗**：自由現金流為負、收入急跌、虧損、過高槓桿或短期流動性不足會否決買入候選。
- **技術時機**：RSI 超賣只代表開始觀察；重上 MA20、資金流轉正、成交量與短期突破才提高進場可信度。
- **使用原則**：所有結果是研究候選，不是投資建議。下單前請核對最新業績、公告、估值同業比較、止損與倉位風險。
""")
