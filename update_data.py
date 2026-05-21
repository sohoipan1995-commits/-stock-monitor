"""GitHub Actions 執行此腳本，抓取最新數據存為 JSON"""
import json, os
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np

HK = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK",
      "0388.HK","0066.HK","0003.HK","0002.HK","0016.HK",
      "0883.HK","2318.HK","1299.HK","0001.HK","9988.HK",
      "0175.HK","0027.HK","2628.HK","0011.HK","0688.HK"]
US = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVBO",
      "ORCL","ASML","UBER","NEE","LITE","CLX","JPM","BAC",
      "SPY","QQQ","SOXL","AMD"]
MACRO = {"VIX":"^VIX","SPX":"^GSPC","HSI":"^HSI",
         "DXY":"DX-Y.NYB","US10Y":"^TNX"}

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

os.makedirs("data", exist_ok=True)

# Macro data
macro_out = {}
for name, tk in MACRO.items():
    try:
        df = yf.download(tk, period="1y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if df.empty: continue
        c  = float(df["close"].iloc[-1])
        p  = float(df["close"].iloc[-2])
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        macro_out[name] = {
            "val": round(c, 3),
            "chg": round((c-p)/p*100, 2),
            "pct": round((c-lo)/(hi-lo)*100, 1) if hi != lo else 50
        }
    except Exception as e:
        print(f"Macro error {name}: {e}")

with open("data/macro.json", "w") as f:
    json.dump({"updated": datetime.now(timezone.utc).isoformat(),
               "data": macro_out}, f, indent=2)
print("✅ macro.json saved")

# Stock snapshots
stock_out = {}
for tk in HK + US:
    try:
        df = yf.download(tk, period="2y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if df.empty or len(df) < 60: continue
        close = df["close"]
        rsi_v = float(calc_rsi(close).iloc[-1])
        hi52  = float(df["high"].iloc[-252:].max())
        lo52  = float(df["low"].iloc[-252:].min())
        cv    = float(close.iloc[-1])
        pv    = float(close.iloc[-2])
        vol_r = float(df["volume"].iloc[-1]) / float(df["volume"].rolling(20).mean().iloc[-1])
        stock_out[tk] = {
            "close": round(cv, 3),
            "chg1d": round((cv-pv)/pv*100, 2),
            "hi52":  round(hi52, 3),
            "lo52":  round(lo52, 3),
            "rsi":   round(rsi_v, 1),
            "vol_ratio": round(vol_r, 2),
            "from_high_pct": round((cv-hi52)/hi52*100, 1)
        }
    except Exception as e:
        print(f"Stock error {tk}: {e}")

with open("data/stocks.json", "w") as f:
    json.dump({"updated": datetime.now(timezone.utc).isoformat(),
               "data": stock_out}, f, indent=2)
print(f"✅ stocks.json saved ({len(stock_out)} tickers)")
