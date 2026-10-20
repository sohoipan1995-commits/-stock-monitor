"""V5.0 三引擎研究版。完整單檔Streamlit應用；不是投資建議或已驗證交易策略。"""
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests

APP_VERSION = "V5.0-three-engines-research"
MODEL_VERSION = "value-downside-decision-1.0"
DB_FILE = os.getenv("STOCK_V5_DB_FILE", "stock_v5.sqlite")
INDEXES = {
    "SP500":("US","https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",490,520),
    "HSI":("HK","https://en.wikipedia.org/wiki/Hang_Seng_Index",50,150),
    "DJIA":("US","https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",30,30),
    "NDX":("US","https://en.wikipedia.org/wiki/Nasdaq-100",95,110),
}
DEFAULT_SYMBOLS = "0700.HK\n0005.HK\n9988.HK\nAAPL\nMSFT\nNVDA"


def number(x, default=np.nan):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except (ValueError,TypeError,OverflowError):
        return default


def positive(x):
    return np.isfinite(number(x)) and number(x)>0


def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x,(np.bool_,bool)): return bool(x)
    if isinstance(x,(np.integer,int)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,datetime)): return x.isoformat()
    return x


def normalize_symbol(value):
    text=re.sub(r"\[[^\]]*\]","",str(value)).strip().upper()
    text=re.sub(r"^(?:SEHK|HKG|HKEX)\s*:\s*","",text)
    if text.endswith(".HK") or re.fullmatch(r"\d{1,5}(?:\.0)?",text):
        text=re.sub(r"\.HK$","",text)
        if text.endswith(".0"): text=text[:-2]
        return str(int(text)).zfill(4)+".HK" if text.isdigit() and 0<int(text)<=99999 else None
    text=text.replace(".","-")
    return text if re.fullmatch(r"[A-Z]{1,6}(?:-[A-Z])?",text) else None


def market_of(ticker):
    return "HK" if ticker.endswith(".HK") or ticker in {"^HSI","^HSCE"} else "US"


def validate_prices(raw):
    if raw is None or raw.empty: raise ValueError("沒有價格資料")
    d=raw.copy()
    if isinstance(d.columns,pd.MultiIndex):
        if len(set(d.columns.get_level_values(-1)))!=1:
            raise ValueError("需要單一股票資料")
        d.columns=d.columns.get_level_values(0)
    d.columns=[str(c).strip().lower().replace(" ","_") for c in d.columns]
    required=["open","high","low","close","volume"]
    if not set(required).issubset(d.columns): raise ValueError("缺少OHLCV欄位")
    cols=required+(["adj_close"] if "adj_close" in d else [])
    d=d[cols].apply(pd.to_numeric,errors="coerce")
    d.index=pd.DatetimeIndex(pd.to_datetime(d.index))
    if d.index.tz is not None: d.index=d.index.tz_localize(None)
    d.index=d.index.normalize()
    if d.index.has_duplicates: raise ValueError("重複交易日期")
    d=d.sort_index()
    # Fully empty rows are missing sessions, not fabricated zero-volume days.
    d=d.dropna(how="all",subset=required)
    if d.empty or not np.isfinite(d[required].to_numpy()).all(): raise ValueError("OHLCV含缺失／無限值")
    if (d[required[:4]]<=0).any().any() or (d.volume<0).any(): raise ValueError("價格／成交量不合法")
    if (d.high<d[["open","close","low"]].max(axis=1)).any() or (d.low>d[["open","close","high"]].min(axis=1)).any():
        raise ValueError("OHLC高低價次序錯誤")
    return d


def adjusted_path(raw):
    d=validate_prices(raw)
    if "adj_close" not in d or not np.isfinite(d.adj_close.to_numpy()).all() or (d.adj_close<=0).any():
        raise ValueError("缺完整Adj Close；無法統一股息復權研究路徑")
    factor=d.adj_close/d.close
    factor=factor/factor.iloc[-1]
    for col in ["open","high","low","close"]: d[col]=d[col]*factor
    return d


def wilder(s,period=14):
    s=pd.Series(s,index=s.index,dtype=float)
    result=pd.Series(np.nan,index=s.index,dtype=float)
    valid=s.dropna()
    if len(valid)<period: return result
    seed=valid.index[period-1]
    result.loc[seed]=valid.iloc[:period].mean()
    started=False
    prev=np.nan
    for key,v in s.items():
        if key==seed: prev=result.loc[key];started=True;continue
        if started and np.isfinite(v):
            prev=(prev*(period-1)+v)/period;result.loc[key]=prev
    return result


def rsi(close,period=14):
    delta=close.diff()
    g=wilder(delta.clip(lower=0),period)
    loss=wilder(-delta.clip(upper=0),period)
    result=100-100/(1+g/loss.replace(0,np.nan))
    result=result.mask((loss==0)&(g>0),100).mask((g==0)&(loss>0),0)
    return result.mask((g==0)&(loss==0),50)


def mfi(d,period=14):
    typical=(d.high+d.low+d.close)/3
    flow=typical*d.volume
    pos=flow.where(typical>typical.shift(),0).rolling(period,min_periods=period).sum()
    neg=flow.where(typical<typical.shift(),0).rolling(period,min_periods=period).sum()
    result=100-100/(1+pos/neg.replace(0,np.nan))
    return result.mask((neg==0)&(pos>0),100).mask((pos==0)&(neg>0),0).mask((pos==0)&(neg==0),50)


def indicators(d):
    prev=d.close.shift()
    tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    macd=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    spread=(d.high-d.low)
    mf=((2*d.close-d.high-d.low)/spread.replace(0,np.nan)).where(spread!=0,0)*d.volume
    return pd.DataFrame(dict(atr=wilder(tr),rsi=rsi(d.close),mfi=mfi(d),macd_hist=hist,
        ma20=d.close.rolling(20).mean(),ma50=d.close.rolling(50).mean(),ma200=d.close.rolling(200).mean(),
        cmf=mf.rolling(20).sum()/d.volume.rolling(20).sum().replace(0,np.nan),
        vol20=d.volume.rolling(20).mean(),realized_vol=d.close.pct_change().rolling(20).std()*np.sqrt(252)),index=d.index)


def confirmed_lows(d,left=10,right=5):
    result=[]
    values=d.low.to_numpy()
    for confirmed in range(left+right,len(d)):
        p=confirmed-right
        if values[p]<min(values[p-left:p]) and values[p]<=min(values[p+1:confirmed+1]):
            result.append(dict(index=p,confirmed_index=confirmed,date=str(d.index[p].date()),
                               confirmed_at=str(d.index[confirmed].date()),price=float(values[p])))
    return result


def relative_strength(stock,benchmark,period=60):
    common=stock.index.intersection(benchmark.index).sort_values()
    if len(common)<period+1: return np.nan
    dates=common[-(period+1):]
    return float((stock.loc[dates[-1],"close"]/stock.loc[dates[0],"close"]-
                  benchmark.loc[dates[-1],"close"]/benchmark.loc[dates[0],"close"])*100)


def quality_engine(info):
    sector=str(info.get("sector") or "")
    financial="financial" in sector.lower()
    blockers,warnings,missing=[],[],[]
    if info.get("quoteType")!="EQUITY": blockers.append("本估值模型只適用普通股；ETF／指數另行研究")
    if not info.get("industry"): missing.append("行業")
    if not info.get("currency") or not info.get("financialCurrency"): missing.append("報價／財報幣種")
    if info.get("currency") and info.get("financialCurrency") and info["currency"]!=info["financialCurrency"]:
        blockers.append("財報與報價幣種不同；本版不自動換算每股財務數字")
    eps=number(info.get("trailingEps"));pb=number(info.get("priceToBook"));book=number(info.get("bookValue"))
    roe=number(info.get("returnOnEquity"));ocf=number(info.get("operatingCashflow"))
    debt=number(info.get("debtToEquity"));growth=number(info.get("revenueGrowth"))
    if financial:
        if not positive(pb) or not positive(book): blockers.append("金融股PB／每股淨資產無效或非正")
    elif not positive(eps): blockers.append("TTM盈利非正或缺失；不套盈利企業模型")
    if not np.isfinite(roe): missing.append("ROE")
    elif roe<0: blockers.append("ROE為負")
    elif roe<.05: warnings.append("ROE低於5%")
    if not np.isfinite(growth): missing.append("營收增長")
    elif growth<-.2: warnings.append("營收按年下跌逾20%")
    if not financial:
        if not np.isfinite(ocf): missing.append("營運現金流")
        elif ocf<0: warnings.append("營運現金流為負")
        if np.isfinite(debt) and debt>200:
            warnings.append("負債權益比高於200%")
            if np.isfinite(ocf) and ocf<0: blockers.append("高負債兼營運現金流為負")
    return dict(financial=financial,blockers=blockers,warnings=warnings,missing=missing,
                eligible=not blockers and not missing,quality_status="不合資格" if blockers else "資料不足" if missing else "有風險旗號" if warnings else "初步合格")


def valuation_engine(ticker,price,info,all_infos,min_peers=8,manual=None):
    q=quality_engine(info)
    metric="PB" if q["financial"] else "PE"
    key="priceToBook" if q["financial"] else "trailingPE"
    per_share=number(info.get("bookValue") if q["financial"] else info.get("trailingEps"))
    peers=[]
    for symbol,other in all_infos.items():
        if symbol==ticker or market_of(symbol)!=market_of(ticker): continue
        if not info.get("industry") or other.get("industry")!=info["industry"]: continue
        if other.get("quoteType")!="EQUITY":continue
        v=number(other.get(key))
        if positive(v) and v< (30 if metric=="PB" else 200): peers.append(v)
    result=dict(method=f"同市場同Yahoo行業TTM {metric}倍數推算；非內在價值DCF",metric=metric,
                peer_count=len(peers),current_multiple=number(info.get(key)),fair_low=np.nan,fair_base=np.nan,fair_high=np.nan,
                discount=np.nan,upside_base=np.nan,confidence="不足",valuation_status="無法可靠估值",
                quality=q,assumptions="",manual=False)
    if manual is not None:
        lo,mid,hi=map(float,manual)
        if not (0<lo<=mid<=hi): raise ValueError("手動倍數必須0<低<=中<=高")
        multiples=(lo,mid,hi)
        result.update(method=f"使用者手動{metric}情境；非自動合理價",manual=True,confidence="手動假設")
    elif len(peers)>=min_peers:
        multiples=tuple(np.quantile(peers,[.25,.5,.75]))
        result["confidence"]="中（倍數代理）" if len(peers)>=15 else "低（倍數代理）"
    else:
        result["assumptions"]=f"同業不足{min_peers}家；不以固定PE/PB門檻捏造合理價"
        return result
    if q["blockers"] or not positive(per_share) or not positive(price):
        result["assumptions"]="；".join(q["blockers"]) or "每股財務數值無效"
        return result
    lo,mid,hi=[per_share*m for m in multiples]
    discount=1-price/lo
    status="重度低估候選" if discount>=.30 else "中度低估候選" if discount>=.20 else "輕度低估候選" if discount>=.10 else "安全邊際不足"
    result.update(fair_low=lo,fair_base=mid,fair_high=hi,discount=discount,upside_base=mid/price-1,
                  valuation_status=status,assumptions=f"{metric}低／中／高={multiples[0]:.2f}/{multiples[1]:.2f}/{multiples[2]:.2f}；未調整同業成長、槓桿與一次性盈利差異；非歷史估值百分位")
    return result


def downside_engine(d,horizons=(20,60),min_windows=30,shock_fund=.20,shock_multiple=.20):
    if not (0<=shock_fund<1 and 0<=shock_multiple<1): raise ValueError("壓力比例必須在0至1之間")
    d=validate_prices(d)
    price=float(d.close.iloc[-1]);ind=indicators(d)
    current_bear=bool(price<ind.ma200.iloc[-1]) if np.isfinite(ind.ma200.iloc[-1]) else None
    results={}
    for h in horizons:
        if h<1: raise ValueError("觀察期須為正")
        episodes=[]
        for i in range(len(d)-h-1,199,-h):
            low=float(d.low.iloc[i+1:i+h+1].min())
            loss=max(0.,1-low/float(d.close.iloc[i]))
            bear=bool(d.close.iloc[i]<ind.ma200.iloc[i])
            episodes.append((loss,bear))
        similar=[v for v,b in episodes if b==current_bear]
        fallback=len(similar)<min_windows
        values=np.array([v for v,b in episodes] if fallback else similar,dtype=float)
        n=len(values)
        enough=n>=min_windows
        p50=float(np.quantile(values,.5)) if enough else np.nan
        p80=float(np.quantile(values,.8)) if enough else np.nan
        p95=float(np.quantile(values,.95)) if n>=100 else np.nan
        tail=values[values>=p95] if np.isfinite(p95) else np.array([])
        results[str(h)]=dict(windows=n,similar_windows=len(similar),all_windows=len(episodes),
             fallback_all_regimes=fallback,p50=p50,p80=p80,p95=p95,
             tail_average=float(tail.mean()) if len(tail)>=5 else np.nan,
             p80_price=price*(1-p80) if enough else np.nan,
             p95_price=price*(1-p95) if np.isfinite(p95) else np.nan,
             status="歷史情境，非已校準預測" if enough else f"不重疊窗口少於{min_windows}；不顯示分位數")
    lows=confirmed_lows(d)
    supports=sorted({round(e["price"],6) for e in lows if e["price"]<price and len(d)-1-e["confirmed_index"]<=120},reverse=True)
    stress_price=price*(1-shock_fund)*(1-shock_multiple)
    return dict(horizons=results,supports=supports[:3],stress_price=stress_price,stress_loss=1-stress_price/price,
                stress_assumptions=f"每股盈利／金融股淨資產-{shock_fund:.0%}；倍數-{shock_multiple:.0%}；按現價基準機械壓力，不附機率",
                realized_vol=number(ind.realized_vol.iloc[-1]),market_trend="低於MA200" if current_bear else "高於MA200" if current_bear is not None else "資料不足")


def decision_engine(d,valuation,downside,benchmark=None,max_p80=.20,min_liquidity=1e7):
    if len(d)<210: return dict(action="資料不足",reason="少於210根日線",confirm_count=0)
    t=indicators(d);x=t.iloc[-1];price=float(d.close.iloc[-1])
    turnover=float((d.close*d.volume).iloc[-20:].mean())
    lows=confirmed_lows(d)
    active=[v for v in lows if price>v["price"] and len(d)-1-v["confirmed_index"]<=60]
    support=active[-1]["price"] if active else np.nan
    divergence=False
    if len(lows)>=2:
        a,b=lows[-2:]
        divergence=bool(b["price"]<a["price"] and price>b["price"] and
            len(d)-1-b["confirmed_index"]<=20 and t.rsi.iloc[b["index"]]>t.rsi.iloc[a["index"]])
    confirms=[price>x.ma20 and x.ma20>t.ma20.iloc[-6],x.macd_hist>0,x.cmf>0,
              d.volume.iloc[-1]>x.vol20*1.2 and d.close.iloc[-1]>d.open.iloc[-1]]
    count=int(sum(bool(v) for v in confirms))
    stop=support-.5*x.atr if positive(support) else price-2*x.atr
    if not (np.isfinite(stop) and 0<stop<price): stop=np.nan
    trigger=max(price,float(d.high.iloc[-5:].max()),number(x.ma20,price))
    plan_valid=positive(stop) and stop<trigger
    target=trigger+2*(trigger-stop) if plan_valid else np.nan
    q=valuation["quality"];discount=number(valuation.get("discount"))
    p80=number(downside["horizons"].get("20",{}).get("p80"))
    rs=relative_strength(d,benchmark) if benchmark is not None else np.nan
    if benchmark is None or benchmark.empty or benchmark.index[-1]!=d.index[-1]:
        action,reason="資料不足","缺同日市場基準，暫停行動分類"
    elif q["blockers"]:
        action,reason="排除／價值陷阱風險","；".join(q["blockers"])
    elif not q["eligible"]:
        action,reason="資料不足","缺少："+"、".join(q["missing"])
    elif turnover<min_liquidity:
        action,reason="排除／流動性不足","20日估算平均成交額低於本市場門檻"
    elif not np.isfinite(discount):
        action,reason="相對估值觀察","同業／估值資料不足，不判斷被低估幅度"
    elif discount<.10:
        action,reason="價格未夠吸引","相對保守倍數情境折讓不足10%"
    elif not np.isfinite(p80):
        action,reason="低估／風險樣本不足","歷史下行窗口不足，不給優先候選"
    elif p80>max_p80 or q["warnings"]:
        action,reason="低估／高風險觀察","較差歷史下行情境超門檻或基本面有風險旗號"
    elif count>=3 and plan_valid:
        action,reason="優先研究／等待觸發","估值、資料及預設確認條件合格；不是已驗證買入訊號"
    elif x.rsi<35 or divergence:
        action,reason="低估／左側觀察","超賣或已確認低點動能改善，未完成入場確認"
    else:
        action,reason="低估／等待確認","估值有折讓，但預設技術條件未完成"
    return dict(action=action,reason=reason,confirm_count=count,rsi=number(x.rsi),mfi=number(x.mfi),
       atr=number(x.atr),turnover=turnover,rs60=rs,bullish_divergence=divergence,
       support=support,trigger=trigger,stop=stop,target_2r=target,
       plan_note="確認價為研究觸發價；2R按觸發價計，不保證成交；跳空需重建計劃")


def estimate_position(entry,stop,account,risk_pct,max_weight,fx=1.,lot=1,fee_bps=10.):
    vals=[entry,stop,account,risk_pct,max_weight,fx,lot,fee_bps]
    if not all(np.isfinite(number(v)) for v in vals): raise ValueError("風控輸入必須有限")
    if not (0<stop<entry and account>0 and fx>0 and 0<risk_pct<=.1 and 0<max_weight<=1 and lot>=1 and int(lot)==lot and fee_bps>=0):
        raise ValueError("需0<止損<入場價、正資金／匯率、有效比例及整數每手")
    fee=fee_bps/10000
    risk_unit=((entry-stop)+(entry+stop)*fee)*fx
    cash_unit=entry*(1+fee)*fx
    shares=int(min(account*risk_pct/risk_unit,account*max_weight/cash_unit)//lot)*int(lot)
    return dict(shares=shares,cash=shares*cash_unit,scenario_loss=shares*risk_unit,
                risk_budget=account*risk_pct,fx=fx,lot=int(lot))


def init_db(path=DB_FILE):
    with sqlite3.connect(path,timeout=30) as con:
        con.execute("CREATE TABLE IF NOT EXISTS v5_watchlist(ticker TEXT PRIMARY KEY,created_at TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS v5_snapshots(run_id TEXT,ticker TEXT,model_version TEXT,asof TEXT,payload TEXT,PRIMARY KEY(run_id,ticker))")
        con.execute("CREATE TABLE IF NOT EXISTS v5_runs(run_id TEXT PRIMARY KEY,created_at TEXT,payload TEXT)")
        for table,required in [("v5_watchlist",{"ticker","created_at"}),("v5_snapshots",{"run_id","ticker","model_version","asof","payload"}),("v5_runs",{"run_id","created_at","payload"})]:
            cols={r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if not required.issubset(cols): raise ValueError(f"{table}結構不相容，請先備份；本版不刪表重建")


def watchlist(path=DB_FILE):
    with sqlite3.connect(path,timeout=30) as con:return [r[0] for r in con.execute("SELECT ticker FROM v5_watchlist ORDER BY ticker")]


def save_run(rows,metadata,path=DB_FILE):
    run_id=datetime.now(timezone.utc).isoformat()+"-"+uuid.uuid4().hex[:8]
    payload=json.dumps(clean_json(metadata),ensure_ascii=False,allow_nan=False)
    with sqlite3.connect(path,timeout=30) as con:
        con.execute("INSERT INTO v5_runs VALUES(?,?,?)",(run_id,datetime.now(timezone.utc).isoformat(),payload))
        for row in rows:
            con.execute("INSERT INTO v5_snapshots VALUES(?,?,?,?,?)",(run_id,row["ticker"],MODEL_VERSION,row["asof"],json.dumps(clean_json(row),ensure_ascii=False,allow_nan=False)))
    return run_id


def completed_session(market):
    import exchange_calendars as xcals
    now=pd.Timestamp.now(tz="UTC")
    calendar=xcals.get_calendar("XHKG" if market=="HK" else "XNYS")
    days=calendar.sessions_in_range((now-pd.Timedelta(days=35)).date(),now.date())
    eligible=[s for s in days if calendar.session_close(s)+pd.Timedelta(minutes=20)<=now]
    if not eligible: raise ValueError("未能確認完整交易日")
    return str(pd.Timestamp(eligible[-1]).date())


def fetch_prices(ticker,period="10y"):
    import yfinance as yf
    raw=yf.download(ticker,period=period,interval="1d",auto_adjust=False,threads=False,progress=False,timeout=20)
    return validate_prices(raw)


def fetch_info(ticker):
    import yfinance as yf
    data=yf.Ticker(ticker).get_info() or {}
    if not data: raise ValueError("Yahoo基本面為空")
    return {**data,"_fetched_at":datetime.now(timezone.utc).isoformat()}


def get_constituents(key):
    market,url,low,high=INDEXES[key]
    resp=requests.get(url,headers={"User-Agent":"Mozilla/5.0 stock-research"},timeout=20)
    resp.raise_for_status()
    choices=[]
    for frame in pd.read_html(StringIO(resp.text)):
        frame.columns=[" ".join(map(str,c)) if isinstance(c,tuple) else str(c) for c in frame.columns]
        for col in frame.columns:
            label=re.sub(r"\[[^\]]*\]","",str(col)).strip().lower()
            if label not in {"symbol","ticker","ticker symbol"}:continue
            items={normalize_symbol(v) for v in frame[col] if not pd.isna(v)}-{None}
            items=sorted(t for t in items if market_of(t)==market)
            if low<=len(items)<=high:choices.append(tuple(items))
    choices=list(set(choices))
    if len(choices)!=1:raise ValueError(f"{key}成分表不唯一／格式改變，未用固定名單冒充")
    return dict(symbols=list(choices[0]),url=url,fetched_at=datetime.now(timezone.utc).isoformat(),official_effective_date_verified=False)


def flatten(ticker,raw,info,v,r,d):
    h20=r["horizons"].get("20",{});h60=r["horizons"].get("60",{})
    return dict(ticker=ticker,name=info.get("shortName",ticker),market=market_of(ticker),currency=info.get("currency"),
       asof=str(raw.index[-1].date()),price=float(raw.close.iloc[-1]),industry=info.get("industry"),
       action=d["action"],reason=d["reason"],valuation_status=v["valuation_status"],confidence=v["confidence"],
       conservative_discount=v["discount"],fair_low=v["fair_low"],fair_base=v["fair_base"],fair_high=v["fair_high"],
       peer_count=v["peer_count"],quality=v["quality"]["quality_status"],
       p80_20=h20.get("p80"),p95_20=h20.get("p95"),windows20=h20.get("windows"),
       p80_60=h60.get("p80"),p95_60=h60.get("p95"),windows60=h60.get("windows"),
       stress_loss=r["stress_loss"],trigger=d.get("trigger"),stop=d.get("stop"),target_2r=d.get("target_2r"),
       fundamentals_fetched_at=info.get("_fetched_at"),valuation=v,downside=r,decision=d,model=MODEL_VERSION)


def analyse(ticker,raw,info,peers,benchmark,cfg,manual=None):
    d=adjusted_path(raw)
    v=valuation_engine(ticker,float(raw.close.iloc[-1]),info,peers,int(cfg["min_peers"]),manual)
    r=downside_engine(d,shock_fund=cfg["shock_fund"],shock_multiple=cfg["shock_multiple"])
    dec=decision_engine(d,v,r,benchmark,max_p80=cfg["max_p80"],min_liquidity=cfg["liq_hk"] if market_of(ticker)=="HK" else cfg["liq_us"])
    return flatten(ticker,raw,info,v,r,dec)
