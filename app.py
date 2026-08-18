import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="撈底監察系統 V4.2", page_icon="📈", layout="wide")

HK_WATCHLIST = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","0066.HK","0003.HK","0002.HK","0016.HK","0883.HK","2318.HK","1299.HK","0001.HK","9988.HK","0175.HK","3690.HK","9618.HK","0981.HK","9999.HK","1211.HK","2688.HK","0762.HK","1810.HK","1024.HK","2020.HK"]
US_WATCHLIST = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","ORCL","AMD","QCOM","INTC","AMAT","LRCX","JPM","BAC","GS","BRK-B","COST","WMT","JNJ","UNH","XOM","NEE","UBER","NFLX","SPY","QQQ"]
INDEXES = {"🇭🇰 恒生指數 HSI": "^HSI", "🇺🇸 納斯達克 100 NDX": "^NDX", "🇺🇸 標普 500 SPX": "^GSPC"}
DB_FILE = Path("signal_events_v42.sqlite")
C_BG, C_PANEL, C_BORDER = "#0d1117", "#161b22", "#30363d"
C_GREEN, C_RED, C_ORANGE, C_BLUE, C_GREY = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#8b949e"

st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background:{C_BG};}} [data-testid="stSidebar"]{{background:{C_PANEL};}}
h1,h2,h3,p,label,.stMarkdown{{color:#e6edf3!important;}}
.card{{background:{C_PANEL};border:1px solid {C_BORDER};border-left:5px solid {C_BLUE};border-radius:10px;padding:15px;margin:8px 0;}}
.good{{border-left-color:{C_GREEN};}} .wait{{border-left-color:{C_ORANGE};}} .bad{{border-left-color:{C_RED};}}
</style>""", unsafe_allow_html=True)


def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS signal_events(
        event_id TEXT PRIMARY KEY,event_time TEXT,ticker TEXT,market TEXT,action TEXT,total_score REAL,
        price REAL,breakout_price REAL,stop_price REAL,target_price REAL,risk_reward REAL,
        valuation REAL,quality REAL,catalyst REAL,technical REAL,risk REAL,snapshot_json TEXT)""")


def save_signal(r):
    event_id = f"{r['ticker']}_{r['event_time'][:10]}_{r['action']}"
    values = (event_id,r["event_time"],r["ticker"],r["market"],r["action"],r["total_score"],r["price"],r["breakout_price"],r["stop"],r["target"],r["rr"],r["valuation"],r["quality"],r["catalyst"],r["technical"],r["risk"],json.dumps(r["snapshot"],ensure_ascii=False))
    with sqlite3.connect(DB_FILE) as con:
        con.execute("INSERT OR REPLACE INTO signal_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)


def load_signals():
    with sqlite3.connect(DB_FILE) as con:
        return pd.read_sql_query("SELECT * FROM signal_events ORDER BY event_time DESC", con)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(x).lower() for x in df.columns]
        required = ["open","high","low","close","volume"]
        return df[required].dropna() if all(c in df.columns for c in required) else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    d = {"name":ticker,"forward_pe":np.nan,"pb":np.nan,"ev_ebitda":np.nan,"fcf_yield":np.nan,"roe":np.nan,"debt_equity":np.nan,"earnings_growth":np.nan}
    try:
        i = yf.Ticker(ticker).info
        cap, fcf = i.get("marketCap"), i.get("freeCashflow")
        d.update({"name":i.get("shortName") or i.get("longName") or ticker,"forward_pe":i.get("forwardPE",np.nan),"pb":i.get("priceToBook",np.nan),"ev_ebitda":i.get("enterpriseToEbitda",np.nan),"fcf_yield":fcf/cap*100 if fcf and cap else np.nan,"roe":i.get("returnOnEquity",np.nan),"debt_equity":i.get("debtToEquity",np.nan),"earnings_growth":i.get("earningsGrowth",np.nan)})
    except Exception:
        pass
    return d


def fetch_many(tickers):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        jobs = {ex.submit(fetch_ohlcv, t):t for t in tickers}
        for job in as_completed(jobs):
            t = jobs[job]
            try: out[t] = job.result()
            except Exception: out[t] = None
    return out


def rsi(close, n=14):
    d = close.diff(); g = d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100 - 100/(1+g/l.replace(0,np.nan))


def macd(close):
    line = close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean(); sig = line.ewm(span=9,adjust=False).mean()
    return line, sig, line-sig


def atr(df, n=14):
    ranges = pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1)
    return ranges.max(axis=1).rolling(n).mean()


def cmf(df, n=20):
    multiplier = (2*df.close-df.high-df.low)/(df.high-df.low).replace(0,np.nan)
    return (multiplier*df.volume).rolling(n).sum()/df.volume.rolling(n).sum().replace(0,np.nan)


def weekly_rsi(df):
    return rsi(df.close.resample("W-FRI").last().dropna(),14)


def avwap(df, lookback=60):
    anchor = df.iloc[-lookback:].low.idxmin(); part = df.loc[anchor:]; typical = (part.high+part.low+part.close)/3
    return float((typical*part.volume).sum()/part.volume.sum()) if part.volume.sum()>0 else np.nan


def pivots(series, order=4):
    lo, hi = [], []
    for k in range(order,len(series)-order):
        w=series.iloc[k-order:k+order+1]
        if series.iloc[k] == w.min(): lo.append(k)
        if series.iloc[k] == w.max(): hi.append(k)
    return lo,hi


def bottom_structure(df):
    p=df.iloc[-100:].reset_index(drop=True); lows,_=pivots(p.low,4)
    if len(lows)<2: return False,np.nan,False,"未形成可辨識底部結構"
    a,b=lows[-2],lows[-1]; la,lb=float(p.low.iloc[a]),float(p.low.iloc[b]); neckline=float(p.high.iloc[a:b+1].max()); close=float(p.close.iloc[-1])
    similar=abs(la-lb)/max(la,.0001)<=.04; higher=lb>=la*.97
    if similar and close>neckline: return True,neckline,higher,"雙底完成，已收市突破頸線"
    if similar or higher: return False,neckline,higher,"有底部雛形，仍等待收市突破確認價"
    return False,neckline,False,"低點結構仍偏弱"


def regime(market):
    bm,vt=("SPY","^VIX") if market=="US" else ("^HSI","^VHSI")
    b,v=fetch_ohlcv(bm,"6mo"),fetch_ohlcv(vt,"6mo")
    if b is None or len(b)<60: return "unknown",0.0,np.nan,bm
    ret=(b.close.iloc[-1]/b.close.iloc[-60]-1)*100; vol=float(v.close.iloc[-1]) if v is not None and len(v) else np.nan
    if ret<-5 and pd.notna(vol) and vol>=25:return "bear_high_vol",ret,vol,bm
    if ret>5 and (pd.isna(vol) or vol<20):return "bull_low_vol",ret,vol,bm
    return "neutral",ret,vol,bm


def valuation(info):
    scores=[]; notes=[]
    pe,pb,ev,fcf=info["forward_pe"],info["pb"],info["ev_ebitda"],info["fcf_yield"]
    if pd.notna(pe) and pe>0: scores.append(90 if pe<10 else 75 if pe<15 else 55 if pe<22 else 25);notes.append(f"Forward PE {pe:.1f}")
    if pd.notna(pb) and pb>0:scores.append(85 if pb<1 else 65 if pb<2 else 35);notes.append(f"PB {pb:.2f}")
    if pd.notna(ev) and ev>0:scores.append(85 if ev<8 else 70 if ev<12 else 45 if ev<20 else 20);notes.append(f"EV/EBITDA {ev:.1f}")
    if pd.notna(fcf):scores.append(90 if fcf>8 else 70 if fcf>4 else 45 if fcf>0 else 10);notes.append(f"FCF Yield {fcf:.1f}%")
    return (float(np.mean(scores))," ｜ ".join(notes)) if scores else (np.nan,"資料不足")


def quality(info):
    score=50;notes=[];roe,debt,growth=info["roe"],info["debt_equity"],info["earnings_growth"]
    if pd.notna(roe):score+=20 if roe>.15 else 10 if roe>.08 else -15;notes.append(f"ROE {roe*100:.1f}%")
    if pd.notna(debt):score+=10 if debt<80 else 0 if debt<180 else -15;notes.append(f"D/E {debt:.0f}")
    if pd.notna(growth):score+=15 if growth>.10 else 5 if growth>0 else -15;notes.append(f"盈利增長 {growth*100:.1f}%")
    return float(np.clip(score,0,100))," ｜ ".join(notes) if notes else "資料不足"


def under_tier(score):
    if pd.isna(score):return "資料不足"
    if score>=80:return "嚴重低估"
    if score>=65:return "中度低估"
    if score>=50:return "輕度低估"
    return "估值不吸引"


def analyse_stock(ticker, df, market, market_regime):
    if df is None or len(df)<210:return None
    info=fetch_info(ticker);close=float(df.close.iloc[-1]);ma20=float(df.close.rolling(20).mean().iloc[-1]);ma200=float(df.close.rolling(200).mean().iloc[-1])
    drsi=float(rsi(df.close).iloc[-1]);wr=weekly_rsi(df);wrsi=float(wr.iloc[-1]) if len(wr) and pd.notna(wr.iloc[-1]) else np.nan
    ml,ms,mh=macd(df.close); c=cmf(df); cm=float(c.iloc[-1]) if pd.notna(c.iloc[-1]) else 0;vw=avwap(df);vr=float(df.volume.iloc[-1]/df.volume.rolling(20).mean().iloc[-1]);high52=float(df.high.iloc[-252:].max());dd=(close/high52-1)*100;at=float(atr(df).iloc[-1])
    confirmed,neck,higher,structure=bottom_structure(df);breakout=neck if pd.notna(neck) else max(ma20,float(df.high.iloc[-10:].max())); above=close>breakout; improving=ml.iloc[-1]>ms.iloc[-1] and mh.iloc[-1]>mh.iloc[-2]
    val,val_note=valuation(info);qual,qual_note=quality(info)
    cat=50+(20 if pd.notna(info["earnings_growth"]) and info["earnings_growth"]>.10 else -10 if pd.notna(info["earnings_growth"]) and info["earnings_growth"]<0 else 0)+(15 if cm>.05 else -10 if cm<-.10 else 0)+(15 if vr>=1.5 and close>ma20 else 0);cat=float(np.clip(cat,0,100))
    tech=(15 if drsi<35 else 8 if drsi<45 else 0)+(15 if pd.notna(wrsi) and 30<=wrsi<=55 else 0)+(20 if improving else 0)+(15 if close>ma20 and close>vw else 0)+(15 if higher else 0)+(20 if confirmed else 0);tech=float(np.clip(tech,0,100))
    risk=float(np.clip(60+(15 if close>ma200 else -20)+(10 if vr>=.7 else -25)+(5 if dd>-55 else -15),0,100))
    stop=max(min(float(df.low.iloc[-20:].min()),close-1.5*at),close-3*at)
    if stop>=close:stop=close-max(at,close*.03)
    per=max(close-stop,.0001);target=close+2*per;rr=(target-close)/per
    eligible=pd.notna(val) and risk>=40 and per/close<=.12 and vr>=.35
    if not eligible: action="不合資格"
    elif val>=60 and qual>=50 and tech<55:action="觀察"
    elif val>=60 and qual>=50 and tech>=55 and not above:action="等待突破"
    elif val>=60 and qual>=50 and tech>=60 and above and rr>=2:action="小量試倉"
    else:action="觀察"
    total=.30*(val if pd.notna(val) else 0)+.25*qual+.20*cat+.15*tech+.10*risk
    if market_regime=="bear_high_vol":total*=.90
    if market_regime=="bull_low_vol" and close<ma20:total*=.90
    reasons=[];missing=[]
    if pd.notna(val) and val>=65:reasons.append(f"{under_tier(val)}：估值分 {val:.0f}")
    elif pd.isna(val):missing.append("估值資料不足")
    if qual>=60:reasons.append(f"財務質素合格：{qual:.0f} 分")
    elif qual<50:missing.append("財務質素未達標")
    if drsi<35:reasons.append(f"日 RSI {drsi:.1f}，短期超賣")
    if higher:reasons.append("形成較高低點，賣壓可能減弱")
    if improving:reasons.append("MACD 動能改善")
    if confirmed:reasons.append("底部結構已突破確認")
    if not above:missing.append(f"未收市突破確認價 {breakout:.2f}")
    if vr<1.5:missing.append("成交量未達確認級別（1.5 倍均量）")
    if close<ma200:missing.append("仍低於 MA200，長期趨勢偏弱")
    snap={"valuation_note":val_note,"quality_note":qual_note,"structure":structure,"daily_rsi":round(drsi,2),"weekly_rsi":round(wrsi,2) if pd.notna(wrsi) else None,"cmf":round(cm,3),"volume_ratio":round(vr,2),"regime":market_regime,"reasons":reasons,"missing":missing}
    return {"ticker":ticker,"name":info["name"],"market":market,"event_time":datetime.now().isoformat(timespec="seconds"),"price":close,"total_score":round(float(total),1),"action":action,"undervaluation":under_tier(val),"valuation":round(val,1) if pd.notna(val) else np.nan,"quality":round(qual,1),"catalyst":round(cat,1),"technical":round(tech,1),"risk":round(risk,1),"daily_rsi":drsi,"weekly_rsi":wrsi,"cmf":cm,"avwap":vw,"vol_ratio":vr,"drawdown":dd,"breakout_price":breakout,"stop":stop,"target":target,"rr":rr,"structure":structure,"snapshot":snap,"df":df}


def stock_chart(r):
    df=r["df"].tail(250);ma20=df.close.rolling(20).mean();ma60=df.close.rolling(60).mean();ma200=df.close.rolling(200).mean();ml,ms,mh=macd(df.close)
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.56,.12,.16,.16],vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name="K線",increasing_line_color=C_GREEN,decreasing_line_color=C_RED),1,1)
    for s,n,c in [(ma20,"MA20",C_ORANGE),(ma60,"MA60",C_BLUE),(ma200,"MA200","#bc8cff")]:fig.add_trace(go.Scatter(x=df.index,y=s,name=n,line=dict(color=c,width=1.1)),1,1)
    fig.add_hline(y=r["avwap"],line_dash="dot",line_color=C_GREEN,annotation_text="AVWAP",row=1,col=1);fig.add_hline(y=r["breakout_price"],line_dash="dash",line_color=C_ORANGE,annotation_text="確認價",row=1,col=1);fig.add_hline(y=r["stop"],line_dash="dot",line_color=C_RED,annotation_text="止損",row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.volume,marker_color=[C_GREEN if c>=o else C_RED for c,o in zip(df.close,df.open)],name="成交量"),2,1)
    fig.add_trace(go.Scatter(x=df.index,y=rsi(df.close),name="RSI(14)",line=dict(color=C_ORANGE)),3,1);fig.add_hline(y=30,line_dash="dash",line_color=C_GREEN,row=3,col=1);fig.add_hline(y=70,line_dash="dash",line_color=C_RED,row=3,col=1)
    fig.add_trace(go.Bar(x=df.index,y=mh,name="MACD Hist",marker_color=[C_GREEN if x>=0 else C_RED for x in mh.fillna(0)]),4,1);fig.add_trace(go.Scatter(x=df.index,y=ml,name="MACD",line=dict(color=C_BLUE)),4,1);fig.add_trace(go.Scatter(x=df.index,y=ms,name="Signal",line=dict(color=C_ORANGE)),4,1)
    fig.update_layout(height=820,paper_bgcolor=C_BG,plot_bgcolor=C_BG,font=dict(color="#e6edf3"),xaxis_rangeslider_visible=False,legend=dict(orientation="h"),margin=dict(l=5,r=5,t=35,b=5));return fig


def card_class(action):return "good" if action=="小量試倉" else "wait" if action in ["觀察","等待突破"] else "bad"


def decision_card(r,prefix):
    reasons="<br>".join("✓ "+x for x in r["snapshot"]["reasons"]) or "—";missing="<br>".join("• "+x for x in r["snapshot"]["missing"]) or "—"
    st.markdown(f"""<div class="card {card_class(r['action'])}"><h3>{r['ticker']}　{r['action']}　|　候選分 {r['total_score']:.1f}</h3><p><b>現價：</b>{r['price']:.2f}　　<b>確認價：</b>{r['breakout_price']:.2f}　　<b>止損：</b>{r['stop']:.2f}　　<b>2R 目標：</b>{r['target']:.2f}</p><p><b>入選原因</b><br>{reasons}</p><p><b>仍欠條件／失效風險</b><br>{missing}</p></div>""",unsafe_allow_html=True)
    a,b,c,d,e=st.columns(5);a.metric("估值","資料不足" if pd.isna(r["valuation"]) else f"{r['valuation']:.0f}",help="Forward PE、PB、EV/EBITDA、FCF Yield 綜合分數。");b.metric("質素",f"{r['quality']:.0f}",help="ROE、負債及盈利增長。");c.metric("催化",f"{r['catalyst']:.0f}",help="盈利增長、CMF 資金流及量價配合。");d.metric("轉勢確認",f"{r['technical']:.0f}",help="超賣、MACD、AVWAP、較高低點及結構突破。");e.metric("R/R",f"{r['rr']:.2f}R",help="目標回報相對止損風險。")
    st.plotly_chart(stock_chart(r),use_container_width=True,key=f"{prefix}_chart_{r['ticker']}")

# --------------------------- Gann turning-window engine ---------------------------

def atr_zigzag(df, multiplier=2.0):
    """ATR reversal ZigZag. All pivots except the last row are confirmed pivots."""
    d=df.dropna().copy(); a=atr(d).bfill(); piv=[]
    if len(d)<30:return pd.DataFrame(columns=["date","price","kind","confirmed"])
    trend=0; high_price=float(d.high.iloc[0]); low_price=float(d.low.iloc[0]); high_date=d.index[0]; low_date=d.index[0]
    for i in range(1,len(d)):
        h,l,close=float(d.high.iloc[i]),float(d.low.iloc[i]),float(d.close.iloc[i]); threshold=max(float(a.iloc[i])*multiplier,close*.004)
        if trend>=0:
            if h>=high_price:high_price,high_date=h,d.index[i]
            if close<=high_price-threshold:
                piv.append((high_date,high_price,"HIGH",True));trend=-1;low_price,low_date=l,d.index[i]
        if trend<=0:
            if l<=low_price:low_price,low_date=l,d.index[i]
            if close>=low_price+threshold:
                piv.append((low_date,low_price,"LOW",True));trend=1;high_price,high_date=h,d.index[i]
    if trend>=0:piv.append((high_date,high_price,"HIGH",False))
    else:piv.append((low_date,low_price,"LOW",False))
    out=pd.DataFrame(piv,columns=["date","price","kind","confirmed"])
    return out.drop_duplicates(subset=["date","kind"],keep="last")


def gann_cycle_windows(anchor_date, as_of, periods=(20,30,45,60,90,120,180), half_window=3):
    rows=[]
    for p in periods:
        target=anchor_date+pd.offsets.BDay(p)
        while target<as_of-pd.offsets.BDay(half_window):target+=pd.offsets.BDay(p)
        distance=abs((target.normalize()-as_of.normalize()).days)
        rows.append({"cycle":p,"target":target.date(),"window_start":(target-pd.offsets.BDay(half_window)).date(),"window_end":(target+pd.offsets.BDay(half_window)).date(),"calendar_distance":distance})
    return pd.DataFrame(rows)


def gann_price_levels(high, low):
    diff=high-low; ratios=[.125,.25,.333,.5,.667,.75,.875]
    rows=[]
    for x in ratios:rows.append({"ratio":f"{x*100:.1f}%","price":low+diff*x})
    return pd.DataFrame(rows)


def gann_assessment(df,pivot_df):
    confirmed=pivot_df[pivot_df.confirmed].copy()
    if len(confirmed)<2:return None
    anchor=confirmed.iloc[-1]; prior=confirmed.iloc[-2]; now=df.index[-1]; close=float(df.close.iloc[-1]); atr_now=float(atr(df).iloc[-1])
    windows=gann_cycle_windows(pd.Timestamp(anchor.date),pd.Timestamp(now)); nearest=int(windows.calendar_distance.min()); time_score=40 if nearest<=3 else 25 if nearest<=7 else 5
    high=max(float(anchor.price),float(prior.price));low=min(float(anchor.price),float(prior.price));levels=gann_price_levels(high,low);nearest_level=float(levels.iloc[(levels.price-close).abs().argsort().iloc[0]].price);price_near=abs(close-nearest_level)<=max(.75*atr_now,close*.005); price_score=30 if price_near else 5
    ml,ms,mh=macd(df.close); rs=float(rsi(df.close).iloc[-1]); last_kind=anchor.kind
    expected="上行波段" if last_kind=="LOW" else "下行波段"
    # A current leg after a confirmed LOW is assessed for an upward exhaustion window; the opposite after HIGH.
    if last_kind=="LOW":confirmed_price=(rs>50 and ml.iloc[-1]<ms.iloc[-1]) or (df.close.iloc[-1]<df.close.rolling(20).mean().iloc[-1])
    else:confirmed_price=(rs<50 and ml.iloc[-1]>ms.iloc[-1]) or (df.close.iloc[-1]>df.close.rolling(20).mean().iloc[-1])
    confirmation_score=30 if confirmed_price else 5; score=time_score+price_score+confirmation_score
    label="高注意轉勢窗" if score>=75 else "中注意轉勢窗" if score>=50 else "低注意／暫無共振"
    return {"anchor":anchor,"prior":prior,"windows":windows,"levels":levels,"close":close,"atr":atr_now,"nearest_level":nearest_level,"price_near":price_near,"nearest_days":nearest,"time_score":time_score,"price_score":price_score,"confirmation_score":confirmation_score,"score":score,"label":label,"expected":expected,"rsi":rs,"macd_confirmation":confirmed_price}


def gann_chart(df,pivot_df,assessment,show_days=260):
    d=df.tail(show_days);fig=go.Figure();fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name="指數",increasing_line_color=C_GREEN,decreasing_line_color=C_RED))
    p=pivot_df[pivot_df.date>=d.index[0]]
    if not p.empty:
        colors=[C_RED if x=="HIGH" else C_GREEN for x in p.kind];symbols=["x" if c else "circle-open" for c in p.confirmed]
        fig.add_trace(go.Scatter(x=p.date,y=p.price,mode="lines+markers",name="ATR ZigZag",line=dict(color=C_BLUE,width=2),marker=dict(size=9,color=colors,symbol=symbols)))
    levels=assessment["levels"]
    for _,row in levels.iterrows():fig.add_hline(y=row.price,line_dash="dot",line_color="rgba(139,148,158,.45)",annotation_text=row.ratio,annotation_font_color=C_GREY)
    anchor=assessment["anchor"];direction=1 if anchor.kind=="LOW" else -1;ref_atr=float(atr(df).iloc[-20:].median());future=pd.bdate_range(d.index[-1],periods=61)[1:]
    for mult,name,color in [(2,"2x1",C_RED),(1,"1x1",C_ORANGE),(.5,"1x2",C_GREEN)]:
        days=np.arange(1,len(future)+1);base=float(d.close.iloc[-1]);path=base+direction*ref_atr*mult*days
        fig.add_trace(go.Scatter(x=future,y=path,mode="lines",name=f"ATR 江恩速度 {name}",line=dict(color=color,dash="dash",width=1)))
    fig.update_layout(height=650,paper_bgcolor=C_BG,plot_bgcolor=C_BG,font=dict(color="#e6edf3"),xaxis_rangeslider_visible=False,legend=dict(orientation="h"),margin=dict(l=5,r=5,t=35,b=5));return fig


init_db()
st.title("📈 撈底監察系統 V4.2 — 決策＋江恩轉勢時間窗")
st.caption("股票頁以估值、質素、催化、轉勢確認及風險找候選；江恩頁只標示值得提高警覺的轉勢時間窗，不預言必然轉向。")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    market_label=st.radio("市場",["🇺🇸 美股","🇭🇰 港股","📋 自選"],key="market_choice")
    custom=st.text_area("自選代碼（每行一個）","AAPL\nNVDA\n0700.HK",key="custom_symbols") if market_label=="📋 自選" else ""
    account=st.number_input("帳戶總值",min_value=1000.0,value=100000.0,step=1000.0);risk_pct=st.slider("每筆最大風險 (%)",.25,2.0,1.0,.25)/100;show_all=st.checkbox("候選頁顯示所有分析股",False)
    st.divider();st.markdown("### 決策定義");st.caption("不合資格：資料、風險或流動性不符。\n\n觀察：便宜或超賣，未確認。\n\n等待突破：有底部條件，未破確認價。\n\n小量試倉：確認突破、止損清晰且 R/R ≥ 2。")

market="US" if "美股" in market_label else "HK";tickers=US_WATCHLIST if market_label=="🇺🇸 美股" else HK_WATCHLIST if market_label=="🇭🇰 港股" else [x.strip().upper() for x in custom.splitlines() if x.strip()]
reg,br,vol,bm=regime(market);reg_text={"bear_high_vol":"熊市高波動","bull_low_vol":"牛市低波動","neutral":"中性","unknown":"未知"}[reg]
a,b,c,d=st.columns(4);a.metric("市場","美股" if market=="US" else "港股");b.metric("市場環境",reg_text);c.metric(f"{bm} 60 日回報",f"{br:.1f}%");d.metric("VIX" if market=="US" else "VHSI","資料不足" if pd.isna(vol) else f"{vol:.1f}")

tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs(["🎯 今日候選","📊 全部掃描","📈 技術圖表","📐 交易計劃","🔮 江恩轉勢窗","📋 訊號紀錄","📖 指標教學"])

with tab1:
    if st.button("🔄 掃描目前名單",type="primary",key="run_scan"):
        with st.spinner(f"正在分析 {len(tickers)} 隻股票..."):
            dm=fetch_many(tickers);res=[analyse_stock(t,dm.get(t),market,reg) for t in tickers];st.session_state["results_v42"]=[x for x in res if x]
    results=st.session_state.get("results_v42",[])
    if not results:st.info("按「掃描目前名單」開始。系統預設只展示等待突破及小量試倉候選。")
    else:
        visible=results if show_all else [x for x in results if x["action"] in ["等待突破","小量試倉"]];counts=pd.Series([x["action"] for x in results]).value_counts();st.caption(f"已分析 {len(results)} 隻｜小量試倉 {counts.get('小量試倉',0)}｜等待突破 {counts.get('等待突破',0)}｜觀察 {counts.get('觀察',0)}｜不合資格 {counts.get('不合資格',0)}")
        if not visible:st.warning("目前沒有值得處理的候選。這代表系統沒有強行給買入訊號。")
        for r in sorted(visible,key=lambda x:(x["action"]!="小量試倉",-x["total_score"])):
            decision_card(r,"candidate")
            if r["action"]=="小量試倉" and st.button(f"保存 {r['ticker']} 試倉訊號",key=f"save_{r['ticker']}"):
                save_signal(r);st.success(f"已保存 {r['ticker']} 訊號快照。")

with tab2:
    results=st.session_state.get("results_v42",[])
    if results:
        rows=[{"代碼":r["ticker"],"名稱":r["name"],"現價":round(r["price"],2),"決策":r["action"],"候選分":r["total_score"],"低估級別":r["undervaluation"],"估值":r["valuation"],"質素":r["quality"],"催化":r["catalyst"],"轉勢":r["technical"],"風險":r["risk"],"日RSI":round(r["daily_rsi"],1),"真周RSI":round(r["weekly_rsi"],1) if pd.notna(r["weekly_rsi"]) else np.nan,"量比":round(r["vol_ratio"],2),"距52周高%":round(r["drawdown"],1),"確認價":round(r["breakout_price"],2),"R/R":round(r["rr"],2)} for r in results]
        table=pd.DataFrame(rows);st.dataframe(table.sort_values("候選分",ascending=False),use_container_width=True,hide_index=True);st.download_button("下載 CSV",table.to_csv(index=False).encode("utf-8-sig"),"v42_scan_results.csv","text/csv",key="csv_scan")
    else:st.info("請先完成掃描。")

with tab3:
    results=st.session_state.get("results_v42",[])
    if results:
        selected=st.selectbox("選擇股票",[x["ticker"] for x in results],key="tech_choice");decision_card(next(x for x in results if x["ticker"]==selected),"technical_tab")
    else:st.info("請先完成掃描。")

with tab4:
    tk=st.text_input("股票代碼","AAPL",key="plan_ticker").upper()
    if st.button("建立交易計劃",key="make_plan"):
        p=analyse_stock(tk,fetch_ohlcv(tk),"HK" if tk.endswith(".HK") else "US",reg)
        if not p:st.error("資料不足，無法建立交易計劃。")
        else:
            cashrisk=account*risk_pct;per=max(p["price"]-p["stop"],.0001);shares=min(int(cashrisk/per),int(account/p["price"]));st.markdown(f"## {tk} — {p['action']}");x1,x2,x3,x4,x5=st.columns(5);x1.metric("參考現價",f"{p['price']:.2f}");x2.metric("確認買入價",f"{p['breakout_price']:.2f}");x3.metric("結構止損",f"{p['stop']:.2f}");x4.metric("2R 目標",f"{p['target']:.2f}");x5.metric("最大股數",f"{shares:,}");st.write(f"每股風險：{per:.2f} ｜ 單筆最大風險：{cashrisk:,.2f} ｜ 最大資金使用：約 {shares*p['price']:,.2f}");st.warning("港股下單前請按每手股數、貨幣換算及實際交易成本向下調整。")

with tab5:
    st.subheader("🔮 江恩轉勢時間窗 — 指數預警")
    st.caption("此頁輸出的是『提高警覺的時間窗』，不是預言。所有轉勢候選仍須以價格結構、動能及成交量確認。")
    gc1,gc2,gc3=st.columns([1.4,1,1]);index_name=gc1.selectbox("分析指數",list(INDEXES.keys()),key="gann_index");period=gc2.selectbox("資料週期",["2y","3y","5y"],index=1,key="gann_period");mult=gc3.slider("ATR ZigZag 敏感度",1.0,4.0,2.0,.25,key="gann_mult",help="越高只保留越大型的確認轉折；越低則較敏感但雜訊更多。")
    if st.button("執行江恩轉勢分析",type="primary",key="run_gann"):
        gdf=fetch_ohlcv(INDEXES[index_name],period);st.session_state["gann_df"]=gdf;st.session_state["gann_name"]=index_name;st.session_state["gann_mult_saved"]=mult
    gdf=st.session_state.get("gann_df")
    if gdf is None:st.info("選擇指數後按「執行江恩轉勢分析」。")
    else:
        piv=atr_zigzag(gdf,st.session_state.get("gann_mult_saved",mult));ass=gann_assessment(gdf,piv)
        if ass is None:st.warning("未能找到足夠確認波段；請增加資料週期或調低 ATR ZigZag 敏感度。")
        else:
            tone="good" if ass["score"]>=75 else "wait" if ass["score"]>=50 else "bad";anchor=ass["anchor"]
            st.markdown(f"""<div class="card {tone}"><h3>{st.session_state['gann_name']}：{ass['label']}　|　江恩共振分 {ass['score']}/100</h3><p><b>目前確認波段：</b>{ass['expected']} ｜ <b>最近確認樞紐：</b>{anchor.kind} {anchor.price:.2f}（{pd.Timestamp(anchor.date).date()}）</p><p><b>時間窗距離：</b>{ass['nearest_days']} 個日曆日 ｜ <b>最近價格共振位：</b>{ass['nearest_level']:.2f} ｜ <b>技術確認：</b>{'已出現初步反向確認' if ass['macd_confirmation'] else '尚未確認'}</p></div>""",unsafe_allow_html=True)
            y1,y2,y3,y4=st.columns(4);y1.metric("時間共振",f"{ass['time_score']}/40",help="距離固定江恩時間窗（20、30、45、60、90、120、180 個交易日）的接近程度。");y2.metric("價格共振",f"{ass['price_score']}/30",help="價格是否接近前一段高低點的比例回撤位。");y3.metric("反向確認",f"{ass['confirmation_score']}/30",help="以 RSI、MACD 與 MA20 尋找初步反向行為。");y4.metric("日 RSI",f"{ass['rsi']:.1f}")
            st.plotly_chart(gann_chart(gdf,piv,ass),use_container_width=True,key=f"gann_chart_{INDEXES[index_name]}_{period}_{mult}")
            left,right=st.columns(2)
            with left:
                st.markdown("### 下一組江恩時間窗")
                display=ass["windows"].copy();display.columns=["週期（交易日）","中心日期","觀察開始","觀察結束","距今天日數"];st.dataframe(display,use_container_width=True,hide_index=True)
            with right:
                st.markdown("### 前一主波段價格比例")
                levels=ass["levels"].copy();levels.columns=["比例","價格位"];levels["價格位"]=levels["價格位"].round(2);st.dataframe(levels,use_container_width=True,hide_index=True)
            with st.expander("如何閱讀江恩頁面？"):
                st.markdown("""1. **先看時間窗**：接近 20、30、45、60、90、120、180 個交易日的循環日，只表示要提高留意，不表示必定轉勢。\n\n2. **再看價格共振**：指數同時接近前一波段的 25%、33.3%、50%、66.7% 或 75% 比例位時，可信度提高。\n\n3. **最後看確認**：上行轉弱要看跌破 MA20、MACD 轉弱或 RSI 轉低；下行轉強要看站回 MA20、MACD 改善或 RSI 回升。\n\n4. **ATR 江恩速度線**：以 ATR 代替畫面上的固定 45 度，顯示目前波段可承受的每日價格速度。它是動態支撐／阻力參考，不是神秘預測線。\n\n5. **最有用的訊號**：高注意時間窗 + 價格共振 + 價格確認，三者同時出現才值得調整指數 ETF 或總倉位。""")

with tab6:
    events=load_signals()
    if events.empty:st.info("尚未有保存的試倉訊號。只有「小量試倉」候選才可保存。")
    else:st.dataframe(events.drop(columns=["snapshot_json"]),use_container_width=True,hide_index=True);st.caption("系統保存當刻分數、價格、確認價、止損、目標及原因快照。累積足夠每日訊號後，可再建立含下一日入場、成本與基準比較的 walk-forward 回測。")

with tab7:
    st.markdown("## 指標應按次序使用，而不是獨立下單")
    guide=pd.DataFrame([
        ["估值","公司相對盈利、資產及現金流是否偏便宜","決定是否值得研究；低估不代表立即買。"],
        ["質素","ROE、負債及盈利趨勢是否健康","排除價值陷阱。"],
        ["催化","盈利、資金流及量價是否開始改善","解釋市場為何可能改變看法。"],
        ["日 RSI","最近跌勢或升勢是否過急","低於 30 只代表超賣，不能當買入訊號。"],
        ["真周 RSI","中期強弱狀態","避免只看日線而過早撈長期弱勢股。"],
        ["MACD","短期動能相對中期動能的改變","低位改善可支持跌勢減弱，但仍要等突破。"],
        ["AVWAP","由重要低點起計的成交加權平均成本","站上 AVWAP 代表近期低位資金可能開始轉為獲利。"],
        ["Higher Low","第二個低點未明顯低於第一個低點","較可靠的賣壓減弱線索。"],
        ["確認價／頸線","底部中間反彈高位或近期阻力","收市突破才由『觀察』升級為『小量試倉』候選。"],
        ["止損與 2R","錯了在哪離場、對了可賺多少","只做可預先定義風險且回報至少兩倍風險的交易。"],
        ["江恩時間窗","由確認轉折點起計的固定交易日週期","只作提高警覺的時間區間，必須配合價格確認。"],
    ],columns=["概念","白話意思","正確使用方式"])
    st.dataframe(guide,use_container_width=True,hide_index=True)
    st.info("正確流程：合資格 → 估值與質素 → 催化 → 轉勢確認 → 止損、倉位及 R/R。任何單一指標都不應單獨決定買入。")
