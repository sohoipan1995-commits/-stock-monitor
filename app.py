import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="撈底監察系統 V4.4", page_icon="📈", layout="wide")

DB = Path("stock_monitor_v44.sqlite")
CORE_US = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","QQQ","SPY"]
CORE_HK = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","9988.HK","3690.HK","9618.HK","1211.HK","0981.HK","9999.HK"]
HK_LIQUID_POOL = ["0700.HK","0005.HK","0939.HK","1398.HK","3988.HK","0388.HK","0066.HK","0003.HK","0002.HK","0016.HK","0883.HK","2318.HK","1299.HK","0001.HK","9988.HK","0175.HK","3690.HK","9618.HK","0981.HK","9999.HK","2382.HK","1211.HK","0267.HK","2688.HK","0762.HK","6862.HK","0960.HK","2020.HK","1810.HK","1024.HK"]
C = {"bg":"#0d1117","panel":"#161b22","border":"#30363d","green":"#3fb950","red":"#f85149","orange":"#d29922","blue":"#58a6ff","grey":"#8b949e"}

st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background:{C['bg']};}} [data-testid="stSidebar"]{{background:{C['panel']};}}
h1,h2,h3,p,label,.stMarkdown{{color:#e6edf3!important;}}
.card{{background:{C['panel']};border:1px solid {C['border']};border-left:5px solid {C['blue']};border-radius:10px;padding:14px;margin:8px 0;}}
.good{{border-left-color:{C['green']};}} .wait{{border-left-color:{C['orange']};}} .bad{{border-left-color:{C['red']};}}
</style>""", unsafe_allow_html=True)


def db_init():
    with sqlite3.connect(DB) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots(
          id TEXT PRIMARY KEY,snapshot_time TEXT,ticker TEXT,market TEXT,source TEXT,data_time TEXT,
          price REAL,action TEXT,total REAL,valuation REAL,quality REAL,catalyst REAL,technical REAL,risk REAL,
          breakout REAL,stop REAL,target REAL,sector TEXT,relative_strength REAL,gann_level TEXT,gann_date TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS positions(
          ticker TEXT PRIMARY KEY,market TEXT,sector TEXT,currency TEXT,shares REAL,cost REAL,stop REAL,target REAL,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS errors(log_time TEXT,module TEXT,ticker TEXT,error TEXT);
        """)


def log_error(module, ticker, error):
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO errors VALUES(?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), module, ticker, str(error)[:500]))


@st.cache_data(ttl=900, show_spinner=False)
def ohlcv(ticker, period="2y"):
    try:
        d = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if d is None or d.empty: return None
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.columns = [str(x).lower() for x in d.columns]
        return d[["open","high","low","close","volume"]].dropna()
    except Exception as e:
        log_error("ohlcv", ticker, e); return None


@st.cache_data(ttl=3600, show_spinner=False)
def info(ticker):
    z={"name":ticker,"sector":"Unknown","pe":np.nan,"pb":np.nan,"ev":np.nan,"fcf":np.nan,"roe":np.nan,"de":np.nan,"growth":np.nan}
    try:
        x=yf.Ticker(ticker).info; cap,fcf=x.get("marketCap"),x.get("freeCashflow")
        z.update({"name":x.get("shortName") or ticker,"sector":x.get("sector") or "Unknown","pe":x.get("forwardPE",np.nan),"pb":x.get("priceToBook",np.nan),"ev":x.get("enterpriseToEbitda",np.nan),"fcf":fcf/cap*100 if fcf and cap else np.nan,"roe":x.get("returnOnEquity",np.nan),"de":x.get("debtToEquity",np.nan),"growth":x.get("earningsGrowth",np.nan)})
    except Exception as e: log_error("fundamentals",ticker,e)
    return z


@st.cache_data(ttl=86400, show_spinner=False)
def sp500_universe():
    try:
        tables=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].astype(str).str.replace(".","-",regex=False).tolist()
    except Exception as e:
        log_error("sp500_universe","S&P500",e); return CORE_US


def indicators(d):
    delta=d.close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rsi=100-100/(1+gain/loss.replace(0,np.nan))
    mac=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean(); sig=mac.ewm(span=9,adjust=False).mean()
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean()
    multiplier=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan); cmf=(multiplier*d.volume).rolling(20).sum()/d.volume.rolling(20).sum().replace(0,np.nan)
    return rsi,mac,sig,mac-sig,atr,cmf


def benchmark(market): return "SPY" if market=="US" else "^HSI"


def sector_relative_strength(d, market):
    b=ohlcv(benchmark(market),"6mo")
    if b is None or len(d)<21 or len(b)<21:return np.nan
    return float((d.close.iloc[-1]/d.close.iloc[-21]-1)-(b.close.iloc[-1]/b.close.iloc[-21]-1))*100


def gann_window(d, mult=2.0):
    a=indicators(d)[4].bfill();trend=0;hp,lp=float(d.high.iloc[0]),float(d.low.iloc[0]);hd,ld=d.index[0],d.index[0];p=[]
    for i in range(1,len(d)):
        h,l,cl=float(d.high.iloc[i]),float(d.low.iloc[i]),float(d.close.iloc[i]);th=max(float(a.iloc[i])*mult,cl*.004)
        if trend>=0:
            if h>=hp:hp,hd=h,d.index[i]
            if cl<=hp-th:p.append((hd,hp,"HIGH",True));trend=-1;lp,ld=l,d.index[i]
        if trend<=0:
            if l<=lp:lp,ld=l,d.index[i]
            if cl>=lp+th:p.append((ld,lp,"LOW",True));trend=1;hp,hd=h,d.index[i]
    confirmed=pd.DataFrame(p,columns=["date","price","kind","confirmed"])
    if len(confirmed)<2:return "無",None,0
    anchor=confirmed.iloc[-1];today=d.index[-1];windows=[]
    for cycle,level in [(20,"輕度"),(30,"輕度"),(45,"輕度"),(60,"中度"),(90,"中度"),(120,"重要"),(144,"重要"),(180,"重要")]:
        target=pd.Timestamp(anchor.date)+pd.offsets.BDay(cycle)
        while target<today-pd.offsets.BDay(3):target+=pd.offsets.BDay(cycle)
        windows.append((target,level,cycle))
    target,level,cycle=min(windows,key=lambda x:abs((x[0]-today).days))
    return level,target.date(),cycle


def valuation_percentile_proxy(d, f):
    """Price/earnings proxy: clearly labeled proxy when true historical PE is unavailable."""
    if pd.isna(f["pe"]) or f["pe"]<=0:return np.nan
    trailing_price_percentile=(d.close.iloc[-1]>=d.close).mean()*100
    return float(trailing_price_percentile)


def analyze(ticker, d, market):
    if d is None or len(d)<210:return None
    f=info(ticker);price=float(d.close.iloc[-1]);rsi,mac,sig,hist,atr,cmf=indicators(d);ma20=float(d.close.rolling(20).mean().iloc[-1]);ma200=float(d.close.rolling(200).mean().iloc[-1]);weekly=float((100-100/(1+(d.close.resample("W-FRI").last().dropna().diff().clip(lower=0).ewm(alpha=1/14,adjust=False).mean()/(-d.close.resample("W-FRI").last().dropna().diff().clip(upper=0)).ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))).iloc[-1])
    vol=float(d.volume.iloc[-1]/d.volume.rolling(20).mean().iloc[-1]);rs=sector_relative_strength(d,market);high52=float(d.high.iloc[-252:].max());dd=(price/high52-1)*100
    scores=[]
    for value,rule in [(f['pe'],lambda x:90 if x<10 else 75 if x<15 else 55 if x<22 else 25),(f['pb'],lambda x:85 if x<1 else 65 if x<2 else 35),(f['ev'],lambda x:85 if x<8 else 70 if x<12 else 45),(f['fcf'],lambda x:90 if x>8 else 70 if x>4 else 45 if x>0 else 10)]:
        if pd.notna(value) and value>0:scores.append(rule(value))
    val=float(np.mean(scores)) if scores else np.nan
    q=float(np.clip(50+(20 if pd.notna(f['roe']) and f['roe']>.15 else 10 if pd.notna(f['roe']) and f['roe']>.08 else -15 if pd.notna(f['roe']) else 0)+(10 if pd.notna(f['de']) and f['de']<80 else -15 if pd.notna(f['de']) and f['de']>180 else 0)+(15 if pd.notna(f['growth']) and f['growth']>.1 else -15 if pd.notna(f['growth']) and f['growth']<0 else 0),0,100))
    cat=float(np.clip(50+(20 if pd.notna(f['growth']) and f['growth']>.1 else -10 if pd.notna(f['growth']) and f['growth']<0 else 0)+(15 if cmf.iloc[-1]>.05 else -10 if cmf.iloc[-1]<-.1 else 0)+(15 if vol>=1.5 and price>ma20 else 0)+(10 if pd.notna(rs) and rs>0 else 0),0,100))
    low20=float(d.low.iloc[-20:].min());breakout=float(d.high.iloc[-10:].max());improve=mac.iloc[-1]>sig.iloc[-1] and hist.iloc[-1]>hist.iloc[-2]
    tech=float(np.clip((15 if rsi.iloc[-1]<35 else 8 if rsi.iloc[-1]<45 else 0)+(15 if 30<=weekly<=55 else 0)+(20 if improve else 0)+(15 if price>ma20 else 0)+(15 if pd.notna(rs) and rs>0 else 0),0,100))
    risk=float(np.clip(60+(15 if price>ma200 else -20)+(10 if vol>=.7 else -25)+(5 if dd>-55 else -15),0,100));stop=max(min(low20,price-1.5*float(atr.iloc[-1])),price-3*float(atr.iloc[-1]));target=price+2*(price-stop);rr=(target-price)/max(price-stop,.0001)
    action="不合資格" if pd.isna(val) or risk<40 or (price-stop)/price>.12 else "觀察" if tech<55 else "等待突破" if price<=breakout else "小量試倉" if q>=50 and rr>=2 else "觀察"
    total=round(float(.30*(val if pd.notna(val) else 0)+.25*q+.20*cat+.15*tech+.10*risk),1);g_level,g_date,g_cycle=gann_window(d);tier="資料不足" if pd.isna(val) else "嚴重低估" if val>=80 else "中度低估" if val>=65 else "輕度低估" if val>=50 else "估值不吸引"
    reasons=[x for x in [f"{tier}（{val:.0f}）" if pd.notna(val) and val>=65 else None,"相對強弱正數" if pd.notna(rs) and rs>0 else None,"MACD 改善" if improve else None,f"日RSI {rsi.iloc[-1]:.1f}" if rsi.iloc[-1]<35 else None] if x];missing=[x for x in [f"未突破 {breakout:.2f}" if price<=breakout else None,"量能不足" if vol<1.5 else None,"低於MA200" if price<ma200 else None] if x]
    return {"ticker":ticker,"market":market,"name":f['name'],"sector":f['sector'],"price":price,"action":action,"total":total,"valuation":val,"quality":q,"catalyst":cat,"technical":tech,"risk":risk,"tier":tier,"rs":rs,"valuation_proxy":valuation_percentile_proxy(d,f),"breakout":breakout,"stop":stop,"target":target,"rr":rr,"gann_level":g_level,"gann_date":g_date,"gann_cycle":g_cycle,"df":d,"snapshot":{"reasons":reasons,"missing":missing,"rsi":round(float(rsi.iloc[-1]),1),"weekly_rsi":round(weekly,1),"cmf":round(float(cmf.iloc[-1]),3),"volume_ratio":round(vol,2)}}


def save_snapshots(results):
    now=datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB) as con:
        for r in results:
            key=f"{now[:10]}_{r['ticker']}"
            con.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(key,now,r['ticker'],r['market'],"Yahoo Finance",r['df'].index[-1].isoformat(),r['price'],r['action'],r['total'],r['valuation'],r['quality'],r['catalyst'],r['technical'],r['risk'],r['breakout'],r['stop'],r['target'],r['sector'],r['rs'],r['gann_level'],str(r['gann_date']),json.dumps(r['snapshot'],ensure_ascii=False)))


def performance_backtest():
    with sqlite3.connect(DB) as con: snap=pd.read_sql_query("SELECT * FROM snapshots WHERE action='小量試倉' ORDER BY snapshot_time",con)
    rows=[]
    for _,x in snap.iterrows():
        d=ohlcv(x.ticker,"2y");entry_date=pd.Timestamp(x.data_time).tz_localize(None) if pd.Timestamp(x.data_time).tzinfo else pd.Timestamp(x.data_time)
        if d is None:continue
        future=d[d.index>entry_date].head(21)
        if len(future)<5:continue
        entry=float(future.open.iloc[0]);stop=float(x.stop);target=float(x.target);exit_price=float(future.close.iloc[-1]);reason="20日到期"
        for _,bar in future.iloc[1:].iterrows():
            if bar.low<=stop:exit_price=stop;reason="止損";break
            if bar.high>=target:exit_price=target;reason="2R目標";break
        cost=.0025;ret=(exit_price/entry-1-cost)*100;rows.append({"代碼":x.ticker,"訊號日":x.snapshot_time[:10],"入場":round(entry,2),"出場":round(exit_price,2),"原因":reason,"回報%":round(ret,2),"R":round((exit_price-entry)/max(entry-stop,.0001),2),"江恩":x.gann_level})
    return pd.DataFrame(rows)


def positions_table():
    with sqlite3.connect(DB) as con:return pd.read_sql_query("SELECT * FROM positions",con)


def chart(r):
    d=r['df'].tail(200);fig=go.Figure();fig.add_trace(go.Candlestick(x=d.index,open=d.open,high=d.high,low=d.low,close=d.close,name='K線',increasing_line_color=C['green'],decreasing_line_color=C['red']));fig.add_trace(go.Scatter(x=d.index,y=d.close.rolling(20).mean(),name='MA20',line=dict(color=C['orange'])));fig.add_trace(go.Scatter(x=d.index,y=d.close.rolling(200).mean(),name='MA200',line=dict(color='#bc8cff')))
    for y,n,c in [(r['breakout'],'確認價',C['orange']),(r['stop'],'止損',C['red']),(r['target'],'2R目標',C['green'])]:fig.add_hline(y=y,line_dash='dash',line_color=c,annotation_text=n)
    fig.update_layout(height=580,paper_bgcolor=C['bg'],plot_bgcolor=C['bg'],font=dict(color='#e6edf3'),xaxis_rangeslider_visible=False);return fig


db_init();st.title('📈 撈底監察系統 V4.4 — 驗證與組合風控版')
with st.sidebar:
    mode=st.radio('顯示模式',['新手模式','進階模式']);market_label=st.radio('市場',['🇺🇸 美股','🇭🇰 港股','📋 自選']);custom=st.text_area('自選代碼（每行一個）','AAPL\nNVDA\n0700.HK') if market_label=='📋 自選' else '';universe_size=st.slider('自動股票池掃描數量',10,60,30,10);account=st.number_input('帳戶總值',1000.0,100000.0,step=1000.0);risk_pct=st.slider('每筆風險%',.25,2.0,1.0,.25)/100
market='US' if '美股' in market_label else 'HK'
if market_label=='🇺🇸 美股': universe=list(dict.fromkeys(CORE_US+sp500_universe()))[:universe_size]
elif market_label=='🇭🇰 港股': universe=HK_LIQUID_POOL[:universe_size]
else: universe=[x.strip().upper() for x in custom.splitlines() if x.strip()]

tabs=st.tabs(['🎯 今日決策','📊 全市場掃描','🏭 行業與估值','🧪 策略驗證','💼 持倉與風控','⚙️ 資料健康'])
with tabs[0]:
    if st.button('🔄 執行掃描並保存每日快照',type='primary'):
        with st.spinner(f'掃描 {len(universe)} 隻股票...'):
            data={};
            with ThreadPoolExecutor(max_workers=8) as ex:
                jobs={ex.submit(ohlcv,t):t for t in universe}
                for j in as_completed(jobs):data[jobs[j]]=j.result()
            results=[x for x in [analyze(t,data.get(t),market) for t in universe] if x];st.session_state['v44']=results;save_snapshots(results)
    results=st.session_state.get('v44',[])
    if not results:st.info('掃描後會保存每日 feature snapshot，供回測使用。')
    else:
        show=[r for r in results if r['action'] in ['等待突破','小量試倉']]
        for r in sorted(show,key=lambda x:(x['action']!='小量試倉',-x['total'])):
            reasons='<br>'.join('✓ '+x for x in r['snapshot']['reasons']) or '—';missing='<br>'.join('• '+x for x in r['snapshot']['missing']) or '—';cls='good' if r['action']=='小量試倉' else 'wait'
            st.markdown(f"<div class='card {cls}'><h3>{r['ticker']}　{r['action']}　|　候選分 {r['total']}</h3><p>現價 {r['price']:.2f}｜確認價 {r['breakout']:.2f}｜止損 {r['stop']:.2f}｜2R目標 {r['target']:.2f}</p><p><b>原因</b><br>{reasons}</p><p><b>下一步</b><br>{missing}</p></div>",unsafe_allow_html=True)
            if mode=='進階模式':st.caption(f"行業：{r['sector']}｜相對{benchmark(market)}強弱：{r['rs']:.2f}%｜江恩：{r['gann_level']} {r['gann_date']}")
with tabs[1]:
    results=st.session_state.get('v44',[])
    if results:
        table=pd.DataFrame([{k:r[k] for k in ['ticker','name','sector','price','action','total','tier','valuation','quality','catalyst','technical','risk','rs','gann_level','gann_date']} for r in results]);table.columns=['代碼','名稱','行業','現價','決策','總分','低估級別','估值','質素','催化','轉勢','風險','相對強弱%','江恩級別','江恩日期'];st.dataframe(table.sort_values('總分',ascending=False),use_container_width=True,hide_index=True);pick=st.selectbox('查看圖表',table['代碼']);st.plotly_chart(chart(next(x for x in results if x['ticker']==pick)),use_container_width=True,key=f'chart_{pick}')
    else:st.info('請先掃描。')
with tabs[2]:
    results=st.session_state.get('v44',[])
    if results:
        t=pd.DataFrame([{ '行業':r['sector'],'代碼':r['ticker'],'估值':r['valuation'],'估值歷史價格百分位代理':r['valuation_proxy'],'相對強弱%':r['rs'],'低估級別':r['tier']} for r in results]);st.dataframe(t.sort_values('相對強弱%',ascending=False),use_container_width=True,hide_index=True);st.caption('「估值歷史價格百分位代理」不是歷史 PE；真正歷史 PE 需要 point-in-time 財報資料累積後才會啟用。')
    else:st.info('請先掃描。')
with tabs[3]:
    bt=performance_backtest()
    if bt.empty:st.info('需要累積已保存的「小量試倉」歷史快照，以及其後至少 20 個交易日資料。')
    else:
        a,b,c,d=st.columns(4);a.metric('交易數',len(bt));b.metric('勝率',f"{(bt['回報%']>0).mean()*100:.1f}%");c.metric('平均回報',f"{bt['回報%'].mean():.2f}%");d.metric('平均R',f"{bt['R'].mean():.2f}");st.dataframe(bt,use_container_width=True,hide_index=True);st.dataframe(bt.groupby('江恩').agg(交易數=('R','count'),平均R=('R','mean'),勝率=('回報%',lambda x:(x>0).mean())).reset_index(),use_container_width=True,hide_index=True)
with tabs[4]:
    pos=positions_table();st.caption('輸入持倉後按儲存。總風險以「現價至止損」計算。')
    edit=st.data_editor(pos if not pos.empty else pd.DataFrame(columns=['ticker','market','sector','currency','shares','cost','stop','target','updated_at']),num_rows='dynamic',use_container_width=True,key='positions_editor')
    if st.button('儲存持倉'):
        edit['updated_at']=datetime.now().isoformat(timespec='seconds')
        with sqlite3.connect(DB) as con:
            con.execute('DELETE FROM positions');edit.to_sql('positions',con,if_exists='append',index=False)
        st.success('已儲存。')
    if not pos.empty:
        current=[]
        for _,p in pos.iterrows():
            d=ohlcv(p.ticker,'6mo');price=float(d.close.iloc[-1]) if d is not None else p.cost;current.append({**p.to_dict(),'current':price,'risk_cash':max(price-float(p.stop),0)*float(p.shares),'market_value':price*float(p.shares)})
        cp=pd.DataFrame(current);a,b,c=st.columns(3);a.metric('持倉市值',f"{cp.market_value.sum():,.0f}");b.metric('總止損風險',f"{cp.risk_cash.sum():,.0f} ({cp.risk_cash.sum()/account*100:.2f}%)");c.metric('最大行業曝險',cp.groupby('sector').market_value.sum().idxmax());st.dataframe(cp[['ticker','sector','shares','cost','current','stop','target','risk_cash','market_value']],use_container_width=True,hide_index=True)
with tabs[5]:
    with sqlite3.connect(DB) as con:err=pd.read_sql_query('SELECT * FROM errors ORDER BY log_time DESC LIMIT 100',con);snap=pd.read_sql_query('SELECT source,data_time,MAX(snapshot_time) AS last_scan,COUNT(*) AS rows FROM snapshots GROUP BY source,data_time',con)
    st.subheader('資料更新與完整度');st.dataframe(snap,use_container_width=True,hide_index=True);st.subheader('錯誤日誌');st.dataframe(err,use_container_width=True,hide_index=True)
