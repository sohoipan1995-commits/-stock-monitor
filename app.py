import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='撈底監察系統 V4.4', page_icon='📈', layout='wide')
DB = Path('stock_monitor_v44.sqlite')
US = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','AMD','QCOM','JPM','SPY','QQQ']
HK = ['0700.HK','0005.HK','0939.HK','1398.HK','3988.HK','0388.HK','9988.HK','3690.HK','9618.HK','0981.HK','1211.HK']

st.markdown("""<style>[data-testid='stAppViewContainer']{background:#0d1117}[data-testid='stSidebar']{background:#161b22}h1,h2,h3,p,label,.stMarkdown{color:#e6edf3!important}.card{background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:14px;margin:8px 0}</style>""", unsafe_allow_html=True)

def migrate(con, table, columns):
    exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if exists:
        actual = [x[1] for x in con.execute(f'PRAGMA table_info({table})').fetchall()]
        if actual != columns:
            con.execute(f'DROP TABLE IF EXISTS {table}_legacy')
            con.execute(f'ALTER TABLE {table} RENAME TO {table}_legacy')

def init_db():
    cols = ['id','ts','ticker','market','price','action','total','valuation','quality','catalyst','technical','risk','breakout','stop','target','sector','rs','gann_level','gann_date','payload']
    with sqlite3.connect(DB) as con:
        migrate(con, 'snapshots', cols)
        con.execute('''CREATE TABLE IF NOT EXISTS snapshots(
            id TEXT PRIMARY KEY,ts TEXT,ticker TEXT,market TEXT,price REAL,action TEXT,total REAL,
            valuation REAL,quality REAL,catalyst REAL,technical REAL,risk REAL,breakout REAL,stop REAL,
            target REAL,sector TEXT,rs REAL,gann_level TEXT,gann_date TEXT,payload TEXT)''')
        con.execute('''CREATE TABLE IF NOT EXISTS positions(
            ticker TEXT PRIMARY KEY,market TEXT,sector TEXT,currency TEXT,shares REAL,cost REAL,stop REAL,target REAL,updated TEXT)''')
        con.execute('''CREATE TABLE IF NOT EXISTS errors(ts TEXT,module TEXT,ticker TEXT,error TEXT)''')

def log_error(module, ticker, error):
    with sqlite3.connect(DB) as con:
        con.execute('INSERT INTO errors VALUES(?,?,?,?)',(datetime.now().isoformat(timespec='seconds'),module,ticker,str(error)[:400]))

@st.cache_data(ttl=900, show_spinner=False)
def fetch(ticker, period='2y'):
    try:
        d = yf.download(ticker,period=period,interval='1d',auto_adjust=True,progress=False)
        if d is None or d.empty:return None
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        d.columns=[str(x).lower() for x in d.columns]
        return d[['open','high','low','close','volume']].dropna()
    except Exception as e:
        log_error('fetch',ticker,e);return None

@st.cache_data(ttl=3600, show_spinner=False)
def meta(ticker):
    x={'name':ticker,'sector':'Unknown','pe':np.nan,'pb':np.nan,'roe':np.nan,'de':np.nan,'growth':np.nan}
    try:
        i=yf.Ticker(ticker).info
        x.update({'name':i.get('shortName') or ticker,'sector':i.get('sector') or 'Unknown','pe':i.get('forwardPE',np.nan),'pb':i.get('priceToBook',np.nan),'roe':i.get('returnOnEquity',np.nan),'de':i.get('debtToEquity',np.nan),'growth':i.get('earningsGrowth',np.nan)})
    except Exception as e:log_error('meta',ticker,e)
    return x

def rsi(s,n=14):
    delta=s.diff();gain=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();loss=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))

def analyse(ticker,d,market):
    if d is None or len(d)<210:return None
    m=meta(ticker);price=float(d.close.iloc[-1]);daily=float(rsi(d.close).iloc[-1]);weekly=float(rsi(d.close.resample('W-FRI').last().dropna()).iloc[-1]);ma20=float(d.close.rolling(20).mean().iloc[-1]);ma200=float(d.close.rolling(200).mean().iloc[-1]);vol=float(d.volume.iloc[-1]/d.volume.rolling(20).mean().iloc[-1]);mac=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean();sig=mac.ewm(span=9,adjust=False).mean();improve=mac.iloc[-1]>sig.iloc[-1]
    val=np.mean([90 if m['pe']<10 else 75 if m['pe']<15 else 55 if m['pe']<22 else 25 for _ in [0] if pd.notna(m['pe']) and m['pe']>0]) if pd.notna(m['pe']) and m['pe']>0 else np.nan
    quality=float(np.clip(50+(20 if pd.notna(m['roe']) and m['roe']>.15 else -10 if pd.notna(m['roe']) else 0)+(10 if pd.notna(m['de']) and m['de']<80 else -15 if pd.notna(m['de']) and m['de']>180 else 0),0,100));catalyst=float(np.clip(50+(20 if pd.notna(m['growth']) and m['growth']>.1 else -10 if pd.notna(m['growth']) and m['growth']<0 else 0)+(15 if vol>=1.5 else 0),0,100));technical=float(np.clip((15 if daily<35 else 8 if daily<45 else 0)+(15 if 30<=weekly<=55 else 0)+(20 if improve else 0)+(15 if price>ma20 else 0),0,100));risk=float(np.clip(60+(15 if price>ma200 else -20)+(10 if vol>=.7 else -25),0,100))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1);atr=float(tr.rolling(14).mean().iloc[-1]);breakout=float(d.high.iloc[-10:].max());stop=max(float(d.low.iloc[-20:].min()),price-2*atr);target=price+2*(price-stop);action='不合資格' if pd.isna(val) or risk<40 else '觀察' if technical<55 else '等待突破' if price<=breakout else '小量試倉';total=round(float(.3*(val if pd.notna(val) else 0)+.25*quality+.2*catalyst+.15*technical+.1*risk),1)
    gann='重要' if len(d)%144<4 else '中度' if len(d)%60<4 else '輕度';gann_date=(d.index[-1]+pd.offsets.BDay(3)).date()
    return {'ticker':ticker,'market':market,'name':m['name'],'sector':m['sector'],'price':price,'action':action,'total':total,'valuation':val,'quality':quality,'catalyst':catalyst,'technical':technical,'risk':risk,'breakout':breakout,'stop':stop,'target':target,'rs':np.nan,'gann_level':gann,'gann_date':str(gann_date),'snapshot':{'daily_rsi':round(daily,1),'weekly_rsi':round(weekly,1),'volume_ratio':round(vol,2)}}

def persist(results):
    now=datetime.now().isoformat(timespec='seconds')
    sql='''INSERT OR REPLACE INTO snapshots(id,ts,ticker,market,price,action,total,valuation,quality,catalyst,technical,risk,breakout,stop,target,sector,rs,gann_level,gann_date,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
    with sqlite3.connect(DB) as con:
        for r in results:
            con.execute(sql,(f"{now[:10]}_{r['ticker']}",now,r['ticker'],r['market'],r['price'],r['action'],r['total'],r['valuation'],r['quality'],r['catalyst'],r['technical'],r['risk'],r['breakout'],r['stop'],r['target'],r['sector'],r['rs'],r['gann_level'],r['gann_date'],json.dumps(r['snapshot'],ensure_ascii=False)))

def walk_forward():
    with sqlite3.connect(DB) as con:s=pd.read_sql_query("SELECT * FROM snapshots WHERE action='小量試倉'",con)
    rows=[]
    for _,x in s.iterrows():
        d=fetch(x.ticker,'2y')
        if d is None:continue
        future=d[d.index>pd.Timestamp(x.ts[:10])].head(21)
        if len(future)<5:continue
        entry=float(future.open.iloc[0]);out=float(future.close.iloc[-1]);why='20日到期'
        for _,bar in future.iloc[1:].iterrows():
            if bar.low<=x.stop:out=float(x.stop);why='止損';break
            if bar.high>=x.target:out=float(x.target);why='2R目標';break
        rows.append({'代碼':x.ticker,'入場':entry,'出場':out,'原因':why,'回報%':round((out/entry-1-.0025)*100,2),'R':round((out-entry)/max(entry-x.stop,.0001),2),'江恩':x.gann_level})
    return pd.DataFrame(rows)

init_db();st.title('📈 撈底監察系統 V4.4 — SQLite 修正版')
with st.sidebar:
    choice=st.radio('市場',['美股','港股','自選']);custom=st.text_area('自選代碼','AAPL\nNVDA\n0700.HK') if choice=='自選' else '';account=st.number_input('帳戶總值',1000.0,100000.0,step=1000.0)
universe=US if choice=='美股' else HK if choice=='港股' else [x.strip().upper() for x in custom.splitlines() if x.strip()];market='US' if choice=='美股' else 'HK'
t1,t2,t3,t4=st.tabs(['🎯 每日決策','📊 掃描結果','🧪 Walk-Forward 回測','⚙️ 資料健康'])
with t1:
    if st.button('掃描並保存每日快照',type='primary'):
        with st.spinner('分析中...'):
            with ThreadPoolExecutor(max_workers=8) as ex:
                jobs={ex.submit(fetch,t):t for t in universe};data={jobs[j]:j.result() for j in as_completed(jobs)}
            st.session_state['results']=[r for r in [analyse(t,data.get(t),market) for t in universe] if r];persist(st.session_state['results'])
    results=st.session_state.get('results',[])
    for r in [x for x in results if x['action'] in ['等待突破','小量試倉']]:
        st.markdown(f"<div class='card'><h3>{r['ticker']}｜{r['action']}｜總分 {r['total']}</h3><p>現價 {r['price']:.2f}｜確認價 {r['breakout']:.2f}｜止損 {r['stop']:.2f}｜2R目標 {r['target']:.2f}</p></div>",unsafe_allow_html=True)
with t2:
    results=st.session_state.get('results',[])
    if results:st.dataframe(pd.DataFrame(results).drop(columns=['snapshot'],errors='ignore'),use_container_width=True,hide_index=True)
    else:st.info('請先掃描。')
with t3:
    bt=walk_forward()
    if bt.empty:st.info('需累積小量試倉訊號及之後至少 20 個交易日。')
    else:st.dataframe(bt,use_container_width=True,hide_index=True);st.metric('平均 R',f"{bt['R'].mean():.2f}")
with t4:
    with sqlite3.connect(DB) as con:
        st.dataframe(pd.read_sql_query('SELECT MAX(ts) AS 最後掃描,COUNT(*) AS 快照數 FROM snapshots',con),use_container_width=True,hide_index=True);st.dataframe(pd.read_sql_query('SELECT * FROM errors ORDER BY ts DESC LIMIT 100',con),use_container_width=True,hide_index=True)
