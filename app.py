import sqlite3, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='撈底監察系統 V4.4',page_icon='📈',layout='wide')
DB=Path('stock_monitor_v44.sqlite')
US_CORE=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','QQQ','SPY']
HK_POOL=['0700.HK','0005.HK','0939.HK','1398.HK','3988.HK','0388.HK','0066.HK','0883.HK','2318.HK','1299.HK','9988.HK','0175.HK','3690.HK','9618.HK','0981.HK','9999.HK','1211.HK','2688.HK','0762.HK','1810.HK','1024.HK','2020.HK']

st.markdown("""<style>[data-testid='stAppViewContainer']{background:#0d1117}[data-testid='stSidebar']{background:#161b22}h1,h2,h3,p,label,.stMarkdown{color:#e6edf3!important}.card{background:#161b22;border:1px solid #30363d;border-left:5px solid #58a6ff;border-radius:10px;padding:14px;margin:8px 0}.good{border-left-color:#3fb950}.wait{border-left-color:#d29922}</style>""",unsafe_allow_html=True)

def init_db():
 with sqlite3.connect(DB) as c:
  c.executescript('''CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY,ts TEXT,ticker TEXT,market TEXT,price REAL,action TEXT,total REAL,val REAL,quality REAL,catalyst REAL,technical REAL,risk REAL,breakout REAL,stop REAL,target REAL,sector TEXT,rs REAL,gann TEXT,payload TEXT);CREATE TABLE IF NOT EXISTS positions(ticker TEXT PRIMARY KEY,market TEXT,sector TEXT,currency TEXT,shares REAL,cost REAL,stop REAL,target REAL,updated TEXT);CREATE TABLE IF NOT EXISTS errors(ts TEXT,module TEXT,ticker TEXT,error TEXT);''')
def err(mod,t,e):
 with sqlite3.connect(DB) as c:c.execute('INSERT INTO errors VALUES(?,?,?,?)',(datetime.now().isoformat(),mod,t,str(e)[:400]))
@st.cache_data(ttl=900,show_spinner=False)
def px(t,period='2y'):
 try:
  d=yf.download(t,period=period,interval='1d',auto_adjust=True,progress=False)
  if d is None or d.empty:return None
  if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
  d.columns=[str(x).lower() for x in d.columns];return d[['open','high','low','close','volume']].dropna()
 except Exception as e:err('prices',t,e);return None
@st.cache_data(ttl=3600,show_spinner=False)
def fin(t):
 o={'name':t,'sector':'Unknown','pe':np.nan,'pb':np.nan,'ev':np.nan,'fcf':np.nan,'roe':np.nan,'de':np.nan,'growth':np.nan}
 try:
  i=yf.Ticker(t).info;cap,fcf=i.get('marketCap'),i.get('freeCashflow');o.update({'name':i.get('shortName') or t,'sector':i.get('sector') or 'Unknown','pe':i.get('forwardPE',np.nan),'pb':i.get('priceToBook',np.nan),'ev':i.get('enterpriseToEbitda',np.nan),'fcf':fcf/cap*100 if fcf and cap else np.nan,'roe':i.get('returnOnEquity',np.nan),'de':i.get('debtToEquity',np.nan),'growth':i.get('earningsGrowth',np.nan)})
 except Exception as e:err('fundamentals',t,e)
 return o
@st.cache_data(ttl=86400,show_spinner=False)
def sp500():
 try:return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].astype(str).str.replace('.','-',regex=False).tolist()
 except Exception:return US_CORE
def rsi(s,n=14):
 d=s.diff();g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+g/l.replace(0,np.nan))
def ind(d):
 m=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean();sig=m.ewm(span=9,adjust=False).mean();tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1);atr=tr.rolling(14).mean();mult=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan);cmf=(mult*d.volume).rolling(20).sum()/d.volume.rolling(20).sum().replace(0,np.nan);return rsi(d.close),m,sig,m-sig,atr,cmf
def weekly_rsi(d):
 w=d.close.resample('W-FRI').last().dropna();return float(rsi(w,14).iloc[-1])
def relative(d,market):
 b=px('SPY' if market=='US' else '^HSI','6mo')
 return np.nan if b is None or len(d)<21 or len(b)<21 else float((d.close.iloc[-1]/d.close.iloc[-21]-1-(b.close.iloc[-1]/b.close.iloc[-21]-1))*100)
def gann(d):
 a=ind(d)[4].bfill();trend=0;hp,lp=float(d.high.iloc[0]),float(d.low.iloc[0]);hd,ld=d.index[0],d.index[0];p=[]
 for i in range(1,len(d)):
  h,l,cl=float(d.high.iloc[i]),float(d.low.iloc[i]),float(d.close.iloc[i]);th=max(float(a.iloc[i])*2,cl*.004)
  if trend>=0:
   if h>=hp:hp,hd=h,d.index[i]
   if cl<=hp-th:p.append((hd,hp,'HIGH'));trend=-1;lp,ld=l,d.index[i]
  if trend<=0:
   if l<=lp:lp,ld=l,d.index[i]
   if cl>=lp+th:p.append((ld,lp,'LOW'));trend=1;hp,hd=h,d.index[i]
 if not p:return '無',None
 anchor=p[-1][0];today=d.index[-1];choices=[]
 for days,level in [(20,'輕度'),(30,'輕度'),(45,'輕度'),(60,'中度'),(90,'中度'),(120,'重要'),(144,'重要'),(180,'重要')]:
  x=anchor+pd.offsets.BDay(days)
  while x<today-pd.offsets.BDay(3):x+=pd.offsets.BDay(days)
  choices.append((x,level))
 x,level=min(choices,key=lambda z:abs((z[0]-today).days));return level,x.date()
def analyse(t,d,market):
 if d is None or len(d)<210:return None
 f=fin(t);price=float(d.close.iloc[-1]);daily,mac,sig,hist,atr,cmf=ind(d);weekly=weekly_rsi(d);ma20=float(d.close.rolling(20).mean().iloc[-1]);ma200=float(d.close.rolling(200).mean().iloc[-1]);vol=float(d.volume.iloc[-1]/d.volume.rolling(20).mean().iloc[-1]);rs=relative(d,market);high=float(d.high.iloc[-252:].max());dd=(price/high-1)*100
 vals=[]
 for v,fun in [(f['pe'],lambda x:90 if x<10 else 75 if x<15 else 55 if x<22 else 25),(f['pb'],lambda x:85 if x<1 else 65 if x<2 else 35),(f['ev'],lambda x:85 if x<8 else 70 if x<12 else 45),(f['fcf'],lambda x:90 if x>8 else 70 if x>4 else 45 if x>0 else 10)]:
  if pd.notna(v) and v>0:vals.append(fun(v))
 val=float(np.mean(vals)) if vals else np.nan
 q=float(np.clip(50+(20 if pd.notna(f['roe']) and f['roe']>.15 else 10 if pd.notna(f['roe']) and f['roe']>.08 else -15 if pd.notna(f['roe']) else 0)+(10 if pd.notna(f['de']) and f['de']<80 else -15 if pd.notna(f['de']) and f['de']>180 else 0)+(15 if pd.notna(f['growth']) and f['growth']>.1 else -15 if pd.notna(f['growth']) and f['growth']<0 else 0),0,100))
 cat=float(np.clip(50+(20 if pd.notna(f['growth']) and f['growth']>.1 else -10 if pd.notna(f['growth']) and f['growth']<0 else 0)+(15 if cmf.iloc[-1]>.05 else -10 if cmf.iloc[-1]<-.1 else 0)+(15 if vol>=1.5 and price>ma20 else 0)+(10 if pd.notna(rs) and rs>0 else 0),0,100));improve=mac.iloc[-1]>sig.iloc[-1] and hist.iloc[-1]>hist.iloc[-2];tech=float(np.clip((15 if daily.iloc[-1]<35 else 8 if daily.iloc[-1]<45 else 0)+(15 if 30<=weekly<=55 else 0)+(20 if improve else 0)+(15 if price>ma20 else 0)+(15 if pd.notna(rs) and rs>0 else 0),0,100));risk=float(np.clip(60+(15 if price>ma200 else -20)+(10 if vol>=.7 else -25)+(5 if dd>-55 else -15),0,100));breakout=float(d.high.iloc[-10:].max());stop=max(min(float(d.low.iloc[-20:].min()),price-1.5*float(atr.iloc[-1])),price-3*float(atr.iloc[-1]));target=price+2*(price-stop);rr=(target-price)/max(price-stop,.0001);action='不合資格' if pd.isna(val) or risk<40 or (price-stop)/price>.12 else '觀察' if tech<55 else '等待突破' if price<=breakout else '小量試倉' if q>=50 else '觀察';total=round(float(.3*(val if pd.notna(val) else 0)+.25*q+.2*cat+.15*tech+.1*risk),1);level,date=gann(d);tier='資料不足' if pd.isna(val) else '嚴重低估' if val>=80 else '中度低估' if val>=65 else '輕度低估' if val>=50 else '估值不吸引';reasons=[x for x in [f'{tier}（{val:.0f}）' if pd.notna(val) and val>=65 else None,'相對強弱正數' if pd.notna(rs) and rs>0 else None,'MACD改善' if improve else None] if x];missing=[x for x in [f'未突破 {breakout:.2f}' if price<=breakout else None,'量能不足' if vol<1.5 else None,'低於MA200' if price<ma200 else None] if x]
 return {'ticker':t,'market':market,'name':f['name'],'sector':f['sector'],'price':price,'action':action,'total':total,'valuation':val,'quality':q,'catalyst':cat,'technical':tech,'risk':risk,'tier':tier,'rs':rs,'breakout':breakout,'stop':stop,'target':target,'rr':rr,'gann':level,'gann_date':date,'df':d,'snapshot':{'reasons':reasons,'missing':missing,'daily_rsi':round(float(daily.iloc[-1]),1),'weekly_rsi':round(weekly,1),'cmf':round(float(cmf.iloc[-1]),3)}}
def persist(results):
 now=datetime.now().isoformat(timespec='seconds')
 with sqlite3.connect(DB) as c:
  for r in results:c.execute('INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(f"{now[:10]}_{r['ticker']}",now,r['ticker'],r['market'],r['price'],r['action'],r['total'],r['valuation'],r['quality'],r['catalyst'],r['technical'],r['risk'],r['breakout'],r['stop'],r['target'],r['sector'],r['rs'],r['gann'],json.dumps(r['snapshot'],ensure_ascii=False)))
def backtest():
 with sqlite3.connect(DB) as c:s=pd.read_sql_query("SELECT * FROM snapshots WHERE action='小量試倉'",c)
 rows=[]
 for _,x in s.iterrows():
  d=px(x.ticker,'2y');date=pd.Timestamp(x.ts[:10])
  if d is None:continue
  f=d[d.index>date].head(21)
  if len(f)<5:continue
  entry=float(f.open.iloc[0]);exit=float(f.close.iloc[-1]);why='20日';
  for _,b in f.iloc[1:].iterrows():
   if b.low<=x.stop:exit=float(x.stop);why='止損';break
   if b.high>=x.target:exit=float(x.target);why='2R目標';break
  rows.append({'代碼':x.ticker,'入場':entry,'出場':exit,'原因':why,'回報%':round((exit/entry-1-.0025)*100,2),'R':round((exit-entry)/max(entry-x.stop,.0001),2),'江恩':x.gann})
 return pd.DataFrame(rows)

init_db();st.title('📈 撈底監察系統 V4.4 — 驗證與組合風控版')
with st.sidebar:
 mode=st.radio('模式',['新手模式','進階模式']);ml=st.radio('市場',['🇺🇸 美股','🇭🇰 港股','📋 自選']);custom=st.text_area('自選代碼','AAPL\nNVDA\n0700.HK') if ml=='📋 自選' else '';n=st.slider('掃描數量',10,60,30,10);account=st.number_input('帳戶總值',1000.0,100000.0,step=1000.0);rp=st.slider('每筆風險%',.25,2.0,1.0,.25)/100
market='US' if '美股' in ml else 'HK';universe=(list(dict.fromkeys(US_CORE+sp500()))[:n] if ml=='🇺🇸 美股' else HK_POOL[:n] if ml=='🇭🇰 港股' else [x.strip().upper() for x in custom.splitlines() if x.strip()])
tabs=st.tabs(['🎯 今日決策','📊 掃描與行業','🧪 回測與江恩','💼 持倉風控','⚙️ 資料健康'])
with tabs[0]:
 if st.button('🔄 掃描並保存每日快照',type='primary'):
  with st.spinner(f'分析 {len(universe)} 隻股票...'):
   with ThreadPoolExecutor(max_workers=8) as ex:
    jobs={ex.submit(px,t):t for t in universe};data={jobs[j]:j.result() for j in as_completed(jobs)}
   st.session_state['v44']=[r for r in [analyse(t,data.get(t),market) for t in universe] if r];persist(st.session_state['v44'])
 results=st.session_state.get('v44',[])
 if not results:st.info('掃描後系統會保存每日 feature snapshot，以建立可回測資料。')
 for r in [x for x in results if x['action'] in ['等待突破','小量試倉']]:
  cls='good' if r['action']=='小量試倉' else 'wait';rs='<br>'.join('✓ '+x for x in r['snapshot']['reasons']) or '—';ms='<br>'.join('• '+x for x in r['snapshot']['missing']) or '—';st.markdown(f"<div class='card {cls}'><h3>{r['ticker']}　{r['action']}｜{r['total']}</h3><p>現價 {r['price']:.2f}｜確認價 {r['breakout']:.2f}｜止損 {r['stop']:.2f}｜2R {r['target']:.2f}</p><p>{rs}<br>{ms}</p></div>",unsafe_allow_html=True)
with tabs[1]:
 results=st.session_state.get('v44',[])
 if results:
  t=pd.DataFrame([{k:r[k] for k in ['ticker','name','sector','price','action','total','tier','valuation','quality','catalyst','technical','risk','rs','gann','gann_date']} for r in results]);t.columns=['代碼','名稱','行業','現價','決策','總分','低估','估值','質素','催化','轉勢','風險','相對強弱%','江恩','江恩日期'];st.dataframe(t.sort_values('總分',ascending=False),use_container_width=True,hide_index=True)
 else:st.info('請先掃描。')
with tabs[2]:
 b=backtest()
 if b.empty:st.info('需累積小量試倉訊號及其後至少 20 個交易日。')
 else:
  a,c,d,e=st.columns(4);a.metric('交易數',len(b));c.metric('勝率',f"{(b['回報%']>0).mean()*100:.1f}%");d.metric('平均R',f"{b['R'].mean():.2f}");e.metric('平均回報',f"{b['回報%'].mean():.2f}%");st.dataframe(b,use_container_width=True,hide_index=True);st.dataframe(b.groupby('江恩').agg(交易數=('R','count'),平均R=('R','mean'),勝率=('回報%',lambda x:(x>0).mean())).reset_index(),use_container_width=True,hide_index=True)
with tabs[3]:
 with sqlite3.connect(DB) as c:p=pd.read_sql_query('SELECT * FROM positions',c)
 edit=st.data_editor(p if not p.empty else pd.DataFrame(columns=['ticker','market','sector','currency','shares','cost','stop','target','updated']),num_rows='dynamic',use_container_width=True)
 if st.button('儲存持倉'):
  edit['updated']=datetime.now().isoformat(timespec='seconds')
  with sqlite3.connect(DB) as c:c.execute('DELETE FROM positions');edit.to_sql('positions',c,if_exists='append',index=False)
  st.success('已儲存。')
 if not p.empty:
  p['止損風險']=np.maximum(p['cost']-p['stop'],0)*p['shares'];a,b,c=st.columns(3);a.metric('持倉市值',f"{(p['cost']*p['shares']).sum():,.0f}");b.metric('總止損風險',f"{p['止損風險'].sum():,.0f} ({p['止損風險'].sum()/account*100:.2f}%)");c.metric('最大行業曝險',p.groupby('sector').apply(lambda x:(x.cost*x.shares).sum()).idxmax());st.dataframe(p,use_container_width=True,hide_index=True)
with tabs[4]:
 with sqlite3.connect(DB) as c:errors=pd.read_sql_query('SELECT * FROM errors ORDER BY ts DESC LIMIT 100',c);snaps=pd.read_sql_query('SELECT MAX(ts) last_scan,COUNT(*) snapshots FROM snapshots',c)
 st.dataframe(snaps,use_container_width=True,hide_index=True);st.dataframe(errors,use_container_width=True,hide_index=True)
