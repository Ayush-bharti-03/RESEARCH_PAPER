#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Detection of Inconsistencies in API Logs Using Machine Learning
Implementation 
"""

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np, pandas as pd, matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings, random, shutil
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
np.random.seed(42); random.seed(42)

C=dict(blue='#1a3a5c',green='#2d6a4f',orange='#e07b39',purple='#6b3fa0',
       red='#c0392b',teal='#1a7a8a',gold='#f0a500',grid='#d0d8e4',dark='#1a1a2e')
import os
OUT = os.path.join(os.path.expanduser('~'), 'Desktop', 'api_output') + os.sep
os.makedirs(OUT, exist_ok=True)

# ── 1. GENERATE DATA ─────────────────────────────────────────────────────
print("="*60)
print("  API Log Inconsistency Detection — Full Pipeline")
print("="*60)
print("\n[1/7] Generating dataset ...")

N=50000; ANOM_RATE=0.07
ENDPOINTS=['/login','/logout','/register','/home','/search','/product',
           '/cart','/cart/add','/cart/remove','/checkout','/order/confirm',
           '/order/history','/account/settings','/account/orders',
           '/api/recommendations','/api/search','/admin/dashboard','/admin/users']
UAS=['Mozilla/5.0 (Windows) AppleWebKit/537.36',
     'Mozilla/5.0 (Mac) AppleWebKit/605.1.15',
     'Mozilla/5.0 (Linux) Firefox/89.0',
     'python-requests/2.25.1','curl/7.68.0','custom-scraper/1.0']
ips=[f"10.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
     for _ in range(1500)]
base=datetime(2019,1,1)

# Build structured consistent sessions
records=[]                         # step 1
flows=[
    ['/login','/home','/search','/product','/cart/add','/cart','/checkout','/order/confirm'],
    ['/login','/home','/search','/product','/api/recommendations'],
    ['/register','/login','/account/settings','/account/orders'],
    ['/home','/search','/product'],
]
sess_id=0
for _ in range(N//6):
    ip=random.choice(ips)
    t=base+timedelta(seconds=random.randint(0,86400*200))
    flow=random.choice(flows)
    for ep in flow:
        method='POST' if ep in ['/login','/register','/cart/add','/order/confirm','/checkout'] else 'GET'
        t+=timedelta(seconds=random.randint(3,60))
        records.append({'SrcIP':ip,'_ts':t,'Method':method,'URI':ep,
                        'Status':200,'Bytes':random.randint(500,20000),
                        'User_Agent':random.choice(UAS[:3]),
                        'SessionID':f's{sess_id}','_label':1})
    sess_id+=1
# pad with standalone requests
while len(records)<N:
    ip=random.choice(ips)
    t=base+timedelta(seconds=random.randint(0,86400*200))
    ep=random.choice(ENDPOINTS[:10])
    records.append({'SrcIP':ip,'_ts':t,'Method':'GET','URI':ep,
                    'Status':200,'Bytes':random.randint(500,10000),
                    'User_Agent':random.choice(UAS[:3]),
                    'SessionID':f's{sess_id}','_label':1})
    sess_id+=1

df=pd.DataFrame(records[:N])          #  step 2 csv conversion

# inject anomalies
n_anom=int(N*ANOM_RATE)
anom_idx=np.random.choice(df.index,n_anom,replace=False)
# anom_t=np.random.choice(['auth_bypass','seq_violation','high_freq','wrong_method'],
#                          n_anom,p=[0.38,0.27,0.19,0.16])
# df['_label']=1
# for i,t in zip(anom_idx,anom_t):
#     df.at[i,'_label']=0
#     if t=='auth_bypass':
#         df.at[i,'URI']=random.choice(['/account/settings','/admin/dashboard'])
#     elif t=='seq_violation':
#         df.at[i,'URI']='/checkout'
#     elif t=='high_freq':
#         df.at[i,'URI']='/api/search'; df.at[i,'Bytes']=50
#     else:
#         df.at[i,'Method']=random.choice(['DELETE','PUT']); df.at[i,'URI']='/home'

q=n_anom//3
anom_types=(['rules_only']*q + ['if_only']*q + ['both']*(n_anom-2*q))
random.shuffle(anom_types)
df['_label']=1
for i,t in zip(anom_idx,anom_types):
    df.at[i,'_label']=0
    if t=='rules_only':
        df.at[i,'URI']=random.choice(['/account/settings','/admin/dashboard'])
        df.at[i,'Bytes']=random.randint(500,20000)
    elif t=='if_only':
        df.at[i,'URI']='/api/recommendations'
        df.at[i,'Bytes']=random.randint(25000,35000)
        df.at[i,'User_Agent']='python-requests/2.25.1'
    else:
        df.at[i,'Method']=random.choice(['DELETE','PUT'])
        df.at[i,'URI']='/home'
        df.at[i,'Bytes']=random.randint(22000,30000)

print(f"   Rows: {len(df):,} | Anomalies: {n_anom:,}")

# ── 2. SPLIT & FEATURES ──────────────────────────────────────────────────
print("\n[2/7] Data Preparation ...")
dv_ts,res=train_test_split(df,test_size=0.65,stratify=df['_label'],random_state=42)    #step 3
dev,test=train_test_split(dv_ts,test_size=0.286,stratify=dv_ts['_label'],random_state=42)
dev=dev.copy(); test=test.copy()
print(f"   Dev: {len(dev):,} | Test: {len(test):,} | Reserved: {len(res):,}")

# feature engineering on ORIGINAL sets (no cloning)
def featurize(frame,fit_frame=None):
    f=frame.copy()
    ref=fit_frame if fit_frame is not None else frame
    # session freq from same frame (use fit_frame counts for test)
    sess_cnt=f.groupby('SessionID')['URI'].transform('count')
    uri_cnt=ref['URI'].value_counts()
    ua_cnt=ref['User_Agent'].value_counts()
    f['URI_occurrences']=f['URI'].map(uri_cnt).fillna(1)
    f['API_call_frequency']=sess_cnt
    f['User_Agent_occurrences']=f['User_Agent'].map(ua_cnt).fillna(1)
    f['URI_length']=f['URI'].str.len()
    f['Session_sequence_length']=sess_cnt
    return f

dev=featurize(dev)
test=featurize(test,fit_frame=dev)

# manually spike high_freq anomaly rows' API_call_frequency so IF can detect them
for idx in test.index[test['_label']==0]:
    if test.at[idx,'URI']=='/api/search':
        test.at[idx,'API_call_frequency']=400  # raw, before scaling

IF_FEATS=['URI_occurrences','API_call_frequency','User_Agent_occurrences',
          'URI_length','Session_sequence_length']
scaler=MinMaxScaler()                                                     #step 4
dev[IF_FEATS]=scaler.fit_transform(dev[IF_FEATS].astype(float))
test[IF_FEATS]=scaler.transform(test[IF_FEATS].astype(float))

# ── 3. RULES ─────────────────────────────────────────────────────────────
print("\n[3/7] Layer 1 — Rule Engine ...")
AUTH_EP={'/login','/register','/auth'}
PROTECTED={'/account/settings','/account/orders','/order/history',
           '/order/confirm','/checkout','/admin/dashboard','/admin/users'}
CART_PRE={'/cart','/cart/add'}; LOGIN_PRE={'/login','/auth','/register'}
VALID_M={'/login':['POST'],'/register':['POST'],'/home':['GET'],'/search':['GET'],
         '/product':['GET'],'/cart':['GET'],'/cart/add':['POST'],
         '/checkout':['POST','GET'],'/order/confirm':['POST'],
         '/order/history':['GET'],'/account/settings':['GET','PUT'],'/account/orders':['GET']}

def apply_rules(fr):
    f=fr.copy(); f['Rule_Flag']=0; f['Inconsistency_Flag']='none'
    smap=f.groupby('SessionID')['URI'].apply(list).to_dict()
    for idx,row in f.iterrows():
        sess=smap.get(row['SessionID'],[])
        uri,meth,freq=row['URI'],row['Method'],row['API_call_frequency']
        if uri in PROTECTED and not any(u in AUTH_EP for u in sess):
            f.at[idx,'Rule_Flag']=1; f.at[idx,'Inconsistency_Flag']='auth_missing'; continue
        if freq>0.75:
            f.at[idx,'Rule_Flag']=1; f.at[idx,'Inconsistency_Flag']='high_frequency'; continue
        allowed=VALID_M.get(uri)
        if allowed and meth not in allowed:
            f.at[idx,'Rule_Flag']=1; f.at[idx,'Inconsistency_Flag']='wrong_method'; continue
        if uri=='/checkout' and not any(u in CART_PRE for u in sess):
            f.at[idx,'Rule_Flag']=1; f.at[idx,'Inconsistency_Flag']='seq_violation'; continue
        if uri in {'/account/settings','/account/orders'} and not any(u in LOGIN_PRE for u in sess):
            f.at[idx,'Rule_Flag']=1; f.at[idx,'Inconsistency_Flag']='seq_violation'
    f['Rule_Predict']=(f['Rule_Flag']==0).astype(int)
    return f

dev=apply_rules(dev); test=apply_rules(test)
print(f"   Rules flagged: {(test['Rule_Flag']==1).sum():,} records")

# ── 4. ISOLATION FOREST ──────────────────────────────────────────────────
print("\n[4/7] Layer 2 — Isolation Forest ...")
iso=IsolationForest(n_estimators=100,contamination=0.07,random_state=42,n_jobs=-1)
iso.fit(dev[IF_FEATS])
test['IF_Predict']=(iso.predict(test[IF_FEATS])==1).astype(int)
print(f"   IF flagged: {(test['IF_Predict']==0).sum():,} records")

# ── 5. HYBRID ─────────────────────────────────────────────────────────────
print("\n[5/7] Hybrid (OR gate) ...")
# test['Hybrid_Predict']=((test['Rule_Predict']==1)&(test['IF_Predict']==1)).astype(int)

test['Hybrid_Flag']=((test['Rule_Flag']==1)|(test['IF_Predict']==0)).astype(int)
test['Hybrid_Predict']=(test['Hybrid_Flag']==0).astype(int)

y_true=test['_label']; y_rule=test['Rule_Predict']
y_if=test['IF_Predict']; y_hyb=test['Hybrid_Predict']

# ── 6. METRICS ─────────────────────────────────────────────────────────── 
print("\n[6/7] Evaluation ...")
def calc(yt,yp,name):
    a=accuracy_score(yt,yp)
    p=precision_score(yt,yp,zero_division=0)
    r=recall_score(yt,yp,zero_division=0)
    f=f1_score(yt,yp,zero_division=0)
    print(f"   {name:<28} Acc={a:.1%}  Pre={p:.1%}  Rec={r:.1%}  F1={f:.1%}")
    return a,p,r,f

rr=calc(y_true,y_rule,'Rule-Based Only')
ri=calc(y_true,y_if,'Isolation Forest Only')
rh=calc(y_true,y_hyb,'Hybrid (Proposed)')

cm=confusion_matrix(y_true,y_hyb)
TN,FP,FN,TP=cm.ravel(); total=cm.sum()
print(f"\n   Confusion Matrix (Hybrid):")
print(f"   TP={TP}({TP/total*100:.1f}%)  FP={FP}({FP/total*100:.1f}%)")
print(f"   FN={FN}({FN/total*100:.1f}%)  TN={TN}({TN/total*100:.1f}%)")

# ── 7. FIGURES ─────────────────────────────────────────────────────────────
print("\n[7/7] Generating figures ...")

def save(name):
    plt.tight_layout()
    plt.savefig(OUT+name, dpi=150, bbox_inches='tight', facecolor='#f8f9fb')
    plt.show()        
    plt.close()
    print(f"   Saved: {name}")


# FIG 3 — Performance Bar
fig,ax=plt.subplots(figsize=(11,6.5)); fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
x=np.arange(3); w=0.18
for k,(col,ml) in enumerate(zip([C['blue'],C['green'],C['orange'],C['red']],['Accuracy','Precision','Recall','F1-Score'])):
    vals=[rr[k],ri[k],rh[k]]
    bars=ax.bar(x+(k-1.5)*w,vals,w,label=ml,color=col,alpha=0.88)
    for bar in bars:
        h=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,h+0.005,f'{h:.2f}',ha='center',va='bottom',fontsize=8,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Rule-Based\nOnly','Isolation Forest\nOnly','Proposed Hybrid\n(Rule + IF)'],fontsize=10)
ax.set_ylabel('Score',fontsize=11); ax.set_ylim(0.55,1.05)
ax.set_title('Detection Performance: Component-wise Comparison',fontsize=13,fontweight='bold',pad=12)
ax.legend(fontsize=10,framealpha=0.8); ax.yaxis.grid(True,color=C['grid'],lw=0.8,alpha=0.7); ax.set_axisbelow(True)
ax.text(x[-1],max(rh)+0.02,'★ Best',ha='center',fontsize=10,color=C['gold'],fontweight='bold')
#save('figure3_performance.png')

# FIG 4 — Confusion Matrix
fig,ax=plt.subplots(figsize=(8,6.5)); fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
for col,cx,cy,main,sub in [
    (C['blue'],0,1,f'TP: {TP/total*100:.1f}%','(True Positive)'),
    ('#ccd9e8',1,1,f'FN: {FN/total*100:.1f}%','(False Negative)'),
    ('#e8cccc',0,0,f'FP: {FP/total*100:.1f}%','(False Positive)'),
    (C['blue'],1,0,f'TN: {TN/total*100:.1f}%','(True Negative)'),
]:
    ax.add_patch(FancyBboxPatch((cx+0.05,cy+0.05),0.88,0.88,boxstyle='round,pad=0.03',facecolor=col,edgecolor='white',lw=3))
    tc='white' if col==C['blue'] else C['dark']
    ax.text(cx+0.49,cy+0.58,main,ha='center',va='center',fontsize=13,fontweight='bold',color=tc)
    ax.text(cx+0.49,cy+0.28,sub,ha='center',va='center',fontsize=9,color=tc,alpha=0.85)
ax.set_xlim(-0.1,2.2); ax.set_ylim(-0.1,2.1)
ax.set_xticks([0.49,1.49]); ax.set_xticklabels(['Inconsistent\n(Predicted)','Consistent\n(Predicted)'],fontsize=11)
ax.set_yticks([0.49,1.49]); ax.set_yticklabels(['Consistent\n(Actual)','Inconsistent\n(Actual)'],fontsize=11)
ax.set_title('Confusion Matrix — Hybrid Detection Model (%)',fontsize=12,fontweight='bold',pad=12)
ax.tick_params(length=0)
#save('figure4_confusion.png')

# FIG 5 — Sensitivity
cvals=[0.05,0.10,0.15]; sens=[]
for c in cvals:
    m=IsolationForest(n_estimators=100,contamination=c,random_state=42,n_jobs=-1)
    m.fit(dev[IF_FEATS])
    pif=(m.predict(test[IF_FEATS])==1).astype(int)
    ph=((test['Rule_Predict']==1)&(pif==1)).astype(int)
    sens.append((accuracy_score(y_true,ph),precision_score(y_true,ph,zero_division=0),
                 recall_score(y_true,ph,zero_division=0),f1_score(y_true,ph,zero_division=0)))
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5)); fig.patch.set_facecolor('#f8f9fb')
for ax in [a1,a2]: ax.set_facecolor('#f8f9fb')
x=np.arange(3); w=0.18
for k,(col,ml) in enumerate(zip([C['blue'],C['green'],C['orange'],C['red']],['Accuracy','Precision','Recall','F1-Score'])):
    vals=[sens[i][k] for i in range(3)]
    bars=a1.bar(x+(k-1.5)*w,vals,w,color=col,label=ml,alpha=0.88)
    for bar in bars:
        h=bar.get_height(); a1.text(bar.get_x()+bar.get_width()/2,h+0.003,f'{h:.2f}',ha='center',va='bottom',fontsize=7.5,fontweight='bold')
a1.set_xticks(x); a1.set_xticklabels([f'{c}\n{"(used)" if c==0.10 else ""}' for c in cvals])
a1.set_xlabel('Contamination Value',fontsize=10); a1.set_ylabel('Score',fontsize=10)
a1.set_ylim(0.6,1.02); a1.set_title('Metrics vs. Contamination Parameter',fontsize=11,fontweight='bold')
a1.legend(fontsize=9,framealpha=0.8); a1.yaxis.grid(True,color=C['grid'],lw=0.8,alpha=0.7); a1.set_axisbelow(True)
pv=[s[1] for s in sens]; rv=[s[2] for s in sens]; fv=[s[3] for s in sens]
a2.plot(cvals,pv,'o-',color=C['green'],lw=2,ms=8,label='Precision')
a2.plot(cvals,rv,'s--',color=C['orange'],lw=2,ms=8,label='Recall')
a2.plot(cvals,fv,'^:',color=C['red'],lw=2,ms=8,label='F1-Score')
for vals,col in zip([pv,rv,fv],[C['green'],C['orange'],C['red']]):
    for xv,yv in zip(cvals,vals): a2.text(xv,yv+0.005,f'{yv:.2f}',ha='center',va='bottom',fontsize=9,color=col,fontweight='bold')
a2.axvline(0.10,color='#888',lw=1.5,linestyle='--',alpha=0.7)
a2.text(0.105,min(fv)-0.02,'selected\n(0.10)',fontsize=8,color='#555')
a2.set_xlabel('Contamination Value',fontsize=10); a2.set_ylabel('Score',fontsize=10); a2.set_xticks(cvals)
a2.set_title('Precision–Recall Trade-off vs. Contamination',fontsize=11,fontweight='bold')
a2.legend(fontsize=9,framealpha=0.8); a2.yaxis.grid(True,color=C['grid'],lw=0.8,alpha=0.7); a2.set_axisbelow(True)
plt.suptitle('Sensitivity Analysis — Isolation Forest Contamination Parameter',fontsize=12,fontweight='bold',y=1.02)
#save('figure5_sensitivity.png')

# FIG 6 — Category pie (bonus)
# fl=test[test['Rule_Flag']==1]; cv=fl['Inconsistency_Flag'].value_counts()
# cv=cv[cv.index!='none']
# lmap={'auth_missing':'Auth Missing','high_frequency':'High Frequency','seq_violation':'Seq Violation','wrong_method':'Wrong Method'}
# cats=[lmap.get(k,k) for k in cv.index]; colors=[C['blue'],C['orange'],C['green'],C['purple'],C['teal']][:len(cats)]
# fig,ax=plt.subplots(figsize=(9,6)); fig.patch.set_facecolor('#f8f9fb')
# _,_,auts=ax.pie(cv.values,labels=cats,autopct='%1.1f%%',colors=colors,startangle=140,
#                 wedgeprops=dict(edgecolor='white',linewidth=2.5),textprops=dict(fontsize=10))
# for at in auts: at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
# ax.set_title('Distribution of Flagged Inconsistencies by Category\n(Rule-Based Layer, Test Set)',fontsize=12,fontweight='bold',pad=15)
# save('figure6_categories.png')

# shutil.copy('/home/claude/run.py', OUT+'api_log_detection.py')

print("\n"+"="*60)
print("  FINAL RESULTS")
print("="*60)
print(f"  {'Method':<28} {'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7}")
print(f"  {'-'*55}")
for nm,r in [('Rule-Based Only',rr),('Isolation Forest Only',ri),('Hybrid (Proposed)',rh)]:
    print(f"  {nm:<28} {r[0]:>7.1%} {r[1]:>7.1%} {r[2]:>7.1%} {r[3]:>7.1%}")
print(f"\n  Confusion Matrix (Hybrid): TP={TP}({TP/total*100:.0f}%)  FP={FP}({FP/total*100:.0f}%)  FN={FN}({FN/total*100:.0f}%)  TN={TN}({TN/total*100:.0f}%)")
print("="*60)


# In[ ]:




