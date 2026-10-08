from __future__ import annotations
import hashlib, json, random, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition
from xtrust_fl.round_engine import build_round_batch
from xtrust_fl.aggregate import fedavg_aggregate
from xtrust_fl.training import apply_delta

SHA='17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1'
DROP={'label','family','is_attack'}
SEED=42; CLIENTS=50; ALPHA=.10; CAP=12000; MU=.01
LOCAL_EPOCHS=1; BATCH=512; LR=1e-3; MAX_ROUNDS=20

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ev(model,X,y,device):
    m=deepcopy(model).to(device).eval(); pp=[]; hh=[]
    with torch.no_grad():
        for i in range(0,len(X),4096):
            z=m(torch.from_numpy(X[i:i+4096]).to(device)); pp.append(torch.softmax(z,1)[:,1].cpu().numpy()); hh.append(torch.argmax(z,1).cpu().numpy())
    p=np.concatenate(pp); h=np.concatenate(hh); cm=confusion_matrix(y,h,labels=[0,1]).tolist()
    return {'accuracy':float(accuracy_score(y,h)),'macro_f1':float(f1_score(y,h,average='macro')),'precision':float(precision_score(y,h,zero_division=0)),'recall':float(recall_score(y,h,zero_division=0)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'confusion':cm,'predicted_classes':sorted(np.unique(h).astype(int).tolist())}

def parts(y,n,a,s):
    for mn in [50,20,10,5,1]:
        try:return dirichlet_partition(y,n,a,s,min_size=mn,max_retries=5000),mn
        except RuntimeError: pass
    raise RuntimeError('Dirichlet partition failed')

def cap_parts(ps,c,s):
    out=[]
    for i,q in enumerate(ps):
        q=np.asarray(q,dtype=np.int64)
        if len(q)>c:q=np.random.default_rng(s*1000+i).choice(q,c,replace=False)
        out.append(q.astype(int).tolist())
    return out

def adequate(v):
    cm=v['confusion']; return (v['predicted_classes']==[0,1] and cm[0][0]>0 and cm[1][1]>0 and v['macro_f1']>=.70 and v['auroc']>=.90)

def main():
    seed_all(SEED); dev='cuda' if torch.cuda.is_available() else 'cpu'
    path=ROOT/'data/external/ciciot2023_clean.parquet'
    if not path.exists(): raise FileNotFoundError(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA: raise RuntimeError('dataset SHA mismatch')
    df=pd.read_parquet(path); lk={str(c).lower().strip():c for c in df}; y=pd.to_numeric(df[lk['is_attack']]).astype('int64').to_numpy(); xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=SEED,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=SEED,stratify=y[tr])
    im=SimpleImputer(strategy='median'); X0=im.fit_transform(xd.iloc[tr]); V0=im.transform(xd.iloc[va]); E0=im.transform(xd.iloc[te]); sc=StandardScaler(); X=sc.fit_transform(X0).astype('float32'); V=sc.transform(V0).astype('float32'); E=sc.transform(E0).astype('float32'); Y=y[tr]; Vy=y[va]; Ey=y[te]
    ps,mn=parts(Y,CLIENTS,ALPHA,SEED); ps=cap_parts(ps,CAP,SEED); ids=list(range(CLIENTS)); model=XTrustMLP(X.shape[1],2)
    history=[]; first=None
    initial={'round':0,'validation':ev(model,V,Vy,dev),'test':ev(model,E,Ey,dev)}; history.append(initial)
    print('round 0 val macro_f1',initial['validation']['macro_f1'],'auroc',initial['validation']['auroc'],'cm',initial['validation']['confusion'],flush=True)
    for r in range(1,MAX_ROUNDS+1):
        t=time.time(); seed_all(SEED+10000*r+11)
        b=build_round_batch(model,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=LOCAL_EPOCHS,batch_size=BATCH,lr=LR,fedprox_mu=MU,device=dev)
        model=apply_delta(model,fedavg_aggregate(b.updates,b.sample_counts)); v=ev(model,V,Vy,dev); e=ev(model,E,Ey,dev); ok=adequate(v)
        if ok and first is None:first=r
        rec={'round':r,'seconds':time.time()-t,'validation':v,'test':e,'adequate':ok}; history.append(rec)
        print('round',r,'val macro_f1',v['macro_f1'],'auroc',v['auroc'],'cm',v['confusion'],'adequate',ok,'seconds',rec['seconds'],flush=True)
    out={'experiment':'final_study_clean_initialization_diagnostic','paper_final':False,'protocol':'docs/FINAL_STUDY_INIT_DIAGNOSTIC_PROTOCOL.md','dataset_sha256':SHA,'device':dev,'config':{'seed':SEED,'clients':CLIENTS,'alpha':ALPHA,'cap':CAP,'mu':MU,'local_epochs':LOCAL_EPOCHS,'batch_size':BATCH,'lr':LR,'max_rounds':MAX_ROUNDS},'partition_min_size':mn,'partition_sizes':[len(q) for q in ps],'first_adequate_round':first,'history':history}
    dest=ROOT/'results/final_study_init_diagnostic_seed42.json'; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('FIRST_ADEQUATE_ROUND=',first,flush=True); print('RESULT=',dest,flush=True)
if __name__=='__main__':main()
