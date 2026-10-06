"""C2-v4 development ablation for parsimonious heterogeneity calibration.

Compares Raw-XAI, reference-loss only, loss+feature-shift, and
loss+feature-shift+effective-quantity calibration. Expected anomaly predictions
are clipped to non-negative values. All fitted references use calibration
clients only. Development seeds must not be reused for final confirmation.
"""
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed
URL="https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"; SHA="17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"; DROP={"label","family","is_attack"}
def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def ensure(p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): urllib.request.urlretrieve(URL,p)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA: raise RuntimeError("SHA mismatch")
def train(m,X,y,d,e):
 m=deepcopy(m).to(d); dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),512,shuffle=True); o=torch.optim.Adam(m.parameters(),lr=1e-3); f=torch.nn.CrossEntropyLoss()
 for _ in range(e):
  m.train()
  for xb,yb in dl:
   xb,yb=xb.to(d),yb.to(d); o.zero_grad(); loss=f(m(xb),yb); loss.backward(); o.step()
 return m.cpu()
def partition(y,n,a,s):
 last=None
 for mn in [50,20,10,5,1]:
  try:return dirichlet_partition(y,n,a,s,min_size=mn,max_retries=5000),mn
  except RuntimeError as e:last=e
 raise RuntimeError("partition failure") from last
def ce(m,x,y):
 m=deepcopy(m).cpu().eval()
 with torch.no_grad():return float(torch.nn.functional.cross_entropy(m(x),y).item())
def calibrate(raw,H,mask):
 H=np.asarray(H,float); H=H[:,None] if H.ndim==1 else H; sc=RobustScaler().fit(H[mask]); Xc=sc.transform(H[mask]); Xa=sc.transform(H); model=HuberRegressor().fit(Xc,raw[mask]); expected=np.maximum(0.0,model.predict(Xa)); return np.maximum(0.0,raw-expected),expected
def main():
 p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,required=True); p.add_argument("--alpha",type=float,default=.1); p.add_argument("--clients",type=int,default=50); p.add_argument("--pretrain-epochs",type=int,default=3); p.add_argument("--local-epochs",type=int,default=1); p.add_argument("--client-cap",type=int,default=12000); p.add_argument("--ref-size",type=int,default=128); p.add_argument("--ig-steps",type=int,default=16); p.add_argument("--out",required=True); a=p.parse_args(); seed_all(a.seed)
 if a.clients<20:raise ValueError("v4 requires >=20 clients")
 path=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(path); df=pd.read_parquet(path); lk={str(c).strip().lower():c for c in df}; y=pd.to_numeric(df[lk["is_attack"]]).astype("int64").to_numpy(); drop=[c for c in df if str(c).strip().lower() in DROP]; xd=df.drop(columns=drop).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
 idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]); im=SimpleImputer(strategy="median"); ti=im.fit_transform(xd.iloc[tr]); vi=im.transform(xd.iloc[va]); ss=StandardScaler(); X=ss.fit_transform(ti).astype("float32"); V=ss.transform(vi).astype("float32"); Y=y[tr]; Vy=y[va]; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); base=train(XTrustMLP(X.shape[1],2),X,Y,dev,a.pretrain_epochs)
 parts,mn=partition(Y,a.clients,a.alpha,a.seed); perm=np.random.default_rng(a.seed+123).permutation(a.clients); ncal=a.clients//2; cal=np.sort(perm[:ncal]); ev=np.sort(perm[ncal:]); mask=np.zeros(a.clients,bool); mask[cal]=1; rr=np.random.default_rng(a.seed+991); ri=rr.choice(len(V),min(a.ref_size,len(V)),replace=False); xr=torch.from_numpy(V[ri]); yr=torch.from_numpy(Vy[ri]); gm=X.mean(0); gs=X.std(0)+1e-6
 fps=[]; used=[]; shift=[]; loss=[]
 for cid,q in enumerate(parts):
  q=np.asarray(q,int)
  if len(q)>a.client_cap:q=np.random.default_rng(a.seed*1000+cid).choice(q,a.client_cap,replace=False)
  used.append(len(q)); lx=X[q]; shift.append(float(np.mean(np.abs((lx.mean(0)-gm)/gs)))); m=train(base,lx,Y[q],dev,a.local_epochs); loss.append(ce(m,xr,yr)); fps.append(attribution_fingerprint(m,xr,yr,a.ig_steps))
 ref=fit_explanation_anomaly_reference([fps[i] for i in cal],top_k=min(10,X.shape[1])); raw=score_explanation_anomaly_fixed(fps,ref); used=np.asarray(used,float); qeff=np.abs(np.log((used+1)/(np.median(used[cal])+1))); loss=np.asarray(loss); shift=np.asarray(shift)
 variants={"loss_only":loss[:,None],"loss_feature":np.column_stack([loss,shift]),"loss_feature_quantity":np.column_stack([loss,shift,qeff])}; rawthr=float(np.quantile(raw[cal],.95)); results={"raw":{"threshold":rawthr,"fp":int((raw[ev]>rawthr).sum()),"fpr":float((raw[ev]>rawthr).mean())}}; details={}
 for name,H in variants.items():
  r,e=calibrate(raw,H,mask); th=float(np.quantile(r[cal],.95)); fl=r[ev]>th; results[name]={"threshold":th,"fp":int(fl.sum()),"fpr":float(fl.mean()),"delta_vs_raw":float(fl.mean()-(raw[ev]>rawthr).mean())}; details[name]={"expected":e.tolist(),"residual":r.tolist()}
 out={"experiment":"C2_v4_parsimonious_ablation","development_only":True,"config":vars(a),"integrity":{"calibration_clients_only":True,"evaluation_clients_in_reference":False,"expected_anomaly_nonnegative":True},"calibration_client_ids":cal.tolist(),"evaluation_client_ids":ev.tolist(),"accepted_partition_min":mn,"signals":{"reference_loss":loss.tolist(),"feature_mean_shift":shift.tolist(),"effective_quantity_skew":qeff.tolist()},"raw_anomaly":raw.tolist(),"results":results,"details":details}
 dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
