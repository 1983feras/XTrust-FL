"""C2-v3 diagnostic: explain benign explanation drift under severe Non-IID.

Development-only runner. It enriches H with feature-distribution shift,
effective training quantity, class imbalance, and reference loss while keeping
all fitted references calibration-only. It also exports rank/top-k/L1 drift
components per client for diagnosis. Do not use development seeds as final
confirmation evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition, client_label_distributions
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed, explanation_features_against_prototype
from xtrust_fl.heterogeneity import js_divergence_to_global, calibrate_explanation_residual
URL="https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"; SHA="17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"; DROP={"label","family","is_attack"}
def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def ensure(p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): urllib.request.urlretrieve(URL,p)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA: raise RuntimeError("SHA mismatch")
def train(m,X,y,d,e=1):
 m=deepcopy(m).to(d); dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),512,shuffle=True); o=torch.optim.Adam(m.parameters(),lr=1e-3); f=torch.nn.CrossEntropyLoss()
 for _ in range(e):
  m.train()
  for xb,yb in dl:
   xb,yb=xb.to(d),yb.to(d); o.zero_grad(); z=f(m(xb),yb); z.backward(); o.step()
 return m.cpu()
def part(y,n,a,s):
 last=None
 for mn in [50,20,10,5,1]:
  try:return dirichlet_partition(y,n,a,s,min_size=mn,max_retries=5000),mn
  except RuntimeError as e:last=e
 raise RuntimeError("partition failure") from last
def ref_loss(model,x,y):
 model=deepcopy(model).cpu().eval()
 with torch.no_grad(): return float(torch.nn.functional.cross_entropy(model(x),y).item())
def main():
 p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,required=True); p.add_argument("--alpha",type=float,default=.1); p.add_argument("--clients",type=int,default=20); p.add_argument("--pretrain-epochs",type=int,default=3); p.add_argument("--local-epochs",type=int,default=1); p.add_argument("--client-cap",type=int,default=12000); p.add_argument("--ref-size",type=int,default=128); p.add_argument("--ig-steps",type=int,default=16); p.add_argument("--out",required=True); a=p.parse_args(); seed_all(a.seed)
 data=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(data); df=pd.read_parquet(data); lk={str(c).strip().lower():c for c in df}; target=lk["is_attack"]; y=pd.to_numeric(df[target]).astype("int64").to_numpy(); dropped=[c for c in df if str(c).strip().lower() in DROP]; xd=df.drop(columns=dropped).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
 ix=np.arange(len(y)); tr,te=train_test_split(ix,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]); im=SimpleImputer(strategy="median"); ti=im.fit_transform(xd.iloc[tr]); vi=im.transform(xd.iloc[va]); sc=StandardScaler(); X=sc.fit_transform(ti).astype("float32"); V=sc.transform(vi).astype("float32"); Y=y[tr]; Vy=y[va]; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); base=train(XTrustMLP(X.shape[1],2),X,Y,dev,a.pretrain_epochs)
 parts,mn=part(Y,a.clients,a.alpha,a.seed); dists=client_label_distributions(Y,parts); js=js_divergence_to_global(dists); orig=np.asarray([len(q) for q in parts],float); perm=np.random.default_rng(a.seed+123).permutation(a.clients); nc=max(5,a.clients//2); cal=np.sort(perm[:nc]); ev=np.sort(perm[nc:]); mask=np.zeros(a.clients,bool); mask[cal]=1
 rr=np.random.default_rng(a.seed+991); ri=rr.choice(len(V),min(a.ref_size,len(V)),replace=False); xr=torch.from_numpy(V[ri]); yr=torch.from_numpy(Vy[ri]); global_mean=X.mean(0); global_std=X.std(0)+1e-6
 fps=[]; used=[]; feature_shift=[]; imbalance=[]; losses=[]
 for cid,q in enumerate(parts):
  q=np.asarray(q,int)
  if len(q)>a.client_cap:q=np.random.default_rng(a.seed*1000+cid).choice(q,a.client_cap,replace=False)
  used.append(len(q)); localX=X[q]; feature_shift.append(float(np.mean(np.abs((localX.mean(0)-global_mean)/global_std)))); counts=np.bincount(Y[q],minlength=2).astype(float); probs=counts/counts.sum(); imbalance.append(float(abs(probs[1]-probs[0]))); m=train(base,localX,Y[q],dev,a.local_epochs); losses.append(ref_loss(m,xr,yr)); fps.append(attribution_fingerprint(m,xr,yr,a.ig_steps))
 ref=fit_explanation_anomaly_reference([fps[i] for i in cal],top_k=min(10,X.shape[1])); raw=score_explanation_anomaly_fixed(fps,ref); comps=explanation_features_against_prototype(fps,ref["prototype"],top_k=ref["top_k"]); used=np.asarray(used,float); qeff=np.abs(np.log((used+1)/(np.median(used[cal])+1))); H=np.column_stack([js,qeff,np.asarray(feature_shift),np.asarray(imbalance),np.asarray(losses)]); residual,expected=calibrate_explanation_residual(raw,H,mask,min_fit_clients=5); rthr=float(np.quantile(raw[cal],.95)); cthr=float(np.quantile(residual[cal],.95)); rf=raw[ev]>rthr; cf=residual[ev]>cthr
 out={"experiment":"C2_v3_diagnostic","development_only":True,"config":vars(a),"integrity":{"prototype_source":"calibration_only","anomaly_scale_source":"calibration_only","heterogeneity_fit_source":"calibration_only","threshold_source":"calibration_only","evaluation_clients_in_reference":False},"calibration_client_ids":cal.tolist(),"evaluation_client_ids":ev.tolist(),"partition":{"accepted_min":mn,"original_sizes":orig.astype(int).tolist(),"used_sizes":used.astype(int).tolist()},"H_columns":["label_js","effective_quantity_skew","feature_mean_shift","class_imbalance","reference_loss"],"H":H.tolist(),"explanation_components_columns":["rank_drift","topk_drift","l1_drift"],"explanation_components":comps.tolist(),"raw":raw.tolist(),"expected":expected.tolist(),"residual":residual.tolist(),"thresholds":{"raw_q95":rthr,"calibrated_q95":cthr},"evaluation":{"raw_fp":int(rf.sum()),"calibrated_fp":int(cf.sum()),"n":len(ev),"raw_fpr":float(rf.mean()),"calibrated_fpr":float(cf.mean()),"delta":float(cf.mean()-rf.mean())}}
 dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
