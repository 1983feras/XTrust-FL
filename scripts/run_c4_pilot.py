"""C4 pilot: severe Non-IID poisoning with paired aggregation.

This runner is intentionally a fast first C4 execution, not the final multi-seed
paper table. It follows docs/C4_FROZEN_PROTOCOL.md and saves full provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, urllib.request
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition
from xtrust_fl.round_engine import build_round_batch, aggregate_same_round, candidate_models
from xtrust_fl.attacks import sign_flip, model_scaling

URL="https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"
SHA="17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"
DROP={"label","family","is_attack"}

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ensure(p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): urllib.request.urlretrieve(URL,p)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA: raise RuntimeError("SHA mismatch")

def partition(y,n,a,s):
 last=None
 for mn in [50,20,10,5,1]:
  try:return dirichlet_partition(y,n,a,s,min_size=mn,max_retries=5000),mn
  except RuntimeError as e:last=e
 raise RuntimeError("partition failure") from last

def cap_parts(parts,cap,seed):
 out=[]
 for cid,q in enumerate(parts):
  q=np.asarray(q,dtype=np.int64)
  if len(q)>cap:q=np.random.default_rng(seed*1000+cid).choice(q,cap,replace=False)
  out.append(q.astype(int).tolist())
 return out

def evaluate(model,X,y,device,batch=4096):
 model=deepcopy(model).to(device).eval(); probs=[]; preds=[]
 with torch.no_grad():
  for i in range(0,len(X),batch):
   xb=torch.from_numpy(X[i:i+batch]).to(device); z=model(xb); p=torch.softmax(z,1)[:,1]
   probs.append(p.cpu().numpy()); preds.append(torch.argmax(z,1).cpu().numpy())
 pr=np.concatenate(probs); yh=np.concatenate(preds)
 return {"accuracy":float(accuracy_score(y,yh)),"macro_f1":float(f1_score(y,yh,average="macro")),"precision":float(precision_score(y,yh,zero_division=0)),"recall":float(recall_score(y,yh,zero_division=0)),"auroc":float(roc_auc_score(y,pr)),"auprc":float(average_precision_score(y,pr)),"confusion":confusion_matrix(y,yh).tolist()}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=31415); p.add_argument("--attack",choices=["sign_flip","scaling"],default="sign_flip"); p.add_argument("--beta",type=float,default=.2); p.add_argument("--alpha",type=float,default=.1); p.add_argument("--clients",type=int,default=50); p.add_argument("--rounds",type=int,default=3); p.add_argument("--client-cap",type=int,default=12000); p.add_argument("--local-epochs",type=int,default=1); p.add_argument("--fedprox-mu",type=float,default=.01); p.add_argument("--out",default="results/c4_pilot.json"); a=p.parse_args()
 if a.beta not in {.1,.2,.3,.4}: raise ValueError("beta must be one of frozen values {0.1,0.2,0.3,0.4}")
 seed_all(a.seed); path=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(path); df=pd.read_parquet(path)
 lk={str(c).strip().lower():c for c in df}; y=pd.to_numeric(df[lk["is_attack"]]).astype("int64").to_numpy(); xd=df.drop(columns=[c for c in df if str(c).strip().lower() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
 idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr])
 im=SimpleImputer(strategy="median"); Xt=im.fit_transform(xd.iloc[tr]); Xv=im.transform(xd.iloc[va]); Xe=im.transform(xd.iloc[te]); sc=StandardScaler(); X=sc.fit_transform(Xt).astype("float32"); V=sc.transform(Xv).astype("float32"); E=sc.transform(Xe).astype("float32"); Y=y[tr]; Vy=y[va]; Ey=y[te]
 parts,mn=partition(Y,a.clients,a.alpha,a.seed); parts=cap_parts(parts,a.client_cap,a.seed); dev="cuda" if torch.cuda.is_available() else "cpu"; models={k:XTrustMLP(X.shape[1],2) for k in ["fedavg","coordinate_median","trimmed_mean","krum","multi_krum"]}
 # Same deterministic malicious identities for every defense and every round.
 rng=np.random.default_rng(a.seed+404); m=max(1,int(round(a.clients*a.beta))); malicious=sorted(rng.choice(a.clients,m,replace=False).astype(int).tolist()); malicious_set=set(malicious)
 history=[]
 for r in range(a.rounds):
  next_models={}; row={"round":r+1,"methods":{}}
  # Each method starts from its own global model, but all clients/partitions/attack IDs are paired.
  # Within a method, all robust aggregators consume exactly the same immutable RoundBatch.
  base=models["fedavg"] if r==0 else models["fedavg"]
  def transform(cid,d):
   if cid not in malicious_set:return d
   return sign_flip(d,1.0) if a.attack=="sign_flip" else model_scaling(d,5.0)
  batch=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),parts,list(range(a.clients)),epochs=a.local_epochs,batch_size=512,lr=1e-3,fedprox_mu=a.fedprox_mu,device=dev,transform_update=transform)
  ag=aggregate_same_round(batch,trim_ratio=.2,krum_f=m)
  cand=candidate_models(base,ag)
  for name,model in cand.items(): row["methods"][name]=evaluate(model,E,Ey,dev); next_models[name]=model.cpu()
  # Pilot deliberately advances FedAvg trajectory only; final C4 runner will maintain independent multi-round trajectories per defense.
  models.update(next_models); history.append(row)
 out={"experiment":"C4_fast_pilot_severe_nonIID_poisoning","paper_final":False,"warning":"Pilot only: multi-round trajectory advances from FedAvg base; use final C4 runner for paper comparisons.","config":vars(a),"dataset":{"sha256":SHA,"n_features":int(X.shape[1]),"train":int(len(X)),"validation":int(len(V)),"test":int(len(E))},"partition":{"accepted_min_size":mn,"used_sizes":[len(q) for q in parts]},"malicious_client_ids":malicious,"device":dev,"history":history}
 dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
