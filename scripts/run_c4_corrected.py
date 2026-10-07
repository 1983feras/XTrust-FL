"""Corrected C4 development runner.
Pretrains a global model, performs a clean calibration round, then evaluates
poisoned paired rounds with update anomaly + server-side IG residual trust.
Outputs are DEVELOPMENT results until multi-seed confirmation is frozen.
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
from xtrust_fl.training import local_train, apply_delta
from xtrust_fl.attacks import sign_flip, model_scaling
from xtrust_fl.update_space import robust_update_anomaly
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed
from xtrust_fl.heterogeneity import calibrate_explanation_residual
from xtrust_fl.scoring import compute_client_scores
URL="https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"; SHA="17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"; DROP={"label","family","is_attack"}
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def ensure(p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): urllib.request.urlretrieve(URL,p)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA: raise RuntimeError("dataset SHA mismatch")
def parts(y,n,a,s):
 for mn in [50,20,10,5,1]:
  try:return dirichlet_partition(y,n,a,s,min_size=mn,max_retries=5000),mn
  except RuntimeError: pass
 raise RuntimeError("Dirichlet partition failed")
def cap(ps,c,s):
 out=[]
 for i,q in enumerate(ps):
  q=np.asarray(q,dtype=np.int64)
  if len(q)>c:q=np.random.default_rng(s*1000+i).choice(q,c,replace=False)
  out.append(q.astype(int).tolist())
 return out
def ev(m,X,y,d):
 m=deepcopy(m).to(d).eval(); pp=[]; yy=[]
 with torch.no_grad():
  for i in range(0,len(X),4096):
   z=m(torch.from_numpy(X[i:i+4096]).to(d)); pp.append(torch.softmax(z,1)[:,1].cpu().numpy()); yy.append(torch.argmax(z,1).cpu().numpy())
 p=np.concatenate(pp); h=np.concatenate(yy); return {"accuracy":float(accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"precision":float(precision_score(y,h,zero_division=0)),"recall":float(recall_score(y,h,zero_division=0)),"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"confusion":confusion_matrix(y,h).tolist()}
def ref_loss(model,x,y):
 model=deepcopy(model).cpu().eval()
 with torch.no_grad(): return float(torch.nn.functional.cross_entropy(model(x),y).item())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=31415); ap.add_argument("--attack",choices=["sign_flip","scaling"],default="sign_flip"); ap.add_argument("--beta",type=float,default=.2); ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--clients",type=int,default=50); ap.add_argument("--alpha",type=float,default=.1); ap.add_argument("--cap",type=int,default=12000); ap.add_argument("--pretrain-epochs",type=int,default=3); ap.add_argument("--mu",type=float,default=.01); ap.add_argument("--ref-size",type=int,default=128); ap.add_argument("--ig-steps",type=int,default=16); ap.add_argument("--out",default="results/c4_corrected.json"); a=ap.parse_args(); seed_all(a.seed); dev="cuda" if torch.cuda.is_available() else "cpu"
 path=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(path); df=pd.read_parquet(path); lk={str(c).lower().strip():c for c in df}; y=pd.to_numeric(df[lk["is_attack"]]).astype("int64").to_numpy(); xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan); idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]); im=SimpleImputer(strategy="median"); X0=im.fit_transform(xd.iloc[tr]); V0=im.transform(xd.iloc[va]); E0=im.transform(xd.iloc[te]); sc=StandardScaler(); X=sc.fit_transform(X0).astype("float32"); V=sc.transform(V0).astype("float32"); E=sc.transform(E0).astype("float32"); Y=y[tr]; Vy=y[va]; Ey=y[te]
 # Pretraining prevents the class-collapse observed in the first C4 pilot.
 model=XTrustMLP(X.shape[1],2); model,_=local_train(model,torch.from_numpy(X),torch.from_numpy(Y),epochs=a.pretrain_epochs,batch_size=1024,lr=1e-3,device=dev); pre=ev(model,E,Ey,dev)
 ps,mn=parts(Y,a.clients,a.alpha,a.seed); ps=cap(ps,a.cap,a.seed); ids=list(range(a.clients)); rng=np.random.default_rng(a.seed+404); nm=max(1,int(round(a.clients*a.beta))); bad=sorted(rng.choice(a.clients,nm,replace=False).astype(int).tolist()); badset=set(bad)
 # Fixed trusted server reference comes only from validation data.
 rr=np.random.default_rng(a.seed+909).choice(len(V),min(a.ref_size,len(V)),replace=False); xr=torch.from_numpy(V[rr]); yr=torch.from_numpy(Vy[rr])
 # Clean calibration round: no poisoning; all clients establish fixed XAI reference.
 clean=build_round_batch(model,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev)
 fps=[]; losses=[]
 for d in clean.updates:
  cm=apply_delta(model,d); fps.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); losses.append(ref_loss(cm,xr,yr))
 eref=fit_explanation_anomaly_reference(fps); raw0=score_explanation_anomaly_fixed(fps,eref); resid0,expected0=calibrate_explanation_residual(raw0,np.asarray(losses)[:,None]); u0=robust_update_anomaly(clean.updates); uth=float(np.quantile(u0,.95)); eth=float(np.quantile(resid0,.95)); prev=np.ones(a.clients,dtype=float)
 history=[]; base=model
 for r in range(a.rounds):
  def attack(cid,d):
   if cid not in badset:return d
   return sign_flip(d,1.0) if a.attack=="sign_flip" else model_scaling(d,5.0)
  b=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,transform_update=attack); uf=robust_update_anomaly(b.updates); pf=[]; rl=[]
  for d in b.updates:
   cm=apply_delta(base,d); pf.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); rl.append(ref_loss(cm,xr,yr))
  raw=score_explanation_anomaly_fixed(pf,eref); # loss-only heterogeneity calibration, frozen C2 choice
  residual,expected=calibrate_explanation_residual(raw,np.asarray(rl)[:,None]); un=uf/max(uth,1e-12); en=residual/max(eth,1e-12); scores=compute_client_scores(un,en,previous_scores=prev); prev=scores.copy(); ag=aggregate_same_round(b,xtrust_scores=scores,trim_ratio=.2,krum_f=nm); cand=candidate_models(base,ag); metrics={k:ev(v,E,Ey,dev) for k,v in cand.items()}; labels=np.asarray([1 if c in badset else 0 for c in b.client_ids]); detected=(scores<.5).astype(int); tp=int(((detected==1)&(labels==1)).sum()); fp=int(((detected==1)&(labels==0)).sum()); tn=int(((detected==0)&(labels==0)).sum()); fn=int(((detected==0)&(labels==1)).sum()); history.append({"round":r+1,"detection":{"tp":tp,"fp":fp,"tn":tn,"fn":fn,"tpr":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1),"scores":scores.tolist(),"update_anomaly":uf.tolist(),"explanation_residual":residual.tolist()},"utility":metrics}); base=cand["xtrust_fl"]
 out={"experiment":"C4_corrected_development","paper_final":False,"config":vars(a),"dataset_sha256":SHA,"device":dev,"pretraining_test":pre,"partition_min_size":mn,"partition_sizes":[len(q) for q in ps],"malicious_client_ids":bad,"reference_indices_validation":rr.astype(int).tolist(),"calibration":{"update_q95":uth,"explanation_residual_q95":eth},"history":history}; dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print("C4 corrected finished:",dest); print("pretraining macro_f1=",pre["macro_f1"]); [print("round",z["round"],"TPR",z["detection"]["tpr"],"FPR",z["detection"]["fpr"],"XTrust Macro-F1",z["utility"]["xtrust_fl"]["macro_f1"]) for z in history]
if __name__=="__main__": main()
