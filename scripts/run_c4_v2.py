"""C4 v2 development: clean-only frozen XAI calibration and clean-calibrated trust threshold."""
from __future__ import annotations
# Reuse validated data/pretraining helpers from corrected runner.
import argparse, json, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from scripts.run_c4_corrected import seed_all, ensure, parts, cap, ev, ref_loss, SHA, DROP
from xtrust_fl.model import XTrustMLP
from xtrust_fl.round_engine import build_round_batch, aggregate_same_round, candidate_models
from xtrust_fl.training import local_train, apply_delta
from xtrust_fl.attacks import sign_flip, model_scaling
from xtrust_fl.update_space import robust_update_anomaly
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed
from xtrust_fl.scoring import compute_client_scores

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=31415); ap.add_argument("--attack",choices=["sign_flip","scaling"],default="sign_flip"); ap.add_argument("--beta",type=float,default=.2); ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--clients",type=int,default=50); ap.add_argument("--alpha",type=float,default=.1); ap.add_argument("--cap",type=int,default=12000); ap.add_argument("--pretrain-epochs",type=int,default=3); ap.add_argument("--mu",type=float,default=.01); ap.add_argument("--ref-size",type=int,default=128); ap.add_argument("--ig-steps",type=int,default=16); ap.add_argument("--out",default="results/c4_v2.json"); a=ap.parse_args(); seed_all(a.seed); dev="cuda" if torch.cuda.is_available() else "cpu"
 path=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(path); df=pd.read_parquet(path); lk={str(c).lower().strip():c for c in df}; y=pd.to_numeric(df[lk["is_attack"]]).astype("int64").to_numpy(); xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan); idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]); im=SimpleImputer(strategy="median"); X0=im.fit_transform(xd.iloc[tr]); V0=im.transform(xd.iloc[va]); E0=im.transform(xd.iloc[te]); sc=StandardScaler(); X=sc.fit_transform(X0).astype("float32"); V=sc.transform(V0).astype("float32"); E=sc.transform(E0).astype("float32"); Y=y[tr]; Vy=y[va]; Ey=y[te]
 model=XTrustMLP(X.shape[1],2); model,_=local_train(model,torch.from_numpy(X),torch.from_numpy(Y),epochs=a.pretrain_epochs,batch_size=1024,lr=1e-3,device=dev); pre=ev(model,E,Ey,dev); ps,mn=parts(Y,a.clients,a.alpha,a.seed); ps=cap(ps,a.cap,a.seed); ids=list(range(a.clients)); rng=np.random.default_rng(a.seed+404); nm=max(1,int(round(a.clients*a.beta))); bad=sorted(rng.choice(a.clients,nm,replace=False).astype(int).tolist()); badset=set(bad); rr=np.random.default_rng(a.seed+909).choice(len(V),min(a.ref_size,len(V)),replace=False); xr=torch.from_numpy(V[rr]); yr=torch.from_numpy(Vy[rr])
 # CLEAN-ONLY calibration. The Huber relationship is frozen before any poisoned client is observed.
 clean=build_round_batch(model,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev); fps=[]; losses=[]
 for d in clean.updates:
  cm=apply_delta(model,d); fps.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); losses.append(ref_loss(cm,xr,yr))
 eref=fit_explanation_anomaly_reference(fps); raw0=score_explanation_anomaly_fixed(fps,eref); hs=RobustScaler().fit(np.asarray(losses)[:,None]); hub=HuberRegressor().fit(hs.transform(np.asarray(losses)[:,None]),raw0); exp0=np.maximum(0.,hub.predict(hs.transform(np.asarray(losses)[:,None]))); resid0=np.maximum(0.,raw0-exp0); u0=robust_update_anomaly(clean.updates); uth=max(float(np.quantile(u0,.95)),1e-12); eth=max(float(np.quantile(resid0,.95)),1e-12); clean_scores=compute_client_scores(u0/uth,resid0/eth,previous_scores=None,temporal_gamma=0.0); trust_thr=float(np.quantile(clean_scores,.05)); prev=clean_scores.copy()
 history=[]; base=model
 for r in range(a.rounds):
  def attack(cid,d): return d if cid not in badset else (sign_flip(d,1.0) if a.attack=="sign_flip" else model_scaling(d,5.0))
  b=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,transform_update=attack); uf=robust_update_anomaly(b.updates); pf=[]; rl=[]
  for d in b.updates:
   cm=apply_delta(base,d); pf.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); rl.append(ref_loss(cm,xr,yr))
  raw=score_explanation_anomaly_fixed(pf,eref); expected=np.maximum(0.,hub.predict(hs.transform(np.asarray(rl)[:,None]))); residual=np.maximum(0.,raw-expected); scores=compute_client_scores(uf/uth,residual/eth,previous_scores=prev); prev=scores.copy(); ag=aggregate_same_round(b,xtrust_scores=scores,xtrust_reject_threshold=trust_thr,trim_ratio=.2,krum_f=nm); cand=candidate_models(base,ag); metrics={k:ev(v,E,Ey,dev) for k,v in cand.items()}; labels=np.asarray([c in badset for c in b.client_ids],dtype=int); detected=(scores<trust_thr).astype(int); tp=int(((detected==1)&(labels==1)).sum()); fp=int(((detected==1)&(labels==0)).sum()); tn=int(((detected==0)&(labels==0)).sum()); fn=int(((detected==0)&(labels==1)).sum()); history.append({"round":r+1,"detection":{"threshold":trust_thr,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"tpr":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1),"scores":scores.tolist(),"update_anomaly":uf.tolist(),"explanation_residual":residual.tolist()},"utility":metrics}); base=cand["xtrust_fl"]
 out={"experiment":"C4_v2_clean_calibrated_trust_development","paper_final":False,"config":vars(a),"dataset_sha256":SHA,"device":dev,"pretraining_test":pre,"partition_min_size":mn,"partition_sizes":[len(q) for q in ps],"malicious_client_ids":bad,"reference_indices_validation":rr.astype(int).tolist(),"calibration":{"update_q95":uth,"explanation_residual_q95":eth,"clean_trust_q05":trust_thr,"clean_scores":clean_scores.tolist()},"history":history}; dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print("C4 v2 finished:",dest); print("clean trust threshold=",trust_thr); [print("round",z["round"],"TPR",z["detection"]["tpr"],"FPR",z["detection"]["fpr"],"XTrust Macro-F1",z["utility"]["xtrust_fl"]["macro_f1"]) for z in history]
if __name__=="__main__": main()
