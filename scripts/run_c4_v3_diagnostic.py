"""C4-v3 frozen development diagnostic: clean-fixed U/XAI references and predeclared ablations."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from scripts.run_c4_corrected import seed_all, ensure, parts, cap, ev, ref_loss, SHA, DROP
from xtrust_fl.model import XTrustMLP
from xtrust_fl.round_engine import build_round_batch, aggregate_same_round, candidate_models
from xtrust_fl.training import local_train, apply_delta
from xtrust_fl.attacks import sign_flip, model_scaling
from xtrust_fl.update_space import fit_update_anomaly_reference, score_update_anomaly_fixed
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed
from xtrust_fl.scoring import compute_client_scores

def trust_from_anomaly(a):
    a=np.asarray(a,dtype=float); return 1.0/(1.0+np.exp(-np.clip(2.0-a,-60.,60.)))

def arm_stats(scores,thr,labels):
    scores=np.asarray(scores,float); labels=np.asarray(labels,int); det=(scores<thr).astype(int)
    tp=int(((det==1)&(labels==1)).sum()); fp=int(((det==1)&(labels==0)).sum()); tn=int(((det==0)&(labels==0)).sum()); fn=int(((det==0)&(labels==1)).sum())
    benign=scores[labels==0]; malicious=scores[labels==1]
    auc=float(roc_auc_score(labels,-scores)) if len(np.unique(labels))==2 else None
    return {"threshold":float(thr),"tp":tp,"fp":fp,"tn":tn,"fn":fn,"tpr":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1),"auroc_detection":auc,"scores":scores.tolist(),"benign":{"mean":float(np.mean(benign)),"median":float(np.median(benign))},"malicious":{"mean":float(np.mean(malicious)),"median":float(np.median(malicious))}}

def q05(x): return float(np.quantile(np.asarray(x,float),.05))
def q95(x): return max(float(np.quantile(np.asarray(x,float),.95)),1e-12)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=31415); ap.add_argument("--attack",choices=["sign_flip","scaling"],default="sign_flip"); ap.add_argument("--beta",type=float,default=.2); ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--clients",type=int,default=50); ap.add_argument("--alpha",type=float,default=.1); ap.add_argument("--cap",type=int,default=12000); ap.add_argument("--pretrain-epochs",type=int,default=3); ap.add_argument("--mu",type=float,default=.01); ap.add_argument("--ref-size",type=int,default=128); ap.add_argument("--ig-steps",type=int,default=16); ap.add_argument("--out",default="results/c4_v3_diagnostic.json"); a=ap.parse_args(); seed_all(a.seed); dev="cuda" if torch.cuda.is_available() else "cpu"
    path=ROOT/"data/external/ciciot2023_clean.parquet"; ensure(path); df=pd.read_parquet(path); lk={str(c).lower().strip():c for c in df}; y=pd.to_numeric(df[lk["is_attack"]]).astype("int64").to_numpy(); xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]); im=SimpleImputer(strategy="median"); X0=im.fit_transform(xd.iloc[tr]); V0=im.transform(xd.iloc[va]); E0=im.transform(xd.iloc[te]); sc=StandardScaler(); X=sc.fit_transform(X0).astype("float32"); V=sc.transform(V0).astype("float32"); E=sc.transform(E0).astype("float32"); Y=y[tr]; Vy=y[va]; Ey=y[te]
    model=XTrustMLP(X.shape[1],2); model,_=local_train(model,torch.from_numpy(X),torch.from_numpy(Y),epochs=a.pretrain_epochs,batch_size=1024,lr=1e-3,device=dev); pre=ev(model,E,Ey,dev); ps,mn=parts(Y,a.clients,a.alpha,a.seed); ps=cap(ps,a.cap,a.seed); ids=list(range(a.clients)); rng=np.random.default_rng(a.seed+404); nm=max(1,int(round(a.clients*a.beta))); bad=sorted(rng.choice(a.clients,nm,replace=False).astype(int).tolist()); badset=set(bad); rr=np.random.default_rng(a.seed+909).choice(len(V),min(a.ref_size,len(V)),replace=False); xr=torch.from_numpy(V[rr]); yr=torch.from_numpy(Vy[rr])
    # Frozen clean calibration: no attacked cohort is used to fit references, scales, regressors, or thresholds.
    clean=build_round_batch(model,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev)
    uref=fit_update_anomaly_reference(clean.updates); u0=score_update_anomaly_fixed(clean.updates,uref); fps=[]; losses=[]
    for d in clean.updates:
        cm=apply_delta(model,d); fps.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); losses.append(ref_loss(cm,xr,yr))
    eref=fit_explanation_anomaly_reference(fps); raw0=score_explanation_anomaly_fixed(fps,eref); hs=RobustScaler().fit(np.asarray(losses)[:,None]); hub=HuberRegressor().fit(hs.transform(np.asarray(losses)[:,None]),raw0); expected0=np.maximum(0.,hub.predict(hs.transform(np.asarray(losses)[:,None]))); cal0=np.maximum(0.,raw0-expected0)
    un=q95(u0); rn=q95(raw0); en=q95(cal0)
    clean_arms={"U":trust_from_anomaly(u0/un),"Eraw":trust_from_anomaly(raw0/rn),"Ecal":trust_from_anomaly(cal0/en),"U+Ecal":compute_client_scores(u0/un,cal0/en,previous_scores=None,temporal_gamma=0.0)}; clean_arms["U+Ecal+T"]=clean_arms["U+Ecal"].copy(); thresholds={k:q05(v) for k,v in clean_arms.items()}; prev=clean_arms["U+Ecal+T"].copy()
    history=[]; base=model
    for r in range(a.rounds):
        def attack(cid,d): return d if cid not in badset else (sign_flip(d,1.0) if a.attack=="sign_flip" else model_scaling(d,5.0))
        b=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,transform_update=attack); uf=score_update_anomaly_fixed(b.updates,uref); pf=[]; rl=[]
        for d in b.updates:
            cm=apply_delta(base,d); pf.append(attribution_fingerprint(cm,xr,yr,steps=a.ig_steps)); rl.append(ref_loss(cm,xr,yr))
        raw=score_explanation_anomaly_fixed(pf,eref); expected=np.maximum(0.,hub.predict(hs.transform(np.asarray(rl)[:,None]))); cal=np.maximum(0.,raw-expected)
        arms={"U":trust_from_anomaly(uf/un),"Eraw":trust_from_anomaly(raw/rn),"Ecal":trust_from_anomaly(cal/en),"U+Ecal":compute_client_scores(uf/un,cal/en,previous_scores=None,temporal_gamma=0.0)}; arms["U+Ecal+T"]=compute_client_scores(uf/un,cal/en,previous_scores=prev); prev=arms["U+Ecal+T"].copy(); labels=np.asarray([c in badset for c in b.client_ids],int); diagnostics={k:arm_stats(v,thresholds[k],labels) for k,v in arms.items()}
        # Utility uses the predeclared combined non-temporal arm; other arms remain diagnostic only.
        ag=aggregate_same_round(b,xtrust_scores=arms["U+Ecal"],xtrust_reject_threshold=thresholds["U+Ecal"],trim_ratio=.2,krum_f=nm); cand=candidate_models(base,ag); metrics={k:ev(v,E,Ey,dev) for k,v in cand.items()}
        history.append({"round":r+1,"signals":{"fixed_update_anomaly":uf.tolist(),"raw_explanation_anomaly":raw.tolist(),"calibrated_explanation_residual":cal.tolist()},"arms":diagnostics,"utility":metrics}); base=cand["xtrust_fl"]
    out={"experiment":"C4_v3_frozen_diagnostic","paper_final":False,"protocol":"docs/C4_V3_DIAGNOSTIC_PROTOCOL.md","config":vars(a),"dataset_sha256":SHA,"device":dev,"pretraining_test":pre,"partition_min_size":mn,"partition_sizes":[len(q) for q in ps],"malicious_client_ids":bad,"reference_indices_validation":rr.astype(int).tolist(),"clean_calibration":{"update_q95":un,"raw_explanation_q95":rn,"calibrated_residual_q95":en,"thresholds":thresholds,"clean_arm_scores":{k:v.tolist() for k,v in clean_arms.items()}},"history":history}
    dest=ROOT/a.out; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding="utf-8"); print("C4-v3 diagnostic finished:",dest)
    for z in history:
        print("round",z["round"],end=" "); print(" | ".join(f"{k}:TPR={v['tpr']:.3f},FPR={v['fpr']:.3f},AUC={v['auroc_detection']:.3f}" for k,v in z["arms"].items()))
if __name__=="__main__": main()
