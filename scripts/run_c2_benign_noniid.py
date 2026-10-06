"""C2: leakage-free benign severe Non-IID mechanism experiment.

Calibration/trusted clients alone define the explanation prototype, robust
feature scales, heterogeneity model, and thresholds. Held-out benign clients
never influence those quantities.
"""
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition, client_label_distributions
from xtrust_fl.explain import attribution_fingerprint, fit_explanation_anomaly_reference, score_explanation_anomaly_fixed
from xtrust_fl.heterogeneity import js_divergence_to_global, calibrate_explanation_residual
URL="https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"
SHA256="17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"; TARGET_DERIVED={"label","family","is_attack"}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_data(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(URL,path)
    d=hashlib.sha256(path.read_bytes()).hexdigest()
    if d!=SHA256: raise RuntimeError(f"SHA-256 mismatch: {d}")

def train_model(model,X,y,device,epochs=1,batch_size=512,lr=1e-3):
    m=deepcopy(model).to(device); loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=batch_size,shuffle=True)
    opt=torch.optim.Adam(m.parameters(),lr=lr); lossfn=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=lossfn(m(xb),yb); loss.backward(); opt.step()
    return m.cpu()

def q95(x): return float(np.quantile(np.asarray(x,dtype=float),.95))

def robust_dirichlet_partition(y,n,alpha,seed):
    last=None
    for minimum in [50,20,10,5,1]:
        try: return dirichlet_partition(y,n,alpha,seed,min_size=minimum,max_retries=5000),minimum
        except RuntimeError as e: last=e
    raise RuntimeError("Could not construct severe Non-IID partition") from last

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--alpha",type=float,default=.1); ap.add_argument("--clients",type=int,default=20); ap.add_argument("--pretrain-epochs",type=int,default=3); ap.add_argument("--local-epochs",type=int,default=1); ap.add_argument("--client-cap",type=int,default=12000); ap.add_argument("--ref-size",type=int,default=128); ap.add_argument("--ig-steps",type=int,default=16); ap.add_argument("--out",default="results/c2_v2_seed42.json"); a=ap.parse_args(); seed_all(a.seed)
    if a.clients<10: raise ValueError("C2 requires at least 10 clients")
    path=ROOT/"data"/"external"/"ciciot2023_clean.parquet"; ensure_data(path); df=pd.read_parquet(path); lookup={str(c).strip().lower():c for c in df.columns}; target=lookup.get("is_attack")
    if target is None: raise ValueError("is_attack target missing")
    y=pd.to_numeric(df[target],errors="raise").astype("int64").to_numpy(); dropped=[c for c in df.columns if str(c).strip().lower() in TARGET_DERIVED]; Xdf=df.drop(columns=dropped).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr])
    imp=SimpleImputer(strategy="median"); Xtr_i=imp.fit_transform(Xdf.iloc[tr]); Xva_i=imp.transform(Xdf.iloc[va]); sc=StandardScaler(); Xtr=sc.fit_transform(Xtr_i).astype("float32"); Xva=sc.transform(Xva_i).astype("float32"); ytr=y[tr]; yva=y[va]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); base=train_model(XTrustMLP(Xtr.shape[1],2),Xtr,ytr,device,epochs=a.pretrain_epochs)
    parts,accepted=robust_dirichlet_partition(ytr,a.clients,a.alpha,a.seed)
    if any(len(p)==0 for p in parts): raise RuntimeError("Partition contains an empty client")
    dists=client_label_distributions(ytr,parts); js=js_divergence_to_global(dists); sizes=np.asarray([len(p) for p in parts],dtype=float); qskew=np.abs(np.log((sizes+1)/(np.median(sizes)+1))); H=np.column_stack([js,qskew])
    rng=np.random.default_rng(a.seed+991); ri=rng.choice(len(Xva),size=min(a.ref_size,len(Xva)),replace=False); xref=torch.from_numpy(Xva[ri]); yref=torch.from_numpy(yva[ri])
    fps=[]; used=[]
    for cid,p in enumerate(parts):
        p=np.asarray(p,dtype=int)
        if len(p)>a.client_cap: p=np.random.default_rng(a.seed*1000+cid).choice(p,size=a.client_cap,replace=False)
        local=train_model(base,Xtr[p],ytr[p],device,epochs=a.local_epochs); fps.append(attribution_fingerprint(local,xref,yref,steps=a.ig_steps)); used.append(int(len(p)))
    perm=np.random.default_rng(a.seed+123).permutation(a.clients); ncal=max(5,a.clients//2); cal=np.sort(perm[:ncal]); ev=np.sort(perm[ncal:]); mask=np.zeros(a.clients,dtype=bool); mask[cal]=True
    topk=min(10,Xtr.shape[1]); ref=fit_explanation_anomaly_reference([fps[i] for i in cal],top_k=topk); raw=score_explanation_anomaly_fixed(fps,ref)
    residual,expected=calibrate_explanation_residual(raw,H,trusted_mask=mask,min_fit_clients=5); raw_thr=q95(raw[mask]); cal_thr=q95(residual[mask]); rf=raw[ev]>raw_thr; cf=residual[ev]>cal_thr
    result={"experiment":"C2_v2_leakage_free_benign_severe_nonIID","hypothesis":"heterogeneity calibration reduces benign false positives under severe Non-IID","provenance":{"url":URL,"sha256":SHA256},"integrity":{"no_malicious_clients":True,"target_metadata_dropped":list(map(str,dropped)),"imputer_fit":"train_only","scaler_fit":"train_only","prototype_source":"calibration_clients_only","anomaly_center_scale_source":"calibration_clients_only","heterogeneity_fit_source":"calibration_clients_only","threshold_source":"calibration_clients_only","evaluation_clients_in_reference":False},"config":vars(a),"device":str(device),"n_features":int(Xtr.shape[1]),"partition":{"requested_alpha":float(a.alpha),"accepted_min_client_size":int(accepted),"actual_min_client_size":int(sizes.min()),"actual_max_client_size":int(sizes.max())},"calibration_client_ids":cal.tolist(),"evaluation_client_ids":ev.tolist(),"client_sizes_original":sizes.astype(int).tolist(),"client_sizes_used":used,"heterogeneity":{"js_divergence":js.tolist(),"quantity_skew":qskew.tolist(),"js_mean":float(js.mean()),"js_max":float(js.max())},"raw_explanation_anomaly":raw.tolist(),"expected_explanation_anomaly":expected.tolist(),"calibrated_residual":residual.tolist(),"thresholds":{"raw_q95":raw_thr,"calibrated_q95":cal_thr},"evaluation":{"raw_false_positives":int(rf.sum()),"calibrated_false_positives":int(cf.sum()),"n_eval_benign_clients":int(len(ev)),"raw_fpr_benign":float(rf.mean()),"calibrated_fpr_benign":float(cf.mean()),"absolute_fpr_change":float(cf.mean()-rf.mean())}}
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
