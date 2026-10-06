"""Run a reproducible real-data baseline on the public cleaned CICIoT2023 Parquet artifact.

Data provenance:
  arpangl/IoT2026, lab2/data/ciciot2023_clean.parquet
  SHA-256 documented upstream: 17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1

This script downloads the artifact, verifies SHA-256, then trains/evaluates the
XTrust MLP. Results are measured results, not synthetic placeholders.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from xtrust_fl.model import XTrustMLP

URL='https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet'
SHA256='17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def ensure_data(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(URL,path)
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    if h != SHA256: raise RuntimeError(f'SHA-256 mismatch: {h}')

def metrics(model,X,y,device):
    model.eval()
    with torch.no_grad():
        p=torch.softmax(model(torch.as_tensor(X,dtype=torch.float32,device=device)),1)[:,1].cpu().numpy()
    pred=(p>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'confusion_matrix':confusion_matrix(y,pred).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--out',default='results/public_parquet_seed42.json'); a=ap.parse_args(); seed_all(a.seed)
    path=ROOT/'data'/'external'/'ciciot2023_clean.parquet'; ensure_data(path); df=pd.read_parquet(path)
    target='is_attack' if 'is_attack' in df.columns else ('label' if 'label' in df.columns else None)
    if target is None: raise ValueError(f'No binary target found. Columns={list(df.columns)}')
    y=pd.to_numeric(df[target],errors='raise').astype('int64').to_numpy(); X=df.drop(columns=[target])
    # Exclude textual/original-label metadata from predictors; retain numeric traffic features only.
    X=X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).dropna(axis=1,how='all')
    X=X.fillna(X.median(numeric_only=True)).astype('float32')
    idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y); tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr])
    scaler=StandardScaler(); Xn=X.to_numpy(); xtr=scaler.fit_transform(Xn[tr]).astype('float32'); xva=scaler.transform(Xn[va]).astype('float32'); xte=scaler.transform(Xn[te]).astype('float32')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=XTrustMLP(xtr.shape[1],2).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossfn=torch.nn.CrossEntropyLoss(); loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(y[tr])),batch_size=a.batch_size,shuffle=True)
    best=-1.; state=None
    for _ in range(a.epochs):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        score=metrics(model,xva,y[va],device)['macro_f1']
        if score>best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); result={'provenance':{'url':URL,'sha256':SHA256},'seed':a.seed,'rows':int(len(df)),'numeric_features':int(xtr.shape[1]),'split':{'train':int(len(tr)),'val':int(len(va)),'test':int(len(te))},'best_val_macro_f1':best,'test':metrics(model,xte,y[te],device)}
    out=ROOT/a.out; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
