"""Final Study v2 mandatory clean-control gate: seed=42, beta=0 only."""
from __future__ import annotations
import hashlib,json,platform,random,subprocess,sys
from copy import deepcopy
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_c4_corrected import seed_all,ensure,parts,cap,ev,SHA,DROP
from xtrust_fl.model import XTrustMLP
from xtrust_fl.final_study import V2_WARMUP_ROUNDS,global_balanced_class_weights,clean_federated_warmup
SEED=42;BETA=0.0;ALPHA=.10;CLIENTS=50;CAP=12000;MU=.01

def git(args):
 try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return f'unavailable:{e}'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def adequate(v):
 cm=v['confusion'];pred=sorted(set(np.argmax(np.asarray(cm),axis=0).tolist())) if False else None
 return cm[0][0]>0 and cm[1][1]>0 and v['macro_f1']>=.70 and v['auroc']>=.90
def main():
 seed_all(SEED);dev='cuda' if torch.cuda.is_available() else 'cpu';path=ROOT/'data/external/ciciot2023_clean.parquet';ensure(path);actual=sha(path);assert actual==SHA
 df=pd.read_parquet(path);lk={str(c).lower().strip():c for c in df};y=pd.to_numeric(df[lk['is_attack']]).astype('int64').to_numpy();xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan);idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.2,random_state=SEED,stratify=y);tr,va=train_test_split(tr,test_size=.125,random_state=SEED,stratify=y[tr]);im=SimpleImputer(strategy='median');X0=im.fit_transform(xd.iloc[tr]);V0=im.transform(xd.iloc[va]);E0=im.transform(xd.iloc[te]);sc=StandardScaler();X=sc.fit_transform(X0).astype('float32');V=sc.transform(V0).astype('float32');E=sc.transform(E0).astype('float32');Y=y[tr];Vy=y[va];Ey=y[te]
 ps,mn=parts(Y,CLIENTS,ALPHA,SEED);ps=cap(ps,CAP,SEED);yt=torch.from_numpy(Y);cw=global_balanced_class_weights(yt);init=XTrustMLP(X.shape[1],2);warm,hist=clean_federated_warmup(init,torch.from_numpy(X),yt,ps,rounds=V2_WARMUP_ROUNDS,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=MU,device=dev,class_weights=cw);val=ev(warm,V,Vy,dev);test=ev(warm,E,Ey,dev);ok=adequate(val)
 out={'experiment':'XTrust_FL_Final_Study_v2_clean_control','paper_final':True,'protocol':'docs/FINAL_STUDY_V2_FROZEN_PROTOCOL.md','config':{'seed':SEED,'beta':BETA,'alpha':ALPHA,'clients':CLIENTS,'cap':CAP,'mu':MU,'warmup_rounds':V2_WARMUP_ROUNDS,'class_weight_formula':'N/(2*N_c)'},'class_weights':cw.tolist(),'partition_min_size':mn,'warmup':hist,'validation':val,'test':test,'clean_control_pass':bool(ok),'provenance':{'git_commit':git(['rev-parse','HEAD']),'git_status':git(['status','--short']),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'device':dev,'dataset_sha256':actual}}
 dest=ROOT/'results/final_v2_clean_control_seed42_beta0.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({'clean_control_pass':ok,'validation':val,'test':test,'class_weights':cw.tolist()},indent=2),flush=True);print('RESULT=',dest,flush=True)
 if not ok:raise SystemExit('STOP: Final Study v2 clean-control gate failed; beta>0 is forbidden.')
 print('PASS: Final Study v2 clean-control gate. Attacked grid may be implemented/executed only from the frozen v2 protocol.',flush=True)
if __name__=='__main__':main()
