"""XTrust-FL Final Study v2 frozen confirmatory runner.

One predetermined seed/beta condition per invocation. No tuning is permitted.
Balanced CE weights are computed once from the clean global training split and
used by warm-up, clean calibration, attacked local training, and FLTrust root
training. Each defense keeps an independent model trajectory.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,random,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_c4_corrected import seed_all,ensure,parts,cap,ev,ref_loss,SHA,DROP
from xtrust_fl.model import XTrustMLP
from xtrust_fl.round_engine import build_round_batch
from xtrust_fl.training import apply_delta
from xtrust_fl.attacks import sign_flip
from xtrust_fl.update_space import fit_update_anomaly_reference,score_update_anomaly_fixed
from xtrust_fl.explain import attribution_fingerprint,fit_explanation_anomaly_reference,score_explanation_anomaly_fixed
from xtrust_fl.scoring import compute_client_scores
from xtrust_fl.aggregate import fedavg_aggregate,clean_threshold_prefilter
from xtrust_fl.baselines import coordinate_median,trimmed_mean,krum,multi_krum,fltrust
from xtrust_fl.final_study import FINAL_SEEDS,FINAL_BETAS,PRIMARY_ALPHA,PREFILTER_QUANTILE,MIN_RETAINED,TRIM_RATIO,V2_WARMUP_ROUNDS,global_balanced_class_weights,clean_federated_warmup,malicious_client_ids,trusted_root_update
METHODS=('fedavg','coordinate_median','trimmed_mean','krum','multi_krum','fltrust','xtrust_u_median','xtrust_ecal_median','xtrust_uecal_median','xtrust_uecal_trimmed')
def pair_seed(s):
 random.seed(s);np.random.seed(s%(2**32-1));torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def q95(x):return max(float(np.quantile(np.asarray(x,float),.95)),1e-12)
def trust(a):a=np.asarray(a,float);return 1/(1+np.exp(-np.clip(2-a,-60,60)))
def git(args):
 try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return f'unavailable:{e}'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def det_stats(score,tau,labels):
 s=np.asarray(score,float);y=np.asarray(labels,int);d=(s<tau).astype(int);tp=int(((d==1)&(y==1)).sum());fp=int(((d==1)&(y==0)).sum());tn=int(((d==0)&(y==0)).sum());fn=int(((d==0)&(y==1)).sum())
 return {'threshold':float(tau),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'tpr':tp/max(tp+fn,1),'fpr':fp/max(fp+tn,1),'auroc_detection':float(roc_auc_score(y,-s)) if len(np.unique(y))==2 else None}
def fit_cal(base,X,Y,ps,ids,xr,yr,a,dev,cw):
 clean=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,class_weights=cw)
 ur=fit_update_anomaly_reference(clean.updates);u=score_update_anomaly_fixed(clean.updates,ur);fps=[];loss=[]
 for d in clean.updates:
  m=apply_delta(base,d);fps.append(attribution_fingerprint(m,xr,yr,steps=a.ig_steps));loss.append(ref_loss(m,xr,yr))
 er=fit_explanation_anomaly_reference(fps);raw=score_explanation_anomaly_fixed(fps,er);hs=RobustScaler().fit(np.asarray(loss)[:,None]);hub=HuberRegressor().fit(hs.transform(np.asarray(loss)[:,None]),raw);cal=np.maximum(0,raw-np.maximum(0,hub.predict(hs.transform(np.asarray(loss)[:,None]))));un,en=q95(u),q95(cal);U=trust(u/un);E=trust(cal/en);UE=compute_client_scores(u/un,cal/en,previous_scores=None,temporal_gamma=0)
 return {'ur':ur,'er':er,'hs':hs,'hub':hub,'un':un,'en':en,'clean':{'U':U,'Ecal':E,'U+Ecal':UE}}
def score_attacked(base,batch,cl,xr,yr,a):
 u=score_update_anomaly_fixed(batch.updates,cl['ur']);fps=[];loss=[]
 for d in batch.updates:
  m=apply_delta(base,d);fps.append(attribution_fingerprint(m,xr,yr,steps=a.ig_steps));loss.append(ref_loss(m,xr,yr))
 raw=score_explanation_anomaly_fixed(fps,cl['er']);cal=np.maximum(0,raw-np.maximum(0,cl['hub'].predict(cl['hs'].transform(np.asarray(loss)[:,None]))));U=trust(u/cl['un']);E=trust(cal/cl['en']);UE=compute_client_scores(u/cl['un'],cal/cl['en'],previous_scores=None,temporal_gamma=0)
 return {'U':U,'Ecal':E,'U+Ecal':UE}
def prefilter_delta(batch,score,clean_score,kind='median'):
 keep,d=clean_threshold_prefilter(score,clean_score,quantile=PREFILTER_QUANTILE,min_retained=MIN_RETAINED);uu=[batch.updates[int(i)] for i in keep];out=coordinate_median(uu) if kind=='median' else trimmed_mean(uu,TRIM_RATIO);return out,keep,d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--beta',type=float,required=True);ap.add_argument('--rounds',type=int,default=3);ap.add_argument('--clients',type=int,default=50);ap.add_argument('--alpha',type=float,default=PRIMARY_ALPHA);ap.add_argument('--cap',type=int,default=12000);ap.add_argument('--mu',type=float,default=.01);ap.add_argument('--ref-size',type=int,default=128);ap.add_argument('--ig-steps',type=int,default=16);ap.add_argument('--out',required=True);a=ap.parse_args()
 if a.seed not in FINAL_SEEDS:raise ValueError(f'seed must be frozen final seed {FINAL_SEEDS}')
 if not any(abs(a.beta-b)<1e-12 for b in FINAL_BETAS):raise ValueError(f'beta must be frozen final beta {FINAL_BETAS}')
 if abs(a.alpha-PRIMARY_ALPHA)>1e-12:raise ValueError('primary runner requires frozen alpha=0.10')
 seed_all(a.seed);dev='cuda' if torch.cuda.is_available() else 'cpu';path=ROOT/'data/external/ciciot2023_clean.parquet';ensure(path);actual=sha(path);assert actual==SHA
 df=pd.read_parquet(path);lk={str(c).lower().strip():c for c in df};y=pd.to_numeric(df[lk['is_attack']]).astype('int64').to_numpy();xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan);idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y);tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]);im=SimpleImputer(strategy='median');X0=im.fit_transform(xd.iloc[tr]);V0=im.transform(xd.iloc[va]);E0=im.transform(xd.iloc[te]);sc=StandardScaler();X=sc.fit_transform(X0).astype('float32');V=sc.transform(V0).astype('float32');E=sc.transform(E0).astype('float32');Y=y[tr];Vy=y[va];Ey=y[te]
 ps,mn=parts(Y,a.clients,a.alpha,a.seed);ps=cap(ps,a.cap,a.seed);ids=list(range(a.clients));yt=torch.from_numpy(Y);cw=global_balanced_class_weights(yt);init=XTrustMLP(X.shape[1],2);warm,warm_hist=clean_federated_warmup(init,torch.from_numpy(X),yt,ps,rounds=V2_WARMUP_ROUNDS,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,class_weights=cw);models={k:deepcopy(warm) for k in METHODS};bad=malicious_client_ids(a.clients,a.beta,a.seed);badset=set(bad);rr=np.random.default_rng(a.seed+909).choice(len(V),min(a.ref_size,len(V)),replace=False);xr=torch.from_numpy(V[rr]);yr=torch.from_numpy(Vy[rr]);history=[]
 for r in range(a.rounds):
  row={'round':r+1,'methods':{}}
  for name in METHODS:
   rt=time.perf_counter();base=models[name];pair_seed(a.seed+10000*(r+1)+101);cl=fit_cal(base,X,Y,ps,ids,xr,yr,a,dev,cw);pair_seed(a.seed+10000*(r+1)+202)
   def attack(cid,d):return sign_flip(d,1.) if cid in badset else d
   batch=build_round_batch(base,torch.from_numpy(X),yt,ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,transform_update=attack,class_weights=cw);labels=np.asarray([cid in badset for cid in batch.client_ids],int);scores=score_attacked(base,batch,cl,xr,yr,a);diag={}
   if name=='fedavg':delta=fedavg_aggregate(batch.updates,batch.sample_counts)
   elif name=='coordinate_median':delta=coordinate_median(batch.updates)
   elif name=='trimmed_mean':delta=trimmed_mean(batch.updates,TRIM_RATIO)
   elif name=='krum':delta=krum(batch.updates,len(bad))
   elif name=='multi_krum':delta=multi_krum(batch.updates,len(bad))
   elif name=='fltrust':
    pair_seed(a.seed+10000*(r+1)+303);root=trusted_root_update(base,xr,yr,epochs=1,batch_size=min(128,len(xr)),lr=1e-3,fedprox_mu=a.mu,device=dev,class_weights=cw);delta,ts=fltrust(batch.updates,root);diag['fltrust_trust']=ts.tolist()
   else:
    key={'xtrust_u_median':'U','xtrust_ecal_median':'Ecal','xtrust_uecal_median':'U+Ecal','xtrust_uecal_trimmed':'U+Ecal'}[name];kind='trimmed' if name.endswith('trimmed') else 'median';delta,keep,pd0=prefilter_delta(batch,scores[key],cl['clean'][key],kind);ret=labels[keep];diag.update(pd0);diag.update({'retained_count':int(len(keep)),'retrospective_benign_retained':int((ret==0).sum()),'retrospective_malicious_retained':int((ret==1).sum())})
   models[name]=apply_delta(base,delta);utility=ev(models[name],E,Ey,dev);det={}
   for key in ('U','Ecal','U+Ecal'):
    tau=float(np.quantile(cl['clean'][key],PREFILTER_QUANTILE));det[key]=det_stats(scores[key],tau,labels)
   row['methods'][name]={'utility':utility,'filter':diag,'detection':det,'seconds':time.perf_counter()-rt}
  history.append(row);print('round',r+1,{k:round(v['utility']['macro_f1'],4) for k,v in row['methods'].items()},flush=True)
 prov={'git_commit':git(['rev-parse','HEAD']),'git_status':git(['status','--short']),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'device':dev,'dataset_sha256':actual};out={'experiment':'XTrust_FL_Final_Study_v2','paper_final':True,'protocol':'docs/FINAL_STUDY_V2_FROZEN_PROTOCOL.md','config':{**vars(a),'warmup_rounds':V2_WARMUP_ROUNDS,'class_weight_formula':'N/(2*N_c)'},'class_weights':cw.tolist(),'partition_min_size':mn,'malicious_client_ids':bad,'warmup':warm_hist,'warmup_test':ev(warm,E,Ey,dev),'history':history,'provenance':prov};dest=ROOT/a.out;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print('FINAL v2 condition complete:',dest)
if __name__=='__main__':main()
