"""C4-v7 frozen diagnostic: unchanged U+Ecal detector -> clean-threshold prefilter -> robust aggregation."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess,sys,time
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
from xtrust_fl.round_engine import build_round_batch,aggregate_same_round,candidate_models
from xtrust_fl.aggregate import relative_trust_aggregate,clean_threshold_prefilter
from xtrust_fl.baselines import coordinate_median,trimmed_mean
from xtrust_fl.training import local_train,apply_delta
from xtrust_fl.attacks import sign_flip,model_scaling
from xtrust_fl.update_space import fit_update_anomaly_reference,score_update_anomaly_fixed
from xtrust_fl.explain import attribution_fingerprint,fit_explanation_anomaly_reference,score_explanation_anomaly_fixed
from xtrust_fl.scoring import compute_client_scores
QMIN=.25; PREFILTER_Q=.05; MIN_RETAINED=3; TRIM=.20

def trust(a):a=np.asarray(a,float);return 1/(1+np.exp(-np.clip(2-a,-60,60)))
def q05(x):return float(np.quantile(np.asarray(x,float),.05))
def q95(x):return max(float(np.quantile(np.asarray(x,float),.95)),1e-12)
def stats(s,t,y):
 s=np.asarray(s,float);y=np.asarray(y,int);d=(s<t).astype(int);tp=int(((d==1)&(y==1)).sum());fp=int(((d==1)&(y==0)).sum());tn=int(((d==0)&(y==0)).sum());fn=int(((d==0)&(y==1)).sum());return {'threshold':float(t),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'tpr':tp/max(tp+fn,1),'fpr':fp/max(fp+tn,1),'auroc_detection':float(roc_auc_score(y,-s)) if len(np.unique(y))==2 else None,'scores':s.tolist()}
def fit_clean(base,X,Y,ps,ids,xr,yr,a,dev):
 clean=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev);ur=fit_update_anomaly_reference(clean.updates);u=score_update_anomaly_fixed(clean.updates,ur);fps=[];loss=[]
 for d in clean.updates:
  m=apply_delta(base,d);fps.append(attribution_fingerprint(m,xr,yr,steps=a.ig_steps));loss.append(ref_loss(m,xr,yr))
 er=fit_explanation_anomaly_reference(fps);raw=score_explanation_anomaly_fixed(fps,er);hs=RobustScaler().fit(np.asarray(loss)[:,None]);hub=HuberRegressor().fit(hs.transform(np.asarray(loss)[:,None]),raw);cal=np.maximum(0,raw-np.maximum(0,hub.predict(hs.transform(np.asarray(loss)[:,None]))));un,rn,en=q95(u),q95(raw),q95(cal);arms={'U':trust(u/un),'Eraw':trust(raw/rn),'Ecal':trust(cal/en),'U+Ecal':compute_client_scores(u/un,cal/en,previous_scores=None,temporal_gamma=0)};arms['U+Ecal+T']=arms['U+Ecal'].copy();return {'ur':ur,'er':er,'hs':hs,'hub':hub,'un':un,'rn':rn,'en':en,'arms':arms,'th':{k:q05(v) for k,v in arms.items()}}
def git(cmd):
 try:return subprocess.check_output(['git',*cmd],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return f'unavailable:{e}'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=31415);ap.add_argument('--attack',choices=['sign_flip','scaling'],default='sign_flip');ap.add_argument('--beta',type=float,default=.2);ap.add_argument('--rounds',type=int,default=3);ap.add_argument('--clients',type=int,default=50);ap.add_argument('--alpha',type=float,default=.1);ap.add_argument('--cap',type=int,default=12000);ap.add_argument('--pretrain-epochs',type=int,default=3);ap.add_argument('--mu',type=float,default=.01);ap.add_argument('--ref-size',type=int,default=128);ap.add_argument('--ig-steps',type=int,default=16);ap.add_argument('--out',default='results/c4_v7_diagnostic_seed31415.json');a=ap.parse_args();seed_all(a.seed);dev='cuda' if torch.cuda.is_available() else 'cpu';path=ROOT/'data/external/ciciot2023_clean.parquet';ensure(path);actual=sha(path);assert actual==SHA
 df=pd.read_parquet(path);lk={str(c).lower().strip():c for c in df};y=pd.to_numeric(df[lk['is_attack']]).astype('int64').to_numpy();xd=df.drop(columns=[c for c in df if str(c).lower().strip() in DROP]).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan);idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.2,random_state=a.seed,stratify=y);tr,va=train_test_split(tr,test_size=.125,random_state=a.seed,stratify=y[tr]);im=SimpleImputer(strategy='median');X0=im.fit_transform(xd.iloc[tr]);V0=im.transform(xd.iloc[va]);E0=im.transform(xd.iloc[te]);sc=StandardScaler();X=sc.fit_transform(X0).astype('float32');V=sc.transform(V0).astype('float32');E=sc.transform(E0).astype('float32');Y=y[tr];Vy=y[va];Ey=y[te]
 base=XTrustMLP(X.shape[1],2);base,_=local_train(base,torch.from_numpy(X),torch.from_numpy(Y),epochs=a.pretrain_epochs,batch_size=1024,lr=1e-3,device=dev);pre=ev(base,E,Ey,dev);ps,mn=parts(Y,a.clients,a.alpha,a.seed);ps=cap(ps,a.cap,a.seed);ids=list(range(a.clients));rng=np.random.default_rng(a.seed+404);nm=max(1,int(round(a.clients*a.beta)));bad=sorted(rng.choice(a.clients,nm,replace=False).astype(int).tolist());badset=set(bad);rr=np.random.default_rng(a.seed+909).choice(len(V),min(a.ref_size,len(V)),replace=False);xr=torch.from_numpy(V[rr]);yr=torch.from_numpy(Vy[rr]);hist=[];prev=None
 for r in range(a.rounds):
  rt=time.perf_counter();cl=fit_clean(base,X,Y,ps,ids,xr,yr,a,dev)
  def attack(cid,d):return d if cid not in badset else(sign_flip(d,1.) if a.attack=='sign_flip' else model_scaling(d,5.))
  b=build_round_batch(base,torch.from_numpy(X),torch.from_numpy(Y),ps,ids,epochs=1,batch_size=512,lr=1e-3,fedprox_mu=a.mu,device=dev,transform_update=attack);u=score_update_anomaly_fixed(b.updates,cl['ur']);fps=[];loss=[]
  for d in b.updates:
   m=apply_delta(base,d);fps.append(attribution_fingerprint(m,xr,yr,steps=a.ig_steps));loss.append(ref_loss(m,xr,yr))
  raw=score_explanation_anomaly_fixed(fps,cl['er']);cal=np.maximum(0,raw-np.maximum(0,cl['hub'].predict(cl['hs'].transform(np.asarray(loss)[:,None]))));arms={'U':trust(u/cl['un']),'Eraw':trust(raw/cl['rn']),'Ecal':trust(cal/cl['en']),'U+Ecal':compute_client_scores(u/cl['un'],cal/cl['en'],previous_scores=None,temporal_gamma=0)};arms['U+Ecal+T']=compute_client_scores(u/cl['un'],cal/cl['en'],previous_scores=prev) if prev is not None else arms['U+Ecal'].copy();prev=arms['U+Ecal+T'].copy();labels=np.asarray([cid in badset for cid in b.client_ids],int)
  ag=aggregate_same_round(b,xtrust_scores=arms['U+Ecal'],xtrust_reject_threshold=cl['th']['U+Ecal'],trim_ratio=TRIM,krum_f=nm);r0,e0=relative_trust_aggregate(b.updates,b.sample_counts,arms['U+Ecal'],cl['arms']['U+Ecal'],q_min=QMIN,clip=False,return_diagnostics=True);r1,e1=relative_trust_aggregate(b.updates,b.sample_counts,arms['U+Ecal'],cl['arms']['U+Ecal'],q_min=QMIN,clip=True,return_diagnostics=True);ag.update({'xtrust_relative_soft':r0,'xtrust_relative_soft_clip':r1})
  keep,pref_diag=clean_threshold_prefilter(arms['U+Ecal'],cl['arms']['U+Ecal'],quantile=PREFILTER_Q,min_retained=MIN_RETAINED);ku=[b.updates[i] for i in keep];ag['xtrust_prefilter_median']=coordinate_median(ku);ag['xtrust_prefilter_trimmed']=trimmed_mean(ku,trim_ratio=TRIM) if len(ku)>=5 else torch.stack(ku).mean(dim=0)
  cand=candidate_models(base,ag);metrics={k:ev(v,E,Ey,dev) for k,v in cand.items()};ret_labels=labels[keep];norms=np.asarray([float(torch.linalg.vector_norm(b.updates[i])) for i in keep]);pref_diag.update({'retained_indices':keep.tolist(),'retained_client_ids':[int(b.client_ids[i]) for i in keep],'excluded_client_ids':[int(cid) for j,cid in enumerate(b.client_ids) if j not in set(keep.tolist())],'retained_count':int(len(keep)),'excluded_count':int(len(b.client_ids)-len(keep)),'retrospective_benign_retained':int((ret_labels==0).sum()),'retrospective_malicious_retained':int((ret_labels==1).sum()),'retained_sample_mass':int(sum(b.sample_counts[i] for i in keep)),'retained_norm_mean':float(norms.mean()),'retained_norm_median':float(np.median(norms))})
  hist.append({'round':r+1,'calibration':{'thresholds':cl['th'],'clean_U+Ecal_scores':cl['arms']['U+Ecal'].tolist()},'arms':{k:stats(v,cl['th'][k],labels) for k,v in arms.items()},'prefilter_diagnostics':pref_diag,'aggregation_diagnostics':{'relative_soft':e0,'relative_soft_clip':e1},'utility':metrics,'round_seconds':time.perf_counter()-rt});base=cand['xtrust_prefilter_median']
 prov={'git_commit':git(['rev-parse','HEAD']),'git_status':git(['status','--short']),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'device':dev,'dataset_sha256':actual};out={'experiment':'C4_v7_xai_prefilter_robust_diagnostic','paper_final':False,'protocol':'docs/C4_V7_FROZEN_PROTOCOL.md','prefilter_quantile':PREFILTER_Q,'min_retained':MIN_RETAINED,'trim_ratio':TRIM,'config':vars(a),'provenance':prov,'pretraining_test':pre,'partition_min_size':mn,'malicious_client_ids':bad,'history':hist};dest=ROOT/a.out;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2),encoding='utf-8');(ROOT/'PROVENANCE.json').write_text(json.dumps(prov,indent=2),encoding='utf-8');print('C4-v7 complete',dest)
 for z in hist:print('round',z['round'],'retained',z['prefilter_diagnostics']['retained_count'],'mal_retained',z['prefilter_diagnostics']['retrospective_malicious_retained'],'detector AUC',z['arms']['U+Ecal']['auroc_detection'],'MacroF1',{k:round(v['macro_f1'],4) for k,v in z['utility'].items()})
if __name__=='__main__':main()
