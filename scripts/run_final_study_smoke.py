"""Lightweight mechanics smoke test; never a paper result."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from xtrust_fl.model import XTrustMLP
from xtrust_fl.final_study import clean_federated_warmup,clone_trajectories,trusted_root_update
from xtrust_fl.round_engine import build_round_batch
from xtrust_fl.aggregate import fedavg_aggregate,clean_threshold_prefilter
from xtrust_fl.baselines import coordinate_median,fltrust
from xtrust_fl.training import apply_delta

def main():
 torch.manual_seed(123);np.random.seed(123);x=torch.randn(120,6);y=(x[:,0]+.3*x[:,1]>0).long();parts=[list(range(i*20,(i+1)*20)) for i in range(6)];m=XTrustMLP(6,2);warm,h=clean_federated_warmup(m,x,y,parts,rounds=1,batch_size=16,device='cpu');states=clone_trajectories(warm,['fedavg','median','fltrust']);b=build_round_batch(warm,x,y,parts,list(range(6)),batch_size=16,device='cpu');states['fedavg'].model=apply_delta(warm,fedavg_aggregate(b.updates,b.sample_counts));states['median'].model=apply_delta(warm,coordinate_median(b.updates));root=trusted_root_update(warm,x[:24],y[:24],batch_size=12,device='cpu');fd,ts=fltrust(b.updates,root);states['fltrust'].model=apply_delta(warm,fd);clean=np.asarray([.9,.8,.7,.6,.5,.4]);att=np.asarray([.9,.8,.7,.6,.1,.05]);keep,d=clean_threshold_prefilter(att,clean,.05,3);assert len(keep)>=3 and np.isfinite(ts).all();assert all(torch.isfinite(p).all() for s in states.values() for p in s.model.parameters());print('FINAL-STUDY SMOKE PASS',{'warmup':h,'retained':keep.tolist(),'threshold':d['threshold']})
if __name__=='__main__':main()
