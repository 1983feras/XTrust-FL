import numpy as np
import torch
from xtrust_fl.model import XTrustMLP
from xtrust_fl.final_study import FINAL_SEEDS,FINAL_BETAS,PREFILTER_QUANTILE,MIN_RETAINED,TRIM_RATIO,malicious_client_ids,clone_trajectories,trusted_root_update


def test_frozen_grid_constants():
    assert FINAL_SEEDS==(42,77,100,999,2026)
    assert FINAL_BETAS==(0.0,.10,.20,.30,.40)
    assert PREFILTER_QUANTILE==.05
    assert MIN_RETAINED==3
    assert TRIM_RATIO==.20


def test_malicious_assignment_deterministic_and_clean_control():
    assert malicious_client_ids(50,0.0,42)==[]
    a=malicious_client_ids(50,.2,42);b=malicious_client_ids(50,.2,42)
    assert a==b and len(a)==10 and len(set(a))==10


def test_independent_trajectory_models():
    m=XTrustMLP(4,2); states=clone_trajectories(m,['fedavg','xtrust'])
    assert states['fedavg'].model is not states['xtrust'].model
    p0=next(states['fedavg'].model.parameters());p1=next(states['xtrust'].model.parameters())
    with torch.no_grad():p0.add_(1)
    assert not torch.equal(p0,p1)


def test_trusted_root_update_is_finite_nonzero():
    torch.manual_seed(7);m=XTrustMLP(4,2);x=torch.randn(24,4);y=(x[:,0]>0).long()
    d=trusted_root_update(m,x,y,epochs=1,batch_size=8,lr=1e-3,device='cpu')
    assert torch.isfinite(d).all();assert float(torch.linalg.vector_norm(d))>0


def test_beta_guardrail():
    for bad in (-.1,.5,1.0):
        try: malicious_client_ids(50,bad,42)
        except ValueError: pass
        else: raise AssertionError('invalid beta accepted')
