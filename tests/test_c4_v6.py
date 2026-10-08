import inspect
import numpy as np
import torch
from xtrust_fl.aggregate import relative_trust_weights, relative_trust_aggregate

def test_relative_weights_bounded_and_finite():
    q,d=relative_trust_weights(np.array([.1,.5,.9]),np.array([.4,.5,.6]),.25); assert np.all(np.isfinite(q)); assert np.all(q>=.25); assert np.all(q<=1); assert d['clean_sigma']>0

def test_monotonic_mapping():
    s=np.linspace(0,1,101); q,_=relative_trust_weights(s,np.array([.4,.5,.6]),.25); assert np.all(np.diff(q)>=0)

def test_zero_mad_safe():
    q,d=relative_trust_weights(np.array([.4,.5,.6]),np.array([.5,.5,.5]),.25); assert np.all(np.isfinite(q)); assert d['clean_sigma']==1e-6

def test_equal_scores_equal_q():
    q,_=relative_trust_weights(np.array([.7,.7]),np.array([.5,.6,.7]),.25); assert q[0]==q[1]

def test_aggregate_deterministic_and_clip_nonincrease():
    u=[torch.tensor([3.,4.]),torch.tensor([1.,0.])]; s=np.array([.4,.8]); c=np.array([.5,.6,.7]); a,d=relative_trust_aggregate(u,[2,3],s,c,clip=True,return_diagnostics=True); b=relative_trust_aggregate(u,[2,3],s,c,clip=True); assert torch.equal(a,b); assert all(x<=y+1e-10 for x,y in zip(d['norm_after'],d['norm_before']))

def test_mapping_api_has_no_labels():
    sig=str(inspect.signature(relative_trust_weights)); assert 'label' not in sig and 'malicious' not in sig
