import numpy as np
import torch
from xtrust_fl.aggregate import soft_trust_weights, soft_trust_aggregate, adaptive_clip_threshold

def U(): return [torch.tensor([3.,4.]),torch.tensor([.3,.4]),torch.tensor([1.,2.])]

def test_weights_bounded_finite():
    q=soft_trust_weights(np.array([-1.,.5,2.]),q_min=.25); assert np.all(np.isfinite(q)); assert np.all(q>=.25); assert np.all(q<=1)

def test_clip_never_increases_norm():
    _,d=soft_trust_aggregate(U(),[1,1,1],np.array([.2,.5,.8]),clip=True,return_diagnostics=True); assert all(a<=b+1e-10 for a,b in zip(d['norm_after'],d['norm_before']))

def test_equal_scores_reduce_to_sample_weighted_no_clip():
    u=U(); n=[1,2,3]; s=np.array([.7,.7,.7]); got=soft_trust_aggregate(u,n,s,clip=False); expected=sum((ni/sum(n))*ui for ni,ui in zip(n,u)); assert torch.allclose(got,expected)

def test_near_zero_update_safe():
    u=[torch.zeros(2),torch.tensor([1.,0.])]; out,d=soft_trust_aggregate(u,[1,1],np.array([0.,1.]),clip=True,return_diagnostics=True); assert torch.isfinite(out).all(); assert np.all(np.isfinite(d['clip_factor']))

def test_denominator_positive_and_deterministic():
    a=soft_trust_aggregate(U(),[1,2,3],np.array([.1,.2,.3]),clip=True); b=soft_trust_aggregate(U(),[1,2,3],np.array([.1,.2,.3]),clip=True); assert torch.equal(a,b)

def test_no_label_argument_in_api():
    import inspect
    sig=str(inspect.signature(soft_trust_aggregate)); assert 'label' not in sig and 'malicious' not in sig

def test_adaptive_threshold_positive(): assert adaptive_clip_threshold(U())>0
