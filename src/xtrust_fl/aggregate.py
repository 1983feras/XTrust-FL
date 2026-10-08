from __future__ import annotations

import numpy as np
import torch


def _validate_updates(updates: list[torch.Tensor]) -> None:
    if not updates: raise ValueError("updates must not be empty")
    shape=updates[0].shape
    if any(u.shape!=shape for u in updates): raise ValueError("all updates must have the same shape")
    if any(not torch.isfinite(u.detach()).all().item() for u in updates): raise ValueError("updates must contain only finite values")

def _validate_counts(sample_counts, expected_len):
    if len(sample_counts)!=expected_len: raise ValueError("sample_counts must align with updates")
    n=np.asarray(sample_counts,float)
    if n.ndim!=1 or not np.all(np.isfinite(n)) or np.any(n<=0): raise ValueError("sample_counts must be finite and strictly positive")
    return n

def adaptive_clip_threshold(updates:list[torch.Tensor],mad_k:float=2.5)->float:
    _validate_updates(updates); norms=np.asarray([torch.linalg.vector_norm(u.detach().cpu()).item() for u in updates],float); med=float(np.median(norms)); mad=float(np.median(np.abs(norms-med))); return max(med+mad_k*1.4826*mad,1e-12)

def clip_update(update:torch.Tensor,threshold:float)->torch.Tensor:
    if threshold<=0 or not np.isfinite(threshold): raise ValueError("threshold must be finite and positive")
    u=update.detach().cpu(); norm=torch.linalg.vector_norm(u).item(); return u if norm<=threshold else u*(threshold/(norm+1e-12))

def soft_trust_weights(client_scores:np.ndarray,q_min:float=.25)->np.ndarray:
    if not 0<q_min<=1: raise ValueError("q_min must be in (0, 1]")
    s=np.asarray(client_scores,float)
    if s.ndim!=1 or not np.all(np.isfinite(s)): raise ValueError("client_scores must be a finite 1-D array")
    return q_min+(1-q_min)*np.clip(s,0,1)

def relative_trust_weights(client_scores:np.ndarray,clean_scores:np.ndarray,q_min:float=.25):
    """C4-v6 frozen mapping. Uses CLEAN score distribution only; consumes no labels."""
    if not 0<q_min<=1: raise ValueError("q_min must be in (0, 1]")
    s=np.asarray(client_scores,float); c=np.asarray(clean_scores,float)
    if s.ndim!=1 or c.ndim!=1 or c.size==0 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(c)): raise ValueError("scores must be finite 1-D arrays and clean_scores nonempty")
    med=float(np.median(c)); mad=float(np.median(np.abs(c-med))); sigma=max(1.4826*mad,1e-6); z=(s-med)/sigma; r=1/(1+np.exp(-np.clip(z,-60,60))); q=q_min+(1-q_min)*r
    return q,{"clean_median":med,"clean_mad":mad,"clean_sigma":sigma,"z":z.tolist(),"relative_sigmoid":r.tolist()}

def clean_threshold_prefilter(client_scores:np.ndarray,clean_scores:np.ndarray,quantile:float=.05,min_retained:int=3):
    """C4-v7 frozen prefilter: clean-only quantile threshold with deterministic top-score fallback."""
    s=np.asarray(client_scores,float); c=np.asarray(clean_scores,float)
    if s.ndim!=1 or c.ndim!=1 or c.size==0 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(c)): raise ValueError("scores must be finite 1-D arrays and clean_scores nonempty")
    if not 0<=quantile<=1: raise ValueError("quantile must be in [0,1]")
    if min_retained<1 or min_retained>len(s): raise ValueError("min_retained must be between 1 and number of clients")
    tau=float(np.quantile(c,quantile)); keep=np.flatnonzero(s>=tau)
    fallback=False
    if keep.size<min_retained:
        order=np.argsort(-s,kind='stable'); keep=np.sort(order[:min_retained]); fallback=True
    return keep.astype(int),{"threshold":tau,"quantile":float(quantile),"min_retained":int(min_retained),"fallback_used":fallback}

def _aggregate_with_q(updates,sample_counts,q,*,clip=False,mad_k=2.5,return_diagnostics=False,extra=None):
    _validate_updates(updates)
    if len(q)!=len(updates): raise ValueError("inputs must align")
    n=_validate_counts(sample_counts,len(updates)); q=np.asarray(q,float)
    if q.ndim!=1 or not np.all(np.isfinite(q)): raise ValueError("weights must be finite")
    threshold=adaptive_clip_threshold(updates,mad_k) if clip else None; used=[]; before=[]; after=[]; factors=[]
    for u in updates:
        u0=u.detach().cpu(); b=float(torch.linalg.vector_norm(u0)); u1=clip_update(u0,threshold) if clip else u0; a=float(torch.linalg.vector_norm(u1)); used.append(u1); before.append(b); after.append(a); factors.append(1. if b<=1e-12 else a/b)
    raw=n*q; den=float(raw.sum())
    if not np.isfinite(den) or den<=0: raise RuntimeError("aggregation denominator must be finite and positive")
    w=raw/den; out=torch.zeros_like(used[0])
    for wi,ui in zip(w,used): out=out+float(wi)*ui
    if not return_diagnostics:return out
    d={"q":q.tolist(),"sample_counts":n.astype(int).tolist(),"raw_weights":raw.tolist(),"normalized_weights":w.tolist(),"clip":bool(clip),"clip_threshold":None if threshold is None else float(threshold),"norm_before":before,"norm_after":after,"clip_factor":factors,"weight_mass":den}; d.update(extra or {}); return out,d

def soft_trust_aggregate(updates,sample_counts,client_scores,*,q_min=.25,clip=False,mad_k=2.5,return_diagnostics=False):
    q=soft_trust_weights(client_scores,q_min); return _aggregate_with_q(updates,sample_counts,q,clip=clip,mad_k=mad_k,return_diagnostics=return_diagnostics,extra={"q_min":float(q_min),"mapping":"absolute_soft"})

def relative_trust_aggregate(updates,sample_counts,client_scores,clean_scores,*,q_min=.25,clip=False,mad_k=2.5,return_diagnostics=False):
    q,cal=relative_trust_weights(client_scores,clean_scores,q_min); cal.update({"q_min":float(q_min),"mapping":"clean_relative_sigmoid"}); return _aggregate_with_q(updates,sample_counts,q,clip=clip,mad_k=mad_k,return_diagnostics=return_diagnostics,extra=cal)

def weighted_clipped_aggregate(updates:list[torch.Tensor],sample_counts:list[int],client_scores:np.ndarray,reject_threshold:float=.15,mad_k:float=2.5)->torch.Tensor:
    _validate_updates(updates); n=_validate_counts(sample_counts,len(updates))
    scores=np.asarray(client_scores,float)
    if scores.ndim!=1 or len(scores)!=len(updates) or not np.all(np.isfinite(scores)): raise ValueError("client_scores must align with updates and be finite")
    threshold=adaptive_clip_threshold(updates,mad_k); weights=[]; clipped=[]
    for u,ni,s in zip(updates,n,scores): weights.append(float(ni)*float(s) if float(s)>=reject_threshold else 0.); clipped.append(clip_update(u,threshold))
    weights=np.asarray(weights,float)
    if weights.sum()<=0:return torch.zeros_like(clipped[0])
    weights/=weights.sum(); out=torch.zeros_like(clipped[0])
    for w,u in zip(weights,clipped):out=out+float(w)*u
    return out

def fedavg_aggregate(updates:list[torch.Tensor],sample_counts:list[int])->torch.Tensor:
    _validate_updates(updates); w=_validate_counts(sample_counts,len(updates)); w/=w.sum(); out=torch.zeros_like(updates[0].detach().cpu())
    for wi,u in zip(w,updates):out=out+float(wi)*u.detach().cpu()
    return out
