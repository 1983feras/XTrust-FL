"""Helpers for the frozen XTrust-FL Final Study v1.

Scientific constants live in the frozen protocol. This module provides
mechanics only: clean federated warm-up, deterministic attack assignment,
clean-root update construction, and method-specific trajectory state.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .aggregate import fedavg_aggregate
from .round_engine import build_round_batch
from .training import apply_delta, local_train, parameter_delta

FINAL_SEEDS = (42, 77, 100, 999, 2026)
FINAL_BETAS = (0.0, 0.10, 0.20, 0.30, 0.40)
PRIMARY_ALPHA = 0.10
PREFILTER_QUANTILE = 0.05
MIN_RETAINED = 3
TRIM_RATIO = 0.20

@dataclass
class TrajectoryState:
    name: str
    model: nn.Module


def malicious_client_ids(num_clients: int, beta: float, seed: int) -> list[int]:
    if num_clients < 1: raise ValueError("num_clients must be positive")
    if not 0.0 <= beta < 0.5: raise ValueError("beta must be in [0, 0.5)")
    n_bad=int(round(num_clients*beta))
    if n_bad==0:return []
    rng=np.random.default_rng(seed+404)
    return sorted(rng.choice(num_clients,n_bad,replace=False).astype(int).tolist())


def clean_federated_warmup(initial_model, x_train, y_train, partitions, *, rounds=3, epochs=1, batch_size=512, lr=1e-3, fedprox_mu=.01, device='cpu'):
    """Benign FedAvg warm-up; never consumes attack labels or attacked outcomes."""
    if rounds < 1: raise ValueError("rounds must be positive")
    model=deepcopy(initial_model).cpu(); ids=list(range(len(partitions))); history=[]
    for r in range(rounds):
        batch=build_round_batch(model,x_train,y_train,partitions,ids,epochs=epochs,batch_size=batch_size,lr=lr,fedprox_mu=fedprox_mu,device=device)
        delta=fedavg_aggregate(batch.updates,batch.sample_counts)
        model=apply_delta(model,delta)
        history.append({'round':r+1,'clients':len(batch.client_ids),'sample_mass':int(sum(batch.sample_counts))})
    return model,history


def trusted_root_update(global_model, x_ref: torch.Tensor, y_ref: torch.Tensor, *, epochs=1, batch_size=128, lr=1e-3, fedprox_mu=.01, device='cpu'):
    """FLTrust-style trusted root update from server-held clean reference data."""
    if len(x_ref)==0: raise ValueError("trusted reference must be nonempty")
    local,_=local_train(global_model,x_ref,y_ref,epochs=epochs,batch_size=batch_size,lr=lr,fedprox_mu=fedprox_mu,device=device)
    return parameter_delta(local,global_model)


def clone_trajectories(base_model, names):
    names=list(names)
    if len(set(names))!=len(names): raise ValueError("trajectory names must be unique")
    return {name:TrajectoryState(name,deepcopy(base_model).cpu()) for name in names}
