"""Helpers for the frozen XTrust-FL Final Study v2."""
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
V2_WARMUP_ROUNDS = 6

@dataclass
class TrajectoryState:
    name: str
    model: nn.Module


def global_balanced_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    """Frozen v2 weights w_c=N/(2*N_c), using clean global training labels only."""
    y=y_train.detach().cpu().long().reshape(-1)
    counts=torch.bincount(y,minlength=2).to(torch.float64)
    if len(counts)!=2 or torch.any(counts<=0): raise ValueError('v2 requires both binary classes in clean training split')
    n=float(y.numel())
    return torch.tensor([n/(2.0*float(counts[0])),n/(2.0*float(counts[1]))],dtype=torch.float32)


def malicious_client_ids(num_clients: int, beta: float, seed: int) -> list[int]:
    if num_clients < 1: raise ValueError("num_clients must be positive")
    if not 0.0 <= beta < 0.5: raise ValueError("beta must be in [0, 0.5)")
    n_bad=int(round(num_clients*beta))
    if n_bad==0:return []
    rng=np.random.default_rng(seed+404)
    return sorted(rng.choice(num_clients,n_bad,replace=False).astype(int).tolist())


def clean_federated_warmup(initial_model, x_train, y_train, partitions, *, rounds=V2_WARMUP_ROUNDS, epochs=1, batch_size=512, lr=1e-3, fedprox_mu=.01, device='cpu', class_weights=None):
    """Benign balanced FedAvg warm-up; never consumes attack labels/outcomes."""
    if rounds != V2_WARMUP_ROUNDS: raise ValueError(f'Final Study v2 requires exactly {V2_WARMUP_ROUNDS} warm-up rounds')
    if class_weights is None: class_weights=global_balanced_class_weights(y_train)
    model=deepcopy(initial_model).cpu(); ids=list(range(len(partitions))); history=[]
    for r in range(rounds):
        batch=build_round_batch(model,x_train,y_train,partitions,ids,epochs=epochs,batch_size=batch_size,lr=lr,fedprox_mu=fedprox_mu,device=device,class_weights=class_weights)
        delta=fedavg_aggregate(batch.updates,batch.sample_counts); model=apply_delta(model,delta)
        history.append({'round':r+1,'clients':len(batch.client_ids),'sample_mass':int(sum(batch.sample_counts))})
    return model,history


def trusted_root_update(global_model, x_ref: torch.Tensor, y_ref: torch.Tensor, *, epochs=1, batch_size=128, lr=1e-3, fedprox_mu=.01, device='cpu', class_weights=None):
    """FLTrust-style trusted root update from server-held clean reference data."""
    if len(x_ref)==0: raise ValueError("trusted reference must be nonempty")
    local,_=local_train(global_model,x_ref,y_ref,epochs=epochs,batch_size=batch_size,lr=lr,fedprox_mu=fedprox_mu,device=device,class_weights=class_weights)
    return parameter_delta(local,global_model)


def clone_trajectories(base_model, names):
    names=list(names)
    if len(set(names))!=len(names): raise ValueError("trajectory names must be unique")
    return {name:TrajectoryState(name,deepcopy(base_model).cpu()) for name in names}
