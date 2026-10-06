"""Paired federated round construction for reproducible XTrust-FL studies.

The key experimental rule is that competing aggregators receive the exact same
client updates from a round. This avoids retraining clients separately for each
defense and enables paired statistical comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .aggregate import fedavg_aggregate, weighted_clipped_aggregate
from .baselines import coordinate_median, trimmed_mean, krum, multi_krum, fltrust
from .training import local_train, parameter_delta, apply_delta


@dataclass
class RoundBatch:
    client_ids: list[int]
    updates: list[torch.Tensor]
    sample_counts: list[int]
    losses: list[float]


def select_clients(num_clients: int, participation_rate: float, seed: int) -> list[int]:
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1]")
    m = max(1, int(np.ceil(num_clients * participation_rate)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(num_clients, size=m, replace=False).astype(int).tolist())


def build_round_batch(
    global_model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    partitions: list[list[int]],
    client_ids: list[int],
    *,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    fedprox_mu: float = 0.0,
    device: str = "cpu",
    transform_update: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> RoundBatch:
    updates, counts, losses = [], [], []
    for cid in client_ids:
        idx = np.asarray(partitions[cid], dtype=np.int64)
        if idx.size == 0:
            continue
        local_model, loss = local_train(
            global_model,
            x_train[idx],
            y_train[idx],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            fedprox_mu=fedprox_mu,
            device=device,
        )
        delta = parameter_delta(local_model, global_model)
        if transform_update is not None:
            delta = transform_update(cid, delta)
        updates.append(delta.detach().cpu())
        counts.append(int(idx.size))
        losses.append(float(loss))
    if not updates:
        raise RuntimeError("No client updates were produced")
    return RoundBatch(client_ids=client_ids[:len(updates)], updates=updates, sample_counts=counts, losses=losses)


def aggregate_same_round(
    batch: RoundBatch,
    *,
    xtrust_scores: np.ndarray | None = None,
    server_update: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    krum_f: int = 0,
) -> dict[str, torch.Tensor]:
    """Run multiple defenses on one immutable set of client updates."""
    u, n = batch.updates, batch.sample_counts
    out: dict[str, torch.Tensor] = {
        "fedavg": fedavg_aggregate(u, n),
        "coordinate_median": coordinate_median(u),
        "trimmed_mean": trimmed_mean(u, trim_ratio=trim_ratio),
    }
    if len(u) >= 2 * krum_f + 3:
        out["krum"] = krum(u, f=krum_f)
        out["multi_krum"] = multi_krum(u, f=krum_f)
    if server_update is not None:
        out["fltrust"] = fltrust(u, server_update)[0]
    if xtrust_scores is not None:
        if len(xtrust_scores) != len(u):
            raise ValueError("xtrust_scores must align with participating clients")
        out["xtrust_fl"] = weighted_clipped_aggregate(u, n, xtrust_scores)
    return out


def candidate_models(global_model: nn.Module, aggregated: dict[str, torch.Tensor]) -> dict[str, nn.Module]:
    """Apply each aggregate to the same starting global model."""
    return {name: apply_delta(global_model, delta) for name, delta in aggregated.items()}
