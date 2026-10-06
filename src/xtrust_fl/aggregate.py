from __future__ import annotations

import numpy as np
import torch


def adaptive_clip_threshold(updates: list[torch.Tensor], mad_k: float = 2.5) -> float:
    norms = np.asarray([torch.linalg.vector_norm(u.detach().cpu()).item() for u in updates], dtype=float)
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med)))
    return max(med + mad_k * 1.4826 * mad, 1e-12)


def clip_update(update: torch.Tensor, threshold: float) -> torch.Tensor:
    update = update.detach().cpu()
    norm = torch.linalg.vector_norm(update).item()
    if norm <= threshold:
        return update
    return update * (threshold / (norm + 1e-12))


def weighted_clipped_aggregate(
    updates: list[torch.Tensor],
    sample_counts: list[int],
    client_scores: np.ndarray,
    reject_threshold: float = 0.15,
    mad_k: float = 2.5,
) -> torch.Tensor:
    if len(updates) == 0:
        raise ValueError("updates must not be empty")
    if not (len(updates) == len(sample_counts) == len(client_scores)):
        raise ValueError("updates, sample_counts, and client_scores must align")

    threshold = adaptive_clip_threshold(updates, mad_k=mad_k)
    weights = []
    clipped = []
    for update, n, score in zip(updates, sample_counts, client_scores):
        weight = float(n) * float(score) if float(score) >= reject_threshold else 0.0
        weights.append(weight)
        clipped.append(clip_update(update, threshold))

    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    out = torch.zeros_like(clipped[0])
    for w, u in zip(weights, clipped):
        out = out + float(w) * u
    return out


def fedavg_aggregate(updates: list[torch.Tensor], sample_counts: list[int]) -> torch.Tensor:
    weights = np.asarray(sample_counts, dtype=float)
    weights = weights / weights.sum()
    out = torch.zeros_like(updates[0].detach().cpu())
    for w, u in zip(weights, updates):
        out = out + float(w) * u.detach().cpu()
    return out
