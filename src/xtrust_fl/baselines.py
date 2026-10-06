"""Robust aggregation baselines used by XTrust-FL experiments.

Implementations are intentionally explicit so their assumptions can be audited.
FLTrust follows the trust-bootstrapping structure: ReLU cosine trust against a
server update, norm normalization to the server update, then trust weighting.
The server/root update must be computed externally from trusted reference data.
"""
from __future__ import annotations

import numpy as np
import torch


def _stack(updates: list[torch.Tensor]) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must not be empty")
    return torch.stack([u.detach().cpu().float() for u in updates], dim=0)


def coordinate_median(updates: list[torch.Tensor]) -> torch.Tensor:
    return torch.median(_stack(updates), dim=0).values


def trimmed_mean(updates: list[torch.Tensor], trim_ratio: float = 0.2) -> torch.Tensor:
    x = _stack(updates)
    n = x.shape[0]
    k = int(np.floor(trim_ratio * n))
    if 2 * k >= n:
        raise ValueError("trim_ratio removes all observations")
    values, _ = torch.sort(x, dim=0)
    return values[k:n-k].mean(dim=0)


def _pairwise_squared_distances(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2).pow(2)


def krum_scores(updates: list[torch.Tensor], f: int) -> np.ndarray:
    x = _stack(updates)
    n = x.shape[0]
    if n < 2 * f + 3:
        raise ValueError("Krum requires n >= 2f + 3")
    d = _pairwise_squared_distances(x)
    neighbour_count = n - f - 2
    scores = []
    for i in range(n):
        row = torch.cat([d[i, :i], d[i, i+1:]])
        scores.append(float(torch.topk(row, k=neighbour_count, largest=False).values.sum()))
    return np.asarray(scores, dtype=float)


def krum(updates: list[torch.Tensor], f: int) -> torch.Tensor:
    scores = krum_scores(updates, f)
    return updates[int(np.argmin(scores))].detach().cpu().float()


def multi_krum(updates: list[torch.Tensor], f: int, m: int | None = None) -> torch.Tensor:
    n = len(updates)
    scores = krum_scores(updates, f)
    max_m = n - f - 2
    if m is None:
        m = max_m
    if not 1 <= m <= max_m:
        raise ValueError("m must satisfy 1 <= m <= n-f-2")
    chosen = np.argsort(scores)[:m]
    return torch.stack([updates[int(i)].detach().cpu().float() for i in chosen]).mean(dim=0)


def fltrust(
    client_updates: list[torch.Tensor],
    server_update: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, np.ndarray]:
    """Aggregate client updates using FLTrust-style trust bootstrapping.

    Returns (aggregated_update, trust_scores). Negative cosine similarity is
    clipped to zero. Each accepted client update is normalized to the L2 norm
    of the trusted server update before trust-weighted aggregation.
    """
    if not client_updates:
        raise ValueError("client_updates must not be empty")
    s = server_update.detach().cpu().float()
    s_norm = float(torch.linalg.vector_norm(s))
    if s_norm <= eps:
        raise ValueError("server_update norm must be positive")

    normalized = []
    trust = []
    for u0 in client_updates:
        u = u0.detach().cpu().float()
        u_norm = float(torch.linalg.vector_norm(u))
        if u_norm <= eps:
            cos = 0.0
            u_scaled = torch.zeros_like(u)
        else:
            cos = float(torch.dot(u, s) / (u_norm * s_norm + eps))
            u_scaled = u * (s_norm / (u_norm + eps))
        trust.append(max(cos, 0.0))
        normalized.append(u_scaled)

    trust_arr = np.asarray(trust, dtype=float)
    total = float(trust_arr.sum())
    if total <= eps:
        # Fail closed to the trusted server direction rather than silently
        # reverting to an untrusted client average.
        return s.clone(), trust_arr

    weights = trust_arr / total
    out = torch.zeros_like(s)
    for w, u in zip(weights, normalized):
        out += float(w) * u
    return out, trust_arr
