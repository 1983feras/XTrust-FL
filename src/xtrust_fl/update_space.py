from __future__ import annotations

import numpy as np
import torch


def coordinate_median(updates: list[torch.Tensor]) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must not be empty")
    stacked = torch.stack([u.detach().cpu() for u in updates], dim=0)
    return torch.median(stacked, dim=0).values


def _mad(values: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(mad, eps)


def update_features(updates: list[torch.Tensor]) -> np.ndarray:
    ref = coordinate_median(updates)
    ref_norm = torch.linalg.vector_norm(ref).item()
    rows = []
    for u in updates:
        u = u.detach().cpu()
        norm = torch.linalg.vector_norm(u).item()
        dist = torch.linalg.vector_norm(u - ref).item()
        denom = (torch.linalg.vector_norm(u).item() * ref_norm) + 1e-12
        cos = float(torch.dot(u, ref).item() / denom) if denom > 0 else 0.0
        rows.append([norm, 1.0 - cos, dist])
    return np.asarray(rows, dtype=float)


def robust_update_anomaly(updates: list[torch.Tensor]) -> np.ndarray:
    features = update_features(updates)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        med, mad = _mad(features[:, j])
        z[:, j] = np.abs(features[:, j] - med) / (1.4826 * mad + 1e-12)
    return np.mean(z, axis=1)
