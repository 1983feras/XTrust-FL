from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def coordinate_median(updates: list[torch.Tensor]) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must not be empty")
    stacked = torch.stack([u.detach().cpu() for u in updates], dim=0)
    return torch.median(stacked, dim=0).values


def _robust_scale(values: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale <= eps:
        q25, q75 = np.percentile(values, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    if scale <= eps:
        scale = float(np.std(values))
    if scale <= eps:
        scale = 0.0
    return med, scale


def update_features_against_reference(
    updates: list[torch.Tensor], reference_update: torch.Tensor
) -> np.ndarray:
    ref = reference_update.detach().cpu()
    ref_norm = torch.linalg.vector_norm(ref).item()
    rows = []
    for u in updates:
        u = u.detach().cpu()
        norm = torch.linalg.vector_norm(u).item()
        dist = torch.linalg.vector_norm(u - ref).item()
        denom = (norm * ref_norm) + 1e-12
        cos = float(torch.dot(u, ref).item() / denom) if denom > 0 else 0.0
        rows.append([norm, 1.0 - cos, dist])
    return np.asarray(rows, dtype=float)


def update_features(updates: list[torch.Tensor]) -> np.ndarray:
    return update_features_against_reference(updates, coordinate_median(updates))


@dataclass(frozen=True)
class UpdateAnomalyReference:
    reference_update: torch.Tensor
    feature_median: np.ndarray
    feature_scale: np.ndarray


def fit_update_anomaly_reference(updates: list[torch.Tensor]) -> UpdateAnomalyReference:
    if not updates:
        raise ValueError("updates must not be empty")
    ref = coordinate_median(updates).detach().cpu().clone()
    features = update_features_against_reference(updates, ref)
    medians = []
    scales = []
    for j in range(features.shape[1]):
        med, scale = _robust_scale(features[:, j])
        medians.append(med)
        scales.append(scale)
    return UpdateAnomalyReference(
        reference_update=ref,
        feature_median=np.asarray(medians, dtype=float),
        feature_scale=np.asarray(scales, dtype=float),
    )


def score_update_anomaly_fixed(
    updates: list[torch.Tensor], reference: UpdateAnomalyReference
) -> np.ndarray:
    features = update_features_against_reference(updates, reference.reference_update)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        scale = float(reference.feature_scale[j])
        if scale > 0.0:
            z[:, j] = np.abs(features[:, j] - reference.feature_median[j]) / scale
        else:
            z[:, j] = 0.0
    return np.mean(z, axis=1)


def robust_update_anomaly(updates: list[torch.Tensor]) -> np.ndarray:
    features = update_features(updates)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        med, scale = _robust_scale(features[:, j])
        if scale > 0.0:
            z[:, j] = np.abs(features[:, j] - med) / scale
        else:
            z[:, j] = 0.0
    return np.mean(z, axis=1)
