from __future__ import annotations

import numpy as np
import torch


def _validate_updates(updates: list[torch.Tensor]) -> None:
    if not updates:
        raise ValueError("updates must not be empty")
    shape = updates[0].shape
    if any(u.shape != shape for u in updates):
        raise ValueError("all updates must have the same shape")
    if any(not torch.isfinite(u.detach()).all().item() for u in updates):
        raise ValueError("updates must contain only finite values")


def adaptive_clip_threshold(updates: list[torch.Tensor], mad_k: float = 2.5) -> float:
    _validate_updates(updates)
    norms = np.asarray([torch.linalg.vector_norm(u.detach().cpu()).item() for u in updates], dtype=float)
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med)))
    return max(med + mad_k * 1.4826 * mad, 1e-12)


def clip_update(update: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and positive")
    update = update.detach().cpu()
    norm = torch.linalg.vector_norm(update).item()
    if norm <= threshold:
        return update
    return update * (threshold / (norm + 1e-12))


def soft_trust_weights(client_scores: np.ndarray, q_min: float = 0.25) -> np.ndarray:
    """Frozen C4-v5 mapping from detector trust to bounded nonzero trust."""
    if not 0.0 < q_min <= 1.0:
        raise ValueError("q_min must be in (0, 1]")
    s = np.asarray(client_scores, dtype=float)
    if s.ndim != 1 or not np.all(np.isfinite(s)):
        raise ValueError("client_scores must be a finite 1-D array")
    return q_min + (1.0 - q_min) * np.clip(s, 0.0, 1.0)


def soft_trust_aggregate(
    updates: list[torch.Tensor], sample_counts: list[int], client_scores: np.ndarray,
    *, q_min: float = 0.25, clip: bool = False, mad_k: float = 2.5,
    return_diagnostics: bool = False,
):
    """C4-v5 soft nonzero trust aggregation; no malicious labels are consumed."""
    _validate_updates(updates)
    if not (len(updates) == len(sample_counts) == len(client_scores)):
        raise ValueError("updates, sample_counts, and client_scores must align")
    n = np.asarray(sample_counts, dtype=float)
    if np.any(n <= 0) or not np.all(np.isfinite(n)):
        raise ValueError("sample counts must be finite and positive")
    q = soft_trust_weights(client_scores, q_min=q_min)
    threshold = adaptive_clip_threshold(updates, mad_k=mad_k) if clip else None
    used, before, after, factors = [], [], [], []
    for u in updates:
        u0 = u.detach().cpu()
        b = float(torch.linalg.vector_norm(u0).item())
        u1 = clip_update(u0, threshold) if clip else u0
        a = float(torch.linalg.vector_norm(u1).item())
        used.append(u1); before.append(b); after.append(a)
        factors.append(1.0 if b <= 1e-12 else a / b)
    raw_w = n * q
    denom = float(raw_w.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("soft trust aggregation denominator must be finite and positive")
    w = raw_w / denom
    out = torch.zeros_like(used[0])
    for wi, ui in zip(w, used):
        out = out + float(wi) * ui
    if not return_diagnostics:
        return out
    return out, {"q_min":float(q_min),"q":q.tolist(),"sample_counts":n.astype(int).tolist(),"raw_weights":raw_w.tolist(),"normalized_weights":w.tolist(),"clip":bool(clip),"clip_threshold":None if threshold is None else float(threshold),"norm_before":before,"norm_after":after,"clip_factor":factors,"weight_mass":denom}


def weighted_clipped_aggregate(
    updates: list[torch.Tensor],
    sample_counts: list[int],
    client_scores: np.ndarray,
    reject_threshold: float = 0.15,
    mad_k: float = 2.5,
) -> torch.Tensor:
    """Historical C4-v4 trust-weighted clipped aggregation with hard rejection."""
    _validate_updates(updates)
    if not (len(updates) == len(sample_counts) == len(client_scores)):
        raise ValueError("updates, sample_counts, and client_scores must align")
    if any(int(n) <= 0 for n in sample_counts):
        raise ValueError("sample counts must be positive")
    scores = np.asarray(client_scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("client scores must be finite")
    if not 0.0 <= reject_threshold <= 1.0:
        raise ValueError("reject_threshold must be in [0, 1]")
    threshold = adaptive_clip_threshold(updates, mad_k=mad_k)
    weights, clipped = [], []
    for update, n, score in zip(updates, sample_counts, scores):
        weight = float(n) * float(score) if float(score) >= reject_threshold else 0.0
        weights.append(weight); clipped.append(clip_update(update, threshold))
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return torch.zeros_like(clipped[0])
    weights /= weights.sum()
    out = torch.zeros_like(clipped[0])
    for w, u in zip(weights, clipped): out = out + float(w) * u
    return out


def fedavg_aggregate(updates: list[torch.Tensor], sample_counts: list[int]) -> torch.Tensor:
    _validate_updates(updates)
    if len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must align")
    weights = np.asarray(sample_counts, dtype=float)
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample counts must be finite and positive")
    weights /= weights.sum()
    out = torch.zeros_like(updates[0].detach().cpu())
    for w, u in zip(weights, updates): out = out + float(w) * u.detach().cpu()
    return out
