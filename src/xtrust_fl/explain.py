from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> torch.Tensor:
    """Integrated Gradients for tabular models using trapezoidal integration."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model = deepcopy(model).cpu().eval()
    x = x.detach().cpu()
    targets = targets.detach().cpu().long().reshape(-1)
    if x.ndim != 2 or len(targets) != len(x):
        raise ValueError("x must be 2D and targets must align with samples")
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach().cpu()
        if baseline.shape != x.shape:
            baseline = baseline.expand_as(x)

    grads = []
    for alpha in torch.linspace(0.0, 1.0, steps + 1):
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits = model(xi)
        selected = logits.gather(1, targets[:, None]).sum()
        grad = torch.autograd.grad(selected, xi, retain_graph=False)[0]
        grads.append(grad.detach())
    stacked = torch.stack(grads, dim=0)
    avg_grad = (0.5 * stacked[0] + stacked[1:-1].sum(dim=0) + 0.5 * stacked[-1]) / float(steps)
    return (x - baseline) * avg_grad


def ig_completeness_error(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
) -> np.ndarray:
    m = deepcopy(model).cpu().eval()
    x_cpu = x.detach().cpu()
    t = targets.detach().cpu().long().reshape(-1)
    b = torch.zeros_like(x_cpu) if baseline is None else baseline.detach().cpu().expand_as(x_cpu)
    attr = integrated_gradients(m, x_cpu, t, baseline=b, steps=steps)
    with torch.no_grad():
        fx = m(x_cpu).gather(1, t[:, None]).squeeze(1)
        fb = m(b).gather(1, t[:, None]).squeeze(1)
    residual = torch.abs(attr.sum(dim=1) - (fx - fb))
    return residual.numpy()


def attribution_fingerprint(
    model: nn.Module,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    steps: int = 32,
) -> np.ndarray:
    attr = integrated_gradients(model, x_ref, y_ref, steps=steps)
    return attr.abs().mean(dim=0).numpy()


def explanation_prototype(fingerprints: list[np.ndarray]) -> np.ndarray:
    if not fingerprints:
        raise ValueError("at least one explanation fingerprint is required")
    stack = np.stack(fingerprints, axis=0)
    if not np.all(np.isfinite(stack)):
        raise ValueError("explanation fingerprints must be finite")
    return np.median(stack, axis=0)


def explanation_features(
    fingerprints: list[np.ndarray],
    top_k: int = 10,
) -> np.ndarray:
    proto = explanation_prototype(fingerprints)
    k = min(max(int(top_k), 1), len(proto))
    proto_top = set(np.argsort(proto)[-k:].tolist())
    rows = []
    denom = np.linalg.norm(proto, ord=1) + 1e-12
    for fp in fingerprints:
        fp = np.asarray(fp, dtype=float)
        if fp.shape != proto.shape:
            raise ValueError("all explanation fingerprints must have the same shape")
        rho = spearmanr(fp, proto).statistic
        if not np.isfinite(rho):
            rho = 0.0
        rank_drift = (1.0 - float(rho)) / 2.0
        client_top = set(np.argsort(fp)[-k:].tolist())
        union = len(proto_top | client_top)
        jaccard = len(proto_top & client_top) / union if union else 1.0
        topk_drift = 1.0 - jaccard
        l1_drift = np.linalg.norm(fp - proto, ord=1) / denom
        rows.append([rank_drift, topk_drift, l1_drift])
    return np.asarray(rows, dtype=float)


def _robust_scale(values: np.ndarray) -> float:
    """Stable robust scale for small/discrete client populations.

    MAD can be exactly zero for top-k Jaccard drift because many clients may
    share the same discrete value. Falling back to 1e-12 turns ordinary drift
    into ~1e10 anomaly scores. Use IQR next and, if that is also degenerate,
    standard deviation. A constant feature contributes zero anomaly.
    """
    x = np.asarray(values, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * float(mad)
    if scale > 1e-8:
        return scale
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(q75 - q25) / 1.349
    if scale > 1e-8:
        return scale
    scale = float(np.std(x, ddof=0))
    return scale if scale > 1e-8 else 0.0


def robust_explanation_anomaly(fingerprints: list[np.ndarray], top_k: int = 10) -> np.ndarray:
    features = explanation_features(fingerprints, top_k=top_k)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        med = np.median(features[:, j])
        scale = _robust_scale(features[:, j])
        if scale > 0.0:
            z[:, j] = np.abs(features[:, j] - med) / scale
        else:
            z[:, j] = 0.0
    return np.mean(z, axis=1)
