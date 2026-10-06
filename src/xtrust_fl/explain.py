from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def integrated_gradients(model: nn.Module, x: torch.Tensor, targets: torch.Tensor, baseline: torch.Tensor | None = None, steps: int = 32) -> torch.Tensor:
    """Integrated Gradients for tabular models using trapezoidal integration."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model = deepcopy(model).cpu().eval()
    x = x.detach().cpu(); targets = targets.detach().cpu().long().reshape(-1)
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
        selected = model(xi).gather(1, targets[:, None]).sum()
        grads.append(torch.autograd.grad(selected, xi, retain_graph=False)[0].detach())
    stacked = torch.stack(grads, dim=0)
    avg_grad = (0.5 * stacked[0] + stacked[1:-1].sum(dim=0) + 0.5 * stacked[-1]) / float(steps)
    return (x - baseline) * avg_grad


def ig_completeness_error(model: nn.Module, x: torch.Tensor, targets: torch.Tensor, baseline: torch.Tensor | None = None, steps: int = 32) -> np.ndarray:
    m = deepcopy(model).cpu().eval(); x_cpu = x.detach().cpu(); t = targets.detach().cpu().long().reshape(-1)
    b = torch.zeros_like(x_cpu) if baseline is None else baseline.detach().cpu().expand_as(x_cpu)
    attr = integrated_gradients(m, x_cpu, t, baseline=b, steps=steps)
    with torch.no_grad():
        fx = m(x_cpu).gather(1, t[:, None]).squeeze(1); fb = m(b).gather(1, t[:, None]).squeeze(1)
    return torch.abs(attr.sum(dim=1) - (fx - fb)).numpy()


def attribution_fingerprint(model: nn.Module, x_ref: torch.Tensor, y_ref: torch.Tensor, steps: int = 32) -> np.ndarray:
    return integrated_gradients(model, x_ref, y_ref, steps=steps).abs().mean(dim=0).numpy()


def explanation_prototype(fingerprints: list[np.ndarray]) -> np.ndarray:
    if not fingerprints:
        raise ValueError("at least one explanation fingerprint is required")
    stack = np.stack(fingerprints, axis=0)
    if not np.all(np.isfinite(stack)):
        raise ValueError("explanation fingerprints must be finite")
    return np.median(stack, axis=0)


def explanation_features_against_prototype(fingerprints: list[np.ndarray], prototype: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Score fingerprints against a fixed externally supplied prototype.

    This is the leakage-free primitive used when calibration/trusted clients
    define the prototype and held-out clients must not influence it.
    """
    proto = np.asarray(prototype, dtype=float)
    if proto.ndim != 1 or not np.all(np.isfinite(proto)):
        raise ValueError("prototype must be a finite 1D vector")
    k = min(max(int(top_k), 1), len(proto)); proto_top = set(np.argsort(proto)[-k:].tolist())
    denom = np.linalg.norm(proto, ord=1) + 1e-12; rows = []
    for fp in fingerprints:
        fp = np.asarray(fp, dtype=float)
        if fp.shape != proto.shape:
            raise ValueError("all explanation fingerprints must match prototype shape")
        rho = spearmanr(fp, proto).statistic
        if not np.isfinite(rho): rho = 0.0
        rank_drift = (1.0 - float(rho)) / 2.0
        client_top = set(np.argsort(fp)[-k:].tolist()); union = len(proto_top | client_top)
        topk_drift = 1.0 - (len(proto_top & client_top) / union if union else 1.0)
        l1_drift = np.linalg.norm(fp - proto, ord=1) / denom
        rows.append([rank_drift, topk_drift, l1_drift])
    return np.asarray(rows, dtype=float)


def explanation_features(fingerprints: list[np.ndarray], top_k: int = 10) -> np.ndarray:
    return explanation_features_against_prototype(fingerprints, explanation_prototype(fingerprints), top_k=top_k)


def _robust_scale(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float); med = np.median(x); mad = np.median(np.abs(x - med)); scale = 1.4826 * float(mad)
    if scale > 1e-8: return scale
    q25, q75 = np.quantile(x, [0.25, 0.75]); scale = float(q75 - q25) / 1.349
    if scale > 1e-8: return scale
    scale = float(np.std(x, ddof=0)); return scale if scale > 1e-8 else 0.0


def fit_explanation_anomaly_reference(fingerprints: list[np.ndarray], top_k: int = 10) -> dict:
    """Fit prototype and robust feature location/scale on trusted clients only."""
    proto = explanation_prototype(fingerprints)
    features = explanation_features_against_prototype(fingerprints, proto, top_k=top_k)
    center = np.median(features, axis=0)
    scale = np.asarray([_robust_scale(features[:, j]) for j in range(features.shape[1])], dtype=float)
    return {"prototype": proto, "center": center, "scale": scale, "top_k": int(top_k)}


def score_explanation_anomaly_fixed(fingerprints: list[np.ndarray], reference: dict) -> np.ndarray:
    """Score arbitrary clients without allowing them to modify the reference."""
    features = explanation_features_against_prototype(fingerprints, reference["prototype"], top_k=reference["top_k"])
    center = np.asarray(reference["center"], dtype=float); scale = np.asarray(reference["scale"], dtype=float)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        if scale[j] > 0.0: z[:, j] = np.abs(features[:, j] - center[j]) / scale[j]
    return np.mean(z, axis=1)


def robust_explanation_anomaly(fingerprints: list[np.ndarray], top_k: int = 10) -> np.ndarray:
    reference = fit_explanation_anomaly_reference(fingerprints, top_k=top_k)
    return score_explanation_anomaly_fixed(fingerprints, reference)
