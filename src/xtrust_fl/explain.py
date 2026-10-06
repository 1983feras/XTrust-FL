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
    steps: int = 16,
) -> torch.Tensor:
    """Minimal Integrated Gradients implementation for tabular models.

    Targets are fixed labels from a trusted reference set to keep candidate-model
    explanations comparable across clients.
    """
    model = deepcopy(model).cpu().eval()
    x = x.detach().cpu()
    targets = targets.detach().cpu().long()
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach().cpu()
        if baseline.shape != x.shape:
            baseline = baseline.expand_as(x)

    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps + 1)[1:]:
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits = model(xi)
        selected = logits.gather(1, targets.view(-1, 1)).sum()
        grads = torch.autograd.grad(selected, xi, retain_graph=False)[0]
        total_grad += grads.detach()
    avg_grad = total_grad / float(steps)
    return (x - baseline) * avg_grad


def attribution_fingerprint(
    model: nn.Module,
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    steps: int = 16,
) -> np.ndarray:
    attr = integrated_gradients(model, x_ref, y_ref, steps=steps)
    return attr.abs().mean(dim=0).numpy()


def explanation_prototype(fingerprints: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(fingerprints, axis=0), axis=0)


def explanation_features(
    fingerprints: list[np.ndarray],
    top_k: int = 10,
) -> np.ndarray:
    proto = explanation_prototype(fingerprints)
    k = min(top_k, len(proto))
    proto_top = set(np.argsort(proto)[-k:].tolist())
    rows = []
    denom = np.linalg.norm(proto, ord=1) + 1e-12
    for fp in fingerprints:
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


def robust_explanation_anomaly(fingerprints: list[np.ndarray], top_k: int = 10) -> np.ndarray:
    features = explanation_features(fingerprints, top_k=top_k)
    z = np.zeros_like(features)
    for j in range(features.shape[1]):
        med = np.median(features[:, j])
        mad = np.median(np.abs(features[:, j] - med))
        z[:, j] = np.abs(features[:, j] - med) / (1.4826 * max(float(mad), 1e-12))
    return np.mean(z, axis=1)
