from __future__ import annotations

import torch


def sign_flip(delta: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return -float(scale) * delta


def model_scaling(delta: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    return float(scale) * delta


def stealth_project(
    malicious_delta: torch.Tensor,
    benign_norm_median: float,
    benign_norm_mad: float,
    k: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project a malicious update into a benign-looking norm band.

    This is only a first stealth-attack approximation. Later experiments should
    also constrain cosine similarity and/or distance from the robust reference.
    """
    target_max = max(benign_norm_median + k * 1.4826 * benign_norm_mad, eps)
    norm = torch.linalg.vector_norm(malicious_delta).item()
    if norm <= target_max:
        return malicious_delta
    return malicious_delta * (target_max / (norm + eps))
