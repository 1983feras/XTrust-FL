from __future__ import annotations

import numpy as np


def compute_client_scores(
    update_anomaly: np.ndarray,
    explanation_residual: np.ndarray,
    previous_scores: np.ndarray | None = None,
    lambda_update: float = 0.5,
    lambda_explanation: float = 0.5,
    bias: float = 2.0,
    temporal_gamma: float = 0.8,
) -> np.ndarray:
    u = np.asarray(update_anomaly, dtype=float)
    e = np.asarray(explanation_residual, dtype=float)
    logits = bias - (lambda_update * u + lambda_explanation * e)
    current = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    if previous_scores is None:
        return current
    prev = np.asarray(previous_scores, dtype=float)
    return temporal_gamma * prev + (1.0 - temporal_gamma) * current
