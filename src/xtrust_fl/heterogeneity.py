from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import HuberRegressor


def js_divergence_to_global(client_distributions: np.ndarray) -> np.ndarray:
    """Compute Jensen-Shannon divergence from each client label distribution to global."""
    client_distributions = np.asarray(client_distributions, dtype=float)
    global_dist = client_distributions.mean(axis=0)
    global_dist = global_dist / (global_dist.sum() + 1e-12)
    out = []
    for p in client_distributions:
        p = p / (p.sum() + 1e-12)
        d = float(jensenshannon(p, global_dist, base=2.0))
        out.append(d * d)
    return np.asarray(out, dtype=float)


def calibrate_explanation_residual(
    raw_explanation_anomaly: np.ndarray,
    heterogeneity_score: np.ndarray,
    trusted_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected benign explanation anomaly from heterogeneity.

    For the MRE, calibration uses a robust Huber regressor. In final experiments,
    the fitting population must be limited to a trusted/warm-up or coarse-screened
    subset to reduce poisoning contamination.
    """
    y = np.asarray(raw_explanation_anomaly, dtype=float)
    h = np.asarray(heterogeneity_score, dtype=float).reshape(-1, 1)
    if trusted_mask is None:
        trusted_mask = np.ones(len(y), dtype=bool)
    else:
        trusted_mask = np.asarray(trusted_mask, dtype=bool)

    if trusted_mask.sum() < 3 or np.allclose(h[trusted_mask], h[trusted_mask][0]):
        expected = np.full_like(y, np.median(y[trusted_mask]))
    else:
        model = HuberRegressor().fit(h[trusted_mask], y[trusted_mask])
        expected = model.predict(h)

    residual = np.maximum(0.0, y - expected)
    return residual, expected
