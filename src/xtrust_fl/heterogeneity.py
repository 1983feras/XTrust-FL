from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler


def js_divergence_to_global(client_distributions: np.ndarray) -> np.ndarray:
    """Compute Jensen-Shannon divergence from each client label distribution to global."""
    p_all = np.asarray(client_distributions, dtype=float)
    if p_all.ndim != 2 or p_all.shape[0] == 0:
        raise ValueError("client_distributions must be a non-empty 2D array")
    if np.any(p_all < 0) or not np.all(np.isfinite(p_all)):
        raise ValueError("client distributions must be finite and non-negative")
    row_sums = p_all.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each client distribution must have positive mass")
    p_all = p_all / row_sums
    global_dist = p_all.mean(axis=0)
    global_dist /= global_dist.sum()
    return np.asarray([
        float(jensenshannon(p, global_dist, base=2.0) ** 2) for p in p_all
    ], dtype=float)


def _as_fingerprint(h: np.ndarray, n: int) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    if h.ndim == 1:
        h = h[:, None]
    if h.ndim != 2 or h.shape[0] != n:
        raise ValueError("heterogeneity fingerprint must have shape (n_clients, n_features)")
    if not np.all(np.isfinite(h)):
        raise ValueError("heterogeneity fingerprint contains non-finite values")
    return h


def calibrate_explanation_residual(
    raw_explanation_anomaly: np.ndarray,
    heterogeneity_score: np.ndarray,
    trusted_mask: np.ndarray | None = None,
    min_fit_clients: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive explanation residual after robust heterogeneity calibration.

    The calibration model is fit only on ``trusted_mask``. A multidimensional
    fingerprint is supported (e.g., label JS divergence, quantity skew, trusted
    reference loss). Robust scaling and Huber regression reduce leverage from
    benign extremes. If too few trusted clients are available, calibration
    fails closed instead of silently fitting on untrusted clients.
    """
    y = np.asarray(raw_explanation_anomaly, dtype=float).reshape(-1)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("raw explanation anomaly must be finite and non-empty")
    h = _as_fingerprint(heterogeneity_score, len(y))
    if trusted_mask is None:
        mask = np.ones(len(y), dtype=bool)
    else:
        mask = np.asarray(trusted_mask, dtype=bool).reshape(-1)
        if len(mask) != len(y):
            raise ValueError("trusted_mask must align with clients")
    if int(mask.sum()) == 0:
        raise ValueError("trusted_mask selects no clients")

    # With insufficient support, use only the trusted median as a conservative
    # intercept. Never borrow untrusted clients to make the fit possible.
    if int(mask.sum()) < int(min_fit_clients):
        expected = np.full_like(y, float(np.median(y[mask])))
    else:
        scaler = RobustScaler().fit(h[mask])
        h_fit = scaler.transform(h[mask])
        h_all = scaler.transform(h)
        informative = np.any(np.ptp(h_fit, axis=0) > 1e-12)
        if not informative:
            expected = np.full_like(y, float(np.median(y[mask])))
        else:
            model = HuberRegressor().fit(h_fit, y[mask])
            expected = model.predict(h_all)

    # Only excess explanation drift is suspicious; legitimate heterogeneity that
    # the calibration predicts is not penalized.
    residual = np.maximum(0.0, y - expected)
    return residual, expected
