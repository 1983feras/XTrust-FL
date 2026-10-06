import numpy as np
import pytest

from xtrust_fl.heterogeneity import calibrate_explanation_residual, js_divergence_to_global


def test_identical_client_distributions_have_zero_js():
    p = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
    d = js_divergence_to_global(p)
    assert np.allclose(d, 0.0)


def test_multidimensional_calibration_explains_benign_drift():
    h = np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.6, 0.3], [0.8, 0.4]])
    raw = 0.1 + 0.5 * h[:, 0] + 0.2 * h[:, 1]
    residual, expected = calibrate_explanation_residual(raw, h)
    assert residual.shape == raw.shape
    assert expected.shape == raw.shape
    assert np.max(residual) < 0.05


def test_empty_trusted_mask_fails_closed():
    raw = np.array([0.1, 0.2, 0.3])
    h = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="selects no clients"):
        calibrate_explanation_residual(raw, h, trusted_mask=np.zeros(3, dtype=bool))


def test_insufficient_trusted_clients_use_trusted_median_only():
    raw = np.array([0.1, 0.2, 2.0, 3.0])
    h = np.arange(4, dtype=float)
    mask = np.array([True, True, False, False])
    residual, expected = calibrate_explanation_residual(raw, h, mask, min_fit_clients=3)
    assert np.allclose(expected, 0.15)
    assert np.all(residual >= 0.0)
