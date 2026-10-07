import numpy as np
import torch

from xtrust_fl.update_space import (
    fit_update_anomaly_reference,
    robust_update_anomaly,
    score_update_anomaly_fixed,
)


def test_zero_mad_update_features_remain_finite_and_bounded():
    updates = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.1]),
        torch.tensor([1.0, 0.9, 1.0]),
    ]
    scores = robust_update_anomaly(updates)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert float(np.max(scores)) < 1e6


def test_fixed_update_reference_is_not_changed_by_other_evaluation_clients():
    calibration = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.9, 0.1, 0.0]),
        torch.tensor([1.1, -0.1, 0.0]),
        torch.tensor([1.0, 0.05, 0.0]),
    ]
    reference = fit_update_anomaly_reference(calibration)
    target = torch.tensor([0.8, 0.2, 0.0])
    unrelated = torch.tensor([-10.0, 20.0, 30.0])
    score_alone = score_update_anomaly_fixed([target], reference)[0]
    score_with_other = score_update_anomaly_fixed([target, unrelated], reference)[0]
    assert np.isfinite(score_alone)
    assert np.isclose(score_alone, score_with_other)


def test_fixed_update_scores_are_finite():
    calibration = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([1.1, -0.1]),
    ]
    reference = fit_update_anomaly_reference(calibration)
    scores = score_update_anomaly_fixed(
        [torch.tensor([1.0, 0.05]), torch.tensor([-1.0, 2.0])], reference
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
