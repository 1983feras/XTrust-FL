import numpy as np
import torch
from torch import nn

from xtrust_fl.explain import (
    fit_explanation_anomaly_reference,
    integrated_gradients,
    ig_completeness_error,
    robust_explanation_anomaly,
    score_explanation_anomaly_fixed,
)


class LinearTwoClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]]))
            self.linear.bias.copy_(torch.tensor([0.3, -0.2]))

    def forward(self, x):
        return self.linear(x)


def test_integrated_gradients_matches_linear_closed_form():
    model = LinearTwoClass()
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.5]])
    targets = torch.tensor([0, 1])
    attr = integrated_gradients(model, x, targets, steps=8)
    weights = model.linear.weight.detach()[targets]
    expected = x * weights
    assert torch.allclose(attr, expected, atol=1e-6)


def test_completeness_error_is_near_zero_for_linear_model():
    model = LinearTwoClass()
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.5]])
    targets = torch.tensor([0, 1])
    err = ig_completeness_error(model, x, targets, steps=8)
    assert np.max(err) < 1e-6


def test_nonzero_baseline_closed_form():
    model = LinearTwoClass()
    x = torch.tensor([[1.0, 2.0, 3.0]])
    baseline = torch.tensor([[0.5, 0.5, 0.5]])
    target = torch.tensor([0])
    attr = integrated_gradients(model, x, target, baseline=baseline, steps=4)
    expected = (x - baseline) * model.linear.weight.detach()[0]
    assert torch.allclose(attr, expected, atol=1e-6)


def test_discrete_zero_mad_features_remain_finite_and_bounded():
    fps = [
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 0.9, 0.1, 0.0]),
        np.array([0.9, 1.0, 0.1, 0.0]),
    ]
    scores = robust_explanation_anomaly(fps, top_k=2)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert float(np.max(scores)) < 1e6


def test_fixed_reference_is_not_changed_by_unrelated_evaluation_clients():
    calibration = [
        np.array([1.0, 0.8, 0.2, 0.1]),
        np.array([0.9, 0.7, 0.3, 0.1]),
        np.array([1.1, 0.9, 0.2, 0.0]),
        np.array([1.0, 0.75, 0.25, 0.05]),
    ]
    reference = fit_explanation_anomaly_reference(calibration, top_k=2)
    target = np.array([0.8, 0.6, 0.4, 0.2])
    unrelated = np.array([10.0, 0.0, 0.0, 10.0])
    score_alone = score_explanation_anomaly_fixed([target], reference)[0]
    score_with_other = score_explanation_anomaly_fixed([target, unrelated], reference)[0]
    assert np.isfinite(score_alone)
    assert np.isclose(score_alone, score_with_other)


def test_fixed_score_output_shape_and_finiteness():
    calibration = [
        np.array([1.0, 0.5, 0.2]),
        np.array([0.9, 0.6, 0.2]),
        np.array([1.1, 0.4, 0.3]),
    ]
    reference = fit_explanation_anomaly_reference(calibration, top_k=2)
    evaluation = [np.array([1.0, 0.55, 0.25]), np.array([0.7, 0.8, 0.1])]
    scores = score_explanation_anomaly_fixed(evaluation, reference)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
