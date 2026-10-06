import numpy as np
import torch
from torch import nn

from xtrust_fl.explain import integrated_gradients, ig_completeness_error


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
