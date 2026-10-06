import numpy as np
import pytest
import torch

from xtrust_fl.aggregate import fedavg_aggregate, weighted_clipped_aggregate


def test_all_rejected_clients_produce_zero_delta():
    updates = [torch.tensor([1.0, 2.0]), torch.tensor([-3.0, 4.0])]
    out = weighted_clipped_aggregate(updates, [10, 20], np.array([0.01, 0.02]), reject_threshold=0.15)
    assert torch.equal(out, torch.zeros(2))


def test_trusted_weighting_uses_sample_count_and_score():
    updates = [torch.tensor([1.0]), torch.tensor([3.0])]
    out = weighted_clipped_aggregate(updates, [1, 1], np.array([1.0, 0.5]), reject_threshold=0.0, mad_k=100.0)
    expected = torch.tensor([(1.0 * 1.0 + 0.5 * 3.0) / 1.5])
    assert torch.allclose(out, expected)


def test_fedavg_rejects_nonpositive_sample_counts():
    with pytest.raises(ValueError):
        fedavg_aggregate([torch.ones(2), torch.ones(2)], [1, 0])


def test_aggregate_rejects_nonfinite_scores():
    with pytest.raises(ValueError):
        weighted_clipped_aggregate([torch.ones(2)], [1], np.array([np.nan]))
