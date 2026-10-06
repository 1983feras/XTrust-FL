import torch

from xtrust_fl.baselines import coordinate_median, trimmed_mean, krum, multi_krum, fltrust


def test_coordinate_median_rejects_single_extreme_update():
    updates = [torch.tensor([1.0, 1.0]), torch.tensor([1.1, 0.9]), torch.tensor([100.0, -100.0])]
    out = coordinate_median(updates)
    assert torch.allclose(out, torch.tensor([1.1, 0.9]))


def test_trimmed_mean_removes_extremes():
    updates = [torch.tensor([x]) for x in [0.0, 1.0, 2.0, 3.0, 100.0]]
    out = trimmed_mean(updates, trim_ratio=0.2)
    assert torch.allclose(out, torch.tensor([2.0]))


def test_krum_prefers_benign_cluster():
    updates = [
        torch.tensor([1.00, 1.00]), torch.tensor([1.02, 0.99]),
        torch.tensor([0.98, 1.01]), torch.tensor([1.01, 1.02]),
        torch.tensor([-10.0, -10.0]),
    ]
    out = krum(updates, f=1)
    assert float(torch.linalg.vector_norm(out - torch.tensor([1.0, 1.0]))) < 0.1


def test_multi_krum_returns_finite_vector():
    updates = [
        torch.tensor([1.00, 1.00]), torch.tensor([1.02, 0.99]),
        torch.tensor([0.98, 1.01]), torch.tensor([1.01, 1.02]),
        torch.tensor([-10.0, -10.0]),
    ]
    out = multi_krum(updates, f=1, m=2)
    assert torch.isfinite(out).all()


def test_fltrust_zeroes_opposite_direction():
    server = torch.tensor([1.0, 0.0])
    clients = [torch.tensor([2.0, 0.0]), torch.tensor([-2.0, 0.0])]
    out, trust = fltrust(clients, server)
    assert trust[0] > 0.0
    assert trust[1] == 0.0
    assert torch.allclose(out, server, atol=1e-6)


def test_fltrust_normalizes_client_magnitude():
    server = torch.tensor([1.0, 0.0])
    clients = [torch.tensor([1000.0, 0.0]), torch.tensor([0.5, 0.0])]
    out, _ = fltrust(clients, server)
    assert torch.allclose(out, server, atol=1e-6)
