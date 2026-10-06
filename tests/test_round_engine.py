import numpy as np
import torch

from xtrust_fl.round_engine import RoundBatch, aggregate_same_round, select_clients


def test_client_selection_is_deterministic():
    a = select_clients(20, 0.5, seed=42)
    b = select_clients(20, 0.5, seed=42)
    assert a == b
    assert len(a) == 10


def test_all_defenses_consume_same_round_updates():
    updates = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.1, 0.9]),
        torch.tensor([0.9, 1.1]),
        torch.tensor([1.05, 0.95]),
        torch.tensor([1.0, 1.05]),
    ]
    batch = RoundBatch(
        client_ids=[0, 1, 2, 3, 4],
        updates=updates,
        sample_counts=[10, 10, 10, 10, 10],
        losses=[0.0] * 5,
    )
    before = [u.clone() for u in batch.updates]
    result = aggregate_same_round(
        batch,
        xtrust_scores=np.ones(5),
        server_update=torch.tensor([1.0, 1.0]),
        trim_ratio=0.2,
        krum_f=1,
    )
    assert {"fedavg", "coordinate_median", "trimmed_mean", "krum", "multi_krum", "fltrust", "xtrust_fl"}.issubset(result)
    for original, after in zip(before, batch.updates):
        assert torch.equal(original, after)


def test_scores_must_align_with_clients():
    batch = RoundBatch(
        client_ids=[0, 1],
        updates=[torch.ones(2), torch.ones(2)],
        sample_counts=[1, 1],
        losses=[0.0, 0.0],
    )
    try:
        aggregate_same_round(batch, xtrust_scores=np.ones(1))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned scores must fail")
