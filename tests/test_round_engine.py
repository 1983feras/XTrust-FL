import numpy as np
import torch
from torch import nn

from xtrust_fl.round_engine import RoundBatch, aggregate_same_round, build_round_batch, select_clients


def test_client_selection_is_deterministic():
    a = select_clients(20, 0.5, seed=42)
    b = select_clients(20, 0.5, seed=42)
    assert a == b
    assert len(a) == 10


def test_build_round_batch_preserves_ids_when_middle_partition_is_empty():
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = torch.tensor([0, 1, 0, 1])
    partitions = [[0, 1], [], [2, 3]]
    batch = build_round_batch(
        model,
        x,
        y,
        partitions,
        client_ids=[0, 1, 2],
        epochs=1,
        batch_size=2,
        lr=1e-3,
    )
    assert batch.client_ids == [0, 2]
    assert len(batch.updates) == len(batch.sample_counts) == len(batch.losses) == 2


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
