from __future__ import annotations

import numpy as np


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_retries: int = 200,
) -> list[list[int]]:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        for cls in classes:
            idx = np.where(y == cls)[0].copy()
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            splits = np.split(idx, cuts)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

        if min(map(len, client_indices)) >= min_size:
            for cid in range(num_clients):
                rng.shuffle(client_indices[cid])
            return client_indices

    raise RuntimeError(
        f"Could not construct a Dirichlet partition with min_size={min_size}; "
        "increase dataset size, alpha, or reduce num_clients."
    )


def client_label_distributions(y: np.ndarray, partitions: list[list[int]]) -> np.ndarray:
    y = np.asarray(y)
    classes = np.unique(y)
    class_to_col = {cls: j for j, cls in enumerate(classes)}
    out = np.zeros((len(partitions), len(classes)), dtype=float)
    for i, idx in enumerate(partitions):
        for label in y[idx]:
            out[i, class_to_col[label]] += 1.0
        if out[i].sum() > 0:
            out[i] /= out[i].sum()
    return out
