from __future__ import annotations

import os
import sys
from copy import deepcopy

import numpy as np
import torch
from sklearn.datasets import make_classification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from xtrust_fl.aggregate import fedavg_aggregate, weighted_clipped_aggregate
from xtrust_fl.attacks import sign_flip, stealth_project
from xtrust_fl.explain import attribution_fingerprint, robust_explanation_anomaly
from xtrust_fl.heterogeneity import calibrate_explanation_residual, js_divergence_to_global
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import client_label_distributions, dirichlet_partition
from xtrust_fl.scoring import compute_client_scores
from xtrust_fl.training import apply_delta, local_train, parameter_delta
from xtrust_fl.update_space import robust_update_anomaly


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    x_np, y_np = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        class_sep=1.0,
        random_state=42,
    )
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    train_x, ref_x = x[:1800], x[1800:]
    train_y, ref_y = y[:1800], y[1800:]

    partitions = dirichlet_partition(
        train_y.numpy(), num_clients=10, alpha=0.5, seed=42, min_size=20
    )
    distributions = client_label_distributions(train_y.numpy(), partitions)
    heterogeneity = js_divergence_to_global(distributions)

    global_model = XTrustMLP(input_dim=20, num_classes=2, dropout=0.1)
    updates = []
    sample_counts = []
    candidate_models = []

    for cid, idx in enumerate(partitions):
        local_model, _ = local_train(
            global_model,
            train_x[idx],
            train_y[idx],
            epochs=1,
            batch_size=64,
            lr=1e-3,
            fedprox_mu=0.0,
        )
        delta = parameter_delta(local_model, global_model)
        updates.append(delta)
        sample_counts.append(len(idx))

    norms = np.asarray([torch.linalg.vector_norm(u).item() for u in updates])
    benign_med = float(np.median(norms))
    benign_mad = float(np.median(np.abs(norms - benign_med)))

    malicious_id = 0
    poisoned = sign_flip(updates[malicious_id], scale=3.0)
    updates[malicious_id] = stealth_project(poisoned, benign_med, benign_mad, k=1.0)

    for delta in updates:
        candidate_models.append(apply_delta(global_model, delta))

    fingerprints = [
        attribution_fingerprint(model, ref_x[:64], ref_y[:64], steps=8)
        for model in candidate_models
    ]

    a_update = robust_update_anomaly(updates)
    a_exp_raw = robust_explanation_anomaly(fingerprints, top_k=5)

    coarse_trusted_mask = a_update <= np.median(a_update)
    a_exp_residual, _ = calibrate_explanation_residual(
        a_exp_raw,
        heterogeneity,
        trusted_mask=coarse_trusted_mask,
    )

    scores = compute_client_scores(
        a_update,
        a_exp_residual,
        lambda_update=0.5,
        lambda_explanation=0.5,
    )

    fedavg_delta = fedavg_aggregate(updates, sample_counts)
    xtrust_delta = weighted_clipped_aggregate(
        updates,
        sample_counts,
        scores,
        reject_threshold=0.15,
        mad_k=2.5,
    )

    assert len(partitions) == 10
    assert a_update.shape == (10,)
    assert a_exp_residual.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert torch.isfinite(fedavg_delta).all()
    assert torch.isfinite(xtrust_delta).all()

    print("XTrust-FL synthetic smoke test PASSED")
    print("NOTE: These are synthetic pipeline diagnostics, not scientific results.")
    print(f"malicious_client={malicious_id}")
    print(f"update_anomaly={a_update.round(4).tolist()}")
    print(f"explanation_residual={a_exp_residual.round(4).tolist()}")
    print(f"client_scores={scores.round(4).tolist()}")


if __name__ == "__main__":
    main()
