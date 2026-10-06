"""C2: benign severe Non-IID experiment for XTrust-FL.

Purpose: test whether heterogeneity calibration reduces false positives caused by
legitimate client drift. There are NO malicious clients in C2. Thresholds are
estimated only from a designated benign calibration subset and evaluated on
held-out benign clients.
"""
from __future__ import annotations
import argparse, hashlib, json, random, urllib.request
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtrust_fl.model import XTrustMLP
from xtrust_fl.partition import dirichlet_partition, client_label_distributions
from xtrust_fl.explain import attribution_fingerprint, robust_explanation_anomaly
from xtrust_fl.heterogeneity import js_divergence_to_global, calibrate_explanation_residual

URL = "https://raw.githubusercontent.com/arpangl/IoT2026/main/lab2/data/ciciot2023_clean.parquet"
SHA256 = "17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1"
TARGET_DERIVED = {"label", "family", "is_attack"}


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_data(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256: raise RuntimeError(f"SHA-256 mismatch: {digest}")


def train_model(model, X, y, device, epochs=1, batch_size=512, lr=1e-3):
    m = deepcopy(model).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lossfn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for xb, yb in loader:
            xb, yb=xb.to(device), yb.to(device); opt.zero_grad(); loss=lossfn(m(xb), yb); loss.backward(); opt.step()
    return m.cpu()


def q95(x):
    return float(np.quantile(np.asarray(x, dtype=float), 0.95))


def robust_dirichlet_partition(y, num_clients, alpha, seed):
    """Construct a severe Non-IID partition without silently changing alpha.

    For very small alpha and imbalanced binary data, a strict minimum client size
    can make rejection sampling fail even on a large dataset. We therefore try a
    short descending minimum-size schedule while keeping the requested alpha,
    client count, labels, and seed family unchanged. The actually accepted
    minimum size is reported in the result artifact.
    """
    attempts = [50, 20, 10, 5, 1]
    last_error = None
    for min_size in attempts:
        try:
            parts = dirichlet_partition(
                y, num_clients, alpha, seed,
                min_size=min_size,
                max_retries=5000,
            )
            return parts, min_size
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        "Could not construct the requested severe Non-IID partition even with min_size=1"
    ) from last_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--client-cap", type=int, default=12000)
    ap.add_argument("--ref-size", type=int, default=128)
    ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--out", default="results/c2_seed42.json")
    a = ap.parse_args(); seed_all(a.seed)
    if a.clients < 10: raise ValueError("C2 requires at least 10 clients")

    path = ROOT / "data" / "external" / "ciciot2023_clean.parquet"; ensure_data(path)
    df = pd.read_parquet(path); lookup = {str(c).strip().lower(): c for c in df.columns}
    target = lookup.get("is_attack")
    if target is None: raise ValueError("is_attack target missing")
    y = pd.to_numeric(df[target], errors="raise").astype("int64").to_numpy()
    dropped = [c for c in df.columns if str(c).strip().lower() in TARGET_DERIVED]
    Xdf = df.drop(columns=dropped).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=.2, random_state=a.seed, stratify=y)
    tr, va = train_test_split(tr, test_size=.125, random_state=a.seed, stratify=y[tr])
    imputer = SimpleImputer(strategy="median"); Xtr_i = imputer.fit_transform(Xdf.iloc[tr]); Xva_i = imputer.transform(Xdf.iloc[va])
    scaler = StandardScaler(); Xtr = scaler.fit_transform(Xtr_i).astype("float32"); Xva = scaler.transform(Xva_i).astype("float32")
    ytr = y[tr]; yva = y[va]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = XTrustMLP(Xtr.shape[1], 2)
    base = train_model(base, Xtr, ytr, device, epochs=a.pretrain_epochs)

    parts, accepted_min_size = robust_dirichlet_partition(ytr, a.clients, a.alpha, a.seed)
    if any(len(p) == 0 for p in parts):
        raise RuntimeError("Partition contains an empty client")
    dists = client_label_distributions(ytr, parts, num_classes=2)
    js = js_divergence_to_global(dists)
    sizes = np.asarray([len(p) for p in parts], dtype=float)
    quantity_skew = np.abs(np.log((sizes + 1.0) / (np.median(sizes) + 1.0)))
    H = np.column_stack([js, quantity_skew])

    rng = np.random.default_rng(a.seed + 991)
    ref_idx = rng.choice(len(Xva), size=min(a.ref_size, len(Xva)), replace=False)
    xref = torch.from_numpy(Xva[ref_idx]); yref = torch.from_numpy(yva[ref_idx])

    fingerprints = []
    used_sizes = []
    for cid, p in enumerate(parts):
        p = np.asarray(p, dtype=int)
        if len(p) > a.client_cap:
            rr = np.random.default_rng(a.seed * 1000 + cid)
            p = rr.choice(p, size=a.client_cap, replace=False)
        local = train_model(base, Xtr[p], ytr[p], device, epochs=a.local_epochs)
        fp = attribution_fingerprint(local, xref, yref, steps=a.ig_steps)
        fingerprints.append(fp); used_sizes.append(int(len(p)))

    raw = robust_explanation_anomaly(fingerprints, top_k=min(10, Xtr.shape[1]))
    perm = np.random.default_rng(a.seed + 123).permutation(a.clients)
    n_cal = max(5, a.clients // 2); cal_ids = np.sort(perm[:n_cal]); eval_ids = np.sort(perm[n_cal:])
    mask = np.zeros(a.clients, dtype=bool); mask[cal_ids] = True
    residual, expected = calibrate_explanation_residual(raw, H, trusted_mask=mask, min_fit_clients=5)

    raw_thr = q95(raw[mask]); cal_thr = q95(residual[mask])
    raw_flags = raw[eval_ids] > raw_thr; cal_flags = residual[eval_ids] > cal_thr
    raw_fpr = float(raw_flags.mean()); cal_fpr = float(cal_flags.mean())

    result = {
        "experiment": "C2_benign_severe_nonIID",
        "hypothesis": "heterogeneity calibration reduces benign false positives under severe Non-IID",
        "provenance": {"url": URL, "sha256": SHA256},
        "integrity": {"no_malicious_clients": True, "target_metadata_dropped": list(map(str, dropped)), "imputer_fit": "train_only", "scaler_fit": "train_only", "threshold_source": "benign_calibration_clients_only"},
        "config": vars(a), "device": str(device), "n_features": int(Xtr.shape[1]),
        "partition": {"requested_alpha": float(a.alpha), "accepted_min_client_size": int(accepted_min_size), "actual_min_client_size": int(sizes.min()), "actual_max_client_size": int(sizes.max())},
        "calibration_client_ids": cal_ids.tolist(), "evaluation_client_ids": eval_ids.tolist(),
        "client_sizes_original": sizes.astype(int).tolist(), "client_sizes_used": used_sizes,
        "heterogeneity": {"js_divergence": js.tolist(), "quantity_skew": quantity_skew.tolist(), "js_mean": float(js.mean()), "js_max": float(js.max())},
        "raw_explanation_anomaly": raw.tolist(), "expected_explanation_anomaly": expected.tolist(), "calibrated_residual": residual.tolist(),
        "thresholds": {"raw_q95": raw_thr, "calibrated_q95": cal_thr},
        "evaluation": {"raw_false_positives": int(raw_flags.sum()), "calibrated_false_positives": int(cal_flags.sum()), "n_eval_benign_clients": int(len(eval_ids)), "raw_fpr_benign": raw_fpr, "calibrated_fpr_benign": cal_fpr, "absolute_fpr_change": float(cal_fpr - raw_fpr)},
    }
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
