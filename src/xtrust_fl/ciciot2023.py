"""Leakage-safe CICIoT2023 CSV preprocessing.

The official CICIoT2023 distribution contains CSV feature files. This loader
streams one or more CSV files, creates binary labels (Benign=0, attack=1),
splits before scaling, and fits StandardScaler on training data only.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def discover_csvs(root: str | Path):
    files = sorted(Path(root).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return files


def _label_column(columns):
    lookup = {str(c).strip().lower(): c for c in columns}
    for key in ("label", "labels", "class"):
        if key in lookup:
            return lookup[key]
    raise ValueError("Could not identify label column")


def load_ciciot2023(root, max_rows_per_file=None, seed=42):
    frames = []
    rng = np.random.default_rng(seed)
    for path in discover_csvs(root):
        df = pd.read_csv(path)
        if max_rows_per_file and len(df) > max_rows_per_file:
            idx = rng.choice(len(df), size=max_rows_per_file, replace=False)
            df = df.iloc[np.sort(idx)]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    label_col = _label_column(data.columns)
    labels_text = data[label_col].astype(str).str.strip()
    y = (~labels_text.str.lower().isin({"benign", "benigntraffic", "normal"})).astype(np.int64)
    X = data.drop(columns=[label_col]).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    constant = X.columns[X.nunique(dropna=False) <= 1]
    X = X.drop(columns=list(constant))
    return X.reset_index(drop=True), y.reset_index(drop=True), labels_text.loc[valid].reset_index(drop=True)


def split_and_scale(X, y, seed=42, test_size=0.20, val_size=0.10):
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(trainval, test_size=relative_val, random_state=seed, stratify=np.asarray(y)[trainval])
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    scaler = StandardScaler()
    X_np = np.asarray(X, dtype=np.float32)
    X_train = scaler.fit_transform(X_np[train]).astype(np.float32)
    X_val = scaler.transform(X_np[val]).astype(np.float32)
    X_test = scaler.transform(X_np[test]).astype(np.float32)
    y_np = np.asarray(y, dtype=np.int64)
    return (X_train, y_np[train]), (X_val, y_np[val]), (X_test, y_np[test]), scaler, {"train": train, "val": val, "test": test}


def audit_dataset(X, y, original_labels, out_path=None):
    report = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "binary_class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "original_label_counts": {str(k): int(v) for k, v in pd.Series(original_labels).value_counts().items()},
        "features": list(map(str, X.columns)),
        "missing_after_cleaning": int(X.isna().sum().sum()),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
