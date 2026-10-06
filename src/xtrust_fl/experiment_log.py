"""Reproducibility utilities for XTrust-FL experiments.

Every scientific run should emit a manifest/result JSON that records enough
information to identify the data, code revision, partition, attack, defense,
and environment. This module deliberately does not manufacture metrics: callers
must provide measurements produced by the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
        })
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


def partition_digest(client_indices: list[np.ndarray]) -> str:
    """Stable digest for an exact federated partition."""
    h = hashlib.sha256()
    for cid, idx in enumerate(client_indices):
        arr = np.asarray(idx, dtype=np.int64)
        h.update(cid.to_bytes(4, "little", signed=False))
        h.update(len(arr).to_bytes(8, "little", signed=False))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def build_manifest(
    *,
    experiment_id: str,
    seed: int,
    dataset_name: str,
    dataset_sha256: str | None,
    split: dict[str, Any],
    partition: dict[str, Any],
    attack: dict[str, Any],
    defense: dict[str, Any],
    model: dict[str, Any],
    training: dict[str, Any],
    malicious_client_ids: list[int] | None = None,
    reference_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "dataset": {"name": dataset_name, "sha256": dataset_sha256},
        "split": split,
        "partition": partition,
        "attack": attack,
        "defense": defense,
        "model": model,
        "training": training,
        "malicious_client_ids": sorted(map(int, malicious_client_ids or [])),
        "reference_set": reference_set or {},
        "environment": environment_info(),
    }


def write_result(
    out_path: str | Path,
    *,
    manifest: dict[str, Any],
    final_metrics: dict[str, Any],
    per_round_metrics: list[dict[str, Any]] | None = None,
    timing: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write a self-describing result artifact.

    Metrics are accepted only from the caller; this function performs no
    interpolation, imputation, or synthetic filling of missing measurements.
    """
    payload = {
        "manifest": manifest,
        "final_metrics": final_metrics,
        "per_round_metrics": per_round_metrics or [],
        "timing": timing or {},
        "notes": notes or [],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
