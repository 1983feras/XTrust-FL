"""Run leakage-safe CICIoT2023 centralized experiments over multiple seeds.

This script launches the existing real-data baseline, stores one JSON per seed,
and produces a paper-ready mean/std summary. It never fabricates metrics.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

METRICS = ["accuracy", "macro_f1", "precision", "recall", "auroc", "auprc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--seeds", default="42,77,100,999,2026")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--out-dir", default="results/centralized")
    a = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    rows = []

    for seed in seeds:
        out = out_dir / f"seed_{seed}.json"
        cmd = [
            sys.executable, str(root / "scripts" / "run_centralized.py"),
            "--data", a.data,
            "--seed", str(seed),
            "--epochs", str(a.epochs),
            "--batch-size", str(a.batch_size),
            "--lr", str(a.lr),
            "--out", str(out),
        ]
        if a.max_rows_per_file is not None:
            cmd += ["--max-rows-per-file", str(a.max_rows_per_file)]
        subprocess.run(cmd, check=True)
        result = json.loads(out.read_text(encoding="utf-8"))
        rows.append({"seed": seed, **result["test"]})

    summary = {"n_seeds": len(seeds), "seeds": seeds, "metrics": {}}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in rows], dtype=float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    summary["runs"] = rows
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCentralized CICIoT2023 research baseline")
    print(f"Seeds: {seeds}")
    for metric in METRICS:
        s = summary["metrics"][metric]
        print(f"{metric:>12}: {s['mean']:.6f} +/- {s['std']:.6f}")


if __name__ == "__main__":
    main()
