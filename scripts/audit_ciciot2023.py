"""Audit real CICIoT2023 CSV files before any research experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xtrust_fl.ciciot2023 import load_ciciot2023, split_and_scale, audit_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Directory containing official CICIoT2023 CSV files")
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/ciciot2023_audit.json")
    args = p.parse_args()
    X, y, labels = load_ciciot2023(args.data, args.max_rows_per_file, args.seed)
    report = audit_dataset(X, y, labels, args.out)
    train, val, test, _, indices = split_and_scale(X, y, args.seed)
    report["split_counts"] = {"train": len(indices["train"]), "val": len(indices["val"]), "test": len(indices["test"])}
    report["train_positive_rate"] = float(train[1].mean())
    report["val_positive_rate"] = float(val[1].mean())
    report["test_positive_rate"] = float(test[1].mean())
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
