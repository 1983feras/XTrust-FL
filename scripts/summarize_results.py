"""Convert XTrust-FL JSON summaries into a compact Markdown results table."""
import argparse
import json
from pathlib import Path

ORDER = ["accuracy", "macro_f1", "precision", "recall", "auroc", "auprc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+", help="method=path/to/summary.json")
    p.add_argument("--out", default="results/table_main.md")
    a = p.parse_args()

    methods = []
    for item in a.summaries:
        if "=" not in item:
            raise ValueError("Use method=summary.json")
        method, path = item.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        methods.append((method, data))

    lines = [
        "# Main Results",
        "",
        "All entries are mean ± sample standard deviation across recorded seeds.",
        "",
        "| Method | " + " | ".join(ORDER) + " |",
        "|---|" + "---:|" * len(ORDER),
    ]
    for method, data in methods:
        cells = []
        for metric in ORDER:
            x = data["metrics"][metric]
            cells.append(f"{x['mean']:.4f} ± {x['std']:.4f}")
        lines.append("| " + method + " | " + " | ".join(cells) + " |")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
