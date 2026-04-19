"""Aggregate training ``results.csv`` files across all runs into a comparison table.

Scans ``<project>/<run_name>/results.csv`` produced by ``DetectionTrainer`` and
extracts the best-epoch metrics (highest mAP50-95). Emits a Markdown table to
stdout and optionally to a file.

Usage
-----
Scan the default runs/detect directory::

    python scripts/summarize_results.py

Filter by name prefix (e.g. only the P0 core set)::

    python scripts/summarize_results.py --filter t0-

Save to Markdown file::

    python scripts/summarize_results.py --output docs/results_table.md

Merge with a pre-measured speed/params table (CSV with columns
``name,params_M,flops_G,fps``)::

    python scripts/summarize_results.py --speed-csv runs/detect/speed.csv

Output columns
--------------
  name | P | R | mAP50 | mAP50-95 | best_epoch
Plus optional: params_M | FLOPs_G | FPS (when speed-csv is merged).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_P = "metrics/precision(B)"
METRIC_R = "metrics/recall(B)"
METRIC_MAP50 = "metrics/mAP50(B)"
METRIC_MAP = "metrics/mAP50-95(B)"


def best_row(csv_path: Path) -> dict | None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def _key(r: dict) -> float:
        try:
            return float(r.get(METRIC_MAP, 0) or 0)
        except ValueError:
            return 0.0

    best = max(rows, key=_key)
    epoch = int(float(best.get("epoch", 0) or 0))
    return {
        "P": float(best.get(METRIC_P, 0) or 0),
        "R": float(best.get(METRIC_R, 0) or 0),
        "mAP50": float(best.get(METRIC_MAP50, 0) or 0),
        "mAP": float(best.get(METRIC_MAP, 0) or 0),
        "best_epoch": epoch,
    }


def load_speed_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        name = r.get("name")
        if not name:
            continue
        out[name] = {k: float(r[k]) for k in ("params_M", "flops_G", "fps") if k in r and r[k]}
    return out


def render_markdown(rows: list[dict], extra_cols: list[str]) -> str:
    headers = ["name", "P", "R", "mAP50", "mAP50-95", "best_epoch"] + extra_cols
    widths = {
        "name": max(24, max(len(r["name"]) for r in rows) if rows else 0),
        "P": 7, "R": 7, "mAP50": 7, "mAP50-95": 9, "best_epoch": 10,
        "params_M": 10, "flops_G": 10, "fps": 8,
    }
    parts = []
    parts.append("| " + " | ".join(f"{h:<{widths[h]}}" for h in headers) + " |")
    parts.append("|" + "|".join("-" * (widths[h] + 2) for h in headers) + "|")
    for r in rows:
        cells = [
            f"{r['name']:<{widths['name']}}",
            f"{r['P']:>{widths['P']}.3f}",
            f"{r['R']:>{widths['R']}.3f}",
            f"{r['mAP50']:>{widths['mAP50']}.3f}",
            f"{r['mAP']:>{widths['mAP50-95']}.3f}",
            f"{r['best_epoch']:>{widths['best_epoch']}d}",
        ]
        for k in extra_cols:
            v = r.get(k)
            cells.append(f"{v:>{widths[k]}.3f}" if v is not None else f"{'-':>{widths[k]}}")
        parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize training results across runs")
    p.add_argument("--root", default="runs/detect", help="root directory of runs")
    p.add_argument("--filter", default="", help="include only runs whose name contains this substring")
    p.add_argument("--sort-by", default="mAP", choices=["name", "mAP", "mAP50"],
                   help="sort key (default: mAP = mAP50-95, descending)")
    p.add_argument("--speed-csv", default=None, help="optional CSV with columns name,params_M,flops_G,fps")
    p.add_argument("--output", default=None, help="path to write Markdown table")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} not found")
        return 1
    speed = load_speed_csv(Path(args.speed_csv)) if args.speed_csv else {}
    # Recursively find all results.csv under root
    csv_paths = sorted(root.rglob("results.csv"))
    rows = []
    for csv_path in csv_paths:
        run_name = csv_path.parent.name
        if args.filter and args.filter not in run_name:
            continue
        b = best_row(csv_path)
        if b is None:
            continue
        row = {"name": run_name, **b}
        if run_name in speed:
            row.update(speed[run_name])
        rows.append(row)

    if args.sort_by == "name":
        rows.sort(key=lambda r: r["name"])
    elif args.sort_by == "mAP50":
        rows.sort(key=lambda r: r["mAP50"], reverse=True)
    else:
        rows.sort(key=lambda r: r["mAP"], reverse=True)

    extra_cols = []
    if speed:
        extra_cols = [k for k in ("params_M", "flops_G", "fps") if any(k in r for r in rows)]

    md = render_markdown(rows, extra_cols)
    print(md)
    print(f"\n({len(rows)} runs)")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md + f"\n\n({len(rows)} runs)\n")
        print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
