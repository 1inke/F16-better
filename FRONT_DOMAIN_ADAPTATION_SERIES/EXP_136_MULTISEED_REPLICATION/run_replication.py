"""Replicate the two surviving non-decoder variants with matched seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SERIES = HERE.parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
BATCH = SERIES / "run_nondecoder_dev_batch.py"
SOURCE_SUMMARY = SERIES / "nondecoder_dev_summary.json"
MODELS = ("baseline", "speed_scale", "dual_temporal")
SPLITS = ("general", "challenge")
SEEDS = (1, 2, 3)
METRICS = (
    "min_ade_20", "min_fde_20", "same_path_fde",
    "top1_ade", "top1_fde", "elapsed_seconds",
)


def run_batch(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON), "-u", str(BATCH),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size),
        "--seeds", "1", "2", "3",
        "--variants", *MODELS,
        "--splits", "all",
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    print("[driver] " + json.dumps(command), flush=True)
    process = subprocess.Popen(
        command, cwd=SERIES, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    with (HERE / "console.log").open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Replication driver failed; see {HERE / 'console.log'}")


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def collect() -> dict:
    if not SOURCE_SUMMARY.is_file():
        raise FileNotFoundError(SOURCE_SUMMARY)
    rows = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))["rows"]
    rows = [
        row for row in rows
        if row["experiment"] in MODELS
        and row["split"] in SPLITS
        and int(row["seed"]) in SEEDS
    ]
    expected = len(MODELS) * len(SPLITS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} completed rows, found {len(rows)}")

    aggregate = []
    for split in SPLITS:
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["split"] == split and row["experiment"] == model),
                key=lambda row: int(row["seed"]),
            )
            item = {
                "split": split,
                "experiment": model,
                "parameters": int(selected[0]["parameters"]),
                "metrics": {
                    metric: mean_std([float(row[metric]) for row in selected])
                    for metric in METRICS
                },
            }
            aggregate.append(item)

    paired = []
    for split in SPLITS:
        baseline = {
            int(row["seed"]): row for row in rows
            if row["split"] == split and row["experiment"] == "baseline"
        }
        for model in MODELS[1:]:
            changes = {metric: [] for metric in METRICS[:-1]}
            for row in rows:
                if row["split"] != split or row["experiment"] != model:
                    continue
                reference = baseline[int(row["seed"])]
                for metric in changes:
                    changes[metric].append(
                        100.0 * (float(row[metric]) / float(reference[metric]) - 1.0)
                    )
            paired.append({
                "split": split,
                "experiment": model,
                "paired_percent_change_vs_baseline": {
                    metric: mean_std(values) for metric, values in changes.items()
                },
            })

    result = {
        "description": "Three-seed matched replication of surviving non-decoder variants",
        "seeds": list(SEEDS),
        "completed_runs": len(rows),
        "aggregate": aggregate,
        "paired_comparisons": paired,
    }
    (HERE / "multiseed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_rows = []
    for item in aggregate:
        flat = {
            "split": item["split"], "experiment": item["experiment"],
            "parameters": item["parameters"],
        }
        for metric, stats in item["metrics"].items():
            flat[metric + "_mean"] = stats["mean"]
            flat[metric + "_std"] = stats["std"]
        flat_rows.append(flat)
    with (HERE / "multiseed_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (PYTHON, BATCH):
        if not path.is_file():
            raise FileNotFoundError(path)
    HERE.mkdir(parents=True, exist_ok=True)
    run_batch(args)
    if args.dry_run:
        print("[done] dry-run passed; no summary was changed", flush=True)
        return
    result = collect()
    print(
        f"[done] {result['completed_runs']} total seed/split/model rows; "
        f"summary={HERE / 'multiseed_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

