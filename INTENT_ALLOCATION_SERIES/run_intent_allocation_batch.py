"""Run EXP-145 first-stage intent-allocation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = ROOT / "intent_allocation_models.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SOURCE_ROOT = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "EXP_143B_WIDE_ENDPOINT_CORRECTION" / "results"
EXPERIMENTS = {
    "balanced": (ROOT / "EXP_145A_BALANCED_INTENT_ALLOCATION", "i1_balanced_intent_allocation"),
    "alternate": (ROOT / "EXP_145B_ALTERNATE_INTENT_ALLOCATION", "i2_alternate_intent_allocation"),
    "balanced_progress": (ROOT / "EXP_145C_BALANCED_INTENT_DYNAMIC_PROGRESS", "i3_balanced_intent_dynamic_progress"),
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
LABELS = ("exp143b_reference", *EXPERIMENTS)


def run_dir(root: Path, split: str, seed: int) -> Path:
    return root / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def read_row(label: str, split: str, seed: int, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    diagnostics = evaluation.get("proposal_diagnostics", {})
    counts = raw["locked_backend"]["parameter_counts"]
    row = {
        "experiment": label, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"],
        "parameters": counts["total"], "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }
    for key in ("anchor_correction_mean", "progress_knot_mean", "progress_knot_p05", "progress_knot_p95"):
        row[key] = json.dumps(diagnostics.get(key), ensure_ascii=False)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    names = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde", "elapsed_seconds")
    output = []
    for split in SPLITS:
        for label in LABELS:
            group = [row for row in rows if row["split"] == split and row["experiment"] == label]
            if not group:
                continue
            metrics = {}
            for name in names:
                values = [float(row[name]) for row in group]
                metrics[name] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "values": values}
            output.append({
                "split": split, "experiment": label,
                "parameters": group[0]["parameters"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "metrics": metrics,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--validation-seed", type=int, default=20260825)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = [PYTHON, TRAINER, DECODER, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    for split in args.splits:
        split_mode, fraction = SPLITS[split]
        for label in args.experiments:
            experiment_root, decoder_name = EXPERIMENTS[label]
            for seed in args.seeds:
                source = run_dir(SOURCE_ROOT, split, seed) / "best_selected.pth"
                if not source.is_file():
                    raise FileNotFoundError(source)
                result_root = experiment_root / "results"
                metrics = run_dir(result_root, split, seed) / "metrics.json"
                if metrics.is_file() and not args.force:
                    print(f"[skip] {label} split={split} seed={seed}", flush=True); continue
                command = [
                    str(PYTHON), "-u", str(TRAINER),
                    "--project-root", str(PROJECT),
                    "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                    "--data-path", str(DATA / "eth_data_train.npy"),
                    "--num-path", str(DATA / "eth_num_train.npy"),
                    "--variant", "f16_temporal_social_cross",
                    "--decoder-experiment", decoder_name, "--decoder-module", str(DECODER),
                    "--seed", str(seed), "--val-scene-id", "-1",
                    "--validation-fraction", str(fraction), "--validation-seed", str(args.validation_seed),
                    "--validation-split-mode", split_mode,
                    "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size), "--device", args.device,
                    "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
                    "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
                    "--output-root", str(result_root / split),
                    "--oracle-coverage-weight", "0.10", "--oracle-coverage-temperature", "0.10",
                    "--oracle-coverage-fde-weight", "0.35", "--checkpoint-criterion", "oracle_combo",
                    "--ema-decay", "0.995", "--focal-heading-normalize", "--augment-reflection",
                    "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
                ]
                if args.force:
                    command.append("--force")
                env = os.environ.copy(); env["EXP145_SOURCE_CHECKPOINT"] = str(source)
                print(f"[run] {label} split={split} seed={seed}", flush=True)
                completed = subprocess.run(command, env=env)
                if completed.returncode:
                    log = run_dir(result_root, split, seed) / "console.log"
                    raise RuntimeError(f"Failed {label} split={split} seed={seed}; see {log}")

    rows = []
    for split in SPLITS:
        for seed in args.seeds:
            source = run_dir(SOURCE_ROOT, split, seed) / "metrics.json"
            if source.is_file():
                rows.append(read_row("exp143b_reference", split, seed, source))
            for label, (experiment_root, _) in EXPERIMENTS.items():
                result = run_dir(experiment_root / "results", split, seed) / "metrics.json"
                if result.is_file():
                    rows.append(read_row(label, split, seed, result))
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-145 batch-balanced first-stage intent query allocation",
        "protocol": {"source": "matched EXP-143B", "num_modes": 20, "cv_reference": False, "seeds": args.seeds},
        "rows": rows, "aggregate": aggregate(rows),
    }
    (ROOT / "multiseed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
