"""Train and test the selected model on all five ETH/UCY leave-one-out folds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
EXPERIMENT = ROOT / "EXP_133_CURRENT_MODEL_FULL_ETH_UCY"
OUTPUT_ROOT = EXPERIMENT / "results"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
DISPLAY = {
    "eth": "ETH", "hotel": "Hotel", "univ": "Univ",
    "zara01": "Zara1", "zara02": "Zara2",
}


def run_dir(fold: str, seed: int) -> Path:
    return (
        OUTPUT_ROOT / fold / "f16_temporal_social_cross" /
        f"seed_{seed}" / "val_scene_-1"
    )


def collect_summary(seed: int) -> dict:
    rows = []
    for fold in FOLDS:
        metrics_path = run_dir(fold, seed) / "metrics.json"
        if not metrics_path.is_file():
            continue
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = raw.get("evaluation_test_best_selected")
        if not test:
            continue
        split = raw["split"]
        oracle = test["oracle"]
        rows.append({
            "fold": fold,
            "display": DISPLAY[fold],
            "train_samples": split["train_samples"],
            "validation_samples": split["validation_samples"],
            "test_samples": test["samples"],
            "min_ade_20": oracle["min_ade_20"],
            "same_path_fde": oracle["same_path_fde"],
            "min_fde_20": oracle["min_fde_20"],
            "top1_ade": test["original"]["ade"],
            "top1_fde": test["original"]["fde"],
            "hit_best_ade": test["original"]["hit_best_ade"],
            "active_modes": test["original"]["active_modes"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "parameters": raw["locked_backend"]["parameter_counts"]["total"],
        })

    metric_names = (
        "min_ade_20", "same_path_fde", "min_fde_20",
        "top1_ade", "top1_fde", "hit_best_ade",
    )
    macro = {}
    micro = {}
    if rows:
        total_samples = sum(row["test_samples"] for row in rows)
        for metric in metric_names:
            macro[metric] = sum(row[metric] for row in rows) / len(rows)
            micro[metric] = sum(
                row[metric] * row["test_samples"] for row in rows
            ) / total_samples
        micro["test_samples"] = total_samples
    summary = {
        "experiment": "EXP_133 current selected model, full ETH/UCY leave-one-out",
        "completed_folds": len(rows),
        "expected_folds": len(FOLDS),
        "model": (
            "F16 temporal-social cross + kinematic K20 + coverage 0.20 + EMA "
            "+ focal-heading normalization + 50% reflection/speed augmentation"
        ),
        "protocol": (
            "Each fold trains on its provided four-dataset training archive, reserves "
            "the chronological tail 5% of every source scene for checkpoint selection, "
            "then evaluates the complete held-out test archive once."
        ),
        "macro_equal_fold": macro,
        "micro_sample_weighted": micro,
        "folds": rows,
    }
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "eth_ucy_full_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (EXPERIMENT / "eth_ucy_full_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def command_for(args: argparse.Namespace, fold: str) -> list[str]:
    command = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / f"{fold}_data_train.npy"),
        "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
        "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "kinematic_k20",
        "--decoder-module", str(DECODER),
        "--seed", str(args.seed), "--val-scene-id", "-1",
        "--validation-fraction", str(args.validation_fraction),
        "--validation-seed", str(args.validation_seed),
        "--validation-split-mode", "scene_tail",
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "warm_restarts", "--output-root", str(OUTPUT_ROOT / fold),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if args.force:
        command.append("--force")
    return command


def validate_files() -> None:
    missing = []
    for fold in FOLDS:
        for suffix in ("data_train", "num_train", "data_test", "num_test"):
            path = DATA / f"{fold}_{suffix}.npy"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing ETH/UCY files:\n" + "\n".join(missing))
    for path in (PYTHON, TRAINER, DECODER):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=20260824)
    parser.add_argument("--only", choices=("all", *FOLDS), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_files()
    folds = FOLDS if args.only == "all" else (args.only,)

    for fold in folds:
        metrics_path = run_dir(fold, args.seed) / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[skip] {DISPLAY[fold]} already completed", flush=True)
            collect_summary(args.seed)
            continue
        command = command_for(args, fold)
        if args.dry_run:
            print(f"[dry-run {DISPLAY[fold]}] " + json.dumps(command), flush=True)
            continue
        destination = run_dir(fold, args.seed)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"[run] {DISPLAY[fold]}", flush=True)
        with (destination / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{DISPLAY[fold]}] {line}", end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f"{DISPLAY[fold]} failed; see {destination / 'console.log'}")
        collect_summary(args.seed)

    if args.dry_run:
        print("[done] dry-run passed", flush=True)
    else:
        summary = collect_summary(args.seed)
        print(
            f"[done] {summary['completed_folds']}/{summary['expected_folds']} folds; "
            f"summary={EXPERIMENT / 'eth_ucy_full_summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
