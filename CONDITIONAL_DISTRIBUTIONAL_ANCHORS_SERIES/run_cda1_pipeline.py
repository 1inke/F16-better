"""Matched-control seed-1 pipeline for Conditional Distributional Anchors."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = W / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = P / "datasets" / "eth_ucy" / "processed_data_diverse"
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
SIDECAR = CSRS / "teacher_cache" / "sample_index_encoded.npy"
PREPARE = ROOT / "prepare_fixed_scale.py"
DIAGNOSTIC = ROOT / "run_cda1_diagnostic.py"
RESULTS = ROOT / "results"
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
VARIANTS = ("cda0_matched_continuation", "cda1_fixed_scale_anchor_mixture")
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def source_run(split, seed):
    return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def scale_path(split, seed):
    return ROOT / "scale" / f"{split}_seed_{seed}.json"


def run_dir(split, variant, seed):
    return RESULTS / split / variant / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def logged(command, path, env=None, cwd=ROOT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout
        for line in process.stdout:
            print(line, end="", flush=True); handle.write(line); handle.flush()
        if process.wait():
            raise RuntimeError(f"failed; see {path}")


def train_command(args, split, variant):
    mode, fraction = SPLITS[split]
    return [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(P),
        "--config-file", str(P / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"), "--num-path", str(DATA / "eth_num_train.npy"),
        "--sample-scene-path", str(SIDECAR), "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", variant, "--decoder-module", str(ROOT / "cda_models.py"),
        "--seed", str(args.seed), "--val-scene-id", "-1", "--validation-fraction", str(fraction),
        "--validation-seed", str(args.validation_seed), "--validation-split-mode", mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--output-root", str(RESULTS / split / variant),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995", "--focal-heading-normalize",
    ]


def read_metrics(path, name, split, seed):
    raw = json.loads(path.read_text(encoding="utf-8")); evaluation = raw["evaluation_best_selected"]
    counts = raw["locked_backend"]["parameter_counts"]
    return {
        "experiment": name, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"], "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"], "top1_fde": evaluation["original"]["fde"],
        "parameters": counts["total"], "trainable_parameters": counts["trainable"], "elapsed_seconds": raw["elapsed_seconds"],
    }


def summarize(seed):
    rows, comparisons = [], []
    for split in SPLITS:
        source = read_metrics(source_run(split, seed) / "metrics.json", "source_b0", split, seed)
        rows.append(source)
        variant_rows = {}
        for variant in VARIANTS:
            path = run_dir(split, variant, seed) / "metrics.json"
            if path.is_file():
                variant_rows[variant] = read_metrics(path, variant, split, seed); rows.append(variant_rows[variant])
        if all(v in variant_rows for v in VARIANTS):
            control, cda = (variant_rows[v] for v in VARIANTS)
            comparisons.append({
                "split": split,
                "cda1_vs_matched_control_percent_change": {m: 100 * (cda[m] / control[m] - 1) for m in METRICS},
                "matched_control_vs_source_percent_change": {m: 100 * (control[m] / source[m] - 1) for m in METRICS},
                "cda1_vs_source_percent_change": {m: 100 * (cda[m] / source[m] - 1) for m in METRICS},
            })
    (ROOT / "seed1_summary.json").write_text(json.dumps({"description": "CDA1 fixed-scale anchor mixture with matched continuation control", "rows": rows, "paired_comparisons": comparisons}, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--val-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.00025); parser.add_argument("--validation-seed", type=int, default=20260828)
    parser.add_argument("--scale-only", action="store_true"); parser.add_argument("--force", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    required = (PYTHON, TRAINER, PREPARE, DIAGNOSTIC, ROOT / "cda_models.py", SIDECAR, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy")
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    prepare_command = [str(PYTHON), "-u", str(PREPARE), "--device", args.device, "--seed", str(args.seed), "--batch-size", str(args.diagnostic_batch_size)]
    if args.dry_run:
        print("[dry-run] scale " + json.dumps(prepare_command))
        for split in SPLITS:
            for variant in VARIANTS: print("[dry-run] train " + json.dumps(train_command(args, split, variant)))
        return
    if args.force or not all(scale_path(split, args.seed).is_file() for split in SPLITS):
        logged(prepare_command, ROOT / "scale.log")
    else:
        print("[skip] train-fold scale calibration already exists")
    if args.scale_only:
        print("[done] scale calibration only"); return
    for split in SPLITS:
        for path in (source_run(split, args.seed) / "best_selected.pth", source_run(split, args.seed) / "metrics.json", scale_path(split, args.seed)):
            if not path.is_file(): raise FileNotFoundError(path)
        for variant in VARIANTS:
            destination = run_dir(split, variant, args.seed)
            if (destination / "metrics.json").is_file() and not args.force:
                print(f"[skip] {split}/{variant}"); continue
            env = os.environ.copy(); env["CDA_SOURCE_CHECKPOINT"] = str(source_run(split, args.seed) / "best_selected.pth"); env["CDA_SCALE_JSON"] = str(scale_path(split, args.seed))
            command = train_command(args, split, variant) + (["--force"] if args.force else [])
            print(f"[run] {split}/{variant} seed={args.seed}")
            logged(command, destination / "console.log", env=env, cwd=P); summarize(args.seed)
    summarize(args.seed)
    logged([str(PYTHON), "-u", str(DIAGNOSTIC), "--device", args.device, "--batch-size", str(args.diagnostic_batch_size), "--seed", str(args.seed)], ROOT / "diagnostic.log")


if __name__ == "__main__":
    main()
