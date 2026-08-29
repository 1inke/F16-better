"""PFRA-gated CFB2 seed-1 pipeline."""
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
IR1 = W / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
SIDECAR = CSRS / "teacher_cache" / "sample_index_encoded.npy"
PFRA = ROOT / "prepare_pfra_cfb2.py"
DIAGNOSTIC = ROOT / "run_cfb2_diagnostic.py"
RESULTS = ROOT / "results"
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def source(split, seed):
    return IR1 / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def baseline(split, seed):
    # Must match CFB2_SOURCE_CHECKPOINT exactly. CSRS contains a different B0 checkpoint.
    return IR1 / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def run_dir(split, seed):
    return RESULTS / split / "cfb2_confidence_gradient_isolated" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def logged(command, log_path, env=None, cwd=ROOT):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line); handle.flush()
        if process.wait():
            raise RuntimeError(f"failed; see {log_path}")


def train_command(args, split):
    mode, fraction = SPLITS[split]
    return [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(P),
        "--config-file", str(P / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"), "--num-path", str(DATA / "eth_num_train.npy"),
        "--sample-scene-path", str(SIDECAR), "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "cfb2_confidence_gradient_isolated", "--decoder-module", str(ROOT / "cfb2_models.py"),
        "--seed", str(args.seed), "--val-scene-id", "-1", "--validation-fraction", str(fraction),
        "--validation-seed", str(args.validation_seed), "--validation-split-mode", mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--output-root", str(RESULTS / split / "cfb2_confidence_gradient_isolated"),
        # Strict isolation: no GT oracle auxiliary is allowed to update the adapter.
        "--oracle-coverage-weight", "0.0", "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995", "--focal-heading-normalize",
    ]


def read_metrics(path, name, split, seed):
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    counts = raw["locked_backend"]["parameter_counts"]
    return {
        "experiment": name, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"], "top1_fde": evaluation["original"]["fde"],
        "parameters": counts["total"], "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def summarize(seed):
    rows, comparisons = [], []
    for split in SPLITS:
        b = read_metrics(baseline(split, seed) / "metrics.json", "baseline_reused_exact_match", split, seed)
        rows.append(b)
        path = run_dir(split, seed) / "metrics.json"
        if path.is_file():
            model = read_metrics(path, "cfb2_confidence_gradient_isolated", split, seed)
            rows.append(model)
            comparisons.append({"split": split, "percent_change": {m: 100 * (model[m] / b[m] - 1) for m in METRICS}})
    result = {"description": "CFB2 confidence-gated gradient-isolated future bank", "rows": rows, "paired_comparisons": comparisons}
    (ROOT / "seed1_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--val-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validation-seed", type=int, default=20260828)
    parser.add_argument("--pfra-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    required = (PYTHON, TRAINER, PFRA, DIAGNOSTIC, ROOT / "cfb2_models.py", SIDECAR, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    pfra_command = [str(PYTHON), "-u", str(PFRA), "--seed", str(args.seed)]
    if args.dry_run:
        print("[dry-run] PFRA " + json.dumps(pfra_command))
        for split in SPLITS:
            print("[dry-run] train " + json.dumps(train_command(args, split)))
        return
    logged(pfra_command, ROOT / "pfra.log")
    pfra = json.loads((ROOT / "pfra" / "combined_summary.json").read_text(encoding="utf-8"))
    if not pfra["gate_passed"]:
        print("[stop] PFRA failed. CFB2 training was intentionally not started.")
        return
    if args.pfra_only:
        print("[done] PFRA passed; training intentionally skipped")
        return
    for split in SPLITS:
        for path in (source(split, args.seed), baseline(split, args.seed) / "metrics.json", ROOT / "cache" / f"{split}_seed_{args.seed}.npz"):
            if not path.is_file():
                raise FileNotFoundError(path)
        destination = run_dir(split, args.seed)
        if (destination / "metrics.json").is_file() and not args.force:
            print(f"[skip] {split}/CFB2")
            continue
        env = os.environ.copy()
        env["CFB2_SOURCE_CHECKPOINT"] = str(source(split, args.seed))
        env["CFB2_CACHE"] = str(ROOT / "cache" / f"{split}_seed_{args.seed}.npz")
        command = train_command(args, split) + (["--force"] if args.force else [])
        print(f"[run] {split}/CFB2 seed={args.seed}")
        logged(command, destination / "console.log", env=env, cwd=P)
        summarize(args.seed)
    summarize(args.seed)
    logged([str(PYTHON), "-u", str(DIAGNOSTIC), "--device", args.device, "--batch-size", str(args.diagnostic_batch_size), "--seed", str(args.seed)], ROOT / "diagnostic.log")


if __name__ == "__main__":
    main()
