"""Run baseline and three non-decoder variants on frozen development splits."""

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
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CUSTOM_ENCODER = ROOT / "nondecoder_models.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
TRAIN_DATA = DATA / "eth_data_train.npy"
TRAIN_NUM = DATA / "eth_num_train.npy"

EXPERIMENTS = {
    "baseline": (ROOT / "EXP_135_BASELINE_CONTROL", "none"),
    "speed_scale": (ROOT / "EXP_135A_SPEED_SCALE_STATE", "n1_speed_scale_state"),
    "dual_temporal": (
        ROOT / "EXP_135B_TARGET_DUAL_TEMPORAL", "n2_target_dual_temporal"
    ),
    "density_motion": (
        ROOT / "EXP_135C_DENSITY_SELF_MOTION", "n3_density_self_motion"
    ),
}
SPLITS = {
    "general": ("stratified_random", 0.05),
    "challenge": ("motion_challenge", 0.10),
}


def run_dir(name: str, split: str, seed: int) -> Path:
    experiment_root, _ = EXPERIMENTS[name]
    return (
        experiment_root / "results" / split / "f16_temporal_social_cross" /
        f"seed_{seed}" / "val_scene_-1"
    )


def command_for(args, name: str, split: str, seed: int) -> list[str]:
    experiment_root, encoder_experiment = EXPERIMENTS[name]
    split_mode, validation_fraction = SPLITS[split]
    command = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(TRAIN_DATA), "--num-path", str(TRAIN_NUM),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", str(seed), "--val-scene-id", "-1",
        "--validation-fraction", str(validation_fraction),
        "--validation-seed", str(args.validation_seed),
        "--validation-split-mode", split_mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "warm_restarts",
        "--output-root", str(experiment_root / "results" / split),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if encoder_experiment != "none":
        command += [
            "--encoder-experiment", encoder_experiment,
            "--encoder-module", str(CUSTOM_ENCODER),
        ]
    if args.force:
        command.append("--force")
    return command


def collect(seeds: list[int]) -> dict:
    rows = []
    for name in EXPERIMENTS:
        _, encoder_experiment = EXPERIMENTS[name]
        for split in SPLITS:
            for seed in seeds:
                metrics_path = run_dir(name, split, seed) / "metrics.json"
                if not metrics_path.is_file():
                    continue
                raw = json.loads(metrics_path.read_text(encoding="utf-8"))
                evaluation = raw["evaluation_best_selected"]
                oracle = evaluation["oracle"]
                rows.append({
                    "experiment": name,
                    "encoder_experiment": encoder_experiment,
                    "split": split,
                    "seed": seed,
                    "validation_samples": evaluation["samples"],
                    "min_ade_20": oracle["min_ade_20"],
                    "same_path_fde": oracle["same_path_fde"],
                    "min_fde_20": oracle["min_fde_20"],
                    "top1_ade": evaluation["original"]["ade"],
                    "top1_fde": evaluation["original"]["fde"],
                    "parameters": raw["locked_backend"]["parameter_counts"]["total"],
                    "elapsed_seconds": raw["elapsed_seconds"],
                })
    summary = {
        "description": (
            "Locked F16/K20 non-decoder development screen; no official held-out "
            "ETH test archive is passed to the trainer."
        ),
        "rows": rows,
    }
    (ROOT / "nondecoder_dev_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (ROOT / "nondecoder_dev_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--validation-seed", type=int, default=20260825)
    parser.add_argument("--variants", nargs="+", choices=("all", *EXPERIMENTS), default=["all"])
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (PYTHON, TRAINER, CUSTOM_ENCODER, DECODER, TRAIN_DATA, TRAIN_NUM):
        if not path.is_file():
            raise FileNotFoundError(path)
    names = list(EXPERIMENTS) if "all" in args.variants else args.variants
    splits = list(SPLITS) if "all" in args.splits else args.splits

    for split in splits:
        for name in names:
            for seed in args.seeds:
                destination = run_dir(name, split, seed)
                metrics_path = destination / "metrics.json"
                if metrics_path.is_file() and not args.force:
                    print(f"[skip] {split} {name} seed={seed}", flush=True)
                    collect(args.seeds)
                    continue
                command = command_for(args, name, split, seed)
                if args.dry_run:
                    print(f"[dry-run] {split} {name} seed={seed} " + json.dumps(command))
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                print(f"[run] {split} {name} seed={seed}", flush=True)
                with (destination / "console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command, cwd=PROJECT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace", bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(f"[{split}/{name}] {line}", end="", flush=True)
                        log.write(line)
                        log.flush()
                    return_code = process.wait()
                if return_code:
                    raise RuntimeError(f"Failed; see {destination / 'console.log'}")
                collect(args.seeds)

    if args.dry_run:
        print("[done] dry-run passed", flush=True)
    else:
        summary = collect(args.seeds)
        print(f"[done] collected {len(summary['rows'])} runs", flush=True)


if __name__ == "__main__":
    main()
