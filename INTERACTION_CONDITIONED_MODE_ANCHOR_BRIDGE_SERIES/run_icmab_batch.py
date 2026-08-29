"""Run matched B0/IC1 seed-1 experiments and the geometry diagnostic."""

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
BASELINE_DECODER = ROOT / "current_baseline_adapter.py"
IC1_DECODER = ROOT / "icmab_models.py"
DIAGNOSTIC = ROOT / "run_geometry_diagnostic.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SOURCE_ROOT = (
    WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" /
    "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
)
RESULTS = ROOT / "results"
EXPERIMENTS = {
    "baseline": ("current_mode_anchor_template", BASELINE_DECODER),
    "ic1_social_mode_anchor_bridge": ("ic1_social_mode_anchor_bridge", IC1_DECODER),
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def run_dir(name: str, split: str, seed: int) -> Path:
    return RESULTS / split / name / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def source_checkpoint(split: str, seed: int) -> Path:
    return (
        SOURCE_ROOT / split / "f16_temporal_social_cross" / f"seed_{seed}" /
        "val_scene_-1" / "best_selected.pth"
    )


def command(args, name: str, split: str, seed: int) -> list[str]:
    split_mode, fraction = SPLITS[split]
    decoder_experiment, decoder_module = EXPERIMENTS[name]
    output = [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"),
        "--num-path", str(DATA / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", decoder_experiment,
        "--decoder-module", str(decoder_module), "--seed", str(seed),
        "--val-scene-id", "-1", "--validation-fraction", str(fraction),
        "--validation-seed", str(args.validation_seed),
        "--validation-split-mode", split_mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
        "--device", args.device, "--max-agents", "9", "--graph-top-k", "4",
        "--graph-radius", "6.0", "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1", "--output-root", str(RESULTS / split / name),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if args.force:
        output.append("--force")
    return output


def read_row(path: Path, name: str, split: str, seed: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    return {
        "experiment": name, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"],
        "parameters": raw["locked_backend"]["parameter_counts"]["total"],
        "trainable_parameters": raw["locked_backend"]["parameter_counts"]["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def summarize(seeds: list[int], names: list[str], splits: list[str]) -> dict:
    rows = []
    for split in splits:
        for name in names:
            for seed in seeds:
                path = run_dir(name, split, seed) / "metrics.json"
                if path.is_file():
                    rows.append(read_row(path, name, split, seed))
    aggregate, paired = [], []
    for split in splits:
        baseline = {row["seed"]: row for row in rows if row["split"] == split and row["experiment"] == "baseline"}
        for name in names:
            group = [row for row in rows if row["split"] == split and row["experiment"] == name]
            if not group:
                continue
            aggregate.append({
                "split": split, "experiment": name,
                "parameters": group[0]["parameters"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "metrics": {metric: {
                    "mean": statistics.fmean(float(row[metric]) for row in group),
                    "std": statistics.stdev(float(row[metric]) for row in group) if len(group) > 1 else 0.0,
                    "values": [float(row[metric]) for row in group],
                } for metric in METRICS},
            })
            if name != "baseline":
                changes = {}
                for metric in METRICS:
                    values = [
                        100.0 * (row[metric] / baseline[row["seed"]][metric] - 1.0)
                        for row in group if row["seed"] in baseline
                    ]
                    if values:
                        changes[metric] = {"mean": statistics.fmean(values), "values": values}
                paired.append({"split": split, "experiment": name, "paired_percent_change": changes})
    result = {
        "description": "IC1 social-only mode-anchor bridge on frozen EXP-146B",
        "seeds": seeds, "experiments": names, "splits": splits,
        "rows": rows, "aggregate": aggregate, "paired_comparisons": paired,
    }
    (ROOT / "multiseed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--val-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--validation-seed", type=int, default=20260828)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--variants", nargs="+", choices=("all", *EXPERIMENTS), default=["all"])
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    args = parser.parse_args()
    names = list(EXPERIMENTS) if "all" in args.variants else args.variants
    splits = list(SPLITS) if "all" in args.splits else args.splits
    required = (
        PYTHON, TRAINER, BASELINE_DECODER, IC1_DECODER, DIAGNOSTIC,
        DATA / "eth_data_train.npy", DATA / "eth_num_train.npy",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for split in splits:
        for name in names:
            for seed in args.seeds:
                source = source_checkpoint(split, seed)
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = run_dir(name, split, seed)
                if (destination / "metrics.json").is_file() and not args.force:
                    print(f"[skip] {split} {name} seed={seed}", flush=True)
                    continue
                cmd = command(args, name, split, seed)
                if args.dry_run:
                    print("[dry-run] source=" + str(source) + " command=" + json.dumps(cmd))
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["ICMAB_SOURCE_CHECKPOINT"] = str(source)
                print(f"[run] {split} {name} seed={seed}", flush=True)
                with (destination / "console.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace", bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(f"[{split}/{name}] {line}", end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise RuntimeError(f"failed; see {destination / 'console.log'}")
                summarize(args.seeds, names, splits)
    if args.dry_run:
        print("[done] dry-run passed")
        return
    result = summarize(args.seeds, names, splits)
    expected = len(names) * len(splits) * len(args.seeds)
    print(f"[done] completed {len(result['rows'])}/{expected}")
    can_diagnose = (
        not args.skip_diagnostic and args.seeds == [1]
        and set(names) == set(EXPERIMENTS) and set(splits) == set(SPLITS)
    )
    if can_diagnose:
        diagnostic_cmd = [
            str(PYTHON), "-u", str(DIAGNOSTIC), "--device", args.device,
            "--batch-size", str(args.diagnostic_batch_size), "--splits", "all",
        ]
        print("[diagnostic] paired candidate geometry", flush=True)
        subprocess.run(diagnostic_cmd, cwd=WORKSPACE, check=True)
    elif not args.skip_diagnostic:
        print("[diagnostic] skipped: requires seed 1, both variants and both splits")


if __name__ == "__main__":
    main()
