"""Run four matched EXP-138 non-decoder experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CUSTOM = ROOT / "targeted_models.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
BASELINE_ROOT = WORKSPACE / "FRONT_DOMAIN_ADAPTATION_SERIES" / "EXP_135_BASELINE_CONTROL"

EXPERIMENTS = {
    "residual_shape_scale": (ROOT / "EXP_138A_RESIDUAL_SHAPE_SCALE", "t1_residual_shape_scale"),
    "future_difficulty_loss": (ROOT / "EXP_138B_FUTURE_DIFFICULTY_LOSS", "none"),
    "wide_speed_augmentation": (ROOT / "EXP_138C_WIDE_SPEED_AUGMENTATION", "none"),
    "curvature_augmentation": (ROOT / "EXP_138D_CURVATURE_AUGMENTATION", "none"),
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde", "elapsed_seconds")


def result_dir(name: str, split: str, seed: int) -> Path:
    root, _ = EXPERIMENTS[name]
    return root / "results" / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def baseline_path(split: str, seed: int) -> Path:
    return BASELINE_ROOT / "results" / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "metrics.json"


def command_for(args, name: str, split: str, seed: int) -> list[str]:
    output_root, encoder_experiment = EXPERIMENTS[name]
    split_mode, validation_fraction = SPLITS[split]
    speed_min, speed_max = (0.75, 1.25) if name == "wide_speed_augmentation" else (0.9, 1.1)
    command = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT), "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"), "--num-path", str(DATA / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", str(seed), "--val-scene-id", "-1",
        "--validation-fraction", str(validation_fraction), "--validation-seed", str(args.validation_seed),
        "--validation-split-mode", split_mode, "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "warm_restarts", "--output-root", str(output_root / "results" / split),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", str(speed_min), "--augment-speed-max", str(speed_max), "--augment-probability", "0.5",
    ]
    if encoder_experiment != "none":
        command += ["--encoder-experiment", encoder_experiment, "--encoder-module", str(CUSTOM)]
    if name == "future_difficulty_loss":
        command += ["--future-difficulty-weight", str(args.future_difficulty_weight)]
    if name == "curvature_augmentation":
        command += ["--augment-curvature-max", str(args.curvature_max)]
    if args.force:
        command.append("--force")
    return command


def row_from(path: Path, name: str, split: str, seed: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8")); evaluation = raw["evaluation_best_selected"]
    return {
        "experiment": name, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"], "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"], "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"], "parameters": raw["locked_backend"]["parameter_counts"]["total"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def summarize(seeds: list[int]) -> dict:
    rows = []
    for split in SPLITS:
        for seed in seeds:
            path = baseline_path(split, seed)
            if path.is_file(): rows.append(row_from(path, "baseline", split, seed))
        for name in EXPERIMENTS:
            for seed in seeds:
                path = result_dir(name, split, seed) / "metrics.json"
                if path.is_file(): rows.append(row_from(path, name, split, seed))
    aggregate, paired = [], []
    for split in SPLITS:
        baseline = {int(r["seed"]): r for r in rows if r["split"] == split and r["experiment"] == "baseline"}
        for name in ("baseline", *EXPERIMENTS):
            selected = sorted((r for r in rows if r["split"] == split and r["experiment"] == name), key=lambda r: int(r["seed"]))
            if not selected: continue
            aggregate.append({"split": split, "experiment": name, "parameters": int(selected[0]["parameters"]), "metrics": {
                metric: {"mean": statistics.fmean(float(r[metric]) for r in selected), "std": statistics.stdev(float(r[metric]) for r in selected) if len(selected) > 1 else 0.0, "values": [float(r[metric]) for r in selected]}
                for metric in METRICS}})
            if name != "baseline":
                changes = {}
                for metric in METRICS[:-1]:
                    values = [100.0 * (float(r[metric]) / float(baseline[int(r["seed"])][metric]) - 1.0) for r in selected if int(r["seed"]) in baseline]
                    changes[metric] = {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "values": values}
                paired.append({"split": split, "experiment": name, "paired_percent_change_vs_baseline": changes})
    result = {"description": "EXP-138 matched three-seed targeted generalization screen", "seeds": seeds, "rows": rows, "aggregate": aggregate, "paired_comparisons": paired}
    (ROOT / "multiseed_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (ROOT / "all_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); parser.add_argument("--validation-seed", type=int, default=20260825)
    parser.add_argument("--future-difficulty-weight", type=float, default=0.8); parser.add_argument("--curvature-max", type=float, default=0.025)
    parser.add_argument("--variants", nargs="+", choices=("all", *EXPERIMENTS), default=["all"])
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--force", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (PYTHON, TRAINER, CUSTOM, DECODER, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"):
        if not path.is_file(): raise FileNotFoundError(path)
    names = list(EXPERIMENTS) if "all" in args.variants else args.variants
    splits = list(SPLITS) if "all" in args.splits else args.splits
    with (ROOT / "console.log").open("a", encoding="utf-8") as driver_log:
        for split in splits:
            for name in names:
                for seed in args.seeds:
                    destination = result_dir(name, split, seed); metrics = destination / "metrics.json"
                    if metrics.is_file() and not args.force:
                        print(f"[skip] {split} {name} seed={seed}", flush=True); continue
                    command = command_for(args, name, split, seed)
                    if args.dry_run:
                        print("[dry-run] " + json.dumps(command), flush=True); continue
                    destination.mkdir(parents=True, exist_ok=True); print(f"[run] {split} {name} seed={seed}", flush=True)
                    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                    assert process.stdout is not None
                    with (destination / "console.log").open("w", encoding="utf-8") as run_log:
                        for line in process.stdout:
                            print(f"[{split}/{name}] {line}", end="", flush=True); run_log.write(line); run_log.flush(); driver_log.write(line); driver_log.flush()
                    if process.wait(): raise RuntimeError(f"Failed; see {destination / 'console.log'}")
                    summarize(args.seeds)
    if args.dry_run: print("[done] dry-run passed", flush=True)
    else:
        result = summarize(args.seeds)
        completed = sum(1 for r in result["rows"] if r["experiment"] in names and r["split"] in splits and int(r["seed"]) in args.seeds)
        print(f"[done] new runs {completed}/{len(names)*len(splits)*len(args.seeds)}; summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__": main()
