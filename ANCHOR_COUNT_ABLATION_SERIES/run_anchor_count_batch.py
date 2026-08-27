"""Run 2/4/6-anchor proposal-only ablations and compare with EXP-139C."""

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
DECODER = ROOT / "anchor_count_models.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
REFERENCE_ROOT = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results"
EXPERIMENTS = {
    "anchor_2": (ROOT / "EXP_141A_TWO_ANCHOR_PROPOSAL", "a2_anchor_proposal"),
    "anchor_4": (ROOT / "EXP_141B_FOUR_ANCHOR_PROPOSAL", "a4_anchor_proposal"),
    "anchor_6": (ROOT / "EXP_141C_SIX_ANCHOR_PROPOSAL", "a6_anchor_proposal"),
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
LABEL_ORDER = ("anchor_2", "anchor_3_reference", "anchor_4", "anchor_6")


def run_dir(root: Path, split: str, seed: int) -> Path:
    return root / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def metric_path(root: Path, split: str, seed: int) -> Path:
    return run_dir(root, split, seed) / "metrics.json"


def read_row(label: str, split: str, seed: int, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    diagnostics = evaluation.get("proposal_diagnostics", {})
    counts = raw["locked_backend"]["parameter_counts"]
    return {
        "experiment": label,
        "split": split,
        "seed": seed,
        "anchor_steps": "-".join(map(str, raw["locked_backend"]["hierarchical_decoder"]["anchor_steps"])),
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"],
        "coarse_min_ade_20": diagnostics.get("coarse_min_ade_20"),
        "coarse_min_fde_20": diagnostics.get("coarse_min_fde_20"),
        "coarse_same_path_fde": diagnostics.get("coarse_same_path_fde"),
        "parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def aggregate(rows: list[dict]) -> list[dict]:
    names = [
        "min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde",
        "elapsed_seconds",
    ]
    output = []
    for split in SPLITS:
        for label in LABEL_ORDER:
            group = [r for r in rows if r["split"] == split and r["experiment"] == label]
            if not group:
                continue
            metrics = {}
            for name in names:
                values = [float(r[name]) for r in group]
                metrics[name] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values),
                    "values": values,
                }
            output.append({
                "split": split,
                "experiment": label,
                "anchor_steps": group[0]["anchor_steps"],
                "parameters": group[0]["parameters"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "metrics": metrics,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
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
                metrics = metric_path(experiment_root / "results", split, seed)
                if metrics.is_file() and not args.force:
                    print(f"[skip] {label} split={split} seed={seed}", flush=True)
                    continue
                command = [
                    str(PYTHON), "-u", str(TRAINER),
                    "--project-root", str(PROJECT),
                    "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                    "--data-path", str(DATA / "eth_data_train.npy"),
                    "--num-path", str(DATA / "eth_num_train.npy"),
                    "--variant", "f16_temporal_social_cross",
                    "--decoder-experiment", decoder_name,
                    "--decoder-module", str(DECODER),
                    "--seed", str(seed), "--val-scene-id", "-1",
                    "--validation-fraction", str(fraction),
                    "--validation-seed", str(args.validation_seed),
                    "--validation-split-mode", split_mode,
                    "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size), "--device", args.device,
                    "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
                    "--scheduler", "warm_restarts",
                    "--output-root", str(experiment_root / "results" / split),
                    "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
                    "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
                    "--focal-heading-normalize", "--augment-reflection",
                    "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
                    "--augment-probability", "0.5",
                ]
                if args.force:
                    command.append("--force")
                print(f"[run] {label} split={split} seed={seed}", flush=True)
                completed = subprocess.run(command)
                if completed.returncode:
                    log = run_dir(experiment_root / "results", split, seed) / "console.log"
                    raise RuntimeError(f"Failed {label} split={split} seed={seed}; see {log}")

    rows = []
    for split in SPLITS:
        for seed in args.seeds:
            reference = metric_path(REFERENCE_ROOT, split, seed)
            if reference.is_file():
                rows.append(read_row("anchor_3_reference", split, seed, reference))
            for label, (experiment_root, _) in EXPERIMENTS.items():
                result = metric_path(experiment_root / "results", split, seed)
                if result.is_file():
                    rows.append(read_row(label, split, seed, result))
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-141 proposal-only learned-anchor count ablation",
        "protocol": {
            "anchor_2": [6, 12], "anchor_3_reference": [4, 8, 12],
            "anchor_4": [3, 6, 9, 12], "anchor_6": [2, 4, 6, 8, 10, 12],
            "num_modes": 20, "cv_reference": False, "use_refiner": False,
            "validation_seed": args.validation_seed,
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (ROOT / "multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

