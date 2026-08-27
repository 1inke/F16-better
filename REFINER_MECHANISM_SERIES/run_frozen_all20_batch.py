"""Run EXP-140A frozen-proposal/all-20-refinement experiments."""

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
DECODER = ROOT / "frozen_all20_refiner_models.py"
SOURCE_ROOT = (
    WORKSPACE / "HIERARCHICAL_DECODER_SERIES"
    / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results"
)
OUTPUT_ROOT = ROOT / "EXP_140A_FROZEN_ALL20_REFINER" / "results"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}


def run_dir(root: Path, split: str, seed: int) -> Path:
    return root / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def source_checkpoint(split: str, seed: int) -> Path:
    return run_dir(SOURCE_ROOT, split, seed) / "best_selected.pth"


def result_row(experiment: str, split: str, seed: int, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    oracle = evaluation["oracle"]
    original = evaluation["original"]
    diagnostics = evaluation.get("proposal_diagnostics", {})
    counts = raw["locked_backend"]["parameter_counts"]
    return {
        "experiment": experiment,
        "split": split,
        "seed": seed,
        "min_ade_20": oracle["min_ade_20"],
        "min_fde_20": oracle["min_fde_20"],
        "same_path_fde": oracle["same_path_fde"],
        "top1_ade": original["ade"],
        "top1_fde": original["fde"],
        "coarse_min_ade_20": diagnostics.get("coarse_min_ade_20"),
        "coarse_min_fde_20": diagnostics.get("coarse_min_fde_20"),
        "coarse_same_path_fde": diagnostics.get("coarse_same_path_fde"),
        "refinement_min_ade_delta": diagnostics.get("refinement_min_ade_delta"),
        "refinement_min_fde_delta": diagnostics.get("refinement_min_fde_delta"),
        "top_m_coarse_oracle_ade_recall": diagnostics.get("top_m_coarse_oracle_ade_recall"),
        "parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = [
        "min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde",
        "coarse_min_ade_20", "coarse_min_fde_20", "coarse_same_path_fde",
        "refinement_min_ade_delta", "refinement_min_fde_delta",
        "top_m_coarse_oracle_ade_recall", "elapsed_seconds",
    ]
    groups = []
    for split in SPLITS:
        for experiment in ("source_anchor_proposal", "frozen_all20_refine"):
            selected = [r for r in rows if r["split"] == split and r["experiment"] == experiment]
            if not selected:
                continue
            item = {
                "split": split,
                "experiment": experiment,
                "parameters": selected[0]["parameters"],
                "trainable_parameters": selected[0]["trainable_parameters"],
                "metrics": {},
            }
            for metric in metrics:
                values = [float(r[metric]) for r in selected if r.get(metric) is not None]
                if values:
                    item["metrics"][metric] = {
                        "mean": statistics.fmean(values),
                        "std": statistics.pstdev(values),
                        "values": values,
                    }
            groups.append(item)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--validation-seed", type=int, default=20260825)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    required = [PYTHON, TRAINER, DECODER, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    for split in args.splits:
        split_mode, validation_fraction = SPLITS[split]
        for seed in args.seeds:
            source = source_checkpoint(split, seed)
            if not source.is_file():
                raise FileNotFoundError(f"Missing matched EXP-139C checkpoint: {source}")
            metrics = run_dir(OUTPUT_ROOT, split, seed) / "metrics.json"
            if metrics.is_file() and not args.force:
                print(f"[skip] {split} seed={seed}", flush=True)
                continue
            command = [
                str(PYTHON), "-u", str(TRAINER),
                "--project-root", str(PROJECT),
                "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                "--data-path", str(DATA / "eth_data_train.npy"),
                "--num-path", str(DATA / "eth_num_train.npy"),
                "--variant", "f16_temporal_social_cross",
                "--decoder-experiment", "r1_frozen_all20_refine",
                "--decoder-module", str(DECODER),
                "--seed", str(seed), "--val-scene-id", "-1",
                "--validation-fraction", str(validation_fraction),
                "--validation-seed", str(args.validation_seed),
                "--validation-split-mode", split_mode,
                "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                "--batch-size", str(args.batch_size), "--device", args.device,
                "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
                "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
                "--output-root", str(OUTPUT_ROOT / split),
                "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
                "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
                "--focal-heading-normalize", "--augment-reflection",
                "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
                "--augment-probability", "0.5",
            ]
            if args.force:
                command.append("--force")
            environment = os.environ.copy()
            environment["EXP140_SOURCE_CHECKPOINT"] = str(source)
            print(f"[run] frozen_all20_refine split={split} seed={seed}", flush=True)
            completed = subprocess.run(command, env=environment)
            if completed.returncode:
                log = run_dir(OUTPUT_ROOT, split, seed) / "console.log"
                raise RuntimeError(f"Failed split={split} seed={seed}; see {log}")

    rows = []
    # Always rebuild the summary from every result already present, even when
    # only one split was requested for a corrective rerun.
    for split in SPLITS:
        for seed in args.seeds:
            source_metrics = run_dir(SOURCE_ROOT, split, seed) / "metrics.json"
            new_metrics = run_dir(OUTPUT_ROOT, split, seed) / "metrics.json"
            if source_metrics.is_file():
                rows.append(result_row("source_anchor_proposal", split, seed, source_metrics))
            if new_metrics.is_file():
                rows.append(result_row("frozen_all20_refine", split, seed, new_metrics))

    fields = list(rows[0]) if rows else []
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-140A frozen EXP-139C proposal/scorer with all-20 refinement",
        "mechanism": {
            "source_frozen": True,
            "source_checkpoint": "matched split/seed EXP-139C best_selected.pth",
            "train_refiner_only": True,
            "train_selection": "all 20",
            "inference_selection": "all 20",
            "teacher_forcing": False,
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (ROOT / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
