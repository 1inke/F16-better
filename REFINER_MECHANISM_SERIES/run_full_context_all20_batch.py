"""Run EXP-140B full-context all-20 refiner against matched EXP-139C sources."""

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
DECODER = ROOT / "full_context_all20_refiner_models.py"
SOURCE_ROOT = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results"
OUTPUT_ROOT = ROOT / "EXP_140B_FROZEN_FULL_CONTEXT_ALL20" / "results"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}


def run_dir(root: Path, split: str, seed: int) -> Path:
    return root / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def read_row(label: str, split: str, seed: int, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    diagnostics = evaluation.get("proposal_diagnostics", {})
    counts = raw["locked_backend"]["parameter_counts"]
    return {
        "experiment": label, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"],
        "coarse_min_ade_20": diagnostics.get("coarse_min_ade_20"),
        "coarse_min_fde_20": diagnostics.get("coarse_min_fde_20"),
        "coarse_same_path_fde": diagnostics.get("coarse_same_path_fde"),
        "refinement_min_ade_delta": diagnostics.get("refinement_min_ade_delta"),
        "refinement_min_fde_delta": diagnostics.get("refinement_min_fde_delta"),
        "top_m_coarse_oracle_ade_recall": diagnostics.get("top_m_coarse_oracle_ade_recall"),
        "parameters": counts["total"], "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }


def summarize(rows: list[dict]) -> list[dict]:
    metric_names = [
        "min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde",
        "coarse_min_ade_20", "coarse_min_fde_20", "coarse_same_path_fde",
        "refinement_min_ade_delta", "refinement_min_fde_delta",
        "top_m_coarse_oracle_ade_recall", "elapsed_seconds",
    ]
    output = []
    for split in SPLITS:
        for label in ("source_anchor_proposal", "full_context_all20_refine"):
            group = [row for row in rows if row["split"] == split and row["experiment"] == label]
            if not group:
                continue
            metrics = {}
            for name in metric_names:
                values = [float(row[name]) for row in group if row.get(name) is not None]
                if values:
                    metrics[name] = {
                        "mean": statistics.fmean(values), "std": statistics.pstdev(values),
                        "values": values,
                    }
            output.append({
                "split": split, "experiment": label,
                "parameters": group[0]["parameters"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "metrics": metrics,
            })
    return output


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
        split_mode, fraction = SPLITS[split]
        for seed in args.seeds:
            source = run_dir(SOURCE_ROOT, split, seed) / "best_selected.pth"
            if not source.is_file():
                raise FileNotFoundError(source)
            metrics = run_dir(OUTPUT_ROOT, split, seed) / "metrics.json"
            if metrics.is_file() and not args.force:
                print(f"[skip] full_context split={split} seed={seed}", flush=True)
                continue
            command = [
                str(PYTHON), "-u", str(TRAINER),
                "--project-root", str(PROJECT),
                "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                "--data-path", str(DATA / "eth_data_train.npy"),
                "--num-path", str(DATA / "eth_num_train.npy"),
                "--variant", "f16_temporal_social_cross",
                "--decoder-experiment", "r2_frozen_full_context_all20",
                "--decoder-module", str(DECODER),
                "--seed", str(seed), "--val-scene-id", "-1",
                "--validation-fraction", str(fraction),
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
            print(f"[run] full_context_all20 split={split} seed={seed}", flush=True)
            completed = subprocess.run(command, env=environment)
            if completed.returncode:
                log = run_dir(OUTPUT_ROOT, split, seed) / "console.log"
                raise RuntimeError(f"Failed split={split} seed={seed}; see {log}")

    rows = []
    for split in SPLITS:
        for seed in args.seeds:
            source = run_dir(SOURCE_ROOT, split, seed) / "metrics.json"
            result = run_dir(OUTPUT_ROOT, split, seed) / "metrics.json"
            if source.is_file(): rows.append(read_row("source_anchor_proposal", split, seed, source))
            if result.is_file(): rows.append(read_row("full_context_all20_refine", split, seed, result))
    if rows:
        with (ROOT / "full_context_all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-140B frozen source with history+anchor full-context all-20 refinement",
        "mechanism": {
            "source_frozen": True, "scorer_frozen": True,
            "train_refiner_only": True, "train_selection": "all 20",
            "inference_selection": "all 20", "teacher_forcing": False,
            "refiner_memory": "8 encoded history tokens + 3 learned anchor tokens",
        },
        "rows": rows, "aggregate": summarize(rows),
    }
    (ROOT / "full_context_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] summary={ROOT / 'full_context_multiseed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

