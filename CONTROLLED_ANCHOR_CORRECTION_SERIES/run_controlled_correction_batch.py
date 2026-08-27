"""Run EXP-142 controlled anchor correction experiments."""

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
DECODER = ROOT / "controlled_anchor_correction_models.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SOURCE_ROOT = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results"
EXPERIMENTS = {
    "controlled": {
        "root": ROOT / "EXP_142A_CONTROLLED_ANCHOR_CORRECTION",
        "decoder": "c1_controlled_anchor_correction",
        "coverage_weight": 0.20, "coverage_fde": 0.25,
        "coverage_temperature": 0.10, "normalized": False,
    },
    "samepath_aligned": {
        "root": ROOT / "EXP_142B_SAMEPATH_ALIGNED_CORRECTION",
        "decoder": "c2_samepath_aligned_correction",
        "coverage_weight": 0.30, "coverage_fde": 0.50,
        "coverage_temperature": 0.25, "normalized": True,
    },
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
LABELS = ("source_anchor_proposal", "controlled", "samepath_aligned")


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


def aggregate(rows: list[dict]) -> list[dict]:
    names = [
        "min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde",
        "coarse_min_ade_20", "coarse_min_fde_20", "coarse_same_path_fde",
        "refinement_min_ade_delta", "refinement_min_fde_delta",
        "top_m_coarse_oracle_ade_recall", "elapsed_seconds",
    ]
    output = []
    for split in SPLITS:
        for label in LABELS:
            group = [r for r in rows if r["split"] == split and r["experiment"] == label]
            if not group: continue
            metrics = {}
            for name in names:
                values = [float(r[name]) for r in group if r.get(name) is not None]
                if values:
                    metrics[name] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "values": values}
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
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = [PYTHON, TRAINER, DECODER, DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    for split in args.splits:
        split_mode, fraction = SPLITS[split]
        for label in args.experiments:
            setting = EXPERIMENTS[label]
            for seed in args.seeds:
                source = run_dir(SOURCE_ROOT, split, seed) / "best_selected.pth"
                if not source.is_file(): raise FileNotFoundError(source)
                result_root = setting["root"] / "results"
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
                    "--decoder-experiment", setting["decoder"], "--decoder-module", str(DECODER),
                    "--seed", str(seed), "--val-scene-id", "-1",
                    "--validation-fraction", str(fraction), "--validation-seed", str(args.validation_seed),
                    "--validation-split-mode", split_mode,
                    "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size), "--device", args.device,
                    "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
                    "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
                    "--output-root", str(result_root / split),
                    "--oracle-coverage-weight", str(setting["coverage_weight"]),
                    "--oracle-coverage-temperature", str(setting["coverage_temperature"]),
                    "--oracle-coverage-fde-weight", str(setting["coverage_fde"]),
                    "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
                    "--focal-heading-normalize", "--augment-reflection",
                    "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
                ]
                if setting["normalized"]: command.append("--oracle-coverage-normalized")
                if args.force: command.append("--force")
                env = os.environ.copy(); env["EXP142_SOURCE_CHECKPOINT"] = str(source)
                print(f"[run] {label} split={split} seed={seed}", flush=True)
                completed = subprocess.run(command, env=env)
                if completed.returncode:
                    log = run_dir(result_root, split, seed) / "console.log"
                    raise RuntimeError(f"Failed {label} split={split} seed={seed}; see {log}")

    rows = []
    for split in SPLITS:
        for seed in args.seeds:
            source_metrics = run_dir(SOURCE_ROOT, split, seed) / "metrics.json"
            if source_metrics.is_file(): rows.append(read_row("source_anchor_proposal", split, seed, source_metrics))
            for label, setting in EXPERIMENTS.items():
                result = run_dir(setting["root"] / "results", split, seed) / "metrics.json"
                if result.is_file(): rows.append(read_row(label, split, seed, result))
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-142 frozen-source bounded three-anchor correction",
        "protocol": {
            "source": "matched EXP-139C best_selected", "source_frozen": True,
            "scorer_frozen": True, "corrected_anchor_steps": [4, 8, 12],
            "max_offsets": [0.20, 0.25, 0.35], "all_20_corrected": True,
            "endpoint_locked": False, "cv_reference": False,
        },
        "rows": rows, "aggregate": aggregate(rows),
    }
    (ROOT / "multiseed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__": main()

