"""Run EXP-143 anchor-correction mechanism validation experiments."""

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
DECODER = ROOT / "anchor_correction_validation_models.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
C_SOURCE = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "EXP_139C_THREE_ANCHOR_PROPOSAL_ONLY" / "results"
A_SOURCE = WORKSPACE / "CONTROLLED_ANCHOR_CORRECTION_SERIES" / "EXP_142A_CONTROLLED_ANCHOR_CORRECTION" / "results"
EXPERIMENTS = {
    "endpoint_only": {
        "root": ROOT / "EXP_143A_ENDPOINT_ONLY_CORRECTION",
        "decoder": "v1_endpoint_only_correction", "source": C_SOURCE,
        "coverage_weight": 0.20, "criterion": "oracle_combo",
    },
    "wide_endpoint": {
        "root": ROOT / "EXP_143B_WIDE_ENDPOINT_CORRECTION",
        "decoder": "v2_wide_endpoint_correction", "source": C_SOURCE,
        "coverage_weight": 0.20, "criterion": "oracle_combo",
    },
    "quality_rescore": {
        "root": ROOT / "EXP_143C_CORRECTED_QUALITY_RESCORING",
        "decoder": "v3_corrected_quality_rescoring", "source": A_SOURCE,
        "coverage_weight": 0.0, "criterion": "original_ade",
    },
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
LABELS = ("source_anchor_proposal", "exp142a_reference", *EXPERIMENTS)


def run_dir(root: Path, split: str, seed: int) -> Path:
    return root / split / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def read_row(label: str, split: str, seed: int, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    evaluation = raw["evaluation_best_selected"]
    diagnostics = evaluation.get("proposal_diagnostics", {})
    counts = raw["locked_backend"]["parameter_counts"]
    row = {
        "experiment": label, "split": split, "seed": seed,
        "min_ade_20": evaluation["oracle"]["min_ade_20"],
        "min_fde_20": evaluation["oracle"]["min_fde_20"],
        "same_path_fde": evaluation["oracle"]["same_path_fde"],
        "top1_ade": evaluation["original"]["ade"],
        "top1_fde": evaluation["original"]["fde"],
        "parameters": counts["total"], "trainable_parameters": counts["trainable"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }
    for key in ("anchor_correction_mean", "anchor_correction_p95", "anchor_correction_max"):
        row[key] = json.dumps(diagnostics.get(key), ensure_ascii=False)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    names = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde", "elapsed_seconds")
    output = []
    for split in SPLITS:
        for label in LABELS:
            group = [row for row in rows if row["split"] == split and row["experiment"] == label]
            if not group:
                continue
            metrics = {}
            for name in names:
                values = [float(row[name]) for row in group]
                metrics[name] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "values": values}
            output.append({
                "split": split, "experiment": label,
                "parameters": group[0]["parameters"],
                "trainable_parameters": group[0]["trainable_parameters"],
                "metrics": metrics,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    for split in args.splits:
        split_mode, fraction = SPLITS[split]
        for label in args.experiments:
            setting = EXPERIMENTS[label]
            for seed in args.seeds:
                source = run_dir(setting["source"], split, seed) / "best_selected.pth"
                if not source.is_file():
                    raise FileNotFoundError(source)
                result_root = setting["root"] / "results"
                metrics = run_dir(result_root, split, seed) / "metrics.json"
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
                    "--oracle-coverage-temperature", "0.10", "--oracle-coverage-fde-weight", "0.25",
                    "--checkpoint-criterion", setting["criterion"], "--ema-decay", "0.995",
                    "--focal-heading-normalize", "--augment-reflection",
                    "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
                ]
                if label == "quality_rescore":
                    command.extend(["--classification-weight", "1.0"])
                if args.force:
                    command.append("--force")
                env = os.environ.copy()
                env["EXP143_SOURCE_CHECKPOINT"] = str(source)
                print(f"[run] {label} split={split} seed={seed}", flush=True)
                completed = subprocess.run(command, env=env)
                if completed.returncode:
                    log = run_dir(result_root, split, seed) / "console.log"
                    raise RuntimeError(f"Failed {label} split={split} seed={seed}; see {log}")

    rows = []
    for split in SPLITS:
        for seed in args.seeds:
            for label, root in (("source_anchor_proposal", C_SOURCE), ("exp142a_reference", A_SOURCE)):
                path = run_dir(root, split, seed) / "metrics.json"
                if path.is_file():
                    rows.append(read_row(label, split, seed, path))
            for label, setting in EXPERIMENTS.items():
                path = run_dir(setting["root"] / "results", split, seed) / "metrics.json"
                if path.is_file():
                    rows.append(read_row(label, split, seed, path))
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "description": "EXP-143 endpoint freedom, anchor necessity, and corrected-candidate scoring validation",
        "protocol": {"splits": SPLITS, "seeds": args.seeds, "epochs": args.epochs, "cv_reference": False},
        "rows": rows, "aggregate": aggregate(rows),
    }
    (ROOT / "multiseed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary={ROOT / 'multiseed_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
