from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PREPROCESSOR = ROOT / "preprocess_sdd_lowcost.py"
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
DATA_DIR = ROOT / "processed_data"
DATA_PATH = DATA_DIR / "sdd_bookstore_lowcost_data.npy"
NUM_PATH = DATA_DIR / "sdd_bookstore_lowcost_num.npy"

EXPERIMENTS = {
    "n9_k4": {
        "folder": "EXP_130_SDD_N9_K4",
        "max_agents": 9,
        "top_k": 4,
    },
    "n16_k6": {
        "folder": "EXP_131_SDD_N16_K6",
        "max_agents": 16,
        "top_k": 6,
    },
}


def collect_summary(pixel_scale: float) -> None:
    rows = []
    for label, spec in EXPERIMENTS.items():
        metrics_path = (
            ROOT / spec["folder"] / "results" / "f16_temporal_social_cross" /
            "seed_1" / "val_scene_2" / "metrics.json"
        )
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = metrics["evaluation_best_selected"]
        oracle = selected["oracle"]
        rows.append({
            "experiment": label,
            "max_agents": spec["max_agents"],
            "top_k": spec["top_k"],
            "samples": selected["samples"],
            "normalized": {
                **oracle,
                "top1_ade": selected["original"]["ade"],
                "top1_fde": selected["original"]["fde"],
            },
            "pixels": {
                "min_ade_20": oracle["min_ade_20"] * pixel_scale,
                "same_path_fde": oracle["same_path_fde"] * pixel_scale,
                "min_fde_20": oracle["min_fde_20"] * pixel_scale,
                "top1_ade": selected["original"]["ade"] * pixel_scale,
                "top1_fde": selected["original"]["fde"] * pixel_scale,
            },
            "parameters": metrics["locked_backend"]["parameter_counts"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        })
    (ROOT / "sdd_lowcost_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-samples-per-video", type=int, default=600)
    parser.add_argument("--pixel-scale", type=float, default=20.0)
    parser.add_argument("--only", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    preprocess_command = [
        str(PYTHON), "-u", str(PREPROCESSOR),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(DATA_DIR),
        "--max-samples-per-video", str(args.max_samples_per_video),
        "--max-agents", "16", "--pixel-scale", str(args.pixel_scale),
    ]
    if args.rebuild_data:
        preprocess_command.append("--rebuild")
    if args.dry_run:
        print("[dry-run] " + json.dumps(preprocess_command), flush=True)
    else:
        subprocess.run(preprocess_command, check=True, cwd=ROOT)

    labels = list(EXPERIMENTS) if args.only == "all" else [args.only]
    for label in labels:
        spec = EXPERIMENTS[label]
        output_root = ROOT / spec["folder"] / "results"
        run_dir = (
            output_root / "f16_temporal_social_cross" /
            "seed_1" / "val_scene_2"
        )
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[skip] {label}: {metrics_path}", flush=True)
            continue
        command = [
            str(PYTHON), "-u", str(TRAINER),
            "--project-root", str(PROJECT),
            "--config-file", str(PROJECT / "configs/motif_rbf_univ.yaml"),
            "--data-path", str(DATA_PATH), "--num-path", str(NUM_PATH),
            "--variant", "f16_temporal_social_cross",
            "--decoder-experiment", "kinematic_k20",
            "--decoder-module", str(DECODER),
            "--seed", "1", "--val-scene-id", "2",
            "--epochs", str(args.epochs), "--val-freq", "2",
            "--batch-size", "32", "--device", args.device,
            "--max-agents", str(spec["max_agents"]),
            "--graph-top-k", str(spec["top_k"]), "--graph-radius", "6.0",
            "--scheduler", "warm_restarts", "--output-root", str(output_root),
            "--oracle-coverage-weight", "0.20",
            "--oracle-coverage-temperature", "0.10",
            "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
            "--focal-heading-normalize", "--augment-reflection",
            "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
            "--augment-probability", "0.5",
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            print("[dry-run] " + json.dumps(command), flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] {label}", flush=True)
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"Failed: {label}; see {run_dir / 'console.log'}")
        collect_summary(args.pixel_scale)

    if args.dry_run:
        print("[done] dry-run validation passed", flush=True)
    else:
        collect_summary(args.pixel_scale)
        print(f"[done] {ROOT / 'sdd_lowcost_summary.json'}", flush=True)


if __name__ == "__main__":
    run()
