"""Preprocess and run one whole-SDD potential experiment with the selected model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
PREPROCESSOR = ROOT / "preprocess_sdd_standard.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
DATA_DIR = ROOT / "processed_data"
OUTPUT_ROOT = ROOT / "EXP_132_CURRENT_MODEL_STANDARD_SDD" / "results"


def pixel_metrics(metrics: dict, pixel_scale: float) -> dict:
    oracle = metrics["oracle"]
    return {
        "samples": metrics["samples"],
        "minADE20_px": oracle["min_ade_20"] * pixel_scale,
        "same_path_FDE_px": oracle["same_path_fde"] * pixel_scale,
        "minFDE20_px": oracle["min_fde_20"] * pixel_scale,
        "top1_ADE_px": metrics["original"]["ade"] * pixel_scale,
        "top1_FDE_px": metrics["original"]["fde"] * pixel_scale,
    }


def write_summary(pixel_scale: float) -> None:
    metrics_path = (
        OUTPUT_ROOT / "f16_temporal_social_cross" / "seed_1" /
        "val_scene_1" / "metrics.json"
    )
    if not metrics_path.is_file():
        return
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = raw.get("evaluation_test_best_selected")
    if not test:
        return
    summary = {
        "experiment": "EXP_132 current selected model on standard whole-SDD split",
        "model": (
            "F16 temporal-social cross + kinematic K20 + coverage 0.20 + EMA "
            "+ focal heading normalization + 50% reflection/speed augmentation"
        ),
        "protocol": "31 train / 12 validation / 17 test videos, 8/12 at 2.5 Hz",
        "checkpoint_selection": "validation oracle_combo; test evaluated once afterward",
        "test_pixels": pixel_metrics(test, pixel_scale),
        "validation_pixels": pixel_metrics(raw["evaluation_best_selected"], pixel_scale),
        "parameters": raw["locked_backend"]["parameter_counts"],
        "elapsed_seconds": raw["elapsed_seconds"],
        "comparison_table_short_SDD_min20_pixels": {
            "PECNet": {"ADE": 9.96, "FDE": 15.88},
            "Trajectron++": {"ADE": 9.90, "FDE": 16.80},
            "MID": {"ADE": 9.73, "FDE": 15.32},
            "MemoNet": {"ADE": 8.56, "FDE": 12.70},
            "GSMNet": {"ADE": 8.30, "FDE": 12.70},
        },
        "interpretation_guardrail": (
            "Use minADE20/minFDE20 for literature comparison. Top-1 and same-path FDE "
            "are diagnostics, not substitutes for the published min20 columns."
        ),
    }
    (ROOT / "sdd_standard_result_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-scale", type=float, default=20.0)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--max-samples-per-video", type=int, default=0)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preprocess = [
        str(PYTHON), "-u", str(PREPROCESSOR),
        "--dataset-root", str(args.dataset_root), "--output-dir", str(DATA_DIR),
        "--max-agents", str(args.max_agents), "--pixel-scale", str(args.pixel_scale),
        "--max-samples-per-video", str(args.max_samples_per_video),
    ]
    if args.rebuild_data:
        preprocess.append("--rebuild")

    train = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs/motif_rbf_univ.yaml"),
        "--data-path", str(DATA_DIR / "sdd_standard_train_val_data.npy"),
        "--num-path", str(DATA_DIR / "sdd_standard_train_val_num.npy"),
        "--test-data-path", str(DATA_DIR / "sdd_standard_test_data.npy"),
        "--test-num-path", str(DATA_DIR / "sdd_standard_test_num.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", "1", "--val-scene-id", "1", "--epochs", str(args.epochs),
        "--val-freq", "2", "--batch-size", str(args.batch_size),
        "--device", args.device, "--max-agents", str(args.max_agents),
        "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "warm_restarts", "--output-root", str(OUTPUT_ROOT),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if args.force:
        train.append("--force")

    if args.dry_run:
        print("[dry-run preprocess] " + json.dumps(preprocess), flush=True)
        print("[dry-run train] " + json.dumps(train), flush=True)
        return

    subprocess.run(preprocess, check=True, cwd=ROOT)
    run_dir = OUTPUT_ROOT / "f16_temporal_social_cross" / "seed_1" / "val_scene_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            train, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"training failed; see {run_dir / 'console.log'}")
    write_summary(args.pixel_scale)
    print(f"[done] {ROOT / 'sdd_standard_result_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
