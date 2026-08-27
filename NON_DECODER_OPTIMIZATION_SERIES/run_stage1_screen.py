"""Screen training/data optimizations with the decoder and F16 architecture locked."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"

JOBS = [
    ("EXP_074_LOCKED_BASELINE", []),
    ("EXP_075_SCENE_SAMPLING", ["--scene-sampling", "sqrt_scene"]),
    ("EXP_076_SCENE_EMA", ["--scene-sampling", "sqrt_scene", "--ema-decay", "0.995"]),
    (
        "EXP_077_TRAJECTORY_AUGMENTATION",
        ["--scene-sampling", "sqrt_scene", "--ema-decay", "0.995",
         "--augment-rotation", "--augment-reflection",
         "--augment-speed-min", "0.9", "--augment-speed-max", "1.1"],
    ),
    (
        "EXP_078_KINEMATIC_CONSISTENCY",
        ["--scene-sampling", "sqrt_scene", "--ema-decay", "0.995",
         "--velocity-loss-weight", "0.10", "--acceleration-loss-weight", "0.02"],
    ),
    (
        "EXP_079_COMBINED_TRAINING",
        ["--scene-sampling", "sqrt_scene", "--ema-decay", "0.995",
         "--augment-rotation", "--augment-reflection",
         "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
         "--velocity-loss-weight", "0.10", "--acceleration-loss-weight", "0.02"],
    ),
]


def main() -> None:
    rows = []
    for name, extra in JOBS:
        output = ROOT / name / "results"
        run_dir = output / "f16_temporal_social_cross" / "seed_1" / "val_scene_1"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
            "--variant", "f16_temporal_social_cross",
            "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
            "--seed", "1", "--val-scene-id", "1", "--epochs", "30", "--val-freq", "5",
            "--device", "cuda:0", "--scheduler", "warm_restarts",
            "--output-root", str(output), "--oracle-coverage-weight", "0.20",
            "--oracle-coverage-temperature", "0.10",
            "--checkpoint-criterion", "oracle_combo", "--force",
            *extra,
        ]
        print(f"[run] {name}", flush=True)
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Failed: {name}; see {run_dir / 'console.log'}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        selected = metrics["evaluation_best_selected"]
        rows.append({
            "experiment": name,
            "min_ade_20": selected["oracle"]["min_ade_20"],
            "same_path_fde": selected["oracle"]["same_path_fde"],
            "min_fde_20": selected["oracle"]["min_fde_20"],
            "top1_ade": selected["original"]["ade"],
            "top1_fde": selected["original"]["fde"],
            "parameters": metrics["locked_backend"]["parameter_counts"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        })
        (ROOT / "stage1_screen_summary.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print("[done] stage 1 screening", flush=True)


if __name__ == "__main__":
    main()
