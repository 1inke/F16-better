"""Three-seed replication using the exact F16 warm-restarts schedule."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
OUTPUT = ROOT / "results_warm_replication"

for seed in (1, 2, 3):
    run_dir = OUTPUT / "f43_relation_type_bias" / f"seed_{seed}" / "val_scene_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(TRAIN), "--project-root", str(PROJECT),
        "--variant", "f43_relation_type_bias",
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", str(seed), "--val-scene-id", "1", "--epochs", "30",
        "--val-freq", "5", "--batch-size", "64", "--device", "cuda:0",
        "--scheduler", "warm_restarts", "--output-root", str(OUTPUT), "--force",
    ]
    print(f"[run] seed={seed}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Failed seed={seed}; see {run_dir / 'console.log'}")
print("[done]", flush=True)
