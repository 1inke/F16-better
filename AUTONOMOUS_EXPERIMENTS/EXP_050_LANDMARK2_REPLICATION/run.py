from __future__ import annotations

import subprocess
from pathlib import Path


W = Path(__file__).resolve().parents[2]
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN = W / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
OUTPUT = Path(__file__).resolve().parent / "results"
VARIANT = "f29_temporal_landmark2"

for seed in (2, 3):
    run_dir = OUTPUT / VARIANT / f"seed_{seed}" / "val_scene_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), str(TRAIN), "--project-root", str(P), "--variant", VARIANT,
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", str(seed), "--val-scene-id", "1", "--epochs", "30",
        "--device", "cuda:0", "--output-root", str(OUTPUT), "--force",
    ]
    print(f"[run] seed={seed}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"seed {seed} failed; see {run_dir / 'console.log'}")
    print(f"[done] seed={seed}", flush=True)
