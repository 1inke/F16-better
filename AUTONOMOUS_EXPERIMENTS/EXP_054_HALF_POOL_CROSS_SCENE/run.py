from __future__ import annotations

import subprocess
from pathlib import Path

W = Path(__file__).resolve().parents[2]
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN = W / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
OUTPUT = Path(__file__).resolve().parent / "results"
VARIANT = "f35_temporal_half_pool"

for scene in (0, 2, 3, 4):
    run_dir = OUTPUT / VARIANT / "seed_1" / f"val_scene_{scene}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PY), str(TRAIN), "--project-root", str(P), "--variant", VARIANT,
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(DECODER),
        "--seed", "1", "--val-scene-id", str(scene), "--epochs", "30",
        "--device", "cuda:0", "--output-root", str(OUTPUT), "--force"]
    print(f"[run] scene={scene}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"scene {scene} failed; see {run_dir / 'console.log'}")
    print(f"[done] scene={scene}", flush=True)
