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
    ("EXP_080_AUGMENT_NO_BALANCE", ["--ema-decay", "0.995", "--augment-rotation",
     "--augment-reflection", "--augment-speed-min", "0.9", "--augment-speed-max", "1.1"]),
    ("EXP_081_KINEMATIC_NO_BALANCE", ["--ema-decay", "0.995",
     "--velocity-loss-weight", "0.10", "--acceleration-loss-weight", "0.02"]),
]
rows = []
for name, extra in JOBS:
    output = ROOT / name / "results"
    run_dir = output / "f16_temporal_social_cross" / "seed_1" / "val_scene_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
        "--variant", "f16_temporal_social_cross", "--decoder-experiment", "kinematic_k20",
        "--decoder-module", str(DECODER), "--seed", "1", "--val-scene-id", "1",
        "--epochs", "30", "--val-freq", "5", "--device", "cuda:0", "--scheduler", "warm_restarts",
        "--output-root", str(output), "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10", "--checkpoint-criterion", "oracle_combo", "--force", *extra]
    print(f"[run] {name}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(run_dir / "console.log")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    evaluation = metrics["evaluation_best_selected"]
    rows.append({"experiment": name, **evaluation["oracle"],
                 "top1_ade": evaluation["original"]["ade"],
                 "top1_fde": evaluation["original"]["fde"]})
    (ROOT / "stage1_isolation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("[done] isolation", flush=True)
