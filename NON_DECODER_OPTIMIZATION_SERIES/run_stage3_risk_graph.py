from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY = Path(r"C:\coding_toobox\envs\astra2\python.exe")
T = W / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
D = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
JOBS = [("EXP_084_TTC_DCA_GRAPH", "f48_ttc_dca_graph"),
        ("EXP_085_TTC_DCA_LIGHT_TEMPORAL", "f49_ttc_dca_multiscale_replace")]
rows = []
for name, variant in JOBS:
    output = ROOT / name / "results"
    run_dir = output / variant / "seed_1" / "val_scene_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PY), str(T), "--project-root", str(P), "--variant", variant,
        "--decoder-experiment", "kinematic_k20", "--decoder-module", str(D),
        "--seed", "1", "--val-scene-id", "1", "--epochs", "30", "--val-freq", "5",
        "--device", "cuda:0", "--scheduler", "warm_restarts", "--output-root", str(output),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--augment-rotation", "--augment-reflection", "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1", "--force"]
    print(f"[run] {name}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=P, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(run_dir / "console.log")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    evaluation = metrics["evaluation_best_selected"]
    rows.append({"experiment": name, "variant": variant, **evaluation["oracle"],
                 "top1_ade": evaluation["original"]["ade"], "top1_fde": evaluation["original"]["fde"],
                 "parameters": metrics["locked_backend"]["parameter_counts"]})
    (ROOT / "stage3_risk_graph_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("[done] risk graph", flush=True)
