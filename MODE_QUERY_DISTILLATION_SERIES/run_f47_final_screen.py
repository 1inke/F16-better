from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXPERIMENT = ROOT / "EXP_120_F47_FINAL_PROTOCOL"
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
VARIANT = "f47_multiscale_temporal_replace"


def run() -> None:
    rows = []
    output = EXPERIMENT / "results"
    for scene in (1, 3):
        run_dir = output / VARIANT / "seed_1" / f"val_scene_{scene}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            command = [
                str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
                "--variant", VARIANT, "--decoder-experiment", "kinematic_k20",
                "--decoder-module", str(DECODER), "--seed", "1",
                "--val-scene-id", str(scene), "--epochs", "30", "--val-freq", "5",
                "--device", "cuda:0", "--scheduler", "warm_restarts",
                "--output-root", str(output), "--oracle-coverage-weight", "0.20",
                "--oracle-coverage-temperature", "0.10", "--checkpoint-criterion",
                "oracle_combo", "--ema-decay", "0.995", "--focal-heading-normalize",
                "--augment-reflection", "--augment-speed-min", "0.9",
                "--augment-speed-max", "1.1", "--augment-probability", "0.5", "--force",
            ]
            print(f"[run] F47 final protocol scene={scene}", flush=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(run_dir / "console.log")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = metrics["evaluation_best_selected"]
        rows.append({
            "variant": VARIANT, "seed": 1, "scene": scene,
            "samples": selected["samples"], **selected["oracle"],
            "top1_ade": selected["original"]["ade"],
            "top1_fde": selected["original"]["fde"],
            "fused_ade": selected["fused"]["ade"],
            "fused_fde": selected["fused"]["fde"],
        })
        (EXPERIMENT / "screen_summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    print("[done] F47 final protocol", flush=True)


if __name__ == "__main__":
    run()
