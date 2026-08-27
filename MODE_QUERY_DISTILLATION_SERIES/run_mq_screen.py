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
DECODER = ROOT / "mode_query_decoder_models.py"
EXPERIMENTS = {
    "mq1": ("EXP_121_MQ1_GEOMETRY_QUERY", "mq1_geometry_k20"),
    "mq2": ("EXP_122_MQ2_TEMPORAL_QUERY", "mq2_temporal_k20"),
}


def run_one(label: str) -> None:
    folder, decoder_name = EXPERIMENTS[label]
    experiment = ROOT / folder
    output = experiment / "results"
    rows = []
    for scene in (1, 3):
        run_dir = output / "f16_temporal_social_cross" / "seed_1" / f"val_scene_{scene}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            command = [
                str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
                "--variant", "f16_temporal_social_cross",
                "--decoder-experiment", decoder_name, "--decoder-module", str(DECODER),
                "--seed", "1", "--val-scene-id", str(scene), "--epochs", "30",
                "--val-freq", "5", "--device", "cuda:0", "--scheduler", "warm_restarts",
                "--output-root", str(output), "--oracle-coverage-weight", "0.20",
                "--oracle-coverage-temperature", "0.10", "--checkpoint-criterion",
                "oracle_combo", "--ema-decay", "0.995", "--focal-heading-normalize",
                "--augment-reflection", "--augment-speed-min", "0.9",
                "--augment-speed-max", "1.1", "--augment-probability", "0.5", "--force",
            ]
            print(f"[run] {label} scene={scene}", flush=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(run_dir / "console.log")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected = metrics["evaluation_best_selected"]
        rows.append({
            "label": label, "decoder": decoder_name, "seed": 1, "scene": scene,
            "samples": selected["samples"], **selected["oracle"],
            "top1_ade": selected["original"]["ade"],
            "top1_fde": selected["original"]["fde"],
            "fused_ade": selected["fused"]["ade"],
            "fused_fde": selected["fused"]["fde"],
        })
        (experiment / "screen_summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    print(f"[done] {label}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=tuple(EXPERIMENTS))
    run_one(parser.parse_args().label)
