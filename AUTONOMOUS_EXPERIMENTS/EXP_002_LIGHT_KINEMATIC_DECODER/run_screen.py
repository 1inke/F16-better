"""Run the seed-1 decoder screen and create a compact summary."""

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT.parents[1] / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
decoders = ["q20", "kinematic_k12", "kinematic_k20"]

for decoder in decoders:
    output = ROOT / "results" / decoder
    command = [
        str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
        "--variant", "f4_relation_tokens", "--decoder-experiment", decoder,
        "--seed", "1", "--val-scene-id", "1", "--epochs", "30",
        "--device", "cuda:0", "--output-root", str(output),
    ]
    if decoder != "q20":
        command += ["--decoder-module", str(ROOT / "decoder_models.py")]
    run_dir = output / "f4_relation_tokens" / "seed_1" / "val_scene_1"
    metrics = run_dir / "metrics.json"
    if metrics.is_file():
        print(f"[skip] {decoder}")
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {decoder}")
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{decoder} failed; see {run_dir / 'console.log'}")

rows = []
for decoder in decoders:
    path = ROOT / "results" / decoder / "f4_relation_tokens" / "seed_1" / "val_scene_1" / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    ev = data["evaluation"]
    rows.append({
        "decoder": decoder,
        "total_parameters": data["locked_backend"]["parameter_counts"]["total"],
        "decoder_parameters": data["locked_backend"]["parameter_counts"]["decoder"],
        "original_ade": ev["original"]["ade"], "original_fde": ev["original"]["fde"],
        "fused_ade": ev["fused"]["ade"], "fused_fde": ev["fused"]["fde"],
        "min_ade_k": ev["oracle"]["min_ade_20"],
        "min_fde_k": ev["oracle"]["min_fde_20"],
        "hit": ev["original"]["hit_best_ade"],
    })
with (ROOT / "SCREEN_SUMMARY.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(ROOT / "SCREEN_SUMMARY.csv")
