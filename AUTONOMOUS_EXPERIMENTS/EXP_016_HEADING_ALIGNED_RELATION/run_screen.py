import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
OUTPUT = ROOT / "results"
run_dir = OUTPUT / "f8_heading_aligned_relation" / "seed_1" / "val_scene_1"
if not (run_dir / "metrics.json").is_file():
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
        "--variant", "f8_heading_aligned_relation", "--decoder-experiment", "kinematic_k20",
        "--decoder-module", str(DECODER), "--seed", "1", "--val-scene-id", "1",
        "--epochs", "30", "--device", "cuda:0", "--output-root", str(OUTPUT)]
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"failed; see {run_dir / 'console.log'}")
print("[done]")
