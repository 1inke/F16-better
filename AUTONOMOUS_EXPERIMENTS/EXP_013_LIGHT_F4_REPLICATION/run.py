import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
SOURCE = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_012_LIGHT_F4_FFN" / "results" / "f6_light_factorized" / "seed_1" / "val_scene_1"
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
OUTPUT = ROOT / "results"
target = OUTPUT / "f6_light_factorized" / "seed_1" / "val_scene_1"
if not (target / "metrics.json").is_file():
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, target, dirs_exist_ok=True)
for seed in (2, 3):
    run_dir = OUTPUT / "f6_light_factorized" / f"seed_{seed}" / "val_scene_1"
    if (run_dir / "metrics.json").is_file():
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
        "--variant", "f6_light_factorized", "--decoder-experiment", "kinematic_k20",
        "--decoder-module", str(DECODER), "--seed", str(seed),
        "--val-scene-id", "1", "--epochs", "30", "--device", "cuda:0",
        "--output-root", str(OUTPUT),
    ]
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"seed {seed} failed; see {run_dir / 'console.log'}")
print("[done]")
