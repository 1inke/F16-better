import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
OUTPUT = ROOT / "results" / "monotonic_cosine"
run_dir = OUTPUT / "f4_relation_tokens" / "seed_1" / "val_scene_1"
if not (run_dir / "metrics.json").is_file():
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(TRAINER), "--project-root", str(PROJECT),
        "--variant", "f4_relation_tokens", "--decoder-experiment", "kinematic_k20",
        "--decoder-module", str(DECODER), "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1", "--seed", "1", "--val-scene-id", "1",
        "--epochs", "30", "--device", "cuda:0", "--output-root", str(OUTPUT),
    ]
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"training failed; see {run_dir / 'console.log'}")
print("[done]")
