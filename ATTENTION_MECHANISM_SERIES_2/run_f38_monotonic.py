from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
OUTPUT = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_061_F38_MONOTONIC_COSINE" / "results"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--val-scenes", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    variant = "f38_recent_exact_early_pool"
    for seed in args.seeds:
        for scene in args.val_scenes:
            run_dir = OUTPUT / variant / f"seed_{seed}" / f"val_scene_{scene}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON), str(TRAIN), "--project-root", str(PROJECT),
                "--variant", variant, "--decoder-experiment", "kinematic_k20",
                "--decoder-module", str(DECODER), "--seed", str(seed),
                "--val-scene-id", str(scene), "--epochs", str(args.epochs),
                "--device", args.device, "--output-root", str(OUTPUT),
                "--scheduler", "monotonic_cosine", "--warmup-epochs", "1.0",
            ]
            if args.force:
                command.append("--force")
            print(f"[run] seed={seed} scene={scene}", flush=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, text=True
                )
            if completed.returncode:
                raise RuntimeError(f"Failed; see {run_dir / 'console.log'}")
            print(f"[done] seed={seed} scene={scene}", flush=True)


if __name__ == "__main__":
    main()

