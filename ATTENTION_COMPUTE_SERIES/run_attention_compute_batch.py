from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS"
    / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
RUNS = [
    ("EXP_043_TEMPORAL_LANDMARK4", "f28_temporal_landmark4"),
    ("EXP_044_TEMPORAL_LANDMARK2", "f29_temporal_landmark2"),
    ("EXP_045_GRAPH_ADDITIVE_ATTENTION", "f30_graph_additive_attention"),
    ("EXP_046_CASCADED_GROUP_ATTENTION", "f31_cascaded_group_attention"),
    ("EXP_047_GROUPED_QUERY_ATTENTION", "f32_grouped_query_attention"),
    ("EXP_048_MULTI_QUERY_ATTENTION", "f33_multi_query_attention"),
    ("EXP_049_IDENTITY_VALUE_ATTENTION", "f34_identity_value_attention"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-scene", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for folder, variant in RUNS:
        experiment = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / folder
        output = experiment / "results"
        run_dir = output / variant / f"seed_{args.seed}" / f"val_scene_{args.val_scene}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(TRAIN), "--project-root", str(PROJECT),
            "--variant", variant, "--decoder-experiment", "kinematic_k20",
            "--decoder-module", str(DECODER), "--seed", str(args.seed),
            "--val-scene-id", str(args.val_scene), "--epochs", str(args.epochs),
            "--device", args.device, "--output-root", str(output),
        ]
        if args.force:
            command.append("--force")
        print(f"[run] {variant}", flush=True)
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, text=True
            )
        if completed.returncode:
            raise RuntimeError(f"failed: {variant}; see {run_dir / 'console.log'}")
        print(f"[done] {variant}", flush=True)


if __name__ == "__main__":
    main()
