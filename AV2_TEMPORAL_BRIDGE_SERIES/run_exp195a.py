"""Orchestrate matched EXP-195A A1/A2/A3 fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from exp195_constants import (
    ASTRA_PROJECT, ASTRA_PYTHON, CONFIGS, DEFAULT_ETH_CHECKPOINT,
    GENERATED_CONFIGS, PREP_PYTHON, ROOT, TRAINER, require_paths,
)


ADAPTER = ROOT / "exp195_adapter.py"
PREPARE = ROOT / "prepare_av2_ped_bridge.py"
EVALUATE = ROOT / "evaluate_exp195a.py"


def checkpoint_for(name: str) -> Path:
    return ROOT / "results" / name / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"


def train_command(args, name: str, config: Path):
    data = ROOT / "prepared"
    return [
        str(ASTRA_PYTHON), "-u", str(TRAINER),
        "--project-root", str(ASTRA_PROJECT), "--config-file", str(config),
        "--data-path", str(data / f"train_{name}_data.npy"),
        "--num-path", str(data / f"train_{name}_num.npy"),
        "--test-data-path", str(data / f"val_{name}_data.npy"),
        "--test-num-path", str(data / f"val_{name}_num.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "exp195_finetune", "--decoder-module", str(ADAPTER),
        "--seed", "1", "--val-scene-id", "-1",
        "--validation-fraction", "0.05", "--validation-seed", "2026",
        "--validation-split-mode", "stratified_random",
        "--epochs", str(args.epochs), "--val-freq", "1",
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--learning-rate", "0.00025", "--output-root", str(ROOT / "results" / name),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ] + (["--force"] if args.force else [])


def run_logged(command: list[str], log_path: Path, environment=None, dry_run=False):
    print(json.dumps(command, ensure_ascii=False), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ASTRA_PROJECT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stages", nargs="+", choices=("a1", "a2", "a3"), default=["a1", "a2", "a3"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    require_paths(ASTRA_PYTHON, TRAINER, ADAPTER, DEFAULT_ETH_CHECKPOINT)
    if args.prepare:
        run_logged(
            [str(PREP_PYTHON), str(PREPARE)] + (["--force"] if args.force else []),
            ROOT / "logs" / "prepare.log", dry_run=args.dry_run,
        )
    configs = GENERATED_CONFIGS
    prepared = ROOT / "prepared"
    required = [prepared / f"{split}_{name}_{kind}.npy" for split in ("train", "val") for name in CONFIGS for kind in ("data", "num")]
    if not args.dry_run:
        require_paths(*required)
    for name in args.stages:
        source = DEFAULT_ETH_CHECKPOINT if name == "a1" else checkpoint_for("a1")
        if not args.dry_run:
            require_paths(source)
        environment = os.environ.copy()
        environment["EXP195_SOURCE_CHECKPOINT"] = str(source)
        run_logged(
            train_command(args, name, configs[name]),
            ROOT / "logs" / f"train_{name}.log",
            environment=environment, dry_run=args.dry_run,
        )
    if not args.skip_evaluation:
        run_logged(
            [str(ASTRA_PYTHON), "-u", str(EVALUATE), "--device", args.device, "--batch-size", str(args.batch_size)],
            ROOT / "logs" / "evaluate.log", dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
