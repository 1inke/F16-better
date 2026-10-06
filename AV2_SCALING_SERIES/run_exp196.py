"""Orchestrate fair, long-history, dense-neighbor, and joint EXP-196 runs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from exp196_constants import (
    A1_CHECKPOINT, ASTRA_PROJECT, ASTRA_PYTHON, LOW_COST_VARIANTS,
    PREP_PYTHON, ROOT, STAGE_VARIANTS, TRAINER, VARIANTS,
    config_for, data_paths, require_paths,
)


ADAPTER = ROOT / "exp196_adapter.py"
MATERIALIZE = ROOT / "materialize_configs.py"
PREPARE = ROOT / "prepare_exp196.py"
EVALUATE = ROOT / "evaluate_exp196.py"
PROFILE = ROOT / "profile_exp196.py"
DEFAULT_BATCH = {"fair": 64, "long": 32, "dense": 16, "joint": 8}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses matched per-stage safe defaults")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_VARIANTS), default=["fair"])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--low-cost", action="store_true", help="Run the eight decision-critical points")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_variants(args) -> list[str]:
    if args.variants:
        names = args.variants
    elif args.low_cost:
        names = list(LOW_COST_VARIANTS)
    else:
        names = [name for stage in args.stages for name in STAGE_VARIANTS[stage]]
    return list(dict.fromkeys(names))


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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def train_command(args, name: str, seed: int):
    spec = VARIANTS[name]
    train_data, train_num = data_paths(name, "train")
    val_data, val_num = data_paths(name, "val")
    batch = args.batch_size or DEFAULT_BATCH[str(spec["stage"])]
    command = [
        str(ASTRA_PYTHON), "-u", str(TRAINER),
        "--project-root", str(ASTRA_PROJECT),
        "--config-file", str(config_for(name)),
        "--data-path", str(train_data), "--num-path", str(train_num),
        "--test-data-path", str(val_data), "--test-num-path", str(val_num),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "exp196_finetune", "--decoder-module", str(ADAPTER),
        "--seed", str(seed), "--val-scene-id", "-1",
        "--validation-fraction", "0.05", "--validation-seed", "2026",
        "--validation-split-mode", "stratified_random",
        "--epochs", str(args.epochs), "--val-freq", "1",
        "--batch-size", str(batch), "--device", args.device,
        "--max-agents", str(spec["max_agents"]),
        "--graph-top-k", str(spec["top_k"]), "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--learning-rate", "0.00025", "--output-root", str(ROOT / "results" / name),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    return command + (["--force"] if args.force else [])


def main():
    args = parse_args()
    names = selected_variants(args)
    require_paths(ASTRA_PYTHON, PREP_PYTHON, TRAINER, ADAPTER, A1_CHECKPOINT)
    run_logged([str(PREP_PYTHON), str(MATERIALIZE)], ROOT / "logs" / "materialize.log", dry_run=args.dry_run)
    if args.prepare:
        groups = sorted({str(VARIANTS[name]["data_source"]) for name in names} & {"temporal", "dense", "joint"})
        if groups:
            command = [str(PREP_PYTHON), "-u", str(PREPARE), "--groups", *groups]
            if args.force:
                command.append("--force")
            run_logged(command, ROOT / "logs" / "prepare.log", dry_run=args.dry_run)
    for name in names:
        if not args.dry_run:
            require_paths(config_for(name), *data_paths(name, "train"), *data_paths(name, "val"))
        for seed in args.seeds:
            environment = os.environ.copy()
            environment["EXP196_SOURCE_CHECKPOINT"] = str(A1_CHECKPOINT)
            run_logged(
                train_command(args, name, seed),
                ROOT / "logs" / f"train_{name}_seed{seed}.log",
                environment=environment, dry_run=args.dry_run,
            )
    common = ["--variants", *names, "--seeds", *[str(seed) for seed in args.seeds], "--device", args.device]
    if not args.skip_evaluation:
        run_logged(
            [str(ASTRA_PYTHON), "-u", str(EVALUATE), *common],
            ROOT / "logs" / "evaluate.log", dry_run=args.dry_run,
        )
    if args.profile:
        run_logged(
            [str(ASTRA_PYTHON), "-u", str(PROFILE), "--variants", *names, "--seed", str(args.seeds[0]), "--device", args.device],
            ROOT / "logs" / "profile.log", dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
