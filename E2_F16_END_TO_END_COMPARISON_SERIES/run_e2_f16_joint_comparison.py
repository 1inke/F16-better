"""Run the matched two-stage official EXP-173 E2/F16 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from evaluate_e2_f16_joint_comparison import summarize

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
OFFICIAL = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"
CONFIG = OFFICIAL / "motif_rbf_official_4p8.yaml"
AUDIT = OFFICIAL / "EXP_156A_PROTOCOL_AUDIT" / "protocol_audit.json"
H3_MODULE = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = ROOT / "joint_v2_adapter.py"
DEFAULT_EXPERIMENT = ROOT / "EXP_173_E2_F16_END_TO_END"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {"e2": "f4_relation_tokens", "e3": "f16_temporal_social_cross"}
STAGES = ("h3", "v2")


def run_dir(experiment: Path, fold: str, stage: str, label: str, seed: int) -> Path:
    return experiment / "results" / fold / stage / label / VARIANTS[label] / f"seed_{seed}" / "val_scene_-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, experiment: Path, fold: str, stage: str, label: str, seed: int) -> list[str]:
    variant = VARIANTS[label]
    if stage == "h3":
        module, decoder_experiment, epochs = H3_MODULE, "h3_anchor_proposal", args.h3_epochs
    else:
        module, decoder_experiment, epochs = V2_MODULE, "v2_joint_wide_endpoint_correction", args.v2_epochs
    cmd = [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
        "--config-file", str(CONFIG),
        "--data-path", str(DATA / f"{fold}_data_train.npy"),
        "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
        "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        "--variant", variant, "--decoder-experiment", decoder_experiment,
        "--decoder-module", str(module), "--seed", str(seed),
        "--val-scene-id", "-1", "--validation-fraction", "0.05",
        "--validation-seed", "20260830", "--validation-split-mode", "scene_tail",
        "--epochs", str(epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--learning-rate", "0.00025", "--output-root", str(experiment / "results" / fold / stage / label),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25", "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995", "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def write_locked_config(destination: Path, args, fold: str, stage: str, label: str, seed: int, source: Path | None, cmd: list[str]) -> None:
    payload = {"experiment": "EXP-173", "fold": fold, "stage": stage, "encoder_label": label, "encoder_variant": VARIANTS[label], "seed": seed, "training_policy": "Stage A joint encoder+H3; Stage B joint encoder+H3+score+V2", "source_checkpoint": str(source) if source else None, "source_sha256": sha256(source) if source and source.is_file() else None, "decoder_initialization_seed": seed if stage == "v2" else None, "command": cmd}
    (destination / "locked_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stage(args, experiment: Path, fold: str, stage: str, label: str, seed: int) -> None:
    destination = run_dir(experiment, fold, stage, label, seed)
    if (destination / "metrics.json").is_file() and not args.force:
        print(f"[skip] {label}/{fold}/{stage}/seed{seed}", flush=True)
        return
    source = run_dir(experiment, fold, "h3", label, seed) / "best_selected.pth" if stage == "v2" else None
    if source is not None and not source.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Missing matched H3 source: {source}")
    cmd = command(args, experiment, fold, stage, label, seed)
    if args.dry_run:
        print(f"[dry-run] {label}/{fold}/{stage}/seed{seed} " + json.dumps(cmd), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if stage == "v2":
        environment["EXP173_H3_SOURCE_CHECKPOINT"] = str(source)
        environment["EXP173_DECODER_INIT_SEED"] = str(seed)
    print(f"[run] {label}/{fold}/{stage}/seed{seed}", flush=True)
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=PROJECT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}/{fold}/{stage}] {line}", end="", flush=True)
            log.write(line); log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Failed: {destination / 'console.log'}")
    write_locked_config(destination, args, fold, stage, label, seed, source, cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--variants", nargs="+", choices=("all", *VARIANTS), default=["e2", "e3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    labels = tuple(VARIANTS) if "all" in args.variants else tuple(args.variants)
    stages = STAGES if "all" in args.stages else tuple(args.stages)
    seeds = tuple(args.seeds)
    required = (PYTHON, TRAINER, CONFIG, AUDIT, H3_MODULE, V2_MODULE)
    missing = [str(path) for path in required if not path.is_file()]
    missing += [str(DATA / f"{fold}_{kind}.npy") for fold in folds for kind in ("data_train", "num_train", "data_test", "num_test") if not (DATA / f"{fold}_{kind}.npy").is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if not json.loads(AUDIT.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Official protocol audit has not passed")
    for label in labels:
        for fold in folds:
            for seed in seeds:
                for stage in stages:
                    run_stage(args, args.experiment_dir, fold, stage, label, seed)
                    if not args.dry_run:
                        summarize(args.experiment_dir, seeds)
    if args.dry_run:
        print("[done] dry-run passed", flush=True)
    else:
        payload = summarize(args.experiment_dir, seeds)
        print(f"[done] available={payload['available_stage_runs']}/{payload['expected_stage_runs']} decision={payload['decision']}", flush=True)


if __name__ == "__main__":
    main()

