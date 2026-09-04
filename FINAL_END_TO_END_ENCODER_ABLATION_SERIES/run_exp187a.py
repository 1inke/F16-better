"""Run matched two-stage EXP-187A E0/E1 end-to-end ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from evaluate_exp187a import summarize

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
AUDIT = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156A_PROTOCOL_AUDIT" / "protocol_audit.json"
H3_MODULE = ROOT / "exp187a_h3_adapter.py"
V2_MODULE = WORKSPACE / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
EXPERIMENT = ROOT / "EXP_187A_END_TO_END_ENCODER_ABLATION"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {"e0": "f0_baseline", "e1": "f3_factorized"}
STAGES = ("h3", "v2")


def run_dir(experiment: Path, fold: str, stage: str, label: str, seed: int) -> Path:
    return experiment / "results" / fold / stage / label / VARIANTS[label] / f"seed_{seed}" / "val_scene_-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_command(args, experiment: Path, fold: str, stage: str, label: str, seed: int) -> list[str]:
    if stage == "h3":
        module, decoder_name, epochs = H3_MODULE, "h3_anchor_proposal", args.h3_epochs
    else:
        module, decoder_name, epochs = V2_MODULE, "v2_joint_wide_endpoint_correction", args.v2_epochs
    command = [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
        "--config-file", str(CONFIG),
        "--data-path", str(DATA / f"{fold}_data_train.npy"),
        "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
        "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        "--variant", VARIANTS[label], "--decoder-experiment", decoder_name,
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
        command.append("--force")
    return command


def run_stage(args, experiment: Path, fold: str, stage: str, label: str, seed: int) -> None:
    destination = run_dir(experiment, fold, stage, label, seed)
    if (destination / "metrics.json").is_file() and not args.force:
        print(f"[skip] {label}/{fold}/{stage}/seed{seed}", flush=True)
        return
    source = run_dir(experiment, fold, "h3", label, seed) / "best_selected.pth" if stage == "v2" else None
    if source is not None and not source.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Missing matched Stage-A checkpoint: {source}")
    command = make_command(args, experiment, fold, stage, label, seed)
    if args.dry_run:
        print(f"[dry-run] {label}/{fold}/{stage}/seed{seed} " + json.dumps(command, ensure_ascii=False), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["EXP187A_H3_INIT_SEED"] = str(seed + 187000)
    if stage == "v2":
        environment["EXP173_H3_SOURCE_CHECKPOINT"] = str(source)
        environment["EXP173_DECODER_INIT_SEED"] = str(seed)
    print(f"[run] {label}/{fold}/{stage}/seed{seed}", flush=True)
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}/{fold}/{stage}] {line}", end="", flush=True)
            log.write(line)
            log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Failed: {destination / 'console.log'}")
    locked = {"experiment": "EXP-187A", "fold": fold, "stage": stage, "encoder": label, "variant": VARIANTS[label], "seed": seed, "protocol": "EXP-173 optimization with encoder-independent matched H3 and fresh Joint-V2 initialization", "h3_initialization_seed": seed + 187000, "v2_initialization_seed": seed, "source_checkpoint": str(source) if source else None, "source_sha256": sha256(source) if source else None, "command": command}
    (destination / "locked_config.json").write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--variants", nargs="+", choices=("all", *VARIANTS), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
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
                    run_stage(args, experiment, fold, stage, label, seed)
                    if not args.dry_run:
                        summarize(experiment, seeds)
    if args.dry_run:
        print("[done] EXP-187A dry-run passed", flush=True)
    else:
        result = summarize(experiment, seeds)
        print(json.dumps({"complete": result["complete"], "available": result["available_stage_runs"], "decision": result["decision"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
