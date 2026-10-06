"""Guarded RGDM 2x2 factorial runner. It never passes a test-set path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = HERE / "rgdm_models.py"
ANALYZER = HERE / "analyze_rgdm_factorial.py"
EXPERIMENT_SPEC = HERE / "experiment_config.json"
CONFIG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
DATA = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_data_train.npy"
NUM = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_num_train.npy"
C00_DIR = (
    ROOT / "QUALITY_AWARE_F16_SERIES_V2" / "results" / "p0_protocol"
    / "C0_EXP173_EXACT" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)
C00 = C00_DIR / "best_selected.pth"
OUTPUT_ROOT = HERE / "results" / "training"
PREFLIGHT = HERE / "preflight.json"
EXPECTED_TRAINABLE = {
    "rgdm_b4_full": 44744,
    "rgdm_g4_generation": 23226,
    "rgdm_ab_joint": 67970,
}


def load_spec() -> dict:
    return json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(condition: str, variant: str) -> Path:
    # train_front_variant names the run directory by encoder variant, while
    # decoder_experiment is recorded inside metrics/contract metadata.
    return OUTPUT_ROOT / condition / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


def expected_validation_ids(spec: dict) -> np.ndarray:
    raw = np.load(DATA, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    parts = []
    for scene in np.unique(scene_ids):
        candidates = np.flatnonzero(scene_ids == scene)
        count = max(1, min(len(candidates) - 1, int(round(len(candidates) * spec["validation_fraction"]))))
        parts.append(candidates[-count:])
    return np.sort(np.concatenate(parts))


def validate_completed_run(condition: str, variant: str, spec: dict) -> Path:
    """Reject stale or protocol-incompatible artifacts before they are skipped."""
    run = run_dir(condition, variant)
    required = [
        run / "metrics.json", run / "split_indices.npz", run / "locked_backend_contract.json",
        run / "best_original_ade.pth", run / "best_selected.pth", run / "final_model.pth",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete completed run for {condition}:\n" + "\n".join(missing))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    checks = {
        "variant": (metrics.get("variant"), "f16_temporal_social_cross"),
        "decoder_experiment": (metrics.get("decoder_experiment"), variant),
        "seed": (metrics.get("seed"), spec["generator_seed"]),
        "epochs": (metrics.get("epochs"), spec["epochs"]),
        "scheduler": (metrics.get("scheduler"), spec["scheduler"]),
    }
    failures = [f"{name}: {actual!r} != {expected!r}" for name, (actual, expected) in checks.items()
                if actual != expected]
    split = metrics.get("split", {})
    split_checks = {
        "strategy": (split.get("strategy"), f"{spec['validation_split_mode']}_holdout_within_training_scenes"),
        "validation_fraction": (split.get("validation_fraction"), spec["validation_fraction"]),
        "validation_seed": (split.get("validation_seed"), spec["validation_seed"]),
        "validation_samples": (split.get("validation_samples"), len(expected_validation_ids(spec))),
        "target_test_accessed": (split.get("target_test_accessed"), False),
    }
    failures += [f"split.{name}: {actual!r} != {expected!r}"
                 for name, (actual, expected) in split_checks.items() if actual != expected]
    for field, locked_path in (("data_path", DATA), ("num_path", NUM)):
        recorded = Path(split.get(field, ""))
        if not recorded.is_file():
            failures.append(f"split.{field} is missing: {recorded}")
        elif sha256(recorded) != sha256(locked_path):
            failures.append(f"split.{field} content differs from the locked input")
    contract = metrics.get("locked_backend", {})
    contract_checks = {
        "decoder_experiment": (contract.get("decoder_experiment"), variant),
        "oracle_coverage_weight": (contract.get("oracle_coverage_weight"), spec["oracle_coverage_weight"]),
        "oracle_coverage_temperature": (
            contract.get("oracle_coverage_temperature"), spec["oracle_coverage_temperature"]
        ),
        "oracle_coverage_fde_weight": (
            contract.get("oracle_coverage_fde_weight"), spec["oracle_coverage_fde_weight"]
        ),
        "ema_decay": (contract.get("ema_decay"), spec["ema_decay"]),
        "checkpoint_criterion": (contract.get("checkpoint_criterion"), "oracle_combo"),
        "trainable_parameters": (
            contract.get("parameter_counts", {}).get("trainable"), EXPECTED_TRAINABLE[variant]
        ),
        "focal_heading_normalize": (contract.get("focal_heading_normalize"), True),
    }
    failures += [f"contract.{name}: {actual!r} != {expected!r}"
                 for name, (actual, expected) in contract_checks.items() if actual != expected]
    expected_augmentation = {
        "rotation": False, "reflection": True, "speed_min": 0.9,
        "speed_max": 1.1, "probability": 0.5, "curvature_max": 0.0,
    }
    if contract.get("augmentation") != expected_augmentation:
        failures.append(
            f"contract.augmentation: {contract.get('augmentation')!r} != {expected_augmentation!r}"
        )
    if metrics.get("evaluation_test_best_selected") is not None:
        failures.append("evaluation_test_best_selected must be null")
    for key in ("evaluation", "evaluation_best_selected"):
        if metrics.get(key, {}).get("samples") != len(expected_validation_ids(spec)):
            failures.append(f"{key}.samples is not the locked validation count")
    with np.load(run / "split_indices.npz", allow_pickle=False) as payload:
        actual_ids = np.asarray(payload["validation"], dtype=np.int64)
    with np.load(C00_DIR / "split_indices.npz", allow_pickle=False) as payload:
        reference_ids = np.asarray(payload["validation"], dtype=np.int64)
    if not np.array_equal(actual_ids, reference_ids):
        failures.append("validation IDs differ from C00")
    if failures:
        raise RuntimeError(f"Completed run contract failed for {condition}:\n" + "\n".join(failures))
    return run


def preflight(spec: dict) -> dict:
    required = [PYTHON, TRAINER, DECODER, ANALYZER, EXPERIMENT_SPEC, CONFIG, DATA, NUM, C00]
    required += [C00_DIR / name for name in (
        "best_original_ade.pth", "best_selected.pth", "final_model.pth",
        "split_indices.npz", "metrics.json",
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    with np.load(C00_DIR / "split_indices.npz", allow_pickle=False) as payload:
        locked_validation = np.asarray(payload["validation"], dtype=np.int64)
    expected = expected_validation_ids(spec)
    if not np.array_equal(locked_validation, expected):
        raise RuntimeError("C00 validation IDs do not match the declared scene-tail split")
    baseline_metrics = json.loads((C00_DIR / "metrics.json").read_text(encoding="utf-8"))
    if int(baseline_metrics["evaluation_best_selected"]["samples"]) != len(expected):
        raise RuntimeError("C00 metric sample count disagrees with split IDs")
    if spec["gate_loss_weight"] != 0.0 or spec["rank_loss_weight"] != 0.0:
        raise RuntimeError("The first factorial must isolate structure; auxiliary losses must remain zero")
    manifest = {
        "protocol": spec["protocol"],
        "status": "PREFLIGHT_PASS_NO_TRAINING",
        "validation_samples": int(len(expected)),
        "validation_id_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "sources": {
            str(path): sha256(path) for path in (
                Path(__file__).resolve(), TRAINER, DECODER, ANALYZER,
                EXPERIMENT_SPEC, CONFIG, C00, DATA, NUM,
            )
        },
        "conditions": spec["conditions"],
        "test_paths_in_training_commands": False,
        "test_access_authorized": False,
        "preflight_action_started_training": False,
    }
    if PREFLIGHT.exists():
        old = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Preflight/source drift detected; review before replacing the lock")
    else:
        PREFLIGHT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for condition, variant in spec["conditions"].items():
        if (run_dir(condition, variant) / "metrics.json").is_file():
            validate_completed_run(condition, variant, spec)
            print(f"[verified existing] {condition}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def command(condition: str, variant: str, spec: dict, device: str, force: bool) -> List[str]:
    output_root = OUTPUT_ROOT / condition
    args = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(ASTRA),
        "--config-file", str(CONFIG),
        "--data-path", str(DATA),
        "--num-path", str(NUM),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", variant,
        "--decoder-module", str(DECODER),
        "--seed", str(spec["generator_seed"]),
        "--val-scene-id", "-1",
        "--validation-fraction", str(spec["validation_fraction"]),
        "--validation-seed", str(spec["validation_seed"]),
        "--validation-split-mode", spec["validation_split_mode"],
        "--epochs", str(spec["epochs"]),
        "--val-freq", str(spec["validation_frequency"]),
        "--batch-size", str(spec["batch_size"]),
        "--learning-rate", str(spec["learning_rate"]),
        "--device", device,
        "--max-agents", "9",
        "--graph-top-k", "4",
        "--graph-radius", "6.0",
        "--scheduler", spec["scheduler"],
        "--warmup-epochs", str(spec["warmup_epochs"]),
        "--output-root", str(output_root),
        "--oracle-coverage-weight", str(spec["oracle_coverage_weight"]),
        "--oracle-coverage-temperature", str(spec["oracle_coverage_temperature"]),
        "--oracle-coverage-fde-weight", str(spec["oracle_coverage_fde_weight"]),
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", str(spec["ema_decay"]),
        "--focal-heading-normalize",
        "--augment-reflection",
        "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if force:
        args.append("--force")
    forbidden = {"--test-data-path", "--test-num-path", "--test-sample-scene-path"}
    if any(item in forbidden for item in args):
        raise RuntimeError("Test path leaked into an RGDM command")
    return args


def train(spec: dict, device: str, dry_run: bool, force: bool) -> None:
    preflight(spec)
    environment = os.environ.copy()
    environment["RGDM_SOURCE_CHECKPOINT"] = str(C00)
    environment["RGDM_GATE_LOSS_WEIGHT"] = str(spec["gate_loss_weight"])
    environment["RGDM_RANK_LOSS_WEIGHT"] = str(spec["rank_loss_weight"])
    for condition, variant in spec["conditions"].items():
        metrics = run_dir(condition, variant) / "metrics.json"
        args = command(condition, variant, spec, device, force)
        if metrics.is_file() and not force:
            validate_completed_run(condition, variant, spec)
            print(f"[skip verified complete] {metrics}")
            continue
        if dry_run:
            print(json.dumps({"condition": condition, "command": args}, ensure_ascii=False, indent=2))
            continue
        print(f"[train] {condition}: {variant}", flush=True)
        subprocess.run(args, cwd=ROOT, env=environment, check=True)
        validate_completed_run(condition, variant, spec)


def status(spec: dict) -> None:
    paths: Dict[str, Path] = {"c00_g0_s0": C00_DIR / "metrics.json"}
    for condition, variant in spec["conditions"].items():
        paths[condition] = run_dir(condition, variant) / "metrics.json"
    paths["analysis"] = HERE / "results" / "analysis" / "verdict.json"
    for name, path in paths.items():
        if name in spec["conditions"] and path.is_file():
            validate_completed_run(name, spec["conditions"][name], spec)
        state = (
            "ready-verified" if name in spec["conditions"] and path.is_file()
            else "ready" if path.is_file()
            else "pending"
        )
        print(f"[{state}] {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "status"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    spec = load_spec()
    if args.stage == "preflight":
        preflight(spec)
    elif args.stage == "train":
        train(spec, args.device, args.dry_run, args.force)
    else:
        status(spec)


if __name__ == "__main__":
    main()
