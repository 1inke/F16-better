"""Run the validation-only ETH/seed-1 H3->V2 diversity stage attribution.

This runner deliberately exposes one stage at a time.  It never passes test data
to the trainer, and it refuses to start C10/C11 unless the newly trained H3-NoDiv
checkpoint exists.  Existing C00/C01 checkpoints are referenced, not retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
TRAINER = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
H3_MODULE = REPO / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
OFFICIAL_CONFIG = REPO / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
CONFIG_005 = HERE / "configs" / "motif_rbf_official_4p8_div005.yaml"
CONFIG_000 = HERE / "configs" / "motif_rbf_official_4p8_div000.yaml"
RESULTS = HERE / "results"

EXP173 = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
H3_BASELINE_RUN = (
    EXP173 / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C00_RUN = (
    EXP173 / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C01_RUN = (
    REPO / "DIVERSITY_LOSS_ABLATION_SERIES" / "results" / "weight_sweep"
    / "divw_0p000_m_0p400" / "f16_temporal_social_cross" / "seed_1"
    / "val_scene_-1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, name))
        return out
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def diff_yaml(left: Path, right: Path) -> dict[str, tuple[Any, Any]]:
    a, b = flatten(load_yaml(left)), flatten(load_yaml(right))
    keys = sorted(set(a) | set(b))
    return {key: (a.get(key), b.get(key)) for key in keys if a.get(key) != b.get(key)}


def assert_config_contract() -> dict[str, Any]:
    official_to_005 = diff_yaml(OFFICIAL_CONFIG, CONFIG_005)
    if official_to_005:
        raise RuntimeError(f"div005 is not identical to official config: {official_to_005}")
    official_to_000 = diff_yaml(OFFICIAL_CONFIG, CONFIG_000)
    allowed = "MODEL.DECODER.DIVERSITY_WEIGHT"
    if set(official_to_000) != {allowed}:
        raise RuntimeError(
            "div000 must differ from official config only at "
            f"{allowed}; found {official_to_000}"
        )
    before, after = official_to_000[allowed]
    if float(before) != 0.05 or float(after) != 0.0:
        raise RuntimeError(f"Unexpected diversity-weight transition: {before!r} -> {after!r}")
    for path in (CONFIG_005, CONFIG_000):
        cfg = load_yaml(path)
        if float(cfg["MODEL"]["DECODER"]["DIVERSITY_MARGIN"]) != 0.4:
            raise RuntimeError(f"DIVERSITY_MARGIN is not locked to 0.4 in {path}")
    return {
        "official_sha256": sha256(OFFICIAL_CONFIG),
        "div005_sha256": sha256(CONFIG_005),
        "div000_sha256": sha256(CONFIG_000),
        "allowed_difference": allowed,
        "official_value": before,
        "div000_value": after,
    }


def run_dir(output_root: Path) -> Path:
    return output_root / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


def stage_paths(astra_root: Path) -> dict[str, Path]:
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    return {
        "data": data_dir / "eth_data_train.npy",
        "num": data_dir / "eth_num_train.npy",
        "h3_new": run_dir(RESULTS / "h3_w000") / "best_selected.pth",
        "h3_new_split": run_dir(RESULTS / "h3_w000") / "split_indices.npz",
        "h3_baseline": H3_BASELINE_RUN / "best_selected.pth",
        "c00": C00_RUN / "best_selected.pth",
        "c01": C01_RUN / "best_selected.pth",
    }


def assert_files(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} missing:\n  " + "\n  ".join(missing))


def preflight(astra_root: Path, require_new_h3: bool = False) -> dict[str, Any]:
    config_audit = assert_config_contract()
    paths = stage_paths(astra_root)
    required = [
        TRAINER, H3_MODULE, V2_MODULE, paths["data"], paths["num"],
        paths["h3_baseline"], paths["c00"], paths["c01"],
        H3_BASELINE_RUN / "split_indices.npz",
        C00_RUN / "split_indices.npz",
        C01_RUN / "split_indices.npz",
    ]
    if require_new_h3:
        required.extend([paths["h3_new"], paths["h3_new_split"]])
    assert_files(required, "Preflight inputs")

    # C01 must really have used the original H3 baseline, not an unrelated source.
    c01_spec_path = C01_RUN / "diversity_ablation_spec.json"
    assert_files([c01_spec_path], "C01 provenance")
    c01_spec = json.loads(c01_spec_path.read_text(encoding="utf-8"))
    declared_source = Path(c01_spec["h3_source_checkpoint"]).resolve()
    if sha256(declared_source) != sha256(paths["h3_baseline"]):
        raise RuntimeError("C01 H3 source does not match the EXP-173 H3 baseline")

    canonical_validation = np.load(H3_BASELINE_RUN / "split_indices.npz")["validation"]
    for label, split_path in {
        "c00": C00_RUN / "split_indices.npz",
        "c01": C01_RUN / "split_indices.npz",
    }.items():
        current = np.load(split_path)["validation"]
        if not np.array_equal(canonical_validation, current):
            raise RuntimeError(f"Existing {label} validation IDs do not match H3 baseline")
    if len(np.unique(canonical_validation)) != len(canonical_validation):
        raise RuntimeError("Canonical validation IDs contain duplicates")

    return {
        "scope": {"fold": "eth", "seed": 1, "split": "validation_only"},
        "test_policy": "test paths are intentionally not passed to new training commands",
        "validation_identity": {
            "sample_count": int(len(canonical_validation)),
            "sample_id_sha256": hashlib.sha256(canonical_validation.tobytes()).hexdigest(),
            "existing_h3_c00_c01_exact_match": True,
        },
        "config_audit": config_audit,
        "inputs": {key: str(value) for key, value in paths.items()},
        "input_hashes": {
            key: sha256(value)
            for key, value in paths.items()
            if value.is_file() and value.suffix in {".pth", ".npz"}
        },
    }


def common_command(
    astra_root: Path,
    config: Path,
    decoder_experiment: str,
    decoder_module: Path,
    epochs: int,
    output_root: Path,
) -> list[str]:
    paths = stage_paths(astra_root)
    return [
        sys.executable, "-u", str(TRAINER),
        "--project-root", str(astra_root),
        "--config-file", str(config),
        "--data-path", str(paths["data"]),
        "--num-path", str(paths["num"]),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", decoder_experiment,
        "--decoder-module", str(decoder_module),
        "--seed", "1",
        "--val-scene-id", "-1",
        "--validation-fraction", "0.05",
        "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail",
        "--epochs", str(epochs),
        "--val-freq", "5",
        "--batch-size", "64",
        "--device", "cuda:0",
        "--max-agents", "9",
        "--graph-top-k", "4",
        "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1",
        "--learning-rate", "0.00025",
        "--output-root", str(output_root),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25",
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995",
        "--focal-heading-normalize",
        "--augment-reflection",
        "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]


def command_for(stage: str, astra_root: Path) -> tuple[list[str], dict[str, str], Path]:
    paths = stage_paths(astra_root)
    env = os.environ.copy()
    if stage == "h3":
        output_root = RESULTS / "h3_w000"
        cmd = common_command(
            astra_root, CONFIG_000, "h3_anchor_proposal", H3_MODULE, 30, output_root
        )
        metadata = {"h3_weight": "0.00", "v2_weight": "not_applicable"}
    elif stage in {"c10", "c11"}:
        weight = "005" if stage == "c10" else "000"
        output_root = RESULTS / (
            "c10_h3w000_v2w005" if stage == "c10" else "c11_h3w000_v2w000"
        )
        env["EXP173_H3_SOURCE_CHECKPOINT"] = str(paths["h3_new"])
        env["EXP173_DECODER_INIT_SEED"] = "1"
        cmd = common_command(
            astra_root,
            CONFIG_005 if weight == "005" else CONFIG_000,
            "v2_joint_wide_endpoint_correction",
            V2_MODULE,
            20,
            output_root,
        )
        metadata = {
            "h3_weight": "0.00",
            "v2_weight": "0.05" if stage == "c10" else "0.00",
            "h3_source_checkpoint": str(paths["h3_new"]),
            "h3_source_sha256": sha256(paths["h3_new"]),
        }
    else:
        raise ValueError(stage)
    return cmd, env, output_root


def quote_command(command: list[str]) -> str:
    return " ".join(f'"{item}"' if any(ch.isspace() for ch in item) else item for item in command)


def execute_stage(stage: str, astra_root: Path, dry_run: bool, force: bool) -> Path:
    require_new_h3 = stage in {"c10", "c11"}
    audit = preflight(astra_root, require_new_h3=require_new_h3)
    command, env, output_root = command_for(stage, astra_root)
    destination = run_dir(output_root)
    metrics = destination / "metrics.json"
    if force:
        command.append("--force")
    elif metrics.is_file():
        print(f"[SKIP] Completed stage {stage}: {metrics}")
        return destination

    print(f"[COMMAND:{stage}] {quote_command(command)}")
    if dry_run:
        return destination

    subprocess.run(command, check=True, cwd=str(REPO), env=env)
    assert_files(
        [destination / "best_selected.pth", destination / "split_indices.npz", metrics],
        f"Completed {stage} outputs",
    )
    spec = {
        "stage": stage,
        "fold": "eth",
        "seed": 1,
        "validation_only": True,
        "test_accessed_by_this_runner": False,
        "command": command,
        "intervention": output_root.name,
        "preflight": audit,
        "checkpoint_sha256": sha256(destination / "best_selected.pth"),
        "split_indices_sha256": sha256(destination / "split_indices.npz"),
    }
    # Keep scientific metadata separate from the process environment, which may
    # contain credentials or unrelated machine-specific values.
    if stage == "h3":
        spec["weights"] = {"h3": 0.0, "v2": None}
    else:
        spec["weights"] = {"h3": 0.0, "v2": 0.05 if stage == "c10" else 0.0}
        spec["h3_source_checkpoint"] = str(stage_paths(astra_root)["h3_new"])
        spec["h3_source_sha256"] = sha256(stage_paths(astra_root)["h3_new"])
    (destination / "stage_attribution_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "h3", "c10", "c11"], required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    astra_root = args.astra_root.resolve()
    if args.stage == "preflight":
        audit = preflight(astra_root, require_new_h3=False)
        output = HERE / "preflight_audit.json"
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        print(f"[PASS] {output}")
        return
    destination = execute_stage(args.stage, astra_root, args.dry_run, args.force)
    print(f"[DONE] {args.stage}: {destination}")


if __name__ == "__main__":
    main()
