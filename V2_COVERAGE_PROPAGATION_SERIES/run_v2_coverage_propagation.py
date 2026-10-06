"""Run the validation-only H3 coverage -> V2 propagation experiment.

Only one new model is trained.  The existing EXP-173 C00 checkpoint is the
coverage-on reference (H3 O=.20 -> V2 O=.20).  The new condition starts V2
from the completed H3 D=.05/O=0 checkpoint while keeping the entire V2
training contract, including V2 oracle coverage=.20, unchanged.
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


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES"))

import run_stage_attribution as stage_runner


ASTRA_DEFAULT = stage_runner.ASTRA_DEFAULT
RESULTS = HERE / "results"
NEW_OUTPUT_ROOT = RESULTS / "v2_from_h3_coverage_off"
NEW_RUN = stage_runner.run_dir(NEW_OUTPUT_ROOT)

H3_ON_RUN = stage_runner.H3_BASELINE_RUN
V2_ON_RUN = stage_runner.C00_RUN
H3_OFF_RUN = (
    REPO / "H3_OBJECTIVE_ATTRIBUTION_SERIES" / "results" / "d1_o0"
    / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def require_files(paths: list[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} missing:\n  " + "\n  ".join(missing))


def validation_ids(run: Path) -> np.ndarray:
    values = np.load(run / "split_indices.npz")["validation"].astype(np.int64)
    if len(np.unique(values)) != len(values):
        raise RuntimeError(f"Duplicate validation IDs in {run}")
    return values


def preflight(astra_root: Path, require_new_v2: bool = False) -> dict[str, Any]:
    inherited = stage_runner.preflight(astra_root, require_new_h3=False)
    required = [
        H3_ON_RUN / "best_selected.pth",
        H3_ON_RUN / "final_model.pth",
        H3_ON_RUN / "split_indices.npz",
        H3_ON_RUN / "metrics.json",
        V2_ON_RUN / "best_selected.pth",
        V2_ON_RUN / "final_model.pth",
        V2_ON_RUN / "split_indices.npz",
        V2_ON_RUN / "metrics.json",
        V2_ON_RUN / "locked_config.json",
        V2_ON_RUN / "locked_backend_contract.json",
        H3_OFF_RUN / "best_selected.pth",
        H3_OFF_RUN / "final_model.pth",
        H3_OFF_RUN / "split_indices.npz",
        H3_OFF_RUN / "metrics.json",
        H3_OFF_RUN / "objective_attribution_spec.json",
        stage_runner.V2_MODULE,
        stage_runner.CONFIG_005,
    ]
    if require_new_v2:
        required.extend([
            NEW_RUN / "best_selected.pth",
            NEW_RUN / "final_model.pth",
            NEW_RUN / "split_indices.npz",
            NEW_RUN / "metrics.json",
        ])
    require_files(required, "Coverage propagation inputs")

    ids = validation_ids(H3_ON_RUN)
    for label, run in {
        "v2_on": V2_ON_RUN,
        "h3_off": H3_OFF_RUN,
        **({"v2_off": NEW_RUN} if require_new_v2 else {}),
    }.items():
        current = validation_ids(run)
        if not np.array_equal(ids, current):
            raise RuntimeError(f"Validation IDs differ for {label}")

    h3_on_metrics = read_json(H3_ON_RUN / "metrics.json")
    h3_off_metrics = read_json(H3_OFF_RUN / "metrics.json")
    v2_on_backend = read_json(V2_ON_RUN / "locked_backend_contract.json")
    v2_on_config = read_json(V2_ON_RUN / "locked_config.json")
    h3_off_spec = read_json(H3_OFF_RUN / "objective_attribution_spec.json")

    if float(h3_on_metrics["locked_backend"]["oracle_coverage_weight"]) != 0.20:
        raise RuntimeError("H3 coverage-on source is not O=.20")
    if float(h3_off_metrics["locked_backend"]["oracle_coverage_weight"]) != 0.0:
        raise RuntimeError("H3 coverage-off source is not O=0")
    if float(h3_off_spec["diversity_weight"]) != 0.05:
        raise RuntimeError("H3 coverage-off source must retain diversity=.05")
    if float(h3_off_spec["oracle_coverage_weight"]) != 0.0:
        raise RuntimeError("H3 coverage-off provenance is inconsistent")
    if v2_on_backend["decoder_experiment"] != "v2_joint_wide_endpoint_correction":
        raise RuntimeError("Existing V2 reference has the wrong decoder")
    if float(v2_on_backend["oracle_coverage_weight"]) != 0.20:
        raise RuntimeError("Existing V2 reference is not using V2 coverage=.20")

    declared_source = Path(v2_on_config["source_checkpoint"]).resolve()
    if sha256(declared_source) != sha256(H3_ON_RUN / "best_selected.pth"):
        raise RuntimeError("Existing V2 reference does not originate from H3 coverage-on")
    if v2_on_config["source_sha256"] != sha256(H3_ON_RUN / "best_selected.pth"):
        raise RuntimeError("Existing V2 source hash does not match the current H3 source")

    audit: dict[str, Any] = {
        "scope": {
            "fold": "eth",
            "seed": 1,
            "split": "validation_only_scene_tail_5pct",
        },
        "design": {
            "reference": "H3 D=.05/O=.20 -> V2 D=.05/O=.20 (existing C00)",
            "treatment": "H3 D=.05/O=0 -> V2 D=.05/O=.20 (one new training)",
            "single_changed_factor": "H3 oracle coverage weight in the source checkpoint",
            "new_training_count": 1,
            "primary_checkpoint": "best_selected.pth",
            "sensitivity_checkpoint": "final_model.pth at epoch 20",
        },
        "test_policy": {
            "new_training_receives_test_paths": False,
            "analysis_reads_test_data": False,
            "existing_reference_note": (
                "The historical C00 artifact contains an old test evaluation, but this "
                "runner and analyzer never load or report it."
            ),
        },
        "validation_identity": {
            "sample_count": int(len(ids)),
            "sample_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "all_required_runs_exact_match": True,
        },
        "sources": {
            "h3_on": str(H3_ON_RUN / "best_selected.pth"),
            "h3_off": str(H3_OFF_RUN / "best_selected.pth"),
            "v2_on": str(V2_ON_RUN / "best_selected.pth"),
            "v2_off": str(NEW_RUN / "best_selected.pth"),
        },
        "source_hashes": {
            "h3_on": sha256(H3_ON_RUN / "best_selected.pth"),
            "h3_off": sha256(H3_OFF_RUN / "best_selected.pth"),
            "v2_on": sha256(V2_ON_RUN / "best_selected.pth"),
        },
        "inherited_preflight": inherited,
    }
    if require_new_v2:
        audit["source_hashes"]["v2_off"] = sha256(NEW_RUN / "best_selected.pth")
    return audit


def command_for(astra_root: Path) -> tuple[list[str], dict[str, str], Path]:
    command = stage_runner.common_command(
        astra_root=astra_root,
        config=stage_runner.CONFIG_005,
        decoder_experiment="v2_joint_wide_endpoint_correction",
        decoder_module=stage_runner.V2_MODULE,
        epochs=20,
        output_root=NEW_OUTPUT_ROOT,
    )
    # V2 coverage is deliberately kept at the standard .20.  Only the source
    # H3 checkpoint was trained without coverage.
    index = command.index("--oracle-coverage-weight")
    command[index + 1] = "0.20"
    forbidden = {"--test-data-path", "--test-num-path"}
    if forbidden.intersection(command):
        raise RuntimeError("Test paths must never be passed to this experiment")
    env = os.environ.copy()
    env["EXP173_H3_SOURCE_CHECKPOINT"] = str(H3_OFF_RUN / "best_selected.pth")
    env["EXP173_DECODER_INIT_SEED"] = "1"
    return command, env, NEW_OUTPUT_ROOT


def run_new_v2(astra_root: Path, force: bool, dry_run: bool) -> Path:
    audit = preflight(astra_root, require_new_v2=False)
    command, env, output_root = command_for(astra_root)
    destination = stage_runner.run_dir(output_root)
    metrics = destination / "metrics.json"
    if force:
        command.append("--force")
    elif metrics.is_file():
        print(f"[SKIP] Completed V2 propagation treatment: {metrics}")
        return destination

    print(f"[COMMAND:v2_off] {stage_runner.quote_command(command)}")
    if dry_run:
        return destination
    subprocess.run(command, check=True, cwd=str(REPO), env=env)
    require_files([
        destination / "best_selected.pth",
        destination / "final_model.pth",
        destination / "split_indices.npz",
        metrics,
    ], "Completed V2 propagation outputs")
    if not np.array_equal(validation_ids(H3_ON_RUN), validation_ids(destination)):
        raise RuntimeError("New V2 validation IDs changed after training")

    spec = {
        "experiment": "V2_COVERAGE_PROPAGATION",
        "fold": "eth",
        "seed": 1,
        "validation_only": True,
        "test_accessed_by_this_runner": False,
        "intervention": {
            "h3_diversity_weight": 0.05,
            "h3_oracle_coverage_weight": 0.0,
            "v2_diversity_weight": 0.05,
            "v2_oracle_coverage_weight": 0.20,
        },
        "h3_source_checkpoint": str(H3_OFF_RUN / "best_selected.pth"),
        "h3_source_sha256": sha256(H3_OFF_RUN / "best_selected.pth"),
        "command": command,
        "checkpoint_sha256": sha256(destination / "best_selected.pth"),
        "final_model_sha256": sha256(destination / "final_model.pth"),
        "split_indices_sha256": sha256(destination / "split_indices.npz"),
        "preflight": audit,
    }
    (destination / "coverage_propagation_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "v2_off"], required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    astra_root = args.astra_root.resolve()

    if args.stage == "preflight":
        audit = preflight(astra_root, require_new_v2=False)
        path = HERE / "preflight_audit.json"
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        print(f"[PASS] {path}")
        return

    destination = run_new_v2(astra_root, args.force, args.dry_run)
    print(f"[DONE] v2_off: {destination}")


if __name__ == "__main__":
    main()
