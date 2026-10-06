"""Run the two missing cells of an H3 Diversity x Oracle-Coverage 2x2.

Existing cells are reused:
  D1O1: diversity=.05, oracle coverage=.20 (EXP-173 H3 baseline)
  D0O1: diversity=0,   oracle coverage=.20 (diversity-stage attribution)

New validation-only ETH/seed-1 cells:
  D1O0: diversity=.05, oracle coverage=0
  D0O0: diversity=0,   oracle coverage=0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES"))
import run_stage_attribution as diversity_runner

ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
RESULTS = HERE / "results"
H3_BASE = diversity_runner.H3_BASELINE_RUN
H3_NODIV = diversity_runner.run_dir(diversity_runner.RESULTS / "h3_w000")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_dir(key: str) -> Path:
    return diversity_runner.run_dir(RESULTS / key)


def preflight(astra_root: Path) -> dict:
    inherited = diversity_runner.preflight(astra_root, require_new_h3=True)
    required = [
        H3_BASE / "best_selected.pth", H3_BASE / "final_model.pth",
        H3_BASE / "split_indices.npz", H3_NODIV / "best_selected.pth",
        H3_NODIV / "final_model.pth", H3_NODIV / "split_indices.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reused H3 inputs:\n  " + "\n  ".join(missing))
    baseline_ids = np.load(H3_BASE / "split_indices.npz")["validation"]
    nodiv_ids = np.load(H3_NODIV / "split_indices.npz")["validation"]
    if not np.array_equal(baseline_ids, nodiv_ids):
        raise RuntimeError("Reused H3 validation IDs do not match")
    inherited["objective_attribution"] = {
        "design": "2x2 diversity weight {.05,0} x oracle coverage weight {.20,0}",
        "new_training_cells": ["d1_o0", "d0_o0"],
        "reused_cells": ["d1_o1", "d0_o1"],
        "primary_checkpoint": "best_selected.pth",
        "matched_epoch_sensitivity": "final_model.pth at epoch 30",
        "validation_id_sha256": hashlib.sha256(baseline_ids.tobytes()).hexdigest(),
        "test_policy": "no test paths are passed",
    }
    return inherited


def command_for(key: str, astra_root: Path) -> tuple[list[str], Path, float]:
    if key not in {"d1_o0", "d0_o0"}:
        raise ValueError(key)
    diversity_weight = 0.05 if key == "d1_o0" else 0.0
    config = (
        diversity_runner.CONFIG_005 if diversity_weight == 0.05
        else diversity_runner.CONFIG_000
    )
    output_root = RESULTS / key
    command = diversity_runner.common_command(
        astra_root=astra_root,
        config=config,
        decoder_experiment="h3_anchor_proposal",
        decoder_module=diversity_runner.H3_MODULE,
        epochs=30,
        output_root=output_root,
    )
    index = command.index("--oracle-coverage-weight")
    command[index + 1] = "0.00"
    return command, output_root, diversity_weight


def run_one(key: str, astra_root: Path, force: bool, dry_run: bool) -> Path:
    audit = preflight(astra_root)
    command, output_root, diversity_weight = command_for(key, astra_root)
    destination = diversity_runner.run_dir(output_root)
    metrics = destination / "metrics.json"
    if force:
        command.append("--force")
    elif metrics.is_file():
        print(f"[SKIP] {key}: {metrics}")
        return destination
    print(f"[COMMAND:{key}] {diversity_runner.quote_command(command)}")
    if dry_run:
        return destination
    subprocess.run(command, check=True, cwd=str(REPO))
    required = [
        destination / "best_selected.pth", destination / "final_model.pth",
        destination / "split_indices.npz", metrics,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed outputs:\n  " + "\n  ".join(missing))
    spec = {
        "condition": key,
        "fold": "eth",
        "seed": 1,
        "diversity_weight": diversity_weight,
        "oracle_coverage_weight": 0.0,
        "validation_only": True,
        "test_accessed_by_this_runner": False,
        "command": command,
        "checkpoint_sha256": sha256(destination / "best_selected.pth"),
        "final_model_sha256": sha256(destination / "final_model.pth"),
        "split_indices_sha256": sha256(destination / "split_indices.npz"),
        "preflight": audit,
    }
    (destination / "objective_attribution_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["preflight", "d1_o0", "d0_o0", "all"], required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    astra_root = args.astra_root.resolve()
    if args.condition == "preflight":
        audit = preflight(astra_root)
        path = HERE / "preflight_audit.json"
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        print(f"[PASS] {path}")
        return
    keys = [args.condition] if args.condition != "all" else ["d1_o0", "d0_o0"]
    for key in keys:
        destination = run_one(key, astra_root, args.force, args.dry_run)
        print(f"[DONE] {key}: {destination}")


if __name__ == "__main__":
    main()
