"""Guarded runner for the two-stop PIF1 feasibility experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON_DEFAULT = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CONFIG = REPO / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
MODEL = HERE / "physical_intent_flow_models.py"
PREPARE = HERE / "prepare_intent_statistics.py"
AUDIT = HERE / "audit_c00_mode_utility.py"
ANALYZE = HERE / "analyze_pif1.py"
C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
RESULT_ROOT = HERE / "results" / "pif1_flow"
RUN_DIR = RESULT_ROOT / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
STATS = HERE / "calibration" / "intent_stats.json"
AUDIT_SUMMARY = HERE / "results" / "mode_utility_audit" / "summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logged(command: list[str], log_path: Path, env: dict | None = None, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); handle.write(line); handle.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Command failed; inspect {log_path}")


def train_command(args, data: Path) -> list[str]:
    return [
        str(args.python), "-u", str(TRAINER),
        "--project-root", str(args.astra_root),
        "--config-file", str(CONFIG),
        "--data-path", str(data / "eth_data_train.npy"),
        "--num-path", str(data / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "pif1_physical_intent_flow",
        "--decoder-module", str(MODEL),
        "--seed", "1", "--val-scene-id", "-1",
        "--validation-fraction", "0.05", "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail",
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--learning-rate", str(args.learning_rate),
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--output-root", str(RESULT_ROOT),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25", "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995", "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
    ]


def preflight(args) -> dict:
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    required = [
        args.python, TRAINER, CONFIG, MODEL, PREPARE, AUDIT, ANALYZE,
        data / "eth_data_train.npy", data / "eth_num_train.npy",
        C00_RUN / "best_selected.pth", C00_RUN / "split_indices.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    payload = {
        "protocol": "pif1_two_stop_validation_only_v1",
        "source_checkpoint": str(C00_RUN / "best_selected.pth"),
        "source_sha256": sha256(C00_RUN / "best_selected.pth"),
        "split_sha256": sha256(C00_RUN / "split_indices.npz"),
        "config_sha256": sha256(CONFIG),
        "test_paths_passed_to_training": False,
        "static_scene_inputs": False,
        "trainable_scope": "conditional_flow_only",
        "num_candidates": 20,
        "flow_layers": 6,
        "flow_hidden_dim": 96,
        "nll_weight": 0.10,
        "explicit_diversity_weight": 0.0,
        "oracle_coverage_weight": 0.20,
    }
    (HERE / "preflight.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "pilot", "analyze"), required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--analysis-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.astra_root = args.astra_root.resolve(); args.python = args.python.resolve()
    preflight(args)
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"

    if args.stage == "audit":
        command = [str(args.python), "-u", str(AUDIT)]
        if args.dry_run:
            print("[dry-run] " + json.dumps(command)); return
        logged(command, HERE / "results" / "mode_utility_audit" / "console.log", cwd=HERE)
        return

    if args.stage == "pilot":
        if not AUDIT_SUMMARY.is_file():
            raise RuntimeError(
                "Stop point 1 has not been completed. Run --stage audit and review "
                f"{HERE / 'results' / 'mode_utility_audit' / 'REPORT.md'} first."
            )
        prepare_command = [str(args.python), "-u", str(PREPARE), "--astra-root", str(args.astra_root)]
        command = train_command(args, data)
        analyze_command = [
            str(args.python), "-u", str(ANALYZE), "--astra-root", str(args.astra_root),
            "--device", args.device, "--batch-size", str(args.analysis_batch_size),
        ]
        if args.dry_run:
            print("[dry-run] prepare " + json.dumps(prepare_command))
            print("[dry-run] train " + json.dumps(command))
            print("[dry-run] analyze " + json.dumps(analyze_command)); return
        if args.force or not STATS.is_file():
            logged(prepare_command, HERE / "calibration" / "prepare.log", cwd=HERE)
        else:
            print(f"[skip] calibration exists: {STATS}", flush=True)
        env = os.environ.copy()
        env.update({
            "PIF_SOURCE_CHECKPOINT": str(C00_RUN / "best_selected.pth"),
            "PIF_STATS_JSON": str(STATS), "PIF_FLOW_LAYERS": "6",
            "PIF_HIDDEN_DIM": "96", "PIF_NLL_WEIGHT": "0.10",
        })
        if (RUN_DIR / "metrics.json").is_file() and not args.force:
            print(f"[skip] completed training: {RUN_DIR}", flush=True)
        else:
            if args.force:
                command.append("--force")
            logged(command, RUN_DIR / "console.log", env=env, cwd=args.astra_root)
        new_ids = np_load_validation(RUN_DIR / "split_indices.npz")
        locked_ids = np_load_validation(C00_RUN / "split_indices.npz")
        if new_ids != locked_ids:
            raise RuntimeError("PIF1 and C00 validation sample IDs differ")
        spec = {
            "preflight": json.loads((HERE / "preflight.json").read_text(encoding="utf-8")),
            "training_command": command, "stats_sha256": sha256(STATS),
            "validation_ids_exactly_match_c00": True,
        }
        (RUN_DIR / "pif1_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        logged(analyze_command, HERE / "results" / "analysis" / "console.log", cwd=HERE)
        return

    analyze_command = [
        str(args.python), "-u", str(ANALYZE), "--astra-root", str(args.astra_root),
        "--device", args.device, "--batch-size", str(args.analysis_batch_size),
    ]
    if args.dry_run:
        print("[dry-run] " + json.dumps(analyze_command)); return
    logged(analyze_command, HERE / "results" / "analysis" / "console.log", cwd=HERE)


def np_load_validation(path: Path) -> list[int]:
    import numpy as np
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)["validation"].astype(np.int64).tolist()


if __name__ == "__main__":
    main()

