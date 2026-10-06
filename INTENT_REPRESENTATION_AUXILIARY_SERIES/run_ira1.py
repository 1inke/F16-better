"""Run the matched IRA0/IRA1 representation-gradient experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON_DEFAULT = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CONFIG = REPO / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
MODEL = HERE / "intent_aux_models.py"
ANALYZE = HERE / "analyze_ira1.py"
TEST = HERE / "test_ira1.py"
SMOKE = HERE / "smoke_check.py"
STATS = REPO / "PHYSICAL_INTENT_FLOW_SERIES" / "calibration" / "intent_stats.json"
PIF_VERDICT = REPO / "PHYSICAL_INTENT_FLOW_SERIES" / "results" / "analysis" / "verdict.json"
C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
ARMS = {
    "ira0_detached_intent_aux": HERE / "results" / "ira0_detached",
    "ira1_joint_intent_aux": HERE / "results" / "ira1_joint",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dir(root: Path) -> Path:
    return root / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


def logged(
    command: list[str], log_path: Path, env: dict | None = None, cwd: Path | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Command failed; inspect {log_path}")


def train_command(args, data: Path, experiment: str, output_root: Path) -> list[str]:
    return [
        str(args.python), "-u", str(TRAINER),
        "--project-root", str(args.astra_root),
        "--config-file", str(CONFIG),
        "--data-path", str(data / "eth_data_train.npy"),
        "--num-path", str(data / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", experiment,
        "--decoder-module", str(MODEL),
        "--seed", "1", "--val-scene-id", "-1",
        "--validation-fraction", "0.05", "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail",
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--learning-rate", str(args.learning_rate),
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--output-root", str(output_root),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25",
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995", "--focal-heading-normalize",
        "--augment-reflection", "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1", "--augment-probability", "0.5",
    ]


def validation_ids(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)["validation"].astype(np.int64)


def preflight(args) -> dict:
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    required = [
        args.python, TRAINER, CONFIG, MODEL, ANALYZE, TEST, SMOKE, STATS, PIF_VERDICT,
        data / "eth_data_train.npy", data / "eth_num_train.npy",
        C00_RUN / "best_selected.pth", C00_RUN / "split_indices.npz",
        C00_RUN / "locked_backend_contract.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    pif = json.loads(PIF_VERDICT.read_text(encoding="utf-8"))
    if pif.get("verdict") != "STOP_PIF1_AFTER_SEED1":
        raise RuntimeError(f"Unexpected PIF1 decision: {pif.get('verdict')}")
    contract = json.loads(
        (C00_RUN / "locked_backend_contract.json").read_text(encoding="utf-8")
    )
    if contract.get("decoder_class") != "JointTrainableV2Decoder":
        raise RuntimeError("Locked C00 decoder identity changed")
    payload = {
        "protocol": "ira1_matched_representation_gradient_validation_only_v1",
        "causal_contrast": "joint_intent_gradient_minus_detached_intent_gradient",
        "source_checkpoint": str(C00_RUN / "best_selected.pth"),
        "source_sha256": sha256(C00_RUN / "best_selected.pth"),
        "split_sha256": sha256(C00_RUN / "split_indices.npz"),
        "stats_sha256": sha256(STATS),
        "config_sha256": sha256(CONFIG),
        "test_paths_passed_to_training_or_analysis": False,
        "inference_candidate_generator": "unchanged_C00",
        "inference_score_head": "unchanged_C00",
        "auxiliary_head_used_at_inference": False,
        "flow_layers": 6,
        "flow_hidden_dim": 96,
        "intent_nll_weight": 0.05,
        "oracle_coverage_weight": 0.20,
        "arms": list(ARMS),
    }
    (HERE / "preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def analyze_command(args) -> list[str]:
    return [
        str(args.python), "-u", str(ANALYZE),
        "--astra-root", str(args.astra_root),
        "--device", args.device,
        "--batch-size", str(args.analysis_batch_size),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "analyze"), required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-freq", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--analysis-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.astra_root = args.astra_root.resolve()
    args.python = args.python.resolve()
    preflight(args)

    if args.stage == "analyze":
        command = analyze_command(args)
        if args.dry_run:
            print("[dry-run] " + json.dumps(command))
            return
        logged(command, HERE / "results" / "analysis" / "console.log", cwd=HERE)
        return

    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    test_command = [str(args.python), "-u", str(TEST)]
    smoke_command = [str(args.python), "-u", str(SMOKE)]
    commands = {
        name: train_command(args, data, name, root) for name, root in ARMS.items()
    }
    final_analysis = analyze_command(args)
    if args.dry_run:
        print("[dry-run] contract " + json.dumps(test_command))
        print("[dry-run] smoke " + json.dumps(smoke_command))
        for name, command in commands.items():
            print(f"[dry-run] {name} " + json.dumps(command))
        print("[dry-run] analyze " + json.dumps(final_analysis))
        return

    logged(test_command, HERE / "results" / "contract_test.log", cwd=HERE)
    logged(smoke_command, HERE / "results" / "smoke_check.log", cwd=HERE)
    env = os.environ.copy()
    env.update({
        "IRA_SOURCE_CHECKPOINT": str(C00_RUN / "best_selected.pth"),
        "IRA_STATS_JSON": str(STATS),
        "IRA_FLOW_LAYERS": "6",
        "IRA_FLOW_HIDDEN_DIM": "96",
        "IRA_NLL_WEIGHT": "0.05",
    })
    locked_ids = validation_ids(C00_RUN / "split_indices.npz")
    for name, command in commands.items():
        destination = run_dir(ARMS[name])
        if (destination / "metrics.json").is_file() and not args.force:
            print(f"[skip] completed training: {destination}", flush=True)
        else:
            if args.force:
                command.append("--force")
            logged(command, destination / "console.log", env=env, cwd=args.astra_root)
        if not np.array_equal(validation_ids(destination / "split_indices.npz"), locked_ids):
            raise RuntimeError(f"{name} validation IDs differ from locked C00")
        spec = {
            "preflight": json.loads(
                (HERE / "preflight.json").read_text(encoding="utf-8")
            ),
            "experiment": name,
            "training_command": command,
            "validation_ids_exactly_match_c00": True,
        }
        (destination / "ira1_spec.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    logged(final_analysis, HERE / "results" / "analysis" / "console.log", cwd=HERE)


if __name__ == "__main__":
    main()
