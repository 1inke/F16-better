"""Prepare SDD semantic assets and run the EXP-147A controlled A/B."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
PREPARE = ROOT / "prepare_sdd_scene_assets.py"
MODEL_MODULE = ROOT / "scene_query_models.py"
SDD_SERIES = WORKSPACE / "SDD_STANDARD_POTENTIAL_SERIES"
DATA_DIR = SDD_SERIES / "processed_data"
ASSET_DIR = ROOT / "processed_scene_assets"
EXPERIMENT = ROOT / "EXP_147A_SEMANTIC_ANCHOR_QUERY"


def parse_seeds(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one seed is required")
    return values


def stream(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"failed with code {code}; see {log_path}")


def collect_summary(decoders: list[str], seeds: list[int]) -> None:
    rows = []
    for decoder in decoders:
        for seed in seeds:
            path = (
                EXPERIMENT / "results" / decoder / "f16_temporal_social_cross"
                / f"seed_{seed}" / "val_scene_1" / "metrics.json"
            )
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            val = metrics["evaluation_best_selected"]["oracle"]
            test = metrics["evaluation_test_best_selected"]["oracle"]
            scale = 20.0
            rows.append({
                "decoder": decoder, "seed": seed,
                "val_minADE20_px": val["min_ade_20"] * scale,
                "val_minFDE20_px": val["min_fde_20"] * scale,
                "test_minADE20_px": test["min_ade_20"] * scale,
                "test_minFDE20_px": test["min_fde_20"] * scale,
                "parameters": metrics["locked_backend"]["parameter_counts"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "test_by_scene": metrics["evaluation_test_best_selected"].get(
                    "by_sample_scene", {}
                ),
                "metrics_path": str(path),
            })
    (EXPERIMENT / "raw_result_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--variants", choices=["both", "control", "scene"], default="both"
    )
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    decoders = {
        "both": ["learned_anchor_control", "semantic_anchor_query"],
        "control": ["learned_anchor_control"],
        "scene": ["semantic_anchor_query"],
    }[args.variants]
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    prepare = [
        str(PYTHON), "-u", str(PREPARE),
        "--dataset-root", str(args.dataset_root),
        "--trajectory-manifest", str(DATA_DIR / "manifest.json"),
        "--output-dir", str(ASSET_DIR), "--device", args.device,
    ]
    if args.force_assets:
        prepare.append("--force")
    env = os.environ.copy()
    env["EXP147_SCENE_ASSET_PATH"] = str(ASSET_DIR / "sdd_semantic_bank.npz")
    env["EXP147_HEADING_NORMALIZED"] = "1"

    commands = []
    for decoder in decoders:
        output_root = EXPERIMENT / "results" / decoder
        for seed in seeds:
            command = [
                str(PYTHON), "-u", str(TRAINER),
                "--project-root", str(PROJECT),
                "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                "--data-path", str(DATA_DIR / "sdd_standard_train_val_data.npy"),
                "--num-path", str(DATA_DIR / "sdd_standard_train_val_num.npy"),
                "--test-data-path", str(DATA_DIR / "sdd_standard_test_data.npy"),
                "--test-num-path", str(DATA_DIR / "sdd_standard_test_num.npy"),
                "--sample-scene-path", str(ASSET_DIR / "sdd_train_val_scene_ids.npy"),
                "--test-sample-scene-path", str(ASSET_DIR / "sdd_test_scene_ids.npy"),
                "--variant", "f16_temporal_social_cross",
                "--decoder-experiment", decoder,
                "--decoder-module", str(MODEL_MODULE),
                "--seed", str(seed), "--val-scene-id", "1",
                "--epochs", str(args.epochs), "--val-freq", "2",
                "--batch-size", str(args.batch_size), "--device", args.device,
                "--max-agents", "16", "--graph-top-k", "4", "--graph-radius", "6.0",
                "--scheduler", "warm_restarts", "--output-root", str(output_root),
                "--oracle-coverage-weight", "0.20",
                "--oracle-coverage-temperature", "0.10",
                "--checkpoint-criterion", "oracle_combo",
                "--ema-decay", "0.995", "--focal-heading-normalize",
            ]
            if args.force:
                command.append("--force")
            commands.append((decoder, seed, command, output_root))

    if args.dry_run:
        print("[prepare] " + json.dumps(prepare, ensure_ascii=False))
        for decoder, seed, command, _ in commands:
            print(f"[{decoder} seed={seed}] " + json.dumps(command, ensure_ascii=False))
        return

    started = time.perf_counter()
    stream(prepare, ROOT, EXPERIMENT / "prepare_scene_assets.log", env)
    for decoder, seed, command, output_root in commands:
        print(f"[run] {decoder} seed={seed}", flush=True)
        log_path = (
            output_root / "f16_temporal_social_cross" / f"seed_{seed}"
            / "val_scene_1" / "console.log"
        )
        stream(command, PROJECT, log_path, env)
        collect_summary(decoders, seeds)
    elapsed = time.perf_counter() - started
    print(f"[done] elapsed={elapsed / 60.0:.1f} min", flush=True)
    print(f"[summary] {EXPERIMENT / 'raw_result_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
