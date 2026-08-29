"""Run the matched current-decoder SDD semantic-anchor-query experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
SDD = WORKSPACE / "SDD_STANDARD_POTENTIAL_SERIES"
PREPROCESS = SDD / "preprocess_sdd_standard.py"
PREPARE_SCENE = WORKSPACE / "SCENE_QUERY_SERIES" / "prepare_sdd_scene_assets.py"
MODEL_MODULE = ROOT / "current_scene_query_models.py"
DATA_DIR = SDD / "processed_data"
ASSET_DIR = ROOT / "processed_scene_assets"
EXPERIMENT = ROOT / "EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY"


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def stream(command: list[str], cwd: Path, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); handle.write(line); handle.flush()
        if process.wait():
            raise RuntimeError(f"failed; see {log}")


def as_pixels(evaluation: dict | None, scale: float = 20.0) -> dict | None:
    if not evaluation:
        return None
    oracle, original = evaluation["oracle"], evaluation["original"]
    return {
        "samples": evaluation["samples"],
        "minADE20_px": oracle["min_ade_20"] * scale,
        "minFDE20_px": oracle["min_fde_20"] * scale,
        "same_path_FDE_px": oracle["same_path_fde"] * scale,
        "top1_ADE_px": original["ade"] * scale,
        "top1_FDE_px": original["fde"] * scale,
        "by_sample_scene": evaluation.get("by_sample_scene", {}),
    }


def collect(variants: list[str], seeds: list[int]) -> None:
    rows = []
    for variant in variants:
        for seed in seeds:
            path = EXPERIMENT / "results" / variant / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_1" / "metrics.json"
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "variant": variant, "seed": seed,
                "parameters": raw["locked_backend"]["parameter_counts"],
                "elapsed_seconds": raw["elapsed_seconds"],
                "validation_pixels": as_pixels(raw.get("evaluation_best_selected")),
                "test_pixels": as_pixels(raw.get("evaluation_test_best_selected")),
                "metrics_path": str(path),
            })
    (EXPERIMENT / "result_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variants", choices=("both", "s0", "s1"), default="both")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    variants = {"both": ["s0_current", "s1_scene_anchor_query"], "s0": ["s0_current"], "s1": ["s1_scene_anchor_query"]}[args.variants]
    seeds = parse_seeds(args.seeds)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["EXP147B_SCENE_ASSET_PATH"] = str(ASSET_DIR / "sdd_semantic_bank.npz")
    env["EXP147B_HEADING_NORMALIZED"] = "1"
    preprocess = [str(PYTHON), "-u", str(PREPROCESS), "--dataset-root", str(args.dataset_root), "--output-dir", str(DATA_DIR), "--max-agents", "16", "--pixel-scale", "20.0"]
    if args.rebuild_data:
        preprocess.append("--rebuild")
    prepare = [str(PYTHON), "-u", str(PREPARE_SCENE), "--dataset-root", str(args.dataset_root), "--trajectory-manifest", str(DATA_DIR / "manifest.json"), "--output-dir", str(ASSET_DIR), "--device", args.device]
    if args.force_assets:
        prepare.append("--force")
    commands = []
    for variant in variants:
        for seed in seeds:
            output = EXPERIMENT / "results" / variant
            command = [str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
                       "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
                       "--data-path", str(DATA_DIR / "sdd_standard_train_val_data.npy"),
                       "--num-path", str(DATA_DIR / "sdd_standard_train_val_num.npy"),
                       "--test-data-path", str(DATA_DIR / "sdd_standard_test_data.npy"),
                       "--test-num-path", str(DATA_DIR / "sdd_standard_test_num.npy"),
                       "--sample-scene-path", str(ASSET_DIR / "sdd_train_val_scene_ids.npy"),
                       "--test-sample-scene-path", str(ASSET_DIR / "sdd_test_scene_ids.npy"),
                       "--variant", "f16_temporal_social_cross", "--decoder-experiment", variant,
                       "--decoder-module", str(MODEL_MODULE), "--seed", str(seed), "--val-scene-id", "1",
                       "--epochs", str(args.epochs), "--val-freq", "2", "--batch-size", str(args.batch_size),
                       "--device", args.device, "--max-agents", "16", "--graph-top-k", "4", "--graph-radius", "6.0",
                       "--scheduler", "warm_restarts", "--output-root", str(output),
                       "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
                       "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995", "--focal-heading-normalize"]
            if args.force:
                command.append("--force")
            commands.append((variant, seed, command, output))
    if args.dry_run:
        print("[preprocess] " + json.dumps(preprocess, ensure_ascii=False))
        print("[prepare-scenes] " + json.dumps(prepare, ensure_ascii=False))
        for variant, seed, command, _ in commands:
            print(f"[{variant} seed={seed}] " + json.dumps(command, ensure_ascii=False))
        return
    started = time.perf_counter()
    stream(preprocess, SDD, EXPERIMENT / "preprocess.log", env)
    stream(prepare, ROOT, EXPERIMENT / "prepare_scene_assets.log", env)
    for variant, seed, command, output in commands:
        run = output / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_1"
        print(f"[run] {variant} seed={seed}", flush=True)
        stream(command, PROJECT, run / "console.log", env)
        collect(variants, seeds)
    print(f"[done] elapsed={(time.perf_counter() - started) / 60.0:.1f} min")
    print(f"[summary] {EXPERIMENT / 'result_summary.json'}")


if __name__ == "__main__":
    main()
