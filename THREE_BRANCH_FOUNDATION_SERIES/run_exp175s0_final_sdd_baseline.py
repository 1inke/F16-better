from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATASET_DEFAULT = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
SERIES = Path(__file__).resolve().parent
EXP_ROOT = SERIES / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
RESULTS = EXP_ROOT / "results"
DATA_DIR = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
MANIFEST = DATA_DIR / "manifest.json"
PREPROCESS = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "preprocess_sdd_standard.py"
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CFG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
H3_MODULE = ROOT / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
VIDEO_ID_DIR = EXP_ROOT / "video_ids"
TRAIN_VIDEO_IDS = VIDEO_ID_DIR / "sdd_train_val_video_ids.npy"
TEST_VIDEO_IDS = VIDEO_ID_DIR / "sdd_test_video_ids.npy"
VIDEO_MAP = VIDEO_ID_DIR / "video_id_manifest.json"
VARIANT = "f16_temporal_social_cross"
PIXEL_SCALE = 20.0


def run_stream(command: list[str], env: dict[str, str] | None = None):
    print("[command] " + subprocess.list2cmdline(command), flush=True)
    proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}")


def ensure_data(dataset_root: Path, rebuild: bool, dry_run: bool):
    required = [DATA_DIR / "sdd_standard_train_val_data.npy", DATA_DIR / "sdd_standard_train_val_num.npy", DATA_DIR / "sdd_standard_test_data.npy", DATA_DIR / "sdd_standard_test_num.npy", MANIFEST]
    if rebuild or not all(path.exists() for path in required):
        command = [sys.executable, "-u", str(PREPROCESS), "--dataset-root", str(dataset_root), "--output-dir", str(DATA_DIR)]
        if rebuild:
            command.append("--rebuild")
        if dry_run:
            print("[would preprocess] " + subprocess.list2cmdline(command))
        else:
            run_stream(command)
    elif not dataset_root.exists():
        raise FileNotFoundError(dataset_root)


def ensure_video_ids(dry_run: bool):
    if not MANIFEST.exists():
        if dry_run:
            return None
        raise FileNotFoundError(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = manifest["videos"]
    stats = {row["video"]: row for row in manifest["video_stats"]}
    ordered = videos["train"] + videos["val"] + videos["test"]
    mapping = {name: i for i, name in enumerate(ordered)}
    train_val_ids = np.concatenate([np.full(int(stats[name]["samples"]), mapping[name], dtype=np.int64) for name in videos["train"] + videos["val"]])
    test_ids = np.concatenate([np.full(int(stats[name]["samples"]), mapping[name], dtype=np.int64) for name in videos["test"]])
    train_val_count = int(np.load(DATA_DIR / "sdd_standard_train_val_num.npy", mmap_mode="r").shape[0])
    test_count = int(np.load(DATA_DIR / "sdd_standard_test_num.npy", mmap_mode="r").shape[0])
    if len(train_val_ids) != train_val_count or len(test_ids) != test_count:
        raise RuntimeError(f"Video-ID/sample mismatch: IDs={len(train_val_ids)}/{len(test_ids)}, arrays={train_val_count}/{test_count}")
    payload = {"mapping": mapping, "test_video_ids": {str(mapping[name]): name for name in videos["test"]}, "split": videos}
    if not dry_run:
        VIDEO_ID_DIR.mkdir(parents=True, exist_ok=True)
        np.save(TRAIN_VIDEO_IDS, train_val_ids)
        np.save(TEST_VIDEO_IDS, test_ids)
        VIDEO_MAP.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_dir(stage: str, seed: int) -> Path:
    return RESULTS / stage / VARIANT / f"seed_{seed}" / "val_scene_1"


def training_command(stage: str, seed: int, epochs: int, batch_size: int, device: str, force: bool) -> list[str]:
    module = H3_MODULE if stage == "h3_anchor_proposal" else V2_MODULE
    experiment = "h3_anchor_proposal" if stage == "h3_anchor_proposal" else "v2_joint_wide_endpoint_correction"
    command = [
        sys.executable, "-u", str(TRAINER), "--project-root", str(ASTRA_ROOT), "--config-file", str(CFG),
        "--data-path", str(DATA_DIR / "sdd_standard_train_val_data.npy"), "--num-path", str(DATA_DIR / "sdd_standard_train_val_num.npy"),
        "--test-data-path", str(DATA_DIR / "sdd_standard_test_data.npy"), "--test-num-path", str(DATA_DIR / "sdd_standard_test_num.npy"),
        "--variant", VARIANT, "--decoder-module", str(module), "--decoder-experiment", experiment,
        "--val-scene-id", "1", "--seed", str(seed), "--device", device, "--epochs", str(epochs), "--val-freq", "2",
        "--batch-size", str(batch_size), "--max-agents", "16", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--checkpoint-criterion", "oracle_combo",
        "--sample-scene-path", str(TRAIN_VIDEO_IDS), "--test-sample-scene-path", str(TEST_VIDEO_IDS),
        "--focal-heading-normalize", "--augment-reflection", "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
        "--learning-rate", "0.00025", "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10", "--oracle-coverage-fde-weight", "0.25",
        "--ema-decay", "0.995", "--output-root", str(RESULTS / stage),
    ]
    if force:
        command.append("--force")
    return command


def stage_metrics(stage: str, seed: int) -> Path:
    return run_dir(stage, seed) / "metrics.json"


def load_metric(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("evaluation_test_best_selected") or data.get("evaluation_test")


def bundle_values(bundle: dict):
    return {
        "samples": int(bundle["samples"]),
        "minADE": float(bundle["oracle"]["min_ade_20"]),
        "minFDE": float(bundle["oracle"]["min_fde_20"]),
        "same_path_ADE": float(bundle.get("same_path_ade", bundle["oracle"]["min_ade_20"])),
        "same_path_FDE": float(bundle["oracle"]["same_path_fde"]),
        "top1_ADE": float(bundle["original"]["ade"]),
        "top1_FDE": float(bundle["original"]["fde"]),
    }


def aggregate_scene(scene_dict: dict, ids: list[str], key: str):
    selected = [(scene_dict[sid], int(scene_dict[sid]["samples"])) for sid in ids if sid in scene_dict]
    total = sum(count for _, count in selected)
    if not total:
        return None
    scene_key = {"minADE": "min_ade_20", "minFDE": "min_fde_20", "same_path_FDE": "same_path_fde"}.get(key)
    if key == "same_path_ADE":
        scene_key = "min_ade_20"
    if key.startswith("top1_"):
        raw_key = key.removeprefix("top1_").lower()
        return sum(float(row["top1"][raw_key]) * count for row, count in selected) / total
    return sum(float(row[scene_key]) * count for row, count in selected) / total


def summarize(seeds: list[int]):
    video_map = json.loads(VIDEO_MAP.read_text(encoding="utf-8"))["test_video_ids"]
    per_seed, per_video = [], []
    for seed in seeds:
        path = stage_metrics("joint_v2", seed)
        if not path.exists():
            continue
        raw, test = load_metric(path)
        values = bundle_values(test)
        row = {
            "seed": seed, "samples": values["samples"],
            "minADE_px": values["minADE"] * PIXEL_SCALE, "minFDE_px": values["minFDE"] * PIXEL_SCALE,
            "same_path_ADE_px": values["same_path_ADE"] * PIXEL_SCALE, "same_path_FDE_px": values["same_path_FDE"] * PIXEL_SCALE,
            "top1_ADE_px": values["top1_ADE"] * PIXEL_SCALE, "top1_FDE_px": values["top1_FDE"] * PIXEL_SCALE,
            "parameters": int(raw.get("locked_backend", {}).get("parameter_counts", {}).get("total", 0)),
        }
        scenes = test.get("by_sample_scene", {})
        little_ids = [sid for sid, name in video_map.items() if name.startswith("little/")]
        non_little_ids = [sid for sid, name in video_map.items() if not name.startswith("little/")]
        for group, ids in (("little", little_ids), ("non_little", non_little_ids)):
            for metric in ("minADE", "minFDE", "same_path_ADE", "same_path_FDE", "top1_ADE", "top1_FDE"):
                value = aggregate_scene(scenes, ids, metric)
                row[f"{group}_{metric}_px"] = None if value is None else value * PIXEL_SCALE
        per_seed.append(row)
        for sid, scene in scenes.items():
            per_video.append({
                "seed": seed, "video_id": sid, "video": video_map.get(sid, f"unknown_{sid}"), "samples": int(scene["samples"]),
                "minADE_px": float(scene["min_ade_20"] * PIXEL_SCALE), "minFDE_px": float(scene["min_fde_20"] * PIXEL_SCALE),
                "same_path_ADE_px": float(scene["min_ade_20"] * PIXEL_SCALE), "same_path_FDE_px": float(scene["same_path_fde"] * PIXEL_SCALE),
                "top1_ADE_px": float(scene["top1"]["ade"] * PIXEL_SCALE), "top1_FDE_px": float(scene["top1"]["fde"] * PIXEL_SCALE),
            })
    if not per_seed:
        return
    efficiency_file = SERIES / "EXP_174A_FINAL_EFFICIENCY" / "efficiency_results.json"
    efficiency = None
    if efficiency_file.exists():
        rows = json.loads(efficiency_file.read_text(encoding="utf-8"))["rows"]
        efficiency = next((r for r in rows if r["model"] == "F16" and r["family"] == "agent_scaling" and r["batch_size"] == 1 and r["agents"] == 16), None)
    numeric = [key for key, value in per_seed[0].items() if key != "seed" and isinstance(value, (int, float))]
    summary = {
        "experiment": "EXP-175S0", "protocol": "standard SDD 31/12/17 videos, obs8/pred12 at 2.5Hz, pedestrian/lost=0",
        "model": "EXP173 F16 + H3 + Joint-V2, no scene/image input", "pixel_scale": PIXEL_SCALE,
        "completed_seeds": [row["seed"] for row in per_seed], "per_seed": per_seed,
        "mean": {key: float(np.mean([row[key] for row in per_seed if row[key] is not None])) for key in numeric},
        "std": {key: float(np.std([row[key] for row in per_seed if row[key] is not None], ddof=1)) if len(per_seed) > 1 else 0.0 for key in numeric},
        "online_efficiency_F16_N16_B1": efficiency,
    }
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("per_seed_metrics.csv", per_seed), ("per_video_metrics.csv", per_video)):
        with (EXP_ROOT / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"[summary] {EXP_ROOT / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="EXP-175S0 final Joint-V2 SDD no-scene baseline")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ensure_data(args.dataset_root, args.rebuild_data, args.dry_run)
    ensure_video_ids(args.dry_run)
    for seed in args.seeds:
        for logical_stage, trainer_stage, epochs in (("h3", "h3_anchor_proposal", args.h3_epochs), ("joint_v2", "joint_v2", args.v2_epochs)):
            metrics = stage_metrics(trainer_stage, seed)
            if metrics.exists() and not args.force:
                print(f"[skip] {logical_stage} seed={seed}: {metrics}")
                continue
            command = training_command(trainer_stage, seed, epochs, args.batch_size, args.device, args.force)
            env = os.environ.copy()
            if logical_stage == "joint_v2":
                source = run_dir("h3_anchor_proposal", seed) / "best_selected.pth"
                if not source.exists() and not args.dry_run:
                    raise FileNotFoundError(source)
                env["EXP173_H3_SOURCE_CHECKPOINT"] = str(source)
                env["EXP173_DECODER_INIT_SEED"] = str(seed)
            if args.dry_run:
                print(f"[would run] seed={seed} stage={logical_stage}\n{subprocess.list2cmdline(command)}")
            else:
                run_stream(command, env)
    if not args.dry_run:
        summarize(args.seeds)
    else:
        print(json.dumps({"status": "preflight_ok", "seeds": args.seeds, "data": str(DATA_DIR), "results": str(RESULTS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
