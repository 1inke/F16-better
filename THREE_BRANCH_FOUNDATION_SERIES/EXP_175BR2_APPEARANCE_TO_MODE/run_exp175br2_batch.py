"""Train and summarize EXP-175B-R2 on standard SDD, seed 1 first."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CFG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
DATA = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
R2 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE"
ADAPTER = HERE / "scene_appearance_anchor_adapter.py"
RESULTS = HERE / "results"
VARIANT = "f16_temporal_social_cross"
EXPERIMENT = "exp175br2_appearance_to_mode"
PIXEL_SCALE = 20.0


def checkpoint(seed: int) -> Path:
    return S0 / "results" / "joint_v2" / VARIANT / f"seed_{seed}" / "val_scene_1" / "best_selected.pth"


def run_dir(seed: int) -> Path:
    return RESULTS / VARIANT / f"seed_{seed}" / "val_scene_1"


def stream(command: list[str], env: dict[str, str]) -> None:
    print("[command] " + subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    log_path = run_dir(int(env["EXP175BR2_SEED"])) / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    code = process.wait()
    if code:
        raise RuntimeError(f"EXP-175B-R2 failed; see {log_path}")


def metric_values(bundle: dict) -> dict[str, float]:
    return {
        "minADE_px": float(bundle["oracle"]["min_ade_20"]) * PIXEL_SCALE,
        "minFDE_px": float(bundle["oracle"]["min_fde_20"]) * PIXEL_SCALE,
        "same_path_FDE_px": float(bundle["oracle"]["same_path_fde"]) * PIXEL_SCALE,
        "top1_ADE_px": float(bundle["original"]["ade"]) * PIXEL_SCALE,
        "top1_FDE_px": float(bundle["original"]["fde"]) * PIXEL_SCALE,
    }


def weighted_group(by_scene: dict, names: dict[str, str], little: bool) -> dict[str, float]:
    selected = []
    for scene_id, row in by_scene.items():
        is_little = names.get(str(scene_id), "").startswith("little/")
        if is_little == little:
            selected.append((row, int(row["samples"])))
    total = sum(n for _, n in selected)
    if not total:
        return {}
    return {
        "minADE_px": sum(float(r["min_ade_20"]) * n for r, n in selected) / total * PIXEL_SCALE,
        "minFDE_px": sum(float(r["min_fde_20"]) * n for r, n in selected) / total * PIXEL_SCALE,
        "same_path_FDE_px": sum(float(r["same_path_fde"]) * n for r, n in selected) / total * PIXEL_SCALE,
        "top1_ADE_px": sum(float(r["top1"]["ade"]) * n for r, n in selected) / total * PIXEL_SCALE,
        "top1_FDE_px": sum(float(r["top1"]["fde"]) * n for r, n in selected) / total * PIXEL_SCALE,
    }


def summarize(seed: int) -> None:
    metrics_path = run_dir(seed) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = metrics["evaluation_test_best_selected"]
    current = metric_values(test)
    baseline_all = json.loads((S0 / "summary.json").read_text(encoding="utf-8"))
    baseline = next(row for row in baseline_all["per_seed"] if int(row["seed"]) == seed)
    video_map = json.loads((S0 / "video_ids" / "video_id_manifest.json").read_text(encoding="utf-8"))["test_video_ids"]
    little = weighted_group(test.get("by_sample_scene", {}), video_map, True)
    non_little = weighted_group(test.get("by_sample_scene", {}), video_map, False)
    improvements = {
        key: (float(baseline[key]) - value) / float(baseline[key])
        for key, value in current.items() if key in baseline
    }
    group_improvements = {}
    for group, values in (("little", little), ("non_little", non_little)):
        group_improvements[group] = {
            key: (float(baseline[f"{group}_{key}"]) - value) / float(baseline[f"{group}_{key}"])
            for key, value in values.items() if f"{group}_{key}" in baseline
        }
    total_params = int(metrics["locked_backend"]["parameter_counts"]["total"])
    trainable_params = int(metrics["locked_backend"]["parameter_counts"]["trainable"])
    gates = {
        "minADE_improvement_ge_5pct": improvements["minADE_px"] >= 0.05,
        "minFDE_improvement_ge_7pct": improvements["minFDE_px"] >= 0.07,
        "little_one_metric_ge_8pct": max(
            group_improvements["little"]["minADE_px"],
            group_improvements["little"]["minFDE_px"],
        ) >= 0.08,
        "non_little_ADE_degradation_le_2pct": group_improvements["non_little"]["minADE_px"] >= -0.02,
        "non_little_FDE_degradation_le_2pct": group_improvements["non_little"]["minFDE_px"] >= -0.02,
        "added_trainable_parameters_le_0p03M": trainable_params <= 30_000,
    }
    payload = {
        "experiment": "EXP-175B-R2",
        "seed": seed,
        "model": "frozen EXP-175S0 F16+H3+Joint-V2 + appearance-to-mode anchor residual",
        "no_future_scene_input": True,
        "data_augmentation_disabled_for_image_alignment": True,
        "baseline": {k: baseline[k] for k in baseline if k != "seed"},
        "scene_model": current,
        "little": little,
        "non_little": non_little,
        "relative_improvement": improvements,
        "group_relative_improvement": group_improvements,
        "parameters": {
            "total": total_params,
            "trainable_scene_branch": trainable_params,
            "added_over_s0": total_params - int(baseline["parameters"]),
        },
        "accuracy_and_parameter_gates": gates,
        "accuracy_and_parameter_gate_pass": all(gates.values()),
        "efficiency_gate_pending": True,
    }
    (HERE / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for scene_id, row in test.get("by_sample_scene", {}).items():
        values = {
            "seed": seed,
            "video_id": scene_id,
            "video": video_map.get(str(scene_id), f"unknown_{scene_id}"),
            "samples": int(row["samples"]),
            "minADE_px": float(row["min_ade_20"]) * PIXEL_SCALE,
            "minFDE_px": float(row["min_fde_20"]) * PIXEL_SCALE,
            "same_path_FDE_px": float(row["same_path_fde"]) * PIXEL_SCALE,
            "top1_ADE_px": float(row["top1"]["ade"]) * PIXEL_SCALE,
            "top1_FDE_px": float(row["top1"]["fde"]) * PIXEL_SCALE,
        }
        rows.append(values)
    with (HERE / "per_video_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"[summary] {HERE / 'summary.json'}", flush=True)


def preflight(seed: int, device: str, batch_size: int) -> None:
    command = [
        sys.executable, "-u", str(HERE / "smoke_test_exp175br2.py"),
        "--device", device, "--seed", str(seed), "--batch-size", str(batch_size),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-175B-R2 SDD appearance-to-mode training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    bank = R2 / "assets" / "appearance_bank.npz"
    required = [
        ASTRA_ROOT, TRAINER, CFG, ADAPTER, bank, checkpoint(args.seed),
        DATA / "sdd_standard_train_val_data.npy", DATA / "sdd_standard_train_val_num.npy",
        DATA / "sdd_standard_test_data.npy", DATA / "sdd_standard_test_num.npy",
        S0 / "video_ids" / "sdd_train_val_video_ids.npy",
        S0 / "video_ids" / "sdd_test_video_ids.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if args.preflight_only:
        preflight(args.seed, args.device, min(args.batch_size, 4))
        return

    metrics = run_dir(args.seed) / "metrics.json"
    if metrics.exists() and not args.force:
        print(f"[skip completed] {metrics}")
        summarize(args.seed)
        return
    command = [
        sys.executable, "-u", str(TRAINER),
        "--project-root", str(ASTRA_ROOT), "--config-file", str(CFG),
        "--data-path", str(DATA / "sdd_standard_train_val_data.npy"),
        "--num-path", str(DATA / "sdd_standard_train_val_num.npy"),
        "--test-data-path", str(DATA / "sdd_standard_test_data.npy"),
        "--test-num-path", str(DATA / "sdd_standard_test_num.npy"),
        "--sample-scene-path", str(S0 / "video_ids" / "sdd_train_val_video_ids.npy"),
        "--test-sample-scene-path", str(S0 / "video_ids" / "sdd_test_video_ids.npy"),
        "--variant", VARIANT,
        "--decoder-module", str(ADAPTER), "--decoder-experiment", EXPERIMENT,
        "--val-scene-id", "1", "--seed", str(args.seed), "--device", args.device,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--max-agents", "16",
        "--graph-top-k", "4", "--graph-radius", "6.0",
        "--checkpoint-criterion", "oracle_combo", "--focal-heading-normalize",
        "--learning-rate", "0.00025", "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1", "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10", "--oracle-coverage-fde-weight", "0.25",
        "--ema-decay", "0.995", "--output-root", str(RESULTS),
    ]
    if args.force:
        command.append("--force")
    env = os.environ.copy()
    env["EXP175BR2_SOURCE_CHECKPOINT"] = str(checkpoint(args.seed))
    env["EXP175BR2_APPEARANCE_BANK"] = str(bank)
    env["EXP175BR2_SEED"] = str(args.seed)
    stream(command, env)
    summarize(args.seed)


if __name__ == "__main__":
    main()

