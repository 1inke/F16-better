"""Run EXP-175C once on SDD seed 1 and produce a paired summary."""
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
B_RESULT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175BR2_APPEARANCE_TO_MODE"
R2 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE"
ADAPTER = HERE / "scene_feasibility_adapter.py"
RESULTS = HERE / "results"
VARIANT = "f16_temporal_social_cross"
EXPERIMENT = "exp175c_soft_appearance_feasibility"
PIXEL_SCALE = 20.0


def source_checkpoint(seed: int) -> Path:
    return S0 / "results" / "joint_v2" / VARIANT / f"seed_{seed}" / "val_scene_1" / "best_selected.pth"


def run_dir(seed: int) -> Path:
    return RESULTS / VARIANT / f"seed_{seed}" / "val_scene_1"


def stream(command: list[str], env: dict[str, str], seed: int) -> None:
    print("[command] " + subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    log_path = run_dir(seed) / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
    if process.wait():
        raise RuntimeError(f"EXP-175C failed; see {log_path}")


def values(bundle: dict) -> dict[str, float]:
    return {
        "minADE_px": float(bundle["oracle"]["min_ade_20"]) * PIXEL_SCALE,
        "minFDE_px": float(bundle["oracle"]["min_fde_20"]) * PIXEL_SCALE,
        "same_path_FDE_px": float(bundle["oracle"]["same_path_fde"]) * PIXEL_SCALE,
        "top1_ADE_px": float(bundle["original"]["ade"]) * PIXEL_SCALE,
        "top1_FDE_px": float(bundle["original"]["fde"]) * PIXEL_SCALE,
    }


def weighted_group(by_scene: dict, names: dict[str, str], little: bool) -> dict[str, float]:
    selected = [
        (row, int(row["samples"])) for sid, row in by_scene.items()
        if names.get(str(sid), "").startswith("little/") == little
    ]
    total = sum(count for _, count in selected)
    return {
        "minADE_px": sum(float(row["min_ade_20"]) * count for row, count in selected) / total * PIXEL_SCALE,
        "minFDE_px": sum(float(row["min_fde_20"]) * count for row, count in selected) / total * PIXEL_SCALE,
        "same_path_FDE_px": sum(float(row["same_path_fde"]) * count for row, count in selected) / total * PIXEL_SCALE,
        "top1_ADE_px": sum(float(row["top1"]["ade"]) * count for row, count in selected) / total * PIXEL_SCALE,
        "top1_FDE_px": sum(float(row["top1"]["fde"]) * count for row, count in selected) / total * PIXEL_SCALE,
    }


def improvement(reference: dict, current: dict) -> dict[str, float]:
    return {
        key: (float(reference[key]) - float(value)) / float(reference[key])
        for key, value in current.items() if key in reference
    }


def summarize(seed: int, loss_weight: float) -> None:
    raw = json.loads((run_dir(seed) / "metrics.json").read_text(encoding="utf-8"))
    test = raw["evaluation_test_best_selected"]
    current = values(test)
    s0_all = json.loads((S0 / "summary.json").read_text(encoding="utf-8"))
    s0 = next(row for row in s0_all["per_seed"] if int(row["seed"]) == seed)
    b = json.loads((B_RESULT / "summary.json").read_text(encoding="utf-8"))["scene_model"]
    names = json.loads((S0 / "video_ids" / "video_id_manifest.json").read_text(encoding="utf-8"))["test_video_ids"]
    little = weighted_group(test["by_sample_scene"], names, True)
    non_little = weighted_group(test["by_sample_scene"], names, False)
    vs_s0 = improvement(s0, current)
    vs_b = improvement(b, current)
    little_ref = {k.removeprefix("little_"): v for k, v in s0.items() if k.startswith("little_")}
    non_little_ref = {k.removeprefix("non_little_"): v for k, v in s0.items() if k.startswith("non_little_")}
    little_gain = improvement(little_ref, little)
    non_little_gain = improvement(non_little_ref, non_little)
    params = raw["locked_backend"]["parameter_counts"]
    guide_gates = {
        "minADE_improvement_ge_5pct": vs_s0["minADE_px"] >= 0.05,
        "minFDE_improvement_ge_7pct": vs_s0["minFDE_px"] >= 0.07,
        "little_one_metric_ge_8pct": max(little_gain["minADE_px"], little_gain["minFDE_px"]) >= 0.08,
        "non_little_ADE_degradation_le_2pct": non_little_gain["minADE_px"] >= -0.02,
        "non_little_FDE_degradation_le_2pct": non_little_gain["minFDE_px"] >= -0.02,
        "trainable_parameters_le_0p03M": int(params["trainable"]) <= 30_000,
    }
    payload = {
        "experiment": "EXP-175C",
        "seed": seed,
        "scene_loss_weight": loss_weight,
        "model": "frozen S0 + appearance anchor residual + soft near-oracle path feasibility",
        "current": current,
        "s0_seed1": {k: v for k, v in s0.items() if k != "seed"},
        "exp175br2": b,
        "relative_improvement_vs_s0": vs_s0,
        "relative_improvement_vs_exp175br2": vs_b,
        "little": little,
        "non_little": non_little,
        "little_improvement_vs_s0": little_gain,
        "non_little_improvement_vs_s0": non_little_gain,
        "parameters": params,
        "guide_gates": guide_gates,
        "guide_gate_pass": all(guide_gates.values()),
        "one_percent_stop_gate": max(vs_s0["minADE_px"], vs_s0["minFDE_px"]) >= 0.01,
    }
    (HERE / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for scene_id, row in test["by_sample_scene"].items():
        rows.append({
            "seed": seed, "video_id": scene_id,
            "video": names.get(str(scene_id), f"unknown_{scene_id}"), "samples": int(row["samples"]),
            "minADE_px": float(row["min_ade_20"]) * PIXEL_SCALE,
            "minFDE_px": float(row["min_fde_20"]) * PIXEL_SCALE,
            "same_path_FDE_px": float(row["same_path_fde"]) * PIXEL_SCALE,
            "top1_ADE_px": float(row["top1"]["ade"]) * PIXEL_SCALE,
            "top1_FDE_px": float(row["top1"]["fde"]) * PIXEL_SCALE,
        })
    with (HERE / "per_video_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"[summary] {HERE / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-175C soft appearance feasibility")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--scene-loss-weight", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    bank = R2 / "assets" / "appearance_bank.npz"
    required = [
        ASTRA_ROOT, TRAINER, CFG, ADAPTER, bank, source_checkpoint(args.seed),
        DATA / "sdd_standard_train_val_data.npy", DATA / "sdd_standard_train_val_num.npy",
        DATA / "sdd_standard_test_data.npy", DATA / "sdd_standard_test_num.npy",
        S0 / "video_ids" / "sdd_train_val_video_ids.npy",
        S0 / "video_ids" / "sdd_test_video_ids.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if args.preflight_only:
        subprocess.run([
            sys.executable, "-u", str(HERE / "smoke_test_exp175c.py"),
            "--device", args.device, "--seed", str(args.seed),
            "--batch-size", str(min(args.batch_size, 4)),
            "--scene-loss-weight", str(args.scene_loss_weight),
        ], cwd=str(ROOT), check=True)
        return
    metrics = run_dir(args.seed) / "metrics.json"
    if metrics.exists() and not args.force:
        print(f"[skip completed] {metrics}")
        summarize(args.seed, args.scene_loss_weight)
        return
    command = [
        sys.executable, "-u", str(TRAINER), "--project-root", str(ASTRA_ROOT),
        "--config-file", str(CFG),
        "--data-path", str(DATA / "sdd_standard_train_val_data.npy"),
        "--num-path", str(DATA / "sdd_standard_train_val_num.npy"),
        "--test-data-path", str(DATA / "sdd_standard_test_data.npy"),
        "--test-num-path", str(DATA / "sdd_standard_test_num.npy"),
        "--sample-scene-path", str(S0 / "video_ids" / "sdd_train_val_video_ids.npy"),
        "--test-sample-scene-path", str(S0 / "video_ids" / "sdd_test_video_ids.npy"),
        "--variant", VARIANT, "--decoder-module", str(ADAPTER),
        "--decoder-experiment", EXPERIMENT, "--val-scene-id", "1",
        "--seed", str(args.seed), "--device", args.device,
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
    env["EXP175C_SOURCE_CHECKPOINT"] = str(source_checkpoint(args.seed))
    env["EXP175C_APPEARANCE_BANK"] = str(bank)
    env["EXP175C_SCENE_LOSS_WEIGHT"] = str(args.scene_loss_weight)
    stream(command, env, args.seed)
    summarize(args.seed, args.scene_loss_weight)


if __name__ == "__main__":
    main()

