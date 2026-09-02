from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
AUDIT = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156A_PROTOCOL_AUDIT" / "protocol_audit.json"
H3_MODULE = ROOT / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
ENCODER_MODULE = EXP_DIR / "exp174d_encoder_adapter.py"
BASELINE = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
EFFICIENCY_BASE = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp174a_final_efficiency.py"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANT = "f16_temporal_social_cross"
LABEL = "q3kv"
ENCODER_EXPERIMENT = "three_representative_temporal_social_kv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_dir(fold: str, stage: str, seed: int) -> Path:
    return RESULTS / fold / stage / LABEL / VARIANT / f"seed_{seed}" / "val_scene_-1"


def baseline_path(fold: str, stage: str, seed: int, filename: str) -> Path:
    return BASELINE / fold / stage / "e3" / VARIANT / f"seed_{seed}" / "val_scene_-1" / filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, fold: str, stage: str, seed: int) -> list[str]:
    module, decoder_experiment, epochs = (
        (H3_MODULE, "h3_anchor_proposal", args.h3_epochs)
        if stage == "h3" else
        (V2_MODULE, "v2_joint_wide_endpoint_correction", args.v2_epochs)
    )
    cmd = [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
        "--config-file", str(CONFIG), "--data-path", str(DATA / f"{fold}_data_train.npy"),
        "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
        "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        "--variant", VARIANT, "--encoder-module", str(ENCODER_MODULE),
        "--encoder-experiment", ENCODER_EXPERIMENT,
        "--decoder-module", str(module), "--decoder-experiment", decoder_experiment,
        "--seed", str(seed), "--val-scene-id", "-1", "--validation-fraction", "0.05",
        "--validation-seed", "20260830", "--validation-split-mode", "scene_tail",
        "--epochs", str(epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--learning-rate", "0.00025", "--output-root", str(RESULTS / fold / stage / LABEL),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25", "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995", "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def run_stage(args, fold: str, stage: str, seed: int):
    destination = run_dir(fold, stage, seed)
    if (destination / "metrics.json").exists() and not args.force:
        print(f"[skip] {fold}/{stage}/seed{seed}", flush=True); return
    source = run_dir(fold, "h3", seed) / "best_selected.pth" if stage == "v2" else None
    if source is not None and not source.exists() and not args.dry_run:
        raise FileNotFoundError(source)
    cmd = command(args, fold, stage, seed)
    if args.dry_run:
        print("[dry-run] " + subprocess.list2cmdline(cmd), flush=True); return
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if stage == "v2":
        env["EXP173_H3_SOURCE_CHECKPOINT"] = str(source)
        env["EXP173_DECODER_INIT_SEED"] = str(seed)
    print(f"[run] {fold}/{stage}/seed{seed}", flush=True)
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{fold}/{stage}] {line}", end="", flush=True)
            log.write(line); log.flush()
    if process.wait():
        raise RuntimeError(f"Failed: {destination / 'console.log'}")
    locked = {
        "experiment": "EXP-174D", "fold": fold, "stage": stage, "seed": seed,
        "encoder": "F16 one-Q + long/short/latest K/V", "source": str(source) if source else None,
        "source_sha256": sha256(source) if source else None, "command": cmd,
    }
    (destination / "exp174d_contract.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")


def metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))["evaluation_test_best_selected"]
    return {
        "minADE": float(data["oracle"]["min_ade_20"]),
        "minFDE": float(data["oracle"]["min_fde_20"]),
        "same_path_FDE": float(data["oracle"]["same_path_fde"]),
        "top1_ADE": float(data["original"]["ade"]),
        "top1_FDE": float(data["original"]["fde"]),
    }


def summarize(seed: int):
    rows = []
    for fold in FOLDS:
        custom = run_dir(fold, "v2", seed) / "metrics.json"
        base = baseline_path(fold, "v2", seed, "metrics.json")
        if not custom.exists():
            continue
        c, b = metrics(custom), metrics(base)
        row = {"fold": fold, "seed": seed}
        for key in c:
            row[f"baseline_{key}"] = b[key]; row[f"q3kv_{key}"] = c[key]
            row[f"relative_change_{key}"] = (c[key] - b[key]) / b[key]
        row["baseline_J"] = b["minADE"] + 0.5 * b["minFDE"]
        row["q3kv_J"] = c["minADE"] + 0.5 * c["minFDE"]
        row["relative_change_J"] = (row["q3kv_J"] - row["baseline_J"]) / row["baseline_J"]
        rows.append(row)
    payload = {"experiment": "EXP-174D", "completed_folds": [r["fold"] for r in rows], "rows": rows}
    if len(rows) == 5:
        macro = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ("fold", "seed")}
        j_degradation = (macro["q3kv_J"] - macro["baseline_J"]) / macro["baseline_J"]
        fde_degradation = (macro["q3kv_minFDE"] - macro["baseline_minFDE"]) / macro["baseline_minFDE"]
        payload.update({"macro": macro, "accuracy_gates": {
            "J_degradation_le_0p5pct": j_degradation <= 0.005,
            "minFDE_degradation_le_1pct": fde_degradation <= 0.01,
            "J_relative_change": j_degradation, "minFDE_relative_change": fde_degradation,
        }})
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (EXP_DIR / "per_fold.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return payload


def efficiency_audit(device_text: str, seed: int, warmup: int, iterations: int) -> dict:
    device = torch.device(device_text)
    base = load_module("exp174d_efficiency_final", EFFICIENCY_BASE)
    adapter = load_module("exp174d_adapter_final", ENCODER_MODULE)
    baseline_model, cfg, trainer = base.build_model("F16", seed, device, 9)
    custom_model, _, _ = base.build_model("F16", seed, device, 9)
    adapter.replace_encoder(custom_model, cfg, ENCODER_EXPERIMENT)
    custom_checkpoint = run_dir("eth", "v2", seed) / "best_selected.pth"
    custom_model.load_state_dict(
        torch.load(custom_checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    custom_model = custom_model.to(device).eval()
    past, valid = base.real_batch(trainer, cfg, 1, device)
    rows = []
    for label, model in (("F16_baseline", baseline_model), ("Q3KV", custom_model)):
        row = {
            "model": label,
            "parameters": sum(p.numel() for p in model.parameters()),
            "profiler_flops": base.profile_flops(model, past, valid, device),
        }
        row.update(base.benchmark(model, past, valid, device, warmup, iterations))
        rows.append(row)
    b, q = rows
    result = {
        "protocol": {"batch_size": 1, "warmup": warmup, "iterations": iterations, "precision": "FP32"},
        "rows": rows,
        "latency_improvement": (b["latency_median_ms"] - q["latency_median_ms"]) / b["latency_median_ms"],
        "flops_improvement": (b["profiler_flops"] - q["profiler_flops"]) / max(b["profiler_flops"], 1),
    }
    result["efficiency_gate"] = bool(
        result["latency_improvement"] >= 0.15 or result["flops_improvement"] >= 0.25
    )
    (EXP_DIR / "efficiency.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def preflight(device_text: str, seed: int):
    device = torch.device(device_text)
    base = load_module("exp174d_efficiency_base", EFFICIENCY_BASE)
    adapter = load_module("exp174d_adapter_preflight", ENCODER_MODULE)
    model, cfg, trainer = base.build_model("F16", seed, device, 9)
    keys_before = set(model.state_dict())
    adapter.replace_encoder(model, cfg, ENCODER_EXPERIMENT)
    if keys_before != set(model.state_dict()):
        raise RuntimeError("Adapter changed checkpoint keys")
    past, valid = base.real_batch(trainer, cfg, 2, device)
    ema_compatible_copy = copy.deepcopy(model)
    with torch.inference_mode():
        prediction = base.forward(model, past, valid)
        copied_prediction = base.forward(ema_compatible_copy, past, valid)
    encoder = model.agent_aware_trans_encoder
    if (getattr(encoder, "last_social_kv_steps", None) != 3
            or not torch.isfinite(prediction).all()
            or not torch.equal(prediction, copied_prediction)):
        raise RuntimeError("EXP-174D forward preflight failed")
    print(json.dumps({"status": "preflight_ok", "output_shape": list(prediction.shape),
                      "social_kv_steps": encoder.last_social_kv_steps,
                      "state_dict_unchanged": True, "ema_deepcopy_compatible": True}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="EXP-174D official five-fold Q3/KV3 experiment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--efficiency-warmup", type=int, default=100)
    parser.add_argument("--efficiency-iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = [PYTHON, TRAINER, CONFIG, AUDIT, H3_MODULE, V2_MODULE, ENCODER_MODULE]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("\n".join(missing))
    if not json.loads(AUDIT.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Official protocol audit did not pass")
    preflight(args.device, args.seed)
    if args.preflight_only: return
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    for fold in folds:
        for stage in ("h3", "v2"):
            run_stage(args, fold, stage, args.seed)
        if not args.dry_run: summarize(args.seed)
    if args.dry_run:
        print("[done] dry-run passed")
    else:
        payload = summarize(args.seed)
        if len(payload["completed_folds"]) == 5:
            efficiency = efficiency_audit(
                args.device, args.seed, args.efficiency_warmup, args.efficiency_iterations
            )
            accuracy = payload.get("accuracy_gates", {})
            final_decision = {
                "efficiency_gate": efficiency["efficiency_gate"],
                "J_gate": accuracy.get("J_degradation_le_0p5pct", False),
                "minFDE_gate": accuracy.get("minFDE_degradation_le_1pct", False),
            }
            final_decision["passed"] = all(final_decision.values())
            payload["efficiency"] = efficiency
            payload["final_decision"] = final_decision
            (EXP_DIR / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"[done] completed_folds={payload['completed_folds']}")


if __name__ == "__main__":
    main()
