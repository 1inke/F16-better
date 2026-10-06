"""Validation-only paired analysis for the RGDM generation/selection 2x2 factorial."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_data_train.npy"
NUM = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_num_train.npy"
C00_DIR = (
    ROOT / "QUALITY_AWARE_F16_SERIES_V2" / "results" / "p0_protocol"
    / "C0_EXP173_EXACT" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
)
C00 = C00_DIR / "best_selected.pth"
H3 = (
    ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1" / "best_selected.pth"
)
TRAINING_ROOT = HERE / "results" / "training"
OUT = HERE / "results" / "analysis"
VIEWS = {
    "best_selection": "best_original_ade.pth",
    "best_generation": "best_selected.pth",
    "final": "final_model.pth",
}
CONDITIONS = {
    "c00_g0_s0": None,
    "s_only_g0_s1": "rgdm_b4_full",
    "g_only_g1_s0": "rgdm_g4_generation",
    "joint_g1_s1": "rgdm_ab_joint",
}
METRICS = (
    "selected_ade", "selected_fde", "minade", "minfde", "samepath_fde",
    "regret", "support_1m", "recall_at_1", "recall_at_3", "endpoint_spread",
    "duplicate_sample_rate", "j_oracle",
)
LOWER_IS_BETTER = {
    "selected_ade", "selected_fde", "minade", "minfde", "samepath_fde",
    "regret", "endpoint_spread", "duplicate_sample_rate", "j_oracle",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup(device: torch.device):
    os.chdir(ASTRA)
    sys.path[:0] = [str(ASTRA), str(ASTRA / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES"), str(HERE)]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg_path = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
    cfg = load_locked_q20(cfg_path)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    front = load_module("rgdm_analysis_front", ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py")
    trainer = load_module("rgdm_analysis_trainer", ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    base = load_module("rgdm_analysis_base", ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py")
    rgdm = load_module("rgdm_analysis_models", HERE / "rgdm_models.py")
    return cfg, front, trainer, base, rgdm


def treatment_run_dir(condition: str, variant: str) -> Path:
    return TRAINING_ROOT / condition / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


def checkpoint(condition: str, view: str) -> Path:
    variant = CONDITIONS[condition]
    if variant is None:
        # The factorial reference is the exact C00 source of every treatment,
        # not another historical C00 checkpoint with different geometry.
        return C00
    return treatment_run_dir(condition, variant) / VIEWS[view]


def audit_frozen_c00_parameters() -> dict:
    """Prove that every non-RGDM tensor still equals the locked C00 source."""
    baseline = torch.load(C00, map_location="cpu", weights_only=True)
    allowed_extra = ("decoder.generation_rgdm.", "decoder.selection_rgdm.")
    audited = {}
    for condition, variant in list(CONDITIONS.items())[1:]:
        audited[condition] = {}
        for view, filename in VIEWS.items():
            path = treatment_run_dir(condition, variant) / filename
            treatment = torch.load(path, map_location="cpu", weights_only=True)
            missing = [key for key in baseline if key not in treatment]
            changed = [key for key, value in baseline.items()
                       if key in treatment and not torch.equal(value, treatment[key])]
            unexpected = [key for key in treatment
                          if key not in baseline and not key.startswith(allowed_extra)]
            if missing or changed or unexpected:
                raise RuntimeError(
                    f"Frozen C00 provenance failed for {condition}/{view}: "
                    f"missing={missing[:3]}, changed={changed[:3]}, unexpected={unexpected[:3]}"
                )
            audited[condition][view] = {
                "checkpoint_sha256": sha256(path),
                "shared_tensors_exact": len(baseline),
                "changed_shared_tensors": 0,
            }
    return audited


def audit_inputs() -> np.ndarray:
    required = [C00, H3, DATA, NUM, C00_DIR / "split_indices.npz", HERE / "preflight.json"]
    for condition, variant in CONDITIONS.items():
        if variant is None:
            continue
        run = treatment_run_dir(condition, variant)
        required.extend([run / "metrics.json", run / "split_indices.npz"])
        required.extend(run / name for name in VIEWS.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Analysis inputs incomplete:\n" + "\n".join(missing))
    with np.load(C00_DIR / "split_indices.npz", allow_pickle=False) as payload:
        reference = np.asarray(payload["validation"], dtype=np.int64)
    for condition, variant in CONDITIONS.items():
        if variant is None:
            continue
        run = treatment_run_dir(condition, variant)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("evaluation_test_best_selected") is not None:
            raise RuntimeError(f"Test result present in new condition {condition}")
        if metrics["split"].get("target_test_accessed", True):
            raise RuntimeError(f"Test-access flag invalid for {condition}")
        with np.load(run / "split_indices.npz", allow_pickle=False) as payload:
            current = np.asarray(payload["validation"], dtype=np.int64)
        if not np.array_equal(reference, current):
            raise RuntimeError(f"Validation IDs differ for {condition}")
    return reference


def build_model(condition: str, view: str, cfg, front, base, rgdm, device: torch.device):
    local_cfg = copy.deepcopy(cfg)
    model = front.FrontEncoderASTRA(local_cfg, "f16_temporal_social_cross")
    variant = CONDITIONS[condition]
    if variant is None:
        os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(H3)
        os.environ["EXP173_DECODER_INIT_SEED"] = "1"
        base.replace_decoder(model, local_cfg, "v2_joint_wide_endpoint_correction")
    else:
        os.environ["RGDM_SOURCE_CHECKPOINT"] = str(C00)
        os.environ["RGDM_GATE_LOSS_WEIGHT"] = "0"
        os.environ["RGDM_RANK_LOSS_WEIGHT"] = "0"
        rgdm.replace_decoder(model, local_cfg, variant)
    path = checkpoint(condition, view)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), path


def motion_change_score(past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    observed = past[:, 0, -1] - past[:, 0, -2]
    future_mean = (future[:, 0, -1] - past[:, 0, -1]) / float(future.shape[-2])
    observed_speed = torch.linalg.vector_norm(observed, dim=-1)
    future_speed = torch.linalg.vector_norm(future_mean, dim=-1)
    product = observed_speed * future_speed
    cosine = (observed * future_mean).sum(dim=-1) / product.clamp_min(1.0e-6)
    cosine = torch.where(product > 1.0e-6, cosine.clamp(-1.0, 1.0), torch.ones_like(cosine))
    turn = ((1.0 - cosine) / 2.0).clamp(0.0, 1.0)
    speed = (
        torch.abs(torch.log((future_speed + 0.05) / (observed_speed + 0.05))) / math.log(2.0)
    ).clamp(0.0, 1.0)
    return torch.maximum(turn, speed)


def focal_scalar(value: torch.Tensor, batch: int) -> np.ndarray:
    if value.ndim == 0:
        value = value.expand(batch)
    elif value.shape[0] == batch:
        if value.ndim >= 2:
            value = value[:, 0]
        if value.ndim > 1:
            value = value.flatten(1).mean(dim=1)
    else:
        raise ValueError(f"Cannot map diagnostic shape {tuple(value.shape)} to batch {batch}")
    return value.detach().cpu().numpy().astype(np.float32)


def infer(model, loader, validation_ids: np.ndarray, trainer, device: torch.device) -> dict:
    ade_parts, fde_parts, logits_parts, endpoint_parts = [], [], [], []
    change_parts = []
    diagnostic_names = (
        "selection_change", "generation_change", "history_confidence",
        "fusion_lambda", "score_residual_abs",
    )
    diagnostics = {name: [] for name in diagnostic_names}
    offset = 0
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            batch = len(past)
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            change_parts.append(motion_change_score(past, future).cpu().numpy())
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past
            )[2] + center.unsqueeze(-3)
            candidates = prediction[:, 0]
            errors = torch.linalg.vector_norm(candidates - future[:, 0, None], dim=-1)
            ade_parts.append(errors.mean(dim=-1).cpu().numpy())
            fde_parts.append(errors[..., -1].cpu().numpy())
            logits_parts.append(model.last_mode_logits[:, 0].cpu().numpy())
            endpoint_parts.append(candidates[..., -1, :].cpu().numpy())
            batch_diag = {name: np.full(batch, np.nan, dtype=np.float32) for name in diagnostic_names}
            stored = getattr(model.decoder, "last_rgdm_diagnostics", {})
            if "selection" in stored:
                values = stored["selection"]
                batch_diag["selection_change"] = focal_scalar(values["change_probability"], batch)
                batch_diag["history_confidence"] = focal_scalar(values["history_confidence"], batch)
                batch_diag["score_residual_abs"] = focal_scalar(values["score_residual"].abs(), batch)
            if "generation" in stored:
                values = stored["generation"]
                batch_diag["generation_change"] = focal_scalar(values["change_probability"], batch)
                batch_diag["fusion_lambda"] = focal_scalar(values["local_lambda"], batch)
            for name in diagnostic_names:
                diagnostics[name].append(batch_diag[name])
            offset += batch
    if offset != len(validation_ids):
        raise RuntimeError(f"Inference count mismatch: {offset} != {len(validation_ids)}")
    result = {
        "sample_id": validation_ids.copy(),
        "ade": np.concatenate(ade_parts).astype(np.float32),
        "fde": np.concatenate(fde_parts).astype(np.float32),
        "logits": np.concatenate(logits_parts).astype(np.float32),
        "endpoints": np.concatenate(endpoint_parts).astype(np.float32),
        "motion_change_score": np.concatenate(change_parts).astype(np.float32),
    }
    result.update({name: np.concatenate(parts) for name, parts in diagnostics.items()})
    return result


def per_sample(cache: dict) -> Dict[str, np.ndarray]:
    ade, fde, logits, endpoints = cache["ade"], cache["fde"], cache["logits"], cache["endpoints"]
    rows = np.arange(len(ade))
    selected = logits.argmax(axis=1)
    oracle = ade.argmin(axis=1)
    order = np.argsort(-logits, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[rows[:, None], order] = np.arange(logits.shape[1])[None]
    pairwise = np.linalg.norm(endpoints[:, :, None] - endpoints[:, None, :], axis=-1)
    upper = np.triu(np.ones(pairwise.shape[1:], dtype=bool), k=1)
    upper_values = pairwise[:, upper]
    minade = ade.min(axis=1)
    minfde = fde.min(axis=1)
    values = {
        "selected_ade": ade[rows, selected],
        "selected_fde": fde[rows, selected],
        "minade": minade,
        "minfde": minfde,
        "samepath_fde": fde[rows, oracle],
        "regret": ade[rows, selected] - minade,
        "support_1m": (ade < 1.0).sum(axis=1).astype(np.float32),
        "recall_at_1": (rank[rows, oracle] < 1).astype(np.float32),
        "recall_at_3": (rank[rows, oracle] < 3).astype(np.float32),
        "endpoint_spread": upper_values.mean(axis=1),
        "duplicate_sample_rate": (upper_values < 0.10).any(axis=1).astype(np.float32),
        "j_oracle": minade + 0.5 * minfde,
    }
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def masks(change: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(len(change), dtype=bool),
        "stable": change < 0.25,
        "transition": (change >= 0.25) & (change < 0.50),
        "change": change >= 0.50,
    }


def summarize(values: Dict[str, np.ndarray], mask: np.ndarray) -> dict:
    if not mask.any():
        raise ValueError("Empty subgroup")
    return {name: float(array[mask].mean()) for name, array in values.items()}


def paired_bootstrap_factorial(
    cell_values: Dict[str, Dict[str, np.ndarray]], mask: np.ndarray, repetitions: int, seed: int
) -> Iterable[dict]:
    names = list(METRICS)
    cell_order = list(CONDITIONS)
    matrix = np.stack(
        [np.stack([cell_values[cell][metric][mask] for metric in names], axis=1) for cell in cell_order],
        axis=1,
    )  # [N,4,M]
    n = len(matrix)
    rng = np.random.default_rng(seed)
    collected = {effect: [] for effect in ("selection_main", "generation_main", "interaction")}
    chunk = 200
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        indices = rng.integers(0, n, size=(size, n))
        means = matrix[indices].mean(axis=1)  # [R,4,M]
        c00, selection, generation, joint = (means[:, index] for index in range(4))
        collected["selection_main"].append(0.5 * ((selection - c00) + (joint - generation)))
        collected["generation_main"].append(0.5 * ((generation - c00) + (joint - selection)))
        collected["interaction"].append(joint - generation - selection + c00)
    point = matrix.mean(axis=0)
    point_effects = {
        "selection_main": 0.5 * ((point[1] - point[0]) + (point[3] - point[2])),
        "generation_main": 0.5 * ((point[2] - point[0]) + (point[3] - point[1])),
        "interaction": point[3] - point[2] - point[1] + point[0],
    }
    for effect, chunks in collected.items():
        samples = np.concatenate(chunks, axis=0)
        lower, upper = np.quantile(samples, (0.025, 0.975), axis=0)
        for index, metric in enumerate(names):
            yield {
                "effect": effect,
                "metric": metric,
                "estimate": float(point_effects[effect][index]),
                "ci_lower": float(lower[index]),
                "ci_upper": float(upper[index]),
                "preferred_direction": "negative" if metric in LOWER_IS_BETTER else "positive",
                "n": int(n),
                "repetitions": int(repetitions),
            }


def relative(new: float, old: float) -> float:
    return (new - old) / old


def decision(summary: dict) -> dict:
    def row(view: str, condition: str) -> dict:
        return summary[view][condition]["all"]

    c_sel, s_best, s_final = row("best_selection", "c00_g0_s0"), row("best_selection", "s_only_g0_s1"), row("final", "s_only_g0_s1")
    c_gen, g_best, g_final = row("best_generation", "c00_g0_s0"), row("best_generation", "g_only_g1_s0"), row("final", "g_only_g1_s0")
    j_sel, j_gen, j_final = row("best_selection", "joint_g1_s1"), row("best_generation", "joint_g1_s1"), row("final", "joint_g1_s1")
    gates = {
        "selection_best_gain_ge_1pct": relative(s_best["selected_ade"], c_sel["selected_ade"]) <= -0.01,
        "selection_final_same_direction": s_final["selected_ade"] < c_sel["selected_ade"],
        "selection_fde_guard": s_best["selected_fde"] <= c_sel["selected_fde"] * 1.005,
        "generation_best_j_gain_ge_2pct": relative(g_best["j_oracle"], c_gen["j_oracle"]) <= -0.02,
        "generation_final_same_direction": g_final["j_oracle"] < c_gen["j_oracle"],
        "generation_selected_ade_guard": g_best["selected_ade"] <= c_gen["selected_ade"] * 1.005,
        "joint_selection_gain": j_sel["selected_ade"] < c_sel["selected_ade"],
        "joint_generation_gain": j_gen["j_oracle"] < c_gen["j_oracle"],
        "joint_final_not_worse": (
            j_final["selected_ade"] <= c_sel["selected_ade"] * 1.005
            and j_final["j_oracle"] <= c_gen["j_oracle"] * 1.005
        ),
    }
    selection = all(gates[name] for name in (
        "selection_best_gain_ge_1pct", "selection_final_same_direction", "selection_fde_guard"
    ))
    generation = all(gates[name] for name in (
        "generation_best_j_gain_ge_2pct", "generation_final_same_direction", "generation_selected_ade_guard"
    ))
    joint = all(gates[name] for name in (
        "joint_selection_gain", "joint_generation_gain", "joint_final_not_worse"
    ))
    if selection and generation and joint:
        verdict = "BOTH_BRANCHES_PROMISING_REQUIRES_SEED_CONFIRMATION"
    elif selection and not generation:
        verdict = "SELECTION_ONLY_PROMISING"
    elif generation and not selection:
        verdict = "GENERATION_ONLY_PROMISING"
    elif joint and (selection or generation):
        verdict = "JOINT_BENEFIT_WITH_PARTIAL_MAIN_EFFECT"
    else:
        verdict = "STOP_OR_REDESIGN_AFTER_SEED1"
    return {"verdict": verdict, "gates": gates, "selection_pass": selection,
            "generation_pass": generation, "joint_pass": joint}


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    validation_ids = audit_inputs()
    frozen_audit = audit_frozen_c00_parameters()
    cfg, front, trainer, base, rgdm = setup(device)
    dataset = trainer.TrajectoryDataset(DATA, NUM, cfg)
    loader = DataLoader(Subset(dataset, validation_ids.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    OUT.mkdir(parents=True, exist_ok=True)

    caches: Dict[str, Dict[str, dict]] = {view: {} for view in VIEWS}
    baseline_model, baseline_path = build_model("c00_g0_s0", "best_generation", cfg, front, base, rgdm, device)
    baseline_cache = infer(baseline_model, loader, validation_ids, trainer, device)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for view in VIEWS:
        caches[view]["c00_g0_s0"] = baseline_cache
    for view in VIEWS:
        for condition in list(CONDITIONS)[1:]:
            model, path = build_model(condition, view, cfg, front, base, rgdm, device)
            print(f"[infer] {view} / {condition}: {path}", flush=True)
            caches[view][condition] = infer(model, loader, validation_ids, trainer, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    cache_root = OUT / "per_sample_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for view, view_caches in caches.items():
        for condition, cache in view_caches.items():
            np.savez_compressed(cache_root / f"{view}__{condition}.npz", **cache)

    # The selection-only arm must be candidate-identical to its C00 source.
    identity = {}
    for view in VIEWS:
        treatment = caches[view]["s_only_g0_s1"]
        error = max(
            float(np.max(np.abs(treatment[key] - baseline_cache[key])))
            for key in ("ade", "fde", "endpoints")
        )
        identity[view] = error
        # Full validation inference on this Windows/CUDA stack can differ from
        # the cached C00 path by about 2.9e-6 even when the selection-only
        # branch provably leaves candidates unchanged. Keep the gate strict,
        # but above that observed floating-point floor.
        if error > 5.0e-6:
            raise RuntimeError(f"Selection-only candidate identity failed at {view}: {error}")

    change = baseline_cache["motion_change_score"]
    subgroup_masks = masks(change)
    summary: Dict[str, Dict[str, dict]] = {}
    per_values: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    condition_rows, effect_rows = [], []
    for view, view_caches in caches.items():
        summary[view], per_values[view] = {}, {}
        for condition, cache in view_caches.items():
            values = per_sample(cache)
            per_values[view][condition] = values
            summary[view][condition] = {}
            for subgroup, mask in subgroup_masks.items():
                stats = summarize(values, mask)
                summary[view][condition][subgroup] = stats
                condition_rows.append({"view": view, "condition": condition, "subgroup": subgroup,
                                       "n": int(mask.sum()), **stats})
        for subgroup, mask in subgroup_masks.items():
            rows = paired_bootstrap_factorial(
                per_values[view], mask, args.bootstrap_repetitions,
                args.bootstrap_seed + list(VIEWS).index(view) * 10 + list(subgroup_masks).index(subgroup),
            )
            for row in rows:
                effect_rows.append({"view": view, "subgroup": subgroup, **row})

    verdict = decision(summary)
    provenance = {
        "status": "ANALYSIS_COMPLETE_VALIDATION_ONLY",
        "baseline_checkpoint": str(baseline_path),
        "baseline_sha256": sha256(baseline_path),
        "validation_samples": int(len(validation_ids)),
        "validation_id_sha256": hashlib.sha256(validation_ids.tobytes()).hexdigest(),
        "selection_candidate_identity_max_error": identity,
        "selection_candidate_identity_tolerance": 5.0e-6,
        "frozen_c00_parameter_audit": frozen_audit,
        "motion_groups": {"stable": "score < 0.25", "transition": "0.25 <= score < 0.50",
                          "change": "score >= 0.50"},
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "sample_bootstrap_scope": "conditional on one trained model per cell; not training-seed uncertainty",
        "test_accessed": False,
    }
    write_csv(OUT / "condition_summary.csv", condition_rows)
    write_csv(OUT / "factorial_effects.csv", effect_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RGDM 生成侧 × 选择侧 2×2 验证报告", "", f"状态：`{verdict['verdict']}`", "",
        "本报告仅使用锁定的 ETH seed-1 scene-tail validation；没有访问 test。样本 bootstrap 只度量固定模型下的配对样本不确定性，不代表训练种子不确定性。",
        "", "## 全体验证指标", "",
    ]
    for view in VIEWS:
        lines += [f"### {view}", "", "| 条件 | Selected ADE | Selected FDE | minADE | minFDE | J | Regret | R@1 | R@3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for condition in CONDITIONS:
            row = summary[view][condition]["all"]
            lines.append(
                f"| {condition} | {row['selected_ade']:.6f} | {row['selected_fde']:.6f} | "
                f"{row['minade']:.6f} | {row['minfde']:.6f} | {row['j_oracle']:.6f} | "
                f"{row['regret']:.6f} | {row['recall_at_1']:.2%} | {row['recall_at_3']:.2%} |"
            )
        lines.append("")
    lines += [
        "## 判定门", "",
        *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in verdict["gates"].items()],
        "", "## 解释边界", "",
        "- `s_only_g0_s1` 必须与 C00 候选严格一致；脚本在分析前执行逐样本 identity gate。",
        "- `best_selection`、`best_generation` 与 `final` 三个视角必须共同审阅，不能只摘取有利 checkpoint。",
        "- motion-change 分组使用未来真值仅做验证后分析；它不进入模型推理输入。",
        "- 只有通过预注册门后才讨论 seed 2/3；脚本不会自动扩展到更多种子、折或 test。",
        "", "完整的配对区间见 `factorial_effects.csv`。",
    ]
    (OUT / "RGDM_FACTORIAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {OUT / 'RGDM_FACTORIAL_REPORT.md'}")


if __name__ == "__main__":
    main()
