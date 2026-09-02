"""EXP-176D: frozen SDD scene-compatible score audit.

No parameters are trained and no candidate coordinates are changed.  The
pre-registered correction is

    corrected_logit = original_logit - 0.10 * within_sample_score_std * violation

where violation is one minus the observation-conditioned appearance
similarity accepted by EXP-175A-R2.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_AUDIT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
R2_AUDIT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE" / "run_exp175ar2_appearance_audit.py"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
BANK = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175AR2_OBSERVATION_CONDITIONED_APPEARANCE" / "assets" / "appearance_bank.npz"
FIXED_VARIANT = "appearance_similarity"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect(split, model, cfg, trainer, base, r2, bank, device, batch_size, max_samples):
    loader = base.make_loader(split, cfg, trainer, batch_size, max_samples)
    pieces: dict[str, list[np.ndarray]] = {}

    def save(name: str, value: torch.Tensor) -> None:
        pieces.setdefault(name, []).append(value.detach().cpu().numpy())

    with torch.inference_mode():
        for absolute_past, absolute_future, num_valid, scene_ids, *_ in loader:
            absolute_past = absolute_past.to(device, non_blocking=True)
            absolute_future = absolute_future.to(device, non_blocking=True)
            num_valid = num_valid.to(device, non_blocking=True).int()
            scene_ids = scene_ids.to(device, non_blocking=True).long()
            past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past
            )
            prediction = (output[2] + center.unsqueeze(-3))[:, 0]
            target = future[:, 0, None]
            distance = torch.linalg.vector_norm(prediction - target, dim=-1)
            ade, fde = distance.mean(dim=-1), distance[..., -1]
            logits = model.last_mode_logits[:, 0]
            global_prediction = base.local_to_global(prediction, absolute_past)
            candidate_values, candidate_inside = bank.query(global_prediction, scene_ids)
            prototype_values, prototype_inside = bank.query(absolute_past, scene_ids)
            valid_agents = torch.arange(absolute_past.shape[1], device=device)[None] < num_valid[:, None]
            prototype_mask = prototype_inside & valid_agents[..., None]
            feasibility = r2.appearance_scores(
                candidate_values, candidate_inside, prototype_values, prototype_mask
            )[FIXED_VARIANT]
            violation = 1.0 - feasibility
            score_std = logits.std(dim=-1, unbiased=False).clamp_min(1e-8)
            correction = 0.10 * score_std[:, None] * violation
            corrected_logits = logits - correction
            save("scene_id", scene_ids)
            save("ade", ade); save("fde", fde); save("logits", logits)
            save("corrected_logits", corrected_logits)
            save("feasibility", feasibility); save("violation", violation)
            save("score_std", score_std); save("correction", correction)
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def selected_metrics(data: dict[str, np.ndarray], selection: np.ndarray) -> dict[str, float | int]:
    rows = np.arange(len(selection))
    ade, fde = data["ade"], data["fde"]
    quality = ade + 0.25 * fde
    selected_quality = quality[rows, selection]
    regret = selected_quality - quality.min(axis=1)
    return {
        "samples": int(len(selection)),
        "top1_ADE": float(ade[rows, selection].mean()),
        "top1_FDE": float(fde[rows, selection].mean()),
        "top1_combined": float(selected_quality.mean()),
        "mean_regret": float(regret.mean()),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "combined_winner_hit": float((selection == quality.argmin(axis=1)).mean()),
        "selected_scene_violation": float(data["violation"][rows, selection].mean()),
        "active_modes": int(np.unique(selection).size),
    }


def oracle_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    ade, fde = data["ade"], data["fde"]
    best_ade = ade.argmin(axis=1)
    rows = np.arange(len(ade))
    return {
        "minADE20": float(ade.min(axis=1).mean()),
        "minFDE20": float(fde.min(axis=1).mean()),
        "same_path_FDE": float(fde[rows, best_ade].mean()),
    }


def improvement(original: dict, corrected: dict, key: str) -> float:
    return (float(original[key]) - float(corrected[key])) / max(float(original[key]), 1e-12)


def analyze(data: dict[str, np.ndarray]) -> dict:
    original_selection = data["logits"].argmax(axis=1)
    corrected_selection = data["corrected_logits"].argmax(axis=1)
    original = selected_metrics(data, original_selection)
    corrected = selected_metrics(data, corrected_selection)
    return {
        "original": original,
        "scene_compatible": corrected,
        "relative_improvement": {
            key: improvement(original, corrected, key)
            for key in ("top1_ADE", "top1_FDE", "top1_combined", "mean_regret", "p90_regret", "p95_regret", "selected_scene_violation")
        },
        "selection_flip_rate": float((original_selection != corrected_selection).mean()),
        "mean_score_std": float(data["score_std"].mean()),
        "mean_abs_score_correction": float(data["correction"].mean()),
        "max_abs_score_correction": float(data["correction"].max()),
        "oracle": oracle_metrics(data),
        "candidate_set_changed": False,
    }


def validation_gate(result: dict) -> dict[str, bool]:
    gain = result["relative_improvement"]
    return {
        "combined_cost_improves": gain["top1_combined"] > 0.0,
        "mean_regret_improves": gain["mean_regret"] > 0.0,
        "p95_regret_not_worse_2pct": gain["p95_regret"] >= -0.02,
        "scene_violation_improves": gain["selected_scene_violation"] > 0.0,
        "nonzero_selection_effect": result["selection_flip_rate"] > 0.0,
    }


def per_video_rows(data: dict[str, np.ndarray], bank) -> list[dict]:
    rows = []
    ids = data["scene_id"].astype(int)
    for scene_id in sorted(np.unique(ids)):
        mask = ids == scene_id
        subset = {key: value[mask] for key, value in data.items()}
        result = analyze(subset)
        original, corrected = result["original"], result["scene_compatible"]
        rows.append({
            "scene_id": scene_id, "video": bank.video_names[scene_id], "samples": int(mask.sum()),
            "original_top1_ADE": original["top1_ADE"], "corrected_top1_ADE": corrected["top1_ADE"],
            "ADE_improvement": improvement(original, corrected, "top1_ADE"),
            "original_top1_FDE": original["top1_FDE"], "corrected_top1_FDE": corrected["top1_FDE"],
            "FDE_improvement": improvement(original, corrected, "top1_FDE"),
            "combined_improvement": improvement(original, corrected, "top1_combined"),
            "mean_regret_improvement": improvement(original, corrected, "mean_regret"),
            "p95_regret_improvement": improvement(original, corrected, "p95_regret"),
            "scene_violation_improvement": improvement(original, corrected, "selected_scene_violation"),
            "selection_flip_rate": result["selection_flip_rate"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-176D fixed scene-compatible score audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "seed_1")
    args = parser.parse_args()
    for path in (BASE_AUDIT, R2_AUDIT, BANK, S0 / "summary.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    base = load("exp176d_base", BASE_AUDIT)
    r2 = load("exp176d_r2", R2_AUDIT)
    model, cfg, trainer = base.build_model(args.seed, device)
    bank = r2.AppearanceBank(BANK, device)
    max_samples = min(args.max_samples or 32, 32) if args.preflight_only else args.max_samples

    print("[collect] validation", flush=True)
    validation_data = collect(
        "validation", model, cfg, trainer, base, r2, bank, device, args.batch_size, max_samples
    )
    validation = analyze(validation_data)
    gate = validation_gate(validation)
    gate_pass = all(gate.values())
    payload = {
        "experiment": "EXP-176D",
        "training": False,
        "candidate_generator": "frozen EXP-175S0 F16+H3+Joint-V2 seed1",
        "appearance_variant": FIXED_VARIANT,
        "score_rule": "original_logit - 0.10 * within_sample_score_std * (1 - appearance_similarity)",
        "lambda_selected_without_test": True,
        "future_used_for_scoring": False,
        "validation": validation,
        "validation_gate": gate,
        "validation_gate_pass": gate_pass,
    }
    if args.preflight_only:
        payload["status"] = "preflight_passed"
        payload["test_accessed"] = False
        out = HERE / "preflight.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return

    if gate_pass:
        print("[validation passed] collect test once", flush=True)
        test_data = collect(
            "test", model, cfg, trainer, base, r2, bank, device, args.batch_size, args.max_samples
        )
        test = analyze(test_data)
        video_rows = per_video_rows(test_data, bank)
        improved_videos = sum(row["combined_improvement"] > 0 for row in video_rows)
        test_gain = test["relative_improvement"]
        confirmation = {
            "combined_cost_improves": test_gain["top1_combined"] > 0.0,
            "mean_regret_improves": test_gain["mean_regret"] > 0.0,
            "p95_regret_not_worse_2pct": test_gain["p95_regret"] >= -0.02,
            "at_least_12_of_17_videos_improve": improved_videos >= 12,
        }
        payload.update({
            "test_accessed": True, "test": test,
            "test_improved_video_count": improved_videos,
            "test_video_count": len(video_rows),
            "test_confirmation": confirmation,
            "accepted": all(confirmation.values()),
        })
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "per_video.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(video_rows[0]))
            writer.writeheader(); writer.writerows(video_rows)
    else:
        payload.update({
            "test_accessed": False, "accepted": False,
            "recommendation": "Stop: fixed scene-compatible score did not pass validation direction gate",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation_gate_pass": gate_pass, "test_accessed": payload["test_accessed"], "accepted": payload["accepted"]}, indent=2))
    print(f"[result] {out}")


if __name__ == "__main__":
    main()

