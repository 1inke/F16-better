"""EXP-178A: zero-trajectory-training SDD topology reachability audit."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from generate_topology_proposals import generate_proposals
from topology_features import candidate_features, row_spearman


SERIES = Path(__file__).resolve().parent
ROOT = SERIES.parent
EXP_DIR = SERIES / "EXP_178A_TOPOLOGY_REACHABILITY"
ASSET_BUILDER = SERIES / "build_walkability_map.py"
DEFAULT_ASSETS = EXP_DIR / "assets"
EXP175A = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
PIXEL_SCALE = 20.0
FEATURE_NAMES = (
    "mean_skeleton_distance", "endpoint_skeleton_distance", "max_skeleton_deviation",
    "reachable_endpoint", "path_connectivity", "direction_consistency",
    "geodesic_path_ratio", "mean_walkability", "topology_consistency",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def load_bank(path: Path):
    raw = np.load(path, allow_pickle=False)
    bank = {key: raw[key] for key in raw.files}
    bank["video_names"] = [str(x) for x in bank["video_names"].tolist()]
    return bank


def ensure_assets(args):
    bank = args.asset_dir / "topology_bank.npz"
    if bank.is_file() and not args.rebuild_assets:
        return bank
    command = [
        sys.executable, "-u", str(ASSET_BUILDER), "--dataset-root", str(args.dataset_root),
        "--output", str(args.asset_dir), "--device", args.device,
        "--epochs", str(args.asset_epochs), "--seed", str(args.asset_seed),
    ]
    if args.rebuild_assets:
        command.append("--force")
    print("[assets] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(SERIES), check=True)
    return bank


def metric_bundle(ade: np.ndarray, fde: np.ndarray):
    best_ade = ade.argmin(axis=1); rows = np.arange(len(ade))
    return {
        "minADE_px": float(ade.min(axis=1).mean()),
        "minFDE_px": float(fde.min(axis=1).mean()),
        "same_path_FDE_px": float(fde[rows, best_ade].mean()),
    }


def relative_gain(original: float, updated: float):
    return 100.0 * (original - updated) / max(original, 1e-8)


def collect(model, cfg, trainer, helper, bank, device, args):
    loader = helper.make_loader("test", cfg, trainer, args.batch_size, args.max_samples)
    records, candidate_rows = [], []
    feature_matrices = {name: [] for name in FEATURE_NAMES}; qualities, scene_ids = [], []
    sample_index = 0
    with torch.inference_mode():
        for absolute_past, absolute_future, num_valid, batch_scene_ids, *_ in loader:
            absolute_past = absolute_past.to(device); absolute_future = absolute_future.to(device)
            num_valid = num_valid.to(device).int(); batch_scene_ids = batch_scene_ids.to(device).long()
            past, future = trainer.normalize_to_focal_heading(absolute_past, absolute_future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            prediction = output[2] if isinstance(output, (tuple, list)) else output
            prediction = (prediction + center.unsqueeze(-3))[:, 0]
            global_candidates = helper.local_to_global(prediction, absolute_past) * PIXEL_SCALE
            past_px = absolute_past[:, 0] * PIXEL_SCALE; future_px = absolute_future[:, 0] * PIXEL_SCALE
            candidates_np = global_candidates.cpu().numpy(); past_np = past_px.cpu().numpy()
            future_np = future_px.cpu().numpy(); ids_np = batch_scene_ids.cpu().numpy().astype(int)
            for index in range(len(candidates_np)):
                sid = int(ids_np[index]); candidates = candidates_np[index]; target = future_np[index]
                distance = np.linalg.norm(candidates - target[None], axis=-1)
                base_ade, base_fde = distance.mean(axis=-1), distance[:, -1]
                quality = base_ade + 0.25 * base_fde
                features = candidate_features(bank, sid, candidates, past_np[index])
                proposals, diagnostic = generate_proposals(bank, sid, past_np[index])
                if len(proposals):
                    topo_distance = np.linalg.norm(proposals - target[None], axis=-1)
                    topo_ade, topo_fde = topo_distance.mean(axis=-1), topo_distance[:, -1]
                    union_ade = np.r_[base_ade, topo_ade]; union_fde = np.r_[base_fde, topo_fde]
                else:
                    topo_ade = topo_fde = np.asarray([], np.float32)
                    union_ade, union_fde = base_ade, base_fde
                base_best = int(base_ade.argmin()); union_best = int(union_ade.argmin())
                record = {
                    "sample_index": sample_index, "scene_id": sid, "video": bank["video_names"][sid],
                    "baseline_minADE_px": float(base_ade.min()), "baseline_minFDE_px": float(base_fde.min()),
                    "baseline_same_path_FDE_px": float(base_fde[base_best]),
                    "topology_minADE_px": float(topo_ade.min()) if len(topo_ade) else None,
                    "topology_minFDE_px": float(topo_fde.min()) if len(topo_fde) else None,
                    "union_minADE_px": float(union_ade.min()), "union_minFDE_px": float(union_fde.min()),
                    "union_same_path_FDE_px": float(union_fde[union_best]),
                    "proposal_count": int(len(proposals)), **diagnostic,
                }
                records.append(record); qualities.append(quality); scene_ids.append(sid)
                for name in FEATURE_NAMES:
                    feature_matrices[name].append(features[name])
                for mode in range(len(candidates)):
                    row = {"sample_index": sample_index, "scene_id": sid, "video": bank["video_names"][sid],
                           "mode": mode, "ADE_px": float(base_ade[mode]), "FDE_px": float(base_fde[mode]),
                           "quality_px": float(quality[mode])}
                    row.update({name: float(features[name][mode]) for name in FEATURE_NAMES})
                    candidate_rows.append(row)
                sample_index += 1
            print(f"[audit] samples={sample_index}", flush=True)
    return records, candidate_rows, {key: np.stack(value) for key, value in feature_matrices.items()}, np.stack(qualities), np.asarray(scene_ids)


def aggregate(records, feature_matrices, quality, scene_ids, bank):
    groups = {"overall": np.ones(len(records), bool)}
    little = np.asarray([bank["video_names"][sid].startswith("little/") for sid in scene_ids])
    groups.update({"little": little, "non_little": ~little})
    per_video_masks = {bank["video_names"][sid]: scene_ids == sid for sid in sorted(set(scene_ids.tolist()))}

    def metrics(mask):
        selected = [records[x] for x in np.flatnonzero(mask)]
        result = {"samples": len(selected)}
        for family in ("baseline", "union"):
            for metric in ("minADE_px", "minFDE_px", "same_path_FDE_px"):
                result[f"{family}_{metric}"] = float(np.mean([row[f"{family}_{metric}"] for row in selected]))
        for metric in ("minADE_px", "minFDE_px", "same_path_FDE_px"):
            result[f"{metric}_improvement_percent"] = relative_gain(result[f"baseline_{metric}"], result[f"union_{metric}"])
        for name in FEATURE_NAMES:
            correlations = row_spearman(feature_matrices[name][mask], -quality[mask])
            result[f"spearman_{name}_neg_quality"] = float(np.nanmean(correlations))
        result["mean_proposal_count"] = float(np.mean([row["proposal_count"] for row in selected]))
        return result

    group_results = {name: metrics(mask) for name, mask in groups.items() if mask.any()}
    per_video = []
    for video, mask in per_video_masks.items():
        row = {"video": video}; row.update(metrics(mask)); per_video.append(row)
    positive = sum(
        row["minADE_px_improvement_percent"] > 0 or row["minFDE_px_improvement_percent"] > 0
        for row in per_video
    )
    overall, little_result = group_results["overall"], group_results.get("little", {})
    oracle_gate = overall["minADE_px_improvement_percent"] >= 5 or overall["minFDE_px_improvement_percent"] >= 5
    video_gate = positive >= 12
    little_gate = little_result.get("minADE_px_improvement_percent", -float("inf")) >= 8 or little_result.get("minFDE_px_improvement_percent", -float("inf")) >= 8
    passed = bool(oracle_gate and video_gate and little_gate)
    return group_results, per_video, {
        "overall_oracle_gain_at_least_5_percent": oracle_gate,
        "positive_test_videos": positive, "positive_test_videos_at_least_12_of_17": video_gate,
        "little_gain_at_least_8_percent": little_gate, "passed": passed,
        "decision": "PROCEED_TO_EXP178B" if passed else "STOP_SCENE_TOPOLOGY_TRAINING",
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def write_analysis(path, summary):
    overall = summary["groups"]["overall"]; gate = summary["gate"]
    text = f"""# EXP-178A result analysis

## Decision

**{gate['decision']}** (passed: `{str(gate['passed']).lower()}`).

## SDD test result

| Metric | Baseline K20 | K20 + topology | Relative improvement |
|---|---:|---:|---:|
| minADE | {overall['baseline_minADE_px']:.4f} px | {overall['union_minADE_px']:.4f} px | {overall['minADE_px_improvement_percent']:.2f}% |
| minFDE | {overall['baseline_minFDE_px']:.4f} px | {overall['union_minFDE_px']:.4f} px | {overall['minFDE_px_improvement_percent']:.2f}% |
| same-path FDE | {overall['baseline_same_path_FDE_px']:.4f} px | {overall['union_same_path_FDE_px']:.4f} px | {overall['same_path_FDE_px_improvement_percent']:.2f}% |

- Positive test videos: {gate['positive_test_videos']}/17.
- Little subset gate: {gate['little_gain_at_least_8_percent']}.
- The topology assets never use validation/test trajectory labels; test futures are accessed only here for final audit metrics.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EXP-178A topological scene reachability audit")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset"))
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--asset-epochs", type=int, default=60); parser.add_argument("--asset-seed", type=int, default=178)
    parser.add_argument("--rebuild-assets", action="store_true"); parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(); args.asset_dir = args.asset_dir.resolve()
    if args.preflight_only and args.max_samples == 0:
        args.max_samples = 8
    bank_path = ensure_assets(args); bank = load_bank(bank_path)
    helper = load_module("exp178_helper", EXP175A)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device); helper.verify_inputs(args.seed)
    model, cfg, trainer = helper.build_model(args.seed, device)
    records, candidates, features, quality, scene_ids = collect(model, cfg, trainer, helper, bank, device, args)
    groups, per_video, gate = aggregate(records, features, quality, scene_ids, bank)
    summary = {
        "experiment": "EXP-178A", "model": "frozen S0 F16 + H3 + Joint-V2, seed1, K20",
        "trajectory_training": False, "dataset": "standard SDD test, 17 videos",
        "topology_source": "offline <10K CNN trained only with training-video pedestrian tracks",
        "validation_or_test_tracks_used_for_topology": False, "test_future_used_only_for_final_metrics": True,
        "asset_contract": json.loads((args.asset_dir / "asset_contract.json").read_text(encoding="utf-8")),
        "groups": groups, "gate": gate,
    }
    output = EXP_DIR / ("preflight" if args.preflight_only else "results")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "per_video.csv", per_video); write_csv(output / "candidate_topology.csv", candidates)
    write_csv(output / "union_metrics.csv", records); write_analysis(output / "RESULT_ANALYSIS.md", summary)
    print(json.dumps({"status": "preflight_ok" if args.preflight_only else "complete", "samples": len(records), **gate}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
