"""EXP-165: leakage-aware official-fold shift and candidate geometry audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DATA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse")
CACHE = WORKSPACE / "CANDIDATE_COMPLEMENTARITY_SERIES" / "EXP_158_CANDIDATE_COMPLEMENTARITY_AUDIT" / "per_sample"
DEFAULT_OUT = HERE / "EXP_165_OFFICIAL_FOLD_SHIFT_CANDIDATE_AUDIT"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
FAMILIES = ("k20", "h3", "v2")
OBS = 8


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict): return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)): return None
    if isinstance(value, np.generic): return value.item()
    return value


def finite_mean(x: np.ndarray) -> float:
    finite = np.asarray(x, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def finite_std(x: np.ndarray) -> float:
    finite = np.asarray(x, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.std(ddof=1)) if len(finite) > 1 else 0.0


def smd(train: np.ndarray, test: np.ndarray) -> float:
    a = np.asarray(train, dtype=np.float64); b = np.asarray(test, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((b.mean() - a.mean()) / pooled) if pooled > 1e-12 else 0.0


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y); x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def observation_features(positions: np.ndarray, num_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Positions [N,A,8,2]; all features are observation-only."""
    focal = positions[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=1)
    acc_norm = np.linalg.norm(acceleration, axis=-1)
    v0, v1 = velocity[:, :-1], velocity[:, 1:]
    denom = np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1)
    cosine = np.sum(v0 * v1, axis=-1) / np.maximum(denom, 1e-8)
    turns = np.arccos(np.clip(cosine, -1.0, 1.0)); turns[denom < 1e-8] = 0.0
    start, end = focal[:, 0], focal[:, -1]
    chord = end - start; chord_norm = np.linalg.norm(chord, axis=-1)
    rel = focal - start[:, None]
    cross = np.abs(rel[..., 0] * chord[:, None, 1] - rel[..., 1] * chord[:, None, 0])
    nonlinearity = np.mean(cross / np.maximum(chord_norm[:, None], 1e-6), axis=1)
    last = positions[:, :, -1]
    distance = np.linalg.norm(last - last[:, :1], axis=-1)
    valid = np.arange(positions.shape[1])[None] < num_valid[:, None]
    neighbor = valid.copy(); neighbor[:, 0] = False
    masked = np.where(neighbor, distance, np.nan)
    nearest = np.full(len(positions), np.nan, dtype=np.float64)
    mean_distance = np.full(len(positions), np.nan, dtype=np.float64)
    has_neighbor = neighbor.any(axis=1)
    nearest[has_neighbor] = np.nanmin(masked[has_neighbor], axis=1)
    mean_distance[has_neighbor] = np.nanmean(masked[has_neighbor], axis=1)
    return {
        "mean_step_speed": speed.mean(axis=1),
        "median_step_speed": np.median(speed, axis=1),
        "last_step_speed": speed[:, -1],
        "obs_displacement": chord_norm,
        "mean_acceleration": acc_norm.mean(axis=1),
        "max_acceleration": acc_norm.max(axis=1),
        "mean_turn_angle": turns.mean(axis=1),
        "observation_nonlinearity": nonlinearity,
        "valid_agent_count": num_valid.astype(np.float64),
        "nearest_neighbor_distance": nearest,
        "neighbor_count_r2": ((distance < 2.0) & neighbor).sum(axis=1).astype(np.float64),
        "neighbor_count_r4": ((distance < 4.0) & neighbor).sum(axis=1).astype(np.float64),
        "neighbor_count_r6": ((distance < 6.0) & neighbor).sum(axis=1).astype(np.float64),
        "mean_neighbor_distance": mean_distance,
    }


def train_features(fold: str) -> dict[str, np.ndarray]:
    raw = np.load(DATA / f"{fold}_data_train.npy", mmap_mode="r")
    count = np.load(DATA / f"{fold}_num_train.npy", mmap_mode="r")
    positions = np.asarray(raw[:, :, :OBS, -2:], dtype=np.float32)
    return observation_features(positions, np.asarray(count))


def greedy_effective_count(path_distance: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    result = np.ones(len(path_distance), dtype=np.float32)
    for sample in range(len(path_distance)):
        chosen = [0]
        for mode in range(1, path_distance.shape[1]):
            if all(path_distance[sample, mode, previous] >= threshold for previous in chosen):
                chosen.append(mode)
        result[sample] = len(chosen)
    return result


def _candidate_metrics_chunk(candidates: np.ndarray, future: np.ndarray, past_last: np.ndarray) -> dict[str, np.ndarray]:
    error = np.linalg.norm(candidates - future[:, None], axis=-1)
    ade = error.mean(axis=-1); fde = error[..., -1]
    winner = ade.argmin(axis=1)
    same_fde = fde[np.arange(len(fde)), winner]
    endpoint = candidates[:, :, -1] - past_last[:, None]
    radius = np.linalg.norm(endpoint, axis=-1)
    pair_endpoint = np.linalg.norm(endpoint[:, :, None] - endpoint[:, None, :], axis=-1)
    pair_path = np.linalg.norm(candidates[:, :, None] - candidates[:, None, :], axis=-1).mean(axis=-1)
    upper = np.triu_indices(candidates.shape[1], k=1)
    endpoint_pairs = pair_endpoint[:, upper[0], upper[1]]
    path_pairs = pair_path[:, upper[0], upper[1]]
    gt_radius = np.linalg.norm(future[:, -1] - past_last, axis=-1)
    coverage_ratio = np.full_like(gt_radius, np.nan, dtype=np.float32)
    np.divide(radius.max(axis=1), gt_radius, out=coverage_ratio, where=gt_radius > .10)
    return {
        "minADE": ade.min(axis=1), "minFDE": fde.min(axis=1), "same_path_FDE": same_fde,
        "J": ade.min(axis=1) + 0.5 * fde.min(axis=1),
        "endpoint_radius_q05": np.quantile(radius, .05, axis=1),
        "endpoint_radius_q25": np.quantile(radius, .25, axis=1),
        "endpoint_radius_q50": np.quantile(radius, .50, axis=1),
        "endpoint_radius_q75": np.quantile(radius, .75, axis=1),
        "endpoint_radius_q95": np.quantile(radius, .95, axis=1),
        "lateral_std": endpoint[..., 1].std(axis=1),
        "lateral_q05": np.quantile(endpoint[..., 1], .05, axis=1),
        "lateral_q95": np.quantile(endpoint[..., 1], .95, axis=1),
        "lateral_max_abs": np.abs(endpoint[..., 1]).max(axis=1),
        "longitudinal_std": endpoint[..., 0].std(axis=1),
        "longitudinal_q05": np.quantile(endpoint[..., 0], .05, axis=1),
        "longitudinal_q95": np.quantile(endpoint[..., 0], .95, axis=1),
        "pairwise_endpoint_distance": endpoint_pairs.mean(axis=1),
        "pairwise_path_distance": path_pairs.mean(axis=1),
        "duplicate_endpoint_010": (endpoint_pairs < .10).mean(axis=1),
        "duplicate_path_010": (path_pairs < .10).mean(axis=1),
        "effective_mode_count": greedy_effective_count(pair_path, .10),
        "radial_shortfall": np.maximum(0.0, gt_radius - radius.max(axis=1)),
        "radial_coverage_ratio": coverage_ratio,
    }


def candidate_metrics(candidates: np.ndarray, future: np.ndarray, past_last: np.ndarray, chunk_size: int = 256) -> dict[str, np.ndarray]:
    """Compute pairwise geometry in bounded-memory chunks (Univ has >24k samples)."""
    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(candidates), chunk_size):
        stop = min(len(candidates), start + chunk_size)
        chunk = _candidate_metrics_chunk(candidates[start:stop], future[start:stop], past_last[start:stop])
        for key, values in chunk.items(): pieces.setdefault(key, []).append(values)
    return {key: np.concatenate(values) for key, values in pieces.items()}


def future_features(future: np.ndarray, past_last: np.ndarray) -> dict[str, np.ndarray]:
    rel = future - past_last[:, None]
    step = np.concatenate((rel[:, :1], np.diff(rel, axis=1)), axis=1)
    v0, v1 = step[:, :-1], step[:, 1:]
    denom = np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1)
    angle = np.arccos(np.clip(np.sum(v0 * v1, axis=-1) / np.maximum(denom, 1e-8), -1.0, 1.0))
    angle[denom < 1e-8] = 0.0
    endpoint = rel[:, -1]
    return {
        "future_endpoint_radius": np.linalg.norm(endpoint, axis=-1),
        "future_lateral_endpoint": endpoint[:, 1],
        "future_longitudinal_endpoint": endpoint[:, 0],
        "future_turn_angle": angle.sum(axis=1),
        "future_path_length": np.linalg.norm(step, axis=-1).sum(axis=1),
    }


def bin_labels(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.digitize(values, boundaries, right=True).astype(np.int32)


def mean_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: finite_mean(np.asarray([row[key] for row in rows], dtype=np.float64)) for key in keys}


def audit_fold(fold: str, max_samples: int | None) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    source = train_features(fold)
    cache_path = CACHE / f"{fold}_seed_1.npz"
    with np.load(cache_path) as cache:
        length = len(cache["sample_id"])
        if max_samples is not None: length = min(length, max_samples)
        sample_id = np.asarray(cache["sample_id"][:length])
        past = np.asarray(cache["past"][:length], dtype=np.float32)
        future = np.asarray(cache["future"][:length], dtype=np.float32)
        num_valid = np.asarray(cache["num_valid"][:length])
        candidates = {family: np.asarray(cache[f"{family}_candidates"][:length], dtype=np.float32) for family in FAMILIES}
    observed = observation_features(past, num_valid)
    past_last = past[:, 0, -1]
    future_diag = future_features(future, past_last)
    speed_boundaries = np.quantile(source["median_step_speed"], [.25, .50, .75])
    density_boundaries = np.quantile(source["neighbor_count_r4"], [1 / 3, 2 / 3])
    speed_bin = bin_labels(observed["median_step_speed"], speed_boundaries)
    density_bin = bin_labels(observed["neighbor_count_r4"], density_boundaries)
    family_metrics = {family: candidate_metrics(candidates[family], future, past_last) for family in FAMILIES}
    correction_endpoint = np.linalg.norm(candidates["v2"][:, :, -1] - candidates["h3"][:, :, -1], axis=-1).mean(axis=1)
    correction_path = np.linalg.norm(candidates["v2"] - candidates["h3"], axis=-1).mean(axis=(1, 2))
    per_sample = []
    for i in range(length):
        row = {"fold": fold, "seed": 1, "sample_id": int(sample_id[i]), "speed_bin": int(speed_bin[i] + 1),
               "density_bin": ("low", "medium", "high")[int(density_bin[i])],
               "h3_v2_endpoint_correction": float(correction_endpoint[i]), "h3_v2_path_correction": float(correction_path[i])}
        for key, values in observed.items(): row[key] = float(values[i])
        for key, values in future_diag.items(): row[key] = float(values[i])
        for family, metrics in family_metrics.items():
            for key, values in metrics.items(): row[f"{family}_{key}"] = float(values[i])
        per_sample.append(row)
    shift_rows = []
    for feature in observed:
        shift_rows.append({"fold": fold, "feature": feature, "train_mean": finite_mean(source[feature]), "train_std": finite_std(source[feature]),
                           "test_mean": finite_mean(observed[feature]), "test_std": finite_std(observed[feature]), "SMD": smd(source[feature], observed[feature]),
                           "speed_q25": float(speed_boundaries[0]), "speed_q50": float(speed_boundaries[1]), "speed_q75": float(speed_boundaries[2]),
                           "density_t1": float(density_boundaries[0]), "density_t2": float(density_boundaries[1])})
    geometry_rows = []
    geometry_keys = tuple(next(iter(family_metrics.values())).keys())
    for family in FAMILIES:
        row = {"fold": fold, "family": family, "seed": 1, "samples": length}
        row.update({key: finite_mean(values) for key, values in family_metrics[family].items()})
        geometry_rows.append(row)
    binned_rows = []
    for family in FAMILIES:
        for speed in range(4):
            for density in range(3):
                mask = (speed_bin == speed) & (density_bin == density)
                if not mask.any(): continue
                row = {"fold": fold, "family": family, "speed_bin": speed + 1, "density_bin": ("low", "medium", "high")[density], "samples": int(mask.sum())}
                for key in ("minADE", "minFDE", "same_path_FDE", "J", "lateral_std", "longitudinal_std", "endpoint_radius_q95",
                            "pairwise_endpoint_distance", "pairwise_path_distance", "duplicate_endpoint_010", "duplicate_path_010",
                            "effective_mode_count", "radial_shortfall", "radial_coverage_ratio"):
                    row[key] = finite_mean(family_metrics[family][key][mask])
                binned_rows.append(row)
    speed_gap = {}
    for family in ("h3", "v2"):
        gaps = []
        for speed in range(4):
            mask = speed_bin == speed
            gaps.append(finite_mean(family_metrics[family]["J"][mask] - family_metrics["k20"]["J"][mask]))
        speed_gap[family] = gaps
    fold_diag = {
        "fold": fold, "samples": length,
        "speed_boundaries": speed_boundaries.tolist(), "density_boundaries": density_boundaries.tolist(),
        "speed_gap_h3_minus_k20": speed_gap["h3"], "speed_gap_v2_minus_k20": speed_gap["v2"],
        "speed_gap_h3_spearman": rank_correlation(np.arange(4), np.asarray(speed_gap["h3"])),
        "speed_gap_v2_spearman": rank_correlation(np.arange(4), np.asarray(speed_gap["v2"])),
        "speed_correction_spearman": rank_correlation(observed["median_step_speed"], correction_endpoint),
    }
    return per_sample, shift_rows, geometry_rows, binned_rows, fold_diag


def group_lookup(rows: list[dict], fold: str, family: str, speed: int | None = None, density: str | None = None) -> dict | None:
    chosen = [r for r in rows if r["fold"] == fold and r["family"] == family]
    if speed is not None: chosen = [r for r in chosen if r["speed_bin"] == speed]
    if density is not None: chosen = [r for r in chosen if r["density_bin"] == density]
    if not chosen: return None
    total = sum(r["samples"] for r in chosen)
    result = {"samples": total}
    for key in ("minADE", "minFDE", "same_path_FDE", "J", "lateral_std", "longitudinal_std", "duplicate_path_010", "radial_shortfall"):
        result[key] = sum(r[key] * r["samples"] for r in chosen) / total
    return result


def decisions(shift_rows: list[dict], geometry_rows: list[dict], binned_rows: list[dict], diagnostics: list[dict]) -> dict:
    shifts = {(r["fold"], r["feature"]): r["SMD"] for r in shift_rows}
    eth_diag = next((x for x in diagnostics if x["fold"] == "eth"), None)
    scale = {"available": eth_diag is not None}
    if eth_diag is not None:
        scale_smd = max(shifts[("eth", key)] for key in ("mean_step_speed", "median_step_speed", "last_step_speed"))
        monotonic = max(eth_diag["speed_gap_h3_spearman"], eth_diag["speed_gap_v2_spearman"]) >= .8
        correction = eth_diag["speed_correction_spearman"] >= .30
        v2_all = next(r for r in geometry_rows if r["fold"] == "eth" and r["family"] == "v2")
        k20_all = next(r for r in geometry_rows if r["fold"] == "eth" and r["family"] == "k20")
        v2_top = group_lookup(binned_rows, "eth", "v2", speed=4); k20_top = group_lookup(binned_rows, "eth", "k20", speed=4)
        v2_degradation = v2_top["J"] / v2_all["J"] - 1 if v2_top else float("nan")
        k20_degradation = k20_top["J"] / k20_all["J"] - 1 if k20_top else float("nan")
        top_speed = v2_degradation >= .05 and v2_degradation - k20_degradation >= .05
        v2_low = group_lookup(binned_rows, "eth", "v2", speed=1)
        radial = bool(v2_top and v2_low and v2_top["radial_shortfall"] > max(.05, 1.10 * v2_low["radial_shortfall"]))
        mechanism = monotonic or correction or top_speed or radial
        scale.update({"scale_SMD": scale_smd, "condition_shift_pass": scale_smd >= .8, "monotonic_gap_pass": monotonic,
                      "correction_correlation_pass": correction, "top_speed_relative_failure_pass": top_speed,
                      "radial_shortfall_pass": radial, "v2_top_speed_J_degradation": v2_degradation,
                      "k20_top_speed_J_degradation": k20_degradation, "mechanism_pass": mechanism,
                      "EXP166A_ALLOWED": bool(scale_smd >= .8 and mechanism)})
    univ_diag = next((x for x in diagnostics if x["fold"] == "univ"), None)
    density = {"available": univ_diag is not None}
    if univ_diag is not None:
        target = {family: group_lookup(binned_rows, "univ", family, speed=1, density="high") for family in FAMILIES}
        overall = {family: next(r for r in geometry_rows if r["fold"] == "univ" and r["family"] == family) for family in FAMILIES}
        enough = all(x is not None and x["samples"] >= 30 for x in target.values())
        if enough:
            target_gap = target["h3"]["J"] - target["k20"]["J"]
            overall_gap = overall["h3"]["J"] - overall["k20"]["J"]
            concentrated = target_gap > 0 and target_gap > overall_gap
            lateral = target["h3"]["lateral_std"] < .95 * target["k20"]["lateral_std"]
            partial = target["h3"]["J"] > target["v2"]["J"] > target["k20"]["J"]
            duplicate_or_longitudinal = (target["h3"]["duplicate_path_010"] > target["k20"]["duplicate_path_010"] + .01 or
                                         target["h3"]["longitudinal_std"] < .95 * target["k20"]["longitudinal_std"])
        else:
            target_gap = overall_gap = float("nan"); concentrated = lateral = partial = duplicate_or_longitudinal = False
        density.update({"target_samples": target["h3"]["samples"] if target["h3"] else 0, "enough_samples": enough,
                        "target_H3_minus_K20_J": target_gap, "overall_H3_minus_K20_J": overall_gap,
                        "concentrated_degradation_pass": concentrated, "lateral_narrowing_pass": lateral,
                        "V2_partial_recovery_pass": partial, "duplicate_or_longitudinal_concentration_pass": duplicate_or_longitudinal,
                        "EXP167_ALLOWED": bool(enough and concentrated and lateral and partial and duplicate_or_longitudinal)})
    return {"scale_hypothesis": scale, "density_lateral_hypothesis": density}


def analysis_text(verdict: dict, geometry: list[dict], complete: bool) -> str:
    scale = verdict["scale_hypothesis"]; density = verdict["density_lateral_hypothesis"]
    scale_decision = "SUPPORTED → EXP-166A allowed" if scale.get("EXP166A_ALLOWED") else "NOT SUPPORTED → do not run EXP-166A"
    density_decision = "SUPPORTED → EXP-167 allowed" if density.get("EXP167_ALLOWED") else "NOT SUPPORTED → do not run EXP-167"
    return (f"# EXP-165 Official Fold Shift + Candidate Geometry Causal Audit\n\n"
            f"## Status: {'COMPLETE' if complete else 'PARTIAL/SMOKE'}\n\n"
            f"## Scale hypothesis: {scale_decision}\n\n```json\n{json.dumps(json_safe(scale), indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Density/lateral hypothesis: {density_decision}\n\n```json\n{json.dumps(json_safe(density), indent=2, ensure_ascii=False)}\n```\n\n"
            "本实验只提供后续训练的准入证据，不使用 outer-test 反复调整阈值或超参数。完整解释应结合 `fold_shift_summary.csv`、"
            "`candidate_geometry_summary.csv` 和 `binned_error_summary.csv`。\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    required = [DATA / f"{fold}_{kind}.npy" for fold in folds for kind in ("data_train", "num_train")]
    required += [CACHE / f"{fold}_seed_1.npz" for fold in folds]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"folds": folds, "cache": str(CACHE), "output": str(args.output_dir), "max_samples": args.max_samples}, indent=2))
        print("[done] dry-run passed"); return
    all_sample: list[dict] = []; all_shift: list[dict] = []; all_geometry: list[dict] = []; all_binned: list[dict] = []; all_diag: list[dict] = []
    for fold in folds:
        print(f"[audit] {fold}", flush=True)
        sample, shift, geometry, binned, diag = audit_fold(fold, args.max_samples)
        all_sample += sample; all_shift += shift; all_geometry += geometry; all_binned += binned; all_diag.append(diag)
    verdict = decisions(all_shift, all_geometry, all_binned, all_diag)
    complete = set(folds) == set(FOLDS) and args.max_samples is None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample_metrics.csv", all_sample)
    write_csv(args.output_dir / "fold_shift_summary.csv", all_shift)
    write_csv(args.output_dir / "candidate_geometry_summary.csv", all_geometry)
    write_csv(args.output_dir / "per_fold_summary.csv", all_geometry)
    write_csv(args.output_dir / "all_runs.csv", all_geometry)
    write_csv(args.output_dir / "binned_error_summary.csv", all_binned)
    payload = {"experiment": "EXP-165 Official Fold Shift Candidate Geometry Causal Audit", "complete": complete,
               "folds": list(folds), "seed": 1, "trained_new_model": False, "verdict": verdict,
               "fold_diagnostics": all_diag, "protocol": {"speed_bins_from_outer_train": True, "density_bins_from_outer_train": True,
               "observation_only_condition_features": True, "future_used_for_diagnostics_only": True}}
    (args.output_dir / "summary.json").write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "RESULT_ANALYSIS.md").write_text(analysis_text(verdict, all_geometry, complete), encoding="utf-8")
    print(json.dumps(json_safe(verdict), indent=2, ensure_ascii=False), flush=True)
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
