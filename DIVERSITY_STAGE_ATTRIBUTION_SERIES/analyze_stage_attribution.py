"""Validation-only analysis for H3/V2 diversity stage attribution.

The script supports an interim H3-only review and a final six-checkpoint review.
It performs inference without future labels, then uses validation labels only for
metrics, fixed-cohort analysis, and paired bootstrap intervals.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module


ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
EXP173 = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
H3_BASE = (
    EXP173 / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C00 = (
    EXP173 / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C01 = (
    REPO / "DIVERSITY_LOSS_ABLATION_SERIES" / "results" / "weight_sweep"
    / "divw_0p000_m_0p400" / "f16_temporal_social_cross" / "seed_1"
    / "val_scene_-1"
)
H3_NEW = HERE / "results" / "h3_w000" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
C10 = HERE / "results" / "c10_h3w000_v2w005" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
C11 = HERE / "results" / "c11_h3w000_v2w000" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    stage: str
    h3_weight: float
    v2_weight: float | None
    run_dir: Path
    h3_source: Path | None = None

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "best_selected.pth"

    @property
    def split_indices(self) -> Path:
        return self.run_dir / "split_indices.npz"


CONDITIONS = {
    "h3_w005": Condition("h3_w005", "H3 w=.05", "h3", 0.05, None, H3_BASE),
    "h3_w000": Condition("h3_w000", "H3 w=0", "h3", 0.0, None, H3_NEW),
    "c00": Condition("c00", "C00", "v2", 0.05, 0.05, C00, H3_BASE / "best_selected.pth"),
    "c01": Condition("c01", "C01", "v2", 0.05, 0.0, C01, H3_BASE / "best_selected.pth"),
    "c10": Condition("c10", "C10", "v2", 0.0, 0.05, C10, H3_NEW / "best_selected.pth"),
    "c11": Condition("c11", "C11", "v2", 0.0, 0.0, C11, H3_NEW / "best_selected.pth"),
}

PAIR_I, PAIR_J = np.triu_indices(20, k=1)
if len(PAIR_I) != 190:
    raise RuntimeError("Expected 190 unordered pairs for 20 modes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_validation_identity(keys: list[str]) -> np.ndarray:
    canonical = np.load(CONDITIONS["h3_w005"].split_indices)["validation"].astype(np.int64)
    for key in keys:
        path = CONDITIONS[key].split_indices
        if not path.is_file():
            raise FileNotFoundError(path)
        current = np.load(path)["validation"].astype(np.int64)
        if not np.array_equal(canonical, current):
            raise RuntimeError(f"Validation sample IDs differ for {key}")
    if len(np.unique(canonical)) != len(canonical):
        raise RuntimeError("Canonical validation IDs contain duplicates")
    return canonical


def load_model(condition: Condition, cfg, device: torch.device) -> torch.nn.Module:
    model_mod = load_module(
        f"stage_attr_front_{condition.key}",
        REPO / "FRONT_ENCODER_SERIES" / "front_encoder_models.py",
    )
    model = model_mod.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    if condition.stage == "h3":
        h3_mod = load_module(
            f"stage_attr_h3_{condition.key}",
            REPO / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py",
        )
        h3_mod.replace_decoder(model, cfg, "h3_anchor_proposal")
    else:
        if condition.h3_source is None or not condition.h3_source.is_file():
            raise FileNotFoundError(condition.h3_source)
        os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(condition.h3_source)
        os.environ["EXP173_DECODER_INIT_SEED"] = "1"
        v2_mod = load_module(
            f"stage_attr_v2_{condition.key}",
            REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py",
        )
        v2_mod.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    state = torch.load(condition.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate(
    condition: Condition,
    base_cfg,
    loader: DataLoader,
    sample_ids: np.ndarray,
    trainer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    cfg = copy.deepcopy(base_cfg)
    model = load_model(condition, cfg, device)
    candidates, ades, fdes, logits = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            past, future, num_valid, *_ = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)
            focal = pred[:, 0]
            focal_gt = future[:, 0]
            focal_logits = model.last_mode_logits[:, 0]
            error = torch.linalg.vector_norm(focal - focal_gt[:, None], dim=-1)
            candidates.append(focal.cpu().numpy())
            ades.append(error.mean(dim=-1).cpu().numpy())
            fdes.append(error[..., -1].cpu().numpy())
            logits.append(focal_logits.cpu().numpy())
    out = {
        "sample_id": sample_ids.copy(),
        "candidate": np.concatenate(candidates).astype(np.float32),
        "ade": np.concatenate(ades).astype(np.float32),
        "fde": np.concatenate(fdes).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
    }
    if out["ade"].shape != (len(sample_ids), 20):
        raise RuntimeError(f"Unexpected candidate shape for {condition.key}: {out['ade'].shape}")
    return out


def derive(raw: dict[str, np.ndarray], margin: float = 0.4) -> dict[str, np.ndarray]:
    candidate, ade, fde, logits = raw["candidate"], raw["ade"], raw["fde"], raw["logits"]
    n = len(ade)
    row = np.arange(n)
    selected_idx = np.argmax(logits, axis=1)
    oracle_idx = np.argmin(ade, axis=1)
    minfde_idx = np.argmin(fde, axis=1)
    score_order = np.argsort(-logits, axis=1)
    inverse_rank = np.empty_like(score_order)
    inverse_rank[row[:, None], score_order] = np.arange(20)[None, :]
    oracle_rank = inverse_rank[row, oracle_idx] + 1

    endpoints = candidate[:, :, -1]
    pair_distance = np.linalg.norm(
        endpoints[:, PAIR_I] - endpoints[:, PAIR_J], axis=-1
    )
    active = pair_distance < margin
    penalty = np.maximum(0.0, margin - pair_distance)
    sorted_ade = np.sort(ade, axis=1)
    selected_ade = ade[row, selected_idx]
    minade = ade[row, oracle_idx]
    regret = selected_ade - minade
    bucket = np.full(n, "R0", dtype="U2")
    bucket[(regret >= 0.20) & (regret < 0.50)] = "R1"
    high = regret >= 0.50
    bucket[high & (minade < 0.50)] = "A"
    bucket[high & (minade >= 0.50) & (minade < 1.00)] = "B"
    bucket[high & (minade >= 1.00)] = "C"

    return {
        "sample_id": raw["sample_id"],
        "selected_idx": selected_idx,
        "oracle_idx": oracle_idx,
        "minfde_idx": minfde_idx,
        "selected_ade": selected_ade,
        "selected_fde": fde[row, selected_idx],
        "minade": minade,
        "minfde": fde[row, minfde_idx],
        "samepath_fde": fde[row, oracle_idx],
        "regret": regret,
        "delta_ade_1_2": sorted_ade[:, 1] - sorted_ade[:, 0],
        "endpoint_spread": pair_distance.mean(axis=1),
        "support_0p5": (fde < 0.5).sum(axis=1).astype(np.float64),
        "support_1p0": (fde < 1.0).sum(axis=1).astype(np.float64),
        "ade_lt_0p5": (ade < 0.5).sum(axis=1).astype(np.float64),
        "ade_lt_1p0": (ade < 1.0).sum(axis=1).astype(np.float64),
        "oracle_rank": oracle_rank.astype(np.float64),
        "oracle_recall_1": (oracle_rank <= 1).astype(np.float64),
        "oracle_recall_3": (oracle_rank <= 3).astype(np.float64),
        "oracle_rank_ge15": (oracle_rank >= 15).astype(np.float64),
        "minade_eq_minfde": (oracle_idx == minfde_idx).astype(np.float64),
        "active_pair_rate_sample": active.mean(axis=1),
        "active_sample": active.any(axis=1).astype(np.float64),
        "diversity_penalty": penalty.mean(axis=1),
        "active_pairs_count": active.sum(axis=1).astype(np.float64),
        "active_pair_distance_sum": (pair_distance * active).sum(axis=1),
        "bucket": bucket,
    }


def summarize(condition: Condition, values: dict[str, np.ndarray]) -> dict[str, Any]:
    active_count = values["active_pairs_count"].sum()
    active_distance_mean = (
        values["active_pair_distance_sum"].sum() / active_count
        if active_count > 0 else math.nan
    )
    return {
        "condition": condition.key,
        "label": condition.label,
        "stage": condition.stage,
        "h3_weight": condition.h3_weight,
        "v2_weight": condition.v2_weight,
        "N": len(values["sample_id"]),
        "selected_ade": float(values["selected_ade"].mean()),
        "selected_fde": float(values["selected_fde"].mean()),
        "minade": float(values["minade"].mean()),
        "minfde": float(values["minfde"].mean()),
        "samepath_fde": float(values["samepath_fde"].mean()),
        "regret": float(values["regret"].mean()),
        "endpoint_spread": float(values["endpoint_spread"].mean()),
        "support_0p5": float(values["support_0p5"].mean()),
        "support_1p0": float(values["support_1p0"].mean()),
        "ade_lt_0p5": float(values["ade_lt_0p5"].mean()),
        "ade_lt_1p0": float(values["ade_lt_1p0"].mean()),
        "delta_ade_1_2_median": float(np.median(values["delta_ade_1_2"])),
        "minade_eq_minfde_rate": float(values["minade_eq_minfde"].mean()),
        "oracle_recall_1": float(values["oracle_recall_1"].mean()),
        "oracle_recall_3": float(values["oracle_recall_3"].mean()),
        "oracle_rank_median": float(np.median(values["oracle_rank"])),
        "oracle_rank_ge15_rate": float(values["oracle_rank_ge15"].mean()),
        "active_pair_rate": float(values["active_pair_rate_sample"].mean()),
        "active_sample_rate": float(values["active_sample"].mean()),
        "mean_diversity_penalty": float(values["diversity_penalty"].mean()),
        "active_pairs_per_sample_mean": float(values["active_pairs_count"].mean()),
        "active_pairs_per_sample_median": float(np.median(values["active_pairs_count"])),
        "active_pair_distance_mean": float(active_distance_mean),
    }


def bucket_rows(condition: Condition, values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for bucket in ("R0", "R1", "A", "B", "C"):
        mask = values["bucket"] == bucket
        if not mask.any():
            continue
        rows.append({
            "condition": condition.key,
            "bucket": bucket,
            "N": int(mask.sum()),
            "rate": float(mask.mean()),
            "selected_ade": float(values["selected_ade"][mask].mean()),
            "minade": float(values["minade"][mask].mean()),
            "regret": float(values["regret"][mask].mean()),
            "endpoint_spread": float(values["endpoint_spread"][mask].mean()),
            "support_1p0": float(values["support_1p0"][mask].mean()),
            "delta_ade_1_2_median": float(np.median(values["delta_ade_1_2"][mask])),
        })
    return rows


def bootstrap_mean_ci(
    delta: np.ndarray,
    repetitions: int,
    seed: int,
    chunk_size: int = 256,
) -> tuple[float, float, float]:
    delta = np.asarray(delta, dtype=np.float64)
    if len(delta) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_rows(
    values_by_key: dict[str, dict[str, np.ndarray]], repetitions: int, seed: int
) -> list[dict[str, Any]]:
    metrics = [
        "endpoint_spread", "support_0p5", "support_1p0", "ade_lt_0p5",
        "ade_lt_1p0", "delta_ade_1_2", "minade", "minfde",
        "selected_ade", "regret",
    ]
    baseline = values_by_key["c00"]
    rows = []
    for condition_index, key in enumerate(("c01", "c10", "c11")):
        if key not in values_by_key:
            continue
        current = values_by_key[key]
        if not np.array_equal(current["sample_id"], baseline["sample_id"]):
            raise RuntimeError(f"Paired sample IDs do not align for {key}")
        for metric_index, metric in enumerate(metrics):
            delta = current[metric] - baseline[metric]
            mean, low, high = bootstrap_mean_ci(
                delta, repetitions, seed + condition_index * 100 + metric_index
            )
            rows.append({
                "condition": key,
                "reference": "c00",
                "metric": metric,
                "delta_mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed + condition_index * 100 + metric_index,
            })
    return rows


def fixed_cohort_rows(values_by_key: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    baseline = values_by_key["c00"]
    fixed_ids = baseline["sample_id"][baseline["bucket"] == "A"]
    metrics = [
        "endpoint_spread", "support_0p5", "support_1p0", "ade_lt_0p5",
        "ade_lt_1p0", "delta_ade_1_2", "minade", "selected_ade", "regret",
    ]
    rows = []
    for key in ("c00", "c01", "c10", "c11"):
        if key not in values_by_key:
            continue
        values = values_by_key[key]
        lookup = {int(sample_id): i for i, sample_id in enumerate(values["sample_id"])}
        try:
            indices = np.asarray([lookup[int(sample_id)] for sample_id in fixed_ids], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Fixed cohort ID missing in {key}: {exc}") from exc
        row: dict[str, Any] = {"condition": key, "N": len(indices)}
        for metric in metrics:
            selected = values[metric][indices]
            row[metric] = float(
                np.median(selected) if metric == "delta_ade_1_2" else selected.mean()
            )
        rows.append(row)
    return fixed_ids, rows


def classify_case(summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    if "h3_w000" not in summaries:
        return "PENDING_H3", "H3-NoDiv has not been evaluated."
    base, new = summaries["h3_w005"], summaries["h3_w000"]
    spread_delta = new["endpoint_spread"] - base["endpoint_spread"]
    support_delta = new["support_1p0"] - base["support_1p0"]
    gap_delta = new["delta_ade_1_2_median"] - base["delta_ade_1_2_median"]
    minade_rel = (new["minade"] - base["minade"]) / base["minade"]
    minfde_rel = (new["minfde"] - base["minfde"]) / base["minfde"]
    geometry = spread_delta <= -0.10 and support_delta >= 0.50 and gap_delta <= -0.02
    degraded = minade_rel >= 0.02 or minfde_rel >= 0.02
    no_h3_effect = (
        abs(spread_delta) < 0.10 and abs(support_delta) < 0.50 and abs(gap_delta) < 0.02
    )
    if geometry and degraded:
        return "CASE_D_MODE_COLLAPSE_RISK", "H3-Off concentrates modes but costs >=2% generation quality."
    if geometry and all(key in summaries for key in ("c10", "c11")):
        rebound = summaries["c10"]["endpoint_spread"] - summaries["c11"]["endpoint_spread"] >= 0.10
        c11_stays_low = summaries["c11"]["endpoint_spread"] <= summaries["c00"]["endpoint_spread"] - 0.10
        if rebound and c11_stays_low:
            return "CASE_C_V2_REDISPERSION", "H3-Off concentrates modes and V2-On disperses them again."
    if geometry:
        return "CASE_A_H3_SHAPING", "H3 diversity has an engineering-meaningful geometry effect without >=2% generation cost."
    if no_h3_effect:
        return "CASE_B_NOT_MAIN_CAUSE", "H3-NoDiv does not cross any preregistered geometry threshold."
    return "MIXED_OR_PARTIAL", "H3 changes are directionally mixed or below the joint Case A/B criteria."


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def render_report(
    output: Path,
    phase: str,
    summaries: dict[str, dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    verdict: tuple[str, str],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# F16 H3→V2 Diversity Stage Attribution",
        "",
        f"- Phase: `{phase}`",
        "- Scope: ETH, seed 1, the frozen 5% scene-tail validation split.",
        "- Test policy: no test path is used by this analysis or by the new training runner.",
        "- Bootstrap: paired over identical validation sample IDs; descriptive validation evidence, not independent confirmation.",
        f"- Verdict: **{verdict[0]}** — {verdict[1]}",
        "",
        "## Stage attribution summary",
        "",
        "| Condition | H3 w | V2 w | Selected ADE | minADE | minFDE | Regret | Spread | Support@1m | ΔADE1,2 median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("c00", "c01", "c10", "c11"):
        if key not in summaries:
            continue
        s = summaries[key]
        lines.append(
            f"| {key.upper()} | {s['h3_weight']:.2f} | {s['v2_weight']:.2f} | "
            f"{fmt(s['selected_ade'])} | {fmt(s['minade'])} | {fmt(s['minfde'])} | "
            f"{fmt(s['regret'])} | {fmt(s['endpoint_spread'], 3)} | "
            f"{fmt(s['support_1p0'], 2)} | {fmt(s['delta_ade_1_2_median'], 3)} |"
        )
    lines += [
        "",
        "## H3-only formation check",
        "",
        "| H3 condition | Spread | Support@0.5 | Support@1.0 | ADE<0.5 | ADE<1.0 | ΔADE1,2 median | minADE | minFDE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("h3_w005", "h3_w000"):
        if key not in summaries:
            continue
        s = summaries[key]
        lines.append(
            f"| {s['label']} | {fmt(s['endpoint_spread'], 3)} | {fmt(s['support_0p5'], 2)} | "
            f"{fmt(s['support_1p0'], 2)} | {fmt(s['ade_lt_0p5'], 2)} | {fmt(s['ade_lt_1p0'], 2)} | "
            f"{fmt(s['delta_ade_1_2_median'], 3)} | {fmt(s['minade'])} | {fmt(s['minfde'])} |"
        )
    lines += [
        "",
        "## Diversity activity at a hypothetical 0.4 m margin",
        "",
        "The activity metrics are calculated for every condition, including weight=0 runs.",
        "",
        "| Condition | Active pair rate | Active sample rate | Mean penalty | Active pairs/sample mean | median | Active distance mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("h3_w005", "h3_w000", "c00", "c01", "c10", "c11"):
        if key not in summaries:
            continue
        s = summaries[key]
        lines.append(
            f"| {s['label']} | {fmt(s['active_pair_rate'], 5)} | {fmt(s['active_sample_rate'], 4)} | "
            f"{fmt(s['mean_diversity_penalty'], 5)} | {fmt(s['active_pairs_per_sample_mean'], 2)} | "
            f"{fmt(s['active_pairs_per_sample_median'], 1)} | {fmt(s['active_pair_distance_mean'], 4)} |"
        )
    if fixed_rows:
        lines += [
            "",
            "## Fixed C00 A-cohort",
            "",
            "The sample IDs are frozen from C00 where regret ≥0.5 m and minADE <0.5 m.",
            "",
            "| Condition | N | Spread | Support@1m | ADE<1m | ΔADE1,2 median | minADE | Selected ADE | Regret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fixed_rows:
            lines.append(
                f"| {row['condition'].upper()} | {row['N']} | {fmt(row['endpoint_spread'], 3)} | "
                f"{fmt(row['support_1p0'], 2)} | {fmt(row['ade_lt_1p0'], 2)} | "
                f"{fmt(row['delta_ade_1_2'], 3)} | {fmt(row['minade'])} | "
                f"{fmt(row['selected_ade'])} | {fmt(row['regret'])} |"
            )
    if paired:
        lines += [
            "",
            "## Paired bootstrap versus C00",
            "",
            "Positive delta means the named condition minus C00; the CSV contains all ten preregistered metrics and 95% CIs.",
            "",
            "| Condition | Metric | Mean delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
        for row in paired:
            lines.append(
                f"| {row['condition'].upper()} | {row['metric']} | {fmt(row['delta_mean'], 5)} | "
                f"[{fmt(row['ci95_low'], 5)}, {fmt(row['ci95_high'], 5)}] |"
            )
    lines += [
        "",
        "## Preregistered engineering scales",
        "",
        "- Endpoint spread: |Δ| ≥ 0.10 m.",
        "- Support@1m: |Δ| ≥ 0.5 candidate/sample.",
        "- ΔADE1,2: |Δ| ≥ 0.02 m.",
        "- minADE/minFDE: relative change ≥ 2% is a meaningful cost or gain.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["h3", "final"], required=True)
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    args = parser.parse_args()

    keys = ["h3_w005", "h3_w000"]
    if args.phase == "final":
        keys += ["c00", "c01", "c10", "c11"]
    for key in keys:
        if not CONDITIONS[key].checkpoint.is_file():
            raise FileNotFoundError(CONDITIONS[key].checkpoint)
    sample_ids = assert_validation_identity(keys)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    astra_root = args.astra_root.resolve()
    cfg = configure_astra_env(REPO, astra_root, device)
    trainer = load_module(
        "stage_attr_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    )
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(
        data_dir / "eth_data_train.npy", data_dir / "eth_num_train.npy", cfg
    )
    loader = DataLoader(
        Subset(dataset, sample_ids.tolist()), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    output_root = HERE / "results" / ("h3_interim_analysis" if args.phase == "h3" else "stage_attribution_analysis")
    cache_root = output_root / "validation_candidate_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    values_by_key: dict[str, dict[str, np.ndarray]] = {}
    summaries_list: list[dict[str, Any]] = []
    all_buckets: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "phase": args.phase,
        "fold": "eth",
        "seed": 1,
        "split": "validation_only_scene_tail_5pct",
        "validation_sample_count": len(sample_ids),
        "validation_sample_id_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "test_accessed_by_this_analysis": False,
        "conditions": {},
    }
    for key in keys:
        condition = CONDITIONS[key]
        print(f"[EVAL] {key}: {condition.checkpoint}")
        raw = evaluate(condition, cfg, loader, sample_ids, trainer, device)
        np.savez_compressed(cache_root / f"{key}.npz", **raw)
        values = derive(raw)
        values_by_key[key] = values
        summary = summarize(condition, values)
        summaries_list.append(summary)
        all_buckets.extend(bucket_rows(condition, values))
        provenance["conditions"][key] = {
            "checkpoint": str(condition.checkpoint),
            "checkpoint_sha256": sha256(condition.checkpoint),
            "split_indices": str(condition.split_indices),
            "split_indices_sha256": sha256(condition.split_indices),
            "h3_source": str(condition.h3_source) if condition.h3_source else None,
            "h3_source_sha256": sha256(condition.h3_source) if condition.h3_source else None,
        }

    summaries = {row["condition"]: row for row in summaries_list}
    paired: list[dict[str, Any]] = []
    fixed_ids = np.empty(0, dtype=np.int64)
    fixed_rows: list[dict[str, Any]] = []
    if args.phase == "final":
        fixed_ids, fixed_rows = fixed_cohort_rows(values_by_key)
        paired = paired_rows(values_by_key, args.bootstrap_repetitions, args.bootstrap_seed)
        np.savetxt(output_root / "fixed_c00_a_cohort_sample_ids.txt", fixed_ids, fmt="%d")

    verdict = classify_case(summaries)
    write_csv(output_root / "condition_summary.csv", summaries_list)
    write_csv(output_root / "dynamic_bucket_summary.csv", all_buckets)
    write_csv(output_root / "fixed_c00_a_cohort.csv", fixed_rows)
    write_csv(output_root / "paired_bootstrap_vs_c00.csv", paired)
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "verdict.json").write_text(
        json.dumps({"code": verdict[0], "explanation": verdict[1]}, indent=2),
        encoding="utf-8",
    )
    report = output_root / ("H3_INTERIM_REPORT.md" if args.phase == "h3" else "STAGE_ATTRIBUTION_REPORT.md")
    render_report(report, args.phase, summaries, fixed_rows, paired, verdict, provenance)
    print(f"[DONE] {report}")


if __name__ == "__main__":
    main()
