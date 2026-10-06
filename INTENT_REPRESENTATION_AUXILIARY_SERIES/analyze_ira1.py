"""Validation-only matched analysis for IRA1 with paired bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

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
C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C00_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "stage_attribution_analysis" / "validation_candidate_cache" / "c00.npz"
)
STATS = REPO / "PHYSICAL_INTENT_FLOW_SERIES" / "calibration" / "intent_stats.json"
MODEL_PATH = HERE / "intent_aux_models.py"
RUNS = {
    "ira0": HERE / "results" / "ira0_detached" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1",
    "ira1": HERE / "results" / "ira1_joint" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1",
}
EXPERIMENTS = {
    "ira0": "ira0_detached_intent_aux",
    "ira1": "ira1_joint_intent_aux",
}
PAIR_I, PAIR_J = np.triu_indices(20, k=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_ira_model(cfg, arm: str, checkpoint: Path, device: torch.device):
    os.environ["IRA_SOURCE_CHECKPOINT"] = str(C00_RUN / "best_selected.pth")
    os.environ["IRA_STATS_JSON"] = str(STATS)
    os.environ["IRA_FLOW_LAYERS"] = "6"
    os.environ["IRA_FLOW_HIDDEN_DIM"] = "96"
    os.environ["IRA_NLL_WEIGHT"] = "0.05"
    models = load_module(
        f"ira_front_models_{arm}",
        REPO / "FRONT_ENCODER_SERIES" / "front_encoder_models.py",
    )
    ira = load_module(f"ira_decoder_analysis_{arm}", MODEL_PATH)
    model = models.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    ira.replace_decoder(model, cfg, EXPERIMENTS[arm])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.to(device).eval()


def evaluate(model, loader, sample_ids: np.ndarray, trainer, device) -> dict[str, np.ndarray]:
    candidates, ades, fdes, logits = [], [], [], []
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future = past.to(device), future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center,
                None,
                None,
                mode="testing",
                num_valid=num_valid,
                graph_loc=past,
            )
            prediction = output[2] + center.unsqueeze(-3)
            focal, target = prediction[:, 0], future[:, 0]
            error = torch.linalg.vector_norm(focal - target[:, None], dim=-1)
            candidates.append(focal.cpu().numpy())
            ades.append(error.mean(dim=-1).cpu().numpy())
            fdes.append(error[..., -1].cpu().numpy())
            logits.append(model.last_mode_logits[:, 0].cpu().numpy())
    result = {
        "sample_id": sample_ids.copy(),
        "candidate": np.concatenate(candidates).astype(np.float32),
        "ade": np.concatenate(ades).astype(np.float32),
        "fde": np.concatenate(fdes).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
    }
    if result["ade"].shape != (len(sample_ids), 20):
        raise RuntimeError(result["ade"].shape)
    return result


def derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    candidate = raw["candidate"].astype(np.float64)
    ade = raw["ade"].astype(np.float64)
    fde = raw["fde"].astype(np.float64)
    logits = raw["logits"].astype(np.float64)
    n = len(ade)
    row = np.arange(n)
    selected = logits.argmax(axis=1)
    oracle = ade.argmin(axis=1)
    minfde = fde.argmin(axis=1)
    order = np.argsort(-logits, axis=1)
    rank = np.empty_like(order)
    rank[row[:, None], order] = np.arange(1, 21)[None]
    oracle_rank = rank[row, oracle]
    endpoints = candidate[:, :, -1]
    pair = np.linalg.norm(endpoints[:, PAIR_I] - endpoints[:, PAIR_J], axis=-1)
    selected_ade = ade[row, selected]
    return {
        "selected_ade": selected_ade,
        "selected_fde": fde[row, selected],
        "minade": ade[row, oracle],
        "minfde": fde[row, minfde],
        "samepath_fde": fde[row, oracle],
        "regret": selected_ade - ade[row, oracle],
        "J": ade[row, oracle] + 0.5 * fde[row, minfde],
        "endpoint_spread": pair.mean(axis=1),
        "support_1p0": (fde < 1.0).sum(axis=1).astype(np.float64),
        "oracle_rank": oracle_rank.astype(np.float64),
        "recall_1": (oracle_rank <= 1).astype(np.float64),
        "recall_3": (oracle_rank <= 3).astype(np.float64),
        "duplicate_sample": (pair < 0.1).any(axis=1).astype(np.float64),
    }


def summarize(label: str, values: dict[str, np.ndarray]) -> dict:
    return {
        "condition": label,
        "N": len(values["minade"]),
        **{key: float(value.mean()) for key, value in values.items()},
    }


def bootstrap(delta: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return float(low), float(high)


def percent_gain(smaller_is_better_joint: float, detached: float) -> float:
    return 100.0 * (1.0 - smaller_is_better_joint / detached)


def gate(detached: dict, joint: dict) -> dict:
    j_gain = percent_gain(joint["J"], detached["J"])
    selected_gain = percent_gain(joint["selected_ade"], detached["selected_ade"])
    minade_gain = percent_gain(joint["minade"], detached["minade"])
    minfde_gain = percent_gain(joint["minfde"], detached["minfde"])
    samepath_gain = percent_gain(joint["samepath_fde"], detached["samepath_fde"])
    recall1_pp = 100.0 * (joint["recall_1"] - detached["recall_1"])
    spread_contraction = 100.0 * (
        1.0 - joint["endpoint_spread"] / detached["endpoint_spread"]
    )
    duplicate_increase_pp = 100.0 * (
        joint["duplicate_sample"] - detached["duplicate_sample"]
    )
    result = {
        "J_gain_percent": j_gain,
        "selected_ADE_gain_percent": selected_gain,
        "recall1_gain_pp": recall1_pp,
        "minADE_gain_percent": minade_gain,
        "minFDE_gain_percent": minfde_gain,
        "samepath_FDE_gain_percent": samepath_gain,
        "spread_contraction_percent": spread_contraction,
        "duplicate_increase_pp": duplicate_increase_pp,
        "generation_pass": j_gain >= 2.0,
        "ranking_pass": selected_gain >= 1.0 or recall1_pp >= 2.0,
        "oracle_safe": (
            min(minade_gain, minfde_gain) >= -0.5
            and max(minade_gain, minfde_gain) > 0.0
        ),
        "samepath_safe": samepath_gain >= -0.5,
        "spread_safe": spread_contraction <= 5.0,
        "duplicate_safe": duplicate_increase_pp <= 3.0,
    }
    result["all_gates_pass"] = all(
        result[key]
        for key in (
            "generation_pass", "ranking_pass", "oracle_safe", "samepath_safe",
            "spread_safe", "duplicate_safe",
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()

    checkpoints = {}
    required = [C00_CACHE, C00_RUN / "best_selected.pth", C00_RUN / "split_indices.npz", STATS]
    for arm, directory in RUNS.items():
        for view, filename in (("best", "best_selected.pth"), ("final", "final_model.pth")):
            checkpoints[f"{arm}_{view}"] = (arm, directory / filename)
        required.append(directory / "split_indices.npz")
    required.extend(checkpoint for _, checkpoint in checkpoints.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing analysis inputs:\n" + "\n".join(missing))

    with np.load(C00_CACHE) as archive:
        c00 = {key: archive[key] for key in archive.files}
    sample_ids = np.load(C00_RUN / "split_indices.npz")["validation"].astype(np.int64)
    for arm, directory in RUNS.items():
        arm_ids = np.load(directory / "split_indices.npz")["validation"].astype(np.int64)
        if not np.array_equal(sample_ids, arm_ids):
            raise RuntimeError(f"{arm} validation IDs do not exactly match C00")
    if not np.array_equal(sample_ids, c00["sample_id"]):
        raise RuntimeError("C00 cache IDs do not match the locked split")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, args.astra_root.resolve(), device)
    trainer = load_module(
        "ira_analysis_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    )
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(
        data / "eth_data_train.npy", data / "eth_num_train.npy", cfg
    )
    loader = DataLoader(
        Subset(dataset, sample_ids.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    output = HERE / "results" / "analysis"
    cache_dir = output / "validation_candidate_cache"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    values = {"c00": derive(c00)}
    summaries = [summarize("c00_historical", values["c00"])]
    provenance = {
        "protocol": "ira1_eth_seed1_matched_validation_only_v1",
        "sample_count": len(sample_ids),
        "test_accessed": False,
        "sample_id_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "c00_checkpoint_sha256": sha256(C00_RUN / "best_selected.pth"),
        "stats_sha256": sha256(STATS),
        "primary_comparison": "ira1_joint_minus_ira0_detached",
        "historical_c00_used_for_verdict": False,
        "conditions": {},
    }
    for label, (arm, checkpoint) in checkpoints.items():
        print(f"[eval] {label}: {checkpoint}", flush=True)
        model = load_ira_model(cfg, arm, checkpoint, device)
        raw = evaluate(model, loader, sample_ids, trainer, device)
        np.savez_compressed(cache_dir / f"{label}.npz", **raw)
        values[label] = derive(raw)
        summaries.append(summarize(label, values[label]))
        provenance["conditions"][label] = {
            "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_by_label = {row["condition"]: row for row in summaries}
    metric_names = list(values["c00"])
    paired_rows = []
    for view_index, view in enumerate(("best", "final")):
        detached_label, joint_label = f"ira0_{view}", f"ira1_{view}"
        for metric_index, metric in enumerate(metric_names):
            delta = values[joint_label][metric] - values[detached_label][metric]
            low, high = bootstrap(
                delta,
                args.bootstrap_repetitions,
                20260923 + 100 * view_index + metric_index,
            )
            paired_rows.append({
                "view": view,
                "metric": metric,
                "detached_mean": float(values[detached_label][metric].mean()),
                "joint_mean": float(values[joint_label][metric].mean()),
                "delta_joint_minus_detached": float(delta.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })

    gates = {
        view: gate(
            summary_by_label[f"ira0_{view}"],
            summary_by_label[f"ira1_{view}"],
        )
        for view in ("best", "final")
    }
    same_direction = (
        gates["best"]["J_gain_percent"] > 0.0
        and gates["final"]["J_gain_percent"] > 0.0
        and (
            gates["best"]["selected_ADE_gain_percent"] > 0.0
            or gates["best"]["recall1_gain_pp"] > 0.0
        )
        and (
            gates["final"]["selected_ADE_gain_percent"] > 0.0
            or gates["final"]["recall1_gain_pp"] > 0.0
        )
    )
    if all(item["all_gates_pass"] for item in gates.values()) and same_direction:
        verdict = "ADVANCE_IRA1_TO_SEEDS_AND_FOLDS"
    elif same_direction:
        verdict = "PROMISING_BUT_BELOW_GATE_REVIEW_ON_VALIDATION_ONLY"
    else:
        verdict = "STOP_IRA1_AFTER_MATCHED_SEED1"
    verdict_payload = {
        "verdict": verdict,
        "gates": gates,
        "best_final_same_direction": same_direction,
        "primary_comparison": "IRA1 joint gradient versus IRA0 detached gradient",
        "test_accessed": False,
    }

    write_csv(output / "condition_summary.csv", summaries)
    write_csv(output / "paired_bootstrap.csv", paired_rows)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "verdict.json").write_text(
        json.dumps(verdict_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# IRA1 matched validation report", "",
        f"Decision: **{verdict}**", "",
        "The causal comparison is IRA1 joint intent gradient versus IRA0 detached intent gradient. Historical C00 is descriptive only.", "",
        "| Condition | minADE | minFDE | Selected ADE | Recall@1 | Mean oracle rank | Spread | Duplicate rate |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['minade']:.6f} | {row['minfde']:.6f} | "
            f"{row['selected_ade']:.6f} | {100*row['recall_1']:.2f}% | "
            f"{row['oracle_rank']:.3f} | {row['endpoint_spread']:.4f} | "
            f"{100*row['duplicate_sample']:.2f}% |"
        )
    lines += ["", "## Gates", ""]
    for view in ("best", "final"):
        item = gates[view]
        lines.append(
            f"- {view}: J {item['J_gain_percent']:+.2f}%, selected ADE "
            f"{item['selected_ADE_gain_percent']:+.2f}%, Recall@1 "
            f"{item['recall1_gain_pp']:+.2f} pp, all gates={item['all_gates_pass']}."
        )
    lines += [
        "", "No test split was loaded. Seed/fold expansion is allowed only by the preregistered verdict."
    ]
    (output / "IRA1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

