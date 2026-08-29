"""PFRA: train-calibrated, validation-audited reliability and CFB2 cache builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = P / "datasets" / "eth_ucy" / "processed_data_diverse"
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
SPLITS = ("general", "challenge")
K = 20
SLOTS = 2


def baseline_dir(split, seed):
    return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def local_data(raw, num):
    """Observed-only retrieval features plus heading-normalized future targets."""
    past = np.asarray(raw[:, :, 0:8, -2:], np.float32)
    future = np.asarray(raw[:, 0, 8:20, -2:], np.float32)
    focal = past[:, 0]
    displacement = focal[:, -1] - focal[:, -4]
    speed = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement); heading[:, 0] = 1.0
    valid_heading = speed > 0.03
    heading[valid_heading] = displacement[valid_heading] / speed[valid_heading, None]
    c, s = heading[:, 0], heading[:, 1]
    center = focal[:, -1]
    delta_p = past - center[:, None, None]
    past_local = np.empty_like(delta_p)
    past_local[..., 0] = delta_p[..., 0] * c[:, None, None] + delta_p[..., 1] * s[:, None, None]
    past_local[..., 1] = -delta_p[..., 0] * s[:, None, None] + delta_p[..., 1] * c[:, None, None]
    delta_f = future - center[:, None]
    future_local = np.empty_like(delta_f)
    future_local[..., 0] = delta_f[..., 0] * c[:, None] + delta_f[..., 1] * s[:, None]
    future_local[..., 1] = -delta_f[..., 0] * s[:, None] + delta_f[..., 1] * c[:, None]
    v_last = past_local[:, 0, -1] - past_local[:, 0, -2]
    v_mean = (past_local[:, 0, -1] - past_local[:, 0, 0]) / 7.0
    v_prev = past_local[:, 0, -2] - past_local[:, 0, -3]
    dv = v_last - v_prev
    early = past_local[:, 0, -4] - past_local[:, 0, -7]
    late = past_local[:, 0, -1] - past_local[:, 0, -4]
    norm_product = np.linalg.norm(early, axis=-1) * np.linalg.norm(late, axis=-1) + 1e-6
    turn = np.stack(((early[:, 0] * late[:, 1] - early[:, 1] * late[:, 0]) / norm_product, (early * late).sum(-1) / norm_product), axis=-1)
    relative = past_local[:, 1:, -1]
    ids = np.arange(1, past.shape[1])[None]
    mask = ids < np.asarray(num)[:, None]
    distance = np.linalg.norm(relative, axis=-1)
    nearest = np.where(mask, distance, np.inf).min(-1); nearest[~np.isfinite(nearest)] = 10.0
    neighbour_velocity = (past_local[:, 1:, -1] - past_local[:, 1:, -4]) / 3.0
    focal_velocity = (past_local[:, 0, -1] - past_local[:, 0, -4])[:, None] / 3.0
    closing = -(relative * (neighbour_velocity - focal_velocity)).sum(-1) / (distance + 1e-6)
    strongest = np.where(mask, closing, -np.inf).max(-1); strongest[~np.isfinite(strongest)] = 0.0
    features = np.concatenate((
        v_last, v_mean, dv, turn,
        np.linalg.norm(past_local[:, 0, -1] - past_local[:, 0, 0], axis=-1, keepdims=True),
        (np.asarray(num) - 1)[:, None] / 8.0, nearest[:, None], strongest[:, None],
    ), axis=-1).astype(np.float32)
    return features, future_local.astype(np.float32)


def candidate_features(query_features, bank_features, retrieval_distance, bank_futures):
    # Pairwise future distance is used only to measure neighbour consensus; query GT is absent.
    pair = np.empty((len(bank_futures), K, K), np.float32)
    for start in range(0, len(bank_futures), 256):
        block = bank_futures[start:start + 256]
        diff = block[:, :, None] - block[:, None, :]
        pair[start:start + len(block)] = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    sorted_pair = np.sort(pair, axis=-1)
    agreement = sorted_pair[:, :, 1:6].mean(axis=-1)
    positive = pair[pair > 0]
    radius = float(np.quantile(positive, 0.25)) if len(positive) else 0.25
    support = (pair <= radius).sum(axis=-1).astype(np.float32) / K
    qspeed = np.linalg.norm(query_features[:, None, 0:2], axis=-1)
    bspeed = np.linalg.norm(bank_features[:, :, 0:2], axis=-1)
    scale_gap = np.abs(np.log((qspeed + 0.03) / (bspeed + 0.03)))
    x = np.stack((retrieval_distance, agreement, 1.0 - support, scale_gap), axis=-1)
    return x.astype(np.float32), pair.astype(np.float32), radius


def utility(bank_futures, query_future):
    error = np.linalg.norm(bank_futures - query_future[:, None], axis=-1)
    return (error.mean(axis=-1) + 0.25 * error[..., -1]).astype(np.float32)


def retrieve(raw, features, futures, train):
    scenes = np.asarray(raw[:, 0, 0, 0], np.int64)
    mean, std = features[train].mean(0), features[train].std(0)
    std[std < 1e-5] = 1.0
    z = (features - mean) / std
    n = len(raw)
    bank_future = np.empty((n, K, 12, 2), np.float32)
    bank_feature = np.empty((n, K, features.shape[1]), np.float32)
    distance = np.empty((n, K), np.float32)
    for scene in np.unique(scenes):
        bank = train[scenes[train] != scene]
        query = np.flatnonzero(scenes == scene)
        nearest = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(z[bank])
        d, loc = nearest.kneighbors(z[query])
        index = bank[loc]
        bank_future[query] = futures[index]
        bank_feature[query] = features[index]
        distance[query] = d
    return bank_future, bank_feature, distance


def select_slots(bank_future, pair, confidence, threshold):
    n = len(bank_future)
    selected = np.zeros((n, SLOTS, 12, 2), np.float32)
    selected_conf = np.zeros((n, SLOTS), np.float32)
    valid = np.zeros((n, SLOTS), bool)
    selected_index = np.full((n, SLOTS), -1, np.int64)
    for i in range(n):
        order = np.argsort(-confidence[i])
        chosen = []
        positive_pair = pair[i][pair[i] > 1e-8]
        separation = float(np.quantile(positive_pair, 0.35)) if len(positive_pair) else 1e-6
        for candidate in order:
            if confidence[i, candidate] < threshold:
                break
            if chosen and pair[i, candidate, chosen[0]] < separation:
                continue
            chosen.append(int(candidate))
            if len(chosen) == SLOTS:
                break
        # Stable responsibility: negative/lower lateral branch -> 18, positive/upper -> 19.
        chosen.sort(key=lambda j: float(bank_future[i, j, -1, 1]))
        if len(chosen) == 1:
            placements = [(0 if bank_future[i, chosen[0], -1, 1] < 0 else 1, chosen[0])]
        else:
            placements = list(enumerate(chosen))
        for slot, candidate in placements:
            selected[i, slot] = bank_future[i, candidate]
            selected_conf[i, slot] = confidence[i, candidate]
            valid[i, slot] = True
            selected_index[i, slot] = candidate
    return selected, selected_conf, valid, selected_index


def subset_report(confidence, util, indices, valid, selected_index):
    c, u = confidence[indices].reshape(-1), util[indices].reshape(-1)
    rho = float(spearmanr(c, -u).statistic)
    q = np.quantile(c, (0.25, 0.75))
    high, low = u[c >= q[1]], u[c <= q[0]]
    chosen_values = []
    for index in indices:
        for slot in range(SLOTS):
            if valid[index, slot]:
                chosen_values.append(float(util[index, selected_index[index, slot]]))
    counts = valid[indices].sum(axis=-1)
    return {
        "candidate_level_spearman_confidence_vs_negative_utility": rho,
        "high_confidence_utility": float(high.mean()),
        "low_confidence_utility": float(low.mean()),
        "high_vs_low_improvement_pct": float(100 * (1 - high.mean() / low.mean())),
        "selected_utility": float(np.mean(chosen_values)) if chosen_values else None,
        "pseudo_usage_fraction": {str(k): float((counts == k).mean()) for k in range(3)},
    }


def process(split, seed, raw, features, futures):
    run = baseline_dir(split, seed)
    split_indices = np.load(run / "split_indices.npz")
    train = np.asarray(split_indices["train"], np.int64)
    val = np.asarray(split_indices["validation"], np.int64)
    bank_future, bank_feature, retrieval_distance = retrieve(raw, features, futures, train)
    x, pair, consensus_radius = candidate_features(features, bank_feature, retrieval_distance, bank_future)
    u = utility(bank_future, futures)
    x_train, y_train = x[train].reshape(-1, x.shape[-1]), np.log1p(u[train].reshape(-1))
    x_mean, x_std = x_train.mean(0), x_train.std(0)
    x_std[x_std < 1e-6] = 1.0
    reg = Ridge(alpha=10.0).fit((x_train - x_mean) / x_std, y_train)
    predicted = reg.predict(((x.reshape(-1, x.shape[-1]) - x_mean) / x_std)).reshape(len(raw), K)
    train_pred = predicted[train].reshape(-1)
    center = float(np.quantile(train_pred, 0.35))
    scale = float(np.quantile(train_pred, 0.75) - np.quantile(train_pred, 0.25))
    scale = max(scale, 1e-4)
    confidence = 1.0 / (1.0 + np.exp((predicted - center) / (0.35 * scale)))
    threshold = 0.55
    selected, selected_conf, valid, selected_index = select_slots(bank_future, pair, confidence, threshold)
    train_report = subset_report(confidence, u, train, valid, selected_index)
    val_report = subset_report(confidence, u, val, valid, selected_index)
    passed = (
        val_report["candidate_level_spearman_confidence_vs_negative_utility"] >= 0.08
        and val_report["high_vs_low_improvement_pct"] >= 5.0
        and val_report["pseudo_usage_fraction"]["2"] < 0.90
    )
    metadata = {
        "split": split, "seed": seed, "bank_policy": "train-only cross-scene",
        "confidence_fit": "Ridge on train fold only; validation never fits coefficients or thresholds",
        "features": ["retrieval_distance", "future_neighbour_agreement", "inverse_cluster_support", "observed_motion_scale_gap"],
        "confidence_threshold": threshold, "pseudo_modes": [18, 19],
    }
    (ROOT / "cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ROOT / "cache" / f"{split}_seed_{seed}.npz",
        pseudo_futures=selected, confidence=selected_conf, valid_mask=valid,
        selected_bank_indices=selected_index, metadata_json=np.array(json.dumps(metadata)),
    )
    result = {
        "split": split, "gate_passed": passed, "train_only_calibration": True,
        "consensus_radius": consensus_radius, "ridge_coefficients": reg.coef_.tolist(),
        "train": train_report, "validation_locked": val_report, "metadata": metadata,
    }
    (ROOT / "pfra").mkdir(parents=True, exist_ok=True)
    (ROOT / "pfra" / f"{split}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PFRA] {split} passed={passed} val_rho={val_report['candidate_level_spearman_confidence_vs_negative_utility']:.3f} high_vs_low={val_report['high_vs_low_improvement_pct']:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    raw = np.load(DATA / "eth_data_train.npy", mmap_mode="r")
    num = np.load(DATA / "eth_num_train.npy", mmap_mode="r")
    features, futures = local_data(raw, num)
    results = [process(split, args.seed, raw, features, futures) for split in SPLITS]
    combined = {
        "gate_passed": all(r["gate_passed"] for r in results),
        "gate_rule": "both splits: locked-val rho>=0.08, high-confidence utility improves >=5%, <90% queries use two pseudo futures",
        "results": results,
    }
    (ROOT / "pfra" / "combined_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PFRA] combined gate_passed={combined['gate_passed']}")


if __name__ == "__main__":
    main()
