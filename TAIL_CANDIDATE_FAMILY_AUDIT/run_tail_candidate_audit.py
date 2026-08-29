"""TCA: zero-training audit of score-ranked tail candidates and rare future bases."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
CSO = W / "CANDIDATE_SET_OPTIMIZATION_SERIES"
GEOM = W / "INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES" / "run_geometry_diagnostic.py"
DATA = P / "datasets" / "eth_ucy" / "processed_data_diverse"
SPLITS = ("general", "challenge")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_run(split, seed):
    return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def local_coordinates(raw):
    past = np.asarray(raw[:, 0, :8, -2:], np.float32); future = np.asarray(raw[:, 0, 8:20, -2:], np.float32)
    displacement = past[:, -1] - past[:, -4]; norm = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement); heading[:, 0] = 1.0; valid = norm > 0.03; heading[valid] = displacement[valid] / norm[valid, None]
    c, s, origin = heading[:, 0], heading[:, 1], past[:, -1]
    def rotate(points):
        delta = points - origin[:, None]; result = np.empty_like(delta)
        result[..., 0] = delta[..., 0] * c[:, None] + delta[..., 1] * s[:, None]
        result[..., 1] = -delta[..., 0] * s[:, None] + delta[..., 1] * c[:, None]
        return result
    return rotate(past), rotate(future)


def kinematics(past, future, num, scenes):
    history_velocity = np.diff(past, axis=1); future_velocity = np.diff(np.concatenate((np.zeros((len(future), 1, 2), np.float32), future), axis=1), axis=1)
    early, late = past[:, 3] - past[:, 0], past[:, 7] - past[:, 4]
    denominator = np.linalg.norm(early, axis=-1) * np.linalg.norm(late, axis=-1) + 1e-6
    history_turn = np.arctan2(early[:, 0] * late[:, 1] - early[:, 1] * late[:, 0], (early * late).sum(-1))
    future_early, future_late = future[:, 3], future[:, 11] - future[:, 7]
    future_turn = np.arctan2(future_early[:, 0] * future_late[:, 1] - future_early[:, 1] * future_late[:, 0], (future_early * future_late).sum(-1))
    line = past[:, :1] + np.linspace(0, 1, 8, dtype=np.float32)[None, :, None] * (past[:, -1:] - past[:, :1])
    nonlinearity = np.linalg.norm(past - line, axis=-1).mean(-1)
    return {
        "history_speed": np.linalg.norm(history_velocity, axis=-1).mean(-1),
        "heading_change_abs": np.abs(history_turn),
        "acceleration": np.linalg.norm(np.diff(history_velocity, axis=1), axis=-1).mean(-1),
        "neighbour_count": np.asarray(num, np.float32) - 1.0,
        "observation_nonlinearity": nonlinearity,
        "future_turn_abs": np.abs(future_turn),
        "future_speed_change": np.linalg.norm(future_velocity[:, -3:], axis=-1).mean(-1) - np.linalg.norm(future_velocity[:, :3], axis=-1).mean(-1),
        "scene": np.asarray(scenes, np.int64),
    }


def infer(split, indices, seed, batch_size, device, diag, cfg, cls, geom):
    run = source_run(split, seed); info = json.loads((run / "split.json").read_text(encoding="utf-8"))
    dataset = geom.IndexedDataset(Path(info["data_path"]), Path(info["num_path"]), indices, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = diag.build(split, "baseline", seed, cfg, cls, device)
    values = diag.infer(model, loader, device); del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return values


def tail_arrays(values):
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1)
    ade, fde = error.mean(-1), error[..., -1]; order = np.argsort(-values["score"], axis=1); rows = np.arange(len(order))[:, None]
    top_ade, tail_ade = ade[rows, order[:, :10]], ade[rows, order[:, 10:]]
    top_fde, tail_fde = fde[rows, order[:, :10]], fde[rows, order[:, 10:]]
    best10_ade, best20_ade = top_ade.min(1), np.minimum(top_ade.min(1), tail_ade.min(1))
    best10_fde, best20_fde = top_fde.min(1), np.minimum(top_fde.min(1), tail_fde.min(1))
    local_ade, local_fde = tail_ade.argmin(1), tail_fde.argmin(1)
    return {
        "order": order, "ade": ade, "fde": fde,
        "gain_ade": best10_ade - best20_ade, "gain_fde": best10_fde - best20_fde,
        "best10_ade": best10_ade, "best20_ade": best20_ade, "best10_fde": best10_fde, "best20_fde": best20_fde,
        "tail_mode_ade": order[np.arange(len(order)), 10 + local_ade], "tail_mode_fde": order[np.arange(len(order)), 10 + local_fde],
    }


def gain_summary(gain, mask):
    value = gain[mask]; positive = value > 1e-8
    return {"samples": int(mask.sum()), "mean": float(value.mean()), "median": float(np.median(value)), "p75": float(np.quantile(value, .75)), "p90": float(np.quantile(value, .90)), "p95": float(np.quantile(value, .95)), "positive_fraction": float(positive.mean()), "positive_mean": float(value[positive].mean()) if positive.any() else 0.0}


def group_report(tail, labels, scenes):
    result = {"overall": {"ade": gain_summary(tail["gain_ade"], np.ones(len(labels), bool)), "fde": gain_summary(tail["gain_fde"], np.ones(len(labels), bool))}}
    for name, mask in (("social_needed", labels), ("social_not_needed", ~labels)):
        result[name] = {"ade": gain_summary(tail["gain_ade"], mask), "fde": gain_summary(tail["gain_fde"], mask)}
    result["by_scene"] = {str(scene): {"ade": gain_summary(tail["gain_ade"], scenes == scene), "fde": gain_summary(tail["gain_fde"], scenes == scene)} for scene in np.unique(scenes)}
    return result


def cluster_tail(future, significant, behaviour, dataset_indices):
    index = np.flatnonzero(significant); anchors = future[index][:, [3, 7, 11]].reshape(len(index), -1)
    scaler = StandardScaler().fit(anchors); features = scaler.transform(anchors); candidates = []
    sample_rng = np.random.default_rng(20260829); silhouette_index = sample_rng.choice(len(features), min(len(features), 5000), replace=False)
    for clusters in range(4, 9):
        labels_by_seed = [KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(features) for seed in (1, 7, 23)]
        silhouette = float(silhouette_score(features[silhouette_index], labels_by_seed[0][silhouette_index]))
        stability = float(np.mean([adjusted_rand_score(labels_by_seed[0], other) for other in labels_by_seed[1:]]))
        candidates.append({"clusters": clusters, "silhouette": silhouette, "stability_ari": stability, "minimum_cluster_fraction": float(np.bincount(labels_by_seed[0], minlength=clusters).min() / len(features))})
    selected_row = max(candidates, key=lambda row: (row["silhouette"] + 0.15 * row["stability_ari"]))
    clusters = selected_row["clusters"]; model = KMeans(n_clusters=clusters, random_state=1, n_init=20).fit(features); labels = model.labels_
    medoid_indices, cluster_rows = [], []
    for cluster in range(clusters):
        members = np.flatnonzero(labels == cluster); distance = np.linalg.norm(features[members] - model.cluster_centers_[cluster], axis=-1); local = members[distance.argmin()]; original = index[local]; medoid_indices.append(original)
        row = {"cluster": cluster, "samples": int(len(members)), "fraction": float(len(members) / len(features)), "medoid_dataset_index": int(dataset_indices[original]), "medoid_anchors": future[original, [3, 7, 11]].reshape(-1).tolist()}
        for key, value in behaviour.items():
            if key != "scene": row[f"mean_{key}"] = float(value[index[members]].mean())
        row["scene_distribution"] = {str(scene): float((value[index[members]] == scene).mean()) for scene in np.unique(behaviour["scene"]) for value in [behaviour["scene"]]}
        cluster_rows.append(row)
    return {"samples": int(len(index)), "tail_fraction": float(significant.mean()), "candidates": candidates, "selected": selected_row, "clusters": cluster_rows}, np.asarray(medoid_indices, np.int64)


def clustering_control(future, count, clusters):
    rng = np.random.default_rng(91); index = rng.choice(len(future), count, replace=False); anchors = future[index][:, [3, 7, 11]].reshape(count, -1); features = StandardScaler().fit_transform(anchors)
    labels = KMeans(n_clusters=clusters, random_state=1, n_init=20).fit_predict(features); sample = rng.choice(count, min(count, 5000), replace=False)
    return {"random_all_future_silhouette": float(silhouette_score(features[sample], labels[sample]))}


def medoid_proxy(values, future_local, validation_indices, medoid_futures):
    tail = tail_arrays(values); order, ade, fde = tail["order"], tail["ade"], tail["fde"]; rows = np.arange(len(order))[:, None]; clusters = len(medoid_futures)
    core_ade, core_fde = ade[rows, order[:, :10]], fde[rows, order[:, :10]]
    keep = max(10 - clusters, 0); kept_ade, kept_fde = ade[rows, order[:, 10:10 + keep]], fde[rows, order[:, 10:10 + keep]]
    target = future_local[validation_indices]; medoid_error = np.linalg.norm(medoid_futures[None] - target[:, None], axis=-1); medoid_ade, medoid_fde = medoid_error.mean(-1), medoid_error[..., -1]
    proxy_ade = np.concatenate((core_ade, kept_ade, medoid_ade), axis=1).min(1); proxy_fde = np.concatenate((core_fde, kept_fde, medoid_fde), axis=1).min(1)
    base_ade, base_fde = ade.min(1), fde.min(1)
    return {"candidate_count": int(10 + keep + clusters), "cluster_medoids": clusters, "kept_original_tail": keep, "baseline": {"min_ade_20": float(base_ade.mean()), "min_fde_20": float(base_fde.mean())}, "proxy": {"min_ade_20": float(proxy_ade.mean()), "min_fde_20": float(proxy_fde.mean())}, "improvement_pct": {"min_ade_20": float(100 * (1 - proxy_ade.mean() / base_ade.mean())), "min_fde_20": float(100 * (1 - proxy_fde.mean() / base_fde.mean()))}}


def process(split, seed, batch_size, device, diag, cfg, cls, geom, future_local, behaviour, labels_all):
    run = source_run(split, seed); arrays = np.load(run / "split_indices.npz"); train = np.asarray(arrays["train"], np.int64); validation = np.asarray(arrays["validation"], np.int64)
    print(f"[infer] {split}/train samples={len(train)}", flush=True); train_values = infer(split, train, seed, batch_size, device, diag, cfg, cls, geom); train_tail = tail_arrays(train_values)
    print(f"[infer] {split}/validation samples={len(validation)}", flush=True); val_values = infer(split, validation, seed, batch_size, device, diag, cfg, cls, geom); val_tail = tail_arrays(val_values)
    # Material tail need: enough absolute gain to matter, defined separately for ADE and FDE.
    significant = (train_tail["gain_ade"] >= 0.05) | (train_tail["gain_fde"] >= 0.10)
    cluster, medoid_index = cluster_tail(future_local[train], significant, {k: v[train] for k, v in behaviour.items()}, train)
    cluster["control"] = clustering_control(future_local[train], cluster["samples"], cluster["selected"]["clusters"])
    medoid_futures = future_local[train][medoid_index]
    proxy = medoid_proxy(val_values, future_local, validation, medoid_futures)
    structured = cluster["samples"] >= 300 and cluster["tail_fraction"] >= 0.05 and cluster["selected"]["silhouette"] >= 0.25 and cluster["selected"]["stability_ari"] >= 0.60 and cluster["selected"]["minimum_cluster_fraction"] >= 0.03
    proxy_pass = max(proxy["improvement_pct"].values()) >= 0.50
    result = {"split": split, "seed": seed, "tail_definition": "score ranks 11-20; ADE and FDE oracle gains computed independently", "significant_threshold": "ADE gain >=0.05 OR FDE gain >=0.10", "train": {"gain": group_report(train_tail, labels_all[train], behaviour["scene"][train]), "cluster": cluster}, "validation": {"gain": group_report(val_tail, labels_all[validation], behaviour["scene"][validation]), "hcb_oracle_proxy": proxy}, "gate": {"structured_tail_pass": structured, "hcb_proxy_pass": proxy_pass, "hcb1_training_allowed": bool(structured and proxy_pass)}}
    out = ROOT / "audit"; out.mkdir(parents=True, exist_ok=True); (out / f"{split}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[audit] {split} tail={cluster['tail_fraction']:.3f} silhouette={cluster['selected']['silhouette']:.3f} stability={cluster['selected']['stability_ari']:.3f} proxy={proxy['improvement_pct']} allowed={result['gate']['hcb1_training_allowed']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=1); parser.add_argument("--batch-size", type=int, default=128); args = parser.parse_args(); device = torch.device(args.device)
    raw = np.load(DATA / "eth_data_train.npy", mmap_mode="r"); num = np.load(DATA / "eth_num_train.npy", mmap_mode="r"); past, future = local_coordinates(raw); scenes = np.asarray(raw[:, 0, 0, 0], np.int64); behaviour = kinematics(past, future, num, scenes)
    diag = load("tca_cso_diag", CSO / "run_cso1_diagnostic.py"); cfg, cls = diag.prepare(device); geom = load("tca_geometry", GEOM)
    results = []
    for split in SPLITS:
        labels = np.asarray(np.load(CSRS / "teacher_cache" / f"{split}_seed_{args.seed}.npz", allow_pickle=False)["need_label"]).astype(bool)
        results.append(process(split, args.seed, args.batch_size, device, diag, cfg, cls, geom, future, behaviour, labels))
    combined = {"zero_training": True, "gate_rule": "both structured clustering and locked-validation medoid proxy >=0.5% on ADE or FDE", "hcb1_training_allowed": all(result["gate"]["hcb1_training_allowed"] for result in results), "results": results}
    (ROOT / "audit" / "combined_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"); print(f"[done] HCB1 allowed={combined['hcb1_training_allowed']}")


if __name__ == "__main__": main()
