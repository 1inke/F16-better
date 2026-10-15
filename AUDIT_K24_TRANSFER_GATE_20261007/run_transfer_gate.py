import sys, time, json, math, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PROTOCOL_DIR = Path(r"C:\Users\linke\Downloads\F16_NEXT_STAGE_PROTOCOL_20261007")
AUDIT_DIR = REPO / "AUDIT_K24_TRANSFER_GATE_20261007"
RAW_TRAIN_PATH = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse\eth_data_train.npy")

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES, macro_f1
)

sys.path.insert(0, str(PROTOCOL_DIR / "scripts"))
from common import (
    explicit_candidates, compute_errors, effective_good_clusters, motion_summary,
    write_json
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing K24_TRANSFER_GATE on device: {device}")

# 1. Load Caches and Base Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
cfg = load_config()

# Precompute past & future for all 28,791 training samples
all_sample_ids = train_cache["sample_id"]
all_raw_obs = np.concatenate((train_cache["motion"], train_cache["social"]), axis=1).astype(np.float32)
all_past, all_future = raw_focal_trajectories(all_sample_ids, train_cache["motion"])
all_labels = maneuver_labels(all_past, all_future, cfg)
all_native = train_cache["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float32)
all_logits = train_cache["logits"].astype(np.float32)
all_groups = train_cache["change_group"]

# 2. Load Global Frozen Intent Classifier
global_saved = torch.load(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/results/models/intent_classifier.pth",
                          map_location="cpu", weights_only=False)
model_global = IntentClassifier(global_saved["input_dim"], 0.0)
model_global.load_state_dict(global_saved["state_dict"])
model_global.to(device).eval()
global_mean = global_saved["mean"]
global_std = global_saved["std"]

def train_isolated_classifier(train_obs, train_lbls, val_obs, val_lbls, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mean = train_obs.mean(0, keepdims=True).astype(np.float32)
    std = train_obs.std(0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    
    val_norm = ((val_obs - mean) / std).astype(np.float32)
    train_norm = ((train_obs - mean) / std).astype(np.float32)
    
    counts = np.bincount(train_lbls, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_weight = torch.from_numpy(weights).to(device)
    
    model = IntentClassifier(train_norm.shape[1], cfg["intent_dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["intent_learning_rate"],
                                  weight_decay=cfg["intent_weight_decay"])
    
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(train_lbls)),
        batch_size=cfg["intent_batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0
    )
    
    best_key = (-float("inf"), float("inf"))
    best_state = None
    best_epoch = -1
    history = []
    
    t0 = time.time()
    for epoch in range(1, cfg["intent_epochs"] + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            
        model.eval()
        with torch.no_grad():
            logits_val = model(torch.from_numpy(val_norm).to(device))
            probs_val = torch.softmax(logits_val, dim=-1).cpu().numpy()
            pred_val = probs_val.argmax(1)
            f1 = macro_f1(val_lbls, pred_val)
            ce = float(-np.log(probs_val[np.arange(len(val_lbls)), val_lbls].clip(1e-8)).mean())
            key = (f1, -ce)
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_macro_f1": float(f1),
                "val_ce": float(ce)
            })
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_key[0]),
        "training_duration_seconds": float(elapsed),
        "history": history
    }

def get_proposals_top4(probs):
    non_cont_classes = [1, 2, 3, 4, 5, 6]
    class_to_prop = {1: 1, 2: 0, 3: 2, 4: 3, 5: 4, 6: 7}
    top_c = np.argsort(-probs[:, non_cont_classes], axis=1)[:, :4]
    props = np.zeros((len(probs), 4), dtype=np.int64)
    for i in range(len(probs)):
        for j in range(4):
            props[i, j] = class_to_prop[non_cont_classes[top_c[i, j]]]
    return props

def get_proposals_prior_weighted(probs):
    prop_scores = np.zeros((len(probs), 8), dtype=np.float32)
    prop_scores[:, 0] = probs[:, 2]        # stop
    prop_scores[:, 1] = probs[:, 1]        # decel
    prop_scores[:, 2] = probs[:, 3] * 1.2  # accel
    prop_scores[:, 3] = probs[:, 4] * 2.0  # mild_left
    prop_scores[:, 4] = probs[:, 5] * 2.0  # mild_right
    prop_scores[:, 5] = probs[:, 4] * 0.1  # sharp_left
    prop_scores[:, 6] = probs[:, 5] * 0.1  # sharp_right
    prop_scores[:, 7] = probs[:, 6]        # reverse
    return np.argsort(-prop_scores, axis=1)[:, :4]

# Define Datasets to Evaluate
datasets_def = [
    {
        "id": "zara_scene_family_outer",
        "name": "Zara Scene-Family Outer",
        "isolation": "scene_family_unseen",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz",
        "outer_key": "outer_eval_index",
        "train_key": "grouped_train_index",
        "cal_key": "inner_cal_index",
        "has_zara_in_global": True
    },
    {
        "id": "target_track_outer",
        "name": "Target-Track Outer",
        "isolation": "pedestrian_id_unseen",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz",
        "outer_key": "outer_eval_index",
        "train_key": "grouped_train_index",
        "cal_key": "inner_cal_index",
        "has_zara_in_global": False
    },
    {
        "id": "recording_time_outer",
        "name": "Recording-Time Outer",
        "isolation": "time_block_unseen",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/splits.npz",
        "outer_key": "outer_eval_index",
        "train_key": "grouped_train_index",
        "cal_key": "inner_cal_index",
        "has_zara_in_global": False
    },
    {
        "id": "original_cal",
        "name": "Original CAL (Held-Out)",
        "isolation": "internal_calibration_held_out",
        "split_file": None,
        "cal_range": (25912, 28791),
        "has_zara_in_global": False
    }
]

group_names = {0: "stable", 1: "transition", 2: "change"}
results_all = {}
training_histories = {}

for ds in datasets_def:
    ds_id = ds["id"]
    ds_name = ds["name"]
    print(f"\n=======================================================")
    print(f"Evaluating Dataset: {ds_name} ({ds_id})")
    print(f"=======================================================")
    
    if ds["split_file"] is not None:
        splits = np.load(ds["split_file"])
        eval_idx = splits[ds["outer_key"]]
        train_idx = splits[ds["train_key"]]
        cal_idx = splits[ds["cal_key"]]
        has_isolated_training = True
    else:
        eval_idx = np.arange(ds["cal_range"][0], ds["cal_range"][1])
        train_idx = np.arange(0, ds["cal_range"][0])
        cal_idx = None
        has_isolated_training = False # CAL was already held out from global classifier training (0..25912)
        
    n_eval = len(eval_idx)
    print(f"Eval samples: {n_eval}")
    
    # Extract subset arrays
    d_native = all_native[eval_idx]
    d_future = all_future[eval_idx]
    d_past = all_past[eval_idx]
    d_logits = all_logits[eval_idx]
    d_groups = all_groups[eval_idx]
    d_obs = all_raw_obs[eval_idx]
    row = np.arange(n_eval)
    
    # Compute 8 explicit proposals
    d_explicit = explicit_candidates(d_past)
    
    # 1. Native20 reference
    met20 = compute_errors(d_native, d_future, 0.2, 0.4)
    cov20 = met20["coverage"]
    c20_count = int(cov20.sum())
    print(f"Native20 Coverage: {c20_count} / {n_eval} ({c20_count / n_eval * 100:.2f}%)")
    
    # 2. Augmented28 (K28) upper bound
    c_k28 = np.concatenate([d_native, d_explicit], axis=1)
    met28 = compute_errors(c_k28, d_future, 0.2, 0.4)
    cov28 = met28["coverage"]
    c28_count = int(cov28.sum())
    res28 = int((cov28 & ~cov20).sum())
    lost28 = int((~cov28 & cov20).sum())
    assert lost28 == 0, f"Lost28 must be 0, got {lost28}"
    print(f"Augmented28 Coverage: {c28_count} / {n_eval} ({c28_count / n_eval * 100:.2f}%), Rescued: {res28}, Lost: {lost28}")
    
    # 3. Global Intent Classifier inference
    val_norm_glob = ((d_obs - global_mean) / global_std).astype(np.float32)
    with torch.no_grad():
        probs_glob = torch.softmax(model_global(torch.from_numpy(val_norm_glob).to(device)), dim=-1).cpu().numpy()
        
    top4_glob = get_proposals_top4(probs_glob)
    prior4_glob = get_proposals_prior_weighted(probs_glob)
    
    # 4. Isolated Intent Classifier (if applicable)
    if has_isolated_training:
        print(f"Training Isolated IntentClassifier on {len(train_idx)} samples (Zero outer exposure)...")
        iso_model, iso_mean, iso_std, iso_hist = train_isolated_classifier(
            all_raw_obs[train_idx], all_labels[train_idx],
            all_raw_obs[cal_idx], all_labels[cal_idx]
        )
        training_histories[ds_id] = iso_hist
        print(f"Isolated Classifier trained in {iso_hist['training_duration_seconds']:.2f}s, best cal F1 = {iso_hist['best_val_macro_f1']:.4f}")
        
        val_norm_iso = ((d_obs - iso_mean) / iso_std).astype(np.float32)
        with torch.no_grad():
            probs_iso = torch.softmax(iso_model(torch.from_numpy(val_norm_iso).to(device)), dim=-1).cpu().numpy()
        top4_iso = get_proposals_top4(probs_iso)
        prior4_iso = get_proposals_prior_weighted(probs_iso)
    else:
        # For original CAL, global classifier was trained on 0..25912, so it is already isolated!
        probs_iso = probs_glob
        top4_iso = top4_glob
        prior4_iso = prior4_glob
        training_histories[ds_id] = {
            "best_epoch": global_saved.get("best_epoch", 24),
            "best_val_macro_f1": global_saved.get("best_cal_macro_f1", 0.384),
            "note": "Global classifier trained strictly on FIT (0..25912), held out CAL (25912..28791)."
        }
        
    # Construct candidate suites to evaluate
    dup_idx = np.argsort(-d_logits, axis=1)[:, :8]
    dup8 = d_native[row[:, None], dup_idx]
    
    candidate_suites = {
        "native20": d_native,
        "null_duplicate8_K28": np.concatenate([d_native, dup8], axis=1),
        "K24_global_top4_classes": np.concatenate([d_native, d_explicit[row[:, None], top4_glob]], axis=1),
        "K24_global_prior_weighted": np.concatenate([d_native, d_explicit[row[:, None], prior4_glob]], axis=1),
        "K24_isolated_top4_classes": np.concatenate([d_native, d_explicit[row[:, None], top4_iso]], axis=1),
        "K24_isolated_prior_weighted": np.concatenate([d_native, d_explicit[row[:, None], prior4_iso]], axis=1),
        "augmented28_full": c_k28
    }
    
    # Null 2: Jitter8 controls
    for sigma in [0.01, 0.02, 0.05]:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, sigma, size=dup8.shape).astype(np.float32)
        candidate_suites[f"null_jitter8_{int(sigma*100):02d}cm_K28"] = np.concatenate([d_native, dup8 + noise], axis=1)
        
    ds_results = {}
    for c_name, c_cand in candidate_suites.items():
        c_met = compute_errors(c_cand, d_future, 0.2, 0.4)
        c_cov = c_met["coverage"]
        c_cov_count = int(c_cov.sum())
        c_rescued = int((c_cov & ~cov20).sum())
        c_lost = int((~c_cov & cov20).sum())
        
        # Verify strict invariant
        assert c_lost == 0, f"Protocol invariant violated: lost must be 0 for {c_name}, got {c_lost}"
        
        # Subgroup metrics
        grp_res = {}
        for g_code, g_name in group_names.items():
            g_mask = (d_groups == g_code)
            gc_cov = c_cov[g_mask]
            g_ref = cov20[g_mask]
            grp_res[g_name] = {
                "n": int(g_mask.sum()),
                "coverage_count": int(gc_cov.sum()),
                "coverage_rate": float(gc_cov.mean()),
                "rescued": int((gc_cov & ~g_ref).sum()),
                "lost": int((~gc_cov & g_ref).sum()),
                "minADE": float(c_met["minADE"][g_mask].mean()),
                "minFDE": float(c_met["minFDE"][g_mask].mean()),
                "mean_J": float(c_met["J"][g_mask].mean())
            }
            
        # Kinematics & clusters
        kin = motion_summary(c_cand, dt=0.4)
        n_clusters, n_distinct = effective_good_clusters(c_cand, c_met["good"], mean_scale=0.1, end_scale=0.2)
        
        capture_pct = float(c_rescued / res28 * 100) if res28 > 0 else 0.0
        
        ds_results[c_name] = {
            "candidate_count_K": int(c_cand.shape[1]),
            "coverage_count": c_cov_count,
            "coverage_rate": float(c_cov.mean()),
            "coverage_gain_count": int(c_cov_count - c20_count),
            "coverage_gain_pp": float((c_cov.mean() - cov20.mean()) * 100),
            "rescued_count": c_rescued,
            "lost_count": c_lost,
            "capture_rate_of_K28_rescue": capture_pct,
            "minADE": float(c_met["minADE"].mean()),
            "minFDE": float(c_met["minFDE"].mean()),
            "mean_J": float(c_met["J"].mean()),
            "early_error": float(c_met["early"].mean()),
            "middle_error": float(c_met["middle"].mean()),
            "late_error": float(c_met["late"].mean()),
            "effective_good_clusters": {
                "mean_good_clusters": float(n_clusters.mean()),
                "mean_distinct_clusters": float(n_distinct.mean())
            },
            "kinematics": kin,
            "groups": grp_res
        }
        print(f"  {c_name:30s} -> Cov: {c_cov_count:4d} ({c_cov.mean()*100:5.2f}%), Rescue: {c_rescued:3d}, Lost: {c_lost:2d}, CapK28: {capture_pct:5.1f}%, J: {c_met['J'].mean():.4f}")
        
    results_all[ds_id] = {
        "dataset_name": ds_name,
        "n_samples": n_eval,
        "isolation_type": ds["isolation"],
        "has_zara_in_global_model": ds["has_zara_in_global"],
        "has_isolated_training": has_isolated_training,
        "results": ds_results
    }

# 5. Output Results and Audit
final_output = {
    "audit_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    "scope": "K24_TRANSFER_GATE across 4 isolation sets (Zara outer, Target-track outer, Recording-time outer, Original CAL)",
    "datasets": results_all
}

write_json(AUDIT_DIR / "RESULT.json", final_output)
write_json(AUDIT_DIR / "training_history.json", training_histories)
print("\nAll experiments in K24_TRANSFER_GATE executed and RESULT.json saved successfully!")
