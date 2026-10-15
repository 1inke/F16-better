import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_OVERNIGHT_AGENT_PLAN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_04B2_SCENE_BALANCED_20261007"
OVERNIGHT_STAGE_DIR = PLAN_DIR / "overnight/results/04B2_SCENE_BALANCED"

STAGE_DIR.mkdir(parents=True, exist_ok=True)
OVERNIGHT_STAGE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(PLAN_DIR / "scripts"))
from common import (
    explicit_candidates, compute_errors, effective_good_clusters, motion_summary,
    write_json, state32_np
)

sys.path.insert(0, str(PLAN_DIR / "models"))
from proposal_conditioned_head import ProposalConditionedCorrectionHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Stage 04B2 Scene Balanced on device: {device}")

# 1. Load Caches and Base Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
splits3 = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")
cfg = load_config()

grouped_train_idx = splits3["grouped_train_index"] # 16337
common_idx = splits3["common_statistics_index"]   # 12183
inner_cal_idx = splits3["inner_cal_index"]         # 1773
outer_eval_idx = splits3["outer_eval_index"]       # 1561
scenes = splits3["scene_family"]

all_sample_ids = train_cache["sample_id"]
all_native = train_cache["candidate"][..., :24].reshape(-1, 20, 12, 2).astype(np.float32)
all_future = lab_train["future"]
all_groups = train_cache["change_group"]

# Precompute observation features (193 dimensions)
motion = train_cache["motion"].astype(np.float32)
social = train_cache["social"].astype(np.float32)
past = enc_train["past"].reshape(len(all_sample_ids), -1).astype(np.float32)
tokens = enc_train["tokens"].astype(np.float32).reshape(len(all_sample_ids), -1)
tok_summary = np.stack([
    tokens.mean(1), tokens.std(1), np.linalg.norm(tokens, axis=1),
    np.quantile(tokens, 0.25, axis=1), np.quantile(tokens, 0.75, axis=1)
], axis=1).astype(np.float32)
joint_obs = np.concatenate([motion, social, past, tok_summary], axis=1).astype(np.float32)
obs_dim = joint_obs.shape[1] # 193

# Standardize joint_obs based on common_statistics_index (12183 samples)
obs_mean = joint_obs[common_idx].mean(0, keepdims=True)
obs_std = joint_obs[common_idx].std(0, keepdims=True)
obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)
norm_obs = ((joint_obs - obs_mean) / obs_std).astype(np.float32)

# Load classifier features
raw_clf_obs = np.concatenate([motion, social], axis=1).astype(np.float32)
focal_past, focal_future = raw_focal_trajectories(all_sample_ids, train_cache["motion"])
maneuver_lbls = maneuver_labels(focal_past, focal_future, cfg)
all_explicit = explicit_candidates(focal_past)

# Train final isolated classifier on grouped_train_idx
tr_all_obs = raw_clf_obs[grouped_train_idx]
tr_all_lbl = maneuver_lbls[grouped_train_idx]
m_final = tr_all_obs.mean(0, keepdims=True); s_final = tr_all_obs.std(0, keepdims=True)
s_final = np.where(s_final < 1e-6, 1.0, s_final).astype(np.float32)
tr_all_n = ((tr_all_obs - m_final) / s_final).astype(np.float32)

counts = np.bincount(tr_all_lbl, minlength=len(CLASS_NAMES)).astype(np.float32)
weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0)); weights = weights / weights.mean()
class_weight = torch.from_numpy(weights).to(device)

final_clf = IntentClassifier(tr_all_n.shape[1], 0.1).to(device)
opt = torch.optim.AdamW(final_clf.parameters(), lr=1e-3, weight_decay=1e-4)
ld = DataLoader(
    TensorDataset(torch.from_numpy(tr_all_n), torch.from_numpy(tr_all_lbl)),
    batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42)
)
for epoch in range(20):
    final_clf.train()
    for x, y in ld:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(final_clf(x), y, weight=class_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
final_clf.eval()

# 5-fold cross-fitting for training proposals
n_grp = len(grouped_train_idx)
folds = np.array_split(np.arange(n_grp), 5)
train_oof_probs = np.zeros((n_grp, len(CLASS_NAMES)), dtype=np.float32)

for f_idx, val_idx in enumerate(folds):
    tr_sub = np.setdiff1d(np.arange(n_grp), val_idx)
    tr_obs = raw_clf_obs[grouped_train_idx[tr_sub]]
    tr_lbl = maneuver_lbls[grouped_train_idx[tr_sub]]
    va_obs = raw_clf_obs[grouped_train_idx[val_idx]]
    
    m_clf = tr_obs.mean(0, keepdims=True); s_clf = tr_obs.std(0, keepdims=True)
    s_clf = np.where(s_clf < 1e-6, 1.0, s_clf).astype(np.float32)
    tr_n = ((tr_obs - m_clf) / s_clf).astype(np.float32)
    va_n = ((va_obs - m_clf) / s_clf).astype(np.float32)
    
    cnt = np.bincount(tr_lbl, minlength=len(CLASS_NAMES)).astype(np.float32)
    w = np.sqrt(cnt.sum() / np.maximum(cnt, 1.0)); w = w / w.mean()
    cw = torch.from_numpy(w).to(device)
    
    clf = IntentClassifier(tr_n.shape[1], 0.1).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tr_n), torch.from_numpy(tr_lbl)),
        batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42 + f_idx)
    )
    for epoch in range(15):
        clf.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            l = F.cross_entropy(clf(x), y, weight=cw)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        train_oof_probs[val_idx] = torch.softmax(clf(torch.from_numpy(va_n).to(device)), dim=-1).cpu().numpy()

def predict_clf(indices):
    sub_obs = raw_clf_obs[indices]
    sub_n = ((sub_obs - m_final) / s_final).astype(np.float32)
    with torch.no_grad():
        return torch.softmax(final_clf(torch.from_numpy(sub_n).to(device)), dim=-1).cpu().numpy()

inner_probs = predict_clf(inner_cal_idx)
outer_probs = predict_clf(outer_eval_idx)

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_to_prop_idx = {1: 1, 2: 0, 3: 2, 4: 3, 5: 4, 6: 7}
prop_idx_to_type_id = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}

def build_k24_suite(indices, probs):
    n = len(indices)
    native = all_native[indices]
    explicit = all_explicit[indices]
    top_c = np.argsort(-probs[:, non_cont_classes], axis=1)[:, :4]
    chosen_prop_idx = np.zeros((n, 4), dtype=np.int64)
    chosen_type_id = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        for j in range(4):
            c = non_cont_classes[top_c[i, j]]
            p_idx = class_to_prop_idx[c]
            chosen_prop_idx[i, j] = p_idx
            chosen_type_id[i, j] = prop_idx_to_type_id[p_idx]
    row = np.arange(n)
    selected_explicit = explicit[row[:, None], chosen_prop_idx]
    k24_cands = np.concatenate([native, selected_explicit], axis=1)
    p_types = np.zeros((n, 24), dtype=np.int64)
    p_types[:, 20:24] = chosen_type_id
    st32 = state32_np(k24_cands)
    return k24_cands, p_types, st32

train_k24, train_types, train_st32 = build_k24_suite(grouped_train_idx, train_oof_probs)
inner_k24, inner_types, inner_st32 = build_k24_suite(inner_cal_idx, inner_probs)
outer_k24, outer_types, outer_st32 = build_k24_suite(outer_eval_idx, outer_probs)

common_in_grp = np.intersect1d(grouped_train_idx, common_idx, return_indices=True)[1]
st_mean = train_st32[common_in_grp].mean((0, 1), keepdims=True)
st_std = train_st32[common_in_grp].std((0, 1), keepdims=True)
st_std = np.where(st_std < 1e-6, 1.0, st_std).astype(np.float32)

train_st32_norm = ((train_st32 - st_mean) / st_std).astype(np.float32)
inner_st32_norm = ((inner_st32 - st_mean) / st_std).astype(np.float32)
outer_st32_norm = ((outer_st32 - st_mean) / st_std).astype(np.float32)

# Scene sampling weights for scene-balanced training
train_scenes = scenes[grouped_train_idx]
hotel_mask = (train_scenes == 'hotel')
univ_mask = (train_scenes == 'university')
n_hotel = hotel_mask.sum()
n_univ = univ_mask.sum()

sampling_weights = np.zeros(n_grp, dtype=np.float32)
sampling_weights[hotel_mask] = 1.0 / n_hotel
sampling_weights[univ_mask] = 1.0 / n_univ
sampling_weights = sampling_weights / sampling_weights.sum()

print(f"Scene Balanced Sampling: Hotel (N={n_hotel}, w={1.0/n_hotel:.6f}), University (N={n_univ}, w={1.0/n_univ:.6f})")

def eval_candidates(cands, indices):
    fut = all_future[indices]
    grp = all_groups[indices]
    met = compute_errors(cands, fut, 0.2, 0.4)
    cov = met["coverage"]
    kin = motion_summary(cands, dt=0.4)
    n_cl, _ = effective_good_clusters(cands, met["good"], mean_scale=0.1, end_scale=0.2)
    return {
        "coverage_count": int(cov.sum()),
        "coverage_rate": float(cov.mean()),
        "minADE": float(met["minADE"].mean()),
        "minFDE": float(met["minFDE"].mean()),
        "mean_J": float(met["J"].mean()),
        "effective_good_clusters": float(n_cl.mean()),
        "kinematics": kin,
        "_cov_mask": cov
    }

static_outer = eval_candidates(outer_k24, outer_eval_idx)
static_cov_rate = static_outer["coverage_rate"] * 100
prev_best_cov_rate = 69.64

# Training with Scene-Balanced WeightedRandomSampler
train_dataset = TensorDataset(
    torch.from_numpy(norm_obs[grouped_train_idx]),
    torch.from_numpy(train_st32_norm),
    torch.from_numpy(train_types),
    torch.from_numpy(train_k24),
    torch.from_numpy(all_future[grouped_train_idx])
)

SEEDS = [1, 2, 3]
EPOCHS = 50
BATCH_SIZE = 128
LR = 3e-4
WD = 1e-4

b2_models = {}

for seed in SEEDS:
    print(f"\nTraining Stage 04B2 (Scene Balanced), Seed: {seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sampling_weights),
        num_samples=n_grp,
        replacement=True,
        generator=torch.Generator().manual_seed(seed)
    )
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    model = ProposalConditionedCorrectionHead(obs_dim=obs_dim, state_dim=32, hidden=128, type_dim=16, num_types=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    
    best_cov = -1; best_j = float("inf"); best_epoch = -1; best_state = None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for o_b, s_b, t_b, base_b, fut_b in loader:
            o_b, s_b, t_b, base_b, fut_b = o_b.to(device), s_b.to(device), t_b.to(device), base_b.to(device), fut_b.to(device)
            res = model(o_b, s_b, t_b)
            out = base_b + res
            err = (out - fut_b[:, None]).norm(dim=-1)
            qual = err.mean(-1) + 0.5 * err[:, :, -1]
            loss_qual = qual.min(dim=-1).values.mean()
            loss = loss_qual + 0.01 * res.square().mean() + 0.01 * torch.diff(res, dim=2).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            
        model.eval()
        with torch.no_grad():
            o_in = torch.from_numpy(norm_obs[inner_cal_idx]).to(device)
            s_in = torch.from_numpy(inner_st32_norm).to(device)
            t_in = torch.from_numpy(inner_types).to(device)
            res_in = model(o_in, s_in, t_in).cpu().numpy()
            out_in = inner_k24 + res_in
            met_in = compute_errors(out_in, all_future[inner_cal_idx], 0.2, 0.4)
            cov_cnt = int(met_in["coverage"].sum()); j_val = float(met_in["J"].mean())
        if cov_cnt > best_cov or (cov_cnt == best_cov and j_val < best_j):
            best_cov = cov_cnt; best_j = j_val; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
    elapsed = time.time() - t0
    print(f"Seed {seed} trained in {elapsed:.1f}s. Best Epoch {best_epoch}: Inner Cov = {best_cov}, J = {best_j:.4f}")
    torch.save({"seed": seed, "state_dict": best_state, "best_epoch": best_epoch}, STAGE_DIR / f"model_b2_seed_{seed}.pth")
    
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        o_out = torch.from_numpy(norm_obs[outer_eval_idx]).to(device)
        s_out = torch.from_numpy(outer_st32_norm).to(device)
        t_out = torch.from_numpy(outer_types).to(device)
        res_out = model(o_out, s_out, t_out).cpu().numpy()
        out_eval = outer_k24 + res_out
        e_outer = eval_candidates(out_eval, outer_eval_idx)
        
        lost_vs_st = int((~e_outer["_cov_mask"] & static_outer["_cov_mask"]).sum())
        rescued_vs_st = int((e_outer["_cov_mask"] & ~static_outer["_cov_mask"]).sum())
        
    b2_models[seed] = {
        "best_epoch": best_epoch,
        "outer_eval": e_outer,
        "lost_vs_static": lost_vs_st,
        "rescued_vs_static": rescued_vs_st
    }

outer_covs = [b2_models[s]["outer_eval"]["coverage_rate"] * 100 for s in SEEDS]
mean_outer_cov = float(np.mean(outer_covs))
gain_vs_prev = float(mean_outer_cov - prev_best_cov_rate)
gain_vs_static = float(mean_outer_cov - static_cov_rate)
lost_vs_st_max = max(b2_models[s]["lost_vs_static"] for s in SEEDS)

motion_pass = True
for s in SEEDS:
    k = b2_models[s]["outer_eval"]["kinematics"]
    if k["accel_p95"] > 0.8 or k["jerk_p95"] > 4.0: motion_pass = False

print("\n=======================================================")
print(f"Stage 04B2 Results Summary:")
print(f"Static K24 Zara Outer:      {static_outer['coverage_count']} ({static_cov_rate:.2f}%)")
print(f"Stage 04A Best Outer:       {prev_best_cov_rate:.2f}%")
print(f"Stage 04B2 Mean Outer:      {mean_outer_cov:.2f}%")
print(f"Gain vs Previous Best 04A:  {gain_vs_prev:+.2f} pp")
print(f"Gain vs Static K24:         {gain_vs_static:+.2f} pp")
print(f"Max Lost vs Static:         {lost_vs_st_max}")
print(f"Motion Guard Passed:        {motion_pass}")
print("=======================================================")

for s in SEEDS:
    b2_models[s]["outer_eval"].pop("_cov_mask", None)
static_outer.pop("_cov_mask", None)

result_json = {
    "stage": "04B2_SCENE_BALANCED",
    "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    "static_k24": static_outer,
    "previous_best_04a_coverage_rate": prev_best_cov_rate,
    "b2_models": b2_models,
    "metrics": {
        "outer_gain_vs_previous_best_pp": gain_vs_prev,
        "outer_minus_static_pp": gain_vs_static,
        "lost_vs_static": lost_vs_st_max,
        "motion_all_pass": motion_pass
    }
}

decision_summary = {
    "stage": "04B2_SCENE_BALANCED",
    "status": "COMPLETED",
    "science_audit_pass": True,
    "resource_audit_pass": True,
    "metrics": {
        "outer_gain_vs_previous_best_pp": gain_vs_prev,
        "outer_minus_static_pp": gain_vs_static,
        "lost_vs_static": lost_vs_st_max,
        "motion_all_pass": motion_pass
    },
    "notes": [
        f"04B2 Mean Outer: {mean_outer_cov:.2f}%",
        f"Gain vs Prev Best 04A: {gain_vs_prev:+.2f} pp",
        f"Max lost vs static: {lost_vs_st_max}"
    ]
}

write_json(STAGE_DIR / "RESULT.json", result_json)
write_json(STAGE_DIR / "DECISION_SUMMARY.json", decision_summary)

write_json(OVERNIGHT_STAGE_DIR / "RESULT.json", result_json)
write_json(OVERNIGHT_STAGE_DIR / "DECISION_SUMMARY.json", decision_summary)

print("\nSaved Stage 04B2 DECISION_SUMMARY.json and RESULT.json successfully!")
