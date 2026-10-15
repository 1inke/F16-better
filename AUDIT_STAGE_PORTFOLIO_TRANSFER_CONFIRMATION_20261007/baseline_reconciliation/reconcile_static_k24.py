import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
OUT_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007" / "baseline_reconciliation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts") if (REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts").exists() else str(PLAN_DIR.parent / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts"))
from common import (
    compute_errors, effective_good_clusters, motion_summary
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running Baseline Reconciliation on: {device}")

# 1. Load Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
splits_sf = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr, n_ot = len(tr_idx), len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
focal_past_outer = raw_focal_past[ot_idx]

# Normalization for Classifier
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
m_clf = raw_clf_obs[tr_idx].mean(0, keepdims=True)
s_clf = raw_clf_obs[tr_idx].std(0, keepdims=True)
s_clf = np.where(s_clf < 1e-6, 1.0, s_clf).astype(np.float32)
tr_norm = ((raw_clf_obs[tr_idx] - m_clf) / s_clf).astype(np.float32)
ot_norm = ((raw_clf_obs[ot_idx] - m_clf) / s_clf).astype(np.float32)
tr_lbl = maneuver_lbls[tr_idx]

cnt = np.bincount(tr_lbl, minlength=len(CLASS_NAMES)).astype(np.float32)
cw = np.sqrt(cnt.sum() / np.maximum(cnt, 1.0)); cw = cw / cw.mean()
cw_t = torch.from_numpy(cw).to(device)

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

def build_static4_from_probs(probs):
    top_c = np.argsort(-probs[:, non_cont_classes], axis=1)[:, :4]
    s0 = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    names_rec = []
    for i in range(n_ot):
        names = [class_names_map[non_cont_classes[top_c[i, j]]] for j in range(4)]
        names_rec.append(names)
        for j in range(4):
            p_def = default_params(names[j], 1)
            tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
            s0[i, j] = tr[0]
    return s0, names_rec, top_c

# Version A: Model loaded from DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES (Global Frozen Canonical from PILOT SERIES)
global_saved_path = REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/results/models/intent_classifier.pth"
has_global_saved = global_saved_path.exists()
print(f"Global saved model exists: {has_global_saved}")

if has_global_saved:
    global_saved = torch.load(global_saved_path, map_location="cpu", weights_only=False)
    clf_global = IntentClassifier(global_saved["input_dim"], 0.0)
    clf_global.load_state_dict(global_saved["state_dict"])
    clf_global.to(device).eval()
    # Normalize with global saved mean and std
    ot_norm_global = ((raw_clf_obs[ot_idx] - global_saved["mean"]) / global_saved["std"]).astype(np.float32)
    with torch.no_grad():
        probs_global = torch.softmax(clf_global(torch.from_numpy(ot_norm_global).to(device)), dim=-1).cpu().numpy()
    s0_global, names_global, topc_global = build_static4_from_probs(probs_global)
    k24_global = np.concatenate([native_outer, s0_global], axis=1)
    err_global = compute_errors(k24_global, future_outer)
    cov_global = err_global["coverage"]
    ch_mask = groups_outer == 2
    print(f"Global Saved Model Control: Overall={cov_global.mean()*100:.2f}% ({cov_global.sum()}/1561), Change={cov_global[ch_mask].mean()*100:.2f}% ({cov_global[ch_mask].sum()}/211)")

# Version B: Retraining with seed=42 explicit RNG seed before model instantiation (Canonical Deterministic Isolated)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
clf_seeded42 = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
opt_seeded42 = torch.optim.AdamW(clf_seeded42.parameters(), lr=1e-3, weight_decay=1e-4)
ldr_seeded42 = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
for _ in range(15):
    clf_seeded42.train()
    for x, y in ldr_seeded42:
        l = F.cross_entropy(clf_seeded42(x.to(device)), y.to(device), weight=cw_t)
        opt_seeded42.zero_grad(set_to_none=True); l.backward(); opt_seeded42.step()
clf_seeded42.eval()
with torch.no_grad():
    probs_seeded42 = torch.softmax(clf_seeded42(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()
s0_seeded42, names_seeded42, topc_seeded42 = build_static4_from_probs(probs_seeded42)
k24_seeded42 = np.concatenate([native_outer, s0_seeded42], axis=1)
err_seeded42 = compute_errors(k24_seeded42, future_outer)
cov_seeded42 = err_seeded42["coverage"]
print(f"Seeded42 Model Control: Overall={cov_seeded42.mean()*100:.2f}% ({cov_seeded42.sum()}/1561), Change={cov_seeded42[ch_mask].mean()*100:.2f}% ({cov_seeded42[ch_mask].sum()}/211)")

# Check multiple seeds or unseeded runs to see why P0, P3, P4 had 88, 89, 87
test_seeds = [0, 1, 42, 43, 44, 100, 1234]
seed_results = {}
for s in test_seeds:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    clf_s = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
    opt_s = torch.optim.AdamW(clf_s.parameters(), lr=1e-3, weight_decay=1e-4)
    ldr_s = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(s))
    for _ in range(15):
        clf_s.train()
        for x, y in ldr_s:
            l = F.cross_entropy(clf_s(x.to(device)), y.to(device), weight=cw_t)
            opt_s.zero_grad(set_to_none=True); l.backward(); opt_s.step()
    clf_s.eval()
    with torch.no_grad():
        probs_s = torch.softmax(clf_s(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()
    s0_s, names_s, _ = build_static4_from_probs(probs_s)
    k24_s = np.concatenate([native_outer, s0_s], axis=1)
    err_s = compute_errors(k24_s, future_outer)
    cov_s = err_s["coverage"]
    tot_cov = int(cov_s.sum())
    ch_cov = int(cov_s[ch_mask].sum())
    seed_results[s] = {"overall": tot_cov, "change": ch_cov}
    print(f"Seed {s:4d} IntentClassifier: Overall={tot_cov}/1561 ({tot_cov/1561*100:.2f}%), Change={ch_cov}/211 ({ch_cov/211*100:.2f}%)")

with open(OUT_DIR / "SEED_SENSITIVITY.json", "w") as f:
    json.dump(seed_results, f, indent=2)
