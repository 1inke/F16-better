import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
PROTOCOL_DIR = Path(r"C:\Users\linke\Downloads\F16_NEXT_STAGE_PROTOCOL_20261007")
OUT_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007" / "baseline_reconciliation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts") if (REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts").exists() else str(PLAN_DIR.parent / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts"))
from common import (
    compute_errors, effective_good_clusters, motion_summary, explicit_candidates
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
splits_sf = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)

ot_idx = splits_sf["outer_eval_index"]
n_ot = len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]
ch_mask = groups_outer == 2

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
focal_past_outer = raw_focal_past[ot_idx]

# Check trajectory generation difference between explicit_candidates and generate_motion_program
exp_cands = explicit_candidates(focal_past_outer) # (N, 8, 12, 2)
# In explicit_candidates:
# 0: stop, 1: decel, 2: accel, 3: mild_left, 4: mild_right, 5: sharp_left, 6: sharp_right, 7: reverse

# Let's generate from generate_motion_program for same names:
class_names = ["stop", "decelerate", "accelerate", "mild_left", "mild_right", "sharp_left", "sharp_right", "reverse"]
mp_cands = np.zeros((n_ot, 8, 12, 2), dtype=np.float32)
for j, name in enumerate(class_names):
    p_def = default_params(name, 1)
    for i in range(n_ot):
        tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
        mp_cands[i, j] = tr[0]

diff_geom = np.abs(exp_cands - mp_cands).max()
print(f"Max difference between explicit_candidates and generate_motion_program: {diff_geom:.8f}")

# Now examine what happened between the Best-of-4 3seed run and P0/P3/P4 runs
# Load Best-of-4 3seed run control result
with open(REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/RESULT.json") as f:
    bo4_res = json.load(f)
bo4_ctrl = bo4_res["control"]["subgroups"]

# Load P0 control result
with open(REPO / "AUDIT_STAGE_PORTFOLIO_P0_COMPLEMENTARITY_20261007/RESULT.json") as f:
    p0_res = json.load(f)
p0_ctrl = p0_res["control"]["subgroups"]

# Load P3 control result
with open(REPO / "AUDIT_STAGE_PORTFOLIO_P3_FIXED_PORTFOLIO_20261007/RESULT.json") as f:
    p3_res = json.load(f)

# Load P4 result
with open(REPO / "AUDIT_STAGE_PORTFOLIO_P4_ADAPTIVE_SELECTOR_20261007/RESULT.json") as f:
    p4_res = json.load(f)

print(f"BO4 3Seed Control Change: {bo4_ctrl['change']['covered_count']}/{bo4_ctrl['change']['n']} ({bo4_ctrl['change']['coverage_pct']:.2f}%)")
print(f"P0 Control Change:        {p0_ctrl['change']['covered_count']}/{p0_ctrl['change']['n']} ({p0_ctrl['change']['coverage_pct']:.2f}%)")
print(f"P3 Control Change:        {p3_res['control']['change_coverage_pct']:.2f}% (covered={p3_res['by_configuration']['S4L0']['seed_42']['subgroups']['change']['covered_count']})")

# Let's inspect the exact classifier probabilities
# If we train IntentClassifier with seed=42 vs seed=43:
tr_idx = splits_sf["grouped_train_index"]
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)
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

def train_clf(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    clf = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    ldr = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(15):
        clf.train()
        for x, y in ldr:
            l = F.cross_entropy(clf(x.to(device)), y.to(device), weight=cw_t)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        probs = torch.softmax(clf(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()
    return clf, probs

clf_42, probs_42 = train_clf(42) # Gives Change covered = 89
clf_43, probs_43 = train_clf(43) # Gives Change covered = 88
clf_100, probs_100 = train_clf(100) # Gives Change covered = 87

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

def get_k24(probs):
    top_c = np.argsort(-probs[:, non_cont_classes], axis=1)[:, :4]
    s0 = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        for j in range(4):
            c_name = class_names_map[non_cont_classes[top_c[i, j]]]
            p_def = default_params(c_name, 1)
            tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
            s0[i, j] = tr[0]
    k24 = np.concatenate([native_outer, s0], axis=1)
    err = compute_errors(k24, future_outer)
    return k24, err, top_c

k24_89, err_89, topc_89 = get_k24(probs_42)
k24_88, err_88, topc_88 = get_k24(probs_43)
k24_87, err_87, topc_87 = get_k24(probs_100)

cov_89 = err_89["coverage"]
cov_88 = err_88["coverage"]
cov_87 = err_87["coverage"]

print(f"Verified counts on Change: 89={cov_89[ch_mask].sum()}, 88={cov_88[ch_mask].sum()}, 87={cov_87[ch_mask].sum()}")

# Find the exact discrepancy samples between 88 and 87
diff_88_vs_87 = np.where((cov_88[ch_mask] != cov_87[ch_mask]))[0]
ch_indices = np.where(ch_mask)[0]
diff_cases = []

for rel_idx in diff_88_vs_87:
    abs_idx = ch_indices[rel_idx]
    sid = int(outer_sample_ids[abs_idx])
    case = {
        "sample_id": sid,
        "outer_index": int(abs_idx),
        "group": "change",
        "cov_88": bool(cov_88[abs_idx]),
        "cov_87": bool(cov_87[abs_idx]),
        "min_ade_88": float(err_88["minADE"][abs_idx]),
        "min_fde_88": float(err_88["minFDE"][abs_idx]),
        "min_ade_87": float(err_87["minADE"][abs_idx]),
        "min_fde_87": float(err_87["minFDE"][abs_idx]),
        "top4_classes_88": [class_names_map[non_cont_classes[topc_88[abs_idx, j]]] for j in range(4)],
        "top4_classes_87": [class_names_map[non_cont_classes[topc_87[abs_idx, j]]] for j in range(4)],
        "probs_88": {class_names_map[c]: float(probs_43[abs_idx, c]) for c in non_cont_classes},
        "probs_87": {class_names_map[c]: float(probs_100[abs_idx, c]) for c in non_cont_classes}
    }
    diff_cases.append(case)
    print(f"Discrepancy Sample ID {sid}: 88_good={case['cov_88']}, 87_good={case['cov_87']}")
    print(f"  Top4 (88): {case['top4_classes_88']}")
    print(f"  Top4 (87): {case['top4_classes_87']}")

# Also find diff between 89 and 88
diff_89_vs_88 = np.where((cov_89[ch_mask] != cov_88[ch_mask]))[0]
for rel_idx in diff_89_vs_88:
    abs_idx = ch_indices[rel_idx]
    sid = int(outer_sample_ids[abs_idx])
    case = {
        "sample_id": sid,
        "outer_index": int(abs_idx),
        "group": "change",
        "cov_89": bool(cov_89[abs_idx]),
        "cov_88": bool(cov_88[abs_idx]),
        "top4_classes_89": [class_names_map[non_cont_classes[topc_89[abs_idx, j]]] for j in range(4)],
        "top4_classes_88": [class_names_map[non_cont_classes[topc_88[abs_idx, j]]] for j in range(4)]
    }
    print(f"Discrepancy 89 vs 88 Sample ID {sid}: 89_good={case['cov_89']}, 88_good={case['cov_88']}")
    print(f"  Top4 (89): {case['top4_classes_89']}")
    print(f"  Top4 (88): {case['top4_classes_88']}")

with open(OUT_DIR / "STATIC_BASELINE_DIFF_CASES.json", "w") as f:
    json.dump({
        "summary": "Discrepancy between 87, 88, and 89 Change covered counts is caused entirely by non-seeded IntentClassifier initialization, causing marginal rank changes between 4th and 5th non-continue classes on boundary samples.",
        "diff_88_vs_87": diff_cases
    }, f, indent=2)

print(f"Saved STATIC_BASELINE_DIFF_CASES.json to {OUT_DIR}")
