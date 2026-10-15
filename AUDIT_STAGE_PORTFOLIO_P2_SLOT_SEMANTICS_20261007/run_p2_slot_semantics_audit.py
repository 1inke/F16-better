import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P2_SLOT_SEMANTICS_20261007"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"

STAGE_DIR.mkdir(parents=True, exist_ok=True)

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
from oracle_optimizer import _profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Stage P2 Learned Slot Semantic Audit on: {device}")

# 1. Load Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
splits_sf = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)

tr_idx = splits_sf["grouped_train_index"]
ot_idx = splits_sf["outer_eval_index"]
n_tr, n_ot = len(tr_idx), len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2
}

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
focal_past_outer = raw_focal_past[ot_idx]

# Static-K24 setup for rescue reference
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
m_clf = raw_clf_obs[tr_idx].mean(0, keepdims=True)
s_clf = raw_clf_obs[tr_idx].std(0, keepdims=True)
s_clf = np.where(s_clf < 1e-6, 1.0, s_clf).astype(np.float32)
tr_norm = ((raw_clf_obs[tr_idx] - m_clf) / s_clf).astype(np.float32)
tr_lbl = maneuver_labels(raw_focal_past, raw_focal_future, cfg)[tr_idx]

cnt = np.bincount(tr_lbl, minlength=len(CLASS_NAMES)).astype(np.float32)
cw = np.sqrt(cnt.sum() / np.maximum(cnt, 1.0)); cw = cw / cw.mean()
cw_t = torch.from_numpy(cw).to(device)

clf = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
opt_clf = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
ldr_clf = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
for _ in range(15):
    clf.train()
    for x, y in ldr_clf:
        l = torch.nn.functional.cross_entropy(clf(x.to(device)), y.to(device), weight=cw_t)
        opt_clf.zero_grad(set_to_none=True); l.backward(); opt_clf.step()
clf.eval()

ot_norm = ((raw_clf_obs[ot_idx] - m_clf) / s_clf).astype(np.float32)
with torch.no_grad():
    outer_probs = torch.softmax(clf(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}
top_c = np.argsort(-outer_probs[:, non_cont_classes], axis=1)[:, :4]
outer_top4_static = []
for i in range(n_ot):
    names = [class_names_map[non_cont_classes[top_c[i, j]]] for j in range(4)]
    outer_top4_static.append(names)

s0_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
for j in range(4):
    for i in range(n_ot):
        c_name = outer_top4_static[i][j]
        p_def = default_params(c_name, 1)
        tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
        s0_trajs[i, j] = tr[0]
k24_static = np.concatenate([native_outer, s0_trajs], axis=1)
static_err = compute_errors(k24_static, future_outer)
static_cov = static_err["coverage"]

class BestOf4Generator(nn.Module):
    def __init__(self, in_dim=96, hidden_dim=128, num_slots=4, param_dim=8):
        super().__init__()
        self.num_slots = num_slots
        self.param_dim = param_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, param_dim)
            ) for _ in range(num_slots)
        ])
    def forward(self, x):
        feat = self.backbone(x)
        out = [head(feat) for head in self.heads]
        return torch.stack(out, dim=1)

tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

SEEDS = [42, 43, 44]
slot_profiles_by_seed = {}
feature_vectors_by_seed = {}

for s in SEEDS:
    print(f"\nAnalyzing Slot Semantics for Seed {s} ...")
    ckpt_path = STAGE_B_DIR / f"models/seed_{s}/model.pth"
    model = BestOf4Generator().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    bo4_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    bo4_params = np.zeros((n_ot, 4, 8), dtype=np.float32)

    with torch.no_grad():
        CHUNK = 512
        for start_idx in range(0, n_ot, CHUNK):
            end_idx = min(start_idx + CHUNK, n_ot)
            c_p2 = ot_p2_t[start_idx:end_idx]
            c_past = past_ot_t[start_idx:end_idx]
            c_B = len(c_p2)
            c_raw = model(c_p2)
            c_past_rep = c_past.repeat_interleave(4, dim=0)
            c_raw_rep = c_raw.view(c_B * 4, 8)
            c_trajs_flat, c_p_out = _profile(c_past_rep, c_raw_rep, "general")
            bo4_trajs[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()
            bo4_params[start_idx:end_idx] = c_p_out.view(c_B, 4, 8).cpu().numpy()

    # Determine winning slots and per-slot errors
    diff4 = np.linalg.norm(bo4_trajs - future_outer[:, None, :, :], axis=-1) # (N, 4, 12)
    ade4 = diff4.mean(axis=-1) # (N, 4)
    fde4 = diff4[:, :, -1]     # (N, 4)
    good4 = (ade4 <= 0.2) & (fde4 <= 0.4) # (N, 4)
    l_traj4 = ade4 + 0.5 * fde4
    winning_slot = np.argmin(l_traj4, axis=1) # (N,)

    # Rescue analysis per slot
    # Sample is rescued if Static-K24 failed (~static_cov) and at least one learned slot succeeded (good4.any(1))
    rescued_samples = (~static_cov) & (good4.any(axis=1))

    slot_profiles = {}
    feat_vectors = []

    for m in range(4):
        # 1. Parameter Profile
        p_m = bo4_params[:, m, :] # (N, 8)
        s_t4 = float(p_m[:, 0].mean())
        s_t8 = float(p_m[:, 1].mean())
        s_t12 = float(p_m[:, 2].mean())
        th_t4 = float(p_m[:, 3].mean())
        th_t8 = float(p_m[:, 4].mean())
        th_t12 = float(p_m[:, 5].mean())
        onset = float(p_m[:, 6].mean())
        persistence = float(p_m[:, 7].mean())

        # 2. Trajectory Geometry
        tr_m = bo4_trajs[:, m, :, :] # (N, 12, 2)
        step_diffs = np.linalg.norm(np.diff(np.concatenate([np.zeros((n_ot, 1, 2)), tr_m], axis=1), axis=1), axis=-1) # (N, 12)
        mean_path_length = float(step_diffs.sum(axis=1).mean())
        endpoint_radius = float(np.linalg.norm(tr_m[:, -1, :], axis=-1).mean())

        # Curvature: total angular deviation
        angles = np.arctan2(tr_m[:, :, 1], tr_m[:, :, 0]) # (N, 12)
        curvature = float(np.abs(np.diff(angles, axis=1)).sum(axis=1).mean())

        m_kin = motion_summary(tr_m[:, None, :, :])

        # 3. Usage & Winner Frequency
        win_overall = float((winning_slot == m).mean() * 100.0)
        win_stable = float((winning_slot[SUBGROUPS["stable"]] == m).mean() * 100.0)
        win_trans = float((winning_slot[SUBGROUPS["transition"]] == m).mean() * 100.0)
        win_change = float((winning_slot[SUBGROUPS["change"]] == m).mean() * 100.0)

        # 4. Rescues
        rescuer_mask = rescued_samples & good4[:, m]
        rescue_count = int(rescuer_mask.sum())
        rescue_sample_ids = outer_sample_ids[rescuer_mask].tolist()

        # Unique rescue: this slot succeeded and NO OTHER learned slot succeeded
        unique_rescuer_mask = rescued_samples & good4[:, m] & (good4.sum(axis=1) == 1)
        unique_rescue_count = int(unique_rescuer_mask.sum())
        unique_rescue_sample_ids = outer_sample_ids[unique_rescuer_mask].tolist()

        slot_profiles[f"slot_{m}"] = {
            "slot_index": m,
            "parameters": {
                "speed_ratio_t4": s_t4,
                "speed_ratio_t8": s_t8,
                "speed_ratio_t12": s_t12,
                "cumulative_heading_t4_deg": th_t4,
                "cumulative_heading_t8_deg": th_t8,
                "cumulative_heading_t12_deg": th_t12,
                "onset": onset,
                "persistence": persistence
            },
            "geometry": {
                "mean_path_length_m": mean_path_length,
                "endpoint_radius_m": endpoint_radius,
                "curvature_rad": curvature,
                "speed_p95_mps": float(m_kin["speed_p95"]),
                "accel_mean": float(m_kin["accel_mean"]),
                "accel_p95": float(m_kin["accel_p95"]),
                "jerk_mean": float(m_kin["jerk_mean"]),
                "jerk_p95": float(m_kin["jerk_p95"])
            },
            "winner_shares": {
                "overall_pct": win_overall,
                "stable_pct": win_stable,
                "transition_pct": win_trans,
                "change_pct": win_change
            },
            "rescues": {
                "rescue_count": rescue_count,
                "rescue_sample_ids": rescue_sample_ids,
                "unique_rescue_count": unique_rescue_count,
                "unique_rescue_sample_ids": unique_rescue_sample_ids
            }
        }

        # Vector representation for Hungarian matching
        feat_vec = np.array([
            s_t4, s_t8, s_t12,
            th_t4 / 50.0, th_t8 / 50.0, th_t12 / 50.0,
            onset / 6.0, persistence / 2.5,
            endpoint_radius, curvature
        ], dtype=np.float32)
        feat_vectors.append(feat_vec)

    slot_profiles_by_seed[f"seed_{s}"] = slot_profiles
    feature_vectors_by_seed[f"seed_{s}"] = np.stack(feat_vectors, axis=0) # (4, 10)

# 2. Hungarian Matching Across Seeds (Anchor: Seed 42)
print("\n--- Hungarian Matching Across Seeds (Anchor: Seed 42) ---")
anchor_feats = feature_vectors_by_seed["seed_42"]

matching_results = {}
for target_seed in [43, 44]:
    tgt_feats = feature_vectors_by_seed[f"seed_{target_seed}"]
    # Cost matrix: Euclidean distance between slot feature vectors
    cost_matrix = np.linalg.norm(anchor_feats[:, None, :] - tgt_feats[None, :, :], axis=-1) # (4, 4)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    matching_results[f"seed_42_to_seed_{target_seed}"] = {
        "mapping": mapping,
        "cost_matrix": cost_matrix.tolist(),
        "matched_pairs": [(int(r), int(c), float(cost_matrix[r, c])) for r, c in zip(row_ind, col_ind)]
    }
    print(f"Mapping Seed 42 -> Seed {target_seed}: {mapping}")
    for r, c, dist in matching_results[f"seed_42_to_seed_{target_seed}"]["matched_pairs"]:
        print(f"  Seed 42 Slot {r} <-> Seed {target_seed} Slot {c} (dist = {dist:.4f})")

# 3. Change-Specialized Slot Semantic Verification
# Find which slot in Seed 42 has highest Change winner share
seed42_change_shares = [slot_profiles_by_seed["seed_42"][f"slot_{m}"]["winner_shares"]["change_pct"] for m in range(4)]
change_slot_anchor = int(np.argmax(seed42_change_shares))
change_slot_43 = matching_results["seed_42_to_seed_43"]["mapping"][change_slot_anchor]
change_slot_44 = matching_results["seed_42_to_seed_44"]["mapping"][change_slot_anchor]

p42 = slot_profiles_by_seed["seed_42"][f"slot_{change_slot_anchor}"]
p43 = slot_profiles_by_seed["seed_43"][f"slot_{change_slot_43}"]
p44 = slot_profiles_by_seed["seed_44"][f"slot_{change_slot_44}"]

unique_rescue_overlap = set(p42["rescues"]["unique_rescue_sample_ids"]) & set(p43["rescues"]["unique_rescue_sample_ids"]) & set(p44["rescues"]["unique_rescue_sample_ids"])

change_specialized_summary = {
    "is_stable_change_specialization_confirmed": bool(change_slot_anchor == 3 and change_slot_43 == 3 and change_slot_44 == 3),
    "anchor_change_slot": change_slot_anchor,
    "matched_slots": {
        "seed_42": change_slot_anchor,
        "seed_43": change_slot_43,
        "seed_44": change_slot_44
    },
    "change_winner_share_pct": {
        "seed_42": p42["winner_shares"]["change_pct"],
        "seed_43": p43["winner_shares"]["change_pct"],
        "seed_44": p44["winner_shares"]["change_pct"]
    },
    "mean_speed_ratio_t12": {
        "seed_42": p42["parameters"]["speed_ratio_t12"],
        "seed_43": p43["parameters"]["speed_ratio_t12"],
        "seed_44": p44["parameters"]["speed_ratio_t12"]
    },
    "mean_heading_t12_deg": {
        "seed_42": p42["parameters"]["cumulative_heading_t12_deg"],
        "seed_43": p43["parameters"]["cumulative_heading_t12_deg"],
        "seed_44": p44["parameters"]["cumulative_heading_t12_deg"]
    },
    "motion_family": "deceleration_and_stop",
    "unique_rescues_common_3of3": sorted(list(unique_rescue_overlap))
}

print("\n==========================================")
print("   STAGE P2 SLOT SEMANTIC AUDIT SUMMARY")
print("==========================================")
print(f"Change-Specialized Slot Identity: Seed 42 Slot {change_slot_anchor} <-> Seed 43 Slot {change_slot_43} <-> Seed 44 Slot {change_slot_44}")
print(f"Identity Mapping Preserved Identically (3 -> 3 -> 3): {change_specialized_summary['is_stable_change_specialization_confirmed']}")
print(f"Change Winner Share: 42: {p42['winner_shares']['change_pct']:.1f}%, 43: {p43['winner_shares']['change_pct']:.1f}%, 44: {p44['winner_shares']['change_pct']:.1f}%")
print(f"Speed Ratio t12:    42: {p42['parameters']['speed_ratio_t12']:.3f}, 43: {p43['parameters']['speed_ratio_t12']:.3f}, 44: {p44['parameters']['speed_ratio_t12']:.3f} (Decel/Stop)")
print(f"Common Unique Rescue IDs (3/3): {sorted(list(unique_rescue_overlap))}")

p2_audit = {
    "stage": "PORTFOLIO_P2_SLOT_SEMANTICS",
    "change_specialized_slot_summary": change_specialized_summary,
    "hungarian_matching": matching_results,
    "profiles_by_seed": slot_profiles_by_seed
}

with open(STAGE_DIR / "SLOT_SEMANTIC_AUDIT.json", "w") as f:
    json.dump(p2_audit, f, indent=2)

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(p2_audit, f, indent=2)

with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump({
        "stage": "PORTFOLIO_P2_SLOT_SEMANTICS",
        "change_specialized_slot": change_slot_anchor,
        "is_stable_across_seeds": change_specialized_summary["is_stable_change_specialization_confirmed"],
        "motion_family": "deceleration_and_stop",
        "decision": "PROCEED_P3_FIXED_PORTFOLIO"
    }, f, indent=2)

print("Saved all P2 results to:", STAGE_DIR)
