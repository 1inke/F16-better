import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P0_COMPLEMENTARITY_20261007"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"

STAGE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts") if (REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts").exists() else str(PLAN_DIR.parent / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts"))
from common import (
    explicit_candidates, compute_errors, effective_good_clusters, motion_summary,
    write_json, load_clear_event_lookup
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params, PARAM_NAMES
from oracle_optimizer import _profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Stage P0 Static4 + Learned4 Complementarity Audit on: {device}")

def unpack_errors_full(cands, future):
    res = compute_errors(cands, future)
    eff_counts, _ = effective_good_clusters(cands, res["good"])
    row = np.arange(len(cands))
    j_idx = res["j_idx"]
    t4 = res["error"][row, j_idx, :4].mean(1)
    t8 = res["error"][row, j_idx, 4:8].mean(1)
    t12 = res["error"][row, j_idx, 8:12].mean(1)
    raw_good = res["good"].sum(axis=1)
    return {
        "minADE": res["minADE"],
        "minFDE": res["minFDE"],
        "J": res["J"],
        "coverage": res["coverage"],
        "eff_clusters": int(eff_counts.sum()),
        "raw_good": raw_good,
        "t4": t4,
        "t8": t8,
        "t12": t12,
        "good_matrix": res["good"]
    }

# 1. Load Data
print("\n--- Loading Data ---")
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
splits_sf = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")

cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)

feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr = len(tr_idx)
n_ot = len(ot_idx)

outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

clear_event_map = load_clear_event_lookup(REPO)
clear_event_outer = np.array([clear_event_map.get(int(sid), False) for sid in outer_sample_ids], dtype=bool)

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2,
    "clear_event": clear_event_outer
}

# 2. Build Control (Static-K24) and Components
print("Building Control (Static-K24) ...")
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
m_clf = raw_clf_obs[tr_idx].mean(0, keepdims=True)
s_clf = raw_clf_obs[tr_idx].std(0, keepdims=True)
s_clf = np.where(s_clf < 1e-6, 1.0, s_clf).astype(np.float32)

tr_norm = ((raw_clf_obs[tr_idx] - m_clf) / s_clf).astype(np.float32)
tr_lbl = maneuver_lbls[tr_idx]

cnt = np.bincount(tr_lbl, minlength=len(CLASS_NAMES)).astype(np.float32)
cw = np.sqrt(cnt.sum() / np.maximum(cnt, 1.0)); cw = cw / cw.mean()
cw_t = torch.from_numpy(cw).to(device)

clf = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
opt_clf = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
ldr_clf = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
for _ in range(15):
    clf.train()
    for x, y in ldr_clf:
        x, y = x.to(device), y.to(device)
        l = F.cross_entropy(clf(x), y, weight=cw_t)
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

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
focal_past_outer = raw_focal_past[ot_idx]

s0_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
for j in range(4):
    for i in range(n_ot):
        c_name = outer_top4_static[i][j]
        p_def = default_params(c_name, 1)
        tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
        s0_trajs[i, j] = tr[0]
k24_static = np.concatenate([native_outer, s0_trajs], axis=1)

static_metrics = unpack_errors_full(k24_static, future_outer)
static_kin = motion_summary(k24_static)
native_only_metrics = unpack_errors_full(native_outer, future_outer)
static4_only_metrics = unpack_errors_full(s0_trajs, future_outer)

native_cov_mask = native_only_metrics["coverage"]
static4_cov_mask = static4_only_metrics["coverage"]
static_cov_mask = static_metrics["coverage"]

print(f"Static-K24 Baseline: Coverage={static_cov_mask.mean()*100:.2f}% ({static_cov_mask.sum()}/{n_ot}), Mean J={static_metrics['J'].mean():.4f}, minADE={static_metrics['minADE'].mean():.4f}")

# Model Definition
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

# Normalization of P2
tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

def summarize_subgroups(metric_dict):
    out = {}
    for sg_name, msk in SUBGROUPS.items():
        n_m = int(msk.sum())
        if n_m == 0:
            out[sg_name] = {"n": 0, "coverage_pct": 0.0, "covered_count": 0, "mean_j": 0.0, "min_ade": 0.0, "min_fde": 0.0, "t4": 0.0, "t8": 0.0, "t12": 0.0, "mean_raw_good": 0.0}
            continue
        out[sg_name] = {
            "n": n_m,
            "coverage_pct": float(metric_dict["coverage"][msk].mean() * 100),
            "covered_count": int(metric_dict["coverage"][msk].sum()),
            "mean_j": float(metric_dict["J"][msk].mean()),
            "min_ade": float(metric_dict["minADE"][msk].mean()),
            "min_fde": float(metric_dict["minFDE"][msk].mean()),
            "t4": float(metric_dict["t4"][msk].mean()),
            "t8": float(metric_dict["t8"][msk].mean()),
            "t12": float(metric_dict["t12"][msk].mean()),
            "mean_raw_good": float(metric_dict["raw_good"][msk].mean())
        }
    return out

control_subgroups = summarize_subgroups(static_metrics)

SEEDS = [42, 43, 44]
p0_seed_results = {}
complementarity_data = {}

for s in SEEDS:
    print(f"\n==========================================")
    print(f"   EVALUATING K28 UNION FOR SEED {s}")
    print(f"==========================================")
    ckpt_path = STAGE_B_DIR / f"models/seed_{s}/model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = BestOf4Generator(in_dim=96, hidden_dim=128, num_slots=4, param_dim=8).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    bo4_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
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
            c_trajs_flat, _ = _profile(c_past_rep, c_raw_rep, "general")
            bo4_trajs[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()

    # 1. Evaluate Bo4-K24
    k24_bo4 = np.concatenate([native_outer, bo4_trajs], axis=1)
    bo4_metrics = unpack_errors_full(k24_bo4, future_outer)
    bo4_cov_mask = bo4_metrics["coverage"]

    # 2. Evaluate Learned4-only (4 proposals)
    learned4_only_metrics = unpack_errors_full(bo4_trajs, future_outer)
    learned4_cov_mask = learned4_only_metrics["coverage"]

    # 3. Construct K28 Union: Native20 + Static4 + Learned4
    k28_union = np.concatenate([native_outer, s0_trajs, bo4_trajs], axis=1) # (N, 28, 12, 2)
    union_metrics = unpack_errors_full(k28_union, future_outer)
    union_cov_mask = union_metrics["coverage"]
    union_kin = motion_summary(k28_union)

    # 4. Strict Invariance Assertions
    lost_vs_static = int(np.sum(static_cov_mask & (~union_cov_mask)))
    assert lost_vs_static == 0, f"Assertion Failed: K28 union lost {lost_vs_static} samples relative to Static-K24!"
    assert union_cov_mask.mean() >= static_cov_mask.mean(), "Assertion Failed: Union coverage < Static coverage!"
    print(f"Assertion PASSED: lost_vs_static == 0, union_coverage ({union_cov_mask.mean()*100:.2f}%) >= static_coverage ({static_cov_mask.mean()*100:.2f}%)")

    rescue_vs_static = int(np.sum((~static_cov_mask) & union_cov_mask))
    union_cov_gain_pp = float((union_cov_mask.mean() - static_cov_mask.mean()) * 100.0)
    union_j_gain_pct = float((static_metrics["J"].mean() - union_metrics["J"].mean()) / static_metrics["J"].mean() * 100.0)
    union_ade_gain_pct = float((static_metrics["minADE"].mean() - union_metrics["minADE"].mean()) / static_metrics["minADE"].mean() * 100.0)
    union_fde_gain_pct = float((static_metrics["minFDE"].mean() - union_metrics["minFDE"].mean()) / static_metrics["minFDE"].mean() * 100.0)

    union_subgroups = summarize_subgroups(union_metrics)
    change_mask = SUBGROUPS["change"]

    # 5. Set-Theoretic Decomposition:
    # Static4 vs Learned4 contributions (Overall and Change)
    def compute_set_theory(msk):
        sub_static4 = static4_cov_mask[msk]
        sub_learned4 = learned4_cov_mask[msk]
        sub_native = native_cov_mask[msk]
        sub_n = int(msk.sum())

        static4_only = (~sub_native) & sub_static4 & (~sub_learned4)
        learned4_only = (~sub_native) & sub_learned4 & (~sub_static4)
        both_proposals = sub_static4 & sub_learned4
        union_proposals = sub_static4 | sub_learned4
        
        # Beyond Native20
        static4_unique_benefit = (~sub_native) & sub_static4
        learned4_unique_benefit = (~sub_native) & sub_learned4
        union_unique_benefit = (~sub_native) & union_proposals

        return {
            "n": sub_n,
            "static4_only_count": int(static4_only.sum()),
            "static4_only_pct": float(static4_only.mean() * 100),
            "learned4_only_count": int(learned4_only.sum()),
            "learned4_only_pct": float(learned4_only.mean() * 100),
            "overlap_both_count": int(both_proposals.sum()),
            "overlap_both_pct": float(both_proposals.mean() * 100),
            "union_proposals_count": int(union_proposals.sum()),
            "union_proposals_pct": float(union_proposals.mean() * 100),
            "static4_unique_benefit_count": int(static4_unique_benefit.sum()),
            "learned4_unique_benefit_count": int(learned4_unique_benefit.sum()),
            "union_unique_benefit_count": int(union_unique_benefit.sum()),
            "jaccard_similarity": float(both_proposals.sum() / max(union_proposals.sum(), 1))
        }

    overall_set_theory = compute_set_theory(SUBGROUPS["overall"])
    change_set_theory = compute_set_theory(SUBGROUPS["change"])

    print(f"Seed {s} K28 Union Results:")
    print(f"  Overall Coverage: {union_cov_mask.mean()*100:.2f}% (Gain: {union_cov_gain_pp:+.2f} pp, Rescue: {rescue_vs_static}, Lost: 0)")
    print(f"  Change Coverage:  {union_subgroups['change']['coverage_pct']:.2f}% vs {control_subgroups['change']['coverage_pct']:.2f}% (Gain: {union_subgroups['change']['coverage_pct'] - control_subgroups['change']['coverage_pct']:+.2f} pp, Rescue: {int(np.sum((~static_cov_mask[change_mask]) & union_cov_mask[change_mask]))}, Lost: 0)")
    print(f"  Set-Theory Overall: Static4-only={overall_set_theory['static4_only_count']}, Learned4-only={overall_set_theory['learned4_only_count']}, Overlap={overall_set_theory['overlap_both_count']}, Union={overall_set_theory['union_proposals_count']}")
    print(f"  Set-Theory Change:  Static4-only={change_set_theory['static4_only_count']}, Learned4-only={change_set_theory['learned4_only_count']}, Overlap={change_set_theory['overlap_both_count']}, Union={change_set_theory['union_proposals_count']}")

    p0_seed_results[f"seed_{s}"] = {
        "overall": {
            "coverage_pct": float(union_cov_mask.mean() * 100),
            "covered_count": int(union_cov_mask.sum()),
            "coverage_gain_pp": union_cov_gain_pp,
            "rescue": rescue_vs_static,
            "lost": lost_vs_static,
            "net_rescue": rescue_vs_static,
            "mean_j": float(union_metrics["J"].mean()),
            "j_gain_pct": union_j_gain_pct,
            "min_ade": float(union_metrics["minADE"].mean()),
            "ade_gain_pct": union_ade_gain_pct,
            "min_fde": float(union_metrics["minFDE"].mean()),
            "fde_gain_pct": union_fde_gain_pct,
            "t4": float(union_metrics["t4"].mean()),
            "t8": float(union_metrics["t8"].mean()),
            "t12": float(union_metrics["t12"].mean()),
            "mean_raw_good": float(union_metrics["raw_good"].mean()),
            "effective_good_clusters": union_metrics["eff_clusters"]
        },
        "subgroups": union_subgroups,
        "set_theory_overall": overall_set_theory,
        "set_theory_change": change_set_theory,
        "kinematics": union_kin
    }

# 6. Aggregate across 3 seeds
mean_union_cov_gain = float(np.mean([p0_seed_results[f"seed_{s}"]["overall"]["coverage_gain_pp"] for s in SEEDS]))
mean_union_rescue = float(np.mean([p0_seed_results[f"seed_{s}"]["overall"]["rescue"] for s in SEEDS]))
mean_union_change_cov_gain = float(np.mean([p0_seed_results[f"seed_{s}"]["subgroups"]["change"]["coverage_pct"] - control_subgroups["change"]["coverage_pct"] for s in SEEDS]))
mean_union_change_rescue = float(np.mean([np.sum((~static_cov_mask[change_mask]) & (p0_seed_results[f"seed_{s}"]["subgroups"]["change"]["covered_count"])) for s in SEEDS]))

# Has significant complementarity?
# Complementarity exists if union clearly beats Static-K24 (e.g. coverage gain > 0 and rescue > 0)
has_complementarity = bool(mean_union_cov_gain > 0.5 and mean_union_rescue >= 15 and mean_union_change_cov_gain > 0.0)

print(f"\n==========================================")
print(f"   STAGE P0 COMPLEMENTARITY AUDIT SUMMARY")
print(f"==========================================")
print(f"Mean Union-K28 Coverage Gain: {mean_union_cov_gain:+.2f} pp ({mean_union_rescue:.1f} rescues, 0 lost)")
print(f"Mean Union-K28 Change Coverage Gain: {mean_union_change_cov_gain:+.2f} pp")
print(f"Static4 + Learned4 Complementarity Confirmed: {has_complementarity}")

# 7. Write RESULT.json & COMPLEMENTARITY_AUDIT.json
p0_result = {
    "stage": "PORTFOLIO_P0_COMPLEMENTARITY",
    "control": {
        "overall": {
            "coverage_pct": float(static_cov_mask.mean() * 100),
            "covered_count": int(static_cov_mask.sum()),
            "mean_j": float(static_metrics["J"].mean()),
            "min_ade": float(static_metrics["minADE"].mean()),
            "min_fde": float(static_metrics["minFDE"].mean()),
            "t4": float(static_metrics["t4"].mean()),
            "t8": float(static_metrics["t8"].mean()),
            "t12": float(static_metrics["t12"].mean()),
            "mean_raw_good": float(static_metrics["raw_good"].mean()),
            "effective_good_clusters": static_metrics["eff_clusters"]
        },
        "subgroups": control_subgroups,
        "kinematics": static_kin
    },
    "seeds": p0_seed_results,
    "aggregate": {
        "mean_union_coverage_gain_pp": mean_union_cov_gain,
        "mean_union_rescue": mean_union_rescue,
        "mean_union_change_coverage_gain_pp": mean_union_change_cov_gain,
        "has_complementarity": has_complementarity,
        "decision": "PROCEED_P1_ORACLE_8TO4" if has_complementarity else "STOP_NO_COMPLEMENTARITY"
    }
}

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(p0_result, f, indent=2)

with open(STAGE_DIR / "COMPLEMENTARITY_AUDIT.json", "w") as f:
    json.dump({
        "stage": "PORTFOLIO_P0_COMPLEMENTARITY",
        "has_complementarity": has_complementarity,
        "mean_union_coverage_gain_pp": mean_union_cov_gain,
        "mean_union_rescue": mean_union_rescue,
        "mean_union_change_coverage_gain_pp": mean_union_change_cov_gain,
        "set_theory_by_seed": {
            f"seed_{s}": {
                "overall": p0_seed_results[f"seed_{s}"]["set_theory_overall"],
                "change": p0_seed_results[f"seed_{s}"]["set_theory_change"]
            } for s in SEEDS
        },
        "decision": "PROCEED_P1_ORACLE_8TO4" if has_complementarity else "STOP_NO_COMPLEMENTARITY"
    }, f, indent=2)

dec_summary = {
    "stage": "PORTFOLIO_P0_COMPLEMENTARITY",
    "has_complementarity": has_complementarity,
    "mean_union_coverage_gain_pp": mean_union_cov_gain,
    "mean_union_rescue": mean_union_rescue,
    "mean_union_change_coverage_gain_pp": mean_union_change_cov_gain,
    "decision": "PROCEED_P1_ORACLE_8TO4" if has_complementarity else "STOP_NO_COMPLEMENTARITY"
}
with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump(dec_summary, f, indent=2)

print("Saved all P0 results to:", STAGE_DIR)
