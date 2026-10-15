import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
TRANSFER_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
OUT_DIR = TRANSFER_DIR / "scene_family"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = TRANSFER_DIR / "baseline_reconciliation"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts") if (REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts").exists() else str(PLAN_DIR.parent / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts"))
from common import (
    compute_errors, effective_good_clusters, motion_summary, load_clear_event_lookup
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params
from oracle_optimizer import _profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running Scene-Family Protocol on: {device}")

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

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr, n_ot = len(tr_idx), len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

# Clear event
ce_map = load_clear_event_lookup(REPO)
clear_event_outer = np.array([ce_map.get(int(sid), False) for sid in outer_sample_ids], dtype=bool)

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2,
    "clear_event": clear_event_outer
}

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
focal_past_outer = raw_focal_past[ot_idx]

# Normalization of P2
tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

# Load Canonical Static Selector
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
saved_clf = torch.load(BASE_DIR / "canonical_intent_classifier.pth", map_location=device, weights_only=False)
clf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
clf.load_state_dict(saved_clf["state_dict"])
clf.eval()

ot_norm = ((raw_clf_obs[ot_idx] - saved_clf["mean"]) / saved_clf["std"]).astype(np.float32)
with torch.no_grad():
    outer_probs = torch.softmax(clf(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}
top_c = np.argsort(-outer_probs[:, non_cont_classes], axis=1)[:, :4]

s0_all = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
for i in range(n_ot):
    for j in range(4):
        c_name = class_names_map[non_cont_classes[top_c[i, j]]]
        p_def = default_params(c_name, 1)
        tr = generate_motion_program(focal_past_outer[i:i+1], p_def)
        s0_all[i, j] = tr[0]

# Canonical Control: Native20 + Static4
k24_control = np.concatenate([native_outer, s0_all], axis=1)
ctrl_err = compute_errors(k24_control, future_outer)
ctrl_cov = ctrl_err["coverage"]
ctrl_ade = ctrl_err["minADE"]
ctrl_fde = ctrl_err["minFDE"]
ctrl_j = ctrl_err["J"]
ctrl_eff, _ = effective_good_clusters(k24_control, ctrl_err["good"])
ctrl_kin = motion_summary(k24_control)

def get_subgroup_metrics(cov, ade, fde, j_val, eff_total):
    res = {}
    for sg_name, msk in SUBGROUPS.items():
        n_m = int(msk.sum())
        if n_m == 0:
            res[sg_name] = {"n": 0, "covered_count": 0, "coverage_pct": 0.0, "min_ade": 0.0, "min_fde": 0.0, "mean_j": 0.0}
            continue
        res[sg_name] = {
            "n": n_m,
            "covered_count": int(cov[msk].sum()),
            "coverage_pct": float(cov[msk].mean() * 100),
            "min_ade": float(ade[msk].mean()),
            "min_fde": float(fde[msk].mean()),
            "mean_j": float(j_val[msk].mean())
        }
    res["effective_good_clusters"] = int(eff_total)
    return res

ctrl_metrics = get_subgroup_metrics(ctrl_cov, ctrl_ade, ctrl_fde, ctrl_j, ctrl_eff.sum())

print(f"CANONICAL CONTROL:")
print(f"  Overall: {ctrl_cov.mean()*100:.2f}% ({ctrl_cov.sum()}/{n_ot})")
print(f"  Change:  {ctrl_cov[SUBGROUPS['change']].mean()*100:.2f}% ({ctrl_cov[SUBGROUPS['change']].sum()}/{SUBGROUPS['change'].sum()})")

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

SEEDS = [42, 43, 44]
ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

results = {
    "protocol": "scene_family",
    "isolation_type": "scene_family_unseen",
    "outer_n": n_ot,
    "control": {
        "metrics": ctrl_metrics,
        "kinematics": ctrl_kin
    },
    "finalists": {
        "S1L3": {},
        "S2L2": {}
    }
}

# Kinematics storage
motion_audit_records = {
    "native20": motion_summary(native_outer),
    "static4": motion_summary(s0_all),
    "control_k24": ctrl_kin,
    "finalists": {"S1L3": {}, "S2L2": {}}
}

# Consistency records
consistency = {
    "S1L3": {"rescues": {}, "losts": {}},
    "S2L2": {"rescues": {}, "losts": {}}
}

for s in SEEDS:
    bo4_ckpt = STAGE_B_DIR / f"models/seed_{s}/model.pth"
    bo4_model = BestOf4Generator().to(device)
    bo4_model.load_state_dict(torch.load(bo4_ckpt, map_location=device))
    bo4_model.eval()

    bo4_ot = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    with torch.no_grad():
        CHUNK = 512
        for start_idx in range(0, n_ot, CHUNK):
            end_idx = min(start_idx + CHUNK, n_ot)
            c_p2 = ot_p2_t[start_idx:end_idx]
            c_past = past_ot_t[start_idx:end_idx]
            c_B = len(c_p2)
            c_raw = bo4_model(c_p2)
            c_past_rep = c_past.repeat_interleave(4, dim=0)
            c_raw_rep = c_raw.view(c_B * 4, 8)
            c_trajs_flat, _ = _profile(c_past_rep, c_raw_rep, "general")
            bo4_ot[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()

    # Finalist 1: S1L3 = Native20 + Static proposal 0 + Learned slots [3, 0, 1]
    s1 = s0_all[:, :1] # (N, 1, 12, 2)
    l3 = bo4_ot[:, [3, 0, 1]] # (N, 3, 12, 2)
    p_s1l3 = np.concatenate([s1, l3], axis=1) # (N, 4, 12, 2)
    k24_s1l3 = np.concatenate([native_outer, p_s1l3], axis=1) # (N, 24, 12, 2)

    # Finalist 2: S2L2 = Native20 + Static proposals [0, 1] + Learned slots [3, 0]
    s2 = s0_all[:, :2] # (N, 2, 12, 2)
    l2 = bo4_ot[:, [3, 0]] # (N, 2, 12, 2)
    p_s2l2 = np.concatenate([s2, l2], axis=1) # (N, 4, 12, 2)
    k24_s2l2 = np.concatenate([native_outer, p_s2l2], axis=1) # (N, 24, 12, 2)

    for f_name, k24_cand, prop_only, static_prop, learned_prop in [
        ("S1L3", k24_s1l3, p_s1l3, s1, l3),
        ("S2L2", k24_s2l2, p_s2l2, s2, l2)
    ]:
        err = compute_errors(k24_cand, future_outer)
        cov = err["coverage"]
        ade = err["minADE"]
        fde = err["minFDE"]
        j_val = err["J"]
        eff, _ = effective_good_clusters(k24_cand, err["good"])
        kin = motion_summary(k24_cand)

        r_msk = (~ctrl_cov) & cov
        l_msk = ctrl_cov & (~cov)
        r_cnt = int(r_msk.sum())
        l_cnt = int(l_msk.sum())
        net_r = r_cnt - l_cnt

        ch_r = int((r_msk & SUBGROUPS["change"]).sum())
        ch_l = int((l_msk & SUBGROUPS["change"]).sum())

        sub_m = get_subgroup_metrics(cov, ade, fde, j_val, eff.sum())

        cov_gain_pp = float((cov.mean() - ctrl_cov.mean()) * 100)
        ch_gain_pp = float(sub_m["change"]["coverage_pct"] - ctrl_metrics["change"]["coverage_pct"])
        ade_gain_pct = float((ctrl_ade.mean() - ade.mean()) / ctrl_ade.mean() * 100)
        fde_gain_pct = float((ctrl_fde.mean() - fde.mean()) / ctrl_fde.mean() * 100)
        ch_ade_gain = float((ctrl_metrics["change"]["min_ade"] - sub_m["change"]["min_ade"]) / ctrl_metrics["change"]["min_ade"] * 100)
        ch_fde_gain = float((ctrl_metrics["change"]["min_fde"] - sub_m["change"]["min_fde"]) / ctrl_metrics["change"]["min_fde"] * 100)

        acc_diff = float((kin["accel_p95"] - ctrl_kin["accel_p95"]) / ctrl_kin["accel_p95"] * 100)
        motion_pass = abs(acc_diff) <= 2.0

        # Rescued winners kinematics
        if r_cnt > 0:
            # find which candidate was good in rescued windows
            res_winners = []
            for idx in np.where(r_msk)[0]:
                best_k = np.argmin(err["error"][idx, :, -1])
                res_winners.append(k24_cand[idx:idx+1, best_k:best_k+1])
            res_winners = np.concatenate(res_winners, axis=0)
            res_kin = motion_summary(res_winners)
        else:
            res_kin = {}

        results["finalists"][f_name][f"seed_{s}"] = {
            "overall_coverage_pct": float(cov.mean() * 100),
            "overall_gain_pp": cov_gain_pp,
            "rescue": r_cnt,
            "lost": l_cnt,
            "net_rescue": net_r,
            "change_coverage_pct": sub_m["change"]["coverage_pct"],
            "change_gain_pp": ch_gain_pp,
            "change_rescue": ch_r,
            "change_lost": ch_l,
            "overall_ade_gain_pct": ade_gain_pct,
            "overall_fde_gain_pct": fde_gain_pct,
            "change_ade_gain_pct": ch_ade_gain,
            "change_fde_gain_pct": ch_fde_gain,
            "accel_p95_diff_pct": acc_diff,
            "motion_guard_pass": motion_pass,
            "subgroups": sub_m,
            "kinematics": kin
        }

        consistency[f_name]["rescues"][f"seed_{s}"] = [int(outer_sample_ids[i]) for i in np.where(r_msk)[0]]
        consistency[f_name]["losts"][f"seed_{s}"] = [int(outer_sample_ids[i]) for i in np.where(l_msk)[0]]

        motion_audit_records["finalists"][f_name][f"seed_{s}"] = {
            "pooled_k24": kin,
            "proposals_only": motion_summary(prop_only),
            "static_prop_only": motion_summary(static_prop),
            "learned_prop_only": motion_summary(learned_prop),
            "rescued_winners": res_kin
        }

        print(f"[{f_name} Seed {s}]")
        print(f"  Overall: {cov.mean()*100:.2f}% ({cov_gain_pp:+.2f} pp, R:{r_cnt}, L:{l_cnt}, Net:{net_r:+d})")
        print(f"  Change:  {sub_m['change']['coverage_pct']:.2f}% ({ch_gain_pp:+.2f} pp, R:{ch_r}, L:{ch_l}, ADE:{ch_ade_gain:+.2f}%, FDE:{ch_fde_gain:+.2f}%)")
        print(f"  Accel p95: {kin['accel_p95']:.4f} ({acc_diff:+.2f}%, Pass: {motion_pass})")

# Compute cross-seed consistency metrics
def calc_consistency(seed_dict):
    s42, s43, s44 = set(seed_dict["seed_42"]), set(seed_dict["seed_43"]), set(seed_dict["seed_44"])
    all_items = s42.union(s43).union(s44)
    common_23 = [x for x in all_items if (int(x in s42) + int(x in s43) + int(x in s44)) >= 2]
    common_33 = [x for x in all_items if (x in s42 and x in s43 and x in s44)]
    def jaccard(a, b):
        return len(a.intersection(b)) / max(len(a.union(b)), 1)
    return {
        "total_unique": len(all_items),
        "common_2_of_3_count": len(common_23),
        "common_3_of_3_count": len(common_33),
        "jaccard_42_43": jaccard(s42, s43),
        "jaccard_42_44": jaccard(s42, s44),
        "jaccard_43_44": jaccard(s43, s44),
        "mean_jaccard": (jaccard(s42, s43) + jaccard(s42, s44) + jaccard(s43, s44)) / 3.0
    }

consistency_summary = {
    "S1L3": {
        "rescues": calc_consistency(consistency["S1L3"]["rescues"]),
        "losts": calc_consistency(consistency["S1L3"]["losts"])
    },
    "S2L2": {
        "rescues": calc_consistency(consistency["S2L2"]["rescues"]),
        "losts": calc_consistency(consistency["S2L2"]["losts"])
    }
}

with open(OUT_DIR / "per_seed_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

with open(OUT_DIR / "rescue_lost_consistency.json", "w") as f:
    json.dump(consistency_summary, f, indent=2)

with open(OUT_DIR / "motion_audit.json", "w") as f:
    json.dump(motion_audit_records, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "protocol": "scene_family",
        "results": results,
        "consistency": consistency_summary
    }, f, indent=2)

print("Saved all Scene-Family evaluation files to:", OUT_DIR)
