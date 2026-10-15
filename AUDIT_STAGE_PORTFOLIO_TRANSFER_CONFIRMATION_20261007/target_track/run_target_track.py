import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
OUT_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007" / "target_track"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Executing Target-Track Isolated Protocol on: {device}")

# 1. Load Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
splits_tt = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)

tr_idx = splits_tt["grouped_train_index"] # 18909
ot_idx = splits_tt["outer_eval_index"]    # 1035
n_tr, n_ot = len(tr_idx), len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

# Invariance assertions
overlap = set(tr_idx).intersection(set(ot_idx))
assert len(overlap) == 0, f"Leakage detected! Overlap = {len(overlap)}"
print(f"Data verified: Train={n_tr}, Outer={n_ot}, Overlap={len(overlap)}")

# Clear Event lookup
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

native_train = train_cache["candidate"][tr_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_train = lab_train["future"][tr_idx].astype(np.float32)
focal_past_train = raw_focal_past[tr_idx]

# P2 Normalization strictly on train split
tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
tr_p2_norm = ((tr_p2 - mu_p2) / std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

# 2. Train Isolated Intent Classifier on Target-Track Train Split
print("Training Isolated Intent Classifier on Target-Track Train Split ...")
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

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

clf = IntentClassifier(tr_norm.shape[1], 0.1).to(device)
opt_clf = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
ldr_clf = DataLoader(TensorDataset(torch.from_numpy(tr_norm), torch.from_numpy(tr_lbl)), batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
for _ in range(15):
    clf.train()
    for x, y in ldr_clf:
        l = F.cross_entropy(clf(x.to(device)), y.to(device), weight=cw_t)
        opt_clf.zero_grad(set_to_none=True); l.backward(); opt_clf.step()
clf.eval()

clf_ckpt_path = MODELS_DIR / "isolated_intent_classifier.pth"
torch.save({"state_dict": clf.state_dict(), "mean": m_clf, "std": s_clf}, clf_ckpt_path)

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

# Canonical Control: Native20 + Static4 on Target-Track
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

print(f"TARGET-TRACK CANONICAL CONTROL:")
print(f"  Overall: {ctrl_cov.mean()*100:.2f}% ({ctrl_cov.sum()}/{n_ot})")
print(f"  Change:  {ctrl_cov[SUBGROUPS['change']].mean()*100:.2f}% ({ctrl_cov[SUBGROUPS['change']].sum()}/{SUBGROUPS['change'].sum()})")

# 3. Model Definition: BestOf4Generator
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
        self._init_heads()

    def _init_heads(self):
        biases = [
            [0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.2, 1.2, 1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.2, -1.2, -1.2, 0.0, 0.0],
            [-0.8, -0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        for head, b in zip(self.heads, biases):
            last_linear = head[-1]
            last_linear.weight.data.normal_(0, 0.01)
            last_linear.bias.data = torch.tensor(b, dtype=torch.float32)

    def forward(self, x):
        feat = self.backbone(x)
        out = [head(feat) for head in self.heads]
        return torch.stack(out, dim=1)

def train_bo4_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = BestOf4Generator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds_tr = TensorDataset(
        torch.from_numpy(tr_p2_norm),
        torch.from_numpy(focal_past_train),
        torch.from_numpy(future_train)
    )
    ldr_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    for ep in range(1, 26):
        model.train()
        tot_l, tot_b = 0.0, 0
        for b_p2, b_past, b_fut in ldr_tr:
            b_p2, b_past, b_fut = b_p2.to(device), b_past.to(device), b_fut.to(device)
            B = len(b_p2)
            raw_params = model(b_p2)
            past_rep = b_past.repeat_interleave(4, dim=0)
            raw_rep = raw_params.view(B * 4, 8)
            trajs_flat, _ = _profile(past_rep, raw_rep, "general")
            cands = trajs_flat.view(B, 4, 12, 2)

            diffs = torch.norm(cands - b_fut.unsqueeze(1), dim=-1)
            ade = diffs.mean(dim=-1)
            fde = diffs[:, :, -1]
            j_score = torch.max(ade / 0.2, fde / 0.4)
            best_idx = torch.argmin(j_score, dim=1)
            b_idx = torch.arange(B, device=device)
            best_ade = ade[b_idx, best_idx]
            best_fde = fde[b_idx, best_idx]
            winner_loss = (best_ade + best_fde).mean()

            cands_exp1 = cands.unsqueeze(2)
            cands_exp2 = cands.unsqueeze(1)
            pw_dist = torch.norm(cands_exp1 - cands_exp2, dim=-1).mean(dim=-1)
            eye_mask = torch.eye(4, device=device).bool().unsqueeze(0)
            pw_dist_off = torch.where(eye_mask, torch.tensor(float('inf'), device=device), pw_dist)
            min_dist = pw_dist_off.min(dim=-1)[0].min(dim=-1)[0]
            div_loss = torch.exp(-min_dist).mean()

            loss = winner_loss + 0.05 * div_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_l += loss.item() * B
            tot_b += B

    s_dir = MODELS_DIR / f"seed_{seed}"
    s_dir.mkdir(parents=True, exist_ok=True)
    m_p = s_dir / "model.pth"
    torch.save(model.state_dict(), m_p)
    return model

SEEDS = [42, 43, 44]
ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

results = {
    "protocol": "target_track",
    "isolation_type": "pedestrian_id_unseen",
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

motion_audit_records = {
    "native20": motion_summary(native_outer),
    "static4": motion_summary(s0_all),
    "control_k24": ctrl_kin,
    "finalists": {"S1L3": {}, "S2L2": {}}
}

consistency = {
    "S1L3": {"rescues": {}, "losts": {}},
    "S2L2": {"rescues": {}, "losts": {}}
}

for s in SEEDS:
    print(f"\n--- Training Bo4 Generator Seed {s} on Target-Track Train Split ---")
    bo4_model = train_bo4_model(s)
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
    s1 = s0_all[:, :1]
    l3 = bo4_ot[:, [3, 0, 1]]
    p_s1l3 = np.concatenate([s1, l3], axis=1)
    k24_s1l3 = np.concatenate([native_outer, p_s1l3], axis=1)

    # Finalist 2: S2L2 = Native20 + Static proposals [0, 1] + Learned slots [3, 0]
    s2 = s0_all[:, :2]
    l2 = bo4_ot[:, [3, 0]]
    p_s2l2 = np.concatenate([s2, l2], axis=1)
    k24_s2l2 = np.concatenate([native_outer, p_s2l2], axis=1)

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

        if r_cnt > 0:
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
        "protocol": "target_track",
        "results": results,
        "consistency": consistency_summary
    }, f, indent=2)

print("Saved all Target-Track evaluation files to:", OUT_DIR)
