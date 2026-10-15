import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P3_FIXED_PORTFOLIO_20261007"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"

STAGE_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Executing Stage P3 Fixed Portfolio K24 Audit on: {device}")

def unpack_errors(cands, future):
    res = compute_errors(cands, future)
    eff_counts, _ = effective_good_clusters(cands, res["good"])
    return res["minADE"], res["minFDE"], res["J"], res["coverage"], int(eff_counts.sum())

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
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)
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

# Static-K24 setup
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
        l = F.cross_entropy(clf(x.to(device)), y.to(device), weight=cw_t)
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
static_ade, static_fde, static_j, static_cov, static_eff = unpack_errors(k24_static, future_outer)
static_kin = motion_summary(k24_static)

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

def get_subgroups(ade, fde, j_val, cov):
    out = {}
    for sg_name, msk in SUBGROUPS.items():
        n_m = int(msk.sum())
        out[sg_name] = {
            "n": n_m,
            "coverage_pct": float(cov[msk].mean() * 100),
            "covered_count": int(cov[msk].sum()),
            "mean_j": float(j_val[msk].mean()),
            "min_ade": float(ade[msk].mean()),
            "min_fde": float(fde[msk].mean())
        }
    return out

control_subgroups = get_subgroups(static_ade, static_fde, static_j, static_cov)

# 2. Portfolio Configurations Specification
# Total proposals = 4, Total K = 24
# Learned priority order locked from P2: [3, 0, 1, 2]
PORTFOLIO_CONFIGS = {
    "S4L0": {"static_idx": [0, 1, 2, 3], "learned_idx": []},
    "S3L1": {"static_idx": [0, 1, 2], "learned_idx": [3]},
    "S2L2": {"static_idx": [0, 1], "learned_idx": [3, 0]},
    "S1L3": {"static_idx": [0], "learned_idx": [3, 0, 1]},
    "S0L4": {"static_idx": [], "learned_idx": [3, 0, 1, 2]}
}

SEEDS = [42, 43, 44]
p3_results = {cfg_name: {} for cfg_name in PORTFOLIO_CONFIGS}

for s in SEEDS:
    print(f"\n==========================================")
    print(f"   EVALUATING FIXED PORTFOLIOS FOR SEED {s}")
    print(f"==========================================")
    ckpt_path = STAGE_B_DIR / f"models/seed_{s}/model.pth"
    model = BestOf4Generator().to(device)
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

    for cfg_name, spec in PORTFOLIO_CONFIGS.items():
        s_idx = spec["static_idx"]
        l_idx = spec["learned_idx"]

        props_list = []
        if len(s_idx) > 0:
            props_list.append(s0_trajs[:, s_idx])
        if len(l_idx) > 0:
            props_list.append(bo4_trajs[:, l_idx])
        port_props = np.concatenate(props_list, axis=1) # (N, 4, 12, 2)
        assert port_props.shape[1] == 4, f"Invalid proposal count: {port_props.shape[1]}"

        k24_cand = np.concatenate([native_outer, port_props], axis=1) # (N, 24, 12, 2)
        assert k24_cand.shape[1] == 24, f"Invalid total K: {k24_cand.shape[1]}"

        c_ade, c_fde, c_j, c_cov, c_eff = unpack_errors(k24_cand, future_outer)
        c_kin = motion_summary(k24_cand)
        c_subgroups = get_subgroups(c_ade, c_fde, c_j, c_cov)

        rescue = int(np.sum((~static_cov) & c_cov))
        lost = int(np.sum(static_cov & (~c_cov)))
        net_rescue = rescue - lost
        cov_gain_pp = float((c_cov.mean() - static_cov.mean()) * 100.0)
        ch_gain_pp = float(c_subgroups["change"]["coverage_pct"] - control_subgroups["change"]["coverage_pct"])
        ade_gain_pct = float((static_ade.mean() - c_ade.mean()) / static_ade.mean() * 100.0)
        fde_gain_pct = float((static_fde.mean() - c_fde.mean()) / static_fde.mean() * 100.0)
        ch_ade_gain_pct = float((control_subgroups["change"]["min_ade"] - c_subgroups["change"]["min_ade"]) / control_subgroups["change"]["min_ade"] * 100.0)
        ch_fde_gain_pct = float((control_subgroups["change"]["min_fde"] - c_subgroups["change"]["min_fde"]) / control_subgroups["change"]["min_fde"] * 100.0)

        speed_diff = float((c_kin["speed_p95"] - static_kin["speed_p95"]) / static_kin["speed_p95"] * 100.0)
        acc_diff = float((c_kin["accel_p95"] - static_kin["accel_p95"]) / static_kin["accel_p95"] * 100.0)
        jerk_diff = float((c_kin["jerk_p95"] - static_kin["jerk_p95"]) / static_kin["jerk_p95"] * 100.0)
        motion_pass = bool(abs(speed_diff) <= 2.0 and abs(acc_diff) <= 2.0 and abs(jerk_diff) <= 2.0)

        p3_results[cfg_name][f"seed_{s}"] = {
            "overall_coverage_pct": float(c_cov.mean() * 100),
            "coverage_gain_pp": cov_gain_pp,
            "rescue": rescue,
            "lost": lost,
            "net_rescue": net_rescue,
            "change_coverage_pct": c_subgroups["change"]["coverage_pct"],
            "change_coverage_gain_pp": ch_gain_pp,
            "change_ade_gain_pct": ch_ade_gain_pct,
            "change_fde_gain_pct": ch_fde_gain_pct,
            "overall_ade_gain_pct": ade_gain_pct,
            "overall_fde_gain_pct": fde_gain_pct,
            "kinematics": c_kin,
            "motion_guard_pass": motion_pass,
            "subgroups": c_subgroups
        }
        print(f"  {cfg_name}: Overall={c_cov.mean()*100:.2f}% ({cov_gain_pp:+.2f} pp, R:{rescue}, L:{lost}, Net:{net_rescue:+d}) | Change={c_subgroups['change']['coverage_pct']:.2f}% ({ch_gain_pp:+.2f} pp) | Acc diff={acc_diff:+.2f}%")

# 3. Aggregate across 3 seeds for each configuration
print(f"\n==========================================")
print(f"   STAGE P3 FIXED PORTFOLIO AGGREGATE SUMMARY")
print(f"==========================================")
aggregate_summary = {}
best_portfolio = None
best_ch_cov_gain = -999.0

for cfg_name in PORTFOLIO_CONFIGS:
    gains_ov = [p3_results[cfg_name][f"seed_{s}"]["coverage_gain_pp"] for s in SEEDS]
    gains_ch = [p3_results[cfg_name][f"seed_{s}"]["change_coverage_gain_pp"] for s in SEEDS]
    gains_ch_ade = [p3_results[cfg_name][f"seed_{s}"]["change_ade_gain_pct"] for s in SEEDS]
    gains_ch_fde = [p3_results[cfg_name][f"seed_{s}"]["change_fde_gain_pct"] for s in SEEDS]
    harms_ov_ade = [max(0.0, -p3_results[cfg_name][f"seed_{s}"]["overall_ade_gain_pct"]) for s in SEEDS]
    harms_ov_fde = [max(0.0, -p3_results[cfg_name][f"seed_{s}"]["overall_fde_gain_pct"]) for s in SEEDS]
    acc_diffs = [(p3_results[cfg_name][f"seed_{s}"]["kinematics"]["accel_p95"] - static_kin["accel_p95"])/static_kin["accel_p95"]*100.0 for s in SEEDS]

    mean_ov_gain = float(np.mean(gains_ov))
    mean_ch_gain = float(np.mean(gains_ch))
    mean_ch_ade = float(np.mean(gains_ch_ade))
    mean_ch_fde = float(np.mean(gains_ch_fde))
    ch_pos_count = sum(1 for g in gains_ch if g > 0.0)
    all_seeds_ov_not_worse = all(g >= -0.1 for g in gains_ov)
    no_harm = max(harms_ov_ade) <= 1.0 and max(harms_ov_fde) <= 1.0

    # Preregistered Fixed Portfolio Success Criteria:
    # 1. 3 seeds Overall not worse than Static-K24
    # 2. Change 3-seed mean coverage > Static-K24
    # 3. Change at least 2/3 seeds positive gain
    # 4. Change minADE or minFDE at least one stably improved
    # 5. No severe motion pathology
    is_positive = bool(
        all_seeds_ov_not_worse and
        mean_ch_gain > 0.0 and
        ch_pos_count >= 2 and
        (mean_ch_ade > 0.0 or mean_ch_fde > 0.0) and
        no_harm
    )

    aggregate_summary[cfg_name] = {
        "mean_overall_coverage_gain_pp": mean_ov_gain,
        "mean_change_coverage_gain_pp": mean_ch_gain,
        "change_positive_seeds_count": ch_pos_count,
        "change_coverage_gains_by_seed": gains_ch,
        "mean_change_ade_gain_pct": mean_ch_ade,
        "mean_change_fde_gain_pct": mean_ch_fde,
        "accel_p95_diffs_pct": acc_diffs,
        "is_fixed_portfolio_positive": is_positive
    }

    print(f"Portfolio {cfg_name:5s}: Overall Cov Gain = {mean_ov_gain:+.2f} pp | Change Cov Gain = {mean_ch_gain:+.2f} pp (Pos: {ch_pos_count}/3: {[round(x,2) for x in gains_ch]}) | Change ADE={mean_ch_ade:+.2f}%, FDE={mean_ch_fde:+.2f}% | Acc diffs={[round(x,2) for x in acc_diffs]}% | Gate={is_positive}")

    if is_positive and mean_ch_gain > best_ch_cov_gain:
        best_ch_cov_gain = mean_ch_gain
        best_portfolio = cfg_name

# Any fixed portfolio positive?
any_portfolio_positive = any(aggregate_summary[c]["is_fixed_portfolio_positive"] for c in PORTFOLIO_CONFIGS)
action_decision = "FIXED_PORTFOLIO_POSITIVE" if any_portfolio_positive else "NO_FIXED_PORTFOLIO_PASS"

print(f"\nFINAL STAGE P3 DECISION: {action_decision} (Best Portfolio: {best_portfolio})")

# 4. Save results
p3_out = {
    "stage": "PORTFOLIO_P3_FIXED_PORTFOLIO",
    "control": {
        "overall_coverage_pct": float(static_cov.mean() * 100),
        "change_coverage_pct": float(control_subgroups["change"]["coverage_pct"]),
        "kinematics": static_kin
    },
    "by_configuration": p3_results,
    "aggregate": aggregate_summary,
    "best_portfolio": best_portfolio,
    "action_decision": action_decision
}

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(p3_out, f, indent=2)

with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump({
        "stage": "PORTFOLIO_P3_FIXED_PORTFOLIO",
        "action_decision": action_decision,
        "best_portfolio": best_portfolio,
        "aggregate": aggregate_summary
    }, f, indent=2)

print("Saved all P3 results to:", STAGE_DIR)
