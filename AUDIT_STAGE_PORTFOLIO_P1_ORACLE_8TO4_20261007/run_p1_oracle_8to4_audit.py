import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P1_ORACLE_8TO4_20261007"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"
P0_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P0_COMPLEMENTARITY_20261007"

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
print(f"Executing Stage P1 Oracle 8->4 Budget Audit on: {device}")

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
static_ade, static_fde, static_j, static_cov, static_eff = unpack_errors(k24_static, future_outer)

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

def get_subgroup_metrics(ade, fde, j_val, cov):
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

control_subgroups = get_subgroup_metrics(static_ade, static_fde, static_j, static_cov)

SEEDS = [42, 43, 44]
p1_seed_results = {}

for s in SEEDS:
    print(f"\n==========================================")
    print(f"   EVALUATING ORACLE 8->4 FOR SEED {s}")
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

    # K28 Union (8 proposals + 20 native)
    k28_union = np.concatenate([native_outer, s0_trajs, bo4_trajs], axis=1)
    union_ade, union_fde, union_j, union_cov, union_eff = unpack_errors(k28_union, future_outer)
    union_subgroups = get_subgroup_metrics(union_ade, union_fde, union_j, union_cov)
    union_rescues = int(np.sum((~static_cov) & union_cov))
    change_mask = SUBGROUPS["change"]
    union_change_rescues = int(np.sum((~static_cov[change_mask]) & union_cov[change_mask]))

    # Pool 8 proposals: (N, 8, 12, 2)
    props8 = np.concatenate([s0_trajs, bo4_trajs], axis=1)

    # Compute errors for all 8 proposals
    diff8 = np.linalg.norm(props8 - future_outer[:, None, :, :], axis=-1) # (N, 8, 12)
    ade8 = diff8.mean(axis=-1) # (N, 8)
    fde8 = diff8[:, :, -1]     # (N, 8)
    j8 = np.maximum(ade8 / 0.2, fde8 / 0.4)
    good8 = (ade8 <= 0.2) & (fde8 <= 0.4)

    # For each sample, select the best 4 proposals
    # Priority: 1. good candidate (True before False), 2. smaller J
    oracle_top4_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        # Sort score: -1000 if good, else J
        sort_key = np.where(good8[i], -1000.0 + j8[i], j8[i])
        best4_idx = np.argsort(sort_key)[:4]
        oracle_top4_trajs[i] = props8[i, best4_idx]

    # Form OracleBest4Of8-K24 (Native 20 + Oracle Top 4)
    k24_oracle = np.concatenate([native_outer, oracle_top4_trajs], axis=1) # (N, 24, 12, 2)
    orc_ade, orc_fde, orc_j, orc_cov, orc_eff = unpack_errors(k24_oracle, future_outer)
    orc_subgroups = get_subgroup_metrics(orc_ade, orc_fde, orc_j, orc_cov)
    orc_rescues = int(np.sum((~static_cov) & orc_cov))
    orc_change_rescues = int(np.sum((~static_cov[change_mask]) & orc_cov[change_mask]))

    rescue_retention_overall = float(orc_rescues / max(union_rescues, 1) * 100.0)
    rescue_retention_change = float(orc_change_rescues / max(union_change_rescues, 1) * 100.0)

    print(f"Seed {s} Oracle 8->4 Results:")
    print(f"  Union Coverage:   {union_cov.mean()*100:.2f}% (Rescues: {union_rescues})")
    print(f"  Oracle4 Coverage: {orc_cov.mean()*100:.2f}% (Rescues: {orc_rescues})")
    print(f"  Overall Rescue Retention: {rescue_retention_overall:.2f}% (Threshold: >= 95.0%)")
    print(f"  Change Rescue Retention:  {rescue_retention_change:.2f}% ({orc_change_rescues} / {union_change_rescues})")
    print(f"  Oracle Change Coverage:   {orc_subgroups['change']['coverage_pct']:.2f}% vs Union: {union_subgroups['change']['coverage_pct']:.2f}% vs Control: {control_subgroups['change']['coverage_pct']:.2f}%")

    p1_seed_results[f"seed_{s}"] = {
        "overall": {
            "union_coverage_pct": float(union_cov.mean() * 100),
            "oracle_coverage_pct": float(orc_cov.mean() * 100),
            "coverage_diff_pp": float((orc_cov.mean() - union_cov.mean()) * 100),
            "union_rescues": union_rescues,
            "oracle_rescues": orc_rescues,
            "rescue_retention_pct": rescue_retention_overall,
            "union_mean_j": float(union_j.mean()),
            "oracle_mean_j": float(orc_j.mean()),
            "union_min_ade": float(union_ade.mean()),
            "oracle_min_ade": float(orc_ade.mean()),
            "union_min_fde": float(union_fde.mean()),
            "oracle_min_fde": float(orc_fde.mean())
        },
        "change": {
            "union_change_coverage_pct": union_subgroups["change"]["coverage_pct"],
            "oracle_change_coverage_pct": orc_subgroups["change"]["coverage_pct"],
            "union_change_rescues": union_change_rescues,
            "oracle_change_rescues": orc_change_rescues,
            "change_rescue_retention_pct": rescue_retention_change
        },
        "subgroups": orc_subgroups
    }

mean_retention_overall = float(np.mean([p1_seed_results[f"seed_{s}"]["overall"]["rescue_retention_pct"] for s in SEEDS]))
mean_retention_change = float(np.mean([p1_seed_results[f"seed_{s}"]["change"]["change_rescue_retention_pct"] for s in SEEDS]))
mean_oracle_cov = float(np.mean([p1_seed_results[f"seed_{s}"]["overall"]["oracle_coverage_pct"] for s in SEEDS]))
mean_oracle_change_cov = float(np.mean([p1_seed_results[f"seed_{s}"]["change"]["oracle_change_coverage_pct"] for s in SEEDS]))

is_budget_sufficient = bool(mean_retention_overall >= 95.0 and mean_retention_change >= 95.0)

print(f"\n==========================================")
print(f"   STAGE P1 ORACLE 8->4 AUDIT SUMMARY")
print(f"==========================================")
print(f"Mean Overall Rescue Retention: {mean_retention_overall:.2f}% (>= 95% Threshold Pass: {mean_retention_overall >= 95.0})")
print(f"Mean Change Rescue Retention:  {mean_retention_change:.2f}% (>= 95% Threshold Pass: {mean_retention_change >= 95.0})")
print(f"Mean Oracle K24 Coverage:      {mean_oracle_cov:.2f}% (+{mean_oracle_cov - static_cov.mean()*100:.2f} pp over Static)")
print(f"Mean Oracle K24 Change Cov:    {mean_oracle_change_cov:.2f}% (+{mean_oracle_change_cov - control_subgroups['change']['coverage_pct']:.2f} pp over Static)")
print(f"Budget 4 Slots Sufficient Upper Bound: {is_budget_sufficient}")

p1_audit = {
    "stage": "PORTFOLIO_P1_ORACLE_8TO4",
    "is_budget_sufficient": is_budget_sufficient,
    "mean_retention_overall_pct": mean_retention_overall,
    "mean_retention_change_pct": mean_retention_change,
    "mean_oracle_coverage_pct": mean_oracle_cov,
    "mean_oracle_change_coverage_pct": mean_oracle_change_cov,
    "seeds": p1_seed_results,
    "conclusion": "4 proposal slots retain 100% of Union benefit under Oracle selection; K=24 budget is NOT the primary bottleneck." if is_budget_sufficient else "Budget limitation detected."
}

with open(STAGE_DIR / "ORACLE_4_OF_8_AUDIT.json", "w") as f:
    json.dump(p1_audit, f, indent=2)

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(p1_audit, f, indent=2)

with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump({
        "stage": "PORTFOLIO_P1_ORACLE_8TO4",
        "is_budget_sufficient": is_budget_sufficient,
        "mean_retention_overall_pct": mean_retention_overall,
        "mean_retention_change_pct": mean_retention_change,
        "decision": "PROCEED_P2_SLOT_SEMANTICS" if is_budget_sufficient else "STOP_K24_PORTFOLIO_CAPACITY_LIMIT"
    }, f, indent=2)

print("Saved all P1 results to:", STAGE_DIR)
