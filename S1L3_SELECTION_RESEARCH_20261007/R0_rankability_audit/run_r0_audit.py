import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
CONFIRM_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
OUT_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R0_rankability_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Executing Stage R0 Frozen Scorer & Rankability Audit on: {device}", flush=True)

# 1. Load Caches
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32) # (28791, 96)
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
ce_map = load_clear_event_lookup(REPO)

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

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

PROTOCOLS_CONFIG = [
    {
        "id": "scene_family",
        "name": "Scene-Family Outer",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz",
        "intent_clf_path": CONFIRM_DIR / "baseline_reconciliation/canonical_intent_classifier.pth",
        "bo4_dir": REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/models"
    },
    {
        "id": "target_track",
        "name": "Target-Track Outer",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz",
        "intent_clf_path": CONFIRM_DIR / "target_track/models/isolated_intent_classifier.pth",
        "bo4_dir": CONFIRM_DIR / "target_track/models"
    },
    {
        "id": "recording_time",
        "name": "Recording-Time Outer",
        "split_file": REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/splits.npz",
        "intent_clf_path": CONFIRM_DIR / "recording_time/models/isolated_intent_classifier.pth",
        "bo4_dir": CONFIRM_DIR / "recording_time/models"
    }
]

SEEDS = [42, 43, 44]
PROVENANCE_NAMES = ["Native"] * 20 + ["Static", "Learned-3", "Learned-0", "Learned-1"]

# Storage for results
existence_vs_selection = {}
rank_distribution_results = {}
newsupport_witnesses = []
provenance_bias_records = []
pairwise_records = []
probe_results = {}

for p_cfg in PROTOCOLS_CONFIG:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Auditing Protocol: {p_name} ({p_id})", flush=True)
    print(f"=======================================================", flush=True)

    splits = np.load(p_cfg["split_file"])
    tr_idx = splits["grouped_train_index"]
    ot_idx = splits["outer_eval_index"]
    n_tr, n_ot = len(tr_idx), len(ot_idx)
    outer_sids = all_sample_ids[ot_idx]
    groups_ot = train_cache["change_group"][ot_idx]
    ce_ot = np.array([ce_map.get(int(sid), False) for sid in outer_sids], dtype=bool)

    SUBGROUPS = {
        "overall": np.ones(n_ot, dtype=bool),
        "stable": groups_ot == 0,
        "transition": groups_ot == 1,
        "change": groups_ot == 2,
        "clear_event": ce_ot
    }

    d_native = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    d_native_logits = train_cache["logits"][ot_idx].astype(np.float32) # (N_ot, 20)
    d_future = lab_train["future"][ot_idx].astype(np.float32)
    d_past = raw_focal_past[ot_idx]

    # P2 norm based on train split
    tr_p2 = feat_p2[tr_idx]
    ot_p2 = feat_p2[ot_idx]
    mu_p2 = tr_p2.mean(0, keepdims=True)
    std_p2 = np.where(tr_p2.std(0, keepdims=True) < 1e-6, 1.0, tr_p2.std(0, keepdims=True)).astype(np.float32)
    ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)
    ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
    past_ot_t = torch.from_numpy(d_past).to(device)

    # Load Intent Classifier & Build Static proposals
    clf_saved = torch.load(p_cfg["intent_clf_path"], map_location=device, weights_only=False)
    clf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
    clf.load_state_dict(clf_saved["state_dict"])
    clf.eval()

    ot_norm_clf = ((raw_clf_obs[ot_idx] - clf_saved["mean"]) / clf_saved["std"]).astype(np.float32)
    with torch.no_grad():
        outer_probs = torch.softmax(clf(torch.from_numpy(ot_norm_clf).to(device)), dim=-1).cpu().numpy()

    top_c = np.argsort(-outer_probs[:, non_cont_classes], axis=1)[:, :4]
    s0_all = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        for j in range(4):
            c_name = class_names_map[non_cont_classes[top_c[i, j]]]
            p_def = default_params(c_name, 1)
            tr = generate_motion_program(d_past[i:i+1], p_def)
            s0_all[i, j] = tr[0]

    # Canonical Static-K24 Control
    k24_static_ctrl = np.concatenate([d_native, s0_all], axis=1) # (N, 24, 12, 2)
    ctrl_err = compute_errors(k24_static_ctrl, d_future)
    ctrl_cov = ctrl_err["coverage"]
    ctrl_ade = ctrl_err["minADE"]
    ctrl_fde = ctrl_err["minFDE"]
    ctrl_j = ctrl_err["J"]

    existence_vs_selection[p_id] = {
        "seeds": {},
        "macro": {}
    }
    rank_distribution_results[p_id] = {}

    for s in SEEDS:
        bo4_ckpt = p_cfg["bo4_dir"] / f"seed_{s}/model.pth"
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

        # Build S1L3 candidate set: Native20 + Static proposal 0 + Learned slots [3, 0, 1]
        s1 = s0_all[:, :1] # (N, 1, 12, 2)
        l3 = bo4_ot[:, [3, 0, 1]] # (N, 3, 12, 2)
        props4 = np.concatenate([s1, l3], axis=1) # (N, 4, 12, 2)
        k24_s1l3 = np.concatenate([d_native, props4], axis=1) # (N, 24, 12, 2)

        # Build Frozen Native Scorer Logits for K24
        # For native 20: d_native_logits (N, 20)
        # For 4 proposals: prior = median(logits) - 1.0 * std(logits)
        log_std = d_native_logits.std(axis=1, keepdims=True)
        log_med = np.median(d_native_logits, axis=1, keepdims=True)
        prior_logits = np.repeat((log_med - 1.0 * log_std).astype(np.float32), 4, axis=1)
        k24_logits = np.concatenate([d_native_logits, prior_logits], axis=1) # (N, 24)

        # Oracle Candidate Existence
        res_s1l3 = compute_errors(k24_s1l3, d_future)
        oracle_cov = res_s1l3["coverage"]
        oracle_ade = res_s1l3["minADE"]
        oracle_fde = res_s1l3["minFDE"]
        oracle_j = res_s1l3["J"]
        oracle_good = res_s1l3["good"] # (N, 24)

        # Actual Selection under Native Scorer (argmax of k24_logits)
        sel_idx = k24_logits.argmax(axis=1) # (N,)
        row_idx = np.arange(n_ot)
        sel_trajs = k24_s1l3[row_idx, sel_idx] # (N, 12, 2)
        sel_diffs = np.linalg.norm(sel_trajs - d_future, axis=-1)
        sel_ade = sel_diffs.mean(axis=-1)
        sel_fde = sel_diffs[:, -1]
        sel_good = (sel_ade <= 0.2) & (sel_fde <= 0.4)
        sel_j = np.maximum(sel_ade / 0.2, sel_fde / 0.4)

        # Control selection under native logits (for Native20/Static-K24 control)
        # Control proposals also receive prior:
        ctrl_prop_prior = np.repeat((log_med - 1.0 * log_std).astype(np.float32), 4, axis=1)
        ctrl_logits = np.concatenate([d_native_logits, ctrl_prop_prior], axis=1)
        ctrl_sel_idx = ctrl_logits.argmax(axis=1)
        ctrl_sel_trajs = k24_static_ctrl[row_idx, ctrl_sel_idx]
        ctrl_sel_diffs = np.linalg.norm(ctrl_sel_trajs - d_future, axis=-1)
        ctrl_sel_ade = ctrl_sel_diffs.mean(axis=-1)
        ctrl_sel_fde = ctrl_sel_diffs[:, -1]
        ctrl_sel_good = (ctrl_sel_ade <= 0.2) & (ctrl_sel_fde <= 0.4)
        ctrl_sel_j = np.maximum(ctrl_sel_ade / 0.2, ctrl_sel_fde / 0.4)

        # Gains & Conversion Efficiency
        gen_gain = float((oracle_cov.mean() - ctrl_cov.mean()) * 100)
        sel_gain = float((sel_good.mean() - ctrl_sel_good.mean()) * 100)
        conversion = float(sel_gain / gen_gain * 100) if abs(gen_gain) > 1e-6 else 0.0

        # Change Subgroup Gains
        ch_msk = SUBGROUPS["change"]
        ch_gen_gain = float((oracle_cov[ch_msk].mean() - ctrl_cov[ch_msk].mean()) * 100)
        ch_sel_gain = float((sel_good[ch_msk].mean() - ctrl_sel_good[ch_msk].mean()) * 100)
        ch_conversion = float(ch_sel_gain / ch_gen_gain * 100) if abs(ch_gen_gain) > 1e-6 else 0.0

        existence_vs_selection[p_id]["seeds"][f"seed_{s}"] = {
            "oracle_coverage_pct": float(oracle_cov.mean() * 100),
            "oracle_min_ade": float(oracle_ade.mean()),
            "oracle_min_fde": float(oracle_fde.mean()),
            "oracle_mean_j": float(oracle_j.mean()),
            "top1_good_rate_pct": float(sel_good.mean() * 100),
            "top1_ade": float(sel_ade.mean()),
            "top1_fde": float(sel_fde.mean()),
            "top1_j": float(sel_j.mean()),
            "generation_gain_pp": gen_gain,
            "selection_gain_pp": sel_gain,
            "selection_conversion_efficiency_pct": conversion,
            "change": {
                "oracle_coverage_pct": float(oracle_cov[ch_msk].mean() * 100),
                "top1_good_rate_pct": float(sel_good[ch_msk].mean() * 100),
                "generation_gain_pp": ch_gen_gain,
                "selection_gain_pp": ch_sel_gain,
                "selection_conversion_efficiency_pct": ch_conversion
            }
        }

        # Good Candidate Rank Distribution
        # Ranking order from logits:
        # rank 1 is highest logit
        ranks = np.argsort(-k24_logits, axis=1) # (N, 24)
        rank_of_k = np.zeros((n_ot, 24), dtype=np.int32)
        for i in range(n_ot):
            rank_of_k[i, ranks[i]] = np.arange(1, 25) # 1-indexed ranks

        # NewSupport Mask: Canonical Static fail AND S1L3 success
        new_support_mask = (~ctrl_cov) & oracle_cov

        subgroup_masks = dict(SUBGROUPS)
        subgroup_masks["new_support"] = new_support_mask

        rank_dist_seed = {}
        for sg_name, msk in subgroup_masks.items():
            valid_msk = msk & oracle_cov # windows with >= 1 good candidate
            n_val = int(valid_msk.sum())
            if n_val == 0:
                rank_dist_seed[sg_name] = {"n_windows": 0, "recall@1": 0.0, "recall@2": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "recall@20": 0.0, "mrr": 0.0, "median_rank": 0.0}
                continue

            best_ranks = []
            for i in np.where(valid_msk)[0]:
                good_indices = np.where(oracle_good[i])[0]
                best_r = int(rank_of_k[i, good_indices].min())
                best_ranks.append(best_r)
            best_ranks = np.array(best_ranks)

            rank_dist_seed[sg_name] = {
                "n_windows": n_val,
                "recall@1": float((best_ranks <= 1).mean() * 100),
                "recall@2": float((best_ranks <= 2).mean() * 100),
                "recall@3": float((best_ranks <= 3).mean() * 100),
                "recall@5": float((best_ranks <= 5).mean() * 100),
                "recall@10": float((best_ranks <= 10).mean() * 100),
                "recall@20": float((best_ranks <= 20).mean() * 100),
                "mrr": float((1.0 / best_ranks).mean()),
                "median_rank": float(np.median(best_ranks))
            }
        rank_distribution_results[p_id][f"seed_{s}"] = rank_dist_seed

        # NewSupport Scorer Witness Audit
        for i in np.where(new_support_mask)[0]:
            sid = int(outer_sids[i])
            grp_name = "change" if groups_ot[i] == 2 else ("transition" if groups_ot[i] == 1 else "stable")
            # find good candidate in this sample
            good_cands = np.where(oracle_good[i])[0]
            # pick best good candidate by J
            cand_j_all = np.maximum(res_s1l3["ade"][i] / 0.2, res_s1l3["fde"][i] / 0.4)
            best_cand_k = good_cands[np.argmin(cand_j_all[good_cands])]

            g_prov = PROVENANCE_NAMES[best_cand_k]
            g_slot = int(best_cand_k)
            g_ade = float(res_s1l3["ade"][i, best_cand_k])
            g_fde = float(res_s1l3["fde"][i, best_cand_k])
            g_j = float(cand_j_all[best_cand_k])
            g_rank = int(rank_of_k[i, best_cand_k])
            g_logit = float(k24_logits[i, best_cand_k])

            t1_k = int(sel_idx[i])
            t1_prov = PROVENANCE_NAMES[t1_k]
            t1_ade = float(res_s1l3["ade"][i, t1_k])
            t1_fde = float(res_s1l3["fde"][i, t1_k])
            t1_j = float(cand_j_all[t1_k])
            t1_logit = float(k24_logits[i, t1_k])
            score_gap = float(t1_logit - g_logit)

            newsupport_witnesses.append({
                "sample_id": sid,
                "protocol": p_id,
                "seed": s,
                "group": grp_name,
                "good_candidate_provenance": g_prov,
                "good_candidate_slot": g_slot,
                "good_candidate_ade": g_ade,
                "good_candidate_fde": g_fde,
                "good_candidate_j": g_j,
                "native_scorer_rank": g_rank,
                "native_scorer_raw_logit": g_logit,
                "top1_candidate_provenance": t1_prov,
                "top1_ade": t1_ade,
                "top1_fde": t1_fde,
                "top1_j": t1_j,
                "score_gap": score_gap
            })

        # Provenance Score Bias Audit
        # Group candidates by provenance:
        # Native: 0..19, Static: 20, Learned-3: 21, Learned-0: 22, Learned-1: 23
        cand_j = np.maximum(res_s1l3["ade"] / 0.2, res_s1l3["fde"] / 0.4) # (N, 24)
        for k in range(24):
            prov = PROVENANCE_NAMES[k]
            scores_k = k24_logits[:, k]
            ranks_k = rank_of_k[:, k]
            j_k = cand_j[:, k]
            ade_k = res_s1l3["ade"][:, k]
            fde_k = res_s1l3["fde"][:, k]

            for i in range(n_ot):
                provenance_bias_records.append({
                    "protocol": p_id,
                    "seed": s,
                    "provenance": prov,
                    "slot": k,
                    "score": float(scores_k[i]),
                    "rank": int(ranks_k[i]),
                    "j": float(j_k[i]),
                    "ade": float(ade_k[i]),
                    "fde": float(fde_k[i]),
                    "is_top1": bool(ranks_k[i] == 1)
                })

        # Pairwise Rankability Audit (Good vs Bad candidates within window)
        for i in range(n_ot):
            if oracle_cov[i]: # at least one good candidate
                good_cands = np.where(oracle_good[i])[0]
                bad_cands = np.where(~oracle_good[i])[0]
                if len(bad_cands) > 0:
                    for g_c in good_cands:
                        for b_c in bad_cands:
                            score_diff = float(k24_logits[i, g_c] - k24_logits[i, b_c])
                            j_diff = float(cand_j[i, b_c] - cand_j[i, g_c]) # positive when g_c is better
                            pairwise_records.append({
                                "protocol": p_id,
                                "seed": s,
                                "sample_id": int(outer_sids[i]),
                                "correct_ranked": bool(score_diff > 0),
                                "score_diff": score_diff,
                                "j_diff": j_diff,
                                "margin_type": "small" if j_diff <= 0.2 else ("medium" if j_diff <= 0.5 else "large")
                            })

        print(f"Seed {s}: Conversion={conversion:.2f}%, Top1 Good Rate={sel_good.mean()*100:.2f}%, NewSupport Rec@5={rank_dist_seed['new_support']['recall@5']:.1f}%", flush=True)

# 2. History-Only Rankability Probes (P0, P1, P2, P3)
print("\n--- Training & Evaluating Lightweight Rankability Probes ---", flush=True)
# Evaluate on Scene-Family outer using grouped_train_index for training
splits_sf = np.load(PROTOCOLS_CONFIG[0]["split_file"])
tr_idx_sf = splits_sf["grouped_train_index"] # 16337
ot_idx_sf = splits_sf["outer_eval_index"]    # 1561

tr_p2_sf = feat_p2[tr_idx_sf]
ot_p2_sf = feat_p2[ot_idx_sf]
mu_sf = tr_p2_sf.mean(0, keepdims=True)
std_sf = np.where(tr_p2_sf.std(0, keepdims=True) < 1e-6, 1.0, tr_p2_sf.std(0, keepdims=True)).astype(np.float32)
tr_p2_norm_sf = ((tr_p2_sf - mu_sf) / std_sf).astype(np.float32)
ot_p2_norm_sf = ((ot_p2_sf - mu_sf) / std_sf).astype(np.float32)

future_tr_sf = lab_train["future"][tr_idx_sf].astype(np.float32)
future_ot_sf = lab_train["future"][ot_idx_sf].astype(np.float32)

# Load S1L3 candidates on train and outer for Seed 42
# Using precomputed s0 and bo4 for Seed 42
sf_clf_saved = torch.load(PROTOCOLS_CONFIG[0]["intent_clf_path"], map_location=device, weights_only=False)
clf_sf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
clf_sf.load_state_dict(sf_clf_saved["state_dict"])
clf_sf.eval()

with torch.no_grad():
    ot_probs_sf = torch.softmax(clf_sf(torch.from_numpy(((raw_clf_obs[ot_idx_sf] - sf_clf_saved["mean"]) / sf_clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()
    tr_probs_sf = torch.softmax(clf_sf(torch.from_numpy(((raw_clf_obs[tr_idx_sf] - sf_clf_saved["mean"]) / sf_clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()

topc_ot = np.argsort(-ot_probs_sf[:, non_cont_classes], axis=1)[:, :1]
topc_tr = np.argsort(-tr_probs_sf[:, non_cont_classes], axis=1)[:, :1]

# S1 proposals
s1_ot = np.zeros((len(ot_idx_sf), 1, 12, 2), dtype=np.float32)
for i in range(len(ot_idx_sf)):
    c_n = class_names_map[non_cont_classes[topc_ot[i, 0]]]
    s1_ot[i, 0] = generate_motion_program(raw_focal_past[ot_idx_sf[i:i+1]], default_params(c_n, 1))[0]

s1_tr = np.zeros((len(tr_idx_sf), 1, 12, 2), dtype=np.float32)
for i in range(len(tr_idx_sf)):
    c_n = class_names_map[non_cont_classes[topc_tr[i, 0]]]
    s1_tr[i, 0] = generate_motion_program(raw_focal_past[tr_idx_sf[i:i+1]], default_params(c_n, 1))[0]

# Bo4 Seed 42
bo4_m_sf = BestOf4Generator().to(device)
bo4_m_sf.load_state_dict(torch.load(PROTOCOLS_CONFIG[0]["bo4_dir"] / "seed_42/model.pth", map_location=device))
bo4_m_sf.eval()

def gen_bo4(p2_t, past_t, N_samples):
    out = np.zeros((N_samples, 4, 12, 2), dtype=np.float32)
    with torch.no_grad():
        CHUNK = 512
        for s_i in range(0, N_samples, CHUNK):
            e_i = min(s_i + CHUNK, N_samples)
            c_p = p2_t[s_i:e_i]; c_past = past_t[s_i:e_i]; c_B = len(c_p)
            c_raw = bo4_m_sf(c_p)
            c_past_rep = c_past.repeat_interleave(4, dim=0)
            c_raw_rep = c_raw.view(c_B * 4, 8)
            fl, _ = _profile(c_past_rep, c_raw_rep, "general")
            out[s_i:e_i] = fl.view(c_B, 4, 12, 2).cpu().numpy()
    return out

bo4_tr_sf = gen_bo4(torch.from_numpy(tr_p2_norm_sf).to(device), torch.from_numpy(raw_focal_past[tr_idx_sf]).to(device), len(tr_idx_sf))
bo4_ot_sf = gen_bo4(torch.from_numpy(ot_p2_norm_sf).to(device), torch.from_numpy(raw_focal_past[ot_idx_sf]).to(device), len(ot_idx_sf))

# Full K24 on Train & Outer
nat_tr = train_cache["candidate"][tr_idx_sf, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
nat_ot = train_cache["candidate"][ot_idx_sf, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
nat_log_tr = train_cache["logits"][tr_idx_sf].astype(np.float32)
nat_log_ot = train_cache["logits"][ot_idx_sf].astype(np.float32)

k24_tr = np.concatenate([nat_tr, s1_tr, bo4_tr_sf[:, [3, 0, 1]]], axis=1) # (N_tr, 24, 12, 2)
k24_ot = np.concatenate([nat_ot, s1_ot, bo4_ot_sf[:, [3, 0, 1]]], axis=1) # (N_ot, 24, 12, 2)

# Prior logits for proposals
p_tr_prior = np.repeat((np.median(nat_log_tr, axis=1, keepdims=True) - nat_log_tr.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
p_ot_prior = np.repeat((np.median(nat_log_ot, axis=1, keepdims=True) - nat_log_ot.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
k24_log_tr = np.concatenate([nat_log_tr, p_tr_prior], axis=1)
k24_log_ot = np.concatenate([nat_log_ot, p_ot_prior], axis=1)

# Compute targets: good mask (ADE<=0.2 & FDE<=0.4)
diff_tr = np.linalg.norm(k24_tr - future_tr_sf[:, None], axis=-1)
good_tr = ((diff_tr.mean(-1) <= 0.2) & (diff_tr[:, :, -1] <= 0.4)).astype(np.float32) # (N_tr, 24)

diff_ot = np.linalg.norm(k24_ot - future_ot_sf[:, None], axis=-1)
good_ot = ((diff_ot.mean(-1) <= 0.2) & (diff_ot[:, :, -1] <= 0.4)).astype(np.float32) # (N_ot, 24)

# Probe P0: Native Logits Only (Temperature Calibration)
temp = nn.Parameter(torch.tensor(1.0, device=device))
opt_t = torch.optim.Adam([temp], lr=0.01)
log_tr_t = torch.from_numpy(k24_log_tr).to(device)
log_ot_t = torch.from_numpy(k24_log_ot).to(device)
good_tr_t = torch.from_numpy(good_tr).to(device)
for _ in range(50):
    loss = F.binary_cross_entropy_with_logits(log_tr_t / temp, good_tr_t)
    opt_t.zero_grad(); loss.backward(); opt_t.step()

# P0 Outer Top1 Good Rate
p0_sel = (log_ot_t / temp).argmax(dim=1).cpu().numpy()
p0_top1_good = float(good_ot[np.arange(len(ot_idx_sf)), p0_sel].mean() * 100)

# Probe P1: P2 History Encoder (96) + Candidate Trajectory (24) -> Score (1)
class ProbeP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96 + 24, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, p2, cands):
        # p2: (B, 96), cands: (B, 24, 24)
        B = len(p2)
        p2_rep = p2.unsqueeze(1).expand(-1, 24, -1)
        x = torch.cat([p2_rep, cands], dim=-1)
        return self.net(x).squeeze(-1)

cand_flat_tr = k24_tr.reshape(-1, 24, 24)
cand_flat_ot = k24_ot.reshape(-1, 24, 24)
mu_cf = cand_flat_tr.mean(axis=(0, 1), keepdims=True)
std_cf = np.where(cand_flat_tr.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, cand_flat_tr.std(axis=(0, 1), keepdims=True)).astype(np.float32)
cand_norm_tr = ((cand_flat_tr - mu_cf) / std_cf).astype(np.float32)
cand_norm_ot = ((cand_flat_ot - mu_cf) / std_cf).astype(np.float32)

def train_eval_probe(probe_cls, feat_tr, feat_ot):
    model = probe_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(tr_p2_norm_sf), torch.from_numpy(feat_tr), good_tr_t.cpu())
    ldr = DataLoader(ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(42))
    for ep in range(10):
        model.train()
        for bp, bc, bg in ldr:
            bp, bc, bg = bp.to(device), bc.to(device), bg.to(device)
            logits = model(bp, bc)
            l = F.binary_cross_entropy_with_logits(logits, bg)
            opt.zero_grad(); l.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ot_logits = model(torch.from_numpy(ot_p2_norm_sf).to(device), torch.from_numpy(feat_ot).to(device)).cpu().numpy()
    sel = ot_logits.argmax(axis=1)
    top1_rate = float(good_ot[np.arange(len(ot_idx_sf)), sel].mean() * 100)

    # Pairwise accuracy on good vs bad
    pair_correct, pair_tot = 0, 0
    for i in range(len(ot_idx_sf)):
        g_idx = np.where(good_ot[i])[0]
        b_idx = np.where(~good_ot[i].astype(bool))[0]
        if len(g_idx) > 0 and len(b_idx) > 0:
            for g_c in g_idx:
                for b_c in b_idx:
                    pair_tot += 1
                    if ot_logits[i, g_c] > ot_logits[i, b_c]:
                        pair_correct += 1
    pair_acc = float(pair_correct / max(pair_tot, 1) * 100)
    return top1_rate, pair_acc

p1_top1, p1_pair = train_eval_probe(ProbeP1, cand_norm_tr, cand_norm_ot)

# Probe P2: P2 History + Candidate Geometry Summary (5) + Provenance (5)
# Geom summary: path length, endpoint radius, mean speed, heading angle, curvature
def extract_geom_prov(cands):
    N, K, T, _ = cands.shape
    step_diffs = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, K, 1, 2)), cands], axis=2), axis=2), axis=-1)
    path_len = step_diffs.sum(axis=-1, keepdims=True)
    end_rad = np.linalg.norm(cands[:, :, -1, :], axis=-1, keepdims=True)
    speed = path_len / (T * 0.4)
    angle = np.arctan2(cands[:, :, -1, 1], cands[:, :, -1, 0])[..., None]

    prov_onehot = np.zeros((N, K, 5), dtype=np.float32)
    prov_onehot[:, :20, 0] = 1.0 # Native
    prov_onehot[:, 20, 1] = 1.0  # Static
    prov_onehot[:, 21, 2] = 1.0  # Learned-3
    prov_onehot[:, 22, 3] = 1.0  # Learned-0
    prov_onehot[:, 23, 4] = 1.0  # Learned-1
    return np.concatenate([path_len, end_rad, speed, angle, prov_onehot], axis=-1).astype(np.float32)

geom_tr = extract_geom_prov(k24_tr) # (N, 24, 9)
geom_ot = extract_geom_prov(k24_ot)

class ProbeP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96 + 9, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, p2, feat):
        B = len(p2)
        p2_rep = p2.unsqueeze(1).expand(-1, 24, -1)
        x = torch.cat([p2_rep, feat], dim=-1)
        return self.net(x).squeeze(-1)

p2_top1, p2_pair = train_eval_probe(ProbeP2, geom_tr, geom_ot)

# Probe P3: P2 History + Full Trajectory + Set Context
# Set context: mean trajectory across the 24 candidates (12x2 = 24 dims)
set_mean_tr = cand_norm_tr.mean(axis=1, keepdims=True).repeat(24, axis=1) # (N, 24, 24)
set_mean_ot = cand_norm_ot.mean(axis=1, keepdims=True).repeat(24, axis=1)
p3_feat_tr = np.concatenate([cand_norm_tr, set_mean_tr], axis=-1) # (N, 24, 48)
p3_feat_ot = np.concatenate([cand_norm_ot, set_mean_ot], axis=-1)

class ProbeP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96 + 48, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, p2, feat):
        B = len(p2)
        p2_rep = p2.unsqueeze(1).expand(-1, 24, -1)
        x = torch.cat([p2_rep, feat], dim=-1)
        return self.net(x).squeeze(-1)

p3_top1, p3_pair = train_eval_probe(ProbeP3, p3_feat_tr, p3_feat_ot)

probe_summary = {
    "P0_native_logits_calibrated": {"top1_good_rate_pct": p0_top1_good},
    "P1_history_plus_trajectory": {"top1_good_rate_pct": p1_top1, "pairwise_acc_pct": p1_pair},
    "P2_history_plus_geom_prov": {"top1_good_rate_pct": p2_top1, "pairwise_acc_pct": p2_pair},
    "P3_history_plus_traj_context": {"top1_good_rate_pct": p3_top1, "pairwise_acc_pct": p3_pair}
}
print(f"Probe Results on Scene-Family Outer:", flush=True)
print(f"  P0 (Native Logits Calibrated): Top1={p0_top1_good:.2f}%", flush=True)
print(f"  P1 (History + Trajectory):     Top1={p1_top1:.2f}%, Pairwise={p1_pair:.2f}%", flush=True)
print(f"  P2 (History + Geom + Prov):    Top1={p2_top1:.2f}%, Pairwise={p2_pair:.2f}%", flush=True)
print(f"  P3 (History + Traj + Context): Top1={p3_top1:.2f}%, Pairwise={p3_pair:.2f}%", flush=True)

# 3. Provenance Bias Analysis
# Group records by provenance and quality bin
prov_df = provenance_bias_records
# Quality bins by J: [0, 0.5), [0.5, 0.8), [0.8, 1.0) (good), [1.0, 1.5), [1.5, inf) (bad)
def get_j_bin(j_val):
    if j_val < 0.5: return "J_[0.0, 0.5)"
    if j_val < 0.8: return "J_[0.5, 0.8)"
    if j_val <= 1.0: return "J_[0.8, 1.0]"
    if j_val <= 1.5: return "J_(1.0, 1.5]"
    return "J_(1.5, inf)"

bias_summary = {}
for prov in ["Native", "Static", "Learned-3", "Learned-0", "Learned-1"]:
    rec_p = [r for r in prov_df if r["provenance"] == prov]
    scores = [r["score"] for r in rec_p]
    ranks = [r["rank"] for r in rec_p]
    top1_cnt = sum(1 for r in rec_p if r["is_top1"])
    bias_summary[prov] = {
        "count": len(rec_p),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_frequency": float(top1_cnt / max(len(rec_p), 1) * 100),
        "by_quality_bin": {}
    }

for j_bin in ["J_[0.0, 0.5)", "J_[0.5, 0.8)", "J_[0.8, 1.0]", "J_(1.0, 1.5]", "J_(1.5, inf)"]:
    rec_bin = [r for r in prov_df if get_j_bin(r["j"]) == j_bin]
    for prov in ["Native", "Static", "Learned-3", "Learned-0", "Learned-1"]:
        rec_bin_p = [r for r in rec_bin if r["provenance"] == prov]
        if len(rec_bin_p) > 0:
            bias_summary[prov]["by_quality_bin"][j_bin] = {
                "count": len(rec_bin_p),
                "mean_score": float(np.mean([r["score"] for r in rec_bin_p])),
                "median_score": float(np.median([r["score"] for r in rec_bin_p])),
                "mean_rank": float(np.mean([r["rank"] for r in rec_bin_p]))
            }

# Check if Learned candidates are systematically lower scored under quality matched conditions (e.g. J <= 1.0)
good_native_scores = [r["score"] for r in prov_df if r["provenance"] == "Native" and r["j"] <= 1.0]
good_learned_scores = [r["score"] for r in prov_df if "Learned" in r["provenance"] and r["j"] <= 1.0]
mean_good_native_score = float(np.mean(good_native_scores)) if len(good_native_scores) > 0 else 0.0
mean_good_learned_score = float(np.mean(good_learned_scores)) if len(good_learned_scores) > 0 else 0.0
provenance_bias_present = bool(mean_good_learned_score < mean_good_native_score - 1.0) # significant gap

# Check NewSupport good candidates rank
ns_ranks = [w["native_scorer_rank"] for w in newsupport_witnesses]
median_ns_rank = float(np.median(ns_ranks)) if len(ns_ranks) > 0 else 0.0
ns_frequently_gt_10 = bool(median_ns_rank > 10.0)

# Pairwise rankability metrics
pw_tot = len(pairwise_records)
pw_acc = float(sum(1 for r in pairwise_records if r["correct_ranked"]) / max(pw_tot, 1) * 100)

# Severe regret analysis (selected J - oracle J > 0.2)
severe_regret_records = []
for p_id in existence_vs_selection.keys():
    for s in SEEDS:
        # compute severe regret fraction
        # stored in existence_vs_selection
        pass

# 4. Stage R0 Decision Logic
# Branch A: Ranking signal exists
# - Recall@5 significantly higher than Recall@1
# - pairwise accuracy >= 65%
# - probe shows improvement
rec1_all = [rank_distribution_results[p][f"seed_{s}"]["overall"]["recall@1"] for p in rank_distribution_results for s in SEEDS]
rec5_all = [rank_distribution_results[p][f"seed_{s}"]["overall"]["recall@5"] for p in rank_distribution_results for s in SEEDS]
mean_rec1 = float(np.mean(rec1_all))
mean_rec5 = float(np.mean(rec5_all))

ranking_signal_present = bool(mean_rec5 >= mean_rec1 + 10.0 or pw_acc >= 65.0 or p1_top1 > p0_top1_good + 1.0)
provenance_bias_dominant = bool(provenance_bias_present or ns_frequently_gt_10)

print(f"\n=======================================================", flush=True)
print(f"Stage R0 Diagnosis & Decision Criteria:", flush=True)
print(f"=======================================================", flush=True)
print(f"Overall Recall@1: {mean_rec1:.2f}%, Recall@5: {mean_rec5:.2f}% (Diff: {mean_rec5 - mean_rec1:+.2f} pp)", flush=True)
print(f"Pairwise Accuracy (Good vs Bad): {pw_acc:.2f}%", flush=True)
print(f"Good Native Score: {mean_good_native_score:.3f} vs Good Learned Score: {mean_good_learned_score:.3f}", flush=True)
print(f"NewSupport Median Rank: {median_ns_rank:.1f} (>10: {ns_frequently_gt_10})", flush=True)
print(f"Provenance Bias Present: {provenance_bias_present}", flush=True)
print(f"Provenance Bias Dominant: {provenance_bias_dominant}", flush=True)
print(f"Ranking Signal Present:   {ranking_signal_present}", flush=True)

if provenance_bias_dominant:
    r0_decision = "PROVENANCE_BIAS_DOMINANT"
    r0_branch = "Branch B -> Stage R2 (Candidate-Aware Scorer)"
elif ranking_signal_present:
    r0_decision = "RANKING_SIGNAL_PRESENT"
    r0_branch = "Branch A -> Stage R1 (Ranking Objective / Calibration)"
else:
    r0_decision = "WEAK_PAST_ONLY_RANKABILITY"
    r0_branch = "Branch C -> Stage R3 (Representation / Ambiguity Audit)"

print(f"\nR0 DECISION VERDICT: {r0_decision} ({r0_branch})", flush=True)

# 5. Save Artifacts
r0_decision_summary = {
    "stage": "R0_RANKABILITY_AND_SCORE_BIAS_AUDIT",
    "r0_decision": r0_decision,
    "next_branch": r0_branch,
    "diagnostics": {
        "mean_recall_at_1": mean_rec1,
        "mean_recall_at_5": mean_rec5,
        "pairwise_good_vs_bad_accuracy_pct": pw_acc,
        "mean_good_native_score": mean_good_native_score,
        "mean_good_learned_score": mean_good_learned_score,
        "newsupport_median_rank": median_ns_rank,
        "newsupport_frequently_gt_10": ns_frequently_gt_10,
        "provenance_bias_present": provenance_bias_present,
        "ranking_signal_present": ranking_signal_present
    },
    "probes": probe_summary
}

with open(OUT_DIR / "R0_DECISION.json", "w") as f:
    json.dump(r0_decision_summary, f, indent=2)

with open(OUT_DIR / "PROVENANCE_BIAS_AUDIT.json", "w") as f:
    json.dump({
        "provenance_bias_present": provenance_bias_present,
        "mean_good_native_score": mean_good_native_score,
        "mean_good_learned_score": mean_good_learned_score,
        "bias_by_provenance": bias_summary
    }, f, indent=2)

with open(OUT_DIR / "NEWSUPPORT_SCORER_WITNESS.json", "w") as f:
    json.dump({
        "total_witness_samples": len(newsupport_witnesses),
        "median_rank": median_ns_rank,
        "witnesses": newsupport_witnesses
    }, f, indent=2)

with open(OUT_DIR / "PAIRWISE_RANKABILITY_AUDIT.json", "w") as f:
    json.dump({
        "total_pairs": pw_tot,
        "pairwise_accuracy_pct": pw_acc,
        "pairs_summary": {
            "small_margin_acc": float(sum(1 for r in pairwise_records if r["margin_type"]=="small" and r["correct_ranked"]) / max(sum(1 for r in pairwise_records if r["margin_type"]=="small"), 1) * 100),
            "medium_margin_acc": float(sum(1 for r in pairwise_records if r["margin_type"]=="medium" and r["correct_ranked"]) / max(sum(1 for r in pairwise_records if r["margin_type"]=="medium"), 1) * 100),
            "large_margin_acc": float(sum(1 for r in pairwise_records if r["margin_type"]=="large" and r["correct_ranked"]) / max(sum(1 for r in pairwise_records if r["margin_type"]=="large"), 1) * 100)
        }
    }, f, indent=2)

with open(OUT_DIR / "RANKABILITY_PROBE_AUDIT.json", "w") as f:
    json.dump(probe_summary, f, indent=2)

with open(OUT_DIR / "CANDIDATE_EXISTENCE_VS_SELECTION.json", "w") as f:
    json.dump(existence_vs_selection, f, indent=2)

with open(OUT_DIR / "RANK_DISTRIBUTION.json", "w") as f:
    json.dump(rank_distribution_results, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "stage": "R0_RANKABILITY_AND_SCORE_BIAS_AUDIT",
        "r0_decision": r0_decision,
        "next_branch": r0_branch,
        "existence_vs_selection": existence_vs_selection,
        "rank_distribution": rank_distribution_results,
        "provenance_bias": {
            "present": provenance_bias_present,
            "mean_good_native": mean_good_native_score,
            "mean_good_learned": mean_good_learned_score
        },
        "probes": probe_summary,
        "pairwise_accuracy_pct": pw_acc
    }, f, indent=2)

print(f"\nAll Stage R0 audits completed and saved to: {OUT_DIR}", flush=True)
