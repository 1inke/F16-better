import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
CONFIRM_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
R2_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R2_candidate_aware_scorer"
M1_DIR = REPO / "CALIBRATED_TOP3_RESEARCH_20261007" / "M1_probability_calibration"
OUT_M2_DIR = REPO / "CALIBRATED_TOP3_RESEARCH_20261007" / "M2_portfolio_composition"
OUT_M3_DIR = REPO / "CALIBRATED_TOP3_RESEARCH_20261007" / "M3_selective_interface"
OUT_M2_DIR.mkdir(parents=True, exist_ok=True)
OUT_M3_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Executing Phases M2 & M3 on: {device}", flush=True)

# 1. Load Base Data
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
cfg = load_config()

all_sample_ids = train_cache["sample_id"]
motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)

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

class CandidateAwareScorer(nn.Module):
    def __init__(self, in_p2=96, in_cand=24, in_geom=4, in_prov=5, in_set=4, hidden=64):
        super().__init__()
        self.p2_net = nn.Sequential(
            nn.Linear(in_p2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
            nn.GELU()
        )
        self.cand_net = nn.Sequential(
            nn.Linear(in_cand + in_geom + in_prov, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
            nn.GELU()
        )
        self.delta_head = nn.Sequential(
            nn.Linear(32 + 32 + in_set, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, p2, cand_feat, set_feat, native_logits):
        B, K, _ = cand_feat.shape
        h_p2 = self.p2_net(p2).unsqueeze(1).expand(-1, K, -1)
        h_cand = self.cand_net(cand_feat)
        h_set = set_feat.unsqueeze(1).expand(-1, K, -1)
        h_cat = torch.cat([h_p2, h_cand, h_set], dim=-1)
        delta = self.delta_head(h_cat).squeeze(-1)
        score_new = native_logits + delta
        return score_new, delta

class Top3ProbabilityCalibrator(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

def extract_features(cands):
    N, K, T, _ = cands.shape
    cand_flat = cands.reshape(N, K, 24)
    end_rad = np.linalg.norm(cands[:, :, -1, :], axis=-1, keepdims=True)
    step_diffs = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, K, 1, 2)), cands], axis=2), axis=2), axis=-1)
    path_len = step_diffs.sum(axis=-1, keepdims=True)
    speed = path_len / (T * 0.4)
    angle = np.arctan2(cands[:, :, -1, 1], cands[:, :, -1, 0])[..., None]
    geom = np.concatenate([end_rad, path_len, speed, angle], axis=-1)

    prov = np.zeros((N, K, 5), dtype=np.float32)
    prov[:, :20, 0] = 1.0
    prov[:, 20, 1] = 1.0
    prov[:, 21, 2] = 1.0
    prov[:, 22, 3] = 1.0
    prov[:, 23, 4] = 1.0

    cand_feat = np.concatenate([cand_flat, geom, prov], axis=-1)
    end_xy = cands[:, :, -1, :]
    set_mean = end_xy.mean(axis=1)
    set_std = end_xy.std(axis=1)
    set_feat = np.concatenate([set_mean, set_std], axis=-1)
    return cand_feat, set_feat

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

PROTOCOLS = [
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

m2_results = {}
m3_results = {}

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Running M2 Portfolio & M3 Interface on: {p_name} ({p_id})", flush=True)
    print(f"=======================================================", flush=True)

    splits = np.load(p_cfg["split_file"])
    tr_idx = splits["grouped_train_index"]
    ot_idx = splits["outer_eval_index"]
    n_tr, n_ot = len(tr_idx), len(ot_idx)
    groups_ot = train_cache["change_group"][ot_idx]

    SUBGROUPS = {
        "overall": np.ones(n_ot, dtype=bool),
        "stable": groups_ot == 0,
        "transition": groups_ot == 1,
        "change": groups_ot == 2
    }

    tr_p2 = feat_p2[tr_idx]
    ot_p2 = feat_p2[ot_idx]
    mu_p2 = tr_p2.mean(0, keepdims=True)
    std_p2 = np.where(tr_p2.std(0, keepdims=True) < 1e-6, 1.0, tr_p2.std(0, keepdims=True)).astype(np.float32)
    tr_p2_norm = ((tr_p2 - mu_p2) / std_p2).astype(np.float32)
    ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

    future_ot = lab_train["future"][ot_idx].astype(np.float32)
    past_ot = raw_focal_past[ot_idx]
    past_tr = raw_focal_past[tr_idx]

    clf_saved = torch.load(p_cfg["intent_clf_path"], map_location=device, weights_only=False)
    clf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
    clf.load_state_dict(clf_saved["state_dict"])
    clf.eval()

    with torch.no_grad():
        ot_probs = torch.softmax(clf(torch.from_numpy(((raw_clf_obs[ot_idx] - clf_saved["mean"]) / clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()

    topc_ot = np.argsort(-ot_probs[:, non_cont_classes], axis=1)[:, :4]

    s_ot = np.zeros((n_ot, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        c_name = class_names_map[non_cont_classes[topc_ot[i, 0]]]
        s_ot[i] = generate_motion_program(past_ot[i:i+1], default_params(c_name, 1))[0]

    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    nat_log_ot = train_cache["logits"][ot_idx].astype(np.float32)
    prior_log_ot = np.repeat((np.median(nat_log_ot, axis=1, keepdims=True) - nat_log_ot.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
    k24_log_ot = np.concatenate([nat_log_ot, prior_log_ot], axis=1)

    sorted_nat_indices = np.argsort(-nat_log_ot[:, :20], axis=1)
    n1_ot = nat_ot[np.arange(n_ot), sorted_nat_indices[:, 0]]
    n2_ot = nat_ot[np.arange(n_ot), sorted_nat_indices[:, 1]]
    n3_ot = nat_ot[np.arange(n_ot), sorted_nat_indices[:, 2]]

    # kNN mode entropy
    ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
    tr_p2_t = torch.from_numpy(tr_p2_norm).to(device)
    dist_mat = torch.cdist(ot_p2_t, tr_p2_t)
    _, knn20_ot_idx = torch.topk(dist_mat, 20, largest=False)
    knn20_ot_np = knn20_ot_idx.cpu().numpy()

    kinds_tr = lab_train["kind"][tr_idx]
    p2_knn_entropy_ot = np.zeros(n_ot, dtype=np.float32)
    for i in range(n_ot):
        counts = np.bincount(kinds_tr[knn20_ot_np[i]], minlength=5)
        probs = counts / 20.0
        probs = probs[probs > 0]
        p2_knn_entropy_ot[i] = -np.sum(probs * np.log2(probs))

    m2_results[p_id] = {"seeds": {}}
    m3_results[p_id] = {"seeds": {}}

    for s in SEEDS:
        print(f"\n--- Protocol {p_id} | Seed {s} ---", flush=True)
        bo4_ckpt = p_cfg["bo4_dir"] / f"seed_{s}/model.pth"
        bo4_m = BestOf4Generator().to(device)
        bo4_m.load_state_dict(torch.load(bo4_ckpt, map_location=device))
        bo4_m.eval()

        def gen_bo4(p2_arr, past_arr, N):
            out = np.zeros((N, 4, 12, 2), dtype=np.float32)
            p2_t = torch.from_numpy(p2_arr).to(device)
            past_t = torch.from_numpy(past_arr).to(device)
            with torch.no_grad():
                CHUNK = 512
                for s_i in range(0, N, CHUNK):
                    e_i = min(s_i + CHUNK, N)
                    c_p = p2_t[s_i:e_i]; c_past = past_t[s_i:e_i]; c_B = len(c_p)
                    c_raw = bo4_m(c_p)
                    c_past_rep = c_past.repeat_interleave(4, dim=0)
                    c_raw_rep = c_raw.view(c_B * 4, 8)
                    fl, _ = _profile(c_past_rep, c_raw_rep, "general")
                    out[s_i:e_i] = fl.view(c_B, 4, 12, 2).cpu().numpy()
            return out

        bo4_ot = gen_bo4(ot_p2_norm, past_ot, n_ot)
        l3_ot = bo4_ot[:, 3]
        l0_ot = bo4_ot[:, 0]
        l1_ot = bo4_ot[:, 1]

        # Candidate Portfolios Comparison (Phase M2)
        portfolios = {
            "C1_canonical_NSL3": np.stack([n1_ot, s_ot, l3_ot], axis=1),
            "C2_dual_learned_NL3L0": np.stack([n1_ot, l3_ot, l0_ot], axis=1),
            "C3_all_learned_NL3L1": np.stack([n1_ot, l3_ot, l1_ot], axis=1),
            "C4_pure_native_N1N2N3": np.stack([n1_ot, n2_ot, n3_ot], axis=1)
        }

        m2_seed_eval = {}
        for c_name, c_trajs in portfolios.items():
            diff_c = np.linalg.norm(c_trajs - future_ot[:, None], axis=-1)
            ade_c = diff_c.mean(axis=-1)
            fde_c = diff_c[:, :, -1]
            good_c = (ade_c <= 0.2) & (fde_c <= 0.4)
            cov_c = good_c.any(axis=-1)
            min_ade = ade_c.min(axis=-1)

            # Pairwise diversity
            d_01 = np.linalg.norm(c_trajs[:, 0] - c_trajs[:, 1], axis=-1).mean(axis=-1)
            d_02 = np.linalg.norm(c_trajs[:, 0] - c_trajs[:, 2], axis=-1).mean(axis=-1)
            d_12 = np.linalg.norm(c_trajs[:, 1] - c_trajs[:, 2], axis=-1).mean(axis=-1)
            mean_dist = float((d_01 + d_02 + d_12).mean() / 3.0)

            ch_msk = SUBGROUPS["change"]
            m2_seed_eval[c_name] = {
                "overall_coverage_pct": float(cov_c.mean() * 100),
                "change_coverage_pct": float(cov_c[ch_msk].mean() * 100),
                "min_ade_m": float(min_ade.mean()),
                "mean_pairwise_dist_m": mean_dist
            }

        m2_results[p_id]["seeds"][f"seed_{s}"] = m2_seed_eval
        print(f"  [M2 Portfolio] C1 (NSL3): Overall={m2_seed_eval['C1_canonical_NSL3']['overall_coverage_pct']:.2f}%, Change={m2_seed_eval['C1_canonical_NSL3']['change_coverage_pct']:.2f}% | C4 (Pure Native Top3): Overall={m2_seed_eval['C4_pure_native_N1N2N3']['overall_coverage_pct']:.2f}%, Change={m2_seed_eval['C4_pure_native_N1N2N3']['change_coverage_pct']:.2f}%", flush=True)

        # Phase M3: Selective Interface Evaluation
        # Load Phase M1 Calibrator
        cal_ckpt = M1_DIR / "models" / p_id / f"seed_{s}" / "calibrator.pth"
        cal_m = Top3ProbabilityCalibrator(in_dim=12).to(device)
        cal_m.load_state_dict(torch.load(cal_ckpt, map_location=device))
        cal_m.eval()

        # Load Stage R2 Scorer
        r2_ckpt = R2_DIR / "models" / p_id / f"seed_{s}" / "scorer.pth"
        r2_m = CandidateAwareScorer().to(device)
        r2_m.load_state_dict(torch.load(r2_ckpt, map_location=device))
        r2_m.eval()

        k24_ot = np.concatenate([nat_ot, s_ot[:, None], bo4_ot[:, [3, 0, 1]]], axis=1)
        cand_feat_ot, set_feat_ot = extract_features(k24_ot)
        mu_cf = cand_feat_ot.mean(axis=(0, 1), keepdims=True)
        std_cf = np.where(cand_feat_ot.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, cand_feat_ot.std(axis=(0, 1), keepdims=True)).astype(np.float32)
        cand_feat_ot_norm = ((cand_feat_ot - mu_cf) / std_cf).astype(np.float32)

        mu_sf = set_feat_ot.mean(axis=0, keepdims=True)
        std_sf = np.where(set_feat_ot.std(axis=0, keepdims=True) < 1e-6, 1.0, set_feat_ot.std(axis=0, keepdims=True)).astype(np.float32)
        set_feat_ot_norm = ((set_feat_ot - mu_sf) / std_sf).astype(np.float32)

        with torch.no_grad():
            r2_scores_ot, _ = r2_m(
                torch.from_numpy(ot_p2_norm).to(device),
                torch.from_numpy(cand_feat_ot_norm).to(device),
                torch.from_numpy(set_feat_ot_norm).to(device),
                torch.from_numpy(k24_log_ot).to(device)
            )
            r2_scores_ot = r2_scores_ot.cpu().numpy()

        n_sel_ot = nat_log_ot[:, :20].argmax(axis=1)
        row = np.arange(n_ot)
        nat_top1_log = nat_log_ot[row, n_sel_ot]
        sorted_nat = np.sort(nat_log_ot[:, :20], axis=1)[:, ::-1]
        nat_margin = sorted_nat[:, 0] - sorted_nat[:, 1]
        r2_sc_n = r2_scores_ot[row, n_sel_ot]
        r2_sc_s = r2_scores_ot[:, 20]
        r2_sc_l = r2_scores_ot[:, 21]
        disagree = (n_sel_ot != r2_scores_ot.argmax(axis=1)).astype(np.float32)
        p_change = ot_probs[:, non_cont_classes].sum(axis=1)
        d_ns = np.linalg.norm(n1_ot - s_ot, axis=-1).mean(axis=-1)
        d_nl = np.linalg.norm(n1_ot - l3_ot, axis=-1).mean(axis=-1)

        X_cal = np.stack([
            nat_top1_log, nat_margin,
            r2_sc_n, r2_sc_s, r2_sc_l,
            r2_sc_l - r2_sc_n, r2_sc_s - r2_sc_n,
            disagree, p_change, p2_knn_entropy_ot,
            d_ns, d_nl
        ], axis=1)

        mu_x = X_cal.mean(axis=0, keepdims=True)
        std_x = np.where(X_cal.std(axis=0, keepdims=True) < 1e-6, 1.0, X_cal.std(axis=0, keepdims=True)).astype(np.float32)
        X_cal_norm = ((X_cal - mu_x) / std_x).astype(np.float32)

        with torch.no_grad():
            _, probs_cal = cal_m(torch.from_numpy(X_cal_norm).to(device))
            probs_cal = probs_cal.cpu().numpy() # (N_ot, 3)

        # Three-Tiered Selective Decision System
        # Tier 1 (Low uncertainty / High Native confidence): p_N >= 0.60 and P2 neighbor entropy < median
        # Tier 3 (Extreme uncertainty): p2_knn_entropy >= 90th percentile
        # Tier 2 (Multimodal Medium uncertainty): output calibrated Top-3 portfolio
        ent_q50 = np.quantile(p2_knn_entropy_ot, 0.50)
        ent_q90 = np.quantile(p2_knn_entropy_ot, 0.90)

        is_tier1 = (probs_cal[:, 0] >= 0.60) & (p2_knn_entropy_ot < ent_q50)
        is_tier3 = p2_knn_entropy_ot >= ent_q90
        is_tier2 = (~is_tier1) & (~is_tier3)

        # Evaluation of Each Tier
        # Tier 1: Single Top-1 output (Native Top-1)
        diff_n1 = np.linalg.norm(n1_ot - future_ot, axis=-1)
        good_n1 = (diff_n1.mean(axis=-1) <= 0.2) & (diff_n1[:, -1] <= 0.4)

        # Tier 2: Top-3 portfolio output (Native, Static, Learned-3)
        diff_top3 = np.linalg.norm(portfolios["C1_canonical_NSL3"] - future_ot[:, None], axis=-1)
        good_top3 = ((diff_top3.mean(axis=-1) <= 0.2) & (diff_top3[:, :, -1] <= 0.4)).any(axis=-1)

        tier1_acc = float(good_n1[is_tier1].mean() * 100) if is_tier1.sum() > 0 else 0.0
        tier2_cov = float(good_top3[is_tier2].mean() * 100) if is_tier2.sum() > 0 else 0.0
        tier3_n1_acc = float(good_n1[is_tier3].mean() * 100) if is_tier3.sum() > 0 else 0.0

        # System effective success: Tier 1 correct OR Tier 2 covered
        system_success_count = good_n1[is_tier1].sum() + good_top3[is_tier2].sum()
        retained_count = is_tier1.sum() + is_tier2.sum()
        system_retained_acc = float(system_success_count / max(retained_count, 1) * 100)

        expected_outputs = float((is_tier1.sum() * 1.0 + is_tier2.sum() * 3.0 + is_tier3.sum() * 0.0) / n_ot)

        m3_seed_eval = {
            "tier_fractions_pct": {
                "tier1_single_top1": float(is_tier1.mean() * 100),
                "tier2_top3_portfolio": float(is_tier2.mean() * 100),
                "tier3_abstained_warning": float(is_tier3.mean() * 100)
            },
            "tier_performance_pct": {
                "tier1_top1_accuracy": tier1_acc,
                "tier2_top3_coverage": tier2_cov,
                "tier3_baseline_accuracy_abstained": tier3_n1_acc
            },
            "system_metrics": {
                "system_retained_accuracy_pct": system_retained_acc,
                "expected_hypotheses_count": expected_outputs
            }
        }
        m3_results[p_id]["seeds"][f"seed_{s}"] = m3_seed_eval
        print(f"  [M3 Interface] Tier 1 ({is_tier1.mean()*100:.1f}%): Acc={tier1_acc:.2f}% | Tier 2 ({is_tier2.mean()*100:.1f}%): Cov={tier2_cov:.2f}% | Tier 3 Abstain ({is_tier3.mean()*100:.1f}%) | Retained Acc={system_retained_acc:.2f}% (Exp Outputs: {expected_outputs:.2f})", flush=True)

# Macro Summary
macro_m2 = {}
for c_name in ["C1_canonical_NSL3", "C2_dual_learned_NL3L0", "C3_all_learned_NL3L1", "C4_pure_native_N1N2N3"]:
    macro_m2[c_name] = {
        "overall_coverage_pct": float(np.mean([[m2_results[p["id"]]["seeds"][f"seed_{s}"][c_name]["overall_coverage_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "change_coverage_pct": float(np.mean([[m2_results[p["id"]]["seeds"][f"seed_{s}"][c_name]["change_coverage_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "min_ade_m": float(np.mean([[m2_results[p["id"]]["seeds"][f"seed_{s}"][c_name]["min_ade_m"] for s in SEEDS] for p in PROTOCOLS])),
        "mean_pairwise_dist_m": float(np.mean([[m2_results[p["id"]]["seeds"][f"seed_{s}"][c_name]["mean_pairwise_dist_m"] for s in SEEDS] for p in PROTOCOLS]))
    }

macro_m3 = {
    "tier1_fraction_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["tier_fractions_pct"]["tier1_single_top1"] for s in SEEDS] for p in PROTOCOLS])),
    "tier2_fraction_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["tier_fractions_pct"]["tier2_top3_portfolio"] for s in SEEDS] for p in PROTOCOLS])),
    "tier3_fraction_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["tier_fractions_pct"]["tier3_abstained_warning"] for s in SEEDS] for p in PROTOCOLS])),
    "tier1_acc_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["tier_performance_pct"]["tier1_top1_accuracy"] for s in SEEDS] for p in PROTOCOLS])),
    "tier2_cov_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["tier_performance_pct"]["tier2_top3_coverage"] for s in SEEDS] for p in PROTOCOLS])),
    "system_retained_accuracy_pct": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["system_metrics"]["system_retained_accuracy_pct"] for s in SEEDS] for p in PROTOCOLS])),
    "expected_hypotheses_count": float(np.mean([[m3_results[p["id"]]["seeds"][f"seed_{s}"]["system_metrics"]["expected_hypotheses_count"] for s in SEEDS] for p in PROTOCOLS]))
}

print("\n--- MACRO PORTFOLIO COMPARISON (M2) ---")
for c_name, st in macro_m2.items():
    print(f"  {c_name:<25}: Overall={st['overall_coverage_pct']:.2f}%, Change={st['change_coverage_pct']:.2f}%, minADE={st['min_ade_m']:.3f}m, Div={st['mean_pairwise_dist_m']:.3f}m")

print("\n--- MACRO SELECTIVE INTERFACE (M3) ---")
print(f"  Tier 1 (Single Top-1):     {macro_m3['tier1_fraction_pct']:.1f}% of windows | Accuracy = {macro_m3['tier1_acc_pct']:.2f}%")
print(f"  Tier 2 (Top-3 Portfolio):  {macro_m3['tier2_fraction_pct']:.1f}% of windows | Coverage = {macro_m3['tier2_cov_pct']:.2f}%")
print(f"  Tier 3 (Abstain Warning):  {macro_m3['tier3_fraction_pct']:.1f}% of windows")
print(f"  Overall Retained Accuracy: {macro_m3['system_retained_accuracy_pct']:.2f}% | Expected Hypotheses = {macro_m3['expected_hypotheses_count']:.2f}")

# Save All Results
with open(OUT_M2_DIR / "PORTFOLIO_COMPOSITION_AUDIT.json", "w") as f:
    json.dump({"results": m2_results, "macro": macro_m2}, f, indent=2)

with open(OUT_M2_DIR / "RESULT.json", "w") as f:
    json.dump({"phase": "M2_PORTFOLIO_COMPOSITION", "macro_summary": macro_m2}, f, indent=2)

with open(OUT_M3_DIR / "SELECTIVE_INTERFACE_AUDIT.json", "w") as f:
    json.dump({"results": m3_results, "macro": macro_m3}, f, indent=2)

with open(OUT_M3_DIR / "RESULT.json", "w") as f:
    json.dump({"phase": "M3_SELECTIVE_INTERFACE", "macro_summary": macro_m3}, f, indent=2)

print("\nPhases M2 and M3 completed successfully!", flush=True)
