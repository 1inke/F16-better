import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
CONFIRM_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
R2_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R2_candidate_aware_scorer"
OUT_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R4_uncertainty_gated_selection"
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
print(f"Executing Stage R4 Uncertainty-Gated Selection on: {device}", flush=True)

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

r4a_results = {}
r4b_results = {}
r4c_results = {}
r4d_results = {}

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Running Stage R4 on Protocol: {p_name} ({p_id})", flush=True)
    print(f"=======================================================", flush=True)

    splits = np.load(p_cfg["split_file"])
    tr_idx = splits["grouped_train_index"]
    ot_idx = splits["outer_eval_index"]
    n_tr, n_ot = len(tr_idx), len(ot_idx)
    outer_sids = all_sample_ids[ot_idx]
    groups_ot = train_cache["change_group"][ot_idx]
    groups_tr = train_cache["change_group"][tr_idx]
    ce_ot = np.array([ce_map.get(int(sid), False) for sid in outer_sids], dtype=bool)

    SUBGROUPS = {
        "overall": np.ones(n_ot, dtype=bool),
        "stable": groups_ot == 0,
        "transition": groups_ot == 1,
        "change": groups_ot == 2,
        "clear_event": ce_ot
    }

    # P2 representation normalized strictly on train split
    tr_p2 = feat_p2[tr_idx]
    ot_p2 = feat_p2[ot_idx]
    mu_p2 = tr_p2.mean(0, keepdims=True)
    std_p2 = np.where(tr_p2.std(0, keepdims=True) < 1e-6, 1.0, tr_p2.std(0, keepdims=True)).astype(np.float32)
    tr_p2_norm = ((tr_p2 - mu_p2) / std_p2).astype(np.float32)
    ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

    future_tr = lab_train["future"][tr_idx].astype(np.float32)
    future_ot = lab_train["future"][ot_idx].astype(np.float32)
    past_tr = raw_focal_past[tr_idx]
    past_ot = raw_focal_past[ot_idx]

    # Load isolated Intent Classifier
    clf_saved = torch.load(p_cfg["intent_clf_path"], map_location=device, weights_only=False)
    clf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
    clf.load_state_dict(clf_saved["state_dict"])
    clf.eval()

    with torch.no_grad():
        tr_probs = torch.softmax(clf(torch.from_numpy(((raw_clf_obs[tr_idx] - clf_saved["mean"]) / clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()
        ot_probs = torch.softmax(clf(torch.from_numpy(((raw_clf_obs[ot_idx] - clf_saved["mean"]) / clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()

    topc_tr = np.argsort(-tr_probs[:, non_cont_classes], axis=1)[:, :4]
    topc_ot = np.argsort(-ot_probs[:, non_cont_classes], axis=1)[:, :4]

    s_tr = np.zeros((n_tr, 1, 12, 2), dtype=np.float32)
    for i in range(n_tr):
        c_name = class_names_map[non_cont_classes[topc_tr[i, 0]]]
        s_tr[i, 0] = generate_motion_program(past_tr[i:i+1], default_params(c_name, 1))[0]

    s_ot = np.zeros((n_ot, 1, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        c_name = class_names_map[non_cont_classes[topc_ot[i, 0]]]
        s_ot[i, 0] = generate_motion_program(past_ot[i:i+1], default_params(c_name, 1))[0]

    nat_tr = train_cache["candidate"][tr_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)

    nat_log_tr = train_cache["logits"][tr_idx].astype(np.float32)
    nat_log_ot = train_cache["logits"][ot_idx].astype(np.float32)

    prior_log_tr = np.repeat((np.median(nat_log_tr, axis=1, keepdims=True) - nat_log_tr.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
    prior_log_ot = np.repeat((np.median(nat_log_ot, axis=1, keepdims=True) - nat_log_ot.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)

    k24_log_tr = np.concatenate([nat_log_tr, prior_log_tr], axis=1)
    k24_log_ot = np.concatenate([nat_log_ot, prior_log_ot], axis=1)

    # Compute kNN neighbor mode entropy (k=20)
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

    # Also compute on train split for gate calibration (using top 21-1 to avoid self-match)
    # Subsample 4000 train samples for fast gate calibration
    SUB_N = min(4000, n_tr)
    sub_tr_p2_t = tr_p2_t[:SUB_N]
    dist_tr_mat = torch.cdist(sub_tr_p2_t, tr_p2_t)
    _, knn21_tr_idx = torch.topk(dist_tr_mat, 21, largest=False)
    knn20_tr_np = knn21_tr_idx[:, 1:].cpu().numpy()
    p2_knn_entropy_tr = np.zeros(SUB_N, dtype=np.float32)
    for i in range(SUB_N):
        counts = np.bincount(kinds_tr[knn20_tr_np[i]], minlength=5)
        probs = counts / 20.0
        probs = probs[probs > 0]
        p2_knn_entropy_tr[i] = -np.sum(probs * np.log2(probs))

    r4a_results[p_id] = {"seeds": {}, "macro": {}}
    r4b_results[p_id] = {"seeds": {}, "macro": {}}
    r4c_results[p_id] = {"seeds": {}, "macro": {}}
    r4d_results[p_id] = {"seeds": {}, "macro": {}}

    for s in SEEDS:
        print(f"\n--- Protocol {p_id} | Seed {s} ---", flush=True)
        # Load Bo4 Generator
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
        bo4_tr_sub = gen_bo4(tr_p2_norm[:SUB_N], past_tr[:SUB_N], SUB_N)

        k24_ot = np.concatenate([nat_ot, s_ot, bo4_ot[:, [3, 0, 1]]], axis=1) # (N_ot, 24, 12, 2)
        k24_tr_sub = np.concatenate([nat_tr[:SUB_N], s_tr[:SUB_N], bo4_tr_sub[:, [3, 0, 1]]], axis=1)

        # Load R2 Scorer Checkpoint
        r2_ckpt = R2_DIR / "models" / p_id / f"seed_{s}" / "scorer.pth"
        r2_m = CandidateAwareScorer().to(device)
        r2_m.load_state_dict(torch.load(r2_ckpt, map_location=device))
        r2_m.eval()

        cand_feat_ot, set_feat_ot = extract_features(k24_ot)
        cand_feat_tr_sub, set_feat_tr_sub = extract_features(k24_tr_sub)

        # Normalization using training split
        cand_feat_tr_all, set_feat_tr_all = extract_features(k24_tr_sub) # representative
        mu_cf = cand_feat_tr_all.mean(axis=(0, 1), keepdims=True)
        std_cf = np.where(cand_feat_tr_all.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, cand_feat_tr_all.std(axis=(0, 1), keepdims=True)).astype(np.float32)
        cand_feat_ot_norm = ((cand_feat_ot - mu_cf) / std_cf).astype(np.float32)
        cand_feat_tr_norm = ((cand_feat_tr_sub - mu_cf) / std_cf).astype(np.float32)

        mu_sf = set_feat_tr_all.mean(axis=0, keepdims=True)
        std_sf = np.where(set_feat_tr_all.std(axis=0, keepdims=True) < 1e-6, 1.0, set_feat_tr_all.std(axis=0, keepdims=True)).astype(np.float32)
        set_feat_ot_norm = ((set_feat_ot - mu_sf) / std_sf).astype(np.float32)
        set_feat_tr_norm = ((set_feat_tr_sub - mu_sf) / std_sf).astype(np.float32)

        with torch.no_grad():
            scores_ot, deltas_ot = r2_m(
                torch.from_numpy(ot_p2_norm).to(device),
                torch.from_numpy(cand_feat_ot_norm).to(device),
                torch.from_numpy(set_feat_ot_norm).to(device),
                torch.from_numpy(k24_log_ot).to(device)
            )
            scores_ot = scores_ot.cpu().numpy()

            scores_tr, deltas_tr = r2_m(
                torch.from_numpy(tr_p2_norm[:SUB_N]).to(device),
                torch.from_numpy(cand_feat_tr_norm).to(device),
                torch.from_numpy(set_feat_tr_norm).to(device),
                torch.from_numpy(k24_log_tr[:SUB_N]).to(device)
            )
            scores_tr = scores_tr.cpu().numpy()

        # Selection under Native Scorer
        native_sel_ot = k24_log_ot.argmax(axis=1)
        row_ot = np.arange(n_ot)
        trajs_nat_ot = k24_ot[row_ot, native_sel_ot]
        diff_nat_ot = np.linalg.norm(trajs_nat_ot - future_ot, axis=-1)
        ade_nat_ot = diff_nat_ot.mean(axis=-1)
        fde_nat_ot = diff_nat_ot[:, -1]
        good_nat_ot = (ade_nat_ot <= 0.2) & (fde_nat_ot <= 0.4)
        j_nat_ot = np.maximum(ade_nat_ot / 0.2, fde_nat_ot / 0.4)

        # Selection under R2 Scorer
        r2_sel_ot = scores_ot.argmax(axis=1)
        trajs_r2_ot = k24_ot[row_ot, r2_sel_ot]
        diff_r2_ot = np.linalg.norm(trajs_r2_ot - future_ot, axis=-1)
        ade_r2_ot = diff_r2_ot.mean(axis=-1)
        fde_r2_ot = diff_r2_ot[:, -1]
        good_r2_ot = (ade_r2_ot <= 0.2) & (fde_r2_ot <= 0.4)
        j_r2_ot = np.maximum(ade_r2_ot / 0.2, fde_r2_ot / 0.4)

        # Oracle metrics on outer set
        err_ot_all = compute_errors(k24_ot, future_ot)
        oracle_j_ot = err_ot_all["J"]
        oracle_good_ot = err_ot_all["good"]
        oracle_cov_ot = err_ot_all["coverage"]

        # Targets
        # Y_override = 1 if R2 achieved strictly smaller regret J than Native
        y_override_ot = (j_r2_ot < j_nat_ot).astype(np.float32)

        # Training set metrics for gate training
        native_sel_tr = k24_log_tr[:SUB_N].argmax(axis=1)
        row_tr = np.arange(SUB_N)
        trajs_nat_tr = k24_tr_sub[row_tr, native_sel_tr]
        diff_nat_tr = np.linalg.norm(trajs_nat_tr - future_tr[:SUB_N], axis=-1)
        j_nat_tr = np.maximum(diff_nat_tr.mean(axis=-1) / 0.2, diff_nat_tr[:, -1] / 0.4)

        r2_sel_tr = scores_tr.argmax(axis=1)
        trajs_r2_tr = k24_tr_sub[row_tr, r2_sel_tr]
        diff_r2_tr = np.linalg.norm(trajs_r2_tr - future_tr[:SUB_N], axis=-1)
        j_r2_tr = np.maximum(diff_r2_tr.mean(axis=-1) / 0.2, diff_r2_tr[:, -1] / 0.4)

        y_override_tr = (j_r2_tr < j_nat_tr).astype(np.float32)

        # Build Deployment-Time GT-Free Uncertainty Features
        def extract_gate_features(nat_logs, r2_scs, probs_clf, p2_knn_ent, cands):
            # 1. Native margin (top1 - top2 logit among native 20)
            sorted_nat = np.sort(nat_logs[:, :20], axis=1)[:, ::-1]
            margin_nat = sorted_nat[:, 0] - sorted_nat[:, 1]
            # 2. Native score entropy
            p_nat = np.exp(nat_logs[:, :20] - np.max(nat_logs[:, :20], axis=1, keepdims=True))
            p_nat = p_nat / np.sum(p_nat, axis=1, keepdims=True)
            ent_nat = -np.sum(p_nat * np.log(np.clip(p_nat, 1e-8, 1.0)), axis=1)
            # 3. R2 score entropy
            p_r2 = np.exp(r2_scs - np.max(r2_scs, axis=1, keepdims=True))
            p_r2 = p_r2 / np.sum(p_r2, axis=1, keepdims=True)
            ent_r2 = -np.sum(p_r2 * np.log(np.clip(p_r2, 1e-8, 1.0)), axis=1)
            # 4. Disagreement between Native Top-1 and R2 Top-1
            disagree_top1 = (nat_logs.argmax(axis=1) != r2_scs.argmax(axis=1)).astype(np.float32)
            # 5. R2 chose proposal
            r2_chose_prop = (r2_scs.argmax(axis=1) >= 20).astype(np.float32)
            # 6. Regime change probability (sum of non-cont classes)
            p_change = probs_clf[:, non_cont_classes].sum(axis=1)
            # 7. Candidate endpoint spread (std)
            cand_end_std = cands[:, :, -1, :].std(axis=1).mean(axis=-1)
            # 8. Learned proposal score advantage
            learned_adv = np.max(r2_scs[:, 21:], axis=1) - np.max(r2_scs[:, :20], axis=1)

            feats = np.stack([
                margin_nat, ent_nat, ent_r2, disagree_top1, r2_chose_prop,
                p_change, p2_knn_ent, cand_end_std, learned_adv
            ], axis=1)
            return feats

        X_ot = extract_gate_features(k24_log_ot, scores_ot, ot_probs, p2_knn_entropy_ot, k24_ot)
        X_tr = extract_gate_features(k24_log_tr[:SUB_N], scores_tr, tr_probs[:SUB_N], p2_knn_entropy_tr, k24_tr_sub)

        feat_names = [
            "margin_native", "entropy_native", "entropy_r2", "disagree_top1",
            "r2_chose_proposal", "prob_non_cont", "p2_knn_entropy", "cand_endpoint_std",
            "learned_advantage"
        ]

        # -------------------------------------------------------------
        # R4A: Override Predictability Audit
        # -------------------------------------------------------------
        mu_X = X_tr.mean(axis=0, keepdims=True)
        std_X = np.where(X_tr.std(axis=0, keepdims=True) < 1e-6, 1.0, X_tr.std(axis=0, keepdims=True))
        X_tr_norm = (X_tr - mu_X) / std_X
        X_ot_norm = (X_ot - mu_X) / std_X

        # Individual feature AUROCs
        indiv_auroc = {}
        for f_idx, f_name in enumerate(feat_names):
            try:
                auc_val = float(roc_auc_score(y_override_ot, X_ot[:, f_idx]))
                if auc_val < 0.5:
                    auc_val = 1.0 - auc_val
                indiv_auroc[f_name] = auc_val
            except Exception:
                indiv_auroc[f_name] = 0.5

        # Fit Logistic Gate on Training Set
        clf_gate = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_gate.fit(X_tr_norm, y_override_tr)
        gate_probs_ot = clf_gate.predict_proba(X_ot_norm)[:, 1] # Uncertainty Score u(h)

        composite_auroc = float(roc_auc_score(y_override_ot, gate_probs_ot))

        r4a_results[p_id]["seeds"][f"seed_{s}"] = {
            "individual_aurocs": indiv_auroc,
            "composite_gate_auroc": composite_auroc,
            "y_override_fraction_pct": float(y_override_ot.mean() * 100),
            "gate_weights": {feat_names[i]: float(clf_gate.coef_[0, i]) for i in range(len(feat_names))}
        }

        print(f"  [R4A Predictability] Composite Gate AUROC = {composite_auroc:.3f} (Override fraction: {y_override_ot.mean()*100:.1f}%)", flush=True)

        # -------------------------------------------------------------
        # R4B: Native / R2 Selective Routing Evaluation
        # -------------------------------------------------------------
        # Sort outer samples by gate uncertainty score u(h) descending
        sort_order = np.argsort(-gate_probs_ot)

        FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]
        routing_sweep = []

        nat_good_rate = float(good_nat_ot.mean() * 100)
        nat_ade_mean = float(ade_nat_ot.mean())
        nat_fde_mean = float(fde_nat_ot.mean())
        ch_msk = SUBGROUPS["change"]
        nat_ch_good_rate = float(good_nat_ot[ch_msk].mean() * 100)
        nat_ch_ade = float(ade_nat_ot[ch_msk].mean())
        nat_regret = j_nat_ot - oracle_j_ot
        nat_severe_regret = float((nat_regret > 0.2).mean() * 100)

        for frac in FRACTIONS:
            num_override = int(round(frac * n_ot))
            override_indices = sort_order[:num_override]

            # Gated selection
            gated_trajs = trajs_nat_ot.copy()
            if num_override > 0:
                gated_trajs[override_indices] = trajs_r2_ot[override_indices]

            diff_gated = np.linalg.norm(gated_trajs - future_ot, axis=-1)
            ade_gated = diff_gated.mean(axis=-1)
            fde_gated = diff_gated[:, -1]
            good_gated = (ade_gated <= 0.2) & (fde_gated <= 0.4)
            j_gated = np.maximum(ade_gated / 0.2, fde_gated / 0.4)

            # Metrics
            g_good_rate = float(good_gated.mean() * 100)
            g_ade_mean = float(ade_gated.mean())
            g_fde_mean = float(fde_gated.mean())

            top1_gain_pp = g_good_rate - nat_good_rate
            # ADE harm: positive means ADE increased/worsened
            ade_harm_pct = float((g_ade_mean - nat_ade_mean) / nat_ade_mean * 100)
            fde_harm_pct = float((g_fde_mean - nat_fde_mean) / nat_fde_mean * 100)

            ch_good_rate = float(good_gated[ch_msk].mean() * 100)
            ch_gain_pp = ch_good_rate - nat_ch_good_rate

            g_regret = j_gated - oracle_j_ot
            g_severe_regret = float((g_regret > 0.2).mean() * 100)
            severe_regret_drop_pp = float(nat_severe_regret - g_severe_regret)

            # Gate criteria check
            # Pass if Change Gain > 0 AND ADE harm < 1.0% AND Severe regret drop > 0
            gate_pass = (ch_gain_pp > 0.0) and (ade_harm_pct < 1.0) and (severe_regret_drop_pp > 0.0)

            routing_sweep.append({
                "override_fraction": frac,
                "n_overridden": num_override,
                "overall_top1_good_rate_pct": g_good_rate,
                "overall_top1_gain_pp": top1_gain_pp,
                "overall_selected_ade": g_ade_mean,
                "overall_selected_fde": g_fde_mean,
                "overall_ade_harm_pct": ade_harm_pct,
                "overall_fde_harm_pct": fde_harm_pct,
                "change_top1_good_rate_pct": ch_good_rate,
                "change_top1_gain_pp": ch_gain_pp,
                "severe_regret_pct": g_severe_regret,
                "severe_regret_drop_pp": severe_regret_drop_pp,
                "gate_passed_at_this_point": gate_pass
            })

        r4b_results[p_id]["seeds"][f"seed_{s}"] = routing_sweep

        # Check optimal fraction
        passing_points = [p for p in routing_sweep if p["gate_passed_at_this_point"] and p["override_fraction"] > 0]
        if passing_points:
            best_pt = max(passing_points, key=lambda x: x["change_top1_gain_pp"])
            print(f"  [R4B Routing] Gate PASSED at fraction {best_pt['override_fraction']*100:.0f}%: Change Gain={best_pt['change_top1_gain_pp']:+.2f} pp, ADE Harm={best_pt['overall_ade_harm_pct']:+.2f}%, Top1 Gain={best_pt['overall_top1_gain_pp']:+.2f} pp", flush=True)
        else:
            # Show 10% point as representative
            pt10 = routing_sweep[2]
            print(f"  [R4B Routing] Gate NOT passed at frac 10%: Change Gain={pt10['change_top1_gain_pp']:+.2f} pp, ADE Harm={pt10['overall_ade_harm_pct']:+.2f}%, Top1 Gain={pt10['overall_top1_gain_pp']:+.2f} pp", flush=True)

        # -------------------------------------------------------------
        # R4C: Selective Prediction (Full Coverage-Risk Curve)
        # -------------------------------------------------------------
        # Uncertainty ordering: rank by gate_probs_ot descending (most uncertain rejected first)
        REJECT_FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        risk_curve = []

        coverages = []
        risks = []
        ch_coverages = []
        ch_risks = []

        for r_frac in REJECT_FRACTIONS:
            num_reject = int(round(r_frac * n_ot))
            retained_idx = sort_order[num_reject:] if num_reject > 0 else np.arange(n_ot)
            coverage = len(retained_idx) / n_ot

            # Selective risk = error rate (1 - Good Rate)
            ret_good = good_nat_ot[retained_idx]
            ret_ade = ade_nat_ot[retained_idx]
            sel_risk = float((1.0 - ret_good.mean()) * 100)
            ret_good_rate = float(ret_good.mean() * 100)
            ret_ade_mean = float(ret_ade.mean())

            # Change specific
            ret_ch_msk = ch_msk[retained_idx]
            if ret_ch_msk.sum() > 0:
                ret_ch_good = good_nat_ot[retained_idx][ret_ch_msk]
                ch_risk = float((1.0 - ret_ch_good.mean()) * 100)
                ch_good_rate = float(ret_ch_good.mean() * 100)
            else:
                ch_risk = 0.0
                ch_good_rate = 0.0

            coverages.append(coverage)
            risks.append(sel_risk / 100.0)
            ch_risks.append(ch_risk / 100.0)

            risk_curve.append({
                "rejected_fraction": r_frac,
                "coverage": coverage,
                "selective_risk_pct": sel_risk,
                "retained_good_rate_pct": ret_good_rate,
                "retained_ade": ret_ade_mean,
                "change_good_rate_pct": ch_good_rate,
                "change_risk_pct": ch_risk
            })

        # Calculate AURC (Area Under Risk-Coverage Curve via trapezoid rule)
        # Sort by coverage ascending
        cov_arr = np.array(coverages)[::-1]
        risk_arr = np.array(risks)[::-1]
        ch_risk_arr = np.array(ch_risks)[::-1]

        aurc = float(np.trapz(risk_arr, cov_arr))
        ch_aurc = float(np.trapz(ch_risk_arr, cov_arr))

        r4c_results[p_id]["seeds"][f"seed_{s}"] = {
            "curve": risk_curve,
            "aurc": aurc,
            "change_aurc": ch_aurc
        }

        pt20_c = [p for p in risk_curve if abs(p["rejected_fraction"] - 0.20) < 1e-4][0]
        print(f"  [R4C Selective Prediction] AURC={aurc:.4f} | At 20% Abstention: Retained Good={pt20_c['retained_good_rate_pct']:.2f}% (+{pt20_c['retained_good_rate_pct'] - nat_good_rate:+.2f} pp)", flush=True)

        # -------------------------------------------------------------
        # R4D: Top-3 Multi-Hypothesis Portfolio Evaluation
        # -------------------------------------------------------------
        # Portfolio: Native Top-1, Static Top-1 (idx 20), Learned Slot 3 (idx 21)
        top3_indices = np.stack([native_sel_ot, np.full(n_ot, 20), np.full(n_ot, 21)], axis=1) # (N_ot, 3)
        top3_trajs = np.zeros((n_ot, 3, 12, 2), dtype=np.float32)
        for i in range(n_ot):
            top3_trajs[i] = k24_ot[i, top3_indices[i]]

        # Compute set coverage: at least 1 candidate good
        top3_good_any = oracle_good_ot[row_ot, native_sel_ot] | oracle_good_ot[:, 20] | oracle_good_ot[:, 21]
        diff_top3 = np.linalg.norm(top3_trajs - future_ot[:, None], axis=-1)
        ade_top3_oracle = diff_top3.mean(axis=-1).min(axis=-1)
        fde_top3_oracle = diff_top3[:, :, -1].min(axis=-1)

        # Effective set size: count distinct endpoints (distance > 0.1m)
        dist_01 = np.linalg.norm(top3_trajs[:, 0, -1] - top3_trajs[:, 1, -1], axis=-1) > 0.1
        dist_02 = np.linalg.norm(top3_trajs[:, 0, -1] - top3_trajs[:, 2, -1], axis=-1) > 0.1
        dist_12 = np.linalg.norm(top3_trajs[:, 1, -1] - top3_trajs[:, 2, -1], axis=-1) > 0.1
        # Distinct endpoints count
        distinct_count = np.ones(n_ot, dtype=np.float32)
        distinct_count += dist_01.astype(np.float32)
        distinct_count += (dist_02 & dist_12).astype(np.float32)

        top3_subgroups = {}
        for g_name, g_mask in SUBGROUPS.items():
            if g_mask.sum() == 0: continue
            top3_subgroups[g_name] = {
                "set_good_rate_pct": float(top3_good_any[g_mask].mean() * 100),
                "gain_pp_vs_native_top1": float((top3_good_any[g_mask].mean() - good_nat_ot[g_mask].mean()) * 100),
                "oracle_min_ade": float(ade_top3_oracle[g_mask].mean()),
                "oracle_min_fde": float(fde_top3_oracle[g_mask].mean())
            }

        r4d_results[p_id]["seeds"][f"seed_{s}"] = {
            "subgroups": top3_subgroups,
            "expected_effective_set_size": float(distinct_count.mean())
        }
        print(f"  [R4D Top-3 Set] Overall Set Good={top3_subgroups['overall']['set_good_rate_pct']:.2f}% (+{top3_subgroups['overall']['gain_pp_vs_native_top1']:+.2f} pp), Change Set Good={top3_subgroups['change']['set_good_rate_pct']:.2f}% (+{top3_subgroups['change']['gain_pp_vs_native_top1']:+.2f} pp)", flush=True)

# -------------------------------------------------------------
# Macro Aggregation and Synthesis
# -------------------------------------------------------------
print(f"\n=======================================================", flush=True)
print(f"Synthesizing Protocol Macro Results for Stage R4", flush=True)
print(f"=======================================================", flush=True)

macro_r4a = {
    "composite_auroc": float(np.mean([[r4a_results[p]["seeds"][f"seed_{s}"]["composite_gate_auroc"] for s in SEEDS] for p in [p["id"] for p in PROTOCOLS]]))
}

# R4B Macro Sweep
FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00]
macro_r4b_sweep = []
for f_idx, frac in enumerate(FRACTIONS):
    pt_gains = []
    pt_ade_harms = []
    pt_ch_gains = []
    pt_sev_drops = []
    protocol_pass_count = 0

    for p in PROTOCOLS:
        p_id = p["id"]
        seed_pts = [r4b_results[p_id]["seeds"][f"seed_{s}"][f_idx] for s in SEEDS]
        mean_ch_gain = np.mean([pt["change_top1_gain_pp"] for pt in seed_pts])
        mean_ade_harm = np.mean([pt["overall_ade_harm_pct"] for pt in seed_pts])
        mean_sev_drop = np.mean([pt["severe_regret_drop_pp"] for pt in seed_pts])

        pt_gains.append(np.mean([pt["overall_top1_gain_pp"] for pt in seed_pts]))
        pt_ade_harms.append(mean_ade_harm)
        pt_ch_gains.append(mean_ch_gain)
        pt_sev_drops.append(mean_sev_drop)

        # Check if 2/3 seeds passed for this protocol
        seed_passes = sum(1 for pt in seed_pts if pt["gate_passed_at_this_point"])
        if seed_passes >= 2:
            protocol_pass_count += 1

    macro_r4b_sweep.append({
        "override_fraction": frac,
        "macro_top1_gain_pp": float(np.mean(pt_gains)),
        "macro_ade_harm_pct": float(np.mean(pt_ade_harms)),
        "macro_change_gain_pp": float(np.mean(pt_ch_gains)),
        "macro_severe_regret_drop_pp": float(np.mean(pt_sev_drops)),
        "protocols_passing_gate_2of3_seeds": f"{protocol_pass_count}/3",
        "gate_passed": protocol_pass_count >= 2 and frac > 0
    })

# R4C Macro AURC
macro_r4c = {
    "aurc": float(np.mean([[r4c_results[p["id"]]["seeds"][f"seed_{s}"]["aurc"] for s in SEEDS] for p in PROTOCOLS])),
    "change_aurc": float(np.mean([[r4c_results[p["id"]]["seeds"][f"seed_{s}"]["change_aurc"] for s in SEEDS] for p in PROTOCOLS]))
}

# R4D Macro Top-3
macro_r4d = {
    "overall_gain_pp": float(np.mean([[r4d_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["gain_pp_vs_native_top1"] for s in SEEDS] for p in PROTOCOLS])),
    "change_gain_pp": float(np.mean([[r4d_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["gain_pp_vs_native_top1"] for s in SEEDS] for p in PROTOCOLS])),
    "overall_set_good_rate": float(np.mean([[r4d_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["set_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
    "change_set_good_rate": float(np.mean([[r4d_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["set_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
    "expected_set_size": float(np.mean([[r4d_results[p["id"]]["seeds"][f"seed_{s}"]["expected_effective_set_size"] for s in SEEDS] for p in PROTOCOLS]))
}

# Print R4B Sweep
print("\nR4B Selective Routing Macro Sweep:")
print(f"{'Override %':>10} | {'Change Gain':>12} | {'ADE Harm %':>12} | {'Top1 Gain':>12} | {'Sev Reg Drop':>12} | {'Gate (2/3)':>10}")
print("-" * 75)
for pt in macro_r4b_sweep:
    print(f"{pt['override_fraction']*100:9.0f}% | {pt['macro_change_gain_pp']:+11.2f} pp | {pt['macro_ade_harm_pct']:+11.2f}% | {pt['macro_top1_gain_pp']:+11.2f} pp | {pt['macro_severe_regret_drop_pp']:+11.2f} pp | {pt['protocols_passing_gate_2of3_seeds']:>10}")

gate_succeeded = any(pt["gate_passed"] for pt in macro_r4b_sweep)
if gate_succeeded:
    final_verdict = "UNCERTAINTY_GATED_SCORER_ACCEPTED"
else:
    final_verdict = "PAST_ONLY_TOP1_LIMITED"

print(f"\nFinal Scientific Verdict for Stage R4: {final_verdict}", flush=True)

# Save All Results
with open(OUT_DIR / "R4A_override_predictability.json", "w") as f:
    json.dump({"results": r4a_results, "macro": macro_r4a}, f, indent=2)

with open(OUT_DIR / "R4B_selective_routing.json", "w") as f:
    json.dump({"results": r4b_results, "macro_sweep": macro_r4b_sweep, "gate_succeeded": gate_succeeded}, f, indent=2)

with open(OUT_DIR / "R4C_selective_prediction.json", "w") as f:
    json.dump({"results": r4c_results, "macro": macro_r4c}, f, indent=2)

with open(OUT_DIR / "R4D_top3_multihypothesis.json", "w") as f:
    json.dump({"results": r4d_results, "macro": macro_r4d}, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "stage": "R4_UNCERTAINTY_GATED_SELECTION",
        "final_verdict": final_verdict,
        "gate_succeeded": gate_succeeded,
        "r4a_predictability_auroc": macro_r4a["composite_auroc"],
        "r4b_sweep": macro_r4b_sweep,
        "r4c_aurc": macro_r4c,
        "r4d_top3_summary": macro_r4d
    }, f, indent=2)

print("\nStage R4 completed and all audit artifacts saved successfully to:", OUT_DIR, flush=True)
