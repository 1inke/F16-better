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
R2_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R2_candidate_aware_scorer"
OUT_DIR = REPO / "CALIBRATED_TOP3_RESEARCH_20261007" / "M1_probability_calibration"
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
print(f"Executing Phase M1: Top-3 Probability Calibration on: {device}", flush=True)

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

# Lightweight Top-3 Probability Calibrator
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
m1_results = {}
checkpoints = {}

def compute_ece(probs, good_mask, num_bins=10):
    # probs: (N, 3), good_mask: (N, 3)
    N = len(probs)
    conf = np.max(probs, axis=1)
    pred_c = np.argmax(probs, axis=1)
    row_idx = np.arange(N)
    corr = good_mask[row_idx, pred_c].astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_stats = []

    for b in range(num_bins):
        in_bin = (conf > bin_edges[b]) & (conf <= bin_edges[b + 1])
        n_in_bin = in_bin.sum()
        if n_in_bin > 0:
            bin_acc = corr[in_bin].mean()
            bin_conf = conf[in_bin].mean()
            ece += (n_in_bin / N) * abs(bin_acc - bin_conf)
            bin_stats.append({
                "bin": b,
                "n_samples": int(n_in_bin),
                "acc": float(bin_acc),
                "conf": float(bin_conf),
                "diff": float(abs(bin_acc - bin_conf))
            })
    return float(ece), bin_stats

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Training Top-3 Probability Calibrator on: {p_name} ({p_id})", flush=True)
    print(f"=======================================================", flush=True)

    splits = np.load(p_cfg["split_file"])
    tr_idx = splits["grouped_train_index"]
    ot_idx = splits["outer_eval_index"]
    n_tr, n_ot = len(tr_idx), len(ot_idx)
    outer_sids = all_sample_ids[ot_idx]
    groups_ot = train_cache["change_group"][ot_idx]
    groups_tr = train_cache["change_group"][tr_idx]

    SUBGROUPS = {
        "overall": np.ones(n_ot, dtype=bool),
        "stable": groups_ot == 0,
        "transition": groups_ot == 1,
        "change": groups_ot == 2
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

    # Subsample 5000 train samples for fast training of the calibrator
    N_SUB_TR = min(5000, n_tr)
    sub_tr_idx = np.arange(N_SUB_TR)

    s_tr_sub = np.zeros((N_SUB_TR, 12, 2), dtype=np.float32)
    for i in range(N_SUB_TR):
        c_name = class_names_map[non_cont_classes[topc_tr[i, 0]]]
        s_tr_sub[i] = generate_motion_program(past_tr[i:i+1], default_params(c_name, 1))[0]

    s_ot = np.zeros((n_ot, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        c_name = class_names_map[non_cont_classes[topc_ot[i, 0]]]
        s_ot[i] = generate_motion_program(past_ot[i:i+1], default_params(c_name, 1))[0]

    nat_tr_sub = train_cache["candidate"][tr_idx[:N_SUB_TR], :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)

    nat_log_tr_sub = train_cache["logits"][tr_idx[:N_SUB_TR]].astype(np.float32)
    nat_log_ot = train_cache["logits"][ot_idx].astype(np.float32)

    prior_log_tr_sub = np.repeat((np.median(nat_log_tr_sub, axis=1, keepdims=True) - nat_log_tr_sub.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
    prior_log_ot = np.repeat((np.median(nat_log_ot, axis=1, keepdims=True) - nat_log_ot.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)

    k24_log_tr_sub = np.concatenate([nat_log_tr_sub, prior_log_tr_sub], axis=1)
    k24_log_ot = np.concatenate([nat_log_ot, prior_log_ot], axis=1)

    # Native top1 selections
    n_sel_tr_sub = nat_log_tr_sub[:, :20].argmax(axis=1)
    n_sel_ot = nat_log_ot[:, :20].argmax(axis=1)

    n_traj_tr_sub = nat_tr_sub[np.arange(N_SUB_TR), n_sel_tr_sub]
    n_traj_ot = nat_ot[np.arange(n_ot), n_sel_ot]

    # Precompute kNN mode entropy (k=20)
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

    # Subsample train kNN
    sub_tr_p2_t = tr_p2_t[:N_SUB_TR]
    dist_tr_mat = torch.cdist(sub_tr_p2_t, tr_p2_t)
    _, knn21_tr_idx = torch.topk(dist_tr_mat, 21, largest=False)
    knn20_tr_np = knn21_tr_idx[:, 1:].cpu().numpy()
    p2_knn_entropy_tr_sub = np.zeros(N_SUB_TR, dtype=np.float32)
    for i in range(N_SUB_TR):
        counts = np.bincount(kinds_tr[knn20_tr_np[i]], minlength=5)
        probs = counts / 20.0
        probs = probs[probs > 0]
        p2_knn_entropy_tr_sub[i] = -np.sum(probs * np.log2(probs))

    m1_results[p_id] = {"seeds": {}, "macro": {}}

    for s in SEEDS:
        print(f"\n--- Protocol {p_id} | Seed {s} ---", flush=True)
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

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
        bo4_tr_sub = gen_bo4(tr_p2_norm[:N_SUB_TR], past_tr[:N_SUB_TR], N_SUB_TR)

        l3_ot = bo4_ot[:, 3]
        l3_tr_sub = bo4_tr_sub[:, 3]

        # S1L3 candidate sets
        k24_ot = np.concatenate([nat_ot, s_ot[:, None], bo4_ot[:, [3, 0, 1]]], axis=1)
        k24_tr_sub = np.concatenate([nat_tr_sub, s_tr_sub[:, None], bo4_tr_sub[:, [3, 0, 1]]], axis=1)

        # Load Stage R2 Scorer
        r2_ckpt = R2_DIR / "models" / p_id / f"seed_{s}" / "scorer.pth"
        r2_m = CandidateAwareScorer().to(device)
        r2_m.load_state_dict(torch.load(r2_ckpt, map_location=device))
        r2_m.eval()

        cand_feat_ot, set_feat_ot = extract_features(k24_ot)
        cand_feat_tr_sub, set_feat_tr_sub = extract_features(k24_tr_sub)

        mu_cf = cand_feat_tr_sub.mean(axis=(0, 1), keepdims=True)
        std_cf = np.where(cand_feat_tr_sub.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, cand_feat_tr_sub.std(axis=(0, 1), keepdims=True)).astype(np.float32)
        cand_feat_ot_norm = ((cand_feat_ot - mu_cf) / std_cf).astype(np.float32)
        cand_feat_tr_norm = ((cand_feat_tr_sub - mu_cf) / std_cf).astype(np.float32)

        mu_sf = set_feat_tr_sub.mean(axis=0, keepdims=True)
        std_sf = np.where(set_feat_tr_sub.std(axis=0, keepdims=True) < 1e-6, 1.0, set_feat_tr_sub.std(axis=0, keepdims=True)).astype(np.float32)
        set_feat_ot_norm = ((set_feat_ot - mu_sf) / std_sf).astype(np.float32)
        set_feat_tr_norm = ((set_feat_tr_sub - mu_sf) / std_sf).astype(np.float32)

        with torch.no_grad():
            r2_scores_ot, _ = r2_m(
                torch.from_numpy(ot_p2_norm).to(device),
                torch.from_numpy(cand_feat_ot_norm).to(device),
                torch.from_numpy(set_feat_ot_norm).to(device),
                torch.from_numpy(k24_log_ot).to(device)
            )
            r2_scores_ot = r2_scores_ot.cpu().numpy()

            r2_scores_tr_sub, _ = r2_m(
                torch.from_numpy(tr_p2_norm[:N_SUB_TR]).to(device),
                torch.from_numpy(cand_feat_tr_norm).to(device),
                torch.from_numpy(set_feat_tr_norm).to(device),
                torch.from_numpy(k24_log_tr_sub).to(device)
            )
            r2_scores_tr_sub = r2_scores_tr_sub.cpu().numpy()

        # Build Calibrator Features for the 3 Hypotheses:
        # Trajectories: [Native Top-1, Static Top-1, Learned-3]
        def build_calibrator_features(nat_logs, r2_scs, n_sel, probs_clf, p2_knn_ent, n_traj, s_traj, l_traj):
            N = len(nat_logs)
            row = np.arange(N)
            # Logit evidence for Native
            nat_top1_log = nat_logs[row, n_sel]
            sorted_nat = np.sort(nat_logs[:, :20], axis=1)[:, ::-1]
            nat_margin = sorted_nat[:, 0] - sorted_nat[:, 1]
            # R2 evidence for the 3 hypotheses
            r2_sc_n = r2_scs[row, n_sel]
            r2_sc_s = r2_scs[:, 20] # slot 20: Static
            r2_sc_l = r2_scs[:, 21] # slot 21: Learned-3
            # Disagreement between Native Top-1 and R2
            disagree = (n_sel != r2_scs.argmax(axis=1)).astype(np.float32)
            # Maneuver change probability
            p_change = probs_clf[:, non_cont_classes].sum(axis=1)
            # Pairwise distances
            d_ns = np.linalg.norm(n_traj - s_traj, axis=-1).mean(axis=-1)
            d_nl = np.linalg.norm(n_traj - l_traj, axis=-1).mean(axis=-1)
            d_sl = np.linalg.norm(s_traj - l_traj, axis=-1).mean(axis=-1)

            feats = np.stack([
                nat_top1_log, nat_margin,
                r2_sc_n, r2_sc_s, r2_sc_l,
                r2_sc_l - r2_sc_n, r2_sc_s - r2_sc_n,
                disagree, p_change, p2_knn_ent,
                d_ns, d_nl
            ], axis=1) # (N, 12)
            return feats

        X_cal_ot = build_calibrator_features(nat_log_ot, r2_scores_ot, n_sel_ot, ot_probs, p2_knn_entropy_ot, n_traj_ot, s_ot, l3_ot)
        X_cal_tr = build_calibrator_features(nat_log_tr_sub, r2_scores_tr_sub, n_sel_tr_sub, tr_probs[:N_SUB_TR], p2_knn_entropy_tr_sub, n_traj_tr_sub, s_tr_sub, l3_tr_sub)

        # Normalize features
        mu_x = X_cal_tr.mean(axis=0, keepdims=True)
        std_x = np.where(X_cal_tr.std(axis=0, keepdims=True) < 1e-6, 1.0, X_cal_tr.std(axis=0, keepdims=True)).astype(np.float32)
        X_cal_tr_norm = ((X_cal_tr - mu_x) / std_x).astype(np.float32)
        X_cal_ot_norm = ((X_cal_ot - mu_x) / std_x).astype(np.float32)

        # Targets for the 3 Hypotheses on Train Split
        # Errors for Native Top-1, Static, Learned-3
        top3_tr_trajs = np.stack([n_traj_tr_sub, s_tr_sub, l3_tr_sub], axis=1) # (N, 3, 12, 2)
        diff_tr = np.linalg.norm(top3_tr_trajs - future_tr[:N_SUB_TR, None], axis=-1)
        ade_tr = diff_tr.mean(axis=-1)
        fde_tr = diff_tr[:, :, -1]
        j_tr = np.maximum(ade_tr / 0.2, fde_tr / 0.4)
        good_tr = (ade_tr <= 0.2) & (fde_tr <= 0.4)

        # Soft utility distribution over the 3 hypotheses
        utility_tr = -2.0 * j_tr + 3.0 * good_tr.astype(np.float32)
        target_probs_tr = torch.softmax(torch.from_numpy(utility_tr).float(), dim=-1).to(device)

        # Targets on Outer Split for Evaluation
        top3_ot_trajs = np.stack([n_traj_ot, s_ot, l3_ot], axis=1) # (N, 3, 12, 2)
        diff_ot = np.linalg.norm(top3_ot_trajs - future_ot[:, None], axis=-1)
        ade_ot = diff_ot.mean(axis=-1)
        fde_ot = diff_ot[:, :, -1]
        j_ot = np.maximum(ade_ot / 0.2, fde_ot / 0.4)
        good_ot = (ade_ot <= 0.2) & (fde_ot <= 0.4)
        target_probs_ot = torch.softmax(torch.from_numpy(-2.0 * j_ot + 3.0 * good_ot.astype(np.float32)).float(), dim=-1).numpy()

        # Train Top3ProbabilityCalibrator
        calibrator = Top3ProbabilityCalibrator(in_dim=12).to(device)
        optimizer = torch.optim.AdamW(calibrator.parameters(), lr=2e-3, weight_decay=1e-4)

        ds_tr = TensorDataset(
            torch.from_numpy(X_cal_tr_norm).to(device),
            target_probs_tr
        )
        ldr_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(s))

        for ep in range(1, 11):
            calibrator.train()
            tot_loss = 0.0
            for bx, by in ldr_tr:
                bx, by = bx.to(device), by.to(device)
                logits, probs = calibrator(bx)
                loss = F.cross_entropy(logits, by)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item() * len(bx)
            if ep % 5 == 0 or ep == 10:
                print(f"Calibrator Ep {ep:2d}/10: Loss={tot_loss/N_SUB_TR:.4f}", flush=True)

        # Save Calibrator Checkpoint
        ckpt_dir = MODELS_DIR / p_id / f"seed_{s}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p = ckpt_dir / "calibrator.pth"
        torch.save(calibrator.state_dict(), ckpt_p)
        checkpoints[f"{p_id}_seed_{s}"] = hashlib.sha256(ckpt_p.read_bytes()).hexdigest()

        # Inference on Outer Split
        calibrator.eval()
        with torch.no_grad():
            _, p3_ot = calibrator(torch.from_numpy(X_cal_ot_norm).to(device))
            p3_ot = p3_ot.cpu().numpy() # (N_ot, 3)

        # Evaluation Metrics
        # 1. Negative Log Likelihood (NLL) against soft utility target
        eps = 1e-12
        nll_val = float(-np.mean(np.sum(target_probs_ot * np.log(np.clip(p3_ot, eps, 1.0)), axis=-1)))
        # Baseline uniform NLL
        unif_p = np.full_like(p3_ot, 1.0 / 3.0)
        nll_unif = float(-np.mean(np.sum(target_probs_ot * np.log(unif_p), axis=-1)))

        # 2. Brier Score
        brier_val = float(np.mean(np.sum((p3_ot - target_probs_ot)**2, axis=-1)))
        brier_unif = float(np.mean(np.sum((unif_p - target_probs_ot)**2, axis=-1)))

        # 3. ECE (Expected Calibration Error)
        ece_val, bin_stats = compute_ece(p3_ot, good_ot, num_bins=10)

        # 4. P_good: expected probability assigned to any good trajectory
        # P_good[i] = sum_{c in good} p_{i, c}
        p_good_per_sample = np.sum(p3_ot * good_ot.astype(np.float32), axis=1)

        # Subgroup probabilities and P_good
        subgroup_cal = {}
        for g_name, g_mask in SUBGROUPS.items():
            if g_mask.sum() == 0: continue
            subgroup_cal[g_name] = {
                "n_samples": int(g_mask.sum()),
                "mean_p_native": float(p3_ot[g_mask, 0].mean()),
                "mean_p_static": float(p3_ot[g_mask, 1].mean()),
                "mean_p_learned": float(p3_ot[g_mask, 2].mean()),
                "mean_p_good": float(p_good_per_sample[g_mask].mean()),
                "target_p_good": float(np.sum(target_probs_ot[g_mask] * good_ot[g_mask], axis=1).mean()),
                "coverage_at_least_one_good_pct": float(good_ot[g_mask].any(axis=1).mean() * 100)
            }

        seed_res = {
            "nll": nll_val,
            "nll_gain_vs_uniform": float(nll_unif - nll_val),
            "brier_score": brier_val,
            "brier_gain_vs_uniform": float(brier_unif - brier_val),
            "ece": ece_val,
            "calibration_bins": bin_stats,
            "subgroups": subgroup_cal
        }
        m1_results[p_id]["seeds"][f"seed_{s}"] = seed_res

        ov = subgroup_cal["overall"]
        ch = subgroup_cal["change"]
        st = subgroup_cal["stable"]
        print(f"  [Overall] P(Native)={ov['mean_p_native']:.3f}, P(Static)={ov['mean_p_static']:.3f}, P(Learned)={ov['mean_p_learned']:.3f} | ECE={ece_val:.4f}, NLL={nll_val:.4f}", flush=True)
        print(f"  [Probability Shift] Stable P(Learned)={st['mean_p_learned']:.3f} -> Change P(Learned)={ch['mean_p_learned']:.3f} ({ch['mean_p_learned']/max(st['mean_p_learned'], 1e-4):.2f}x increase!)", flush=True)
        print(f"  [Expected Mass on Good Trajectories] Overall P_good={ov['mean_p_good']:.3f}, Change P_good={ch['mean_p_good']:.3f}", flush=True)

# Macro Summary
macro_m1 = {
    "nll_mean": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["nll"] for s in SEEDS] for p in PROTOCOLS])),
    "brier_mean": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["brier_score"] for s in SEEDS] for p in PROTOCOLS])),
    "ece_mean": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["ece"] for s in SEEDS] for p in PROTOCOLS])),
    "subgroups": {
        "overall": {
            "p_native": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["mean_p_native"] for s in SEEDS] for p in PROTOCOLS])),
            "p_static": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["mean_p_static"] for s in SEEDS] for p in PROTOCOLS])),
            "p_learned": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["mean_p_learned"] for s in SEEDS] for p in PROTOCOLS])),
            "p_good": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["overall"]["mean_p_good"] for s in SEEDS] for p in PROTOCOLS]))
        },
        "stable": {
            "p_native": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["stable"]["mean_p_native"] for s in SEEDS] for p in PROTOCOLS])),
            "p_static": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["stable"]["mean_p_static"] for s in SEEDS] for p in PROTOCOLS])),
            "p_learned": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["stable"]["mean_p_learned"] for s in SEEDS] for p in PROTOCOLS])),
            "p_good": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["stable"]["mean_p_good"] for s in SEEDS] for p in PROTOCOLS]))
        },
        "change": {
            "p_native": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["mean_p_native"] for s in SEEDS] for p in PROTOCOLS])),
            "p_static": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["mean_p_static"] for s in SEEDS] for p in PROTOCOLS])),
            "p_learned": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["mean_p_learned"] for s in SEEDS] for p in PROTOCOLS])),
            "p_good": float(np.mean([[m1_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"]["change"]["mean_p_good"] for s in SEEDS] for p in PROTOCOLS]))
        }
    }
}

print("\n--- MACRO PROBABILITY CALIBRATION SUMMARY ---")
print(f"Overall Metrics: ECE={macro_m1['ece_mean']:.4f}, Brier={macro_m1['brier_mean']:.4f}, NLL={macro_m1['nll_mean']:.4f}")
print(f"Overall Probabilities: P(Native)={macro_m1['subgroups']['overall']['p_native']:.3f}, P(Static)={macro_m1['subgroups']['overall']['p_static']:.3f}, P(Learned)={macro_m1['subgroups']['overall']['p_learned']:.3f}")
print(f"Stable Probabilities:  P(Native)={macro_m1['subgroups']['stable']['p_native']:.3f}, P(Static)={macro_m1['subgroups']['stable']['p_static']:.3f}, P(Learned)={macro_m1['subgroups']['stable']['p_learned']:.3f}")
print(f"Change Probabilities:  P(Native)={macro_m1['subgroups']['change']['p_native']:.3f}, P(Static)={macro_m1['subgroups']['change']['p_static']:.3f}, P(Learned)={macro_m1['subgroups']['change']['p_learned']:.3f}")
print(f"Change vs Stable Learned Shift: {macro_m1['subgroups']['change']['p_learned'] / max(macro_m1['subgroups']['stable']['p_learned'], 1e-4):.2f}x increase")
print(f"Expected Probability Mass on Good Trajectories (P_good): Overall={macro_m1['subgroups']['overall']['p_good']:.3f}, Stable={macro_m1['subgroups']['stable']['p_good']:.3f}, Change={macro_m1['subgroups']['change']['p_good']:.3f}")

# Save All Results
with open(OUT_DIR / "PROBABILITY_CALIBRATION_AUDIT.json", "w") as f:
    json.dump({"results": m1_results, "macro": macro_m1, "checkpoints": checkpoints}, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "phase": "M1_TOP3_PROBABILITY_CALIBRATION",
        "macro_summary": macro_m1,
        "checkpoints": checkpoints,
        "conclusions": {
            "ece": macro_m1["ece_mean"],
            "brier_score": macro_m1["brier_mean"],
            "learned_probability_shift_ratio": macro_m1["subgroups"]["change"]["p_learned"] / max(macro_m1["subgroups"]["stable"]["p_learned"], 1e-4),
            "expected_p_good_overall": macro_m1["subgroups"]["overall"]["p_good"],
            "expected_p_good_change": macro_m1["subgroups"]["change"]["p_good"]
        }
    }, f, indent=2)

print("\nPhase M1 Top-3 Probability Calibration completed and saved to:", OUT_DIR, flush=True)
