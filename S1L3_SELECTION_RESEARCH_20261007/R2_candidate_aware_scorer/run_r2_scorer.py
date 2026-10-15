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
OUT_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R2_candidate_aware_scorer"
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
print(f"Executing Stage R2 Candidate-Aware Scorer on: {device}", flush=True)

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

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

# R2 Architecture: Candidate-Aware Scorer with Zero-Init Residual
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
        # Strict zero-initialization of the residual head
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, p2, cand_feat, set_feat, native_logits):
        # p2: (B, 96), cand_feat: (B, 24, 33), set_feat: (B, 4), native_logits: (B, 24)
        B, K, _ = cand_feat.shape
        h_p2 = self.p2_net(p2).unsqueeze(1).expand(-1, K, -1) # (B, 24, 32)
        h_cand = self.cand_net(cand_feat) # (B, 24, 32)
        h_set = set_feat.unsqueeze(1).expand(-1, K, -1) # (B, 24, 4)
        h_cat = torch.cat([h_p2, h_cand, h_set], dim=-1) # (B, 24, 68)
        delta = self.delta_head(h_cat).squeeze(-1) # (B, 24)
        score_new = native_logits + delta
        return score_new, delta

def extract_features(cands):
    # cands: (N, 24, 12, 2)
    N, K, T, _ = cands.shape
    cand_flat = cands.reshape(N, K, 24)

    # Geometry: endpoint distance, path length, speed, heading angle
    end_rad = np.linalg.norm(cands[:, :, -1, :], axis=-1, keepdims=True)
    step_diffs = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, K, 1, 2)), cands], axis=2), axis=2), axis=-1)
    path_len = step_diffs.sum(axis=-1, keepdims=True)
    speed = path_len / (T * 0.4)
    angle = np.arctan2(cands[:, :, -1, 1], cands[:, :, -1, 0])[..., None]
    geom = np.concatenate([end_rad, path_len, speed, angle], axis=-1) # (N, K, 4)

    # Provenance one-hot: 0..19 Native, 20 Static, 21 Learned-3, 22 Learned-0, 23 Learned-1
    prov = np.zeros((N, K, 5), dtype=np.float32)
    prov[:, :20, 0] = 1.0
    prov[:, 20, 1] = 1.0
    prov[:, 21, 2] = 1.0
    prov[:, 22, 3] = 1.0
    prov[:, 23, 4] = 1.0

    cand_feat = np.concatenate([cand_flat, geom, prov], axis=-1) # (N, K, 33)

    # Set context: mean/std of endpoint x and y across the 24 candidates
    end_xy = cands[:, :, -1, :] # (N, K, 2)
    set_mean = end_xy.mean(axis=1) # (N, 2)
    set_std = end_xy.std(axis=1)   # (N, 2)
    set_feat = np.concatenate([set_mean, set_std], axis=-1) # (N, 4)
    return cand_feat, set_feat

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
r2_results = {}
checkpoints = {}

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Training R2 Candidate-Aware Scorer on: {p_name} ({p_id})", flush=True)
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

    # P2 representation normalized strictly on training split
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

    # Generate Static 4 proposals
    s_tr = np.zeros((n_tr, 4, 12, 2), dtype=np.float32)
    for i in range(n_tr):
        for j in range(4):
            c_name = class_names_map[non_cont_classes[topc_tr[i, j]]]
            s_tr[i, j] = generate_motion_program(past_tr[i:i+1], default_params(c_name, 1))[0]

    s_ot = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        for j in range(4):
            c_name = class_names_map[non_cont_classes[topc_ot[i, j]]]
            s_ot[i, j] = generate_motion_program(past_ot[i:i+1], default_params(c_name, 1))[0]

    # Canonical Static-K24 control coverage
    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    k24_ctrl = np.concatenate([nat_ot, s_ot], axis=1)
    ctrl_cov = compute_errors(k24_ctrl, future_ot)["coverage"]

    # Native logits
    nat_log_tr = train_cache["logits"][tr_idx].astype(np.float32)
    nat_log_ot = train_cache["logits"][ot_idx].astype(np.float32)

    prior_log_tr = np.repeat((np.median(nat_log_tr, axis=1, keepdims=True) - nat_log_tr.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)
    prior_log_ot = np.repeat((np.median(nat_log_ot, axis=1, keepdims=True) - nat_log_ot.std(axis=1, keepdims=True)).astype(np.float32), 4, axis=1)

    k24_log_tr = np.concatenate([nat_log_tr, prior_log_tr], axis=1)
    k24_log_ot = np.concatenate([nat_log_ot, prior_log_ot], axis=1)

    r2_results[p_id] = {"seeds": {}, "macro": {}}

    for s in SEEDS:
        print(f"\n--- Training R2 Scorer Seed {s} on {p_id} ---", flush=True)
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

        # Load Bo4 Generator
        bo4_ckpt = p_cfg["bo4_dir"] / f"seed_{s}/model.pth"
        bo4_m = BestOf4Generator().to(device)
        bo4_m.load_state_dict(torch.load(bo4_ckpt, map_location=device))
        bo4_m.eval()

        def gen_bo4_subset(p2_arr, past_arr, N):
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

        bo4_tr = gen_bo4_subset(tr_p2_norm, past_tr, n_tr)
        bo4_ot = gen_bo4_subset(ot_p2_norm, past_ot, n_ot)

        # S1L3 candidates: Native20 + Static0 + Learned[3, 0, 1]
        nat_tr = train_cache["candidate"][tr_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
        k24_tr = np.concatenate([nat_tr, s_tr[:, :1], bo4_tr[:, [3, 0, 1]]], axis=1) # (N_tr, 24, 12, 2)
        k24_ot = np.concatenate([nat_ot, s_ot[:, :1], bo4_ot[:, [3, 0, 1]]], axis=1) # (N_ot, 24, 12, 2)

        # Extract features
        cand_feat_tr, set_feat_tr = extract_features(k24_tr)
        cand_feat_ot, set_feat_ot = extract_features(k24_ot)

        # Normalize candidate features by training statistics
        mu_cf = cand_feat_tr.mean(axis=(0, 1), keepdims=True)
        std_cf = np.where(cand_feat_tr.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, cand_feat_tr.std(axis=(0, 1), keepdims=True)).astype(np.float32)
        cand_feat_tr_norm = ((cand_feat_tr - mu_cf) / std_cf).astype(np.float32)
        cand_feat_ot_norm = ((cand_feat_ot - mu_cf) / std_cf).astype(np.float32)

        mu_sf = set_feat_tr.mean(axis=0, keepdims=True)
        std_sf = np.where(set_feat_tr.std(axis=0, keepdims=True) < 1e-6, 1.0, set_feat_tr.std(axis=0, keepdims=True)).astype(np.float32)
        set_feat_tr_norm = ((set_feat_tr - mu_sf) / std_sf).astype(np.float32)
        set_feat_ot_norm = ((set_feat_ot - mu_sf) / std_sf).astype(np.float32)

        # Compute Training Target Probabilities
        # Compute errors and J
        diff_tr = np.linalg.norm(k24_tr - future_tr[:, None], axis=-1)
        ade_tr = diff_tr.mean(-1)
        fde_tr = diff_tr[:, :, -1]
        j_tr = np.maximum(ade_tr / 0.2, fde_tr / 0.4)
        good_tr = (ade_tr <= 0.2) & (fde_tr <= 0.4)

        # Target utility: softmax of negative J with high bonus for good candidates
        utility_tr = -2.0 * j_tr + 3.0 * good_tr.astype(np.float32)
        target_probs_tr = torch.softmax(torch.from_numpy(utility_tr).float(), dim=-1).to(device)

        # Train CandidateAwareScorer
        scorer = CandidateAwareScorer().to(device)
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)

        ds_tr = TensorDataset(
            torch.from_numpy(tr_p2_norm),
            torch.from_numpy(cand_feat_tr_norm),
            torch.from_numpy(set_feat_tr_norm),
            torch.from_numpy(k24_log_tr),
            target_probs_tr
        )
        ldr_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(s))

        for ep in range(1, 16):
            scorer.train()
            tot_loss = 0.0
            for bp, bc, bs, bn, bq in ldr_tr:
                bp, bc, bs, bn, bq = bp.to(device), bc.to(device), bs.to(device), bn.to(device), bq.to(device)
                pred_scores, delta = scorer(bp, bc, bs, bn)
                loss = F.cross_entropy(pred_scores, bq)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
                optimizer.step()
                tot_loss += loss.item() * len(bp)
            if ep % 5 == 0 or ep == 15:
                print(f"Scorer Ep {ep:2d}/15: Loss={tot_loss/n_tr:.4f}", flush=True)

        # Save Scorer Checkpoint
        ckpt_dir = MODELS_DIR / p_id / f"seed_{s}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p = ckpt_dir / "scorer.pth"
        torch.save(scorer.state_dict(), ckpt_p)
        checkpoints[f"{p_id}_seed_{s}"] = hashlib.sha256(ckpt_p.read_bytes()).hexdigest()

        # Inference on Outer Split
        scorer.eval()
        with torch.no_grad():
            ot_scores, ot_deltas = scorer(
                torch.from_numpy(ot_p2_norm).to(device),
                torch.from_numpy(cand_feat_ot_norm).to(device),
                torch.from_numpy(set_feat_ot_norm).to(device),
                torch.from_numpy(k24_log_ot).to(device)
            )
            ot_scores = ot_scores.cpu().numpy()
            ot_deltas = ot_deltas.cpu().numpy()

        # Evaluate Selection Performance
        sel_idx = ot_scores.argmax(axis=1)
        row_idx = np.arange(n_ot)
        sel_trajs = k24_ot[row_idx, sel_idx]
        diff_ot = np.linalg.norm(sel_trajs - future_ot, axis=-1)
        sel_ade = diff_ot.mean(axis=-1)
        sel_fde = diff_ot[:, -1]
        sel_good = (sel_ade <= 0.2) & (sel_fde <= 0.4)
        sel_j = np.maximum(sel_ade / 0.2, sel_fde / 0.4)

        # Native Scorer Baseline on S1L3
        native_sel_idx = k24_log_ot.argmax(axis=1)
        nat_trajs = k24_ot[row_idx, native_sel_idx]
        diff_nat = np.linalg.norm(nat_trajs - future_ot, axis=-1)
        nat_ade = diff_nat.mean(axis=-1)
        nat_fde = diff_nat[:, -1]
        nat_good = (nat_ade <= 0.2) & (nat_fde <= 0.4)
        nat_j = np.maximum(nat_ade / 0.2, nat_fde / 0.4)

        # Oracle metrics for comparison
        err_ot_all = compute_errors(k24_ot, future_ot)
        oracle_cov = err_ot_all["coverage"]
        oracle_good = err_ot_all["good"]

        # Ranks under new scorer
        ranks_new = np.argsort(-ot_scores, axis=1)
        rank_of_k = np.zeros((n_ot, 24), dtype=np.int32)
        for i in range(n_ot):
            rank_of_k[i, ranks_new[i]] = np.arange(1, 25)

        # NewSupport Recall@1 and Recall@5
        new_support_mask = (~ctrl_cov) & oracle_cov
        ns_count = int(new_support_mask.sum())
        if ns_count > 0:
            ns_ranks = []
            for i in np.where(new_support_mask)[0]:
                good_cands = np.where(oracle_good[i])[0]
                ns_ranks.append(int(rank_of_k[i, good_cands].min()))
            ns_ranks = np.array(ns_ranks)
            ns_rec1 = float((ns_ranks <= 1).mean() * 100)
            ns_rec5 = float((ns_ranks <= 5).mean() * 100)
            ns_mrr = float((1.0 / ns_ranks).mean())
        else:
            ns_rec1, ns_rec5, ns_mrr = 0.0, 0.0, 0.0

        # Regret metrics: selected J - oracle J
        oracle_j = err_ot_all["J"]
        regret = sel_j - oracle_j
        severe_regret_frac = float((regret > 0.2).mean() * 100)

        nat_regret = nat_j - oracle_j
        nat_severe_regret_frac = float((nat_regret > 0.2).mean() * 100)

        # Gains vs Native Scorer
        top1_gain_pp = float((sel_good.mean() - nat_good.mean()) * 100)
        ch_msk = SUBGROUPS["change"]
        ch_top1_gain_pp = float((sel_good[ch_msk].mean() - nat_good[ch_msk].mean()) * 100)
        ade_gain_pct = float((nat_ade.mean() - sel_ade.mean()) / nat_ade.mean() * 100)
        fde_gain_pct = float((nat_fde.mean() - sel_fde.mean()) / nat_fde.mean() * 100)
        ch_ade_gain = float((nat_ade[ch_msk].mean() - sel_ade[ch_msk].mean()) / nat_ade[ch_msk].mean() * 100)
        ch_fde_gain = float((nat_fde[ch_msk].mean() - sel_fde[ch_msk].mean()) / nat_fde[ch_msk].mean() * 100)

        # Recalls
        rec1 = float((sel_good.sum()) / max(oracle_cov.sum(), 1) * 100)

        seed_res = {
            "top1_good_rate_pct": float(sel_good.mean() * 100),
            "top1_gain_pp_vs_native_scorer": top1_gain_pp,
            "top1_ade": float(sel_ade.mean()),
            "top1_fde": float(sel_fde.mean()),
            "top1_j": float(sel_j.mean()),
            "ade_gain_pct": ade_gain_pct,
            "fde_gain_pct": fde_gain_pct,
            "severe_regret_pct": severe_regret_frac,
            "severe_regret_drop_pp": float(nat_severe_regret_frac - severe_regret_frac),
            "change": {
                "top1_good_rate_pct": float(sel_good[ch_msk].mean() * 100),
                "top1_gain_pp": ch_top1_gain_pp,
                "change_ade": float(sel_ade[ch_msk].mean()),
                "change_fde": float(sel_fde[ch_msk].mean()),
                "change_ade_gain_pct": ch_ade_gain,
                "change_fde_gain_pct": ch_fde_gain
            },
            "new_support": {
                "n_samples": ns_count,
                "recall_at_1_pct": ns_rec1,
                "recall_at_5_pct": ns_rec5,
                "mrr": ns_mrr
            },
            "mean_delta_learned": float(ot_deltas[:, 21:].mean()),
            "mean_delta_native": float(ot_deltas[:, :20].mean())
        }
        r2_results[p_id]["seeds"][f"seed_{s}"] = seed_res

        print(f"[{p_id} Seed {s}] R2 Scorer Results:", flush=True)
        print(f"  Top1 Good Rate: {sel_good.mean()*100:.2f}% ({top1_gain_pp:+.2f} pp vs Native Scorer)", flush=True)
        print(f"  Change Good Rate: {sel_good[ch_msk].mean()*100:.2f}% ({ch_top1_gain_pp:+.2f} pp, ADE:{ch_ade_gain:+.2f}%, FDE:{ch_fde_gain:+.2f}%)", flush=True)
        print(f"  NewSupport Rec@1: {ns_rec1:.1f}%, Rec@5: {ns_rec5:.1f}% (was 0.0%)", flush=True)
        print(f"  Severe Regret: {severe_regret_frac:.2f}% (Dropped {nat_severe_regret_frac - severe_regret_frac:+.2f} pp)", flush=True)

# Save R2 Results
with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "stage": "R2_CANDIDATE_AWARE_SCORER",
        "results": r2_results,
        "checkpoints": checkpoints
    }, f, indent=2)

print("\nSaved all R2 results to:", OUT_DIR, flush=True)
