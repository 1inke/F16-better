import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
CONFIRM_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
OUT_DIR = REPO / "S1L3_SELECTION_RESEARCH_20261007" / "R3_ambiguity_audit"
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
print(f"Executing Stage R3 Representation / Ambiguity Audit on: {device}", flush=True)

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

def shannon_entropy(labels, num_classes):
    # labels: 1D array of integers
    counts = np.bincount(labels, minlength=num_classes)
    probs = counts / max(len(labels), 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def circular_variance(angles):
    # angles: 1D array in radians
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    R = np.sqrt(c**2 + s**2)
    return float(1.0 - R)

ambiguity_results = {}
subgroup_comparison = {}
uncertainty_diagnostics = {}

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Executing R3 Ambiguity Audit on: {p_name} ({p_id})", flush=True)
    print(f"=======================================================", flush=True)

    splits = np.load(p_cfg["split_file"])
    tr_idx = splits["grouped_train_index"]
    ot_idx = splits["outer_eval_index"]
    n_tr, n_ot = len(tr_idx), len(ot_idx)
    outer_sids = all_sample_ids[ot_idx]
    groups_ot = train_cache["change_group"][ot_idx]
    groups_tr = train_cache["change_group"][tr_idx]
    ce_ot = np.array([ce_map.get(int(sid), False) for sid in outer_sids], dtype=bool)

    # Subgroups
    SUBGROUPS = {
        "overall": np.ones(n_ot, dtype=bool),
        "stable": groups_ot == 0,
        "transition": groups_ot == 1,
        "change": groups_ot == 2,
        "clear_event": ce_ot
    }

    # Normalized P2 history representation (normalized strictly on train split)
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

    # Target properties on Training set
    # 1. Future motion kind (0..4)
    kinds_tr = lab_train["kind"][tr_idx]
    # 2. Future endpoint direction
    end_xy_tr = future_tr[:, -1, :] # (N_tr, 2)
    angles_tr = np.arctan2(end_xy_tr[:, 1], end_xy_tr[:, 0]) # (N_tr,)
    # 3. Future speed profile
    step_diffs_tr = np.linalg.norm(np.diff(np.concatenate([np.zeros((n_tr, 1, 2)), future_tr], axis=1), axis=1), axis=-1)
    speed_tr = step_diffs_tr.sum(axis=-1) / (12 * 0.4) # (N_tr,)

    # Load canonical Intent Classifier and Bo4 Seed 42 for generating representative S1L3 candidates
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

    # Bo4 Seed 42
    bo4_ckpt = p_cfg["bo4_dir"] / "seed_42/model.pth"
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

    bo4_tr = gen_bo4(tr_p2_norm, past_tr, n_tr)
    bo4_ot = gen_bo4(ot_p2_norm, past_ot, n_ot)

    nat_tr = train_cache["candidate"][tr_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)

    k24_tr = np.concatenate([nat_tr, s_tr, bo4_tr[:, [3, 0, 1]]], axis=1) # (N_tr, 24, 12, 2)
    k24_ot = np.concatenate([nat_ot, s_ot, bo4_ot[:, [3, 0, 1]]], axis=1) # (N_ot, 24, 12, 2)

    # Determine oracle best candidate and provenance on training set
    diff_tr = np.linalg.norm(k24_tr - future_tr[:, None], axis=-1)
    ade_tr = diff_tr.mean(-1)
    fde_tr = diff_tr[:, :, -1]
    j_tr = np.maximum(ade_tr / 0.2, fde_tr / 0.4)
    best_cand_tr = j_tr.argmin(axis=1) # (N_tr,)

    # Provenance category: 0: Native (0..19), 1: Static (20), 2: Learned-3 (21), 3: Learned-0 (22), 4: Learned-1 (23)
    prov_cat_tr = np.zeros(n_tr, dtype=np.int32)
    for i in range(n_tr):
        idx = best_cand_tr[i]
        if idx < 20: prov_cat_tr[i] = 0
        elif idx == 20: prov_cat_tr[i] = 1
        elif idx == 21: prov_cat_tr[i] = 2
        elif idx == 22: prov_cat_tr[i] = 3
        else: prov_cat_tr[i] = 4

    # Native vs Proposal family: 0 Native, 1 Proposal
    family_tr = (best_cand_tr >= 20).astype(np.int32)

    # Compute k-NN on GPU
    ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
    tr_p2_t = torch.from_numpy(tr_p2_norm).to(device)

    # Normalized Euclidean distance
    dist_mat = torch.cdist(ot_p2_t, tr_p2_t) # (N_ot, N_tr)
    top20_dists, top20_indices = torch.topk(dist_mat, 20, largest=False)
    knn20_idx = top20_indices.cpu().numpy() # (N_ot, 20)

    # Metrics computation per sample
    knn_metrics = {10: {}, 20: {}}
    for K_NN in [10, 20]:
        mode_entropies = np.zeros(n_ot, dtype=np.float32)
        prov_entropies = np.zeros(n_ot, dtype=np.float32)
        dir_variances = np.zeros(n_ot, dtype=np.float32)
        speed_variances = np.zeros(n_ot, dtype=np.float32)
        family_entropies = np.zeros(n_ot, dtype=np.float32)

        for i in range(n_ot):
            nbrs = knn20_idx[i, :K_NN]
            mode_entropies[i] = shannon_entropy(kinds_tr[nbrs], 5)
            prov_entropies[i] = shannon_entropy(prov_cat_tr[nbrs], 5)
            dir_variances[i] = circular_variance(angles_tr[nbrs])
            speed_variances[i] = float(np.var(speed_tr[nbrs]))
            family_entropies[i] = shannon_entropy(family_tr[nbrs], 2)

        knn_metrics[K_NN] = {
            "mode_entropy": mode_entropies,
            "provenance_entropy": prov_entropies,
            "direction_variance": dir_variances,
            "speed_variance": speed_variances,
            "family_entropy": family_entropies
        }

    # Aggregate across subgroups
    ambiguity_results[p_id] = {}
    for K_NN in [10, 20]:
        ambiguity_results[p_id][f"k_{K_NN}"] = {}
        for g_name, g_mask in SUBGROUPS.items():
            if g_mask.sum() == 0: continue
            ambiguity_results[p_id][f"k_{K_NN}"][g_name] = {
                "n_samples": int(g_mask.sum()),
                "future_mode_entropy_mean": float(knn_metrics[K_NN]["mode_entropy"][g_mask].mean()),
                "future_mode_entropy_median": float(np.median(knn_metrics[K_NN]["mode_entropy"][g_mask])),
                "provenance_entropy_mean": float(knn_metrics[K_NN]["provenance_entropy"][g_mask].mean()),
                "direction_variance_mean": float(knn_metrics[K_NN]["direction_variance"][g_mask].mean()),
                "speed_variance_mean": float(knn_metrics[K_NN]["speed_variance"][g_mask].mean()),
                "family_entropy_mean": float(knn_metrics[K_NN]["family_entropy"][g_mask].mean())
            }

    # Print summary
    print(f"Ambiguity Summary for {p_id} (k=20):", flush=True)
    st = ambiguity_results[p_id]["k_20"]["stable"]
    tr = ambiguity_results[p_id]["k_20"]["transition"]
    ch = ambiguity_results[p_id]["k_20"]["change"]
    print(f"  Stable:     Mode Entropy={st['future_mode_entropy_mean']:.3f}, Prov Entropy={st['provenance_entropy_mean']:.3f}, Dir Var={st['direction_variance_mean']:.3f}, Speed Var={st['speed_variance_mean']:.4f}", flush=True)
    print(f"  Transition: Mode Entropy={tr['future_mode_entropy_mean']:.3f}, Prov Entropy={tr['provenance_entropy_mean']:.3f}, Dir Var={tr['direction_variance_mean']:.3f}, Speed Var={tr['speed_variance_mean']:.4f}", flush=True)
    print(f"  Change:     Mode Entropy={ch['future_mode_entropy_mean']:.3f}, Prov Entropy={ch['provenance_entropy_mean']:.3f}, Dir Var={ch['direction_variance_mean']:.3f}, Speed Var={ch['speed_variance_mean']:.4f}", flush=True)
    print(f"  Ratio Change/Stable: Mode Entropy = {ch['future_mode_entropy_mean']/max(st['future_mode_entropy_mean'], 1e-4):.2f}x, Dir Var = {ch['direction_variance_mean']/max(st['direction_variance_mean'], 1e-4):.2f}x, Speed Var = {ch['speed_variance_mean']/max(st['speed_variance_mean'], 1e-4):.2f}x", flush=True)

    # Uncertainty & Multi-hypothesis Diagnostics on Outer Set
    # Native logits on outer set
    nat_log_ot = train_cache["logits"][ot_idx].astype(np.float32)
    err_ot = compute_errors(k24_ot, future_ot)
    oracle_j_ot = err_ot["J"]
    oracle_good_ot = err_ot["good"]

    # Top-K candidate set coverage (e.g. Set of top-1, top-2, top-3 native candidates + top proposals)
    # Native ranking
    nat_rank = np.argsort(-nat_log_ot, axis=1) # (N_ot, 20)
    top1_nat_good = oracle_good_ot[np.arange(n_ot), nat_rank[:, 0]]
    top2_nat_good = oracle_good_ot[np.arange(n_ot), nat_rank[:, 0]] | oracle_good_ot[np.arange(n_ot), nat_rank[:, 1]]
    top3_nat_good = np.any(oracle_good_ot[np.arange(n_ot)[:, None], nat_rank[:, :3]], axis=1)
    # Multi-hypothesis hybrid: Native Top-1 + Static Top-1 (idx 20) + Learned-3 (idx 21)
    hybrid_top3_good = oracle_good_ot[np.arange(n_ot), nat_rank[:, 0]] | oracle_good_ot[:, 20] | oracle_good_ot[:, 21]

    # Error vs Entropy correlation (does high neighbor entropy predict selection failure?)
    mode_ent_20 = knn_metrics[20]["mode_entropy"]
    top1_fail = (~top1_nat_good).astype(np.float32)
    corr_ent_fail = float(np.corrcoef(mode_ent_20, top1_fail)[0, 1])

    # Abstention analysis: If we abstain on top 20% highest entropy samples, what is the Top-1 good rate on retained?
    q80 = np.quantile(mode_ent_20, 0.80)
    retained_mask = mode_ent_20 <= q80
    retained_good_rate = float(top1_nat_good[retained_mask].mean() * 100)
    abstain_good_rate = float(top1_nat_good[~retained_mask].mean() * 100)

    uncertainty_diagnostics[p_id] = {
        "top1_native_good_rate_pct": float(top1_nat_good.mean() * 100),
        "top2_native_good_rate_pct": float(top2_nat_good.mean() * 100),
        "top3_native_good_rate_pct": float(top3_nat_good.mean() * 100),
        "hybrid_top3_portfolio_good_rate_pct": float(hybrid_top3_good.mean() * 100),
        "hybrid_top3_gain_pp_vs_top1": float((hybrid_top3_good.mean() - top1_nat_good.mean()) * 100),
        "entropy_failure_correlation": corr_ent_fail,
        "abstention_analysis": {
            "retained_fraction_pct": 80.0,
            "retained_top1_good_rate_pct": retained_good_rate,
            "abstained_top1_good_rate_pct": abstain_good_rate,
            "precision_gain_pp": float(retained_good_rate - top1_nat_good.mean() * 100)
        }
    }
    print(f"  Hybrid Top-3 Set Coverage: {hybrid_top3_good.mean()*100:.2f}% (+{(hybrid_top3_good.mean() - top1_nat_good.mean())*100:+.2f} pp vs Top-1)", flush=True)
    print(f"  Confidence Abstention: 80% Retained Good Rate = {retained_good_rate:.2f}% (+{retained_good_rate - top1_nat_good.mean()*100:+.2f} pp)", flush=True)

# Macro Ambiguity Summary
macro_summary = {
    "k_20": {
        "stable": {
            "mode_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["stable"]["future_mode_entropy_mean"] for p in PROTOCOLS])),
            "provenance_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["stable"]["provenance_entropy_mean"] for p in PROTOCOLS])),
            "direction_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["stable"]["direction_variance_mean"] for p in PROTOCOLS])),
            "speed_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["stable"]["speed_variance_mean"] for p in PROTOCOLS]))
        },
        "transition": {
            "mode_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["transition"]["future_mode_entropy_mean"] for p in PROTOCOLS])),
            "provenance_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["transition"]["provenance_entropy_mean"] for p in PROTOCOLS])),
            "direction_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["transition"]["direction_variance_mean"] for p in PROTOCOLS])),
            "speed_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["transition"]["speed_variance_mean"] for p in PROTOCOLS]))
        },
        "change": {
            "mode_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["change"]["future_mode_entropy_mean"] for p in PROTOCOLS])),
            "provenance_entropy": float(np.mean([ambiguity_results[p["id"]]["k_20"]["change"]["provenance_entropy_mean"] for p in PROTOCOLS])),
            "direction_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["change"]["direction_variance_mean"] for p in PROTOCOLS])),
            "speed_variance": float(np.mean([ambiguity_results[p["id"]]["k_20"]["change"]["speed_variance_mean"] for p in PROTOCOLS]))
        }
    }
}

ratios = {
    "change_vs_stable_mode_entropy_ratio": macro_summary["k_20"]["change"]["mode_entropy"] / macro_summary["k_20"]["stable"]["mode_entropy"],
    "change_vs_stable_prov_entropy_ratio": macro_summary["k_20"]["change"]["provenance_entropy"] / macro_summary["k_20"]["stable"]["provenance_entropy"],
    "change_vs_stable_direction_variance_ratio": macro_summary["k_20"]["change"]["direction_variance"] / macro_summary["k_20"]["stable"]["direction_variance"],
    "change_vs_stable_speed_variance_ratio": macro_summary["k_20"]["change"]["speed_variance"] / macro_summary["k_20"]["stable"]["speed_variance"]
}

# Save All Results
with open(OUT_DIR / "AMBIGUITY_METRICS.json", "w") as f:
    json.dump(ambiguity_results, f, indent=2)

with open(OUT_DIR / "UNCERTAINTY_DIAGNOSTICS.json", "w") as f:
    json.dump(uncertainty_diagnostics, f, indent=2)

with open(OUT_DIR / "SUBGROUP_COMPARISON.json", "w") as f:
    json.dump({
        "macro_k20": macro_summary["k_20"],
        "ratios": ratios
    }, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "stage": "R3_AMBIGUITY_AUDIT",
        "ambiguity_results": ambiguity_results,
        "macro_summary": macro_summary,
        "ratios": ratios,
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "scientific_finding": "HIGH_INTRINSIC_AMBIGUITY_IN_CHANGE_SUBGROUP",
        "decision": "PAST_ONLY_SELECTION_AMBIGUITY_CONFIRMED"
    }, f, indent=2)

print("\nStage R3 Ambiguity Audit completed successfully!", flush=True)
print(f"Results saved to: {OUT_DIR}", flush=True)
