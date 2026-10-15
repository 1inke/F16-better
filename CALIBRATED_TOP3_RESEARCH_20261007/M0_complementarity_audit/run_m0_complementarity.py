import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
CONFIRM_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
OUT_DIR = REPO / "CALIBRATED_TOP3_RESEARCH_20261007" / "M0_complementarity_audit"
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
print(f"Executing Phase M0: Top-3 Complementarity Audit on: {device}", flush=True)

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

SEEDS = [42, 43, 44]

m0_results = {}

for p_cfg in PROTOCOLS:
    p_id = p_cfg["id"]
    p_name = p_cfg["name"]
    print(f"\n=======================================================", flush=True)
    print(f"Auditing Top-3 Complementarity on: {p_name} ({p_id})", flush=True)
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

    # Normalized P2 history representation
    tr_p2 = feat_p2[tr_idx]
    ot_p2 = feat_p2[ot_idx]
    mu_p2 = tr_p2.mean(0, keepdims=True)
    std_p2 = np.where(tr_p2.std(0, keepdims=True) < 1e-6, 1.0, tr_p2.std(0, keepdims=True)).astype(np.float32)
    ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

    future_ot = lab_train["future"][ot_idx].astype(np.float32)
    past_ot = raw_focal_past[ot_idx]

    # Load isolated Intent Classifier
    clf_saved = torch.load(p_cfg["intent_clf_path"], map_location=device, weights_only=False)
    clf = IntentClassifier(raw_clf_obs.shape[1], 0.1).to(device)
    clf.load_state_dict(clf_saved["state_dict"])
    clf.eval()

    with torch.no_grad():
        ot_probs = torch.softmax(clf(torch.from_numpy(((raw_clf_obs[ot_idx] - clf_saved["mean"]) / clf_saved["std"]).astype(np.float32)).to(device)), dim=-1).cpu().numpy()

    topc_ot = np.argsort(-ot_probs[:, non_cont_classes], axis=1)[:, :4]

    # Generate Static-1 Proposal (top-1 non-cont class)
    s_ot = np.zeros((n_ot, 12, 2), dtype=np.float32)
    for i in range(n_ot):
        c_name = class_names_map[non_cont_classes[topc_ot[i, 0]]]
        s_ot[i] = generate_motion_program(past_ot[i:i+1], default_params(c_name, 1))[0]

    # Native candidates and native top-1
    nat_ot = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
    nat_log_ot = train_cache["logits"][ot_idx, :20].astype(np.float32)
    native_sel_idx = nat_log_ot.argmax(axis=1)
    n_traj_ot = nat_ot[np.arange(n_ot), native_sel_idx] # (N_ot, 12, 2)

    # Errors for Native Top-1
    diff_n = np.linalg.norm(n_traj_ot - future_ot, axis=-1)
    ade_n = diff_n.mean(axis=-1)
    fde_n = diff_n[:, -1]
    good_n = (ade_n <= 0.2) & (fde_n <= 0.4)
    j_n = np.maximum(ade_n / 0.2, fde_n / 0.4)

    # Errors for Static Top-1
    diff_s = np.linalg.norm(s_ot - future_ot, axis=-1)
    ade_s = diff_s.mean(axis=-1)
    fde_s = diff_s[:, -1]
    good_s = (ade_s <= 0.2) & (fde_s <= 0.4)
    j_s = np.maximum(ade_s / 0.2, fde_s / 0.4)

    m0_results[p_id] = {"seeds": {}, "macro": {}}

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
        l3_traj_ot = bo4_ot[:, 3] # Learned slot 3: (N_ot, 12, 2)

        # Errors for Learned Slot 3
        diff_l = np.linalg.norm(l3_traj_ot - future_ot, axis=-1)
        ade_l = diff_l.mean(axis=-1)
        fde_l = diff_l[:, -1]
        good_l = (ade_l <= 0.2) & (fde_l <= 0.4)
        j_l = np.maximum(ade_l / 0.2, fde_l / 0.4)

        # Stack the 3 trajectories: (N_ot, 3, 12, 2)
        top3_trajs = np.stack([n_traj_ot, s_ot, l3_traj_ot], axis=1)

        # Pairwise Trajectory Distances (average over 12 timesteps)
        d_ns = np.linalg.norm(n_traj_ot - s_ot, axis=-1).mean(axis=-1)
        d_nl = np.linalg.norm(n_traj_ot - l3_traj_ot, axis=-1).mean(axis=-1)
        d_sl = np.linalg.norm(s_ot - l3_traj_ot, axis=-1).mean(axis=-1)
        mean_pairwise_dist = (d_ns + d_nl + d_sl) / 3.0

        # Endpoint Distances (at t=12)
        end_d_ns = np.linalg.norm(n_traj_ot[:, -1] - s_ot[:, -1], axis=-1)
        end_d_nl = np.linalg.norm(n_traj_ot[:, -1] - l3_traj_ot[:, -1], axis=-1)
        end_d_sl = np.linalg.norm(s_ot[:, -1] - l3_traj_ot[:, -1], axis=-1)

        # Effective distinct modes count (threshold > 0.1m endpoint distance)
        distinct_modes = np.ones(n_ot, dtype=np.float32)
        distinct_modes += (end_d_ns > 0.1).astype(np.float32)
        distinct_modes += ((end_d_nl > 0.1) & (end_d_sl > 0.1)).astype(np.float32)

        # Combined Top-3 Oracle Metrics
        top3_ade_all = np.stack([ade_n, ade_s, ade_l], axis=1) # (N_ot, 3)
        top3_fde_all = np.stack([fde_n, fde_s, fde_l], axis=1) # (N_ot, 3)
        top3_j_all = np.stack([j_n, j_s, j_l], axis=1)         # (N_ot, 3)

        top3_min_ade = top3_ade_all.min(axis=1)
        top3_min_fde = top3_fde_all.min(axis=1)
        top3_min_j = top3_j_all.min(axis=1)

        # Number of good trajectories per window: 0, 1, 2, 3
        good_count = good_n.astype(np.int32) + good_s.astype(np.int32) + good_l.astype(np.int32)

        # Full 8-way Venn decomposition
        venn_n_only = good_n & (~good_s) & (~good_l)
        venn_s_only = (~good_n) & good_s & (~good_l)
        venn_l_only = (~good_n) & (~good_s) & good_l
        venn_ns = good_n & good_s & (~good_l)
        venn_nl = good_n & (~good_s) & good_l
        venn_sl = (~good_n) & good_s & good_l
        venn_nsl = good_n & good_s & good_l
        venn_none = (~good_n) & (~good_s) & (~good_l)

        top3_union_good = good_n | good_s | good_l

        # Subgroup decomposition
        subgroup_decomp = {}
        for g_name, g_mask in SUBGROUPS.items():
            if g_mask.sum() == 0: continue
            g_N = int(g_mask.sum())
            subgroup_decomp[g_name] = {
                "n_samples": g_N,
                "native_top1_good_rate_pct": float(good_n[g_mask].mean() * 100),
                "static_top1_good_rate_pct": float(good_s[g_mask].mean() * 100),
                "learned3_good_rate_pct": float(good_l[g_mask].mean() * 100),
                "top3_union_good_rate_pct": float(top3_union_good[g_mask].mean() * 100),
                "union_gain_pp_vs_native": float((top3_union_good[g_mask].mean() - good_n[g_mask].mean()) * 100),
                "venn_counts": {
                    "none_good": int(venn_none[g_mask].sum()),
                    "native_only": int(venn_n_only[g_mask].sum()),
                    "static_only": int(venn_s_only[g_mask].sum()),
                    "learned_only": int(venn_l_only[g_mask].sum()),
                    "native_and_static": int(venn_ns[g_mask].sum()),
                    "native_and_learned": int(venn_nl[g_mask].sum()),
                    "static_and_learned": int(venn_sl[g_mask].sum()),
                    "all_three_good": int(venn_nsl[g_mask].sum())
                },
                "venn_fractions_pct": {
                    "none_good": float(venn_none[g_mask].mean() * 100),
                    "native_only": float(venn_n_only[g_mask].mean() * 100),
                    "static_only": float(venn_s_only[g_mask].mean() * 100),
                    "learned_only": float(venn_l_only[g_mask].mean() * 100),
                    "native_and_static": float(venn_ns[g_mask].mean() * 100),
                    "native_and_learned": float(venn_nl[g_mask].mean() * 100),
                    "static_and_learned": float(venn_sl[g_mask].mean() * 100),
                    "all_three_good": float(venn_nsl[g_mask].mean() * 100)
                },
                "net_rescues": {
                    "learned_only_rescue_pct": float(venn_l_only[g_mask].mean() * 100),
                    "static_only_rescue_pct": float(venn_s_only[g_mask].mean() * 100),
                    "static_and_learned_rescue_pct": float(venn_sl[g_mask].mean() * 100),
                    "total_proposal_rescues_pct": float(((good_s | good_l) & (~good_n))[g_mask].mean() * 100)
                },
                "quality_metrics": {
                    "native_ade": float(ade_n[g_mask].mean()),
                    "native_fde": float(fde_n[g_mask].mean()),
                    "top3_min_ade": float(top3_min_ade[g_mask].mean()),
                    "top3_min_fde": float(top3_min_fde[g_mask].mean()),
                    "ade_reduction_pct": float((ade_n[g_mask].mean() - top3_min_ade[g_mask].mean()) / ade_n[g_mask].mean() * 100),
                    "fde_reduction_pct": float((fde_n[g_mask].mean() - top3_min_fde[g_mask].mean()) / fde_n[g_mask].mean() * 100)
                },
                "good_trajectories_distribution": {
                    "0_of_3_pct": float((good_count[g_mask] == 0).mean() * 100),
                    "1_of_3_pct": float((good_count[g_mask] == 1).mean() * 100),
                    "2_of_3_pct": float((good_count[g_mask] == 2).mean() * 100),
                    "3_of_3_pct": float((good_count[g_mask] == 3).mean() * 100)
                },
                "diversity": {
                    "mean_pairwise_dist_m": float(mean_pairwise_dist[g_mask].mean()),
                    "expected_distinct_modes": float(distinct_modes[g_mask].mean())
                }
            }

        seed_res = {
            "subgroups": subgroup_decomp
        }
        m0_results[p_id]["seeds"][f"seed_{s}"] = seed_res

        ov = subgroup_decomp["overall"]
        ch = subgroup_decomp["change"]
        print(f"  [Overall] Native Top-1: {ov['native_top1_good_rate_pct']:.2f}% -> Top-3: {ov['top3_union_good_rate_pct']:.2f}% ({ov['union_gain_pp_vs_native']:+.2f} pp)", flush=True)
        print(f"  [Change]  Native Top-1: {ch['native_top1_good_rate_pct']:.2f}% -> Top-3: {ch['top3_union_good_rate_pct']:.2f}% ({ch['union_gain_pp_vs_native']:+.2f} pp)", flush=True)
        print(f"  [Change Rescues] Learned-only: {ch['net_rescues']['learned_only_rescue_pct']:.2f} pp, Static-only: {ch['net_rescues']['static_only_rescue_pct']:.2f} pp, Shared S+L: {ch['net_rescues']['static_and_learned_rescue_pct']:.2f} pp", flush=True)
        print(f"  [Quality] minADE@3: {ov['quality_metrics']['top3_min_ade']:.3f}m ({ov['quality_metrics']['ade_reduction_pct']:+.1f}%), Modes: {ov['diversity']['expected_distinct_modes']:.2f}/3.0", flush=True)

# Macro Summary Across Protocols
print(f"\n=======================================================", flush=True)
print(f"Synthesizing Macro Top-3 Complementarity Results", flush=True)
print(f"=======================================================", flush=True)

macro_decomp = {}
for g_name in ["overall", "stable", "transition", "change"]:
    macro_decomp[g_name] = {
        "native_good_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["native_top1_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "static_good_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["static_top1_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "learned_good_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["learned3_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "top3_good_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["top3_union_good_rate_pct"] for s in SEEDS] for p in PROTOCOLS])),
        "top3_gain_pp": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["union_gain_pp_vs_native"] for s in SEEDS] for p in PROTOCOLS])),
        "net_rescues": {
            "learned_only_pp": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["net_rescues"]["learned_only_rescue_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "static_only_pp": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["net_rescues"]["static_only_rescue_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "shared_static_learned_pp": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["net_rescues"]["static_and_learned_rescue_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "total_rescues_pp": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["net_rescues"]["total_proposal_rescues_pct"] for s in SEEDS] for p in PROTOCOLS]))
        },
        "quality": {
            "native_ade": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["quality_metrics"]["native_ade"] for s in SEEDS] for p in PROTOCOLS])),
            "top3_min_ade": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["quality_metrics"]["top3_min_ade"] for s in SEEDS] for p in PROTOCOLS])),
            "ade_reduction_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["quality_metrics"]["ade_reduction_pct"] for s in SEEDS] for p in PROTOCOLS]))
        },
        "good_trajectories_distribution": {
            "0_of_3_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["good_trajectories_distribution"]["0_of_3_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "1_of_3_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["good_trajectories_distribution"]["1_of_3_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "2_of_3_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["good_trajectories_distribution"]["2_of_3_pct"] for s in SEEDS] for p in PROTOCOLS])),
            "3_of_3_pct": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["good_trajectories_distribution"]["3_of_3_pct"] for s in SEEDS] for p in PROTOCOLS]))
        },
        "diversity": {
            "mean_pairwise_dist_m": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["diversity"]["mean_pairwise_dist_m"] for s in SEEDS] for p in PROTOCOLS])),
            "expected_distinct_modes": float(np.mean([[m0_results[p["id"]]["seeds"][f"seed_{s}"]["subgroups"][g_name]["diversity"]["expected_distinct_modes"] for s in SEEDS] for p in PROTOCOLS]))
        }
    }

print("\n--- MACRO VENN DECOMPOSITION SUMMARY ---")
print(f"Overall: Native {macro_decomp['overall']['native_good_pct']:.2f}% -> Top-3 {macro_decomp['overall']['top3_good_pct']:.2f}% ({macro_decomp['overall']['top3_gain_pp']:+.2f} pp)")
print(f"Change:  Native {macro_decomp['change']['native_good_pct']:.2f}% -> Top-3 {macro_decomp['change']['top3_good_pct']:.2f}% ({macro_decomp['change']['top3_gain_pp']:+.2f} pp)")
print(f"  Change Rescues Breakdown:")
print(f"    - Learned-only Rescue:         {macro_decomp['change']['net_rescues']['learned_only_pp']:.2f} pp ({macro_decomp['change']['net_rescues']['learned_only_pp']/max(macro_decomp['change']['net_rescues']['total_rescues_pp'], 1e-4)*100:.1f}% of total rescues)")
print(f"    - Static-only Rescue:          {macro_decomp['change']['net_rescues']['static_only_pp']:.2f} pp ({macro_decomp['change']['net_rescues']['static_only_pp']/max(macro_decomp['change']['net_rescues']['total_rescues_pp'], 1e-4)*100:.1f}% of total rescues)")
print(f"    - Shared (Static & Learned):   {macro_decomp['change']['net_rescues']['shared_static_learned_pp']:.2f} pp ({macro_decomp['change']['net_rescues']['shared_static_learned_pp']/max(macro_decomp['change']['net_rescues']['total_rescues_pp'], 1e-4)*100:.1f}% of total rescues)")
print(f"    - Total Net Rescue:            {macro_decomp['change']['net_rescues']['total_rescues_pp']:.2f} pp")
print(f"  Quality: minADE@3 = {macro_decomp['overall']['quality']['top3_min_ade']:.3f}m ({macro_decomp['overall']['quality']['ade_reduction_pct']:+.1f}% vs Native {macro_decomp['overall']['quality']['native_ade']:.3f}m)")
print(f"  Diversity: Mean Pairwise Dist = {macro_decomp['overall']['diversity']['mean_pairwise_dist_m']:.3f}m, Distinct Modes = {macro_decomp['overall']['diversity']['expected_distinct_modes']:.2f}/3.0")

# Save All Results
with open(OUT_DIR / "COMPLEMENTARITY_AUDIT.json", "w") as f:
    json.dump({"results": m0_results, "macro": macro_decomp}, f, indent=2)

with open(OUT_DIR / "RESULT.json", "w") as f:
    json.dump({
        "stage": "M0_TOP3_COMPLEMENTARITY_AUDIT",
        "macro_summary": macro_decomp,
        "conclusions": {
            "change_top3_gain_pp": macro_decomp["change"]["top3_gain_pp"],
            "learned_share_of_change_rescues_pct": macro_decomp["change"]["net_rescues"]["learned_only_pp"] / max(macro_decomp["change"]["net_rescues"]["total_rescues_pp"], 1e-4) * 100,
            "static_share_of_change_rescues_pct": macro_decomp["change"]["net_rescues"]["static_only_pp"] / max(macro_decomp["change"]["net_rescues"]["total_rescues_pp"], 1e-4) * 100,
            "shared_share_pct": macro_decomp["change"]["net_rescues"]["shared_static_learned_pp"] / max(macro_decomp["change"]["net_rescues"]["total_rescues_pp"], 1e-4) * 100,
            "distinct_modes_ratio": macro_decomp["overall"]["diversity"]["expected_distinct_modes"] / 3.0
        }
    }, f, indent=2)

print("\nPhase M0 Top-3 Complementarity Audit completed and saved to:", OUT_DIR, flush=True)
