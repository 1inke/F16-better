import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_P4_ADAPTIVE_SELECTOR_20261007"
STAGE_B_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"

STAGE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = STAGE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    IntentClassifier, CLASS_NAMES
)

sys.path.insert(0, str(REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts") if (REPO / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts").exists() else str(PLAN_DIR.parent / "F16_OVERNIGHT_AGENT_PLAN_20261007/scripts"))
from common import (
    compute_errors, effective_good_clusters, motion_summary
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params
from oracle_optimizer import _profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Stage P4 Adaptive Portfolio Selector on: {device}")

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
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr, n_ot = len(tr_idx), len(ot_idx)
outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2
}

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

# 2. Intent Classifier & Static Proposals for Both Train and Outer
print("Generating Intent Classifier and Static Proposals ...")
raw_clf_obs = np.concatenate([motion_all, social_all], axis=1).astype(np.float32)
m_clf = raw_clf_obs[tr_idx].mean(0, keepdims=True)
s_clf = raw_clf_obs[tr_idx].std(0, keepdims=True)
s_clf = np.where(s_clf < 1e-6, 1.0, s_clf).astype(np.float32)
tr_norm = ((raw_clf_obs[tr_idx] - m_clf) / s_clf).astype(np.float32)
tr_lbl = maneuver_labels(raw_focal_past, raw_focal_future, cfg)[tr_idx]

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

# Outer Static proposals
ot_norm = ((raw_clf_obs[ot_idx] - m_clf) / s_clf).astype(np.float32)
with torch.no_grad():
    outer_probs = torch.softmax(clf(torch.from_numpy(ot_norm).to(device)), dim=-1).cpu().numpy()
    train_probs = torch.softmax(clf(torch.from_numpy(tr_norm).to(device)), dim=-1).cpu().numpy()

non_cont_classes = [1, 2, 3, 4, 5, 6]
class_names_map = {1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}

def build_static4(probs, focal_past, n_samples):
    top_c = np.argsort(-probs[:, non_cont_classes], axis=1)[:, :4]
    s0 = np.zeros((n_samples, 4, 12, 2), dtype=np.float32)
    for j in range(4):
        for i in range(n_samples):
            c_name = class_names_map[non_cont_classes[top_c[i, j]]]
            p_def = default_params(c_name, 1)
            tr = generate_motion_program(focal_past[i:i+1], p_def)
            s0[i, j] = tr[0]
    return s0

focal_past_outer = raw_focal_past[ot_idx]
focal_past_train = raw_focal_past[tr_idx]

s0_trajs_ot = build_static4(outer_probs, focal_past_outer, n_ot)
s0_trajs_tr = build_static4(train_probs, focal_past_train, n_tr)

native_outer = train_cache["candidate"][ot_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
native_train = train_cache["candidate"][tr_idx, :, :24].reshape(-1, 20, 12, 2).astype(np.float32)
future_outer = lab_train["future"][ot_idx].astype(np.float32)
future_train = lab_train["future"][tr_idx].astype(np.float32)

k24_static = np.concatenate([native_outer, s0_trajs_ot], axis=1)
static_ade, static_fde, static_j, static_cov, static_eff = unpack_errors(k24_static, future_outer)
static_kin = motion_summary(k24_static)
control_subgroups = get_subgroups(static_ade, static_fde, static_j, static_cov)

# Native ground-truth coverage on train split (for rescue-critical weight)
native_tr_err = compute_errors(native_train, future_train)
native_good_tr = native_tr_err["coverage"]
tr_rescue_weight = np.where(~native_good_tr, 2.5, 1.0).astype(np.float32)

# Normalization of P2
tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
tr_p2_norm = ((tr_p2 - mu_p2) / std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

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

def extract_geom_features(props8):
    # props8: (N, 8, 12, 2)
    # Returns: (N, 8, 5)
    N, K, T, _ = props8.shape
    # 1. provenance: 0 for 0..3, 1 for 4..7
    prov = np.zeros((N, K, 1), dtype=np.float32)
    prov[:, 4:, 0] = 1.0

    # 2. path length
    step_diffs = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, K, 1, 2)), props8], axis=2), axis=2), axis=-1) # (N, K, 12)
    path_len = step_diffs.sum(axis=-1, keepdims=True) # (N, K, 1)

    # 3. endpoint radius
    end_rad = np.linalg.norm(props8[:, :, -1, :], axis=-1, keepdims=True) # (N, K, 1)

    # 4. mean speed
    speed = path_len / (T * 0.4) # (N, K, 1)

    # 5. net heading angle
    end_p = props8[:, :, -1, :] # (N, K, 2)
    angle = np.arctan2(end_p[:, :, 1], end_p[:, :, 0])[..., None] # (N, K, 1)

    geom = np.concatenate([prov, path_len, end_rad, speed, angle], axis=-1)
    return geom

# ProposalPortfolioSelector Architecture
class ProposalPortfolioSelector(nn.Module):
    def __init__(self, in_p2=96, in_geom=5, hidden_dim=64):
        super().__init__()
        self.p2_net = nn.Sequential(
            nn.Linear(in_p2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU()
        )
        self.geom_net = nn.Sequential(
            nn.Linear(in_geom, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )
        self.score_head = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_p2, x_geom):
        # x_p2: (B, 96), x_geom: (B, 8, 5)
        B = x_p2.shape[0]
        h_p2 = self.p2_net(x_p2).unsqueeze(1).expand(-1, 8, -1) # (B, 8, 32)
        h_geom = self.geom_net(x_geom) # (B, 8, 16)
        h_cat = torch.cat([h_p2, h_geom], dim=-1) # (B, 8, 48)
        logits = self.score_head(h_cat).squeeze(-1) # (B, 8)
        return logits

SEEDS = [42, 43, 44]
p4_seed_results = {}
checkpoints = {}

ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)
tr_p2_t = torch.from_numpy(tr_p2_norm).to(device)
past_tr_t = torch.from_numpy(focal_past_train).to(device)

for s in SEEDS:
    print(f"\n==========================================")
    print(f"   TRAINING ADAPTIVE SELECTOR SEED {s}")
    print(f"==========================================")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    bo4_ckpt = STAGE_B_DIR / f"models/seed_{s}/model.pth"
    bo4_model = BestOf4Generator().to(device)
    bo4_model.load_state_dict(torch.load(bo4_ckpt, map_location=device))
    bo4_model.eval()

    # Generate Bo4 proposals on Train split
    bo4_tr = np.zeros((n_tr, 4, 12, 2), dtype=np.float32)
    with torch.no_grad():
        CHUNK = 512
        for start_idx in range(0, n_tr, CHUNK):
            end_idx = min(start_idx + CHUNK, n_tr)
            c_p2 = tr_p2_t[start_idx:end_idx]
            c_past = past_tr_t[start_idx:end_idx]
            c_B = len(c_p2)
            c_raw = bo4_model(c_p2)
            c_past_rep = c_past.repeat_interleave(4, dim=0)
            c_raw_rep = c_raw.view(c_B * 4, 8)
            c_trajs_flat, _ = _profile(c_past_rep, c_raw_rep, "general")
            bo4_tr[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()

    # Generate Bo4 proposals on Outer split
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

    # Form 8 proposals on train and outer
    props8_tr = np.concatenate([s0_trajs_tr, bo4_tr], axis=1) # (N_tr, 8, 12, 2)
    props8_ot = np.concatenate([s0_trajs_ot, bo4_ot], axis=1) # (N_ot, 8, 12, 2)

    # Proposal Geometry
    geom8_tr = extract_geom_features(props8_tr)
    geom8_ot = extract_geom_features(props8_ot)

    # Normalize geometry features based on train statistics
    mu_geom = geom8_tr.mean(axis=(0, 1), keepdims=True)
    std_geom = geom8_tr.std(axis=(0, 1), keepdims=True)
    std_geom = np.where(std_geom < 1e-6, 1.0, std_geom).astype(np.float32)

    geom8_tr_norm = ((geom8_tr - mu_geom) / std_geom).astype(np.float32)
    geom8_ot_norm = ((geom8_ot - mu_geom) / std_geom).astype(np.float32)

    # Compute training usefulness target labels (ADE <= 0.2 and FDE <= 0.4)
    diff8_tr = np.linalg.norm(props8_tr - future_train[:, None, :, :], axis=-1) # (N_tr, 8, 12)
    ade8_tr = diff8_tr.mean(axis=-1)
    fde8_tr = diff8_tr[:, :, -1]
    good8_tr = ((ade8_tr <= 0.2) & (fde8_tr <= 0.4)).astype(np.float32) # (N_tr, 8)

    # Train Selector
    selector = ProposalPortfolioSelector().to(device)
    opt_sel = torch.optim.AdamW(selector.parameters(), lr=1e-3, weight_decay=1e-4)

    ds_sel = TensorDataset(
        torch.from_numpy(tr_p2_norm),
        torch.from_numpy(geom8_tr_norm),
        torch.from_numpy(good8_tr),
        torch.from_numpy(tr_rescue_weight)
    )
    ldr_sel = DataLoader(ds_sel, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(s))

    t0 = time.time()
    for ep in range(1, 16):
        selector.train()
        tot_l = 0.0
        for bx_p2, bx_geom, by_good, bw in ldr_sel:
            bx_p2, bx_geom, by_good, bw = bx_p2.to(device), bx_geom.to(device), by_good.to(device), bw.to(device)
            logits = selector(bx_p2, bx_geom) # (B, 8)
            bce = F.binary_cross_entropy_with_logits(logits, by_good, reduction='none') # (B, 8)
            loss = (bce * bw.unsqueeze(1)).mean()
            opt_sel.zero_grad(set_to_none=True)
            loss.backward()
            opt_sel.step()
            tot_l += loss.item() * len(bx_p2)
        if ep % 5 == 0 or ep == 15:
            print(f"Selector Seed {s} Epoch {ep:2d}/15: Loss={tot_l/n_tr:.4f}")

    # Save selector checkpoint
    ckpt_dir = MODELS_DIR / f"seed_{s}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = ckpt_dir / "selector.pth"
    torch.save(selector.state_dict(), ckpt_p)
    h_sha = hashlib.sha256(ckpt_p.read_bytes()).hexdigest()
    checkpoints[f"seed_{s}"] = h_sha

    # Inference on Outer Split
    selector.eval()
    with torch.no_grad():
        ot_geom_t = torch.from_numpy(geom8_ot_norm).to(device)
        ot_logits = selector(ot_p2_t, ot_geom_t).cpu().numpy() # (N_ot, 8)

    # Select Top 4 proposals per outer sample
    adaptive_proposals = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    selected_indices_record = []
    for i in range(n_ot):
        top4 = np.argsort(-ot_logits[i])[:4]
        adaptive_proposals[i] = props8_ot[i, top4]
        selected_indices_record.append(top4.tolist())

    # Form K24 Adaptive candidate set
    k24_adapt = np.concatenate([native_outer, adaptive_proposals], axis=1) # (N_ot, 24, 12, 2)
    ad_ade, ad_fde, ad_j, ad_cov, ad_eff = unpack_errors(k24_adapt, future_outer)
    ad_kin = motion_summary(k24_adapt)
    ad_subgroups = get_subgroups(ad_ade, ad_fde, ad_j, ad_cov)

    rescue = int(np.sum((~static_cov) & ad_cov))
    lost = int(np.sum(static_cov & (~ad_cov)))
    net_rescue = rescue - lost
    cov_gain_pp = float((ad_cov.mean() - static_cov.mean()) * 100.0)
    ch_gain_pp = float(ad_subgroups["change"]["coverage_pct"] - control_subgroups["change"]["coverage_pct"])
    ade_gain_pct = float((static_ade.mean() - ad_ade.mean()) / static_ade.mean() * 100.0)
    fde_gain_pct = float((static_fde.mean() - ad_fde.mean()) / static_fde.mean() * 100.0)
    ch_ade_gain = float((control_subgroups["change"]["min_ade"] - ad_subgroups["change"]["min_ade"]) / control_subgroups["change"]["min_ade"] * 100.0)
    ch_fde_gain = float((control_subgroups["change"]["min_fde"] - ad_subgroups["change"]["min_fde"]) / control_subgroups["change"]["min_fde"] * 100.0)

    speed_diff = float((ad_kin["speed_p95"] - static_kin["speed_p95"]) / static_kin["speed_p95"] * 100.0)
    acc_diff = float((ad_kin["accel_p95"] - static_kin["accel_p95"]) / static_kin["accel_p95"] * 100.0)
    jerk_diff = float((ad_kin["jerk_p95"] - static_kin["jerk_p95"]) / static_kin["jerk_p95"] * 100.0)
    motion_pass = bool(abs(speed_diff) <= 2.0 and abs(acc_diff) <= 2.0 and abs(jerk_diff) <= 2.0)

    # Static vs Learned share chosen by adaptive selector
    chosen_static_count = sum(sum(1 for idx in row if idx < 4) for row in selected_indices_record)
    chosen_learned_count = sum(sum(1 for idx in row if idx >= 4) for row in selected_indices_record)
    total_slots = n_ot * 4
    static_share = float(chosen_static_count / total_slots * 100.0)
    learned_share = float(chosen_learned_count / total_slots * 100.0)

    print(f"Seed {s} Adaptive Selector Results:")
    print(f"  Overall Coverage: {ad_cov.mean()*100:.2f}% ({cov_gain_pp:+.2f} pp, R:{rescue}, L:{lost}, Net:{net_rescue:+d})")
    print(f"  Change Coverage:  {ad_subgroups['change']['coverage_pct']:.2f}% vs {control_subgroups['change']['coverage_pct']:.2f}% ({ch_gain_pp:+.2f} pp)")
    print(f"  Change ADE Gain:  {ch_ade_gain:+.2f}%, FDE Gain: {ch_fde_gain:+.2f}%")
    print(f"  Chosen Portfolio Composition: Static={static_share:.1f}%, Learned={learned_share:.1f}%")
    print(f"  Kinematics: Accel p95 diff = {acc_diff:+.2f}% (Motion Pass: {motion_pass})")

    p4_seed_results[f"seed_{s}"] = {
        "overall": {
            "coverage_pct": float(ad_cov.mean() * 100),
            "coverage_gain_pp": cov_gain_pp,
            "rescue": rescue,
            "lost": lost,
            "net_rescue": net_rescue,
            "mean_j": float(ad_j.mean()),
            "min_ade": float(ad_ade.mean()),
            "ade_gain_pct": ade_gain_pct,
            "min_fde": float(ad_fde.mean()),
            "fde_gain_pct": fde_gain_pct
        },
        "change": {
            "change_coverage_pct": ad_subgroups["change"]["coverage_pct"],
            "change_coverage_gain_pp": ch_gain_pp,
            "change_ade_gain_pct": ch_ade_gain,
            "change_fde_gain_pct": ch_fde_gain
        },
        "subgroups": ad_subgroups,
        "kinematics": ad_kin,
        "motion_guard_pass": motion_pass,
        "accel_p95_diff_pct": acc_diff,
        "portfolio_composition": {
            "static_share_pct": static_share,
            "learned_share_pct": learned_share
        }
    }

# 3. Aggregate across 3 seeds
mean_ad_cov_gain = float(np.mean([p4_seed_results[f"seed_{s}"]["overall"]["coverage_gain_pp"] for s in SEEDS]))
mean_ad_ch_gain = float(np.mean([p4_seed_results[f"seed_{s}"]["change"]["change_coverage_gain_pp"] for s in SEEDS]))
mean_ad_ch_ade = float(np.mean([p4_seed_results[f"seed_{s}"]["change"]["change_ade_gain_pct"] for s in SEEDS]))
mean_ad_ch_fde = float(np.mean([p4_seed_results[f"seed_{s}"]["change"]["change_fde_gain_pct"] for s in SEEDS]))
ch_gains_list = [p4_seed_results[f"seed_{s}"]["change"]["change_coverage_gain_pp"] for s in SEEDS]
ch_pos_seeds = sum(1 for g in ch_gains_list if g > 0.0)
acc_diffs_list = [p4_seed_results[f"seed_{s}"]["accel_p95_diff_pct"] for s in SEEDS]
motion_all_pass = all(p4_seed_results[f"seed_{s}"]["motion_guard_pass"] for s in SEEDS)

# Primary Gate:
# - Overall coverage >= Static-K24
# - Change coverage mean >= Static-K24 +1.0 pp
# - Change >= 2/3 seeds positive
# - Overall minADE/FDE harm <= 1%
# - Change minADE/FDE at least one improved
# - motion guard pass
primary_gate_pass = bool(
    mean_ad_cov_gain >= 0.0 and
    mean_ad_ch_gain >= 1.0 and
    ch_pos_seeds >= 2 and
    (mean_ad_ch_ade > 0.0 or mean_ad_ch_fde > 0.0) and
    motion_all_pass
)

# Output action:
# If primary gate passed -> PORTFOLIO_K24_ACCEPTED
# If failed to beat fixed -> FREEZE_BEST_FIXED_OR_STATIC_K24
if primary_gate_pass:
    final_action = "PORTFOLIO_K24_ACCEPTED"
elif mean_ad_ch_gain <= 0.0:
    final_action = "STOP_PORTFOLIO_CHANGE_NOT_SOLVED"
else:
    final_action = "FREEZE_BEST_FIXED_OR_STATIC_K24"

print(f"\n==========================================")
print(f"   STAGE P4 ADAPTIVE SELECTOR SUMMARY")
print(f"==========================================")
print(f"Mean Overall Coverage Gain: {mean_ad_cov_gain:+.2f} pp")
print(f"Mean Change Coverage Gain:  {mean_ad_ch_gain:+.2f} pp (Seeds: {[round(x, 2) for x in ch_gains_list]}, Pos: {ch_pos_seeds}/3)")
print(f"Mean Change ADE Gain:       {mean_ad_ch_ade:+.2f}%, FDE Gain: {mean_ad_ch_fde:+.2f}%")
print(f"Accel p95 Diffs:            {[round(x, 2) for x in acc_diffs_list]}% (Motion All Pass: {motion_all_pass})")
print(f"Primary Gate Pass:          {primary_gate_pass}")
print(f"FINAL DECISION ACTION:      {final_action}")

# 4. Save results
p4_out = {
    "stage": "PORTFOLIO_P4_ADAPTIVE_SELECTOR",
    "primary_gate_pass": primary_gate_pass,
    "final_action": final_action,
    "aggregate": {
        "mean_overall_coverage_gain_pp": mean_ad_cov_gain,
        "mean_change_coverage_gain_pp": mean_ad_ch_gain,
        "change_positive_seeds_count": ch_pos_seeds,
        "change_gains_by_seed": ch_gains_list,
        "mean_change_ade_gain_pct": mean_ad_ch_ade,
        "mean_change_fde_gain_pct": mean_ad_ch_fde,
        "accel_p95_diffs_pct": acc_diffs_list,
        "motion_guards_all_pass": motion_all_pass
    },
    "by_seed": p4_seed_results,
    "checkpoints": checkpoints
}

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(p4_out, f, indent=2)

with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump({
        "stage": "PORTFOLIO_P4_ADAPTIVE_SELECTOR",
        "primary_gate_pass": primary_gate_pass,
        "final_action": final_action,
        "mean_overall_coverage_gain_pp": mean_ad_cov_gain,
        "mean_change_coverage_gain_pp": mean_ad_ch_gain,
        "change_positive_seeds_count": ch_pos_seeds
    }, f, indent=2)

print("Saved all P4 results to:", STAGE_DIR)
