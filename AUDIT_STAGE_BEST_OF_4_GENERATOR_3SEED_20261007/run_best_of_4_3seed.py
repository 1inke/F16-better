import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007"
MODELS_DIR = STAGE_DIR / "models"

STAGE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Executing Stage B Best-of-4 Generator 3-Seed Formal Validation on: {device}")

def unpack_errors(cands, future):
    res = compute_errors(cands, future)
    eff_counts, _ = effective_good_clusters(cands, res["good"])
    return res["minADE"], res["minFDE"], res["J"], res["coverage"], int(eff_counts.sum()), res["error"]

# 1. Load Data
print("\n--- Loading Data ---")
train_cache = load_npz(REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz")
lab_train = load_npz(REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz")
enc_train = load_npz(REPO / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz")
splits_sf = np.load(REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz")

cfg = load_config()

all_sample_ids = train_cache["sample_id"]
N_TOTAL = len(all_sample_ids)

motion_all = train_cache["motion"].astype(np.float32)
social_all = train_cache["social"].astype(np.float32)
raw_focal_past, raw_focal_future = raw_focal_trajectories(all_sample_ids, motion_all)
maneuver_lbls = maneuver_labels(raw_focal_past, raw_focal_future, cfg)

# P2 Representation (Encoder Output: 96 dims)
feat_p2 = enc_train["tokens"].mean(axis=1).astype(np.float32)
print(f"P2 representation shape: {feat_p2.shape}")

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr = len(tr_idx)
n_ot = len(ot_idx)

outer_sample_ids = all_sample_ids[ot_idx]
groups_outer = train_cache["change_group"][ot_idx]

# Clear Event lookup
clear_event_map = load_clear_event_lookup(REPO)
clear_event_outer = np.array([clear_event_map.get(int(sid), False) for sid in outer_sample_ids], dtype=bool)
n_ce = int(clear_event_outer.sum())
print(f"Clear Event count in outer set: {n_ce}/{n_ot}")

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2,
    "clear_event": clear_event_outer
}

def compute_subgroup_metrics(ade, fde, j_val, cov):
    out = {}
    for sg_name, msk in SUBGROUPS.items():
        n_m = int(msk.sum())
        if n_m == 0:
            out[sg_name] = {"n": 0, "coverage_pct": 0.0, "covered_count": 0, "mean_j": 0.0, "min_ade": 0.0, "min_fde": 0.0}
            continue
        sub_ade = ade[msk]
        sub_fde = fde[msk]
        sub_j = j_val[msk]
        sub_cov = cov[msk]
        out[sg_name] = {
            "n": n_m,
            "coverage_pct": float(sub_cov.mean() * 100),
            "covered_count": int(sub_cov.sum()),
            "mean_j": float(sub_j.mean()),
            "min_ade": float(sub_ade.mean()),
            "min_fde": float(sub_fde.mean())
        }
    return out

# 2. Build Control (Static-K24)
print("Building Control (Static-K24) ...")
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

static_ade, static_fde, static_j, static_cov, static_eff, _ = unpack_errors(k24_static, future_outer)
static_kin = motion_summary(k24_static)
control_subgroups = compute_subgroup_metrics(static_ade, static_fde, static_j, static_cov)

# Native-only and static4 proposals kinematics
native_kin = motion_summary(native_outer)
static4_kin = motion_summary(s0_trajs)

print(f"CONTROL (Static-K24):")
print(f"  Overall Coverage: {static_cov.mean()*100:.2f}% ({static_cov.sum()}/{n_ot}), Mean J: {static_j.mean():.4f}, minADE: {static_ade.mean():.4f}, minFDE: {static_fde.mean():.4f}")
print(f"  Change Coverage: {control_subgroups['change']['coverage_pct']:.2f}% ({control_subgroups['change']['covered_count']}/{control_subgroups['change']['n']})")

# 3. Model Definition: BestOf4Generator
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
        self._init_heads()

    def _init_heads(self):
        # Quadrant diversity initialization
        biases = [
            # Slot 0: Straight / Accel (s ~ 1.3, theta ~ 0)
            [0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Slot 1: Turn Left (s ~ 1.0, theta ~ +60 deg)
            [0.0, 0.0, 0.0, 1.2, 1.2, 1.2, 0.0, 0.0],
            # Slot 2: Turn Right (s ~ 1.0, theta ~ -60 deg)
            [0.0, 0.0, 0.0, -1.2, -1.2, -1.2, 0.0, 0.0],
            # Slot 3: Decel / Stop (s ~ 0.4, theta ~ 0 deg)
            [-1.8, -1.8, -1.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
        for slot_idx, head in enumerate(self.heads):
            final_linear = head[-1]
            nn.init.xavier_uniform_(final_linear.weight, gain=0.1)
            final_linear.bias.data = torch.tensor(biases[slot_idx], dtype=torch.float32)

    def forward(self, x):
        feat = self.backbone(x)
        out = [head(feat) for head in self.heads] # 4 x (B, 8)
        return torch.stack(out, dim=1) # (B, 4, 8)

# 4. Training Data Preparation
tr_p2 = feat_p2[tr_idx]
ot_p2 = feat_p2[ot_idx]
mu_p2 = tr_p2.mean(axis=0, keepdims=True)
std_p2 = tr_p2.std(axis=0, keepdims=True)
std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)

tr_p2_norm = ((tr_p2 - mu_p2) / std_p2).astype(np.float32)
ot_p2_norm = ((ot_p2 - mu_p2) / std_p2).astype(np.float32)

tr_focal_past = raw_focal_past[tr_idx]
tr_future = lab_train["future"][tr_idx].astype(np.float32)

LAMBDA_DIV = 0.05
DELTA_DIV = 0.30
EPOCHS = 25
SEEDS = [42, 43, 44]

seed_results = {}
checkpoints = {}
rescued_sample_ids_by_seed = {}
lost_sample_ids_by_seed = {}
change_rescued_ids_by_seed = {}
change_lost_ids_by_seed = {}
kinematics_attributions = {}

ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

for s in SEEDS:
    print(f"\n==========================================")
    print(f"   TRAINING BEST-OF-4 GENERATOR SEED {s}")
    print(f"==========================================")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    model = BestOf4Generator(in_dim=96, hidden_dim=128, num_slots=4, param_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.from_numpy(tr_p2_norm),
        torch.from_numpy(tr_focal_past),
        torch.from_numpy(tr_future)
    )
    g_seed = torch.Generator().manual_seed(s)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=g_seed)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        tot_best_ade = 0.0
        tot_div_loss = 0.0
        slot_win_counts = np.zeros(4, dtype=np.int64)

        for bx_p2, b_past, b_gt in loader:
            B_cur = len(bx_p2)
            bx_p2, b_past, b_gt = bx_p2.to(device), b_past.to(device), b_gt.to(device)

            raw_params = model(bx_p2) # (B, 4, 8)
            
            past_rep = b_past.repeat_interleave(4, dim=0) # (4B, 8, 2)
            raw_rep = raw_params.view(B_cur * 4, 8)       # (4B, 8)
            trajs_flat, _ = _profile(past_rep, raw_rep, "general")
            trajs_4 = trajs_flat.view(B_cur, 4, 12, 2)

            diff = torch.linalg.vector_norm(trajs_4 - b_gt.unsqueeze(1), dim=-1)
            ade_slot = diff.mean(dim=-1)
            fde_slot = diff[:, :, -1]
            l_traj = ade_slot + 0.5 * fde_slot

            best_loss, best_idx = torch.min(l_traj, dim=1)
            l_best4 = best_loss.mean()

            div_loss = torch.tensor(0.0, device=device)
            pair_cnt = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    d_ij = torch.linalg.vector_norm(trajs_4[:, i] - trajs_4[:, j], dim=-1).mean(dim=-1)
                    pen = F.relu(DELTA_DIV - d_ij) ** 2
                    div_loss = div_loss + pen.mean()
                    pair_cnt += 1
            div_loss = div_loss / pair_cnt

            loss = l_best4 + LAMBDA_DIV * div_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * B_cur
            tot_best_ade += ade_slot.gather(1, best_idx.unsqueeze(1)).mean().item() * B_cur
            tot_div_loss += div_loss.item() * B_cur
            slot_win_counts += np.bincount(best_idx.detach().cpu().numpy(), minlength=4)

        if epoch % 5 == 0 or epoch == EPOCHS:
            win_pcts = slot_win_counts / n_tr * 100
            print(f"Seed {s} Epoch {epoch:2d}/{EPOCHS}: Loss={tot_loss/n_tr:.4f}, Best ADE={tot_best_ade/n_tr:.4f} | Winners: [{win_pcts[0]:.1f}%, {win_pcts[1]:.1f}%, {win_pcts[2]:.1f}%, {win_pcts[3]:.1f}%]")

    # Save checkpoint
    ckpt_dir = MODELS_DIR / f"seed_{s}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pth"
    torch.save(model.state_dict(), ckpt_path)
    h_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    checkpoints[f"seed_{s}"] = h_sha
    print(f"Checkpoint saved: {ckpt_path} (SHA: {h_sha[:16]}...)")

    # Outer Inference
    model.eval()
    bo4_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
    bo4_params = np.zeros((n_ot, 4, 8), dtype=np.float32)

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
            c_trajs_flat, c_p_out = _profile(c_past_rep, c_raw_rep, "general")

            bo4_trajs[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()
            bo4_params[start_idx:end_idx] = c_p_out.view(c_B, 4, 8).cpu().numpy()

    # Evaluation with K=24
    k24_bo4 = np.concatenate([native_outer, bo4_trajs], axis=1)
    bo4_ade, bo4_fde, bo4_j, bo4_cov, bo4_eff, bo4_err_matrix = unpack_errors(k24_bo4, future_outer)
    bo4_kin = motion_summary(k24_bo4)
    treatment_subgroups = compute_subgroup_metrics(bo4_ade, bo4_fde, bo4_j, bo4_cov)

    # Slot Collapse & Diversity
    pw_dist = np.zeros((n_ot, 6), dtype=np.float32)
    p_idx = 0
    for i in range(4):
        for j in range(i + 1, 4):
            d_ij = np.linalg.norm(bo4_trajs[:, i] - bo4_trajs[:, j], axis=-1).mean(axis=-1)
            pw_dist[:, p_idx] = d_ij
            p_idx += 1

    mean_pw_traj_dist = float(pw_dist.mean())
    median_pw_traj_dist = float(np.median(pw_dist))
    min_pw_traj_dist = float(pw_dist.min())

    effective_slots_per_sample = []
    for n in range(n_ot):
        eff_cnt = 0
        for i in range(4):
            min_d = min(np.linalg.norm(bo4_trajs[n, i] - bo4_trajs[n, j], axis=-1).mean() for j in range(4) if j != i)
            if min_d >= 0.10:
                eff_cnt += 1
        effective_slots_per_sample.append(eff_cnt)
    effective_slots_mean = float(np.mean(effective_slots_per_sample))

    fut_diff = np.linalg.norm(bo4_trajs - future_outer[:, None, :, :], axis=-1)
    outer_ade_slots = fut_diff.mean(axis=-1)
    outer_fde_slots = fut_diff[:, :, -1]
    outer_ltraj_slots = outer_ade_slots + 0.5 * outer_fde_slots
    winning_slot_idx = np.argmin(outer_ltraj_slots, axis=1)

    slot_winners_overall = (np.bincount(winning_slot_idx, minlength=4) / n_ot).tolist()
    slot_winners_stable = (np.bincount(winning_slot_idx[SUBGROUPS["stable"]], minlength=4) / int(SUBGROUPS["stable"].sum())).tolist()
    slot_winners_transition = (np.bincount(winning_slot_idx[SUBGROUPS["transition"]], minlength=4) / int(SUBGROUPS["transition"].sum())).tolist()
    slot_winners_change = (np.bincount(winning_slot_idx[SUBGROUPS["change"]], minlength=4) / int(SUBGROUPS["change"].sum())).tolist()
    slot_winners_ce = (np.bincount(winning_slot_idx[SUBGROUPS["clear_event"]], minlength=4) / max(int(SUBGROUPS["clear_event"].sum()), 1)).tolist()

    # Slot Entropy: H = -sum(p * log2(p))
    def calc_entropy(probs):
        return float(-sum(p * math.log2(p) for p in probs if p > 1e-8))

    slot_entropy_overall = calc_entropy(slot_winners_overall)
    slot_entropy_change = calc_entropy(slot_winners_change)
    max_slot_dominance = float(max(slot_winners_overall) * 100)
    is_slot_collapsed = bool(mean_pw_traj_dist < 0.10 or max_slot_dominance > 75.0 or effective_slots_mean < 2.0)

    # Rescue & Lost Sets
    rescue_mask = (~static_cov) & bo4_cov
    lost_mask = static_cov & (~bo4_cov)
    rescue_ids = outer_sample_ids[rescue_mask].tolist()
    lost_ids = outer_sample_ids[lost_mask].tolist()
    rescued_sample_ids_by_seed[f"seed_{s}"] = rescue_ids
    lost_sample_ids_by_seed[f"seed_{s}"] = lost_ids

    change_mask = SUBGROUPS["change"]
    change_rescue_ids = outer_sample_ids[rescue_mask & change_mask].tolist()
    change_lost_ids = outer_sample_ids[lost_mask & change_mask].tolist()
    change_rescued_ids_by_seed[f"seed_{s}"] = change_rescue_ids
    change_lost_ids_by_seed[f"seed_{s}"] = change_lost_ids

    rescue = len(rescue_ids)
    lost = len(lost_ids)
    net_rescue = rescue - lost
    cov_gain_pp = float((bo4_cov.mean() - static_cov.mean()) * 100.0)
    j_gain_pct = float((static_j.mean() - bo4_j.mean()) / static_j.mean() * 100.0)
    ade_gain_pct = float((static_ade.mean() - bo4_ade.mean()) / static_ade.mean() * 100.0)
    fde_gain_pct = float((static_fde.mean() - bo4_fde.mean()) / static_fde.mean() * 100.0)

    change_cov_gain_pp = float(treatment_subgroups["change"]["coverage_pct"] - control_subgroups["change"]["coverage_pct"])
    change_ade_gain_pct = float((control_subgroups["change"]["min_ade"] - treatment_subgroups["change"]["min_ade"]) / control_subgroups["change"]["min_ade"] * 100.0)
    change_fde_gain_pct = float((control_subgroups["change"]["min_fde"] - treatment_subgroups["change"]["min_fde"]) / control_subgroups["change"]["min_fde"] * 100.0)

    speed_p95_diff_pct = float((bo4_kin["speed_p95"] - static_kin["speed_p95"]) / static_kin["speed_p95"] * 100.0)
    acc_p95_diff_pct = float((bo4_kin["accel_p95"] - static_kin["accel_p95"]) / static_kin["accel_p95"] * 100.0)
    jerk_p95_diff_pct = float((bo4_kin["jerk_p95"] - static_kin["jerk_p95"]) / static_kin["jerk_p95"] * 100.0)

    motion_guard_pass = bool(
        abs(speed_p95_diff_pct) <= 2.0 and
        abs(acc_p95_diff_pct) <= 2.0 and
        abs(jerk_p95_diff_pct) <= 2.0
    )

    # Kinematics Attribution:
    # 1. Learned 4 proposals across all outer samples:
    learned4_kin = motion_summary(bo4_trajs)
    # 2. Rescued winner proposals specifically:
    if rescue > 0:
        rescue_indices = np.where(rescue_mask)[0]
        rescued_winning_trajs = []
        for r_idx in rescue_indices:
            # find which of the 4 bo4 proposals was good (ade <= 0.2 and fde <= 0.4)
            r_ade = outer_ade_slots[r_idx] # (4,)
            r_fde = outer_fde_slots[r_idx] # (4,)
            is_good = (r_ade <= 0.2) & (r_fde <= 0.4)
            if is_good.any():
                w_slot = np.where(is_good)[0][0]
            else:
                w_slot = winning_slot_idx[r_idx]
            rescued_winning_trajs.append(bo4_trajs[r_idx, w_slot:w_slot+1])
        rescued_winning_trajs_np = np.concatenate(rescued_winning_trajs, axis=0)[:, None, :, :] # (N_rescue, 1, 12, 2)
        rescued_winners_kin = motion_summary(rescued_winning_trajs_np)
    else:
        rescued_winners_kin = {"speed_p95": 0.0, "accel_mean": 0.0, "accel_p95": 0.0, "jerk_mean": 0.0, "jerk_p95": 0.0}

    kinematics_attributions[f"seed_{s}"] = {
        "k24_full": bo4_kin,
        "learned4_all": learned4_kin,
        "rescued_winners": rescued_winners_kin
    }

    seed_results[f"seed_{s}"] = {
        "overall": {
            "coverage_pct": float(bo4_cov.mean() * 100),
            "covered_count": int(bo4_cov.sum()),
            "coverage_gain_pp": cov_gain_pp,
            "rescue": rescue,
            "lost": lost,
            "net_rescue": net_rescue,
            "mean_j": float(bo4_j.mean()),
            "j_gain_pct": j_gain_pct,
            "min_ade": float(bo4_ade.mean()),
            "ade_gain_pct": ade_gain_pct,
            "min_fde": float(bo4_fde.mean()),
            "fde_gain_pct": fde_gain_pct,
            "effective_good_clusters": int(bo4_eff)
        },
        "subgroups": treatment_subgroups,
        "gains": {
            "change_coverage_gain_pp": change_cov_gain_pp,
            "change_ade_gain_pct": change_ade_gain_pct,
            "change_fde_gain_pct": change_fde_gain_pct
        },
        "kinematics": {
            "speed_p95": float(bo4_kin["speed_p95"]),
            "speed_p95_diff_pct": speed_p95_diff_pct,
            "accel_mean": float(bo4_kin["accel_mean"]),
            "accel_p95": float(bo4_kin["accel_p95"]),
            "accel_p95_diff_pct": acc_p95_diff_pct,
            "jerk_mean": float(bo4_kin["jerk_mean"]),
            "jerk_p95": float(bo4_kin["jerk_p95"]),
            "jerk_p95_diff_pct": jerk_p95_diff_pct,
            "motion_guard_pass": motion_guard_pass
        },
        "slot_collapse_monitoring": {
            "mean_pairwise_traj_dist_m": mean_pw_traj_dist,
            "median_pairwise_traj_dist_m": median_pw_traj_dist,
            "min_pairwise_traj_dist_m": min_pw_traj_dist,
            "effective_slots_mean": effective_slots_mean,
            "slot_winners_overall_pct": [round(x * 100, 2) for x in slot_winners_overall],
            "slot_winners_change_pct": [round(x * 100, 2) for x in slot_winners_change],
            "slot_entropy_overall_bits": slot_entropy_overall,
            "slot_entropy_change_bits": slot_entropy_change,
            "max_slot_dominance_pct": max_slot_dominance,
            "is_slot_collapsed": is_slot_collapsed
        }
    }

    print(f"Seed {s} Results:")
    print(f"  Overall Coverage: {bo4_cov.mean()*100:.2f}% [{cov_gain_pp:+.2f} pp, Rescue: {rescue}, Lost: {lost}, Net: {net_rescue:+d}]")
    print(f"  Change Coverage: {treatment_subgroups['change']['coverage_pct']:.2f}% vs {control_subgroups['change']['coverage_pct']:.2f}% [{change_cov_gain_pp:+.2f} pp]")
    print(f"  Change minADE Gain: {change_ade_gain_pct:+.2f}%, Change minFDE Gain: {change_fde_gain_pct:+.2f}%")
    print(f"  Kinematics: Speed diff={speed_p95_diff_pct:+.2f}%, Acc diff={acc_p95_diff_pct:+.2f}% -> {motion_guard_pass}")
    print(f"  Slot Collapse: {is_slot_collapsed} (mean dist={mean_pw_traj_dist:.3f}m, entropy={slot_entropy_overall:.3f} bits)")

# 5. Cross-Seed Consistency Audit
print("\n--- Cross-Seed Sample Consistency Audit ---")
def find_overlap(id_lists):
    # id_lists is list of lists
    from collections import Counter
    all_items = [x for sub in id_lists for x in sub]
    cnt = Counter(all_items)
    at_least_2 = [k for k, v in cnt.items() if v >= 2]
    all_3 = [k for k, v in cnt.items() if v == 3]
    return sorted(at_least_2), sorted(all_3)

r_lists = [rescued_sample_ids_by_seed[f"seed_{s}"] for s in SEEDS]
l_lists = [lost_sample_ids_by_seed[f"seed_{s}"] for s in SEEDS]
r_2of3, r_3of3 = find_overlap(r_lists)
l_2of3, l_3of3 = find_overlap(l_lists)

cr_lists = [change_rescued_ids_by_seed[f"seed_{s}"] for s in SEEDS]
cl_lists = [change_lost_ids_by_seed[f"seed_{s}"] for s in SEEDS]
cr_2of3, cr_3of3 = find_overlap(cr_lists)
cl_2of3, cl_3of3 = find_overlap(cl_lists)

print(f"Overall Rescues: 2/3 seeds common = {len(r_2of3)}, 3/3 seeds common = {len(r_3of3)}")
print(f"Overall Losts:   2/3 seeds common = {len(l_2of3)}, 3/3 seeds common = {len(l_3of3)}")
print(f"Change Rescues:  2/3 seeds common = {len(cr_2of3)}, 3/3 seeds common = {len(cr_3of3)}")
print(f"Change Losts:    2/3 seeds common = {len(cl_2of3)}, 3/3 seeds common = {len(cl_3of3)}")

consistency_audit = {
    "overall": {
        "rescues_2of3_count": len(r_2of3),
        "rescues_2of3_sample_ids": r_2of3,
        "rescues_3of3_count": len(r_3of3),
        "rescues_3of3_sample_ids": r_3of3,
        "lost_2of3_count": len(l_2of3),
        "lost_2of3_sample_ids": l_2of3,
        "lost_3of3_count": len(l_3of3),
        "lost_3of3_sample_ids": l_3of3
    },
    "change": {
        "change_rescues_2of3_count": len(cr_2of3),
        "change_rescues_2of3_sample_ids": cr_2of3,
        "change_rescues_3of3_count": len(cr_3of3),
        "change_rescues_3of3_sample_ids": cr_3of3,
        "change_lost_2of3_count": len(cl_2of3),
        "change_lost_2of3_sample_ids": cl_2of3,
        "change_lost_3of3_count": len(cl_3of3),
        "change_lost_3of3_sample_ids": cl_3of3
    }
}
with open(STAGE_DIR / "CONSISTENCY_AUDIT.json", "w") as f:
    json.dump(consistency_audit, f, indent=2)

# Save Kinematics Attribution Audit
kinematics_attribution_audit = {
    "control_proposals": {
        "native20": native_kin,
        "static4": static4_kin,
        "static_k24": static_kin
    },
    "by_seed": kinematics_attributions
}
with open(STAGE_DIR / "KINEMATICS_ATTRIBUTION_AUDIT.json", "w") as f:
    json.dump(kinematics_attribution_audit, f, indent=2)

# 6. Aggregate & Preregistered Gate Evaluation
cov_gains = [seed_results[f"seed_{s}"]["overall"]["coverage_gain_pp"] for s in SEEDS]
change_cov_gains = [seed_results[f"seed_{s}"]["gains"]["change_coverage_gain_pp"] for s in SEEDS]
change_ade_gains = [seed_results[f"seed_{s}"]["gains"]["change_ade_gain_pct"] for s in SEEDS]
change_fde_gains = [seed_results[f"seed_{s}"]["gains"]["change_fde_gain_pct"] for s in SEEDS]
overall_ade_harms = [max(0.0, -seed_results[f"seed_{s}"]["overall"]["ade_gain_pct"]) for s in SEEDS]
overall_fde_harms = [max(0.0, -seed_results[f"seed_{s}"]["overall"]["fde_gain_pct"]) for s in SEEDS]
accel_diffs = [seed_results[f"seed_{s}"]["kinematics"]["accel_p95_diff_pct"] for s in SEEDS]
speed_diffs = [seed_results[f"seed_{s}"]["kinematics"]["speed_p95_diff_pct"] for s in SEEDS]
jerk_diffs = [seed_results[f"seed_{s}"]["kinematics"]["jerk_p95_diff_pct"] for s in SEEDS]

mean_cov_gain = float(np.mean(cov_gains))
mean_change_cov_gain = float(np.mean(change_cov_gains))
mean_change_ade_gain = float(np.mean(change_ade_gains))
mean_change_fde_gain = float(np.mean(change_fde_gains))
change_cov_pos_count = sum(1 for g in change_cov_gains if g > 0.0)
all_seeds_no_collapse = all(not seed_results[f"seed_{s}"]["slot_collapse_monitoring"]["is_slot_collapsed"] for s in SEEDS)
overall_harm_acceptable = (max(overall_ade_harms) <= 1.0) and (max(overall_fde_harms) <= 1.0)
motion_guards_all_pass = all(seed_results[f"seed_{s}"]["kinematics"]["motion_guard_pass"] for s in SEEDS)

# Gate check
# Criteria:
# - Change coverage at least 2/3 seeds positive
# - 3-seed mean Change coverage gain >= +1.0 pp
# - Overall minADE/minFDE harm <= 1%
# - Slot does not collapse
# - Motion guard <= 2%
quality_gate_passed = bool(
    change_cov_pos_count >= 2 and
    mean_change_cov_gain >= 1.0 and
    overall_harm_acceptable and
    all_seeds_no_collapse
)

# Determine final action string based on preregistered instructions:
# If quality gates pass but only acceleration guard slightly fails -> BO4_POSITIVE_MOTION_REPAIR_REQUIRED
# If Overall/Stable improves but Change exhibits stable negative gains -> STOP_BO4_CHANGE_NOT_SOLVED
# If only Seed 42 is positive -> STOP_SEED_SPECIFIC
# If >=2/3 seeds Change positive, mean >= +1.0 pp, and all major guards pass -> BO4_STRONG_PASS
if quality_gate_passed and motion_guards_all_pass:
    action_decision = "BO4_STRONG_PASS"
elif quality_gate_passed and not motion_guards_all_pass:
    action_decision = "BO4_POSITIVE_MOTION_REPAIR_REQUIRED"
elif change_cov_pos_count == 1 and change_cov_gains[0] > 0:
    action_decision = "STOP_SEED_SPECIFIC"
elif change_cov_pos_count == 0 or (mean_cov_gain > 0 and mean_change_cov_gain < 0):
    action_decision = "STOP_BO4_CHANGE_NOT_SOLVED"
else:
    # Partial or ambiguous outcome
    action_decision = "REVIEW_REQUIRED_CHANGE_GAIN_BELOW_THRESHOLD"

print(f"\n==========================================")
print(f"   STAGE B FORMAL 3-SEED AUDIT SUMMARY")
print(f"==========================================")
print(f"Change Coverage Gains (42, 43, 44): {[round(x, 2) for x in change_cov_gains]} pp (Positive: {change_cov_pos_count}/3)")
print(f"Mean Change Coverage Gain: {mean_change_cov_gain:+.2f} pp (Threshold: >= +1.0 pp)")
print(f"Mean Change minADE Gain: {mean_change_ade_gain:+.2f}%, minFDE Gain: {mean_change_fde_gain:+.2f}%")
print(f"Mean Overall Coverage Gain: {mean_cov_gain:+.2f} pp")
print(f"Overall Harm Acceptable: {overall_harm_acceptable}")
print(f"All Seeds Slot Collapse Ruled Out: {all_seeds_no_collapse}")
print(f"Motion Guards All Pass: {motion_guards_all_pass} (Acc diffs: {[round(x, 2) for x in accel_diffs]} %)")
print(f"Quality Gate Pass: {quality_gate_passed}")
print(f"FINAL PREREGISTERED ACTION DECISION: {action_decision}")

# 7. Write RESULT.json & DECISION_SUMMARY.json
final_result = {
    "stage": "BEST_OF_4_GENERATOR_3SEED",
    "control": {
        "overall": {
            "coverage_pct": float(static_cov.mean() * 100),
            "covered_count": int(static_cov.sum()),
            "mean_j": float(static_j.mean()),
            "min_ade": float(static_ade.mean()),
            "min_fde": float(static_fde.mean()),
            "effective_good_clusters": int(static_eff)
        },
        "subgroups": control_subgroups,
        "kinematics": static_kin
    },
    "seeds": seed_results,
    "checkpoints": checkpoints,
    "aggregate": {
        "mean_coverage_gain_pp": mean_cov_gain,
        "mean_change_coverage_gain_pp": mean_change_cov_gain,
        "mean_change_ade_gain_pct": mean_change_ade_gain,
        "mean_change_fde_gain_pct": mean_change_fde_gain,
        "change_cov_positive_seeds_count": change_cov_pos_count,
        "overall_harm_acceptable": overall_harm_acceptable,
        "all_seeds_no_collapse": all_seeds_no_collapse,
        "motion_guards_all_pass": motion_guards_all_pass,
        "quality_gate_passed": quality_gate_passed,
        "action_decision": action_decision
    }
}
with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(final_result, f, indent=2)

dec_summary = {
    "stage": "BEST_OF_4_GENERATOR_3SEED",
    "quality_gate_passed": quality_gate_passed,
    "motion_guards_all_pass": motion_guards_all_pass,
    "mean_coverage_gain_pp": mean_cov_gain,
    "mean_change_coverage_gain_pp": mean_change_cov_gain,
    "change_cov_positive_seeds_count": change_cov_pos_count,
    "action_decision": action_decision
}
with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump(dec_summary, f, indent=2)

print("\nSaved all audit files to:", STAGE_DIR)
