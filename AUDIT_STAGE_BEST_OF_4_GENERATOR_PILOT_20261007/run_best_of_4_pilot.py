import sys, time, json, math, hashlib, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_BEST_OF_4_GENERATOR_PILOT_20261007"
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
print(f"Executing Stage A Best-of-4 Generator Pilot on device: {device}")

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

SUBGROUPS = {
    "overall": np.ones(n_ot, dtype=bool),
    "stable": groups_outer == 0,
    "transition": groups_outer == 1,
    "change": groups_outer == 2
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

print(f"CONTROL (Static-K24):")
print(f"  Overall Coverage: {static_cov.mean()*100:.2f}% ({static_cov.sum()}/{n_ot}), Mean J: {static_j.mean():.4f}, minADE: {static_ade.mean():.4f}, minFDE: {static_fde.mean():.4f}")
print(f"  Change Coverage: {control_subgroups['change']['coverage_pct']:.2f}% ({control_subgroups['change']['covered_count']}/{control_subgroups['change']['n']}), Change minADE: {control_subgroups['change']['min_ade']:.4f}, Change minFDE: {control_subgroups['change']['min_fde']:.4f}")

# 3. Model Definition: Best-of-4 Generator
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
        # Semantic diversity initialization
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

# 5. Train Best-of-4 Pilot (Seed 42)
SEED = 42
print(f"\n==========================================")
print(f"   TRAINING BEST-OF-4 GENERATOR PILOT (SEED {SEED})")
print(f"==========================================")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = BestOf4Generator(in_dim=96, hidden_dim=128, num_slots=4, param_dim=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

dataset = TensorDataset(
    torch.from_numpy(tr_p2_norm),
    torch.from_numpy(tr_focal_past),
    torch.from_numpy(tr_future)
)
g_seed = torch.Generator().manual_seed(SEED)
loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=g_seed)

LAMBDA_DIV = 0.05
DELTA_DIV = 0.30 # 30 cm margin
EPOCHS = 25

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
        
        # Batch trajectory decoder
        past_rep = b_past.repeat_interleave(4, dim=0) # (4B, 8, 2)
        raw_rep = raw_params.view(B_cur * 4, 8)       # (4B, 8)
        trajs_flat, _ = _profile(past_rep, raw_rep, "general")
        trajs_4 = trajs_flat.view(B_cur, 4, 12, 2)

        # L_traj per slot
        diff = torch.linalg.vector_norm(trajs_4 - b_gt.unsqueeze(1), dim=-1) # (B, 4, 12)
        ade_slot = diff.mean(dim=-1) # (B, 4)
        fde_slot = diff[:, :, -1]    # (B, 4)
        l_traj = ade_slot + 0.5 * fde_slot # (B, 4)

        best_loss, best_idx = torch.min(l_traj, dim=1) # (B,)
        l_best4 = best_loss.mean()

        # Pairwise diversity loss between 4 slots
        # 6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        div_loss = torch.tensor(0.0, device=device)
        pair_cnt = 0
        for i in range(4):
            for j in range(i + 1, 4):
                d_ij = torch.linalg.vector_norm(trajs_4[:, i] - trajs_4[:, j], dim=-1).mean(dim=-1) # (B,)
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
        
        b_wins = np.bincount(best_idx.detach().cpu().numpy(), minlength=4)
        slot_win_counts += b_wins

    if epoch % 5 == 0 or epoch == EPOCHS:
        win_pcts = slot_win_counts / n_tr * 100
        print(f"Epoch {epoch:2d}/{EPOCHS}: Total Loss={tot_loss/n_tr:.4f}, Best ADE={tot_best_ade/n_tr:.4f}, Div Loss={tot_div_loss/n_tr:.4f} | Winners: [{win_pcts[0]:.1f}%, {win_pcts[1]:.1f}%, {win_pcts[2]:.1f}%, {win_pcts[3]:.1f}%]")

train_time = time.time() - t0
print(f"Training completed in {train_time:.2f}s")

# Save model checkpoint
ckpt_path = MODELS_DIR / "bo4_pilot_seed_42.pth"
torch.save(model.state_dict(), ckpt_path)
h_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
print(f"Checkpoint saved: {ckpt_path} (SHA: {h_sha[:16]}...)")

# 6. Inference on Outer Split
print("\n--- Running Inference on Outer Set ---")
model.eval()
ot_p2_t = torch.from_numpy(ot_p2_norm).to(device)
past_ot_t = torch.from_numpy(focal_past_outer).to(device)

bo4_trajs = np.zeros((n_ot, 4, 12, 2), dtype=np.float32)
bo4_params = np.zeros((n_ot, 4, 8), dtype=np.float32)

with torch.no_grad():
    # Chunked inference to avoid OOM
    CHUNK = 512
    for start_idx in range(0, n_ot, CHUNK):
        end_idx = min(start_idx + CHUNK, n_ot)
        c_p2 = ot_p2_t[start_idx:end_idx]
        c_past = past_ot_t[start_idx:end_idx]
        c_B = len(c_p2)

        c_raw = model(c_p2) # (c_B, 4, 8)
        c_past_rep = c_past.repeat_interleave(4, dim=0)
        c_raw_rep = c_raw.view(c_B * 4, 8)
        c_trajs_flat, c_p_out = _profile(c_past_rep, c_raw_rep, "general")

        bo4_trajs[start_idx:end_idx] = c_trajs_flat.view(c_B, 4, 12, 2).cpu().numpy()
        bo4_params[start_idx:end_idx] = c_p_out.view(c_B, 4, 8).cpu().numpy()

# 7. Evaluate Candidates with Bo4-K24
k24_bo4 = np.concatenate([native_outer, bo4_trajs], axis=1) # (N, 24, 12, 2)
bo4_ade, bo4_fde, bo4_j, bo4_cov, bo4_eff, bo4_err_matrix = unpack_errors(k24_bo4, future_outer)
bo4_kin = motion_summary(k24_bo4)
treatment_subgroups = compute_subgroup_metrics(bo4_ade, bo4_fde, bo4_j, bo4_cov)

# 8. Detailed Slot Collapse & Diversity Monitoring on Outer Set
print("\n--- Detailed Slot Collapse & Diversity Monitoring ---")
# Pairwise trajectory distances between 4 slots
# Shape: (N, 4, 12, 2)
pw_dist = np.zeros((n_ot, 6), dtype=np.float32)
pair_names = ["(0,1)", "(0,2)", "(0,3)", "(1,2)", "(1,3)", "(2,3)"]
p_idx = 0
for i in range(4):
    for j in range(i + 1, 4):
        d_ij = np.linalg.norm(bo4_trajs[:, i] - bo4_trajs[:, j], axis=-1).mean(axis=-1) # (N,)
        pw_dist[:, p_idx] = d_ij
        p_idx += 1

mean_pw_traj_dist = float(pw_dist.mean())
median_pw_traj_dist = float(np.median(pw_dist))
min_pw_traj_dist = float(pw_dist.min())

# Effective slot count (slots with min distance to any other slot >= 0.10 m)
effective_slots_per_sample = []
for n in range(n_ot):
    eff_cnt = 0
    for i in range(4):
        min_d = min(np.linalg.norm(bo4_trajs[n, i] - bo4_trajs[n, j], axis=-1).mean() for j in range(4) if j != i)
        if min_d >= 0.10:
            eff_cnt += 1
    effective_slots_per_sample.append(eff_cnt)
effective_slots_mean = float(np.mean(effective_slots_per_sample))

# Trajectory vs Parameter Diversity
# Normalized param distance:
param_norm = (bo4_params - bo4_params.mean(axis=(0, 1), keepdims=True)) / np.maximum(bo4_params.std(axis=(0, 1), keepdims=True), 1e-6)
pw_pdist = []
for i in range(4):
    for j in range(i + 1, 4):
        d_p = np.linalg.norm(param_norm[:, i] - param_norm[:, j], axis=-1)
        pw_pdist.append(d_p)
mean_pw_param_dist = float(np.mean(pw_pdist))

# Winner Frequency & Subgroup Usage
# Which slot had lowest L_traj on the outer set?
fut_diff = np.linalg.norm(bo4_trajs - future_outer[:, None, :, :], axis=-1) # (N, 4, 12)
outer_ade_slots = fut_diff.mean(axis=-1) # (N, 4)
outer_fde_slots = fut_diff[:, :, -1]    # (N, 4)
outer_ltraj_slots = outer_ade_slots + 0.5 * outer_fde_slots
winning_slot_idx = np.argmin(outer_ltraj_slots, axis=1) # (N,)

slot_winners_overall = (np.bincount(winning_slot_idx, minlength=4) / n_ot * 100).tolist()
slot_winners_stable = (np.bincount(winning_slot_idx[SUBGROUPS["stable"]], minlength=4) / int(SUBGROUPS["stable"].sum()) * 100).tolist()
slot_winners_transition = (np.bincount(winning_slot_idx[SUBGROUPS["transition"]], minlength=4) / int(SUBGROUPS["transition"].sum()) * 100).tolist()
slot_winners_change = (np.bincount(winning_slot_idx[SUBGROUPS["change"]], minlength=4) / int(SUBGROUPS["change"].sum()) * 100).tolist()

max_slot_dominance = float(max(slot_winners_overall))
is_slot_collapsed = bool(mean_pw_traj_dist < 0.10 or max_slot_dominance > 75.0 or effective_slots_mean < 2.0)

print(f"Mean Pairwise Traj Distance: {mean_pw_traj_dist:.4f} m (Median: {median_pw_traj_dist:.4f} m, Min: {min_pw_traj_dist:.4f} m)")
print(f"Mean Effective Slots (>= 0.1m distinct): {effective_slots_mean:.2f} / 4.0")
print(f"Mean Normalized Param Distance: {mean_pw_param_dist:.4f}")
print(f"Winner Frequencies Overall: {[round(x, 1) for x in slot_winners_overall]} %")
print(f"Winner Frequencies Stable:  {[round(x, 1) for x in slot_winners_stable]} %")
print(f"Winner Frequencies Trans:   {[round(x, 1) for x in slot_winners_transition]} %")
print(f"Winner Frequencies Change:  {[round(x, 1) for x in slot_winners_change]} %")
print(f"Max Dominance: {max_slot_dominance:.1f}% -> Slot Collapse: {is_slot_collapsed}")

# 9. Comparison with Static-K24
rescue = int(np.sum((~static_cov) & bo4_cov))
lost = int(np.sum(static_cov & (~bo4_cov)))
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

# 10. Paired Change Sample Case Analysis
change_mask = SUBGROUPS["change"]
change_indices = np.where(change_mask)[0]

static_change_cov = static_cov[change_mask]
bo4_change_cov = bo4_cov[change_mask]

# Static success -> Bo4 failure
lost_change_indices = change_indices[static_change_cov & (~bo4_change_cov)]
# Static failure -> Bo4 success
rescue_change_indices = change_indices[(~static_change_cov) & bo4_change_cov]

print(f"\n--- Change Subgroup Paired Case Analysis (N_change = {len(change_indices)}) ---")
print(f"Static Success -> Bo4 Failure (Lost on Change): {len(lost_change_indices)}")
print(f"Static Failure -> Bo4 Success (Rescue on Change): {len(rescue_change_indices)}")
print(f"Net Change Rescue: {len(rescue_change_indices) - len(lost_change_indices):+d}")

representative_lost_cases = []
for idx in lost_change_indices[:5]:
    sid = int(outer_sample_ids[idx])
    static_min_ade = float(static_ade[idx])
    static_min_fde = float(static_fde[idx])
    bo4_min_ade = float(bo4_ade[idx])
    bo4_min_fde = float(bo4_fde[idx])
    w_slot = int(winning_slot_idx[idx])
    representative_lost_cases.append({
        "sample_id": sid,
        "static_min_ade": static_min_ade,
        "static_min_fde": static_min_fde,
        "bo4_min_ade": bo4_min_ade,
        "bo4_min_fde": bo4_min_fde,
        "winning_bo4_slot": w_slot
    })

representative_rescue_cases = []
for idx in rescue_change_indices[:5]:
    sid = int(outer_sample_ids[idx])
    static_min_ade = float(static_ade[idx])
    static_min_fde = float(static_fde[idx])
    bo4_min_ade = float(bo4_ade[idx])
    bo4_min_fde = float(bo4_fde[idx])
    w_slot = int(winning_slot_idx[idx])
    representative_rescue_cases.append({
        "sample_id": sid,
        "static_min_ade": static_min_ade,
        "static_min_fde": static_min_fde,
        "bo4_min_ade": bo4_min_ade,
        "bo4_min_fde": bo4_min_fde,
        "winning_bo4_slot": w_slot
    })

# 11. Pilot Decision Rules
# If:
# 1. Change Coverage fails to improve (<= 41.71%)
# 2. Change minADE and minFDE both fail to improve
# 3. Or 4 slots visibly collapse
# 4. Or improvements only come from Stable while Change degrades
# -> STOP_GENERATION_BRANCH
pilot_positive_signal = bool(
    change_cov_gain_pp > 0.0 and
    (change_ade_gain_pct > 0.0 or change_fde_gain_pct > 0.0) and
    not is_slot_collapsed and
    ade_gain_pct >= -1.0
)

print(f"\n==========================================")
print(f"   STAGE A PILOT EVALUATION SUMMARY")
print(f"==========================================")
print(f"Overall Coverage: {bo4_cov.mean()*100:.2f}% vs {static_cov.mean()*100:.2f}% [{cov_gain_pp:+.2f} pp, Rescue: {rescue}, Lost: {lost}, Net: {net_rescue:+d}]")
print(f"Mean J: {bo4_j.mean():.4f} [{j_gain_pct:+.2f}%], minADE: {bo4_ade.mean():.4f} [{ade_gain_pct:+.2f}%], minFDE: {bo4_fde.mean():.4f} [{fde_gain_pct:+.2f}%]")
print(f"Change Coverage: {treatment_subgroups['change']['coverage_pct']:.2f}% vs {control_subgroups['change']['coverage_pct']:.2f}% [{change_cov_gain_pp:+.2f} pp]")
print(f"Change minADE Gain: {change_ade_gain_pct:+.2f}%, Change minFDE Gain: {change_fde_gain_pct:+.2f}%")
print(f"Kinematics Guard: Speed diff={speed_p95_diff_pct:+.2f}%, Acc diff={acc_p95_diff_pct:+.2f}% -> {motion_guard_pass}")
print(f"Slot Collapse Check: Collapsed={is_slot_collapsed} (Mean dist: {mean_pw_traj_dist:.3f}m, Max dominance: {max_slot_dominance:.1f}%)")
print(f"Pilot Positive Signal on Change: {pilot_positive_signal}")

# 12. Save Results JSON
out_result = {
    "stage": "BEST_OF_4_GENERATOR_PILOT",
    "seed": SEED,
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
        "kinematics": {
            "speed_p95": float(static_kin["speed_p95"]),
            "accel_mean": float(static_kin["accel_mean"]),
            "accel_p95": float(static_kin["accel_p95"]),
            "jerk_mean": float(static_kin["jerk_mean"]),
            "jerk_p95": float(static_kin["jerk_p95"])
        }
    },
    "treatment": {
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
        }
    },
    "collapse_and_diversity_monitoring": {
        "mean_pairwise_traj_dist_m": mean_pw_traj_dist,
        "median_pairwise_traj_dist_m": median_pw_traj_dist,
        "min_pairwise_traj_dist_m": min_pw_traj_dist,
        "effective_slots_mean": effective_slots_mean,
        "mean_pairwise_param_dist": mean_pw_param_dist,
        "slot_winners_overall_pct": slot_winners_overall,
        "slot_winners_stable_pct": slot_winners_stable,
        "slot_winners_transition_pct": slot_winners_transition,
        "slot_winners_change_pct": slot_winners_change,
        "max_slot_dominance_pct": max_slot_dominance,
        "is_slot_collapsed": is_slot_collapsed
    },
    "change_paired_cases": {
        "n_change": len(change_indices),
        "lost_change_count": len(lost_change_indices),
        "rescue_change_count": len(rescue_change_indices),
        "net_change_rescue": len(rescue_change_indices) - len(lost_change_indices),
        "representative_lost_cases": representative_lost_cases,
        "representative_rescue_cases": representative_rescue_cases
    },
    "checkpoint": {
        "path": str(ckpt_path),
        "sha256": h_sha
    },
    "pilot_decision": {
        "pilot_positive_signal": pilot_positive_signal,
        "recommended_action": "PROCEED_STAGE_B_3SEED" if pilot_positive_signal else "STOP_GENERATION_BRANCH"
    }
}

with open(STAGE_DIR / "RESULT.json", "w") as f:
    json.dump(out_result, f, indent=2)

dec_summary = {
    "stage": "BEST_OF_4_GENERATOR_PILOT",
    "seed": SEED,
    "pilot_positive_signal": pilot_positive_signal,
    "overall_coverage_gain_pp": cov_gain_pp,
    "change_coverage_gain_pp": change_cov_gain_pp,
    "change_ade_gain_pct": change_ade_gain_pct,
    "change_fde_gain_pct": change_fde_gain_pct,
    "net_rescue": net_rescue,
    "net_change_rescue": len(rescue_change_indices) - len(lost_change_indices),
    "is_slot_collapsed": is_slot_collapsed,
    "motion_guard_pass": motion_guard_pass,
    "recommended_action": "PROCEED_STAGE_B_3SEED" if pilot_positive_signal else "STOP_GENERATION_BRANCH"
}
with open(STAGE_DIR / "DECISION_SUMMARY.json", "w") as f:
    json.dump(dec_summary, f, indent=2)

print("\nSaved RESULT.json and DECISION_SUMMARY.json to:", STAGE_DIR)
