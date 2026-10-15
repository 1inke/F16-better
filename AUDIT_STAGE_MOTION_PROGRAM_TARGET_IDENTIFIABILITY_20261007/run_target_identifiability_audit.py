import sys, time, json, math, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(r"c:\Users\linke\Documents\New project")
PLAN_DIR = Path(r"C:\Users\linke\Downloads\F16_MOTION_PROGRAM_LONG_RUN_20261007")
STAGE_DIR = REPO / "AUDIT_STAGE_MOTION_PROGRAM_TARGET_IDENTIFIABILITY_20261007"
PLAN_STAGE_DIR = PLAN_DIR / "results/MOTION_PROGRAM_TARGET_IDENTIFIABILITY_AUDIT"

STAGE_DIR.mkdir(parents=True, exist_ok=True)
PLAN_STAGE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES"))
from directional_regime_pilot import (
    load_npz, raw_focal_trajectories, maneuver_labels, load_config,
    CLASS_NAMES
)

sys.path.insert(0, str(PLAN_DIR / "src"))
from motion_program import generate_motion_program, default_params, PARAM_NAMES
from oracle_optimizer import _profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Target Identifiability Audit on device: {device}")

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

# Cached Oracle Params on Train
ORACLE_CACHE = REPO / "AUDIT_STAGE_02P_PARAMETER_IDENTIFIABILITY_20261007/train_oracle_params.npz"
all_oracle_params = np.load(ORACLE_CACHE)["params"]

LBL_TO_CLASS = {0: "continue", 1: "decelerate", 2: "stop", 3: "accelerate", 4: "mild_left", 5: "mild_right", 6: "reverse"}
ALL_8_CLASSES = ["stop", "decelerate", "accelerate", "mild_left", "mild_right", "sharp_left", "sharp_right", "reverse"]

tr_idx = splits_sf["grouped_train_index"] # 16337
ot_idx = splits_sf["outer_eval_index"]    # 1561
n_tr = len(tr_idx)
n_ot = len(ot_idx)

groups_outer = train_cache["change_group"][ot_idx]
change_mask_ot = (groups_outer == 2)
stable_mask_ot = (groups_outer == 0)

change_indices_local = np.where(change_mask_ot)[0]
stable_indices_local = np.where(stable_mask_ot)[0]

print(f"Zara Outer: Total = {n_ot}, Change = {len(change_indices_local)}, Stable = {len(stable_indices_local)}")

# ====================================================================
# EXPERIMENT A: Multi-Restart Oracle Parameter Uniqueness Audit
# ====================================================================
print("\n=======================================================")
print("  EXPERIMENT A: Oracle Parameter Uniqueness Audit (16 Restarts)")
print("=======================================================")

def optimize_multirestart(past_batch, future_batch, class_name, num_restarts=16, steps=160, lr=0.06):
    """
    past_batch: (B, 8, 2)
    future_batch: (B, 12, 2)
    Returns:
      restarts_params: (B, num_restarts, 8)
      restarts_trajs: (B, num_restarts, 12, 2)
      restarts_loss: (B, num_restarts)
    """
    B = len(past_batch)
    dev = device
    past_t = torch.as_tensor(past_batch, dtype=torch.float32, device=dev)
    future_t = torch.as_tensor(future_batch, dtype=torch.float32, device=dev)
    
    all_params = []
    all_trajs = []
    all_losses = []
    
    for r in range(num_restarts):
        # Diverse randomized initializations
        raw = nn.Parameter(torch.randn(B, 8, device=dev) * (0.2 + 0.1 * r) + torch.randn(1, 8, device=dev) * 0.5)
        opt = torch.optim.Adam([raw], lr=lr)
        for _ in range(steps):
            traj, params = _profile(past_t, raw, class_name)
            err = torch.linalg.vector_norm(traj - future_t, dim=-1)
            loss = (err.mean(-1) + 0.5 * err[:, -1]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            traj, params = _profile(past_t, raw, class_name)
            err = torch.linalg.vector_norm(traj - future_t, dim=-1)
            q = (err.mean(-1) + 0.5 * err[:, -1]).cpu().numpy()
            all_losses.append(q)
            all_params.append(params.cpu().numpy())
            all_trajs.append(traj.cpu().numpy())
            
    # Stack: (B, num_restarts, ...)
    res_losses = np.stack(all_losses, axis=1) # (B, R)
    res_params = np.stack(all_params, axis=1) # (B, R, 8)
    res_trajs = np.stack(all_trajs, axis=1)   # (B, R, 12, 2)
    return res_losses, res_params, res_trajs

def analyze_uniqueness(sample_indices, subset_name):
    B = len(sample_indices)
    sub_ot_indices = ot_idx[sample_indices]
    sub_past = raw_focal_past[sub_ot_indices]
    sub_future = raw_focal_future[sub_ot_indices]
    sub_lbls = maneuver_lbls[sub_ot_indices]
    
    # We test on each sample's ground truth maneuver class
    classes_to_run = {}
    for i, lbl in enumerate(sub_lbls):
        c_name = LBL_TO_CLASS[lbl]
        classes_to_run.setdefault(c_name, []).append(i)
        
    d_tau_list = []
    d_z_list = []
    d_z_norm_list = []
    n_near_list = []
    param_std_list = [] # (8,)
    onset_theta_corr_list = []
    speed_persist_corr_list = []
    
    t0 = time.time()
    for c_name, local_idx_list in classes_to_run.items():
        sub_B = len(local_idx_list)
        batch_past = sub_past[local_idx_list]
        batch_future = sub_future[local_idx_list]
        
        losses, params, trajs = optimize_multirestart(batch_past, batch_future, c_name, num_restarts=16)
        
        # Analyze each sample in this class
        for b in range(sub_B):
            b_losses = losses[b] # (16,)
            q_best = b_losses.min()
            near_mask = (b_losses <= 1.01 * q_best)
            n_near = near_mask.sum()
            n_near_list.append(int(n_near))
            
            if n_near >= 2:
                near_p = params[b, near_mask] # (K, 8)
                near_t = trajs[b, near_mask]  # (K, 12, 2)
                
                # Pairwise ADE in trajectory space
                # near_t: (K, 12, 2) -> pairwise diff: (K, K, 12)
                diff_t = np.linalg.norm(near_t[:, None] - near_t[None, :], axis=-1).mean(-1) # (K, K)
                triu_idx = np.triu_indices(n_near, k=1)
                pw_ade = diff_t[triu_idx].mean()
                d_tau_list.append(float(pw_ade))
                
                # Pairwise normalized distance in parameter space
                # Normalization scales: s ~ [0.15, 1.85] (range 1.7), theta ~ [-110, 110] (range 220), onset ~ [0, 11] (11), persistence ~ [0.25, 5] (4.75)
                norm_scales = np.array([1.7, 1.7, 1.7, 220.0, 220.0, 220.0, 11.0, 4.75], dtype=np.float32)
                norm_p = near_p / norm_scales[None, :]
                diff_z_norm = np.linalg.norm(norm_p[:, None] - norm_p[None, :], axis=-1)
                pw_z_norm = diff_z_norm[triu_idx].mean()
                d_z_norm_list.append(float(pw_z_norm))
                
                raw_diff_z = np.linalg.norm(near_p[:, None] - near_p[None, :], axis=-1)
                d_z_list.append(float(raw_diff_z[triu_idx].mean()))
                
                param_std_list.append(near_p.std(axis=0))
                
                # Cross-parameter correlation inside near-optimal set
                if n_near >= 4:
                    if near_p[:, 6].std() > 1e-4 and near_p[:, 5].std() > 1e-4:
                        r_ot = np.corrcoef(near_p[:, 6], near_p[:, 5])[0, 1]
                        if not np.isnan(r_ot): onset_theta_corr_list.append(float(r_ot))
                    if near_p[:, :3].mean(1).std() > 1e-4 and near_p[:, 7].std() > 1e-4:
                        r_sp = np.corrcoef(near_p[:, :3].mean(1), near_p[:, 7])[0, 1]
                        if not np.isnan(r_sp): speed_persist_corr_list.append(float(r_sp))
            else:
                d_tau_list.append(0.0)
                d_z_list.append(0.0)
                d_z_norm_list.append(0.0)
                param_std_list.append(np.zeros(8))
                
    elapsed = time.time() - t0
    
    d_tau_arr = np.array(d_tau_list)
    d_z_norm_arr = np.array(d_z_norm_list)
    
    # Degeneracy criterion: trajectory difference < 0.05m but normalized parameter difference > 0.15
    degenerate_mask = (d_tau_arr <= 0.05) & (d_z_norm_arr >= 0.15)
    degen_fraction = float(degenerate_mask.mean())
    
    mean_std = np.mean(param_std_list, axis=0)
    
    print(f"\n[{subset_name}] Evaluated {B} samples in {elapsed:.2f}s:")
    print(f"  Avg Near-Optimal Restarts (within 1% loss): {np.mean(n_near_list):.1f} / 16")
    print(f"  Trajectory Space Dispersion D_tau (Pairwise ADE): {d_tau_arr.mean():.4f} m (Median: {np.median(d_tau_arr):.4f} m)")
    print(f"  Normalized Parameter Dispersion D_z: {d_z_norm_arr.mean():.4f} (Median: {np.median(d_z_norm_arr):.4f})")
    print(f"  Degenerate Samples Fraction (D_tau <= 0.05m & D_z >= 0.15): {degen_fraction*100:.2f}%")
    print(f"  Parameter Std: s1={mean_std[0]:.3f}, s2={mean_std[1]:.3f}, s3={mean_std[2]:.3f}, th1={mean_std[3]:.1f}°, th2={mean_std[4]:.1f}°, th3={mean_std[5]:.1f}°, onset={mean_std[6]:.2f}, persist={mean_std[7]:.2f}")
    if onset_theta_corr_list:
        print(f"  Onset-Theta Correlation: {np.mean(onset_theta_corr_list):+.3f} (positive indicates later onset compensates larger turn)")
    if speed_persist_corr_list:
        print(f"  Speed-Persistence Correlation: {np.mean(speed_persist_corr_list):+.3f}")
        
    return {
        "n_samples": B,
        "mean_near_optimal_solutions": float(np.mean(n_near_list)),
        "mean_trajectory_dispersion_ade_m": float(d_tau_arr.mean()),
        "median_trajectory_dispersion_ade_m": float(np.median(d_tau_arr)),
        "mean_normalized_param_dispersion": float(d_z_norm_arr.mean()),
        "median_normalized_param_dispersion": float(np.median(d_z_norm_arr)),
        "degenerate_solutions_fraction": degen_fraction,
        "mean_parameter_std": {
            "s1": float(mean_std[0]), "s2": float(mean_std[1]), "s3": float(mean_std[2]),
            "theta1_deg": float(mean_std[3]), "theta2_deg": float(mean_std[4]), "theta3_deg": float(mean_std[5]),
            "onset_step": float(mean_std[6]), "persistence": float(mean_std[7])
        },
        "mean_onset_theta_correlation": float(np.mean(onset_theta_corr_list)) if onset_theta_corr_list else 0.0,
        "mean_speed_persistence_correlation": float(np.mean(speed_persist_corr_list)) if speed_persist_corr_list else 0.0
    }

# Run Experiment A on Change (N=211) and Stable (N=211 sample)
exp_a_change = analyze_uniqueness(change_indices_local, "CHANGE SUBSET")
exp_a_stable = analyze_uniqueness(stable_indices_local[:211], "STABLE SUBSET")


# ====================================================================
# EXPERIMENT B: Conditional Multimodality in kNN History Space
# ====================================================================
print("\n=======================================================")
print("  EXPERIMENT B: Past-Conditioned Multimodality Audit (kNN=20)")
print("=======================================================")

def analyze_conditional_multimodality(k=20):
    tr_p2 = feat_p2[tr_idx] # (16337, 96)
    ot_p2 = feat_p2[ot_idx] # (1561, 96)
    
    # Normalize features with train statistics
    mu_p2 = tr_p2.mean(axis=0, keepdims=True)
    std_p2 = tr_p2.std(axis=0, keepdims=True)
    std_p2 = np.where(std_p2 < 1e-6, 1.0, std_p2).astype(np.float32)
    
    tr_norm = (tr_p2 - mu_p2) / std_p2
    ot_norm = (ot_p2 - mu_p2) / std_p2
    
    # Normalize vectors to unit sphere for cosine distance
    tr_unit = tr_norm / np.linalg.norm(tr_norm, axis=-1, keepdims=True)
    ot_unit = ot_norm / np.linalg.norm(ot_norm, axis=-1, keepdims=True)
    
    tr_unit_t = torch.from_numpy(tr_unit).to(device)
    
    tr_lbls = maneuver_lbls[tr_idx] # (16337,)
    tr_params = all_oracle_params[tr_idx] # (16337, 8)
    
    # We analyze each of the 211 Change samples on Zara Outer
    change_ot_unit = ot_unit[change_indices_local] # (211, 96)
    change_ot_unit_t = torch.from_numpy(change_ot_unit).to(device)
    
    # Compute cosine similarity: (211, 16337)
    sim = torch.matmul(change_ot_unit_t, tr_unit_t.T) # (211, 16337)
    topk_vals, topk_indices = torch.topk(sim, k=k, dim=1) # (211, 20)
    
    topk_indices = topk_indices.cpu().numpy()
    topk_vals = topk_vals.cpu().numpy()
    
    n_unique_classes_list = []
    class_entropy_list = []
    theta3_std_list = []
    theta3_span_list = []
    s3_std_list = []
    s3_span_list = []
    bimodal_theta_count = 0
    multimodal_class_count = 0
    
    for i in range(len(change_indices_local)):
        nn_idx = topk_indices[i] # (20,)
        nn_lbls = tr_lbls[nn_idx] # (20,)
        nn_params = tr_params[nn_idx] # (20, 8)
        
        # 1. Class Distribution
        classes_present = np.unique(nn_lbls)
        n_unique_classes_list.append(len(classes_present))
        
        counts = np.bincount(nn_lbls, minlength=len(CLASS_NAMES)).astype(np.float32)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        class_entropy_list.append(entropy)
        
        if len(classes_present) >= 3:
            multimodal_class_count += 1
            
        # 2. Theta Distribution
        theta3_vals = nn_params[:, 5] # theta3 degrees
        theta3_std = theta3_vals.std()
        theta3_span = theta3_vals.max() - theta3_vals.min()
        theta3_std_list.append(theta3_std)
        theta3_span_list.append(theta3_span)
        
        # Bimodal turning: has at least one strong left (> 15 deg) and one strong right (< -15 deg)
        has_left = np.any(theta3_vals > 15.0)
        has_right = np.any(theta3_vals < -15.0)
        has_straight = np.any(np.abs(theta3_vals) <= 10.0)
        if (has_left and has_right) or ((has_left or has_right) and has_straight and theta3_span >= 30.0):
            bimodal_theta_count += 1
            
        # 3. Speed Ratio Distribution
        s3_vals = nn_params[:, 2] # s3
        s3_std_list.append(s3_vals.std())
        s3_span_list.append(s3_vals.max() - s3_vals.min())
        
    n_change = len(change_indices_local)
    mean_sim = float(topk_vals.mean())
    mean_classes = float(np.mean(n_unique_classes_list))
    mean_entropy = float(np.mean(class_entropy_list))
    mean_th_std = float(np.mean(theta3_std_list))
    mean_th_span = float(np.mean(theta3_span_list))
    mean_s3_std = float(np.mean(s3_std_list))
    mean_s3_span = float(np.mean(s3_span_list))
    multimodal_class_pct = float(multimodal_class_count / n_change * 100.0)
    bimodal_theta_pct = float(bimodal_theta_count / n_change * 100.0)
    
    print(f"Evaluated {n_change} Change samples against k={k} nearest past neighbors:")
    print(f"  Mean Top-{k} Cosine Similarity: {mean_sim:.4f}")
    print(f"  Mean Distinct Maneuver Classes in Top-{k}: {mean_classes:.2f} (Max: {np.max(n_unique_classes_list)})")
    print(f"  Mean Class Entropy: {mean_entropy:.3f} bits (Max possible: ~2.81 bits)")
    print(f"  Samples with >= 3 Distinct Maneuver Classes: {multimodal_class_pct:.2f}% ({multimodal_class_count}/{n_change})")
    print(f"  Mean Turning Angle Theta3 Std: {mean_th_std:.1f}° (Mean Span: {mean_th_span:.1f}°)")
    print(f"  Samples with Substantial Angular Branching: {bimodal_theta_pct:.2f}% ({bimodal_theta_count}/{n_change})")
    print(f"  Mean Speed Ratio s3 Std: {mean_s3_std:.3f} (Mean Span: {mean_s3_span:.3f})")
    
    return {
        "n_samples": n_change,
        "k_neighbors": k,
        "mean_topk_similarity": mean_sim,
        "mean_distinct_classes_in_k": mean_classes,
        "mean_class_entropy_bits": mean_entropy,
        "samples_with_ge_3_classes_pct": multimodal_class_pct,
        "mean_theta3_std_deg": mean_th_std,
        "mean_theta3_span_deg": mean_th_span,
        "samples_with_angular_bimodality_pct": bimodal_theta_pct,
        "mean_s3_std": mean_s3_std,
        "mean_s3_span": mean_s3_span
    }

exp_b_results = analyze_conditional_multimodality(k=20)


# ====================================================================
# SYNTHESIS & ROUTING DECISION MATRIX
# ====================================================================
print("\n=======================================================")
print("  DIAGNOSTIC SYNTHESIS & ROUTING DECISION")
print("=======================================================")

degen_pct = exp_a_change["degenerate_solutions_fraction"] * 100.0
bimodal_pct = exp_b_results["samples_with_angular_bimodality_pct"]
multi_class_pct = exp_b_results["samples_with_ge_3_classes_pct"]

# Decision logic
is_parameter_degenerate = bool(degen_pct >= 30.0) # > 30% of solutions have multiple distinct parameter sets producing identical trajectories
is_conditional_multimodal = bool(bimodal_pct >= 35.0 or multi_class_pct >= 40.0) # > 35% of samples exhibit bifurcated futures given similar history

if is_parameter_degenerate and is_conditional_multimodal:
    recommended_route = "BEST_OF_4_TRAJECTORY_SPACE_GENERATOR"
    route_reason = "Both internal parameter degeneracy and true conditional multimodality are prominent. Trajectory-space multi-head generation with min-of-4 loss circumvents parameter ill-posedness while directly preserving diverse modes under K=24 budget."
elif is_parameter_degenerate and not is_conditional_multimodal:
    recommended_route = "CANONICALIZED_OR_TRAJECTORY_SPACE_DETERMINISTIC"
    route_reason = "Parameter non-uniqueness dominates over multimodality; supervision must move to trajectory space."
elif not is_parameter_degenerate and is_conditional_multimodal:
    recommended_route = "PROBABILISTIC_MDN_JOINT_ALLOCATOR"
    route_reason = "Parameters are identifiable but past conditional distribution is multimodal; 3-component MDN required."
else:
    recommended_route = "PREDICTOR_CAPACITY_OR_CONDITIONING_AUDIT"
    route_reason = "Neither parameter degeneracy nor severe multimodality dominates; inspect network architecture."

print(f"Finding 1 (Parameter Degeneracy Rate): {degen_pct:.2f}% (Threshold: 30%) -> {'POSITIVE' if is_parameter_degenerate else 'NEGATIVE'}")
print(f"Finding 2 (Conditional Multimodality Rate): Branching Angular={bimodal_pct:.2f}%, Multi-Class={multi_class_pct:.2f}% (Threshold: 35%) -> {'POSITIVE' if is_conditional_multimodal else 'NEGATIVE'}")
print(f"\nRecommended Next Scientific Route: {recommended_route}")
print(f"Scientific Rationale: {route_reason}")

# Package Results
audit_output = {
    "stage": "MOTION_PROGRAM_TARGET_IDENTIFIABILITY_AUDIT",
    "status": "COMPLETED",
    "experiment_a_parameter_uniqueness": {
        "change_subset": exp_a_change,
        "stable_subset": exp_a_stable,
        "key_finding": {
            "parameter_degeneracy_rate_pct": degen_pct,
            "is_parameter_degenerate": is_parameter_degenerate,
            "mean_onset_theta_correlation": exp_a_change["mean_onset_theta_correlation"]
        }
    },
    "experiment_b_conditional_multimodality": {
        "metrics": exp_b_results,
        "key_finding": {
            "samples_with_angular_bimodality_pct": bimodal_pct,
            "samples_with_ge_3_classes_pct": multi_class_pct,
            "is_conditional_multimodal": is_conditional_multimodal
        }
    },
    "routing_decision": {
        "selected_route": recommended_route,
        "rationale": route_reason,
        "is_parameter_degenerate": is_parameter_degenerate,
        "is_conditional_multimodal": is_conditional_multimodal,
        "budget_constraint": "K=24 (Native 20 frozen + 4 learned program slots)"
    }
}

# Write results to stage and plan directories
(STAGE_DIR / "RESULT.json").write_text(json.dumps(audit_output, indent=2, ensure_ascii=False), encoding="utf-8")
(PLAN_STAGE_DIR / "RESULT.json").write_text(json.dumps(audit_output, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nAudit results successfully saved to:")
print(f" - {STAGE_DIR / 'RESULT.json'}")
print(f" - {PLAN_STAGE_DIR / 'RESULT.json'}")
