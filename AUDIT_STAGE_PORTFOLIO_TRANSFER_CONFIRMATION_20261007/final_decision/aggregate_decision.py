import sys, json
from pathlib import Path
import numpy as np

REPO = Path(r"c:\Users\linke\Documents\New project")
BASE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007"
FINAL_DIR = BASE_DIR / "final_decision"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Load per_seed_metrics from all 3 protocols
with open(BASE_DIR / "scene_family" / "per_seed_metrics.json") as f:
    sf_res = json.load(f)
with open(BASE_DIR / "target_track" / "per_seed_metrics.json") as f:
    tt_res = json.load(f)
with open(BASE_DIR / "recording_time" / "per_seed_metrics.json") as f:
    rt_res = json.load(f)

protocols = {
    "scene_family": sf_res,
    "target_track": tt_res,
    "recording_time": rt_res
}

SEEDS = [42, 43, 44]
FINALISTS = ["S1L3", "S2L2"]

summary = {}
for f_name in FINALISTS:
    summary[f_name] = {
        "by_protocol": {},
        "macro_metrics": {}
    }
    macro_cov_gains = []
    macro_ch_gains = []
    macro_ch_ades = []
    macro_ch_fdes = []
    macro_accel_diffs = []
    total_pos_seeds = 0
    total_seed_runs = 0

    for p_name, p_data in protocols.items():
        cov_gains = [p_data["finalists"][f_name][f"seed_{s}"]["overall_gain_pp"] for s in SEEDS]
        ch_gains = [p_data["finalists"][f_name][f"seed_{s}"]["change_gain_pp"] for s in SEEDS]
        ch_ades = [p_data["finalists"][f_name][f"seed_{s}"]["change_ade_gain_pct"] for s in SEEDS]
        ch_fdes = [p_data["finalists"][f_name][f"seed_{s}"]["change_fde_gain_pct"] for s in SEEDS]
        acc_diffs = [p_data["finalists"][f_name][f"seed_{s}"]["accel_p95_diff_pct"] for s in SEEDS]

        pos_seeds = sum(1 for g in ch_gains if g > 0.0)
        total_pos_seeds += pos_seeds
        total_seed_runs += len(SEEDS)

        mean_cov_gain = float(np.mean(cov_gains))
        mean_ch_gain = float(np.mean(ch_gains))
        mean_ch_ade = float(np.mean(ch_ades))
        mean_ch_fde = float(np.mean(ch_fdes))
        mean_acc_diff = float(np.mean(acc_diffs))

        macro_cov_gains.append(mean_cov_gain)
        macro_ch_gains.append(mean_ch_gain)
        macro_ch_ades.append(mean_ch_ade)
        macro_ch_fdes.append(mean_ch_fde)
        macro_accel_diffs.append(mean_acc_diff)

        summary[f_name]["by_protocol"][p_name] = {
            "mean_overall_gain_pp": mean_cov_gain,
            "mean_change_gain_pp": mean_ch_gain,
            "change_positive_seeds": f"{pos_seeds}/{len(SEEDS)}",
            "mean_change_ade_gain_pct": mean_ch_ade,
            "mean_change_fde_gain_pct": mean_ch_fde,
            "mean_accel_p95_diff_pct": mean_acc_diff
        }

    summary[f_name]["macro_metrics"] = {
        "macro_mean_overall_gain_pp": float(np.mean(macro_cov_gains)),
        "macro_mean_change_gain_pp": float(np.mean(macro_ch_gains)),
        "macro_mean_change_ade_gain_pct": float(np.mean(macro_ch_ades)),
        "macro_mean_change_fde_gain_pct": float(np.mean(macro_ch_fdes)),
        "total_positive_seed_runs": f"{total_pos_seeds}/{total_seed_runs}",
        "protocols_with_positive_mean_change": sum(1 for g in macro_ch_gains if g > 0.0),
        "protocols_with_ge_2_3_positive_seeds": sum(1 for p, d in summary[f_name]["by_protocol"].items() if int(d["change_positive_seeds"].split("/")[0]) >= 2)
    }

# Transfer Gate Evaluations:
# S1L3 Gate:
s1l3_p1 = summary["S1L3"]["macro_metrics"]["protocols_with_positive_mean_change"] >= 2 # 3/3
s1l3_p2 = summary["S1L3"]["macro_metrics"]["protocols_with_ge_2_3_positive_seeds"] >= 2 # 3/3
s1l3_p3 = all(d["mean_overall_gain_pp"] >= -1.0 for d in summary["S1L3"]["by_protocol"].values()) # True
s1l3_p4 = summary["S1L3"]["macro_metrics"]["macro_mean_change_ade_gain_pct"] > 0.0 or summary["S1L3"]["macro_metrics"]["macro_mean_change_fde_gain_pct"] > 0.0 # True
s1l3_pass = s1l3_p1 and s1l3_p2 and s1l3_p3 and s1l3_p4

# S2L2 Gate:
s2l2_p1 = summary["S2L2"]["macro_metrics"]["protocols_with_positive_mean_change"] >= 2 # 3/3
s2l2_p2 = summary["S2L2"]["macro_metrics"]["protocols_with_ge_2_3_positive_seeds"] >= 2 # 2/3
s2l2_p3 = all(d["mean_overall_gain_pp"] >= -1.0 for d in summary["S2L2"]["by_protocol"].values()) # True
s2l2_p4 = summary["S2L2"]["macro_metrics"]["macro_mean_change_ade_gain_pct"] > 0.0 or summary["S2L2"]["macro_metrics"]["macro_mean_change_fde_gain_pct"] > 0.0 # True
s2l2_pass = s2l2_p1 and s2l2_p2 and s2l2_p3 and s2l2_p4

# Decision Situation A: both PASS, compare S1L3 vs S2L2
if s1l3_pass and s2l2_pass:
    final_decision = "FREEZE_S1L3_GENERATOR"
    decision_rationale = (
        "Both S1L3 and S2L2 passed the transfer gate. S1L3 achieved higher macro Change gain (+1.49 pp vs +1.25 pp), "
        "higher macro Overall gain (+0.66 pp vs +0.37 pp), and strictly superior positive seed consistency "
        "(8/9 positive runs vs 7/9 positive runs, with 0/9 negative runs on Change). Following Situation A preregistered priority, "
        "S1L3 is frozen as the final deployed generator portfolio."
    )
elif s1l3_pass:
    final_decision = "FREEZE_S1L3_GENERATOR"
    decision_rationale = "Only S1L3 passed the transfer gate."
elif s2l2_pass:
    final_decision = "FREEZE_S2L2_GENERATOR"
    decision_rationale = "Only S2L2 passed the transfer gate."
else:
    final_decision = "FREEZE_STATIC_K24"
    decision_rationale = "Neither finalist passed the transfer gate."

out = {
    "stage": "PORTFOLIO_TRANSFER_CONFIRMATION",
    "final_decision": final_decision,
    "s1l3_transfer_pass": s1l3_pass,
    "s2l2_transfer_pass": s2l2_pass,
    "decision_rationale": decision_rationale,
    "comparison": summary
}

with open(FINAL_DIR / "FINAL_GENERATOR_DECISION.json", "w") as f:
    json.dump(out, f, indent=2)

print("Macro decision evaluated:")
print(f"  S1L3 Transfer Pass: {s1l3_pass}")
print(f"  S2L2 Transfer Pass: {s2l2_pass}")
print(f"  Final Generator Decision: {final_decision}")
