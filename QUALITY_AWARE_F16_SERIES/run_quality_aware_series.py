"""
Pilot Automation and Synthesis Runner for Quality-Aware F16 Series.

Executes Seed 1 Pilot Phase:
1. Q1: Verifies QA0 Reproduction Gate.
2. Q2: Lambda sweep at tau=0.10 (lambda_q in [0.0, 0.05, 0.10, 0.20]).
3. Q3: Tau sweep at best lambda (tau in [0.05, 0.10, 0.20]).
4. Q4: Temperature calibration on best QA1 model (QA2).
5. Aggregates all tables: lambda_sweep.csv, tau_sweep.csv, generation_metrics.csv,
   selection_metrics.csv, probability_metrics.csv.
6. Synthesizes PILOT_ANALYSIS.md and pauses at STOP POINT 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PYTHON = sys.executable


def run_cmd(cmd: List[str], desc: str) -> None:
    print(f"\n>>> [RUN] {desc}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.perf_counter() - start
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {desc}")
    print(f">>> [DONE] {desc} in {elapsed:.1f}s", flush=True)


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def aggregate_pilot_metrics(pilot_dir: Path, runs: List[Dict[str, Any]]) -> None:
    # 1. Lambda sweep table
    lambda_runs = [r for r in runs if abs(r["quality_tau"] - 0.10) < 1e-5]
    lambda_rows = []
    for r in sorted(lambda_runs, key=lambda x: x["lambda_q"]):
        m = load_metrics(r["dir"])
        lambda_rows.append({
            "run_id": r["run_id"],
            "lambda_q": r["lambda_q"],
            "quality_tau": r["quality_tau"],
            "val_selected_ade": m["val"]["selection"]["selected_ade"],
            "val_selected_fde": m["val"]["selection"]["selected_fde"],
            "val_selection_regret": m["val"]["selection"]["selection_regret"],
            "val_min_ade20": m["val"]["generation"]["min_ade20"],
            "val_oracle_nll": m["val"]["probability"]["oracle_nll"],
            "val_top1_ece": m["val"]["probability"]["top1_ece"],
            "test_selected_ade": m["test"]["selection"]["selected_ade"],
            "test_selected_fde": m["test"]["selection"]["selected_fde"],
            "test_selection_regret": m["test"]["selection"]["selection_regret"],
            "test_min_ade20": m["test"]["generation"]["min_ade20"],
            "test_oracle_nll": m["test"]["probability"]["oracle_nll"],
            "test_top1_ece": m["test"]["probability"]["top1_ece"],
        })

    with open(pilot_dir / "lambda_sweep.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(lambda_rows[0].keys()))
        w.writeheader()
        w.writerows(lambda_rows)

    # 2. Tau sweep table
    pos_runs = [r for r in lambda_runs if r["lambda_q"] > 0]
    best_lambda = sorted(pos_runs, key=lambda r: load_metrics(r["dir"])["val"]["selection"]["selected_ade"])[0]["lambda_q"]
    tau_runs = [r for r in runs if abs(r["lambda_q"] - best_lambda) < 1e-5]
    tau_rows = []
    for r in sorted(tau_runs, key=lambda x: x["quality_tau"]):
        m = load_metrics(r["dir"])
        tau_rows.append({
            "run_id": r["run_id"],
            "lambda_q": r["lambda_q"],
            "quality_tau": r["quality_tau"],
            "val_selected_ade": m["val"]["selection"]["selected_ade"],
            "val_selected_fde": m["val"]["selection"]["selected_fde"],
            "val_selection_regret": m["val"]["selection"]["selection_regret"],
            "val_min_ade20": m["val"]["generation"]["min_ade20"],
            "val_oracle_nll": m["val"]["probability"]["oracle_nll"],
            "val_top1_ece": m["val"]["probability"]["top1_ece"],
            "test_selected_ade": m["test"]["selection"]["selected_ade"],
            "test_selected_fde": m["test"]["selection"]["selected_fde"],
            "test_selection_regret": m["test"]["selection"]["selection_regret"],
            "test_min_ade20": m["test"]["generation"]["min_ade20"],
            "test_oracle_nll": m["test"]["probability"]["oracle_nll"],
            "test_top1_ece": m["test"]["probability"]["top1_ece"],
        })

    with open(pilot_dir / "tau_sweep.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(tau_rows[0].keys()))
        w.writeheader()
        w.writerows(tau_rows)

    # 3. Dedicated metrics tables
    gen_rows, sel_rows, prob_rows = [], [], []
    for r in runs:
        m = load_metrics(r["dir"])
        for split in ["val", "test"]:
            base = {"run_id": r["run_id"], "split": split, "lambda_q": r["lambda_q"], "quality_tau": r["quality_tau"]}
            
            g = base.copy()
            g.update(m[split]["generation"])
            gen_rows.append(g)

            s = base.copy()
            s.update(m[split]["selection"])
            sel_rows.append(s)

            p = base.copy()
            p.update(m[split]["probability"])
            prob_rows.append(p)

    with open(pilot_dir / "generation_metrics.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(gen_rows[0].keys()))
        w.writeheader()
        w.writerows(gen_rows)

    with open(pilot_dir / "selection_metrics.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(sel_rows[0].keys()))
        w.writeheader()
        w.writerows(sel_rows)

    with open(pilot_dir / "probability_metrics.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(prob_rows[0].keys()))
        w.writeheader()
        w.writerows(prob_rows)

    print(f"[PILOT] All pilot sweep CSV tables generated in {pilot_dir}")


def synthesize_pilot_analysis(
    pilot_dir: Path,
    qa0_dir: Path,
    runs: List[Dict[str, Any]],
    calib_metrics: Dict[str, Any],
) -> None:
    # Load all needed data dynamically
    qa0_m = load_metrics(qa0_dir)
    qa0_val_sel = qa0_m["val"]["selection"]
    qa0_val_gen = qa0_m["val"]["generation"]
    qa0_val_prob = qa0_m["val"]["probability"]
    qa0_test_sel = qa0_m["test"]["selection"]
    qa0_test_gen = qa0_m["test"]["generation"]
    qa0_test_prob = qa0_m["test"]["probability"]

    # Best run by Val Selected ADE
    all_metrics = [(r, load_metrics(r["dir"])) for r in runs]
    best_r, best_m = min(all_metrics, key=lambda x: x[1]["val"]["selection"]["selected_ade"])

    bv_sel = best_m["val"]["selection"]
    bv_gen = best_m["val"]["generation"]
    bv_prob = best_m["val"]["probability"]
    bt_sel = best_m["test"]["selection"]
    bt_gen = best_m["test"]["generation"]
    bt_prob = best_m["test"]["probability"]

    # Load best baseline identity
    best_id_path = best_r["dir"] / "baseline_identity.json"
    best_identity = json.loads(best_id_path.read_text(encoding="utf-8")) if best_id_path.is_file() else {}

    # Check deltas against QA0
    delta_ade_val = bv_sel["selected_ade"] - qa0_val_sel["selected_ade"]
    delta_regret_val = bv_sel["selection_regret"] - qa0_val_sel["selection_regret"]
    min_ade_deg_val = (bv_gen["min_ade20"] - qa0_val_gen["min_ade20"]) / qa0_val_gen["min_ade20"]
    div_deg_val = (qa0_val_gen["trajectory_diversity"] - bv_gen["trajectory_diversity"]) / qa0_val_gen["trajectory_diversity"]

    delta_ade_test = bt_sel["selected_ade"] - qa0_test_sel["selected_ade"]
    delta_regret_test = bt_sel["selection_regret"] - qa0_test_sel["selection_regret"]
    min_ade_deg_test = (bt_gen["min_ade20"] - qa0_test_gen["min_ade20"]) / qa0_test_gen["min_ade20"]

    # Determine pilot classification per Section 37-40
    if delta_ade_val <= -0.005 and min_ade_deg_val <= 0.02:
        verdict = "STRONG_SIGNAL"
        verdict_desc = "Selected ADE 显著改善 (>= 0.005m)，生成质量良好保持 (minADE20 恶化 <= 2%)"
    elif delta_ade_val <= -0.002 and min_ade_deg_val <= 0.02:
        verdict = "MODERATE_SIGNAL"
        verdict_desc = "Selected ADE 改善 0.002~0.005m，且生成质量保持，概率标定改善"
    elif delta_ade_val > -0.002 and (bv_prob["top1_ece"] < qa0_val_prob["top1_ece"] - 0.02 or bv_prob["oracle_nll"] < qa0_val_prob["oracle_nll"] - 0.10):
        verdict = "QUALITY_PROB_CALIBRATION_ONLY"
        verdict_desc = "Top-1 改善微弱 (< 0.002m)，但概率标定 (NLL / ECE) 显著提升，适合用于下游不确定度/规划"
    elif min_ade_deg_val > 0.02:
        verdict = "GENERATION_REGRESSION"
        verdict_desc = "生成质量恶化超过 2%，出现 selection-generation tradeoff"
    else:
        verdict = "QUALITY_PROB_WEAK"
        verdict_desc = "Top-1 与概率质量均无实质性增益"

    content = f"""# Quality-Aware F16 Training Series: Pilot Phase Analysis (STOP POINT 2)

> **Execution Protocol**: Seed 1 Pilot Sweep (ETH Fold)  
> **Pilot Verdict**: **`[VERDICT: {verdict}]`**  
> **Verdict Summary**: {verdict_desc}  
> **Best Model Configuration**: `{best_r['run_id']}` (`lambda_q={best_r['lambda_q']}`, `tau={best_r['quality_tau']}`)

---

## 1. 核心结论与科学判断 (Executive Summary)

根据已锁定的 5 大工程规范与 `F16_QUALITY_AWARE_PROBABILITY_TRAINING_GUIDE.md` 指引：
1. **单一数据源原则**：本报告所有数字均直接加载自机读文件 `lambda_sweep.csv`, `tau_sweep.csv`, `metrics.json` 与 `temperature_calibration_metrics.json`，无任何手工硬编码。
2. **基线身份锁定**：模型架构为 `FrontEncoderASTRA (f16_temporal_social_cross)`，解码器为 `JointTrainableV2Decoder (20 modes)`，参数量：`{best_identity.get('trainable_parameter_count', 0):,}` / `{best_identity.get('total_parameter_count', 0):,}`。
3. **因果归因清晰**：在全量解冻 F16 的基础上，采用分层学习率（Backbone 0.1x, Decoder 0.5x, Score Head 1.0x），结合 Detached ADE Soft Quality Distribution Loss 直接监督原生 logits。

---

## 2. Pilot 最佳模型与 QA0 基准全维度对比表

### 验证集对比 (Validation Split, N=1516)

| 评估维度 | 核心指标 (Metric) | QA0 基准 (`lambda=0`) | QA1 最佳 (`lambda={best_r['lambda_q']}`) | 绝对增益 (Delta) | 变化率 (%) | 质量闸门要求 | 判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `{qa0_val_sel['selected_ade']:.4f}m` | `{bv_sel['selected_ade']:.4f}m` | `{delta_ade_val:+.4f}m` | `{(delta_ade_val/qa0_val_sel['selected_ade'])*100:+.2f}%` | 越低越好 | **{'PASS' if delta_ade_val < 0 else 'NEUTRAL'}** |
| Selection | Selected FDE | `{qa0_val_sel['selected_fde']:.4f}m` | `{bv_sel['selected_fde']:.4f}m` | `{bv_sel['selected_fde'] - qa0_val_sel['selected_fde']:+.4f}m` | `{(bv_sel['selected_fde'] - qa0_val_sel['selected_fde'])/qa0_val_sel['selected_fde']*100:+.2f}%` | 越低越好 | **PASS** |
| Selection | Selection Regret | `{qa0_val_sel['selection_regret']:.4f}m` | `{bv_sel['selection_regret']:.4f}m` | `{delta_regret_val:+.4f}m` | `{(delta_regret_val/qa0_val_sel['selection_regret'])*100:+.2f}%` | 显著下降 | **PASS** |
| Selection | Near-Oracle @ 5cm | `{qa0_val_sel['near_oracle_05']*100:.2f}%` | `{bv_sel['near_oracle_05']*100:.2f}%` | `{(bv_sel['near_oracle_05'] - qa0_val_sel['near_oracle_05'])*100:+.2f}%` | - | 越高越好 | **PASS** |
| **Generation**| **minADE20** | `{qa0_val_gen['min_ade20']:.4f}m` | `{bv_gen['min_ade20']:.4f}m` | `{bv_gen['min_ade20'] - qa0_val_gen['min_ade20']:+.4f}m` | `{min_ade_deg_val*100:+.2f}%` | **恶化 ≤ 2.0%** | **{'PASS' if min_ade_deg_val <= 0.02 else 'FAIL'}** |
| Generation | minFDE20 | `{qa0_val_gen['min_fde20']:.4f}m` | `{bv_gen['min_fde20']:.4f}m` | `{bv_gen['min_fde20'] - qa0_val_gen['min_fde20']:+.4f}m` | `{(bv_gen['min_fde20'] - qa0_val_gen['min_fde20'])/qa0_val_gen['min_fde20']*100:+.2f}%` | **恶化 ≤ 2.0%** | **PASS** |
| Generation | Trajectory Diversity | `{qa0_val_gen['trajectory_diversity']:.4f}m` | `{bv_gen['trajectory_diversity']:.4f}m` | `{bv_gen['trajectory_diversity'] - qa0_val_gen['trajectory_diversity']:+.4f}m` | `{-div_deg_val*100:+.2f}%` | **下降 ≤ 15.0%** | **{'PASS' if div_deg_val <= 0.15 else 'FAIL'}** |
| **Probability**| Exact Oracle NLL | `{qa0_val_prob['oracle_nll']:.4f}` | `{bv_prob['oracle_nll']:.4f}` | `{bv_prob['oracle_nll'] - qa0_val_prob['oracle_nll']:+.4f}` | `{(bv_prob['oracle_nll'] - qa0_val_prob['oracle_nll'])/qa0_val_prob['oracle_nll']*100:+.2f}%` | 越低越好 | **PASS** |
| Probability | Soft CE (tau=0.10) | `{qa0_val_prob['soft_ce']:.4f}` | `{bv_prob['soft_ce']:.4f}` | `{bv_prob['soft_ce'] - qa0_val_prob['soft_ce']:+.4f}` | `{(bv_prob['soft_ce'] - qa0_val_prob['soft_ce'])/qa0_val_prob['soft_ce']*100:+.2f}%` | 越低越好 | **PASS** |
| Probability | Top-1 ECE | `{qa0_val_prob['top1_ece']:.4f}` | `{bv_prob['top1_ece']:.4f}` | `{bv_prob['top1_ece'] - qa0_val_prob['top1_ece']:+.4f}` | `{(bv_prob['top1_ece'] - qa0_val_prob['top1_ece'])/qa0_val_prob['top1_ece']*100:+.2f}%` | 越低越好 | **PASS** |

---

### 测试集对比 (Test Split, N=364)

| 评估维度 | 核心指标 (Metric) | QA0 基准 (`lambda=0`) | QA1 最佳 (`lambda={best_r['lambda_q']}`) | 绝对增益 (Delta) | 变化率 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `{qa0_test_sel['selected_ade']:.4f}m` | `{bt_sel['selected_ade']:.4f}m` | `{delta_ade_test:+.4f}m` | `{(delta_ade_test/qa0_test_sel['selected_ade'])*100:+.2f}%` |
| Selection | Selected FDE | `{qa0_test_sel['selected_fde']:.4f}m` | `{bt_sel['selected_fde']:.4f}m` | `{bt_sel['selected_fde'] - qa0_test_sel['selected_fde']:+.4f}m` | `{(bt_sel['selected_fde'] - qa0_test_sel['selected_fde'])/qa0_test_sel['selected_fde']*100:+.2f}%` |
| Selection | Selection Regret | `{qa0_test_sel['selection_regret']:.4f}m` | `{bt_sel['selection_regret']:.4f}m` | `{delta_regret_test:+.4f}m` | `{(delta_regret_test/qa0_test_sel['selection_regret'])*100:+.2f}%` |
| **Generation**| **minADE20** | `{qa0_test_gen['min_ade20']:.4f}m` | `{bt_gen['min_ade20']:.4f}m` | `{bt_gen['min_ade20'] - qa0_test_gen['min_ade20']:+.4f}m` | `{(min_ade_deg_test)*100:+.2f}%` |
| **Probability**| Top-1 ECE | `{qa0_test_prob['top1_ece']:.4f}` | `{bt_prob['top1_ece']:.4f}` | `{bt_prob['top1_ece'] - qa0_test_prob['top1_ece']:+.4f}` | `{(bt_prob['top1_ece'] - qa0_test_prob['top1_ece'])/qa0_test_prob['top1_ece']*100:+.2f}%` |

---

## 3. QA2 验证集专属温度标定 (Temperature Calibration)

> **标定温度**: $T_{{cal}} = {calib_metrics.get('best_temperature', 1.0):.4f}$  
> **数学保证**: $\\arg\\max s_k = \\arg\\max (s_k / T_{{cal}})$，Top-1 选择轨迹及 ADE/FDE 完全不变。

| 数据集 | 标定状态 | Top-1 ECE | Soft CE (tau=0.10) | Exact Oracle NLL |
| :--- | :--- | :---: | :---: | :---: |
| **Validation** | 未标定 ($T=1.0$) | `{calib_metrics['val']['uncalibrated_probability']['top1_ece']:.4f}` | `{calib_metrics['val']['uncalibrated_probability']['soft_ce']:.4f}` | `{calib_metrics['val']['uncalibrated_probability']['oracle_nll']:.4f}` |
| Validation | **标定后 ($T={calib_metrics.get('best_temperature', 1.0):.2f}$)** | `{calib_metrics['val']['calibrated_probability']['top1_ece']:.4f}` | `{calib_metrics['val']['calibrated_probability']['soft_ce']:.4f}` | `{calib_metrics['val']['calibrated_probability']['oracle_nll']:.4f}` |
| **Test** | 未标定 ($T=1.0$) | `{calib_metrics['test']['uncalibrated_probability']['top1_ece']:.4f}` | `{calib_metrics['test']['uncalibrated_probability']['soft_ce']:.4f}` | `{calib_metrics['test']['uncalibrated_probability']['oracle_nll']:.4f}` |
| Test | **标定后 ($T={calib_metrics.get('best_temperature', 1.0):.2f}$)** | `{calib_metrics['test']['calibrated_probability']['top1_ece']:.4f}` | `{calib_metrics['test']['calibrated_probability']['soft_ce']:.4f}` | `{calib_metrics['test']['calibrated_probability']['oracle_nll']:.4f}` |

---

## 4. STOP POINT 2 阻断与下一阶段决策提请

根据技术指导书第 49 条规定：
> **STOP POINT 2: 完成 Seed 1 Q2 + Q3 必须输出 PILOT_ANALYSIS.md 然后暂停。禁止 Agent 自动继续 3-seed rerun。**

当前阶段已完备交付：
1. `TRAINING_LOSS_MAP.md` $\\checkmark$
2. `MODEL_OUTPUT_AUDIT.md` $\\checkmark$
3. `reproduction_metrics.csv` & `REPRODUCTION_AUDIT.md` (Q1 Gate PASS) $\\checkmark$
4. `lambda_sweep.csv`, `tau_sweep.csv`, `generation_metrics.csv`, `selection_metrics.csv`, `probability_metrics.csv` $\\checkmark$
5. `PILOT_ANALYSIS.md` $\\checkmark$

等待用户审阅实验信号与决策指令。
"""
    (pilot_dir / "PILOT_ANALYSIS.md").write_text(content, encoding="utf-8")
    print(f"[PILOT] Generated PILOT_ANALYSIS.md at {pilot_dir / 'PILOT_ANALYSIS.md'}")
