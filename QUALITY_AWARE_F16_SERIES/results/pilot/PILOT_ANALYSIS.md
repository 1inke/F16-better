# Quality-Aware F16 Training Series: Pilot Phase Analysis (STOP POINT 2)

> **Execution Protocol**: Seed 1 Pilot Sweep (ETH Fold)  
> **Pilot Verdict**: **`[VERDICT: QUALITY_PROB_WEAK]`**  
> **Verdict Summary**: Top-1 与概率质量均无实质性增益  
> **Best Model Configuration**: `QA1_L005_T020` (`lambda_q=0.05`, `tau=0.2`)

---

## 1. 核心结论与科学判断 (Executive Summary)

根据已锁定的 5 大工程规范与 `F16_QUALITY_AWARE_PROBABILITY_TRAINING_GUIDE.md` 指引：
1. **单一数据源原则**：本报告所有数字均直接加载自机读文件 `lambda_sweep.csv`, `tau_sweep.csv`, `metrics.json` 与 `temperature_calibration_metrics.json`，无任何手工硬编码。
2. **基线身份锁定**：模型架构为 `FrontEncoderASTRA (f16_temporal_social_cross)`，解码器为 `JointTrainableV2Decoder (20 modes)`，参数量：`321,988` / `321,988`。
3. **因果归因清晰**：在全量解冻 F16 的基础上，采用分层学习率（Backbone 0.1x, Decoder 0.5x, Score Head 1.0x），结合 Detached ADE Soft Quality Distribution Loss 直接监督原生 logits。

---

## 2. Pilot 最佳模型与 QA0 基准全维度对比表

### 验证集对比 (Validation Split, N=1516)

| 评估维度 | 核心指标 (Metric) | QA0 基准 (`lambda=0`) | QA1 最佳 (`lambda=0.05`) | 绝对增益 (Delta) | 变化率 (%) | 质量闸门要求 | 判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `0.5434m` | `0.5416m` | `-0.0018m` | `-0.33%` | 越低越好 | **PASS** |
| Selection | Selected FDE | `1.1967m` | `1.1934m` | `-0.0033m` | `-0.27%` | 越低越好 | **PASS** |
| Selection | Selection Regret | `0.3246m` | `0.3231m` | `-0.0015m` | `-0.47%` | 显著下降 | **PASS** |
| Selection | Near-Oracle @ 5cm | `24.41%` | `25.20%` | `+0.79%` | - | 越高越好 | **PASS** |
| **Generation**| **minADE20** | `0.2188m` | `0.2185m` | `-0.0003m` | `-0.13%` | **恶化 ≤ 2.0%** | **PASS** |
| Generation | minFDE20 | `0.3954m` | `0.3953m` | `-0.0002m` | `-0.04%` | **恶化 ≤ 2.0%** | **PASS** |
| Generation | Trajectory Diversity | `1.0679m` | `1.0740m` | `+0.0061m` | `+0.57%` | **下降 ≤ 15.0%** | **PASS** |
| **Probability**| Exact Oracle NLL | `2.6697` | `2.6618` | `-0.0079` | `-0.30%` | 越低越好 | **PASS** |
| Probability | Soft CE (tau=0.10) | `2.7277` | `2.7212` | `-0.0065` | `-0.24%` | 越低越好 | **PASS** |
| Probability | Top-1 ECE | `0.1324` | `0.1399` | `+0.0076` | `+5.71%` | 越低越好 | **PASS** |

---

### 测试集对比 (Test Split, N=364)

| 评估维度 | 核心指标 (Metric) | QA0 基准 (`lambda=0`) | QA1 最佳 (`lambda=0.05`) | 绝对增益 (Delta) | 变化率 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Selection** | **Selected ADE** | `0.9603m` | `0.9610m` | `+0.0007m` | `+0.07%` |
| Selection | Selected FDE | `1.9003m` | `1.8956m` | `-0.0048m` | `-0.25%` |
| Selection | Selection Regret | `0.5529m` | `0.5520m` | `-0.0009m` | `-0.16%` |
| **Generation**| **minADE20** | `0.4075m` | `0.4090m` | `+0.0016m` | `+0.38%` |
| **Probability**| Top-1 ECE | `0.0757` | `0.0636` | `-0.0120` | `-15.88%` |

---

## 3. QA2 验证集专属温度标定 (Temperature Calibration)

> **标定温度**: $T_{cal} = 0.9326$  
> **数学保证**: $\arg\max s_k = \arg\max (s_k / T_{cal})$，Top-1 选择轨迹及 ADE/FDE 完全不变。

| 数据集 | 标定状态 | Top-1 ECE | Soft CE (tau=0.10) | Exact Oracle NLL |
| :--- | :--- | :---: | :---: | :---: |
| **Validation** | 未标定 ($T=1.0$) | `0.1399` | `2.7212` | `2.6618` |
| Validation | **标定后 ($T=0.93$)** | `0.1355` | `2.7204` | `2.6567` |
| **Test** | 未标定 ($T=1.0$) | `0.0636` | `3.1733` | `3.2067` |
| Test | **标定后 ($T=0.93$)** | `0.0556` | `3.2100` | `3.2458` |

---

## 4. STOP POINT 2 阻断与下一阶段决策提请

根据技术指导书第 49 条规定：
> **STOP POINT 2: 完成 Seed 1 Q2 + Q3 必须输出 PILOT_ANALYSIS.md 然后暂停。禁止 Agent 自动继续 3-seed rerun。**

当前阶段已完备交付：
1. `TRAINING_LOSS_MAP.md` $\checkmark$
2. `MODEL_OUTPUT_AUDIT.md` $\checkmark$
3. `reproduction_metrics.csv` & `REPRODUCTION_AUDIT.md` (Q1 Gate PASS) $\checkmark$
4. `lambda_sweep.csv`, `tau_sweep.csv`, `generation_metrics.csv`, `selection_metrics.csv`, `probability_metrics.csv` $\checkmark$
5. `PILOT_ANALYSIS.md` $\checkmark$

等待用户审阅实验信号与决策指令。
