# F16 Confidence Objective Refinement Series V2

> **Repository**: `1inke/pedestrian-trajectory-research`  
> **Status**: Stage 1 Completed (`P0-A PASS`, `P0-B PASS`, `P0-C PASS`, `P1 MEDIUM REDUNDANCY`). Paused at **STOP POINT 1**.

---

## 1. 核心定位与原则

本系列针对 F16 模态置信度监督进行第二代严格控制实验（Confidence Objective Refinement V2），修正第一代实验中发现的核心问题：
1. **完全复用 Native Trainer**：正式实验全部通过 `FRONT_ENCODER_SERIES/train_front_variant.py` + `build_loss()` 接口运行，不重写第二套 Trainer，严守 EXP-173 原始训练协议（单学习率 0.00025、Grad Clip 5.0、EMA 0.995、`oracle_combo` checkpointing）。
2. **纯 Objective Replacement**：保持原生总分类损失权重 $0.5$ 不变，不再盲目叠加辅助 Soft CE 损失，隔离置信度目标形式（Objective Form）。
3. **分阶段硬性决策闸门（Decision Gates）**：严格遵循 P0 协议与评测等价验证、P1 目标冗余审计、P2 替换试点、P3 多种子泛化验证阶段。

---

## 2. 阶段审计与交付物体系

```text
QUALITY_AWARE_F16_SERIES_V2/
├── README.md
├── PROTOCOL_LOCK.md                     <-- P0-A 严格复现验证报告 [STATUS: PASS]
├── EVAL_EQUIVALENCE_AUDIT.md            <-- P0-B 推理评测调用等价报告 [STATUS: PASS]
├── LOSS_EQUIVALENCE_AUDIT.md            <-- P0-C 损失与全图梯度等价报告 [STATUS: PASS]
├── TARGET_REDUNDANCY_REPORT.md          <-- P1 目标与梯度冗余度审计报告 [VERDICT: MEDIUM REDUNDANCY]
│
├── confidence_objective_adapter.py      <-- 注入 build_loss 的干净 Adapter
├── confidence_objective_loss.py         <-- 模块化置信度目标 ModeQueryLoss
├── run_confidence_v2_series.py          <-- 离散阶段调度器 (禁止 --stage all)
├── run_protocol_reproduction.py         <-- P0-A 协议执行脚本
├── audit_eval_equivalence.py            <-- P0-B 推理等价执行脚本
├── audit_loss_equivalence.py            <-- P0-C 损失等价执行脚本
├── audit_target_redundancy.py           <-- P1 冗余度分析执行脚本
│
├── common/
│   ├── confidence_targets.py            <-- 原生/物理/ADE 分布目标构建
│   ├── redundancy_metrics.py            <-- JS/KL 散度、Jaccard、有效模态数
│   ├── gradient_audit.py                <-- 解析 Logits 梯度与余弦相似度
│   └── identity.py                      <-- 协议 SHA256 与差异比对工具
│
└── results/
    ├── state.json                       <-- 全局状态机跟踪
    ├── p0_protocol/                     <-- P0-A 复现数据与指标表
    ├── p0_eval/                         <-- P0-B 推理等价数值记录
    ├── p0_loss/                         <-- P0-C 损失与梯度偏差记录
    └── p1_redundancy/                   <-- P1 全样本与分层场景冗余数据
```

---

## 3. 当前运行状态 (`results/state.json`)

```json
{
  "p0_protocol": "PASS",
  "p0_eval": "PASS",
  "p0_loss": "PASS",
  "p1_redundancy": "MEDIUM REDUNDANCY",
  "p2_pilot": "BLOCKED",
  "p3_multiseed": "BLOCKED"
}
```

- **P0-A (Protocol)**: `PASS`，C0 在 ETH Seed 1 上 Selected ADE、minADE20、minFDE20 均与 EXP-173 达到 0.00mm 绝对复现。
- **P0-B (Evaluation)**: `PASS`，`mode="testing"` 与 `mode="val"` 推理偏差均为 `0.00e+00`。
- **P0-C (Loss & Gradients)**: `PASS`，`original_exact` 模式损失偏差 `0.00e+00`，梯度偏差 `0.00e+00`，Winner 匹配率 100%。
- **P1 (Target Redundancy)**: **`MEDIUM REDUNDANCY`**，整体梯度余弦 `0.8163`，但在高混淆困难样本（$\Delta ADE_{1,2} < 2cm$）梯度余弦显著降至 `0.4216`，证明新目标在难例场景具备实质性区分梯度，判定解除叠加损失路线，推荐进入 P2 纯替换试点。
