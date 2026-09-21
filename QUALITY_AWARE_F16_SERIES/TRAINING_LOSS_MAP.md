# F16 Native Training Loss Audit Map (EXP-173 Baseline)

> **Audit Target**: Exact identification and mapping of all loss components in F16 (EXP-173 E3 Stage V2).  
> **Source Files**:
> - `FRONT_ENCODER_SERIES/train_front_variant.py`
> - `FRONT_ENCODER_SERIES/locked_mode_query_loss.py`
> - `OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml`
> - `E2_F16_END_TO_END_COMPARISON_SERIES/run_e2_f16_joint_comparison.py`
> - `E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py`

---

## 1. 损失组件全景总表

| 损失组件名称 (Loss Component) | 源文件与函数 / 类 | 输入张量 (Input Tensors) | 损失权重 (Weight) | 影响轨迹生成 (Traj Generation) | 影响评分头 (Score / Logit Head) | 梯度作用机制与设计初衷 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Regression (轨迹回归损失)** | `locked_mode_query_loss.py`<br>`WeightedTrajectoryLoss` | `winner_prediction` `[B, 1, 12, 2]`<br>`target` `[B, 1, 12, 2]` | **1.0**<br>(exp penalty 1.0~2.0) | **是** | **否**<br>(winner 由 quality.detach() 索引) | 仅对距离真值最近的 winner mode 反传 SmoothL1，沿时间步按指数衰减递增惩罚，优化赢家候选形状。 |
| **2. Classification (原生模式分类损失)** | `locked_mode_query_loss.py`<br>`ModeQueryLoss.forward` | `logits` `[B, 1, 20]`<br>`quality` `[B, 1, 20]`<br>`winner` `[B, 1]` | **0.5**<br>(soft + 0.25*hard) | **否**<br>(quality.detach()) | **是**<br>(对 logits 反传) | 采用软分类目标 (`soft_temperature=0.5`) 结合硬 winner 交叉熵，监督原生 20-way logits 分类概率分布。 |
| **3. Endpoint (终点对齐损失)** | `locked_mode_query_loss.py`<br>`ModeQueryLoss.forward` | `winner_prediction[..., -1, :]`<br>`target[..., -1, :]` | **0.25** | **是** | **否** | 重点约束赢家候选轨迹终点与真实终点的位移偏差，降低终点误差 (FDE)。 |
| **4. Diversity (模态多样性排斥损失)** | `locked_mode_query_loss.py`<br>`ModeQueryLoss.forward` | 全部 20 个候选终点<br>`endpoints` `[B, 1, 20, 2]` | **0.05**<br>(margin = 0.4m) | **是** | **否** | 计算全部 20 模式终点间的两两成对距离，当距离小于 0.4m 时施加 ReLU 惩罚，防止模态坍缩为同一种轨迹。 |
| **5. Short Horizon (短时程局部约束)** | `locked_mode_query_loss.py`<br>`ModeQueryLoss.forward` | `winner_prediction[..., :2, :]`<br>`target[..., :2, :]` | **0.20**<br>(前 2 个时间步) | **是** | **否** | 强化前 2 步短时历史平滑性，抑制起点跳变。 |
| **6. Oracle Coverage (多候选覆盖强化损失)** | `train_front_variant.py`<br>行 1066~1102 | 全部 20 模式预测 `prediction[:, :1]`<br>`target` `[B, 1, 12, 2]` | **0.20**<br>(temp=0.10, fde=0.25) | **是** | **否**<br>(quality.detach()) | 使用 detached 责任权重对多个接近真值的潜在可行模态施加吸引力梯度，大幅压低 Oracle minADE20/minFDE20。 |
| **7. Quality-Aware Probability Loss (本轮新增辅助损失)** | `train_quality_aware_f16.py`<br>`build_quality_target_ade` | `focal_logits` `[B, 20]`<br>`target_q` `[B, 20]` (detached) | **$\lambda_q$**<br>(Pilot 0.05~0.20) | **否**<br>(target_q 严格 detach) | **是**<br>(仅对 focal_logits 反传) | **核心目标**：将 20 候选真实轨迹质量 (ADE) 映射为软概率目标，直接监督原生 logits，收敛 Top-1 与 Oracle 间距。 |

---

## 2. 原始损失保护与接口规范

1. **零改动保留原则**：
   $$
   \mathcal{L}_{total} = \mathcal{L}_{F16, original} + \lambda_q \mathcal{L}_{quality}
   $$
   严格保留前 6 项原生损失组件，禁止删除、替换或修改权重。
2. **Quality Target 必须完全 Detach (P0 规约)**：
   $$
   q_k = \text{softmax}\left(-\frac{ADE_k(\text{detached})}{\tau}\right)
   $$
   $$
   \mathcal{L}_{quality} = -\sum_{k=1}^{20} q_k \log \text{softmax}(s_k)
   $$
   严禁 $ADE_k$ 对生成器反向传播，确保 Quality Loss 只对分数头提供概率校准梯度，杜绝欺骗性退化。
3. **单主体监督原则**：
   第一阶段仅对 Focal Agent（索引 `[:, 0]`）计算 $\mathcal{L}_{quality}$，邻居 Agents 维持原生监督，保证因果归因纯净。
