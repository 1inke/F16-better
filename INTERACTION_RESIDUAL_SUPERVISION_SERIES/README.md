# Interaction Residual Supervision（IR1）

## 实验目标

IC1已证明社会交互信息可以改变mode和anchors，但“移动幅度”与ADE/FDE收益无关。IR1验证：给social-only representation增加明确的Future Response语义监督后，能否让候选几何朝真实未来移动。

## 对原建议的实现调整

原建议将“相对惯性轨迹的6D残差”直接加到已有learned anchors。当前底座的anchors本身不是恒速锚点，直接相加可能重复修正；同一个物理残差施加给20个mode也可能损害多模态结构。

因此本实验采用更严格的语义瓶颈：

```text
F16 gated social-only token [96]
          ↓
LayerNorm → Linear(96→16) → GELU → Linear(16→6)
          ↓                         ↘ 0.1 × weighted Future Response L1
predicted response [3 anchors × 2D]
          ↓
zero-init Linear(6→64)
          ↓
mode-specific gate(base mode, response hidden)
          ↓
原 proposal / refiner / template / score heads
```

这保留了建议的Future Response语义与mode-specific利用方式，同时避免把相对惯性的残差直接重复加到learned anchor坐标。

## Future Response标签

训练时仅用最后观测位移构造三个惯性位置：

```text
inertia_t = last_position + t × last_observed_step,  t ∈ {4,8,12}
response_target_t = ground_truth_t - inertia_t
```

惯性轨迹只用于定义辅助训练标签：

- 不作为decoder候选中心；
- 不替换或混入20个learned modes；
- 推理阶段不读取未来，也不需要计算惯性轨迹；
- `cv_reference=False`保持不变。

## Interaction权重

使用只依赖观测历史的固定物理强度：

```text
opportunity = exp(-distance/3) × sigmoid(closing/0.5)
conflict = exp(-TTC/3) × exp(-DCA/1.5) × approaching
weight = max_neighbor 0.5 × (opportunity + conflict)
```

无效padding agent被严格屏蔽。辅助损失固定为：

```text
L_total = L_original + 0.1 × mean(weight × response_L1)
```

不进行辅助权重搜索。

## 固定底座与参数

- F16 temporal-social encoder、Top-K4 sparse graph和relation路径不变；
- EXP-146B、20 modes、三锚点 `(4,8,12)`、template/refiner/score不变且冻结；
- optimizer、augmentation、EMA、split、checkpoint criterion不变；
- 新增2,375参数；预计总参数355,048→357,423（+0.669%）。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\INTERACTION_RESIDUAL_SUPERVISION_SERIES\run_irs_batch.py" --device cuda:0 --seeds 1 --variants all --splits all --epochs 12
```

依次运行General B0/IR1、Challenge B0/IR1，随后自动运行Future Response与候选几何诊断。预计训练加诊断约35～55分钟。

仅预检命令：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\INTERACTION_RESIDUAL_SUPERVISION_SERIES\run_irs_batch.py" --device cuda:0 --seeds 1 --variants all --splits all --epochs 12 --dry-run
```

## 自动输出

训练：

```text
INTERACTION_RESIDUAL_SUPERVISION_SERIES/results/
INTERACTION_RESIDUAL_SUPERVISION_SERIES/multiseed_summary.json
INTERACTION_RESIDUAL_SUPERVISION_SERIES/all_runs.csv
```

诊断：

```text
INTERACTION_RESIDUAL_SUPERVISION_SERIES/response_diagnostic/
```

诊断包括：

- step4/8/12 response MAE、L2、cosine；
- 6D整体response direction cosine；
- low/medium/high interaction响应误差；
- response alignment与逐样本ADE/FDE gain的Spearman关系；
- step4/8/12 anchor coverage与endpoint coverage；
- anchor shift、mode differential、winner change和candidate diversity；
- high-interaction子集；
- response head、gate和semantic mode delta激活量；
- 逐样本`paired_response_samples.csv`与诊断图。

## 晋级条件

精度：至少一个划分minADE/minFDE改善不低于0.5%，另一划分不明显反向；或high-interaction改善不低于约2%。

机制还必须同时满足：

- response direction cosine明显大于0；
- high-interaction response alignment更好；
- step8/12 coverage跨划分一致改善；
- alignment越高，ADE/FDE delta越负。

若response方向学不到，停止当前F16 social-token语义路线；若response能预测但oracle不改善，转向candidate allocation/mode assignment，而不是扩大辅助头。

## 已完成验证

- 零初始化与B0预测/锚点严格一致；
- 未来信息仅进入训练辅助损失；
- EMA deepcopy、padding屏蔽、interaction weight范围检查通过；
- response head梯度非零；
- 激活语义桥后产生mode-specific anchor变化；
- 四任务dry-run通过；
- 一轮完整训练、EMA、checkpoint与最终评估通过；
- response head、6→64语义映射和mode gate均已实际更新。

