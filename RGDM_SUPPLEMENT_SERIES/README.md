# RGDM 补充模块（首轮 2×2 实验已完成）

本目录把《基于 GatedDelta 的轨迹生成与筛选双支路模块设计》补充到当前锁定的 EXP-173/C00 路线上。它不是新的离线 reranker，也不会替换已经验证的 F16、H3 或 V2；新增模块直接位于当前 decoder 内，并保持 C00 的 20 条候选和原生 score 路径。

## 当前问题与对应修正

现有 C00 在锁定验证集上具有较好的候选下界，但原生 Top-1 与 oracle 之间仍有明显差距；同时 V2 会修改三段 anchor，而原生 score 在修正前计算，存在“候选已变、分数未同步”的结构性风险。RGDM-S 因此读取 V2 修正后的 anchors，并使用 observation-only 的 Gated Delta memory 做候选条件化校正。RGDM-G 仅作为后续 generation miss 的补充支路，当前不优先训练。

## 关键实现

1. `RegimeAwareGatedDeltaMemory` 严格实现：

   \[
   S_t=\alpha_t S_{t-1}(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top.
   \]

2. `alpha/beta` 只由编码后的观测 token 与观测期速度、加速度、速度变化和转向证据产生。
3. RGDM-S 的 candidate query 读取同一历史状态的不同方向，并使用修正后的三段 anchors。
4. 原生 logits 被保留。完整选择分数为原生 centered logits 的 change-aware 校准加一个零初始化 residual，因此初始输出与 C00 完全一致。
5. RGDM-G 使用零强度残差融合，初始候选同样与 C00 完全一致。
6. 未来真值只存在于可选的训练辅助 gate/rank loss；两个权重默认均为 0，推理 API 不接收未来真值。

生成融合强度和历史分数校准强度使用零初始化的有界 `tanh` 参数，而不是零点截断；因此初始仍为严格 identity，同时梯度可以向正负两个方向更新。历史校准若学习为正值表示在 change evidence 下减弱原生历史惯性，若为负值则表示数据更支持强化原生分数差异，报告必须如实保留其符号。

## 已实现消融

| 变体 | 新增内容 | 候选是否改变 | 当前优先级 |
|---|---|---:|---:|
| `rgdm_b1_history_gate` | change-aware 原生分数置信门 | 否 | 1 |
| `rgdm_b2_memory` | Gated Delta memory，共享 query | 否 | 2 |
| `rgdm_b3_candidate_query` | candidate-specific memory query | 否 | 3 |
| `rgdm_b4_full` | memory + candidate query + history gate | 否 | 4 |
| `rgdm_g4_generation` | generation-side residual fusion | 是 | 延后 |
| `rgdm_ab_joint` | G + S 联合 | 是 | 最后 |

这里的 B1–B4 是嵌入 decoder 的端到端增量支路，与此前冻结候选的外部 selector 不同；但历史 selector 的负结果仍然是重要先验，因此新实验必须以最小、分阶段、带停止点的方式进行。

## 当前实验进度（2026-09-29）

- `s_only_g0_s1`、`g_only_g1_s0` 与 `joint_g1_s1` 均已完成 12 个 epoch；C00 主干在三个条件中保持冻结。
- 完整 2×2 validation-only 分析已完成，预注册判定为 `STOP_OR_REDESIGN_AFTER_SEED1`；当前结构不进入 seed 2/3 或 test。
- 已修复批量脚本把 decoder 变体名误当成输出目录名的问题；所有结果现在均能被正确识别和审计。
- 跳过已有结果前会核对变体、epoch、scheduler、可训练参数量、增强设置、输入文件哈希、验证 ID、checkpoint 完整性和 test 隔离；完整分析前还会逐 tensor 验证所有冻结 C00 权重与源 checkpoint 完全相同。
- 已完成的筛选分支没有访问 test；当前结果均来自固定的 1,516 条 ETH scene-tail validation 样本。

## 当前边界

- 方法、模型适配器、单元测试、三个训练条件和完整分析均已完成；没有读取测试集。
- `replace_decoder` 要求显式设置 `RGDM_SOURCE_CHECKPOINT`，并只允许从锁定的 C00 完整 checkpoint 补载新增参数。
- 默认冻结整个 C00，仅训练被选择的 RGDM 支路，避免把继续训练收益误归因于 Gated Delta。
- 每个消融中未参与前向的 RGDM 子组件也保持冻结，参数统计只计入实际有梯度路径的部分。
- 是否启用 gate supervision、pairwise rank、训练预算、checkpoint 选择与验证分组，留待实验设计评审后确定。

`preflight_rgdm.py` 只用合成观测检查真实 C00 checkpoint 的加载、严格输出等价和可训练参数范围；它不读取数据集，也不进入训练循环。

## 已准备的首轮实验

首轮采用最小 2×2 因子矩阵，并复用已有 C00，不重新训练 baseline：

| 条件 | 生成侧 RGDM-G | 选择侧 RGDM-S | 新训练次数 |
|---|---:|---:|---:|
| `c00_g0_s0` | 关 | 关 | 0，复用锁定 checkpoint |
| `s_only_g0_s1` | 关 | 开 | 1 |
| `g_only_g1_s0` | 开 | 关 | 1 |
| `joint_g1_s1` | 开 | 开 | 1 |

三个新条件都从同一个 C00 `best_selected.pth` 开始，只训练实际启用的 RGDM 参数。结构试验中 gate/rank 辅助损失固定为 0，避免把监督变化和结构变化混在一起。训练限定为 ETH、generator seed 1、12 epochs、validation-only；脚本不接受 test 路径，也不会自动扩展种子或折。

分析同时读取 `best_original_ade.pth`、`best_selected.pth` 和 `final_model.pth`，分别形成 best-selection、best-generation 和 final 三个视角。报告包含选择主效应、生成主效应、交互效应、motion-change 子组、候选 identity gate 和 10,000 次配对 bootstrap。样本 bootstrap 不代表训练种子不确定性。

在当前 RTX 5070 Laptop 和既有 C00 训练速度下，预计：

- 预检与加载检查：约 1 分钟；
- 三个 12-epoch 新条件：约 10–18 分钟；
- 十次 validation inference、缓存和 bootstrap 分析：约 2–5 分钟；
- 正常总计约 13–24 分钟；GPU 同时被占用时按 25–35 分钟预留。

可直接执行的 PowerShell 命令：

```powershell
& "C:\Users\linke\Documents\New project\RGDM_SUPPLEMENT_SERIES\run_experiment.ps1" -Phase Preflight
& "C:\Users\linke\Documents\New project\RGDM_SUPPLEMENT_SERIES\run_experiment.ps1" -Phase Train -Device "cuda:0" -DryRun
& "C:\Users\linke\Documents\New project\RGDM_SUPPLEMENT_SERIES\run_experiment.ps1" -Phase All -Device "cuda:0"
```

第一条只做预检；第二条只打印并验证三个训练命令；第三条才会正式训练和分析。不要添加 `-Force`，除非明确要覆盖已有试验。

## 实验前必须先决定

1. 首轮是否只比较 B0/B1/B2/B3/B4，还是先做 B0/B1/B4 的最小筛选。
2. 是否保留默认的纯交叉熵监督，或增加 gate/rank 辅助项；不能在同一轮同时改结构和损失后再归因。
3. 使用 continuation 的 epoch 数和学习率，以及 best/final 一致性规则。
4. motion-change 子组只能用于验证分析，不能作为推理时的 GT gate。
5. 正式扩展前必须先完成生成器整条链的样本隔离审计；首轮仍应限定为 ETH、seed 1、validation-only。
