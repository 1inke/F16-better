# 下一轮：固定候选下的观察历史条件评分

当前唯一研究目标是改善候选轨迹选择。固定预算时间选择方向暂停；本系列不安排该方向、不改变候选生成器，也不自动确定更远期研究路线。

## 这轮要回答的问题

之前对冻结 latent、候选几何、Top-1 loss 和集合排序器的尝试没有形成大的稳定收益，但这不证明原始观察历史没有可恢复的信号。本轮检验：**在相同候选上，独立训练的候选条件历史读取器，能否比只利用恒速信息的同架构评分器选得更好？**

只有评分结果发生变化，最终输出始终是原候选集合中的一条轨迹，不对轨迹做平均、修正或另行生成。

## 实验矩阵

| 名称 | 脚本标识 | 实验内容 | 作用 |
|---|---|---|---|
| N0 | Native | 缓存原生 logits 的 argmax | 原生参照，不训练 |
| L0 | linear | 冻结 mode/context、最终轨迹及位移、末步速度、原生分数的线性评分 | 新协议下的低成本对照；不是历史 Tiny L5 的复现 |
| H0 | cv_history | 候选查询读取恒速重建历史，输出候选相对效用 | 匹配模型容量、当前速度与训练预算 |
| H1 | real_history | 与 H0 完全相同的网络，读取真实观察历史 | 唯一主比较：H1 对 H0 |
| D0 | CV replacement | H1 训练后，将历史换为自身恒速历史 | 判断是否依赖非恒速历史；不参与调参 |
| D1 | residual shuffle | H1 训练后，在样本间打乱偏离恒速的历史残差 | 固定末点和末步速度，检查历史对应关系；不参与调参 |

H0/H1 均采用同一个 GRU 历史编码器、候选查询注意力和残差评分头。隐藏宽度与优化参数全部在 config.json 中锁定；参数量从模型实时统计，特征维度从真实张量推导。相同种子下初始化、样本顺序与预算相同。

恒速对照保留了当前运动方向与速度，同时不包含过去的非恒速变化。两组都有冻结 context/mode，因此这轮检验的是独立历史读取路径的增量作用，而不是“原模型完全不知道历史”。它也不涉及选帧或改变观察预算。

## 统一训练目标

预测候选相对效用 `u_k = -(ADE_k - mean_j ADE_j)`，所有训练组均使用 MSE。ADE只用于训练标签、校准指标和最终评估；特征构造只允许读取观察历史、模型生成候选、冻结 latent 和原生 logits。

输出分数为 `native_logits + alpha * scale * predicted_utility`。scale只从拟合集计算，alpha和检查点只从校准集选择，alpha网格包含零以允许退回原生选择。整个训练期间不使用 development 集早停或选择alpha；每个训练完成的模型才评估一次development集。额外报告不融合原生分数时的utility选择ADE，用于判断失效来自效用预测还是融合。

这是一个新协议下的控制实验。与旧R3、旧Tiny或旧Transformer不可直接进行架构效果归因。本轮不进行损失函数网格搜索。

## 数据范围与边界

- 默认固定当前 ETH / generator seed 1 / EXP-173 V2 已有 train、val 缓存与C0权重。
- 从原train缓存按显式sample_id确定性划分评分器拟合集与校准集；使用旧val作为development诊断集。划分规则不读取标签。
- 预检验证sample_id唯一性、拟合与development不相交、与原生成器split一致；从future独立重算ADE/FDE，检查标签及索引。
- future数组只在预检和监督评估中使用；预测脚本根本不加载future或ADE/FDE字段。
- 原生成器已经训练过拟合/校准样本，旧val也被此前实验观察过；这不是新的独立确认集。当前没有场景/轨迹分组元数据，不假装样本独立，不输出显著性置信区间。
- 三个种子是评分器初始化种子，生成器仍固定。结果是单个生成器下的development诊断，不是三生成器稳定性，也不是五折Macro结果。
- 本轮复用已做过一致性审计的缓存；预检锁定当前缓存和权重哈希，但不重新执行完整在线生成器等价性检查。不能将文件哈希本身当作缓存生成来源的证明。
- 没有读取官方test的代码路径，不启动生成器训练，不继续运行旧脚本中被阻断的后续阶段。

## 运行方法（PowerShell，可在任意目录执行）

先运行指标、未来信息隔离与结构检查：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\test_selection.py"
```

预检只核验数据和锁定身份，不训练：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\run_selection_experiments.py" --stage prepare
```

正式运行本轮试点（自动先预检，三个模型×三个评分器种子；完成后停止）：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\run_selection_experiments.py" --stage pilot --device cuda:0
```

小样本冒烟测试已在交付时执行。需要自行重测时使用新的输出目录：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\run_selection_experiments.py" --stage smoke --device cuda:0 --output-dir "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\smoke_recheck"
```

正式运行完成后可从机读结果重新生成报告：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\run_selection_experiments.py" --stage report
```

CPU可用 `--device cpu` 显式选择。已有训练目录不会被覆盖；中断后重跑请指定新的 `--output-dir`。改变配置、源数据或核心代码也必须使用新目录，不允许污染既有结果。

## 输出

- `results/preflight.json`：缓存/生成器权重/划分/配置/核心代码哈希、动态维度、适用范围。
- `results/split_ids.npz`：拟合、校准、development显式ID。
- `results/<模型>_seed<种子>/selector.pth`：实际训练权重、归一化、alpha、scale和来源身份。
- `history.json`：训练误差、校准ADE、alpha及scale逐epoch记录。
- `predictions.npz`：sample_id、分数、选中索引和逐样本ADE。
- `metrics.json`：ADE/FDE、regret、near-oracle、尾部regret、误干预和获益/损害幅度。
- H1附带`history_ablation.json`，区分小oracle gap与大native regret交叉子集。
- `leaderboard.csv`、`summary.json`、`RESULT_ANALYSIS.md`、`state.json`：动态汇总与审阅停止点。

实际拟合时间被记录，但本脚本不宣称端到端推理加速。smoke_results中的数字仅检查执行，不可用来判断方法有效性。

## 如何做下一步决策

主比较预先固定为H1对H0；本轮建议的筛选门槛为：平均Selected ADE相对改善至少2%，至少两个评分器种子改善，平均FDE相对恶化不超过1%，ADE优于Native且不差于L0，严重误干预率不增加。这些是config.json中可审阅的资源决策阈值，不是统计显著性标准。必须在正式运行前锁定，不根据结果降低。

若通过，下一步仍只围绕选择：先审阅误干预、尾部表现与历史消融，再决定做邻居观察条件评分或跨生成器种子确认。若需要正式五折验证，必须先建立整条生成器—缓存—评分器链的外层隔离，不能直接拼接旧OOF缓存。

若没通过，应根据日志区分：alpha归零、效用预测不佳、校准到development分布偏移、历史分支优化不足、或这一路径没有稳定收益。不要马上扩大模型，也不要写成信息论极限。更远期路线保持未定。

## 单独推理

`predict_selector.py` 接收训练检查点以及只含sample_id和五个输入字段的NPZ，输出原候选中的选中轨迹；不需要真实未来。

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\predict_selector.py" --checkpoint "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\results\real_history_seed1\selector.pth" --input "C:\Users\linke\Documents\New project\F16_RANKABILITY_PROBE_SERIES\results\cache\val_seed1.npz" --output "C:\Users\linke\Documents\New project\CANDIDATE_HISTORY_SELECTION_SERIES\results\real_history_seed1\inference_replay.npz" --device cuda:0
```

输入NPZ即使包含标签，推理脚本也只加载 `sample_id / past_relative / candidate_final / mode_feature / context_feature / native_logits`。接入新生成器时必须重建、核验特征来源，不能只因维度相同就直接套用。
