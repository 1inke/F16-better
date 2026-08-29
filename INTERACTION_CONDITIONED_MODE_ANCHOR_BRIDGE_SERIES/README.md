# Interaction-Conditioned Mode–Anchor Bridge（ICMAB）

## 本轮问题

验证 F16 已计算出的社会交互增量，能否通过一个轻量、mode-specific 的通路，直接改变20个候选及三锚点几何，并改善 GT coverage 与 minADE/minFDE。

## 固定部分

- F16 temporal-social cross attention；
- 原 Top-K4 稀疏图、relation encoder/bias/value；
- 20 modes；
- 三锚点 `(4, 8, 12)`；
- EXP-146B mode template、anchor refiner、trajectory refiner 与 score head；
- `cv_reference=False`；
- 原 loss、augmentation、EMA、训练划分和 checkpoint criterion。

原解码器参数全部冻结；B0 和 IC1 均从对应 split/seed 的 EXP-146B checkpoint 开始微调 F16 encoder。

## IC1 改动

从 F16 原 social gate 中无损读取：

```text
social_delta = sigmoid(social_gate) * social_vector   [B, N, 96]
```

使用：

```text
96 → Linear(8) → GELU → Linear(64, zero-init)
                              ↓
                       20个mode-specific gates
                              ↓
                     conditioned mode queries
                              ↓
                  原 proposal/refiner/template heads
```

真实结构中 social vector 是96维、mode query 是64维，因此桥为 `96→8→64`。新增参数为2,652，总参数预计从355,048增加至357,700，增幅约0.747%。

最后一个 bridge 投影和 mode gate 均为零初始化。零初始化时 IC1 与 B0 的预测、logits 和三锚点严格一致。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES\run_icmab_batch.py" --device cuda:0 --seeds 1 --variants all --splits all --epochs 12
```

脚本依次运行：

- General：B0、IC1；
- Challenge：B0、IC1；
- 四项训练结束后，自动运行严格配对的候选几何诊断。

预计总耗时约30～50分钟，具体取决于GPU占用；最后的离线几何诊断通常再需要1～4分钟。锁屏不影响运行，但不要让电脑进入睡眠或休眠。

如只想检查命令而不训练：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES\run_icmab_batch.py" --device cuda:0 --seeds 1 --variants all --splits all --epochs 12 --dry-run
```

## 输出

训练结果：

```text
INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES/results/
INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES/multiseed_summary.json
INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES/all_runs.csv
```

几何诊断：

```text
INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES/geometry_diagnostic/
```

诊断自动报告：

- 第4/8/12步 anchor shift 的 mean/median/P90/P95；
- mode-differential shift；
- endpoint/anchor/trajectory candidate diversity；
- oracle winner change；
- GT 第4/8/12步 anchor coverage 与 endpoint coverage；
- bridge/social token/gate 实际激活量；
- 使用训练集 Q75 固定阈值的 high-interaction 子集结果；
- 逐样本 `paired_geometry_samples.csv` 和 `geometry_and_coverage.png`。

## Seed 1 晋级标准

精度至少满足其一：

- General 或 Challenge 的 minADE20/minFDE20 至少一个改善不低于0.5%，另一个不明显退化；
- 或 high-interaction 子集改善约2%～3%。

同时需要机制证据：

- anchor shift 可测量；
- mode-differential shift 非零；
- candidate diversity 不明显下降；
- anchor/endpoint coverage 有改善趋势。

若 anchor 明显变化而 oracle/coverage 不改善，不扩大 bridge；若只改善 Top-1 而 oracle 不改善，也停止本方向。

## 已完成验证

- Python 编译与四任务 dry-run：通过；
- 零初始化严格匹配 B0：通过；
- EMA deepcopy token 传递：通过；
- bridge 输出层梯度：非零且有限；
- 非零 bridge 能改变不同 mode 的 anchor geometry：通过；
- 一轮完整训练、EMA、checkpoint 与最终评估：通过；
- 一轮训练后 bridge 与 mode gate 参数均离开初始化值。

