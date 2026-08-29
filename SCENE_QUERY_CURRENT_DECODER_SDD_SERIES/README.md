# EXP-147B：当前三锚点模式模板解码器上的 SDD 语义锚点查询

本轮用完整 SDD `31/12/17` 视频划分，严格比较两组从零训练的模型：

- `s0_current`：F16 稀疏社会编码器 + CV-free 三锚点 + anchor refiner + anchored mode template；
- `s1_scene_anchor_query`：与 S0 完全相同，只在 refiner 后的三个 focal anchors 查询离线语义图并给出有界修正，然后进入同一 mode-template 对齐与细化链。

因此 S0 是当前解码器的正式 SDD 基线；S0 vs S1 才是场景信息的因果对照。旧 EXP-132 的 K20 结果仅作为历史参考，不能替代 S0。

语义图由 SegFormer-B0 对每段视频的一张 `reference.jpg` 离线生成。训练与推理不会运行图像网络；在线部分仅在每个候选的三个锚点读取 6 个语义属性并经过小 MLP。

## 首轮运行

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SCENE_QUERY_CURRENT_DECODER_SDD_SERIES\run_exp147b_current_sdd.py" --device cuda:0 --seeds 1 --variants both --epochs 30
```

首次运行会准备语义资产；若本机尚未缓存 SegFormer-B0，会下载一次模型。结果写入 `EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY/results/`，汇总为 `result_summary.json`。原始数据、语义资产、权重和控制台日志均不进入 Git。

## 筛选判据

进入多种子前，S1 相对 S0 应至少满足：`minFDE20` 改善 5%，`minADE20` 不退化超过 1%，且 `little` 子集改善、非 little 不出现系统性损伤。

## EXP-148：尺度反事实审计

当 S1 未达到筛选门槛且 `little` 仍明显较差时，运行：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SCENE_QUERY_CURRENT_DECODER_SDD_SERIES\run_video_scale_audit.py" --device cuda:0
```

该审计加载已完成的 S0 checkpoint，只在测试集推理一次；不会训练或改写模型。它报告历史位移、图像尺度、GT 终点半径、候选终点跨度，以及允许候选仅做幅值缩放时的 oracle 覆盖上界。输出在 `EXP_148_SDD_VIDEO_SCALE_COUNTERFACTUAL_AUDIT/audit_summary.json`。

## EXP-149：严格 K=20 尺度族公平审计

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" -u "C:\Users\linke\Documents\New project\SCENE_QUERY_CURRENT_DECODER_SDD_SERIES\run_fixed_k20_scale_family_audit.py" --device cuda:0
```

它保持每个样本严格 20 条候选：按模型预测的终点半径分成近程、标准、远程族，并额外检查仅由观察期速度控制的门控版本。若 `little` 的 minFDE20 有明显改善、非 little 不受损，才进入真正的可学习幅值分配训练。
