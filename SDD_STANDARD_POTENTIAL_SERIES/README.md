# 当前模型的整体 SDD 潜力实验

本系列只回答一个问题：当前筛选模型在覆盖全部 Stanford Drone Dataset 的标准场景划分上，是否具有继续研究的潜力。

## 固定模型

- F16 temporal-social cross 编码主干；
- 运动学有向稀疏图，最多 16 个智能体、每个节点 Top-K=4；
- Kinematic K20 候选生成；
- Coverage 0.20、EMA 0.995；
- 目标航向规范化；
- 50% 概率的镜像与 0.9–1.1 速度增强。

这不是早期 ASTRA 复现实验，而是当前多轮实验后保留的模型组合。

## 数据与指标

- 使用公开 PECNet/Y-Net 协议对应的 31/12/17 视频 train/val/test 划分，覆盖全部 60 段视频；
- 仅保留 `Pedestrian` 且 `lost=0` 的轨迹；
- 从 30 fps 下采样至 2.5 Hz，观察 8 帧、预测 12 帧；
- 验证集只负责选择 checkpoint，测试集随后只评估一次；
- 文献对比只使用像素单位的 `minADE20` 和 `minFDE20`。

默认不截断每段视频的样本，因此仍然是全场景结果；计算量小主要来自模型本身较轻、轨迹窗口总量有限，而不是只抽一个场景。

## 运行

```powershell
python "C:\Users\linke\Documents\New project\SDD_STANDARD_POTENTIAL_SERIES\run_sdd_standard_potential.py" --device cuda:0 --epochs 30
```

若显存不足，可增加 `--batch-size 32`。若要完全重建预处理数据并覆盖旧实验，加上 `--rebuild-data --force`。

运行结束后，把 `sdd_standard_result_summary.json` 或终端结束信息发给 Codex 进行正式分析。
