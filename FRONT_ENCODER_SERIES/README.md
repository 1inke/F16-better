# 前端编码器系列实验（固定 Q20 后端）

本目录用于隔离本轮“只改前端、不再改解码”的实验。`ModeQueryEndpointPCADecoder`、20 个模式、`ModeQueryLoss` 和 Social Verifier 的融合系数 0.30 均保持不变。这里的“固定”是固定结构、损失和评分协议；编码器改变后模型仍需端到端重新训练，解码器参数不会被冻结。

## 第一批变体

| 编号 | 唯一新增因素 |
|---|---|
| F0 | 当前 Q20 基线 |
| F1 | 鲁棒多尺度运动状态（短/长速度、平滑加速度、航向、转向与置信度） |
| F2 | F1 + recent/trend/turn 动态时间查询；保留原混合注意力编码器 |
| F3 | F1 + 时间编码与社会编码因子化 |
| F4 | F3 + 随时间聚合的关系 token（相对位置/速度、距离、closing、TTC、DCA） |
| F5 | F4 + 局部邻居/风险邻居双通道图 |

## 推荐先跑：单场景、单随机种子的筛选实验

```powershell
python "C:\Users\linke\Documents\New project\FRONT_ENCODER_SERIES\run_front_encoder_batch.py" --device cuda:0 --seeds 1 --val-scenes 1 --epochs 30
```

批量入口会自动选择带 CUDA PyTorch 的 `astra2` 环境。也可以显式指定：

```powershell
python "C:\Users\linke\Documents\New project\FRONT_ENCODER_SERIES\run_front_encoder_batch.py" --python-executable "C:\coding_toobox\envs\astra2\python.exe" --device cuda:0 --seeds 1 --val-scenes 1 --epochs 30
```

这会顺序训练 F0-F5。验证集是训练文件中完整留出的 scene 1，不读取目标测试集。每个实验输出在：

`C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\results\front_encoder_series\<variant>\seed_<seed>\val_scene_<scene>`

汇总表位于：

`C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\results\front_encoder_series\batch_summary\matched_results.csv`

## 扩展复验

第一轮选出 1–2 个候选后，再补 3 个随机种子和四个源场景：

```powershell
python "C:\Users\linke\Documents\New project\FRONT_ENCODER_SERIES\run_front_encoder_batch.py" --device cuda:0 --variants f0_baseline f4_relation_tokens f5_dual_graph --seeds 1 2 3 --val-scenes 0 1 3 4 --epochs 30
```

已有 `metrics.json` 的任务默认跳过；确需重跑时追加 `--force`。单独调试某一变体可直接运行 `train_front_variant.py`。

## 判读顺序

首先看 `min_ade_20`：它衡量编码器是否让候选集合本身更好。然后看 `original_ade/fde`：它衡量原 Q20 logits 是否能选中好候选。最后看固定协议下的 `fused_ade/fde`。如果 oracle 改善但 original/fused 不改善，问题主要在表示到模式打分的接口；如果 oracle 也不改善，该前端结构本身没有提供有效信息。
