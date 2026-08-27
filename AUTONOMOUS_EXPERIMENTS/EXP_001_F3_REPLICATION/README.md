# EXP-001：F3 因子化编码稳定性复验

## 目的

现有单种子、单场景实验显示 F3（时间编码与社会编码因子化）相对 F0 有明显选择精度收益，但 oracle `minADE20` 略差。本实验先固定 scene 1，补齐 seeds 1/2/3，判断收益是否稳定，并保留 F1、F4 用于贡献归因。

## 对照组

- F0：原 Q20 编码器基线；
- F1：仅加入鲁棒多尺度状态；
- F3：F1 + 时间/社会因子化；
- F4：F3 + 动态关系 token。

## 固定条件

- Q20 `ModeQueryEndpointPCADecoder`；
- 20 modes、相同 ModeQueryLoss；
- 稀疏运动学图先验不变；
- Social Verifier `alpha=0.30`；
- 留出训练数据中的完整 scene 1；
- 30 epochs，batch size 64。

## 运行命令

```powershell
python "C:\Users\linke\Documents\New project\FRONT_ENCODER_SERIES\run_front_encoder_batch.py" --device cuda:0 --variants f0_baseline f1_robust_state f3_factorized f4_relation_tokens --seeds 1 2 3 --val-scenes 1 --epochs 30 --output-root "C:\Users\linke\Documents\New project\AUTONOMOUS_EXPERIMENTS\EXP_001_F3_REPLICATION\results"
```

结果完成后写入 `RESULT_ANALYSIS.md`。
