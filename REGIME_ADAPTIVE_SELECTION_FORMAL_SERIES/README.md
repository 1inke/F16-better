# Regime-Adaptive Selection Formal Series

这是 Observation-Only Regime Detector 通过 H1/H2 后的正式 Stage-C 实验。实验不再同时修改生成器：五个 ETH/UCY 留一场景折均冻结已有 EXP-156B F16-V2 seed-1 生成器，只训练 observation-only detector 与候选筛选修正头。

## 因果对照

- `constant_gate`：D1 训练集平均概率作为常数门控；与自适应门控具有相同筛选头容量。
- `shuffled_d1`：保持 D1 概率边际分布，但打乱样本对应关系。
- `d0_motion_gate`：由纯运动 detector 概率控制筛选修正。
- `d1_social_gate`：由运动+冻结社交表征 detector 概率控制筛选修正；正式主条件。
- 原生 V2 logits 始终作为零训练 reference。

筛选头只接收 observation-only 运动特征、冻结候选轨迹及原生 logits。未来真值只用于 detector 标签和训练期 soft candidate-quality target；模型推理接口不接收未来。

## 严格阶段顺序

`All` 会按以下顺序执行：预检 → 只生成 train/dev 缓存 → 完成并锁定全部 5×3×(2 detector + 4 reranker) 检查点 → 才首次生成 test 缓存 → 一次性 test 推理 → Macro/分层 bootstrap 分析。测试集不参与 epoch、模型或变体选择。

三个 seed 是 detector/reranker seed；每折生成器固定为已有 seed-1，因此不能表述为“三个完整生成器种子”。正式结论以五折 Macro 为主，Micro 仅作描述。

## 运行

```powershell
& "C:\Users\linke\Documents\New project\REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES\run_experiment.ps1" -Phase All -Device "cuda:0"
```

不要在已有结果上使用 `-Force`，除非确实需要删除并重做对应缓存或检查点。
