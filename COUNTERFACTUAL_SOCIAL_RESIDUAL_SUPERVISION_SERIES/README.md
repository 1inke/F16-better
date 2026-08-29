# CSRS1：Counterfactual Social Residual Supervision

## 研究问题

在CSNA确认F16社会上下文整体有效、但手工interaction potential无法判断社会必要性之后，本实验验证：模型能否从F16的`fused_agent-individual_agent`社会增量中学习：

1. 当前样本是否需要社会修正；
2. 需要时，三个未来锚点应如何修正。

## 严格设计

- 教师：IR1实验中的冻结B0 seed1 checkpoint。
- Need标签：`min_k(ADE_self,k+0.5FDE_self,k)-min_k(ADE_full,k+0.5FDE_full,k)>0.01`。
- 连续置信度：`clip(max(g,0)/0.10,0,1)`。
- Residual目标：`GT anchors - best self-only learned candidate anchors`。
- 辅助损失：`0.10 L_res + 0.05 L_need`。
- 推理只运行Full Social编码器；Self-Only和GT只用于离线教师。
- 解码仍是20模式、三锚点、模板与细化结构；不使用CV候选或CV参考。
- B0和CSRS1均关闭随机反射/速度增强，避免离线残差标签与随机坐标变换不一致。

组合代价必须先对同一个候选计算ADE和FDE再取最小值，不能把独立minADE与minFDE相加。

## 一键运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES\run_csrs_seed1_batch.py" --device cuda:0 --seed 1 --splits all --variants all --epochs 12
```

流程依次为：样本索引侧车验证 → General/Challenge教师缓存 → 4组匹配训练 → 机制诊断。

## 输出

- `teacher_cache/`：带checkpoint和split指纹的Full/Self教师缓存。
- `results/`：B0与CSRS1训练结果。
- `seed1_summary.json`：主指标配对汇总。
- `diagnostic/combined_summary.json`：Need、Residual、增益相关性和anchor coverage诊断。
- `RESULT_ANALYSIS.md`：实验完成后的分析。

