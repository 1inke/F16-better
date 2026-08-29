# MA1：Interaction-Aware Mode Allocation

## 评估结论

FCRA1 已证明 post-hoc residual correction 的收益上限很低，因此本轮把社会信息的作用位置前移到训练期 mode matching。建议方向成立，但原建议中“可学习 compatibility 同时构造 target 并接受该 target 监督”存在循环监督风险。本实现作了以下约束：

- Candidate generator、F16 稀疏交互编码器、Mode + 3-Anchor、20 modes、4/8/12 anchors 均保持不变。
- 不增加 residual、不平移 anchor、不修改推理轨迹；compatibility 只是训练期分配先验。
- 几何责任目标完全停止梯度。
- Compatibility 另行拟合纯几何 soft responsibility，避免通过篡改 target 降低 loss。
- 最终 matching cost 使用 `ADE + 0.25 FDE + 0.25 anchor error - social prior`，保留 Top-5 soft responsibility。
- Top-5 responsibility 真正用于轨迹回归、终点和短时损失，不只是增加一个分类辅助项。
- 复用 CSRS1 严格匹配 B0，不重复训练确定性相同的 baseline。

## 第一轮

只训练：

- General：MA1 seed1，与既有 B0 严格配对。
- Challenge：MA1 seed1，与既有 B0 严格配对。

总计两个新训练。完成后自动生成 overall、social-needed 子集、winner entropy、mode usage、assignment confidence、responsibility shift、anchor coverage 和 diversity 诊断。

## 运行命令

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\INTERACTION_AWARE_MODE_ALLOCATION_SERIES\run_ma1_batch.py" --device cuda:0 --seed 1 --splits all --epochs 12
```

预计在当前机器上约 12–18 分钟。实际耗时主要取决于 Challenge split 和显卡是否同时被其他程序占用。

## 晋级标准

- Overall minADE 或 minFDE 至少改善 0.5%，另一主要 oracle 指标不得明显退化；或 social-needed 子集至少改善 2%。
- Challenge 至少复现改善方向。
- Winner usage 不更集中，never-win modes 不增加。
- Social-needed 与 non-needed responsibility 分布产生可测差异。
- Diversity 不明显下降，中后段 anchor coverage 方向一致。

不满足则不跑 Seed2/3，并转向 candidate-set/distribution-level objective。
