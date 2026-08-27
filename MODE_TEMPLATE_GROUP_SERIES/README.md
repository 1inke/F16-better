# EXP-146：锚点对齐运动模板与软模式分组

本轮验证两个假设：完整轨迹不应只由锚点线性插值；20个无组织模式应形成粗组和组内细模式。

- **EXP-146A Soft Group Reranking**：冻结全部轨迹，仅训练5组软层级评分，测量同组命中率、跨组错误率和组后悔值。它只验证选择问题，不会改变minADE/minFDE。
- **EXP-146B Anchored Mode Template**：每个模式学习一条12步离散运动模板；模板先对齐第4/8/12步锚点，再由轻量解码器一次性输出12步残差。锚点处残差为零，先隔离检验“线性插值是否是瓶颈”。
- **EXP-146C Grouped Anchored Template**：在B基础上把20个模式按源模板几何软组织为5组，组向量参与锚点和残差生成，组先验参与层级评分；不做硬候选屏蔽。

所有方案从匹配split/seed的EXP-143B初始化，保持20候选、稀疏交互、运动学输入和无CV外推。初始输出严格复现来源模型。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\MODE_TEMPLATE_GROUP_SERIES\run_mode_template_group_batch.py" --device cuda:0 --epochs 30 --batch-size 64 --seeds 1 2 3
```

默认共18个任务。A自动限制为15 epochs，B/C使用30 epochs。脚本支持断点续跑，不要添加`--force`。
