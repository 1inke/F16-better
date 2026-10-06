# C00 与 H3 的同样本预测可视化

本目录使用项目现有候选缓存，对 C00 的直接来源 H3 与 C00 做同一批 ETH validation 样本、同坐标范围的比较。

- `c00_vs_h3_overview`：全体 validation 样本的配对结果，包括 minADE 散点、逐样本变化分布、均值相对变化及 oracle 候选分数排名。
- `c00_vs_h3_representative_trajectories`：四种预先定义情形的同样本轨迹对照；样本由分位数和明确条件自动选择，不手工挑选图片。
- 灰色为其他候选，蓝色为观测历史，黑色虚线为真实未来，红色为模型最高分候选，绿色为 minADE oracle；紫色表示最高分候选恰好也是 oracle。

重新生成：

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\C00_VISUAL_COMPARISON\visualize_c00_vs_h3.py"
```

该比较只使用锁定的 ETH validation 数据，不读取测试集。代表性样本只用于解释整体统计中已经存在的异质性，不用于选择模型或作总体推断。
