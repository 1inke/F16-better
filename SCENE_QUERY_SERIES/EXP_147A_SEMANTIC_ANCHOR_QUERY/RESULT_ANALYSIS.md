# EXP-147A 结果分析

状态：等待实验完成。

本文件只保留分析位置，不预填任何结果。实验完成后，以 `raw_result_summary.json`、各运行目录的 `metrics.json` 和 `history.csv` 为依据填写。

重点比较：

1. S1 相对 S0 的测试集 minADE20 / minFDE20 改变量；
2. 增益是否集中于 `little` 等跨尺度困难场景；
3. 语义锚点修正是否同时改善路径和终点，而非只转移 ADE/FDE 冲突；
4. 新增在线参数量，以及离线 SegFormer 不计入在线模型后的参数口径。
