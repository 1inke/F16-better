# EXP-159 Fixed-K20 Family Mixture

- 输入：EXP-158B 五折 V2-host 候选缓存
- 最终候选数：严格 K=20
- K20/V2 配额：4/16 至 16/4
- 选择规则：仅使用预测候选的几何距离，不使用 GT
- 正式基准：matched K20 和当前 M2 的官方五折 seed 1 结果
- 结果文件：`summary.json`、`per_fold_summary.csv`、`RESULT_ANALYSIS.md`
- 仅为自动推荐配额保存 `selected_candidates/*.npz`，供 EXP-161 使用

