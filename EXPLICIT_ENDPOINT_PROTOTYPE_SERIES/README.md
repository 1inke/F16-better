# EXP-163 Explicit Endpoint Prototype K20

第一阶段只验证候选生成。F16编码器从当前V2 checkpoint加载并冻结；新解码器从fold-local训练未来构造的`5 endpoint × 4 path-shape = 20`固定原型出发，只训练原型条件残差和原始形式的mode score head。

本实验不加入M2、新scorer、scene branch、attention或K40筛选。第一轮使用seed1官方五折，与matched K20和V2比较minADE20、minFDE20和same-path FDE。

运行入口：`run_exp163_batch.py`。脚本会自动构造fold-local原型、训练五折、评估held-out test并生成`summary.json`、`per_fold_summary.csv`和`RESULT_ANALYSIS.md`。
