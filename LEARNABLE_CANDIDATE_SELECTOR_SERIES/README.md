# EXP-159C Learnable K40-to-K20 Candidate Selector

EXP-159/159B 证明静态配额无法同时适配 ETH、Hotel、Univ、Zara1、Zara2。本系列冻结 F16/V2 encoder、V2 decoder 与 transplanted K20 decoder，只训练一个小型共享候选质量网络，对每个样本动态从 K40 中保留 K20。

严格折内流程：

1. 使用当前 fold 的训练集生成 K40 候选；
2. 训练集内部固定划分 90%/10% 训练与早停验证；
3. held-out fold 不参与训练、超参数选择或早停；
4. 最终报告 learned Top-20 oracle、K40 oracle、Top-1、winner retention 与动态 K20 数量分布。

## 正式命令

```powershell
C:\coding_toobox\envs\astra2\python.exe -u "C:\Users\linke\Documents\New project\LEARNABLE_CANDIDATE_SELECTOR_SERIES\run_exp159c_learnable_selector.py" --device cuda:0 --folds all --seed 1 --epochs 20 --candidate-batch-size 64 --selector-batch-size 256
```

