# EXP-159 Fixed-K20 K20/V2 Mixture

本系列读取 EXP-158B 的 V2-host 共享编码器候选缓存，在不重新推理、不训练新参数的条件下，将 K20 与 V2 的 K40 诊断集合压缩为严格 K=20。

候选选择不接触 GT：

1. V2 候选优先进行几何多样性覆盖；
2. K20 候选按照相对已选 V2 的新增覆盖进行选择；
3. 根据终点与整轨迹距离抑制重复；
4. 若重复抑制导致候选不足，从对应候选族按最大新增多样性补足。

测试配额：`4+16, 6+14, 8+12, 10+10, 12+8, 14+6, 16+4`，前者为 K20 数量，后者为 V2 数量。

EXP-159 的纯几何压缩已被拒绝。EXP-159B 改为 score 主导、几何多样性辅助，并把搜索范围收缩到 `0+20, 2+18, 4+16, 6+14, 8+12`。

## 正式命令

```powershell
C:\coding_toobox\envs\astra2\python.exe -u "C:\Users\linke\Documents\New project\FIXED_K20_MIXTURE_SERIES\run_exp159_fixed_k20_mixture.py" --seed 1 --allocations 4 6 8 10 12 14 16
```

## EXP-159B 正式命令

```powershell
C:\coding_toobox\envs\astra2\python.exe -u "C:\Users\linke\Documents\New project\FIXED_K20_MIXTURE_SERIES\run_exp159b_score_guided_compression.py" --seed 1 --allocations 0 2 4 6 8 --diversity-weights 0 0.05 0.10 0.20
```

