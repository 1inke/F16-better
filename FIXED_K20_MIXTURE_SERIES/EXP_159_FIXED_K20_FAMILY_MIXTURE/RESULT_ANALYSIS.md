# EXP-159 Result Analysis

## Decision: REJECTED

自动选择的最佳固定配额为K20=14、V2=6，宏平均minADE/minFDE/same-path FDE为0.2556/0.4490/0.5376。相对matched K20，最坏折ADE/FDE退化37.31%/47.97%。

## Next step

固定配额和纯几何去重不能在K=20下保留K40互补性，转向分数引导的压缩验证。
