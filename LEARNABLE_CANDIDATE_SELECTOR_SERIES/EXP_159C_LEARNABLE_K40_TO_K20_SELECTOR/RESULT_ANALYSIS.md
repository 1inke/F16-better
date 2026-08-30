# EXP-159C Result Analysis

## Decision: REJECTED

学习式Top-20选择器的宏平均minADE/minFDE/same-path FDE为0.2468/0.4257/0.4976。相对matched K20，最坏折ADE/FDE退化17.64%/19.75%，没有保留K40的候选上界。

## Next step

停止K40到K20候选压缩主线，回到固定V2/M2候选并独立审计评分失配。
