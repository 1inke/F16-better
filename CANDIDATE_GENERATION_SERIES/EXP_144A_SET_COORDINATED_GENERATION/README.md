# EXP-144A：跨候选集合协同生成

来源为对应 split/seed 的 EXP-143B。冻结原模型，仅训练 CandidateSetGenerator；20个候选先交换集合信息，再生成额外三锚点残差。目标是减少候选重复并提高覆盖。
