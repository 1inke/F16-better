# EXP-043 Temporal Landmark-4

固定 F16 与 K20，将每个邻居的 8 个历史 K/V 用共享可学习时间投影压缩为 4 个 landmarks。实际缩小 temporal-social 的 K/V 投影输入和注意力分数张量。
