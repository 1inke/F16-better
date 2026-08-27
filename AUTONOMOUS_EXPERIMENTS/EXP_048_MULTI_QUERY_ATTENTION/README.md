# EXP-048 Multi-Query Attention

4 个 query heads 共享 1 个 K/V head。相对 F16 将 K/V 参数、投影量和未展开缓存减少 75%。
