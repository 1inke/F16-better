# EXP-055 Fixed Half Projection

F35 的工程等价实现。用固定、无参数的 2×8 投影矩阵分别平均前四帧和后四帧；与 `reshape(...).mean()` 数学等价，但调用矩阵化 einsum kernel。直接加载 F35 checkpoint，先验证输出等价，再比较 CUDA 延迟；不需要重新训练才能判断实现层效率。
