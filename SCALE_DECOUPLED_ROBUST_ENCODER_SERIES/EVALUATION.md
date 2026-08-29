# SDRE1建议评估

SDRE1是DSA之后合理的最小模型实验：它直接针对F16将速度、位移和密度尺度写入latent幅值的现象，同时保留这些条件的预测价值。相比直接对抗性域适配，它的风险更低，也不与“Challenge高速、低密度是真实条件”的证据冲突。

本实现保留两项约束：

1. raw relation path和原始F16 temporal/social attention不被替换；normalized relation只作为零初始化残差加入。
2. normalized motion shape同样只经过零初始化残差注入agent summary，并由显式scale/density condition门控。

建议中的额外representation LayerNorm未在第一轮重复加入：锁定F16已在social fusion和输出端应用LayerNorm；再插入强制归一化会让训练起点不再等价于B0，并混淆“尺度解耦”与“直接重标定”。本轮改为保留已有LayerNorm，同时将absolute scale/density显式传入两条normalized residual的门控。

目前没有加入domain adversarial loss、MMD、CORAL或tail-risk监督。若SDRE1不能在Challenge的oracle指标和表示诊断中共同改善，应停止继续堆叠robustness trick。
