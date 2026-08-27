# EXP-028 α=1.5 Entmax 注意力

由用户已读的 Stochastic Non-Autoregressive Transformer 工作启发。固定 F10+K20 和原稀疏图，仅将图内 softmax 换为 α=1.5 entmax。它位于 softmax 与 sparsemax 之间，目标是保留精确零权重，同时减少 F17 的过度稀疏和模式收缩。
