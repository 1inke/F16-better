# Pair-Aware Attention Series

本系列固定当前 F16 稀疏时间—社会编码框架、运动学关系、Kinematic K20 解码器、Coverage Loss 0.20、EMA 和训练协议，只逐项替换注意力内部机制。

## 实验顺序

1. `EXP_112_NSCA`：Q/K 归一化与每头可学习缩放余弦注意力；
2. KDTA 基础版本；
3. 根据 KDTA 的采样位置、精度分解与训练稳定性进行针对性适配。
4. `EXP_117_MATKS`：运动自适应局部时间平滑；
5. `EXP_118_RGVA_G4`：关系分组向量注意力；
6. `EXP_119_CDSA`：计算量受控的差分稀疏注意力。

NSCA 不改变 token 数、稀疏图、relation bias/value、V 投影和解码器。其变体名为 `f51_scaled_cosine_attention`。

上述六个阶段覆盖长调研报告中的五个正式方法族（KDTA 包含多轮适配）。统一结论和加权比较见 `FINAL_REPORT.md`。
