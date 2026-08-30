# EXP-158B Shared-Encoder Complementarity Control

- 类型：零训练、冻结模型、官方 ETH/UCY 五折、seed 1
- 主控制：V2-host，即 V2 编码器产生一次 encoded tokens，同时送入 V2 与 K20 decoder
- 对称控制：K20-host，即 K20 编码器产生一次 encoded tokens，同时送入 K20 与 V2 decoder
- K40 union 仅为诊断上限
- 通过标准：V2-host 下，union 相对更好的单族在五折宏平均 minADE/minFDE 均改善至少 5%，且每个 fold 两项指标均有正增益

若通过，下一步以 V2-host 为共享编码器进入固定 K=20 配额实验。

