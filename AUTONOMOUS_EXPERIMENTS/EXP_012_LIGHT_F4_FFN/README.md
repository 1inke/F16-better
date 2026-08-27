# EXP-012：F4 时间 FFN 轻量化

固定规则 K20、关系 token、稀疏运动学图和所有训练协议。将 F4 时间 Transformer 的 FFN hidden dimension 从 192 降到 96（等于 token dimension），其余结构不变。

目标是在保持 F4 稳定选择能力的同时减少编码器参数；若精度接近，优先选择更轻版本。

