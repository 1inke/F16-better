# EXP-063 显式关系类型偏置

## 假设

连续关系 token 未必能稳定分离跟随、对向、横穿和潜在冲突。将六个软关系指标直接映射为逐头 attention bias，可能提高社会交互信息的可辨识性。

## 控制变量

- 编码主干：F16 temporal-social cross attention
- 解码：固定 Q20
- 偏置末层零初始化，初始行为与 F16 对齐
- 首轮：Univ, seed=1, val_scene=1, 30 epochs
