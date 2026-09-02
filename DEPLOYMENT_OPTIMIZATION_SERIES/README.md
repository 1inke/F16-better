# Deployment Optimization Series

## EXP-180A Q3KV Runtime Stability Audit

比较 EXP-173 F16 baseline 与独立训练的 EXP-174D Q3KV。两者均使用 ETH seed1
Joint-V2 正式权重、FP32 eager和相同输入。

正式协议锁定为5个顺序启动的独立Python进程。每个进程测试batch 1和64，每个组合
预热100次、同步计时2,000次。不同进程和batch交替模型测量顺序，降低固定先后顺序与
GPU热状态造成的偏差。

每个batch都必须满足平均P95改善至少15%，且至少4/5个进程为正，才把Q3KV保留为
deployment variant。P95改善不足10%或跨进程不稳定时删除Q3KV。

## EXP-180B Deployment Backend Audit

固定 EXP-173 final F16 Joint-V2，对 ETH/UCY 的 `B=1,N=9` 和 SDD 的
`B=1,N=16` 分别导出固定尺寸模型，比较 PyTorch FP32 eager、ONNX Runtime CUDA
FP32 与 ONNX Runtime CUDA FP16。脚本先核验候选坐标、logits 和 minADE/minFDE，
通过 0.2% 精度门槛后再按中位延迟 20% / 吞吐率 25% 门槛判定。
