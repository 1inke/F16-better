# EXP-187B Full-Dataset ORT Equivalence Audit

## 正式结论

六个 profile、36,840 个测试样本全部完成。结论为：

**METRIC_PASS_TENSOR_ANOMALY_AUDIT_REQUIRED**

ORT CUDA FP32 通过 minADE/minFDE 的聚合误差门槛，但存在候选与 logit 的少数大幅离群差异，暂时不能宣告与 PyTorch FP32 严格数值等价，也不能正式锁定为最终部署后端。

## 聚合指标

ETH/UCY 五折宏平均：

| Metric | PyTorch | ORT CUDA FP32 | Relative delta |
|---|---:|---:|---:|
| minADE20 | 0.218105 | 0.217991 | 0.0522% |
| minFDE20 | 0.353215 | 0.353104 | 0.0316% |
| same-path FDE | 0.423579 | 0.422884 | 0.1642% |
| Top1 ADE | 0.518412 | 0.517855 | 0.1076% |
| Top1 FDE | 1.075664 | 1.074069 | 0.1483% |
| Top1 J | 1.056244 | 1.054889 | 0.1283% |

SDD：minADE 相对偏差 0.0154%，minFDE 相对偏差 0.0302%，聚合指标同样通过。

这些差异不能解释为 ORT 提升了准确率，因为后端理论上运行同一个冻结模型；误差变好或变坏都属于数值路径差异。

## 分 profile 审计

| Profile | Samples | ΔminADE | ΔminFDE | Candidate mean/max | Logit mean/max | Top1 changed | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| ETH | 364 | 0.0414% | 0.0469% | 0.000890/0.1869 | 0.001717/0.1834 | 0.55% | metric pass |
| Hotel | 1,197 | 0.0940% | 0.0159% | 0.002947/1.1136 | 0.007057/1.0936 | 1.17% | tensor review |
| Univ | 24,334 | 0.0107% | 0.0131% | 0.000570/2.0014 | 0.002784/2.6670 | 0.24% | tensor review |
| Zara01 | 2,356 | 0.0076% | 0.0163% | 0.000957/0.1217 | 0.002595/0.0804 | 0.30% | metric pass |
| Zara02 | 5,910 | 0.1915% | 0.1060% | 0.005013/1.1913 | 0.016443/1.6635 | 2.23% | tensor review |
| SDD | 2,679 | 0.0154% | 0.0302% | 0.001467/0.8393 | 0.001739/0.7483 | 0.19% | tensor review |

全数据按样本加权的 Top1 索引改变率约为 0.5945%。此外，ETH 与 Hotel 的 same-path FDE 偏差分别为 0.305% 和 0.369%，虽非预注册主门槛，也说明候选身份会在少数样本上改变。

## 原因判断

普通 FP32 运算顺序变化可以解释很小的平均绝对差，但不能充分解释候选坐标最大差 2.00、logit 最大差 2.67。运行时持续出现 ORT CUDA `ScatterND` 重复索引警告；模型的稀疏图路由包含 TopK 与 scatter 操作。最可能的机制是：

1. TopK 在并列或极接近的邻居分数上选择了不同索引；
2. scatter 在重复/边界索引上的执行语义或并行顺序不同；
3. 一个很小的前级差异触发离散邻居集合变化，随后经注意力传播成为少数大离群输出。

这仍是待验证推断，不能仅凭警告直接认定具体算子。

## 下一步

不需要重新训练模型。先做一个部署算子定位审计：保存离群样本索引与 `num_valid`，在相同 ONNX 上比较 ORT CPU 与 ORT CUDA，并记录 PyTorch/ORT 的稀疏邻居 TopK。如果差异来自 ScatterND/TopK lowering，则把布尔 scatter 改写为等价的 deterministic one-hot/reduction 导出表达，再重新运行六个 profile。

只有修复后仍满足 minADE/minFDE <0.2%，且不再出现大幅候选/logit离群和显著Top1改变，才能设置：

`FINAL_DEPLOYMENT_BACKEND = ORT CUDA FP32`
