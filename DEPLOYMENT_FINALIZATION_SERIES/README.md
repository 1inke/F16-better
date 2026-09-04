# EXP-187B：全测试集 ORT FP32 等价性审计

固定模型为 F16 + H3 + Joint-V2 + original score + K20。分别加载 ETH/UCY 五折正式 seed1 checkpoint 与 SDD standard 正式 checkpoint，在完整测试集上比较 PyTorch FP32 eager 和 ONNX Runtime CUDA FP32。

脚本逐样本流式累计指标，不保存全量候选张量。每完成一个 profile 会写入独立 JSON；中断后再次运行会自动跳过已完成 profile。ONNX 文件属于可再生部署产物，不提交 Git。

若全量指标通过但出现大张量离群，运行 `run_exp187b_r1_operator_audit.py`，在同一导出图上比较 PyTorch、ORT CPU 与 ORT CUDA，定位导出图和 CUDA provider 的责任边界。

