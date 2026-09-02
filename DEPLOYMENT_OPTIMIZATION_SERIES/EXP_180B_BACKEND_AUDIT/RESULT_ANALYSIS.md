# EXP-180B Deployment Backend Audit

固定模型：EXP-173 final F16 + H3 + Joint-V2 + original score + K20。

## 判定标准

- ONNX 相对 PyTorch 的 minADE/minFDE 变化均小于 0.2%。
- 中位延迟降低至少 20%，或吞吐率提高至少 25%。

## Runtime

| Profile | Backend | Median ms | P95 ms | Throughput/s | Median gain | Verdict |
|---|---:|---:|---:|---:|---:|---|
| eth_n9 | B0_pytorch_fp32 | 10.8666 | 12.2426 | 91.97 | - | REFERENCE |
| eth_n9 | B1_ort_cuda_fp32 | 6.6811 | 11.8747 | 142.96 | 38.52% | PASS |
| eth_n9 | B2_ort_cuda_fp16 | 6.0274 | 13.3341 | 137.03 | 44.53% | FAIL |
| sdd_n16 | B0_pytorch_fp32 | 10.5369 | 19.3340 | 84.40 | - | REFERENCE |
| sdd_n16 | B1_ort_cuda_fp32 | 5.1184 | 10.0262 | 167.04 | 51.42% | PASS |
| sdd_n16 | B2_ort_cuda_fp16 | 5.8405 | 9.8992 | 155.19 | 44.57% | FAIL |

## Correctness

| Profile | Backend | ΔminADE | ΔminFDE | Candidate max abs | Verdict |
|---|---:|---:|---:|---:|---|
| eth_n9 | B1_ort_cuda_fp32 | 0.0350% | 0.0566% | 0.019873 | PASS |
| eth_n9 | B2_ort_cuda_fp16 | 0.3711% | 0.2286% | 3.002576 | FAIL |
| sdd_n16 | B1_ort_cuda_fp32 | 0.0515% | 0.0758% | 0.156387 | PASS |
| sdd_n16 | B2_ort_cuda_fp16 | 0.9204% | 0.8372% | 1.079559 | FAIL |

## 自动结论

存在通过 EXP-180B 门槛的部署组合：eth_n9/B1_ort_cuda_fp32, sdd_n16/B1_ort_cuda_fp32。正式保留前需结合两个 profile 综合判断。

说明：ONNX 导出对固定尺寸 `EyeLike` 与旧导出器的 `atan2(0,0)` 展开做了等价兼容处理。
