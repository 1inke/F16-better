# 编码与训练精度实验系列

本系列固定 F16 稀疏 temporal-social 编码、Kinematic K20、Coverage 0.20、EMA 0.995、航向规范化和既有增强，只验证编码器或训练约束的单变量变化。

正式批处理包含：

1. EXP-123：同批 F16 基线；
2. EXP-124：镜像轨迹集合等变一致性，权重 0.02；
3. EXP-125：镜像轨迹集合等变一致性，权重 0.05；
4. EXP-126：基础运动状态 token 的纯通道门控；
5. EXP-127：既有8维关系特征的纯通道门控；
6. EXP-128：F16 时间编码后的零门控 depthwise TCN 残差；
7. EXP-129：只作用于时间 Q/K 的 RoPE。

脚本可断点续跑：已有 `metrics.json` 的运行会自动跳过。`--force` 才会覆盖重跑。

可先加 `--dry-run` 检查全部命令和路径而不启动训练。

正式实验已经完成，统一数值见 `batch_summary.json`，分析结论见 `FINAL_REPORT.md`。本批没有变体通过oracle精度晋级标准。

## 正式运行

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\ENCODER_TRAINING_PRECISION_SERIES\run_encoder_precision_batch.py" --device cuda:0 --seeds 1 --val-scenes 1,3 --epochs 30
```

中断后重复执行同一命令即可续跑；脚本会跳过已有完整 `metrics.json` 的实验。
