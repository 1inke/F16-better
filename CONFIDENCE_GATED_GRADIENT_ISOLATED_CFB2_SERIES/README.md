# CFB2：置信度门控与梯度隔离 Future Bank

## 实验目标

CFB1 改善了 K=2/5/10，却损伤 K=20。本轮只验证一个核心假设：可靠 pseudo future 是否能在不污染原主干的情况下改善候选覆盖。

完整流程为：

1. PFRA：只用训练折拟合线性置信度，验证折只做锁定评估；
2. PFRA 两个 split 均通过才训练；
3. 每个 query 使用 0、1 或最多 2 条高置信 pseudo future；
4. 两条 pseudo branch 稳定映射至 mode 18/19，其余 18 条候选保持 B0；
5. 原编码器、mode query、三锚点模板、score head 全部冻结；
6. pseudo loss 只能更新约 5K 参数的轻量 residual adapter；
7. 训练完成后自动执行 K 曲线、tail winner、social-needed、主干保护和 adapter usage 诊断。

## 与建议文件相比的必要修正

- PFRA 的回归系数、标准化参数、置信度中心和阈值全部由训练折确定，禁止用 validation GT 调参。
- “排除 GT winner”不是严格隔离。本实现冻结整个 B0，并对 adapter 输入显式 stop-gradient。
- 关闭 trainer 原有 oracle coverage 辅助项，避免 GT 梯度绕过自定义损失进入 adapter。
- 不动态寻找最近 mode；低横向终点固定对应 mode 18，高横向终点对应 mode 19。

## 一次运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONFIDENCE_GATED_GRADIENT_ISOLATED_CFB2_SERIES\run_cfb2_pipeline.py" --device cuda:0 --seed 1 --epochs 12
```

只跑 PFRA：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONFIDENCE_GATED_GRADIENT_ISOLATED_CFB2_SERIES\run_cfb2_pipeline.py" --device cuda:0 --seed 1 --pfra-only
```

查看完整命令但不运行：

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CONFIDENCE_GATED_GRADIENT_ISOLATED_CFB2_SERIES\run_cfb2_pipeline.py" --dry-run
```

## 预计耗时

- PFRA 检索、校准和缓存：本机实测约 15 秒，保守按 0.5～1 分钟；
- General + Challenge 两次 12 epoch：参考 CFB1 实测，约 18～26 分钟；
- 完整诊断：约 2～4 分钟；
- 合计保守预计：约 21～31 分钟。

若 PFRA 不通过，脚本会在训练前自动停止。

## 晋级标准

- General/Challenge 的 minADE20 或 minFDE20 至少改善 0.5%；
- K=5/10 保留 CFB1 增益，K=20 不再退化；
- top-1 不明显变差；
- mode 0～17 输出与 B0 应接近数值零偏移；
- 两个 split 方向基本一致。
