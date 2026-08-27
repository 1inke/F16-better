# EXP-145：第一阶段意图查询均衡分配

本轮不再给冻结候选追加校正插件，而是重新训练第一阶段20个意图查询。使用批次级 Sinkhorn 最优传输，根据整条轨迹和终点误差为查询分配职责，并约束不同查询在批次中获得近似均衡的训练机会。

- **EXP-145A Balanced**：均衡意图分配基础版本。
- **EXP-145B Alternate**：以35%概率屏蔽当前最佳查询，迫使备用候选学习可行轨迹，检验候选覆盖不足是否源于备用查询缺少监督。
- **EXP-145C Balanced + Dynamic Progress**：均衡意图分配与EXP-144验证有效的轻量动态进度联合训练。

三组均从对应 split/seed 的 EXP-143B 开始，保留稀疏交互、运动学输入、20候选和无CV外推。编码器、交互模块及受控校正器冻结，只训练意图查询、锚点模板、候选生成/评分头；C额外训练动态进度头。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\INTENT_ALLOCATION_SERIES\run_intent_allocation_batch.py" --device cuda:0 --epochs 30 --batch-size 64 --seeds 1 2 3
```

默认执行3个方案 × General/Challenge × 3 seeds，共18次训练。支持断点续跑，不要添加 `--force`。
