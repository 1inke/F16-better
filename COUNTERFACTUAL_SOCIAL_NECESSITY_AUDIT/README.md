# Counterfactual Social Necessity Audit（CSNA）

## 实验目的

在不训练、不改变参数、不删除邻居输入的条件下，判断当前F16社会上下文是否真实改善轨迹预测。

同一个IR1实验中的B0 checkpoint执行两次严格配对推理：

- **Full Social**：原始稀疏Top-K时序—社会交互。
- **Self-Only**：仅把焦点行人的接收边改为self-loop；其他智能体、张量形状、时间编码、解码器、候选数和参数完全不变。

组合oracle代价固定为 `ADE + 0.5 × FDE`，与当前checkpoint选择标准一致。

## 输出

- `results/combined_summary.json`：总结果与是否进入CSRS的判定。
- `results/{split}/summary.json`：Overall、高交互、十分位趋势、聚类bootstrap置信区间。
- `results/{split}/paired_samples.csv`：逐样本严格配对数据。
- `results/interaction_decile_social_gain.png`：交互强度十分位趋势图。

由于ETH/UCY窗口重叠，95% CI采用“场景×20帧连续时间块”的聚类bootstrap，不采用iid窗口bootstrap。

## 正式运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT\run_csna_batch.py" --device cuda:0 --seed 1 --splits all --bootstrap 2000
```

CSNA是零训练审计。只有General和Challenge的机制证据通过门槛，才编写并运行CSRS训练实验。

