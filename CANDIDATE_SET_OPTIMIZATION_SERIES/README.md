# CSO1：Winner-Protected Candidate-Set Optimization

## 对建议的评估

方向合理，但当前 B0 已经包含 endpoint margin diversity loss。因此本轮不能简单把 repulsion 再加一次，而应验证真正新增的两个机制：

1. 用 winner-protected repulsion **替换**原来的 all-pair repulsion；winner 只接受单 winner GT regression。
2. 在此基础上加入极轻量的跨样本 mode prototype consistency。

Prototype 可能与 winner regression 冲突，因此拆成两个递进变体，避免把两个变量捆绑后无法解释结果。

## 实验变体

- `CSO1-R`：`cso1_winner_protected_repulsion`。只替换为 non-winner endpoint squared-hinge repulsion。
- `CSO1-P`：`cso1_repulsion_prototype`。CSO1-R + winner mode 三锚点位移 EMA prototype，权重 0.01。

两者均：

- 保留单 winner trajectory regression；
- 不使用 Top-M soft regression；
- 不新增 decoder branch；
- 不直接改变推理轨迹；
- 保持 F16、稀疏图、Mode + 3-Anchor、20 modes 和现有优化设置；
- 复用严格匹配 B0。

## 运行

```powershell
C:\coding_toobox\envs\astra2\python.exe "C:\Users\linke\Documents\New project\CANDIDATE_SET_OPTIMIZATION_SERIES\run_cso1_batch.py" --device cuda:0 --seed 1 --splits all --variants all --epochs 12
```

将新增四次训练：两个变体分别在 General/Challenge 运行。预计约 22–30 分钟，完成后自动生成严格配对诊断。

## 自动诊断

- minADE/minFDE/Same-path/Top1；
- score-ranked ADE@K、FDE@K，K=1/2/5/10/20；
- endpoint pairwise mean/median/P10/P25/P75 和 margin 内比例；
- endpoint/trajectory/anchor diversity；
- 4/8/12/endpoint nearest coverage；
- winner entropy、winner usage 和 dead modes。

## 晋级标准

至少一个主要 oracle 指标改善 0.5%，另一项不得明显退化；或 ADE@10/ADE@20 出现一致改善。同时要求 diversity/coverage 不下降、winner usage 不更集中且不增加 dead modes。
