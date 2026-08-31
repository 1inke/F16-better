# H3-to-V2 Correction Attribution Audit

该实验在同一次 V2 推理中读取 H3 coarse trajectories 与 V2 corrected trajectories，使用相同的 20 个 mode index 和相同 score，区分 correction 的真实轨迹收益、oracle winner 切换和评分同步问题。

不训练、不修改 checkpoint。主要输出：

- high-acc / non-high-acc 的 H3 与 V2 minADE、minFDE、same-path FDE、J；
- 固定 source winner 和 fixed corrected winner 的 same-mode correction gain；
- t4/t8/t12 point-oracle 与 same-mode gain；
- winner change、mode improved fraction、top1 及 oracle winner score rank；
- H3 inner 与 V2 outer 官方复现检查。

```powershell
python "C:\Users\linke\Documents\New project\H3_V2_CORRECTION_ATTRIBUTION_AUDIT\run_h3_v2_correction_attribution.py" --device cuda:0 --folds all --seed 1 --batch-size 64
```
