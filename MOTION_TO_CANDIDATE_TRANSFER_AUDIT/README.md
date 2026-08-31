# Motion-to-Candidate Transfer Audit

该实验承接 Individual Motion Representation Audit：加速度信息在 F16 表征中仍可解码，因此先检查它是否被 V2 候选生成和锚点修正正确使用。

本实验不训练、不修改 checkpoint，复用 EXP-164 official V2 seed 1。它在 t4/t8/t12 输出：

- 候选半径、纵向/横向 spread；
- GT future geometry 与候选 envelope；
- source anchor 与 V2 corrected anchor 的最近距离及 oracle 改善；
- correction norm 和 cap-hit rate；
- 几何量对加速度/非线性的 source-inner 与 outer 响应斜率。

正式运行：

```powershell
python "C:\Users\linke\Documents\New project\MOTION_TO_CANDIDATE_TRANSFER_AUDIT\run_motion_to_candidate_transfer_audit.py" --device cuda:0 --folds all --seed 1 --batch-size 64
```
