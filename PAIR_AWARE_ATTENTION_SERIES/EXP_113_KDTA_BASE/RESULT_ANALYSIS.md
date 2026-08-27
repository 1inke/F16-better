# EXP-113/114 KDTA 初步诊断

## 结果

| 方法（scene 1/3 样本加权） | minADE20 | same-path FDE | minFDE20 |
|---|---:|---:|---:|
| F16 | **0.177130** | **0.390575** | **0.331855** |
| KDTA 基础 | 0.178241 | 0.401070 | 0.336351 |
| KDTA recent-exact | 0.178134 | 0.399676 | 0.336154 |

## 采样位置诊断

- scene 1 最佳 checkpoint 为 epoch 5，采样均值约 `[1.529, 4.506, 7]`，几乎未离开初始化；
- scene 3 最佳 checkpoint 为 epoch 30，采样均值约 `[0.355, 5.769, 7]`；
- scene 3 的早期采样大量靠近左边界，第二点靠近第 6 帧；
- 基础版两个场景的 same-path FDE 和 minFDE 都退化，说明插值中后期 token 不足以保持近期动态。

## recent-exact 适配

改为 `[dynamic early, exact 6, exact 7]` 后，scene 1 minFDE 从 F16 的 0.211124 改善到 0.209177，但 minADE 和 same-path FDE 仍弱；scene 3 Oracle 仍退化。这证明最近两帧保护方向正确，但单个早期事件不能完整表示路径趋势。

下一步 EXP-115 增加无参数 early trend mean，与 dynamic event 和 recent-exact 共同组成 4-token K/V。

