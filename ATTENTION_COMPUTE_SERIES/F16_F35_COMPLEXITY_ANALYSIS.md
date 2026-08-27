# F16 与 F35 注意力计算复杂度核算

## 固定设置

- 最大行人数 `N=9`
- 观测长度 `T=8`
- token 维度 `d=96`
- attention heads `H=4`，每头维度 `dh=24`
- F35 双段池化后 `L=2`
- 参数量：F16 与 F35 均为 284,642

以下只计算 temporal-social attention 核心的主要乘加，不包含共同的 temporal encoder、关系编码、运动学图、门控和 K20 解码器。

## F16

| 操作 | 主要 MAC |
|---|---:|
| Q projection | `N d² = 82,944` |
| K/V projection | `2 N T d² = 1,327,104` |
| QK 与 weighted-V | `2 N² T d = 124,416` |
| output projection | `N d² = 82,944` |
| 合计 | **1,617,408** |

attention score 元素数（不计 batch）：`H N² T = 2,592`。

## F35

| 操作 | 主要 MAC/操作量 |
|---|---:|
| 前后半段均值 | 约 `N (T-L) d = 5,184` 次加法，另有缩放 |
| Q projection | `N d² = 82,944` |
| K/V projection | `2 N L d² = 331,776` |
| QK 与 weighted-V | `2 N² L d = 31,104` |
| output projection | `N d² = 82,944` |
| 合计（将池化加法近似计入） | **约 533,952** |

attention score 元素数：`H N² L = 648`。

## 理论变化

- temporal-social attention score 张量：**减少 75%**。
- K/V token 数与 K/V projection：**减少 75%**。
- temporal-social 核心主要操作：约从 1.62M 降至 0.53M，**减少约 67%**。
- 参数量：**不变**，因为双段池化无参数，Q/K/V/out 结构仍保留。
- 完整模型峰值前向显存实测：58.44 MiB → 54.58 MiB，**减少约 6.6%**。

核心注意力减少 67% 不代表完整模型 FLOPs 也减少 67%。temporal encoder、鲁棒状态编码、关系特征、运动学图和 K20 解码器没有变化，所以端到端收益被稀释。

## 延迟实测

独立顺序测试会因 GPU 时钟漂移产生相反排名。最终采用 18 轮、每轮旋转模型顺序的交替基准：batch=1 时 F16/F29/F35/F36 均约 10.55 ms；batch=64 分别约 13.20/13.40/12.94/13.84 ms，波动区间明显重叠，没有稳定显著加速。

因此当前只能可靠宣称 attention 核心理论规模下降和峰值显存下降，不能宣称明显端到端加速。论文或汇报中必须分别报告理论规模、峰值显存和实测延迟，不能用 75% attention-axis reduction 代替端到端加速比例。
