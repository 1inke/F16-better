# EXP-136：非解码候选三种子复验

补跑seed 2和3，并与已有seed 1共同汇总。比较baseline、速度尺度状态编码和目标短—长期双时间分支，分别使用general与challenge冻结开发划分。

新增训练数为 `3个模型 × 2个划分 × 2个新种子 = 12`。已有seed 1自动跳过，不访问官方ETH测试集。

脚本结束后生成：

- `multiseed_summary.json`；
- `multiseed_summary.csv`；
- 各指标逐种子相对基线百分比变化。

