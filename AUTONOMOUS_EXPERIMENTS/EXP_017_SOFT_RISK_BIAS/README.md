# EXP-017：原稀疏图上的软风险偏置

不改变任何邻接边，不增加稠密注意力。根据 closing、distance、TTC+DCA 构造连续风险分数，并用每个注意力头一个从零初始化的可学习强度加入 attention logits。新增仅 4 个参数，初始化行为严格等同 F4。

