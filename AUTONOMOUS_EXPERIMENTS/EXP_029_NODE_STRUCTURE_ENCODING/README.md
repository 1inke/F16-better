# EXP-029 节点结构编码

由 Graph Transformer 的 structural encoding 思路启发。固定 F10+K20，在社会注意力前，为每个节点编码稀疏图度数、邻居平均距离、平均风险距离和最大接近速度，并用低初始门控注入 agent token。测试“节点在交互图中的角色”是否是关系边编码之外的缺失信息。
