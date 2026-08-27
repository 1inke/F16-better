# EXP-030 Agent Proxy Attention

由 Agent Attention 启发。固定 F10 的稀疏成对交互，额外使用两个代理 token 以 O(NP) 代价汇聚全局群体信息并广播回节点。它不恢复全连接 N² 注意力，用于检验稀疏局部图是否遗漏远距离群体趋势。
