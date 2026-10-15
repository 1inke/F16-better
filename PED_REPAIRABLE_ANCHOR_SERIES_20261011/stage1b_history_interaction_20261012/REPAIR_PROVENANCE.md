# 预检1历史置换构造修复来源

2026-10-12。首次预检通过真实观测对齐、actual128 scalar/vector EXACT、两arm513重复fit/独立NumPy stationarity+Hessian reference+FD、1033 throwaway fit。失败仅在fixed ID-hash cyclic shift无法产生全不同track的置换。没有生产worker、OOF H4或保存训练模型，没有按科学结果选择置换。

保留首次features/DESIGN/protocol/preflight/common源码及原PREFLIGHT_FAILURE原位置；修复为identity-only SHA track分块+块内ID排序+固定maxblock循环偏移，预先要求maxblock<=N/2并实际断言全donor异track。所有候选和GT不参与构造。模型/数值门/5pp和CI/标签划分/资源不变。完整重新预检。
