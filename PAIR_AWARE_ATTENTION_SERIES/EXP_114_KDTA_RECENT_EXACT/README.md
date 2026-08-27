# EXP-114 KDTA Recent-Exact

针对 EXP-113 的 FDE 退化与采样位置诊断进行适配：每个目标—邻居对只预测一个早期连续时刻（初始 1.5、范围 `[0,3]`），同时精确保留最近两帧 6 和 7。

相比基础版 `[dynamic early, dynamic middle/recent, exact last]`，本版本为 `[dynamic early, exact 6, exact 7]`。目标是恢复终点所需的最近速度/转向，同时保持 pair-conditioned 早期事件选择和 8→3 token 压缩。

