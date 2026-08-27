# EXP-038 F16 + Temporal DropKey

借鉴视觉 Transformer 的 DropKey。仅在训练时随机屏蔽 10% 的“邻居×历史时刻”key，并始终保留最后观测时刻；推理时与 F16 完全相同，无新增参数和推理开销。
