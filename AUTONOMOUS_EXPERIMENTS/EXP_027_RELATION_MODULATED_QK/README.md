# EXP-027 关系调制 Q/K

固定 F10+K20。借鉴 Relational Transformer，不再只把关系 token 作为标量 bias 和 value，而是生成逐边、逐头的 Q/K 位移后计算相似度。投影零初始化，使训练从 F10 的注意力形式平滑起步。
