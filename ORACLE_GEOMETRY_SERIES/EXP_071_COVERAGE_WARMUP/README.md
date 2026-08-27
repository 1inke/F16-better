# EXP-071 覆盖损失渐进启用

固定 coverage weight 0.15、temperature 0.10，在前5个epoch线性增加覆盖损失权重，避免训练早期候选责任分配不稳定。
