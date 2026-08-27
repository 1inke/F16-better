# EXP-121 MQ-1 Geometry Query

固定F16、K20 anchors和现有有界endpoint/四阶路径residual。唯一变量是将每个固定K20 anchor的longitudinal、lateral、radius和angle通过小型零初始化MLP，加入已有learned mode embedding。

该实验不改变候选数、不学习anchor坐标、不增加第二套refinement，也不改变scorer输入形式。目标是判断显式几何先验能否稳定组织20个mode的责任分配。

