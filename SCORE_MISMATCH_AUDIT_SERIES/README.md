# EXP-160 Score–Trajectory Mismatch Audit

本目录由 EXP-158/160 联合审计脚本生成结果。它冻结 H3、V2、M2 的全部参数，衡量旧 score head 与各阶段候选几何质量之间的同步程度。

主要输出包括 Spearman score-quality correlation、best-ADE/FDE/combined winner hit、Top-1/2/3/5/10 recovery、regret 分位数、score entropy、active modes 与 top-mode histogram。

只有该审计确认评分失配后，才进入 EXP-161 frozen candidate listwise scorer。

