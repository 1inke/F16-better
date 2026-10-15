# M1 单项时间预算授权

2026-10-14，人类在获悉完整规划39.91分钟、估计范围27.93–59.86分钟、原30分钟门未通过后，明确回复“允许启动”。本文件只记录M1单项预算例外；不修改父预注册或旧PREFLIGHT.json的STOP_BUDGET历史，不放宽任何科学门，不减少人口/轮数/审核，不扩展M2/M3/V预算或解除旧任务暂停。

仅final_source_review.py与start_independent.ps1的启动/预算控制发生授权差异，旧版本保存在budget_release_provenance，旧3738源表保留budget_review/SOURCE_SNAPSHOT.json。全部科学代码、输入、参数/统计、loss、seed与25epoch必须与该快照SHA EXACT，才可复用已真实通过的GPU/子集两轮全optimizer-RNG/saved数学预检。新预启动审查只是CPU源码/资源核验，不冒称再次GPU预检。

本次source_manifest将同时保留原3738表的引用与逐项授权转换记录，不静默覆盖旧hash。实际worker唯一OS按需0trigger，唯一startup及一次50%预计耗时检查；完成必须双SUCCESS＋独立全saved审核，不能以训练epoch结束冒充分析完成。预计耗时不是强制终止健康worker的定时器。
