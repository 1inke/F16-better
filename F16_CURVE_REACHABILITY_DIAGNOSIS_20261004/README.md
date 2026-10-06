# 曲线可达性与实际学习差距诊断

人类2026-10-04批准执行先前NEXT_DESIGN；非正式第三次训练，response study仍2/3。自回归方向放下；预测节点后的GT不进入预测模型。

仅原FIT25912/CAL2879、native K20和既有单Gaussian脉冲族。GT仅用于每窗、每slot的事后构造，不改模型参数、损失、数据或旧结果。既有mean_std constrained三seed只读重放，forward只past/tokens/base/speed。所有新增代码、缓存、日志、失败与结果只在此目录。

运行`diagnosis.py --preflight`检查全输入内存、生产函数对齐、数值梯度、边界、独立路径重算、旧SHA和分析块恢复。随后仅`start_independent.ps1`启动零trigger任务。按实测ETA一次50%TIME检查，完成核验后解释分类，等待人类；不自动新训练。

过程预算6GiB，系统physical/commit各至少6GiB余量；64行chunk。旧21780清单基础上加全部上一诊断/recovery artifact，前后SHA核验，控制文件与锁/临时/bytecode排除。数学拟合不是模型训练，未找到不等不可达；原formalmajor和全组运动学保护不降低。
