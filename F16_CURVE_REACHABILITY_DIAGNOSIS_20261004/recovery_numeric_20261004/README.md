# 同科学协议的数值校验恢复

人类2026-10-04“继续执行任务”批准可靠性验证后恢复原分类诊断。
所有新增内容只在此子目录；父失败源码、日志、exit及171完成块原样保留。
原模型/F16/decoder/scorer/损失/候选质量/solver预算/数据全部不变。仅FIT25912/CAL2879，不读DEV/heldtest，不训练新模型。GT仅事后构造目标，不进预测forward。formal response study仍2/3。

固定维度无关的6e-6profile零和断言改为逐候选高精度参照及浮点传播误差上界；原2e-5m路径/终点、clip/几何下界、保存算术审核保留，并新增真实旧components同候选质量及window witness不变审核。已完成块SHA通过、同数据分类/有效去重计数重算与新校验通过后只读链接，不重拟合/覆盖旧块。

`diagnosis.py --preflight`执行全输入/资源/原model/source/math预检，171旧块完整算术及真实生产一致性、原失败64块复现丢弃、故意profile破坏和source错配拒绝。通过后由start_independent.ps1唯一独立按需0trigger任务启动。按实测ETA单次50%TIME检查与完成分析节点退出对话，不轮询。

约17GB可用hostRAM；过程6GiB，physical/commit各>=6GiB余量；64行块。任何实质方法/映射/资源/源锁阻碍即停止等待人类，不能放宽质量标准强行通过。
