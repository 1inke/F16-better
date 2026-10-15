# 模型诊断：首阶段零训练

2026-10-09。依据用户提供的 ped_model_diagnosis_and_understanding_plan.md，保留其建议与执行权限的区别；目前优先已有模型/输出、跳过长训练的限制继续有效。

## 本次实施

D1a 全人口朝向兜底/起步贡献；D2 严格同路径混淆表与 top1/meanK 残差收缩解析诊断；D3 逐窗口 Jensen 分散度、固定物理箱、CV失败区分AUROC；D6 与平方距离medoid/最近CV/随机期望对照及窗口内AUROC；D7 新位置-only K32与已审计速度/位置速度K32对照、train/test分布距离；D5c 保存的真实 noNbr 输出响应随原CV近距变化。

五fold三seed逐一报告 all/P1，track聚类2000次bootstrap，不把重复seed/window视为独立数据集。所有旧文件只读、SHA前后校验，新增仅本目录。无torch、optimizer、模型前向或恢复旧E3，无OS任务/自动化激活。

## 解释约束与反方检查1（同助手，非外审）

- λstar由测试GT算出仅诊断，不是合法调参或部署收益。λ<1不单独证明方向正确；必须同时看内积符号、损害/救回及分母。
- CV严格成功并非旧训练分位数CV_success；两种名称不得混用。严格同路径阈值不放宽。meanK不是集合内slot，也没有自动运动学安全。
- best-of-K候选不是后验抽样，分散度/误差比≠1不是校准失败的充分证据。固定物理箱允许为空，不根据测试好坏改箱。
- medoid使用全12时刻平方L2，相当于最接近meanK的原候选，不是欧氏meanpath medoid。窗口内good/bad AUROC与跨窗口总体AUROC不可混淆。
- 位置-only kNN不是地图模型；跨场景坐标不对齐。KS仅分布距离，无iid显著性；输入数量与训练分布是替代解释。
- D5c沿原closest-away投影，非严格侧向；移除邻居改变输入且可OOD，GT不是邻居反事实。
- 小探针失败不能推断本质不可预测；A≈−1也可由同负向偏置产生；A≈0可能响应都近零或不同窗口抵消。附文这两条判读需收窄。
- T1/T2尚未在首阶段执行；T2必须另做同GPU真实cache对齐和全参数固定预检，不把之前E8 CPU STOP绕过去。T3长训练跳过，不恢复暂停90fits。

## 预检与完成

数值fixtures：同路径分类/λ解析/相同候选tie AUC/medoid身份/邻居32全tie排序、资源检查。按已锁done及历史manifest验证实际使用文件；模型/缓存/统计不改。保存逐window sufficient statistics与选择指标、所有摘要和新位置-only邻居IDs。另CPU从保存统计独立重算摘要/部分实际nearest brute，明确不是所有模型再次重放。结尾报告完成/未完成表、数据/反例/限制和下一优先方向。6GiB进程及physical/commit各6GiB余量；最多600秒CPU计算，不使用外部数据、不改变baseline/cycle11或wholecomplete。
