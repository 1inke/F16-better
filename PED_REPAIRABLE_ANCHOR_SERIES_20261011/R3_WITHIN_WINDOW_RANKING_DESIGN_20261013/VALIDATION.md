# 验证记录：仅设计交付与合成代码测试

日期：2026-10-13（香港）。

## 实际执行

命令：`C:/coding_toobox/python.exe -B -m unittest -v test_core`。

实际结果：**15 tests，0.280s，OK，exit code 0**。这是unittest报告的时间，不是实际实验ETA。

通过项目：

- pointwise总20N、track等权、窗口P/N各半、U权重0。
- pair总权重与窗口质量一致，索引均是同窗口已知P/N，无跨窗口或U配对。
- 正margin公式与对称±差值BCE的标量、梯度、Hessian误差<=1e-12。
- 独立逐窗口逐pair循环核对目标标量。
- pairwise与pointwise：固定方向两eps=1e-4/1e-5的gradient/Hessian有限差分误差<=1e-8；这是合成方向导数检查，不声称所有真实参数/所有方向逐坐标FD。
- U特征修改不进入拟合pair；固定map下shared-history差值为0，其梯度只来自正则。
- 窗口公共平移与slot排列的理论不变性；固定并列physical排序；非法输入拒绝。
- 已证命中/未知上界与全N/含U漏选分解。
- 五比较共享bootstrap抽样、窗口加权、99%分位数、重复调用确定性。
- 合成384形状的统计screen必须同时满足收益及change门；该fixture不是项目384窗口。

## 未执行，不能声称通过

没有调用`load_inputs`读取真实R1/R2数组，没有拟合任何选择器，没有调用Sklearn.fit、轨迹solver、Torch/模型前向，也没有worker/PID/资源/任务查询或自动化更新。

未进行真实384数据契约审核、全部继承来源SHA复核、真实统计重构、原R2pointwise逐bit复现、真实pair优化收敛、真实资源吞吐/ETA、科学OOF/CI结果、保存及独立全量审核。`frozen_inputs.py`为核心适配器，真实端到端路径尚待执行阶段验证。

当前已有设计/协议入口SHA来自只读指定旧文件哈希，不是完整3244来源/全部历史模型复审。原旧科学文件和结果未改变。

## 交付过程记录

第一次添加evaluation/test的审批因自动审批服务用量限制失败，操作未执行；用户随后“继续任务”后通过同一require_escalated审批路径补齐文件。没有绕过权限、覆盖旧结果或启动实验。此前已经写入的设计、protocol、core、adapter保持。

同助手ARS内联检查已经写入设计，非外部审核。统计实现测试不证明Bootstrap的独立性或外推效力，标签未知性/样本选择/三折训练重叠/重复探索限制仍适用。
