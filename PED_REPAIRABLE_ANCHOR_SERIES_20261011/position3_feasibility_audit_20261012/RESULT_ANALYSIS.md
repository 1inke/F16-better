# Position3 真实可修复性审计：结果与下一步
日期：2026-10-12。状态：**完成并复核；NONFORMAL 诊断，不是新模型训练或 S2。**

## 1. 结论
**现有 position3 输出结构，在部分现存失败窗口中确实可以产生合格、安全的修复路径；目前的问题不能简单归结为“这个结构根本修不动”。但现有 S1b 寻址仍没有证明能可靠找到这些锚点。**

固定64个有包络潜力且 native20 全失败的训练侧窗口，检查88个潜力锚点、176次有界数值求解。42个窗口得到独立核验的安全见证；12个只找到质量达标但运动学不达标的路径；10个仍未知。

这不是“新模型覆盖率提高65.6%”：样本先筛过机会窗口，修复使用未来GT逐例求解，不是可部署策略，也没有训练共享神经网络。

## 2. 实验设计
- 数据：只用既有训练侧A7334，不用B/outer/inner/CAL求解。64窗口来自56条目标轨迹；组内轨迹不重复，跨组可重复。
- 抽样：固定SHA256标签 position3-feasibility-v1，变化32、过渡16、稳定16；所有被抽中窗口都 native20 无同路径 .2/.4 合格候选、且至少一个潜力锚点。
- 每个窗口全部潜力锚点逐个检查，不根据S1b得分选案例。每次只修改一条原生路径，其余19条保持精确不变，K=20。
- 保持原 position3：3节点(4,8,12)、原tanh/L2控制、原全路径lambda、每步位移界 `.20*(t/12)^2`、float32 raw/float64路径。
- 两种固定求解：shape只约束形状与位移；pointwise增加编辑路径逐时刻v/a/j≤原路径1.01的更强内域。两者都以同路径 `J=max(ADE/.2,FDE/.4)` 为目标，SLSQP零初始化、150次迭代上限、ftol=1e-9，无重启或调参。
- 最终安全标准始终是**实际重构路径** ADE≤.2 AND FDE≤.4、位置界以及完整K20的五项运动学≤native1.01；求解器容差不是科学评价epsilon。
- pointwise内域比最终科学门更严；失败既不证明原门下无解，也不证明四锚联合修复无解。

### 核心代码
```python
# 六维逐例控制，不训练神经模型
raw32 = atanh(clip(unit_controls, -1 + 1e-7, 1 - 1e-7)).astype(float32)
path = original_position3_decode(native_anchor, raw32)
bank = native20.copy()
bank[physical_slot] = path
safe = (
    same_candidate_ADE(path, GT) <= .2
    and same_candidate_FDE(path, GT) <= .4
    and every_step_residual_within_original_bound(path)
    and all(five_motion(bank) <= 1.01 * five_motion(native20))
)
# 第二种独立重构也必须通过；没有safe只记UNKNOWN/QUALITY_WITNESS_ONLY
```
实际代码：[numerics.py](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/numerics.py>)、[run.py](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/run.py>)、[audit_saved.py](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/audit_saved.py>)。此处GT只用于离线存在性审计，不允许直接当部署输入或自动转成训练teacher。

## 3. 分组结果
| 预先固定的组 | 窗口数 | 安全见证 | 仅质量见证 | 未知 |
|---|---:|---:|---:|---:|
| 稳定 |16|11|3|2|
| 过渡 |16|10|2|4|
| 变化 |32|21|7|4|
| 合计 |64|42|12|10|

42/64=65.625%只是此选择性诊断样本中**已证实的可行比例下界**；不是总体估计，另外22个也不是已证实不可修复。
12个quality-only窗口的保存路径均实际未过运动学：9个涉及加速度均值和jerk均值，2个涉及jerk p95，1个涉及加速度p95。没有用浮点边界分类分歧解释掉这些失败。

### 求解状态不能代替科学审核
- shape：88次，85次返回收敛成功；31窗口找到安全路径。
- pointwise：88次，78次返回收敛成功；29窗口找到安全路径。
- 两者安全窗口交集18、并集42；pointwise另外找到11个shape未找到的安全窗口。
- 3条返回非成功码的候选仍通过保存路径的严格安全核验，因此可作为存在性见证；不称它们为最优解。
- 未收敛、约束未满足或未找到合格路径均不构成不可行证明。本轮没有全局对偶不可行证书。

## 4. 保存的GT诊断组合确实通过样本组运动学
每窗口按事先规定的slot升序、shape优先选首个安全见证，其他窗口保持native。此组合依赖GT，是诊断oracle，不是推理策略。
每个成功窗口仅编辑一条；所有原窗口本来全失败，因此新增合格路径来自真实修复，不是删除不利样本。

| 组 | speed p95变化 | accel均值变化 | accel p95变化 | jerk均值变化 | jerk p95变化 |
|---|---:|---:|---:|---:|---:|
|总体|+0.0000%|+0.0175%|+0.0203%|+0.0175%|+0.0000%|
|稳定|+0.0000%|-0.0112%|+0.0815%|-0.0112%|+0.0000%|
|过渡|+0.0000%|-0.0193%|+0.4744%|-0.0193%|+0.0000%|
|变化|+0.0000%|+0.0530%|-0.0848%|+0.0530%|+0.0000%|

全部四组五项均通过原1%上限；最大正变化为过渡组加速度p95 +0.4744%。单窗口安全不自动推出pooled分位数安全，因此这里另做了完整聚合复核。
GT-free K20原SciPy complete-link（.10 meanpath/.20 endpoint）也已重算；总体有效good簇均值 .65625、distinct簇均值19.90625，不是复制轨迹凑覆盖。

诊断oracle的总体 minADE .208147→.179302m，minFDE .350165→.261529m；变化组 .211106→.181186m / .369620→.277493m。minADE和minFDE各自可能来自不同slot；**42个安全见证采用的是同一candidate同时过两项阈值**，不能用独立minima替代。

## 5. 现有寻址离这些已找到的锚点有多远？
下表只问“所选4锚中是否至少包含一个本轮已经证实可安全修复的锚”。它不是实际训练后覆盖率，也不是完整真实repairability召回率。

| 组 | 有安全见证窗口 | S1b interaction命中 | 固定随机4命中 | bottom-logit4命中 |
|---|---:|---:|---:|---:|
| 稳定 |11|3|3|1|
| 过渡 |10|1|3|0|
| 变化 |21|8|6|3|
| 合计 |42|12|12|4|

S1b在42个已证实机会窗口中只命中12个（28.57%），漏掉30个；随机4同样12个，bottom4只有4个。变化组8比6是小样本描述，不声称显著胜出。
这与此前S1b全A OOF gate FAIL一致，**不能因为oracle好看就跳过S1门直接S2**。此外，本轮已经看过所有这些GT结果，因此不得把本样本冒充后续调参的独立验证集。

## 6. 如何更新判断与下一步
1. **保留position3作为一个可行结构基线。** 已有42个严格见证，暂不以“结构容量绝对不足”为由扩大head或再扫节点/幅度。
2. **优先研究寻址目标与表征，而不是重训scorer。** 原potential标签只是乐观包络；已发现安全锚与当前排名脱节。下一项若推进，应单独预注册“真实安全可修复下界标签 vs 包络标签”的训练侧数据设计与目标轨迹OOF；不能把solver未找到的锚全部标0。
3. **区分寻址和共享映射的可学习性。** 本次只证明逐例控制存在，不证明observed-history能识别它，也不证明统一神经head能拟合。即使寻址通过，仍需后续单因素、匹配预算的学习实验。
4. **不立即把64个oracle案例拿来训练S2。** 样本太小且偏向包络机会、含自适应分析。先确定可靠标签/未知处理、正负与轨迹分离规则，再做短预算寻址验证。现有S1b失败门不被本次诊断覆盖。
5. 对12个质量-only和10个未知保留全部证据，不通过调宽物理门、加大位移界或多次重启把它们强行变为PASS。它们只能提示继续区分安全耦合与求解能力，不能认定不可预测。

本轮到此闭环；没有启动新的神经训练、S2、scorer或长任务，也没有恢复旧暂停实验。

## 7. 执行与审计
- 唯一零trigger任务，实际worker29872/parent2688：12:17:26.020957→12:18:34.368883香港，墙钟68.348秒；科学主程序68.125秒。
- 唯一startup12:17:39；唯一半程计划12:19:40.162、实际12:19:59.653（迟约19.5秒）发现双SUCCESS；没有重复completed PID查询或再安排完成节点。
- 2,030项**相关**科学源/既有结果digest初末不变，继承1993原digest和22个S1b输出；不是全部历史34k文件校验。
- 完整176解保存路径独立重构max 2.220446049250313e-16；独立历史差分公式对176解五motion误差0；完整64组合GT-free聚簇与同路径质量复核通过；所有保存科学输出SHA初末通过。
- 第一保存复核7.778秒；另一次CPU补充复核负责历史独立差分/各组oracle指标与求解退出分类。没有重复数值求解、模型前向或梯度。
- 生产private峰值 .171726GiB、RSS .190926GiB，最小可用物理16.15155GiB、commit27.83860GiB，未降低资源门。
- ARS设计与以下反方检查由**同一助手inline**完成，不是外部独立同行评审。独立算术指实现路径不同，不指独立研究人员。

## 8. 反方检查2与3
**反方2：最强反驳**——64个本来就按乐观潜力筛选，逐例看GT自由优化，安全成功不能说明真实预测容易。成立。报告只主张存在性，不外推覆盖提升、不宣称理解运动或SOTA。
**反方3：寻址是否真是唯一瓶颈？**——不能。12个质量-only提示安全耦合，10个未知提示有限求解局限；历史输入是否足够、共享函数拟合、统计泛化仍未回答。S1b与随机相同也不证明所有非线性寻址无效。
当前可支持的最窄结论：**部分失败窗口存在现有结构内的安全修复，但现有选锚方法没稳定找到这些机会；下一步应先检验更贴近真实修复的寻址，而不是无条件扩生成器或改scorer。**

## 原始证据
- [预注册设计](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/DESIGN.md>)
- [实际结果](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/RESULT.json>)
- [完整保存复核](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/completion_review/REVIEW.json>)
- [补充运动学与求解状态](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/completion_review/SUPPLEMENT.json>)
- [逐窗口分类与安全slots](<C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/position3_feasibility_audit_20261012/completion_review/CASES.json>)
