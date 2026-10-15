# 从局部修复转回候选生成：一次大分支快速筛选

日期：2026-10-14。当前用户要求重新审视主线并快速测试；不是续开 R5。

## 研究问题与范围

目标是同一完整候选 ADE≤0.2m 且 FDE≤0.4m 的覆盖、变化行为的质量和物理合理性，不以可修复标签 Hit@4 替代生成能力。

本次先用一次无训练的结构诊断区分：错误主要落在去向/位移，还是在到达同一终点的时间过程。不是无止境的诊断链：此筛选结束后给出直接生成器的单次短训练方案，不继续 R6 排序、特征、修复器标签探针。

输入仅为方向3 sum/seed42 已保存的 outer1035（primary）、inner1856、original_CAL2879。seed43/44 的 native/GT 是同一输入，不重复当成独立模型证据。保留全部窗口/三运动组/scene，不选择有利难例。不新读取 held-test/fold/external 数据。

## 在读取新诊断结果前固定的四个端口

所有坐标是原保存的末观测点为原点的 focal 坐标，dt=0.4s。

1. `native20`：原始20条候选。
2. `oracle_endpoint_native20`：q[k,t]=p[k,t]+(t/12)*(GT[12]-p[k,12])。只补偿整体位移；无 .20m 局部界、不选slot。末端精确对齐，离散二阶/三阶差分理论不变，但速度及与最后观测速度的衔接不受保护。它不是全体目标条件生成器的性能上界，也不证明可预测真实终点。
3. `oracle_endpoint_line1`：q[t]=(t/12)*GT[12]。知道去向但匀速直线到达，只有1条不同路径；不重复20次伪造多样性。
4. `oracle_waypoint3_1`：固定 t=4,8,12 的 GT 点加起点0，分段线性插值。只有1条路径，提供了中间未来信息。测试低维位置程序的重构能力，非新架构效果、非 GT-free 预测，分段拐点可能造成 jerk。

不得混淆 oracle 和学习结果，不将不同未来信息量/不同K解释为公平模型比较。终点误差被人为消掉，需重点看 ADE/同路径覆盖以及原生失败的 rescue/lost。

## 评价

- native 的 FDE≤.4 的端点可达比例、ADE≤.2 比例和同路径 AND；同时报告独立 minima 拼接的假成功数量。
- 四端口 full population/group 的 coverage、rescue/lost、minADE/minFDE/J、同一 Jclosest 早中晚误差。
- 原 SciPy complete-link 距离 max(meanpath/.10,endpoint/.20)，cut=1；effective-good 不追求越多越好。
- 五项 pooled motion：speed p95、acceleration mean/p95、jerk mean/p95，literal≤1.01×native，原标准不放宽。GT 自身同口径只作参照，不据此修改门槛。
- 每split/group whole-track bootstrap 2000次，固定 RNG20261014；探索性95%区间，不把三split当独立重复，不据此宣称确认性突破。
- 保存完整构造路径、ID、GT、group、track、scene、指标；独立标量插值/误差和第二差分实现校验。不导入旧运行器、不实例化旧模型。

## 解释规则

- endpoint-native 在原生失败中大幅 rescue：目标/位移分支有结构空间，不证明预测目标容易。
- 已知 endpoint 的 line 仍差、waypoints 改善：终点不足，需生成时间过程/路径形状。
- oracle coverage 高但 motion FAIL：不能作为安全部署候选；应设计连续速度/曲率过程，不继续任意 XY 修正。
- 上述均不证明历史信息不存在、心理意图已被理解、F16 decoder 是唯一根因。

## 运行与完整性

一次同步短 CPU 脚本；process≤6GiB、physical/commit各留6GiB，内部检查，无OS/pagefile调整、无旧PID查询。实际输入与父 RESULT 已注册 SHA 一致，前后复核；本地新增代码/设计 SHA 锁定。只核验本轮相关文件，不宣称全历史所有文件复核。
不创建 automation，不恢复暂停E3，不启动神经训练。无GT-free模型效果的声称，不增加formal计数、baseline或完成原研究目标。

ARS研究设计与反方检查由同一助手完成，非外审。主要反驳已纳入：oracle的未来信息远多于真实模型、变换非完整表示上界、路径平滑不等物理可行、重复评估人口不是新确认集。
