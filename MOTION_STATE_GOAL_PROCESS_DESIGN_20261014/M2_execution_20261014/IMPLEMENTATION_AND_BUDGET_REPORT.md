# M2 实施与预算报告（未启动生产）

日期：2026-10-14，香港。状态：**数值/实现预检 PASS；完整生产规划超30分钟，STOP_BUDGET_WAIT_HUMAN。**

## 1. 已完成什么

按原目录和事前规则实现M2，不是只写了设计。M1已完成且S1 FAIL，父来源固定为seed42/43/44各自原G1，没有选M1-A、seed44或CAL结果，也没有重训G1。

本目录新增9个Python实现文件和2个PowerShell启动/核验脚本，涵盖数据与来源隔离、模型、原质量/运动学、训练保存、20000次统计、数值恢复预检、独立CPU保存审核及唯一OS worker流程。源码静态AST解析通过。当前没有正式worker、PID、Windows任务、生产checkpoint、25epoch结果或预计完成时刻；启动与全量审核代码尚未经历生产执行。

预检实际运行55.929秒，最后在预算断言以exit1停止。这个退出不是数值失败或训练失败，更不是科学S2/Q2 FAIL；S2/Q2/major均**尚未评价**。没有重复预检、追加优化、降低预算系数或偷偷减工作量。

## 2. 方案和已验证的关键项

共同原G1/context/query/centres/scalers，A只训练过程，B额外训练新goal head。实测两臂总参数33306，A可训练7000、B12314。B额外5314可训练参数是重要混杂，不能把未来收益简单解释成纯过程理解。

过程84→64→24 GELU：输入h64、query16、固定FIT标准化预测goal2与观测v0_2；NNfloat32，积分float64。H12、dt.4、w12..1、S650，a=b+(w/650)[(g−4.8v0)/.16−Σwb]。不使用GT目标、额外幅值、clip或新物理loss。过程自由度22，不是24满秩。

b=0时两臂为同一预测目标的最小平方加速度路径，不是原生或G1直线identity。计划完整保存12个P0与24个最终科学job；所有FIT18909/outer1035/inner1856/CAL2879人口保留。

实际预检证据见 [PREFLIGHT.json](PREFLIGHT.json)：

| 项目 | 实际结果 |
|---|---|
| 三seed父来源 | 实际GPU128 G1保存目标 EXACT；两臂全部初始state和输出 EXACT |
| 实际GPU8×FIT256 | 两臂全部合法可训练参数梯度按zero-last分阶段正有限；A父无梯度，冻结state EXACT |
| B目标双链 | goal输入标准化和endpoint投影路径均有正有限连接；梯度范数约.02805和1543.75388（不同诊断量，不当训练收益） |
| canonical原loss | AST标量与全部输出坐标梯度 EXACT；独立NumPy标量误差0 |
| 独立投影 | NumPy显式时序积分/torch结果最大差7.1054e-15；CV、nullspace方向和S650核验PASS |
| 实际保存几何 | 两臂actual128保存/重读/投影独立重构最大1.1546e-14，小于原阈值，不是科学评价epsilon |
| 参数有限差分 | 固定两eps1e-4/1e-5；process及Bgoal参数误差最高5.9769e-10<1e-8，无winner/hinge活动面切换 |
| 真实恢复 | 两臂FIT513+inner12子集两轮连续/中断，ALLparams/buffers/optimizer/Torch CPUCUDA/loader/NumPy/PythonRNG/history EXACT |
| 拒绝错误来源 | done SHA-skip、坏source/output及resume合同真实拒绝 |
| 保存质量 | 两个actual128 smoke的samecandidate质量、原SciPy ALLK complete-link和五pooled运动学独立复算一致 |
| FIT统计 | 每seed初始G1 FIT全部378180目标，v0每窗口一次；31/256累加差最大4.4587e-13<1e-11 |
| 来源 | 原M1全部3746相关原digest继承不覆盖，并扩107输出/报告/补充/失败/新代码等，共3930源初末SHA核验PASS |

子集两轮不是完整18909人口两轮；数值FD不是float32量化导数证明；保存算术不是所有模型和梯度第二次回放，也不是外部评审。没有实例化或训练旧F16/原decoder/scorer，只读取immutable cache和新G1父模型。3930相关源不是全部历史34k文件。

预检源表与SHA见 [preflight_source_snapshot.json](preflight_source_snapshot.json)；这只是保留的预算STOP证据快照，**不是生产source-lock**。未生成source_manifest.json，不修改已注册科学协议或阈值。

## 3. 为什么不能按短预算启动

[实测BENCHMARK](BENCHMARK.json)在每臂4次warmup后测32次FIT256更新，不保存benchmark模型为科学结果，不按效果选方案。

| 项目 | 当前实测/估算 |
|---|---:|
| A每FIT256更新 | .0167242秒 |
| B每FIT256更新 | .0222887秒 |
| 6run、11100更新估算 | 216.522秒（3.609分钟） |
| 保存/重读双算术每窗口A/B | .00696395/.00761725秒 |
| 独立审核每窗口A/B | .00103477/.00106476秒 |
| 全36job保存、几何、quality聚簇、motion及独立审核估算 | 1955.358秒（32.589分钟） |
| 统计与独立重构估算 | 123.697秒（2.062分钟） |
| 初始相关source SHA核验 | 15.959秒 |
| 含终SHA的nominal | 2327.495秒（38.792分钟） |
| 原统一25%余量+150秒后规划 | **50.989分钟** |
| 吞吐范围（非保证） | **35.693–76.484分钟** |

这不是“训练要51分钟”，也不是已有运行任务的结束ETA。更大的b/a/v/minimum/null等全量保存、压缩读回、完整聚簇与审核占主要部分。当前36job含24最终＋12全量初始化，不能为了低于30分钟删FIT/P0、少跑epoch/seed、减少20000draw或跳过独立审核。

M1实际比其规划快，不能因此把M2的已注册预算门判成通过；M1“允许启动”的单项约40分钟例外也不能延用M2。此次没有因超预算继续优化/重跑计时，也没有以nominal替代完整规划来启动。

## 4. 资源、证据和当前状态

预检private峰值2.68436GiB、RSS1.80303GiB；系统physical最低12.89955GiB、commit最低16.99614GiB。保持process6GiB、两系统余量各6GiB，无OS/pagefile/用户程序干预。这是预检峰值，不是未发生的生产峰值。

预检时diskfree40.879GiB，基于实际128保存体积保守估算本项新artifact8.621GiB，加额外2GiB余量的磁盘门通过；资源/磁盘不是本次停止原因。

保留 [failures/preflight1/FAILURE.json](failures/preflight1/FAILURE.json)、原代码、smoke、子集恢复checkpoint、三seed统计、原预检与吞吐文件。旧科学/source/model/result及M1结果不变。没有正式run，没有S2/Q2/major成绩，没有新增formal完成数，M1仍1阶段完成、整系列未完成，baseline/historycycle11/unfinishedgoal不变。

本次20:00一次实施节点已完成，因没有合法生产启动，不建立half/completion节点；仅移除本次已过期m1跟进，旧E3及automation/automation-2 PAUSED保持，不查询、恢复或kill。

## 5. 下一步需要的具体决定

保留当前协议继续生产，需要**明确批准M2这一项约51分钟规划、吞吐范围约36–76分钟的预算例外**；这不是重新询问已给过的系列权限，而是超过现行30分钟上限所需的新决定。未收到该决定保持停止，不自动重试、轮询资源、创建worker或新事件。

若人类不希望投入该预算，则保持M2未执行并标明预算原因，不靠缩小数据/seed/审核或更改门限把它包装成“完成”。若之后获准，须先确认当时来源/预检兼容与资源，再形成新的显式启动预算合同，唯一OS任务实际启动后才给结束时刻和一次半程；不覆盖这份STOP来源。

## ARS反方核查1（同助手inline，非外审）

最强反驳：联合臂额外可训练5314参数且可调整终点，未来覆盖改善可能主要来自目标变化而非过程生成能力。方案已预定P0/final、endpoint、minimum/null过程、同一完整slot和motion分解来限制结论，不宣称纯容量因果。

次要限制：20个预测终点的仿射投影只保证离散代数，不保证a1=0、v1=v0、五motion、安全或迁移；所有原1%运动学门保留。当前无科学结果，因此不产生DA2/3虚构的结果分析。使用academic-research-suite组织预检核查，属于AI辅助、同助手自查。

证据：[设计](DESIGN.md)、[协议](protocol.json)、[数值预检](PREFLIGHT.json)、[吞吐](BENCHMARK.json)、[失败来源](failures/preflight1/FAILURE.json)。科学执行和保存审计脚本已实现但未生产验证，不能称完整系列已完成。
