# 脉冲响应实验结果分析（2026-10-04）

## 结论

实验完整正常结束并通过完整性审核，但正式major标准未通过。可学习时间位置/跨度确实发生变化；相对原F16，变化组minADE/minFDE分别改善 **1.95% / 1.47%**。然而相对同结构固定时间对照，只有 **0.235% / 0.021%** 的平均增量，配对置信区间均跨0，三个种子仅seed1同时优于固定对照。因此不能认为“学习变化时间就把已有线索转化成正确响应”已经成立，更不能升级baseline。[results/summary.json：gates、aggregate、bootstrap]

严格同路径达标候选覆盖没有改善：469个明确变化样例，native92个、可学习三seed均88个；救回8/4/5个，却损失12/8/9个。平均误差的小幅改善不等于解决无法生成高质量候选的问题。[completion_review/REVIEW.json：temporal_rows]

## 完成与证据完整性

- 13:02:33.304375启动，14:27:33.339346（香港时间）worker SUCCESS，实际85分00秒。人类“分析结果”触发本次完成复核，直接读取完成worker及结果文件；未再次查询已完成PID/task，未重训。
- 6个全新配对训练，每个80epochs；18policy条目是3seed×2arm×3checkpoint策略，不是18个独立训练。
- worker和results exit均SUCCESS；independent_audit、saved_metrics_recompute均PASS。18checkpoint SHA、18显式1516IDs有限20×12×2输出、原decoder逐项冻结/原F16与scorer源锁、256source和19558旧文件SHA均通过。
- 全部保存sample指标、nativegroups、24ALLpolicygroup aggregates、72perseed group rows、96bootstrapCI、temporal数组、pulseprofile及终点分解独立重算通过，sample最大误差0。
- 本轮新CPU completion_review.py补充核验18预测minADE/FDE、同路径rescue/lost、条件事件统计及6×80history；839个生产输入/源/checkpoint SHA前后不变，min指标误差0。这是保存坐标算术，不是所有模型再次重放。19558全部旧文件核验来自已完成worker，未伪称补充脚本再次全核验。[results/independent_audit.json；results/saved_metrics_recompute.json；completion_review/REVIEW.json]

所有新增分析只在本实验新文件夹；未改变已有科学源码、模型或生成结果。

## 正式结果：原constrained策略，不用DEV另选checkpoint

| formal change组 | minADE（m） | minFDE（m） |
|---|---:|---:|
| 原F16/native K20 | .372175 | .696326 |
| 固定响应时间对照 | .365773 | .686225 |
| 可学习时间位置/跨度 | .364915 | .686081 |

固定对照相对native也改善约1.72%/1.45%。两新组都保留候选观测attention、终点漂移及零和pulse；无法将共同收益单独归因于pulse本身。仅本次预先锁定因素——学习center/span——没有可靠额外增量。[results/summary.json：aggregate]

三个可学习seed均相对native改善ADE/FDE；但相对配对固定对照，seed1好，seed2/3的ADE和FDE都更差。组平均不可替代配对重复一致性。[completion_review/REVIEW.json：per_seed_metrics]

变化组delta=新−参照（负为好）：

| 参照 | 指标 | delta（m） | 保存bootstrap CI |
|---|---|---:|---|
| native | ADE | −.007260 | [−.013994, −.002683] |
| native | FDE | −.010246 | [−.025965, +.000926] |
| native | joint同路径 | −.013132 | [−.027827, −.003890] |
| fixed | ADE | −.000858 | [−.006020, +.004005] |
| fixed | FDE | −.000145 | [−.010457, +.007591] |
| fixed | joint同路径 | −.002893 | [−.013651, +.005921] |

native ADE/joint存在探索性负向CI信号，但nativeFDE跨0、fixed三项均跨0，未满足正式条件。只有三配对seed及相关场景窗口、反复ETH DEV探索，不能当独立场景确认/人口置信区间/SOTA。[results/summary.json：bootstrap]

通过：保护组误差≤1%伤害、每seed相对native改善、CAL可行性、原coverage/重复/硬运动学/同路径joint保护。失败：双5%收益、native及freshcontrol CI、全组运动分布≤1%伤害。原标准完全保留。

## 用户关心的高质量候选：同时近GT，而非两个不同的最小值

以下固定描述标准为“存在同一候选ADE≤.2m且FDE≤.4m”，不是SOTA标准或新选模条件。formal change与clear_event是不同分组。这里clear_event来自原事件规则，共469例。

| clear_event469 | native | 固定seed1/2/3 | 可学习seed1/2/3 |
|---|---:|---|---|
| 存在严格达标候选 | 92 | 92 / 88 / 87 | 88 / 88 / 88 |
| 可学习救回native原失败 | — | — | 8 / 4 / 5 |
| 可学习损失native原达标 | — | — | 12 / 8 / 9 |

原缺少好候选的377例，仅救回约2.12%/1.06%/1.33%，且每seed净减4例。相同IDs在各seed重复，不能相加为17个独立被救回的人。全体1516例严格覆盖由native733变723/728/727；因此当前方法未改善用户的“无法生成高质量候选”覆盖目标。[completion_review/REVIEW.json：temporal_rows、paired_adequacy]

宽松.3/.5的clear_event计数native184，新组188/185/183，仅小幅且非一致变化；不能看严格标准失败就切换标准称成功。

## 时序响应是否真的学了

可学习constrained模型的center标准差约.0397–.0509、span约.0177–.0231，固定组center=.5/span=.275。响应幅度非零；保存profile/终点分解已核验，最大终点分解误差约3.58e−7m。不是分支未执行或输出恒零。[results/pulse/summary.json；results/saved_metrics_recompute.json]

但这只能证明响应参数在用，不能证明模型准确识别了真实事件时刻或持续时间。Gaussian中心/跨度并非语义onset/duration；同code终点与profile分离也不是最终两组训练终点匹配。

clear_event同一GT辅助jointoracle候选的早/中/晚误差：
native约.15797/.38998/.57512m；
可学习三seed均值约.15317/.38549/.57284m。
主要是小幅早中段改善，晚段仍约.573m，没有完整路径形状纠正的显著跃迁。这里使用GT事后选joint候选，不是冻结评分器部署argmax。[completion_review/REVIEW.json：temporal_rows]

已检测事件条件下onset误差由1.829steps变约1.717steps，同时检测覆盖native67.38% vs新均值约67.09%。不能把未检出事件onset99当实际时差，不能只报条件精度忽略缺失。转向方向需truth-turn331条件分母；不能用所有非转向0值制造高方向准确率。

## 副作用与学习/泛化

| constrained可学习 vs native | 平均加速度变化 | 平均jerk变化 |
|---|---:|---:|
| overall | +14.60% | +1.41% |
| stable | +12.06% | +1.12% |
| transition | +18.75% | +1.84% |
| change | +23.79% | +2.48% |

speedp95、accelp95及jerkp95增幅较小、硬速度/加速度极限通过，并不等于所有分布保护通过。添加pulse产生更多动态变化，但不证明这些动态与GT吻合。平均加速度上升也不单独证明轨迹物理不可能；只是违反此前锁定的≤1%无伤害要求，不能事后放宽。[results/summary.json：native、aggregate]

训练loss epoch1→80下降可学习约6.55–7.03%、固定5.58–5.80%；可学习CAL change ADE反而增加.55–2.66%、FDE增加1.47–2.72%。首epoch已训练，不是精确nativeCAL，因此仅是该run内泛化缺口描述，不能归因唯一瓶颈。[completion_review/REVIEW.json：histories]

final描述输出change改善约2.69%/2.49%，仍未5%，FDE/freshcontrolCI仍跨0，transitionADE/FDE伤害约2.33%/2.27%、changeaccelmean+29.34%。不能因DEV看起来更好就选final替代原constrained，也没有理由盲目加epoch。

## 对研究方向的含义

1. 有“共享结构下变化组路径误差有所改善”的探索性线索，比此前free12increments的约.19%/.21%更大；但架构和训练不同，这不是受控跨实验优势证明。
2. 本次负结果更具体：只增加可学习时间位置/跨度，配合当前原损失和冻结特征，不足以可靠增加高质量候选覆盖。问题不只是“没有时间自由度”。
3. 可能仍包括单pulse表达受限、多阶段响应、幅度/方向与时序协同、winner/保护正则约束、特征对精确未来的不足与FIT/CAL泛化。均为待检假设，不能说decoder是唯一根因或线索已经被有效利用。
4. 建议下一次决策先区分“这类曲线是否能表达失败样例GT响应”与“能表达但现有训练学不准”。可讨论FIT/CAL-only表达上界和时序参数误差诊断；事件辅助监督/更换损失是独立干预，需要精确方案与授权，不能由此次分析擅自启动。

本response study现在完成2/3；第三项批准正式完成后才做整组review。当前仅交付分析并等待人类，不自动第三训练，不更广解冻/改评分器/读heldtest或跨fold。

## 内联审查与局限

Academic Research Suite用于匹配对照、反证及条件分母审查，非外部独立审稿。

- Checkpoint1：仅回答固定/学习clock的特定对照；名义参数相同不等有效容量，原科学范围不扩张。结论限定后无Critical阻塞，容量耦合为Major局限。
- Checkpoint2：把native误差小幅改善与freshcontrol无稳健增量、严格覆盖净减、motion副作用、CAL恶化并列；最强反证是无法证明学时钟带来准确完整响应。
- Checkpoint3：不把有效参数变化当收益、不把不同candidate的minADE/FDE当一条好路径、不事后选final/阈值；保存算术审核不称全模型重放。结论与已核验文件一致。
- 无人工盲评新判断、无新身份信息/外部上传；AI协助分析与记录、人类授权监督。本文不声称伦理机构批准或公开注册。三seed/场景相关/native曾见数据/反复DEV只支持探索性判断。
