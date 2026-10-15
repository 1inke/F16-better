# Native-anchor三项 mandatory fullreview

状态：三项完整完成、3/3 ANALYZED；所有执行与保存审核PASS，三个major均FAIL。AI辅助/academic-research-suite同助手反方核验，非外部审稿。基线、历史cycle11和未完成的长期目标不变。

全数据：[216个arm×seed×split×group单元](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/completion_review/FULLTHREE_DATA.json)、[第三独立审核](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/completion_review/REVIEW.json)。第一、第二已完成的独立保存审核通过且当前1193相关SHA锁继承保护；本复盘引用其保存汇总，不声称再次重放全部旧模型/梯度/聚簇。

## 1. 三个问题与三次回答

| 对照 | 唯一比较因素 | 主outer变化覆盖，native110/253 | 三seed运动学结果 | 判断 |
|---|---|---|---|---|
| 第一：CV vs native anchor | 同输入/MLP/损失，输出基底改变 | CV108/108/108；native111/111/111 | native36/36 accelmeanFAIL；CV30/36 speedp95FAIL | 原生基底比CV保留更多变化支撑，但小幅位置残差不够物理安全 |
| 第二：free12 vs Bernstein | 相同12bounded controls的时间映射 | 两者111/111/111，全部二元救回/丢失集合相同 | 两arm72/72 accelmeanFAIL | 残差jerk降86.6–89.9%，完整路径加速度仍失败；不能唯一归因高频 |
| 第三：Bernstein vs physical-ray | 同Bernstein下训练内ray可行收缩 | 不投影111/111/111；投影110/110/110 | 不投影36/36 accelmeanFAIL；投影36/36全五项PASS | 物理保护恢复，但修正近identity、零新增覆盖 |

三项同target grouped train18909/outer1035/inner1856/CAL2879、3seed固定25、每arm25304参数、每window4条refined路径加16条exact保留路径、K20。六训练×三项=18个fresh训练执行、18checkpoint、450epoch行、54保存科学jobs；共108bootstrap记录432scalar探索CI。重复控制不是新的独立泛化，216组也不是216个独立场景。

## 2. 全组/所有CI，不只选变化组

全216组指标和motion失败明细见FULLTHREE_DATA，各项全部36CI见各自completion_review/SUMMARY.json。protected overall/stable/transition同样计入，未拿有效簇补偿coverage或把window-p95平均当pooled globalp95。

第一native总体outer628各seed、inner931各seed、CAL1493/1494/1495；CV总体outer638/637/639、inner968/963/967、CAL1525/1526/1529主要来自稳定组，而变化CAL241/239/240低于native243。因此“CV总体更好”不能推广成修好变化行为。CV初始zero总体也不同，已报告own-zero，不能把输出基底差全算学习。

第二Bernstein/free的36paired组二元coverage集合、rescue/lost ID完全相同；这是当前样本不跨门槛，而非两个总体方法等价。两者outer变化coverage+1/253的探索CI跨0，连续质量均无重大收益。Bernstein只把残差jerk平滑，原生加残差的全路径仍accelmean+2.665–3.073%。第一free/native对照也存在+2.613–3.054%的同问题。

第三physical各组救0/损0、outer/inner/CAL变化110/166/243均原生，effectivegood没有新增覆盖。所有五motionPASS来自近原生输出；alpha median0、53.76–56.82%候选alpha=0，最大位置残差约0.04184毫米。不能把原生负对照被保留说成积极突破。其对native change ADE/FDE/J为1e-8级差异，CI跨0；coverage CI[0,0]由相同二元集合造成。

三个major都不满足change ADE与FDE各≥5%、protected≤1%、三seed一致/native与fresh主要CI排除0、同路径覆盖/重复/硬物理和每seed每组五pooledmotion≤native1.01的全AND。第一/第二motion失败，第三收益缺失。首次全PASS才进行scene/sample相关及3seed充分性审核，当前未到该节点。

## 3. 反馈、容量、泛化：哪些结论不能下

同一canonical硬min4(ADE+.5FDE)+六pair平方hinge.3*.05，AdamW .001/wd.0001/batch256/noclip/25epochs保持。fixed16没有偷偷加入训练winner，全部K20质量评价与四路径loss分母不同。第一/第二四路径loss仅降约2.4–2.5%，diversity为0时相对百分比未定义；第三physical训练loss几乎不降。没有完整FIT set质量快照，不能继续套用旧persistent架构“FIT大幅成功→CAL失败”为本系列新证据。

第一输出anchor改变初始支撑与reachable set，不是“newhead理解运动”的纯因果；第二basis fullrank但有界可行几何/条件数与优化变了，不是等有效容量；第三31逐时间逐候选native相对约束强于原pooled门槛，native零/微小加速度jerk可压缩整条方向。三项不能推出“所有可行候选不存在”“head完全无法学”“loss已免责”或“只能改scorer”。

前系列原始state/中心化/标准化、fixedcandidate-route、物理P、训练内端点保持投影、softset/energy/mixture、freeincrements/timequery/velocitybasis、conditioning/mode/social、GT witness与tinyFIT/CAL的反证均不删除。它们不是同canonical协议的单因素扩展，不能从不同容量/输入/协议横比均值认定唯一根因。

## 4. 重复控制与可信度

第二free的九输出11个原有数组与第一native实际bitwise EXACT；第三unprojected的九输出同字段与第二Bernstein实际bitwise EXACT。可验证流程一致性，但不是18个新独立模型发现。paired初始化、共同train-only统计、IDs与cache冻结是每项内部对照，不代表anchor/basis/ray之间有效函数跨度不变。

当前CPU复核六个manifest/checkpoint SHA及150history，18全部质量/原SciPy GTfree聚簇/五motion/assignment/bounds16/Balpha/36CI独立最大误差0。原旧F16/decoder/scorer未实例化或optimizer访问，nativecache/SHA保护，不假称wholeoldGPU replay。资源始终process6GiB/physical和commit各reserve6GiB，无OS/pagefile/userkill。

保留所有工程失败：第一144→120输入schema误写；第二paired汇总72field实际36pairedrows的命名纠正；第三roundoff误判与被拒fallback、大步长FD面切换、小步长收敛、单次时间授权。第三11个rawpointwise微超额最大2.63e-15由推导证书覆盖，**原pooled研究阈值仍literal**。控制台大JSON读取被截断只影响查看，已生成的FULLTHREE_DATA完整且216行assert通过，不冒称控制台parse成功。

## 5. ARS反方2：最强替代解释

PASS限域。最强反驳：ray保护太强，把与可观察输入相关的有效修正也挤没了，失败可能来自受限方向/预算而非generation无信息。必须保留；安全与改进的联合缺口确实未解决，但不是普遍不可能性定理。第二仅一个basis不能证明所有时间结构无用，第一CV与native的起点差也不能孤立head学习能力。相关窗口/3scene/3seed/CAL重复/F16旧pool已见均限制独立泛化。

## 6. ARS反方3及方向更新

PASS_EXPLORATORY_NEGATIVE_SERIES。拒绝“理解运动”“SOTA”“独立wholemodel泛化”“decoder唯一根因”，不删native支持/预算/全组motion反例。意义是排除本组三种局部修补足以形成安全互补，不是放弃候选生成。

下一优先问题转到**物理可行且原生支撑友好的非退化生成几何**：输出方向本身应有可用可行跨度，不依靠事后原生零预算共同alpha把整条修正压到近0。应先预注册一个结构层级、单因素的新方向并核验非GT可行域/梯度/原门槛，而不是第四margin、degree、bound、width/LR/epoch局部扫描。

不进入scorer优先分支，也不自动改quality assignment/loss、旧F16/decoder解冻或外部数据。原六小时窗口结束，当前单第三+全复盘已归因闭环；本轮不再新启动训练。近期跨系列综合另见[RECENT_EXPERIMENT_SYNTHESIS](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/RECENT_EXPERIMENT_SYNTHESIS.md)。

