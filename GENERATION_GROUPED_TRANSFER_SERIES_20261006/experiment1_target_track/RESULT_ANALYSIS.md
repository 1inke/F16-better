# 行人分组迁移对照：完成分析（2026-10-06）

## 结论与目前分支
实验完整完成，不是仍在训练。实际生产09:07:16.580045–09:42:20.072694HK，35分03.493秒；唯一完成检查09:55:59.4766771，未重复半程/旧PID查询。六次全新80epoch训练，18checkpoint×3split=54模型评估+3原生，共57科学jobs/114输出。18policy不是18次独立训练。

结果更支持“适应后的预测精度部分依赖同一行人的相关窗口暴露”，但不证明完全无法迁移，也不证明共享head干扰是唯一根因。首先继续检查迁移/相关性，而不是现在改scorer、hardwinner或扫学习率。下一项已精确预注册同录像时间块隔离，待实施完整审计后启动；目前无第二worker。

完整好候选必须同一条轨迹 ADE<=.2m AND FDE<=.4m；以下覆盖均按此标准，3seed结果不能相加成独立样本。固定epoch80/final为主分析，绝不选外层表现最好的checkpoint。

## 一、主评估：相同1035个外层窗口
训练两臂各18909，sequence×原loss group精确匹配，outer1035/inner1856相同；统计只由15572共同训练交集估计。Random暴露1028个评估目标身份，1024/1031已知时间窗口有同目标20索引重叠；grouped目标暴露0。旧FIT/CAL是ordered前25912/后2879，并非原本随机划分。

|固定epoch80|整体好窗口/1035，seed1/2/3|变化好窗口/253，seed1/2/3|变化minADE/FDE三seed均值(m)|
|---|---|---|---|
|原生F16|625|110|.249147 / .429378|
|random-window|642 / 649 / 632|113 / 115 / 113|.249305 / .426035|
|target-grouped|642 / 622 / 632|107 / 108 / 109|.264060 / .455268|

整体覆盖原生60.39%，random61.93%，grouped61.06%。变化覆盖原生43.48%，random44.93%，grouped42.69%。Grouped相对random整体少0.87个百分点、变化少2.24个百分点；不能称“覆盖大幅崩溃”，它们的探索性簇CI均跨0。

然而，Grouped–Random的连续误差/J差异更明确：
|主外层指标差，grouped-random|三seed均值|95%探索CI|
|---|---|---|
|整体ADE(m)|+.006638|[+.002314,+.010908]|
|整体FDE(m)|+.015677|[+.003995,+.027446]|
|变化ADE(m)|+.014755|[+.004404,+.028485]|
|变化FDE(m)|+.029233|[+.002360,+.061650]|
|变化J=minKmax(ADE/.2,FDE/.4)|+.068403|[+.020129,+.137461]|
|整体完整覆盖比例|-.008696|[-.031983,+.011771]|
|变化完整覆盖比例|-.022398|[-.062411,+.016598]|

正误差/J差表示grouped更差。CI使用相同评估目标轨迹簇+3模型seed的2000次层次bootstrap，是探索性区间，多指标未作确认性纠正，不是3独立数据集。

Random相对原生变化ADE没有改善、FDE只约.78%改善，变化误差与覆盖的native-reference CI均跨0。因此“random表现很好而grouped全失效”的强前提并未实现，不能将这次结果夸大为纯泄漏证明。Grouped变化ADE/FDE相对原生约5.99%/6.03%变差，三个seed变化覆盖均低于原生。

## 二、rescue–lost与候选数量
主外层变化组random救26/26/24、损23/21/21；grouped救21/21/21、损24/23/22。并非全无救回，只是grouped net=-3/-2/-1。
整体random救71/74/59、损54/50/52；grouped救73/62/73、损56/65/66。
整体有效好候选簇总数random平均918.33、grouped948.33，而grouped覆盖更低、误差更高。更多有效簇不能替代至少一条完整好候选；原始候选数也不最大化。完整GT-free ALLK20 complete-link .10meanpath/.20endpoint定义未变。
同一Jclosest候选变化组前/中/后段误差：random .12097/.26911/.37648m，grouped .12521/.28725/.40015m。后段差异更大但前段也有差异，不能单凭此归因decoder尾段结构；未拼接不同oracle槽位。

## 三、必须保留的反例
共享inner1856的final整体覆盖原生932，random939/953/975，grouped948/959/961；inner变化497原生166，random152/156/172，grouped165/172/179。Grouped在另一个身份隔离群体的覆盖可增加，推翻“按行人分组后完全不能泛化”的过强表述，但其误差仍未稳定优于原生。
原CAL2879只是事后描述：整体原生1490，random1442/1481/1470，grouped1473/1476/1484；变化700原生243，random228/232/237，grouped224/239/217。所有final变化覆盖仍低于原生，grouped整体比random平均好一些，因此随机暴露不在所有split占优。
按场景：outer变化hotel17，原生10好，random均12、grouped均12.33；university211原生91，random93.33、grouped88.33；zara25原生9，random8.33、grouped7.33。Hotel小样本改善是反例，不能说所有场景都退化。三个场景极不均衡，禁止伪造可靠scene CI。
Inner选取constrained epochs random8/9/8、grouped7/7/8，其outer变化覆盖random110/110/110、grouped110/111/110，接近原生；原CAL变化random243/244/246、grouped244/243/243。早期选择保护住不少质量，但不等于学出显著新响应或把secondary升为primary。

## 四、运动学与训练损失
端点保持投影的每window加速度均值严格1%保护通过；五项pooledALLK仍单独审核。
Grouped所有policy/split/group均通过五项1%。Random只有5个final行违反speedp95：outer变化seed3 +1.021%；inner变化seed1/2/3 +2.088/+1.569/+1.364%；原CAL变化seed2 +1.041%。即使轻微也不能放宽1%标准。加速度投影不是速度或所有运动学保证。
Random首末trainloss约.370→.300–.303，grouped约.369→.298–.300。两者训练目标均下降，grouped的训练loss还略低，但主外层误差更差；这比“loss算错”更符合迁移/相关性疑问，不过不同训练窗口的loss不可当同ID配对因果，亦不排除objective局限。
所有原major条件没有满足：没有变化ADE/FDE各5%稳定收益或可靠完整覆盖提升，不能升级F16baseline，不声明运动意图/SOTA/decoder唯一根因。

## 五、完成核验与独立保存算术
生产worker与results exit双SUCCESS；ARITHMETIC/FROZEN/PRESERVATION PASS，565源/32754旧文件前后SHA不变，18checkpoint所有参数/布尔/scaler缓冲固定；6完整80epoch history与objective trace。
completion_review/REVIEW.json：CPU仅保存路径/标签，不torch/model-forward/backward/optimizer；57jobs114输出18models前后SHA；全部quality列、216组汇总、原SciPyGT-free complete-link重构、pooledmotion、180探索CI重算全部EXACT，legacy minADE/FDE最大误差0，perwindow严格physics/端点PASS。不是所有gradient/model再次重放或外部独立审稿。
补充脚本首次默认Python缺psutil、正确环境下CSV BOM导致epoch键读取失败，只新补充读取器utf-8-sig修复，失败脚本/FAILURE/SETUP_PROVENANCE保留，完整重新审核PASS；未改生产科学文件/标准。
生产RSS峰2.5931/private3.9016GiB，系统physical/commit最低16.1190/15.4963GiB。补充private.2043GiB单独记录，不能替代生产峰。持续process6GiB和系统physical+commit各6GiBreserve，约17GiB资源约束未放宽。

## 六、下一步精确方向与规划节点
研究分支暂定：存在相关目标窗口依赖信号，但尚未达到“grouped稳定好→只查CAL distribution shift”条件，也未证明“grouped完全失败→马上MoE”。
第2项 experiment2_recording_time/DESIGN.md 已预注册50源索引block，同录像所有行人的完整20索引间隔purge，random保留部分相同时间块训练暴露，两边原模型/目标/预算不变。前置元数据规模：两臂16104、共同统计12544、outer566/change142、sharedinner2134、未知时间123单独保留记账；只设计规模核验，不是来源/模型/GPU预检PASS。
10:20HK实施身份/区间/匹配审核，之后实际科学/source/resource/双arm真实2epoch恢复预检；只有全部PASS才benchmark实测ETA、独立OS solelaunch及一次50%节点。当前第二训练未启动，无生产ETA。时间隔离并非行人隔离、仍不是独立未见F16/新场景验证；场景系列另精确注册。
任何实质科学/源/ID/资源/权限阻碍STOP等人类，不搜索blockwidth/coeff/LR/epoch使PASS，不改loss/scorer/旧decoder或更多数据范围。新研究已完成1/3；每3正式完整审核mandatory fullreview，历史cycle11不重置。

## Inline反方检查2：PASS_WITH_LIMITATIONS（同助手，非外审）
最强反驳：连续误差差异可能来自训练多样性/有限优化及条件分布，而非同目标记忆；覆盖CI跨0、inner与hotel反例不可删。已限制主结论为相关暴露支持信号，不称不可泛化/因果泄漏。重大限制target-disjoint不是邻居时间上下文独立，下一已精确设计purge而非直接改架构。

## Inline反方检查3：PASS_WITH_LIMITATIONS（同助手，非外审）
不能将随机arm微小覆盖收益、grouped加速度保护、更多好簇或constrained稳定当major完成。保留5个严格speed失败，所有原成功阈值不变。冻结F16已见、原CAL单university、只有3seed/3不均场景及多指标区间探索均保留。AI辅助整理与算术审核可复现，但不是外部审稿或独立泛化保证。

依据：本实验results/RESULT.json，ARITHMETIC_AUDIT.json，FROZEN_AUDIT.json，PRESERVATION_AUDIT.json，RESOURCE_USAGE.json；completion_review/REVIEW.json、PER_SCENE.json；父identity_audit/RESULT.json、splits.npz；预注册DESIGN/protocol及6history。全部只读旧科学证据，新报告只本次目录。

