# R4：冻结表示的跨轨迹邻域一致性诊断

2026-10-13；状态：**设计与核心代码，未启动真实实验**。NONFORMAL，零模型拟合、零神经前向、零标签求解，不进入B/S2。

## 1. 问题、证据与边界

在R3的固定200维可观测表示里，漏选的安全可修复锚点P，是已经具有跨行人邻域支持但排不到前四，还是缺乏训练覆盖、或邻近候选的标签并不支持P？

R3证据：同263 known-P窗口，pointwise77/263，pairwise82/263；净增5=16新增−11丢失，未过门。pairwise最佳P第1–4/5–8/9–20名分别82/75/106；change为20/26/31。181漏选中161的top4全N，不能主要归因于U直接占槽。来源为上一目录production_20261013的RESULT_ANALYSIS、RESULT及completion_review/REVIEW，不重跑旧worker。

本轮是**当前表示与距离度量下的经验诊断**，不是新选择器竞赛，不证明信息论不可预测、Bayes误差、行人意图或PNU因果关系。P/N依赖真实未来与固定position3修复界；观测相近不意味着未来必须相同。最近邻不一致也可能来自距离失真、数据稀疏、条件样本选择或见证发现偏差。

## 2. 固定输入与六个计算单元

- 原A384、P329/N7027/U324、263 known-P窗口、115条known-P轨迹，原三个target-track OOF折不变。共同训练177/180/169，所有训练窗口至少有1P和1N。原discovery56轨迹排除不变。
- 每折直接复用R2的136标准化特征+64标准化RBF；中心/标准化/顺序完全冻结。不重新训练表示，不用R3系数加权距离。
- 每折两个计算单元：训练库内部leave-one-track-out的无标签距离标定；该折所有验证窗口×20候选的检索。三折合计6个诊断单元，不是六次训练。
- 7680个验证candidate查询全部完成后，才使用P/N/U、运动组、场景及R3已存分数做事后统计。不得先选P再改变检索规则。
- 原训练库是条件于hasP&hasN的窗口；本诊断只解释这一库，不外推到所有FIT或无P窗口训练分布。
- 无B、外部数据、held test、新标签solver、GT曲线拟合、old F16/decoder/scorer或repair训练。

## 3. 邻居检索：只用可观测特征

唯一距离固定为：`d(z,z') = sqrt(mean_200((z-z')²))`。不扫距离、特征块权重、K、带宽或场景匹配策略。

固定K=5：

1. 候选库包含该折共同训练窗口的全部20槽，包括P/N/U，不依标签过滤。
2. 排除query所在整条target track；OOF本来已排除，训练标定也必须整track排除。
3. 每条训练track只允许一个邻居：选该track里距离最近的window-slot。
4. 再取最近5条不同track的代表。所有并列按`distance → integer sampleID → physical slot`稳定处理；不按P/N/U、原logit或预测分数解并列。
5. 不足5条不同轨迹即STOP_INPUT_SUPPORT，不自动减K。选中U保留，不找更远的P/N补位。

检索API不接收GT、标签、group、scene或score；track只用于排除/去重，ID只用于并列稳定排序。target-disjoint仍不等于scene/context独立，须报告邻居同场景比例但不据此重选邻居。

### 覆盖半径，不把“最近”偷换成“很近”

在同折训练库内部，对每个训练candidate执行上述整track排除检索，记录第5邻居距离d5。以track等权、其内window等权、其内20candidate等权计算**加权95%分位数r95**（最小累计权重达到.95的观测值）。

验证candidate的`near = d5 <= r95`。这是无标签训练距离参照，不是理论OOD边界；报告d1/d5、d5/r95与原数值。若r95=0，near按精确d5==0，比例输出null，不加epsilon或扩大半径。map本身来自整个训练库，因此训练标定不是完全重拟表示的独立LOTO验证。

## 4. 标签一致性：必须校正N占多数

检索完成后，记录5个邻居P/N/U数量。`known = P+N`；至少3个已知邻居才有充足标签支持。U不变成N，也不补邻居。

令所选第j个已知邻居来自训练窗口w_j，该窗口所有20槽中有p_j个P、n_j个N。定义：

    observed_P = 邻居中的P数 / known
    expected_P = mean_j [p_j / (p_j+n_j)]
    enrichment = observed_P - expected_P

expected_P是**保持所选窗口与U位置不变、在其已知槽之间随机打乱P/N**的解析期望。每个邻居来自不同track，故也来自不同窗口。另报至少一个P的观测值和参考期望`1 - product_j(1-p_j/(p_j+n_j))`。

无需实际跑置换或新增随机样本。该期望只是类别基率校正参照，**不是有效随机化p值**：真实P/N不满足已知槽可交换的确认性假设。高N比例/高N预测准确率不能单独作为冲突证据。

同时保存所有P查询与所有N查询的类条件统计、U比例；先在每window各类内平均，再跨window平均，禁止让大量N槽淹没P。P/N类条件宏平均一致性可描述，但不作为模型成功门。

## 5. 主要分析人口与预注册比较

主分析每个known-P窗口只取一个query：**按已存R3 pairwise分数最高的P，分数并列physical slot最小**。此选择用标签但严格在检索后，仅用于解释旧模型失败，不是部署选槽。不是挑邻域最好的P。所有329P作为次要完整报告，不能替代主分析。

固定三组：bestP rank1–4（82）、rank5–8（75）、rank9–20（106）；change对应20/26/31。保留R2→R3排名转移与新增/丢失命中ID作为描述，不GT合并两个选择器。

每组先报全部分母，再报near比例、known>=3比例、两者交集比例，不能静默丢弃远邻/U多的窗口。`supported = near AND known>=3`；下列推断只限supported子人口，所有人口亦报原始描述。

固定六个探索性统计量（overall与change各三个）：

1. rank5–8的平均enrichment。
2. rank9–20的平均enrichment。
3. 上述两组均值之差。

2000次整query-track bootstrap，rng20261013，同一抽样表同时计算六项；窗口加权，保留同track跨group关联。六比较Bonferroni，总alpha=.05，区间99.1667%；不重拟库、map或检索，训练库不确定性未计入。

每个被比较单元必须在**全部人口中supported>=80%**，且supported>=20窗口/10轨迹；不满足则估计仍描述、推断CI为null、结论INCONCLUSIVE_SUPPORT。每个统计量至少1900/2000个有效重采样才给CI，保存空抽样数。该80%是诊断解释支撑门，不是原模型质量/安全门，也不通过改半径来满足。小hotel或其他分层只描述，报告零分母null。

## 6. 解释分支，不自动开启后续训练

| 结果形态 | 可支持的下一步 | 不能声称 |
|---|---|---|
| 大量rank9–20缺少near或已知邻居 | 优先训练覆盖/可观测表示/标签发现机制的精确设计 | 不可预测、模型必须更大 |
| supported充分，rank5–8有正enrichment，rank9–20也有 | 当前排序未充分利用此度量中的局部信息；可设计唯一top4集合目标对照 | 新loss必定成功、已具部署效果 |
| supported充分，rank5–8正而rank9–20不确定 | 边界子集有证据，远端仍未定位；保持两类问题分开 | 所有失败都是截断问题 |
| supported充分，某组enrichment上界<=0 | 当前度量下未见超过所选窗口基率的P邻域支持；优先表示/监督可观测性问题 | 所有表示都无信息、Bayes不可学 |
| CI跨0/三个fold相反/支撑不足 | 明确不确定，停止本诊断，不扫K/距离求结论 | 显著无效或方法等价 |

这些可能同时成立，不强迫选择唯一根因。本轮不产生新Hit4选择器，不按邻居标签部署新策略；R3原FAIL仍然FAIL。原生成质量、change5%ADE/FDE、protected1%、五motion<=native1.01等门原样保留，全部不由本轮诊断宣称通过。

## 7. 实施与保存要求（尚未执行）

1. 继承R3原3313相关来源及47科学输出digest，再扩入原报告/本设计；不覆盖旧hash。核对R1输入/标签、三个fold训练统计与map、R3 OOF分数/ID/rank及报告人口。
2. 真实输入预检：384唯一ID、track隔离、中心只来自train、各折map重构逐数组一致；两种独立距离计算/逐track手工loop与chunk实现核对。设计合成测试不能代替此项。
3. 真实小批查询测时及内存，只查无标签几何、不据label选择速度/方案。计入训练LOTO标定、7680验证query、标签统计、CI与独立保存复算。planning>30min则STOP_BUDGET，不抽小样本/降K/少做fold。没有生产ETA直到实际benchmark。
4. CPU单线程、process<=6GiB、physical和commit各reserve6GiB；按query/chunk guard，不查旧PID、不调OS/pagefile/用户进程。不构造整人口query×bank×200大tensor。
5. 以后获执行授权且预检PASS才建立唯一OS任务/一次startup/一次half→completion；本次没有launcher或worker。
6. 每折保存训练与验证query显式ID/slot/track、neighbor ID/slot/track/index/dist、距离分量、r95及权重、PNU计数/基率/支持标记。全局保存bestP/rankbin、每层全部分母、六项统计及draws/无效数、source/outputSHA/RESOURCE。
7. 独立CPU保存审核重构全query最近5条track、无标签检索独立性、PNU统计、r95、六CI及全部SHA；不得只验hash不验几何。验证输出字段不可与旧失败同名错误混淆。完成source兼容才analysis-skip；无训练optimizer/RNG恢复声称。

代码是纯诊断核心与显式未来接口，不包含生产资源guard、源锁管理、持久化/launcher或完整审计器。这些在执行授权后补齐，不把代码交付标为科学完成。

## 8. 内联反方检查（ARS，同助手非外审）

- 检查1 Major：高维距离并非语义等价。固定200D等权只检验一个预先选择度量；不从负结果外推所有可观测信息不足。
- 检查2 Major：N数量远大于P、相同track候选相关。已用不同track5邻居、窗口内已知标签基率校正、窗口级主统计；bank hasP&hasN选择偏差仍保留。
- 检查3 Major：bestP分层来自同一批已反复研究的A，属于事后机制诊断；CI不覆盖历史适应性选择、重叠fold训练或context依赖。不给因果、心理意图或确认性结论。

沿用既有合法去标识化本地数据，不招募/联系参与者、不再识别身份、不上传数据。报告按计算实验可复现性要求描述来源与限制，不冒称新IRB审批或外部注册；本方案是本地事先固定的探索设计。
