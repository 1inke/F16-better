# 三项时序响应研究完整总复盘：3/3 REVIEWED

## 决策摘要

TIME→PULSE→PERSISTENT均正常完成，独立保存审计PASS，但没有major。结束局部曲线自由度探索，转向**修正映射的候选运动状态条件化**。不放宽质量/运动学标准，不更换baseline，不把更多动态变化当更正确意图。原生候选的运动状态是已有模型预测，不是预测起点后的真实观测。

## 三假设与各自新鲜对照

| 实验 | 唯一注册比较 | constrained change ADE/FDE gain vsnative | freshcontrol结果与结论 |
|---|---|---|---|
| TIME | clip后4step分段平均/repeat vs12步自由增量，同43426名义参数 | .193%/.208% | block对照更好，三项CI跨0；自由增量未稳定救回 |
| PULSE | 固定center/span vs可学习clock，同39590名义参数 | 1.951%/1.471% | 对照同结构也改善；只seed1优于control，全部freshCI跨0 |
| PERSISTENT | 双learnedclock Gaussian瞬态 vssigmoid持续，同39590名义参数 | 1.287%/1.346% | 持续误差均值逊于瞬态，freshCI全部跨0；动态代价较小但未过1% |

TIME projection改变函数类及优化几何；PULSE fixedclock数据梯度0所以有效容量不等；PERSISTENT clock-终点耦合/幅度几何/函数类共同改变。不能包装为纯时间定位/纯持续性因果证明。

十八个fresh80epoch run，54policy entries，**不是54独立训练**。当前Gaussian对照3seed×3policy的9份完整路径与PULSE learned输出逐项bitwise相同，fresh执行而非独立泛化；也不能把两次同结果当新验证。所有假设/控制/完整24policy-group aggregates+96CI/6history各，合共72aggregate、288CI、18history已完整汇集在completion_review/ALL_THREE_POLICY_GROUPS_AND_CIS.json（无遗漏只挑显著项）。

## 所有保护组和联合路径

| treatment constrained | overall ADE/FDE | stable ADE/FDE | transition ADE/FDE | change ADE/FDE |
|---|---|---|---|---|
| native | .219154/.399725 | .175177/.313804 | .275791/.512714 | .372175/.696326 |
| TIME | .219471/.400256 | .175295/.314180 | .278071/.515929 | .371457/.694880 |
| PULSE | .217983/.398973 | .174925/.314402 | .276290/.514956 | .364915/.686081 |
| PERSISTENT | .218646/.399322 | .175531/.315542 | .275439/.511058 | .367385/.686951 |

三个constrained误差保护组≤1%都通过，不能抵消其它major失败。TIME seed3 FDE比native差；PULSE/PERSISTENT相对native微改善三seed一致，但对照增量不足。各组joint、coverage、duplicate、family/spread、hardmotion和全部运动分布列完整保留在全量JSON。minADE/FDE可能不同slot，joint/samepath另报。

TIME native和freshADE/FDE/joint CI全跨0；PULSE只有nativeADE/joint探索性负CI、nativeFDE跨0、fresh全跨0；PERSISTENT只有nativeADE负CI、nativeFDE/joint与fresh全跨0。没有三项共同独立确认，不生成虚假的合并meta-analysis CI。

## 同路径高质量覆盖与数量

固定ADE≤.2 AND FDE≤.4，未按结果改阈值；.3/.5仅敏感性，不删难样例。全体1516 native733，clear469 native92。

| treatment seed1/2/3 | 全体达标覆盖 | clear469覆盖 |
|---|---|---|
| TIME | 734/735/734 | 90/91/93 |
| PULSE | 723/728/727 | 88/88/88 |
| PERSISTENT | 733/730/735 | 93/90/93 |

TIME全体小净增不等变化样例一致改善；PULSE净损，不符合“困难候选能生成”的阶段目标；PERSISTENT回到近native仍没有稳健净增。全部rescue/lost应配对看同IDs，不能各seed或相邻窗口相加为被救回行人数。

只有本次追加质量聚簇审计覆盖所有18政策；native有效好簇929 vs持续934/930/935，小幅且与coverage几乎不变。PULSE等同当前Gaussian控制的有效簇909/911/817来自本轮相同保存轨迹重算；不冒称原PULSE当时已存该全量结果。GT-free簇不是语义意图类别，静止例自然接近可合并，数量不是越多越好。输入不同slot的minima不能拼为合格路径。

当前早中晚normalizedadequacy-closest与原两项jointoracle不同，均持有同slot全程；不能拼不同阶段oracle解释响应更好。晚段仍是大误差，不等新模型特有问题。原scorer仅继承slot，当前281→282/282/280近不变，不能自动启动scorer解决未生成的路径。

## 运动质量、校准和过拟合

TIME、PULSE、PERSISTENT均未通过全组平均accel/jerk等1%要求。PULSE增加加速度整体约14.6%/change23.8%；PERSISTENT pooled全K变小到整体1.53–3.39%/change2.65–5.27%，仍每seed每组超1%。平滑副作用减少是真实有意思的折中，不是重大提升或曲线正确的证明；硬极限通过也不代表分布无伤害。

TIME/PULSE/PERSISTENT trainloss均继续改善而CAL最终不保证改善；已选checkpoint集中较早，部分中后期不CAL可行。当前PERSISTENT selected24/24/19，末轮80不作新基线；不能盲加epochs或根据DEV改选final。全三项每seed早末loss、CAL末值、不可行epoch数、constrained/legacy选择点已存全量JSON。

## 中间诊断与历史反证

- GT辅助3anchor和pulse witness证明部分当前几何约束下有可行输出；不证明past-only共享头可以学到全部witness，更不是GT forward部署性能。
- 原目标兼容诊断为输出空间loss/梯度检查，不是head Jacobian/共享映射可学性；保护没有在所查局部处普遍消掉学习信号，不等原损失与泛化完美。
- Head-learning小集FIT失败128救回54–58，CAL失败77救回11–12但fullCAL169→167/161/165；FIT整体明显改善、CAL整体和动态恶化，证明部分可学而非稳定泛化。
- 38人工批注富集23rescues/15lost；late failures很多、候选改善可见，但只有2/23rescues inheritedslot合格。旧slot不重新评分纠正了“新模型偏向维持状态”的概率解释。
- 旧free-increments/timequery/velocitybasis、所有历史模式/social/head/集合监督失败，不证明意图不可预测，也禁止无限加自由度。旧decoder conditioning dense/rank/tangent泛化失败要继续作为反证。
- 可观测风险/事件兼容不是精确完整未来可读；未检测线索不是不可预测率。不能剔除这些样例来改善整体headline。

## 新大方向和可辨识性

原新head输入为context64+原modeMLP64、observed-token attention；native候选完整路径仅在输出加法处使用。mode隐式含候选结构，但head没有直接读取候选实际生成的速度/转向/终点状态。下一对照只加入**原生候选预测运动状态的显式条件分支**：相同新分支与全部初始化，control标准化后state乘0，treatment读取actual；双方同common持续响应和原损失/数据/clip/K20。先FIT-only数值非退化、源ID实际GPU和两arm real2epoch exactresume，再精确ETA/唯一独立启动。

这是假设检验，不宣称新增信息（state是已有native输出的确定函数）；输入可用性与有效容量/优化差异仍耦合。旧mode/social条件化失败降低期望，不是忽略历史换名称；新的比较明确落在**新修正头读取实际候选运动**而不是原mode或decoder解冻。候选state不能GT/未来观测/原logits/group混入，不能改变native候选生成/scorer。

合法范围现行human持续授权覆盖；新文件夹GENERATION_CANDIDATE_STATE_SERIES_20261005保存精确design后实施。当前没有此项生产worker，不重置历史cycle或计formal完成；新study只能正式预检通过并真实启动后登记0/3。需要新监督/scorer/进一步解冻/独立数据确认时实质阻碍STOP等人类，不以新名称规避。

## 来源、反方检查与限制

当前RESULT_ANALYSIS/summary/quality、前两RESULT_ANALYSIS、三completion审计、头学习和GT曲线诊断是本项目一手技术实验。AI辅助inline检查，非外部独立复现/审稿；本地schema修复有来源保留。

反方2：持续曲线motion更低，但误差不胜fresh，strictcoverage近native；不是已证明响应理解更好，profile因素不是单因果语义。
反方3：候选运动state来自已知native/mode，可能完全冗余、放大错误路径或仍过拟合；必须以freshcontrol和strictcoverage/motion判定，失败不归因未来不可预测。
只有3seeds/3scene族、窗口同轨迹相关、原F16训练见数据、CAL单university、重复DEV与人类富集选择，故不声称独立泛化、SOTA/心理意图、可靠sceneCI或原decoder唯一根因。历史cycle11不作新复盘确认，不重置cycle12。
