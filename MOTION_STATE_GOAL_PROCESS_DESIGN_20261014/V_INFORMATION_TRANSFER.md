# V：信息利用与迁移验证

版本1.0；CONDITIONAL DESIGN ONLY。不是预先承诺运行全部诊断，更不是新增长任务授权。仅使用已有合法raw观测和既有内部划分，不新增数据/测试fold。

## 1. 何时执行、验证谁

- V1冻结信息诊断：按固定优先级，M3-S3通过验证M3-A/B；否则M2-Q2通过验证M2-A/B；否则仅M1-S1通过时验证M1-A/B的目标信息。若三者均无信号，停止，不为阴性结果继续堆诊断。
- V2/V3迁移：只对M2-Q2或M3-S3通过的完整生成对照开展。M1仅有终点信号不触发长预算的完整生成迁移。
- 不挑最好seed或只验证赢家，始终保留其直接匹配对照和三seed。
- V1是在损坏/变换输入后看性能与敏感性，不是真实反事实未来预测；V2/V3是在旧数据上的新head迁移，不是全模型未见泛化。

## 2. V1：冻结信息利用诊断

### 2.1 固定人口与条件

使用全部outer1035原ID，无GT选例；3seed×2arms×5conditions＝30个评价job。原始条件可在完全兼容的saved输出上复用，不重新训练。诊断保存只在新V目录。

| condition | raw自身历史 | P2 | 新motion支路 | 可解释范围 |
|---|---|---|---|---|
| ORIGINAL | 原始 | 原始 | 原始 | 主参照 |
| BRANCH_CV | 原始保留 | 原始 | 将自身历史替换为末速度反推的CV历史 | 仅新增支路依赖，不是全历史依赖 |
| COHERENT_CV | 以最后两点确定v_last，反推全部前6点；最后两点不变 | 用完整raw重新算冻结F16 | 同一变换raw | 历史趋势损坏的整体敏感性 |
| PREFIX_REVERSE | 前6个点逆序，最后两点不变 | 同一变换raw重算 | 同一变换raw | OOD顺序敏感性 |
| PREFIX_SHUFFLE | 前6点按SHA256(tag,ID)固定置换，最后两点不变 | 同一变换raw重算 | 同一变换raw | OOD顺序敏感性 |

tag=`motion-goal-process-v1-20261014`。物理时间戳保持原位置；反转/打乱点可能产生不自然速度/加速度，必须保存输入运动学分布，不把这些序列当真实行为。所有scalers和centre保持训练时原值，不重拟。

所有条件仅改变自身历史，邻居raw观测保持原值。必须通过现有focal归一化和完整冻结encoder重新计算P2，不能直接改cached tokens/paths。末两点相同使原点/末速度固定，但仍须真实核验坐标管线；P2可能含邻居交互，因此响应不能全归于自身GRU。

如果最终模型没有新motion支路，BRANCH_CV明确记N/A，不伪造“零影响”结果；其他条件仍有效。只实例化旧模型做冻结raw前向，不训练/覆盖旧checkpoint，也不查询已完成任务PID。

### 2.2 指标与解释

所有条件报告endpoint/full coverage、ADE/FDE/J、rescue/lost、全部motion以及路径集合距离。相同candidate索引差异和GT-free对称nearest-set距离都报，避免query交换导致虚构大响应。

GT仍是原始记录的未来，用于“输入破坏后原任务性能”比较，不是这些合成历史的真实counterfactual未来。不能报告合成历史的“反事实ADE改善”。

无训练必要性对照时，性能下降可能由OOD而非有用信息被移除；无下降可能是P2冗余/模型忽略支路，而非真实世界不需要历史。模型输出变化只表示敏感，只有原始held-out质量、定向消融和迁移共同支持，才谨慎称“利用了相关运动线索”。

本项为描述性机制诊断，不将众多干预95%区间当新显著发现；不得用V1最漂亮条件决定后续结构。若以后要做真实同当前状态匹配的历史对照，必须另行固定匹配规则与可比性，不本轮临时搜好配对。

## 3. V2：recording-time迁移

精确复用已有[时间划分协议](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/protocol.json)的grouped训练16104、outer566、inner2134；原CAL2879只posthoc。50为SOURCE索引block不是秒/视频frame；保留原unknown/boundary资格记录和所有ID，不重新搜索blockwidth/hash。

时间隔离不等于target/context独立。F16原编码器已见旧pool，明确这是新生成器适应迁移。

### 3.1 如何避免成本和泄漏被隐藏

M2入选时：每seed在当前time-train上fresh训练一个固定来源类型的goal前置模型25epoch，再复制给frozen/joint两臂，各训练process25epoch。因此每split是3goal＋6process＝**9训练阶段、225epoch**，不是“只要6个短run”。

M3入选时：每seed先在当前time-train fresh训练一个表示来源模型25epoch取得共同冻结h，再训练两种M3预算各25epoch，也是9阶段。即使原target训练的encoder更好，也不能暖启动它，因为其训练ID可能含time-outer。

每个split的P2标准化、动态特征统计、centres和过程输入尺度都只由其train重算，算法固定；不能从target或当前outer带入训练统计。旧F16只冻结，不声称消除了旧F16见过数据的限制。

### 3.2 判定

同split同ID中比较B−A和B−native。报告三seed、原四组、fullcoverage/endpoint/motion以及所有负结果，不跨split配对或平均绝对coverage判断grouping因果。

V2支持继续V3的门：B−A change覆盖三seed均非负、均值≥5pp、下述V-family校正CI下界>0；B outer变化覆盖各seed≥native，outer保护组各seed ADE/FDE≤native1.01，全部四split全组五motion通过。未通过则停止迁移扩展，结论为收益尚未稳定迁移，不重新选block或seed。

## 4. V3：scene-family迁移

仅V2通过且单项预算合格时执行。复用[场景划分协议](../GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/protocol.json)：既定zara外层1561、grouped训练16337、inner1773，原CAL2879posthoc；保持未知时间记录和资格，不因结果换held-out族。

同样fresh重训前置表示/goal和两臂，9训练阶段225epoch。time/scene两个split全做是18阶段450epoch；预算不合格明确标“未验证”，不能用只做冻结预测代替训练排除证据。

只有一个held-out族、少量recordings且曾被研究反复查看，不能声称普适跨场景或独立确认。使用相同V2门判定迁移支持，任何失败均保留。

## 5. Transfer统计与预算

V2/V3预留16项主family：每split两个对比(B−A、B−native)×change四指标(C/minADE/minFDE/J)×2splits。Bonferroni percentile区间，20,000draws，rng20261014，分位数0.05/(2×16)及1−该值；未运行V3仍不缩减除数。该family不校正整段项目的历史自适应搜索，不宣称确认性。

V2 primary按sequence-timeblock聚类与3modelseed crossed重采样，复用原anchor block ID；target-track聚类为预定敏感性，不能替换primary。V3按sequence+target-track聚类；三个zara recording只描述，不能伪造可靠独立scene bootstrap。

只有14时间outer blocks等小cluster问题必须标注，不因大量bootstrap次数就称样本充分。次级全组指标和每seed效果完整报告。

每个split的完整9阶段成本含前置训练、24最终sciencejobs、初始化数据、CI/全saved审核，真实benchmark后才报ETA。若规划>30min，跳过该生产对照并写明原因，不能悄悄只训一个seed或减epoch；是否另开长预算必须是新明确决定。

## 6. 最终允许的结论

- 只有原始数据收益：候选质量改善，不能称理解。
- 原始收益＋信息诊断：模型对相关历史有敏感性，仍需OOD和冗余解释。
- 再有time/scene迁移＋完整物理门：较强支持可迁移运动预测能力，但仍不是心理意图、独立whole-model泛化或SOTA证明。
- 任何环节未完成：在最终报告列“未执行/因预算跳过/因前置门失败”，不写整套完成。
