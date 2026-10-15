# R1 安全可修复监督的选锚对照
2026-10-12；人类已批准“根据以下内容执行新的实验”，覆盖此前设计-only状态。预注册快照，不表示已生产启动。

## 范围与固定抽样
仅原A7334，排除旧64发现窗口对应全部56轨迹；native20失败且包络potential任一真；按SHA256("repairability-supervision-v1|sample_id")逐组排序，stable/transition/change各128，同组每targettrack最多4，沿原三OOF。规模不足/预算>30min停止，不换哈希/配额。现只读scale通过384windows/511potentialanchors（1022固定求解），不是完整科学预检或安全labels。B/outer/inner/CAL封存不评价；registry全ID只用于对齐。

## 标签和求解
所有511potential按抽样窗口顺序/physicalslot升序，shape/pointwise各一次。原SLSQP zero/maxiter150/ftol1e-9/no retry，原position3节点4/8/12/tanh rawfloat32→float64原trajectorylambda/半径.20(t/12)^2；逐例GT数值oracle仅生成训练标签，不训练轨迹网络。
P=原samecandidate ADE<=.2 AND FDE<=.4/原位移界/其余19EXACT/完整K20五pooled motiondt.4 literal<=native1.01/独立decode均通过的实际保存见证，solver.success不是标签。
N仅端点必要排除：保存坐标精确Fraction差平方 > [float(.20)+float(.40)+1e-9]^2。1e-9只保守减少负例，科学门无epsilon；先核验64*float64eps*coordinate_scale*sqrt2<1e-9覆盖roundoff，超标STOP。若P/N或E/N冲突STOP。U=其余全部，含qualityonly/求解失败，非不可行。全部potential均求解，不按选中4决定求解。
单锚安全不保证联合4安全，P是已发现下界、U可有未发现解，不能PU-MAR/trueprob/心理意图/不可预测率主张。

## 监督唯一因素
原锁定S1b136features/linearLogRegC1L2newton-choleskytol1e-6max500/one thread不变；class_weight=None使用显式sample_weight避免doublebalance。Envelope positives=E，Witness positives=P；共同N，其他候选各自mask。所有训练fold共用包含>=1P和>=1N的相同窗口，validation不删。每track总权重相同、每window正/负各半，归一权重sum=20*N_common，两armC实际强度相同。共享commontrain全部20anchorsfloat64mean/std，std<1e-6→1。
这是positive membership+mask监督策略差异，不纯标量label替换。原136固定history下仍nativepath affine，不暗增MLP/capacity/features。
仅训练fold允许P/N筛common；validation所有384保持。不得用标签筛推理anchors。推理rawscore稳定descending、physicaltie，选4后physicalascending。

## 评价与门
Primary已发现P机会窗口上的Hit4；同时allwindowhit、P/Ulowerupper、unknownfraction、各组/scene/fold、旧S1bOOF分数描述性参考、fixedrandom4及原bottom4稳定ties、uniform4解析期望1-C(20-m,4)/C(20,4)。
Witness对Envelope/fixedrandom/bottom overall分别>=5pp且三个98.333...%pairedtargettrackCI下界>0；bootstrap2000 rng20261012/13/14，quantiles alpha=.05/6。change每fold不低于Envelope+fixedrandom；每validationfold>=20Pwindows且>=10Ptracks。支持不足则证据不足且不fit，不用in-sample冒OOF；门过只推荐独立Btransfer，不直接S2。保留全部反例/未知上下界，无subgroup替换primary。
两种fit相同fold/sharedIDs/statistics/weight rule/source绑定，6OOFfit无Torch/神经forward/backward；线性CPU优化和GT逐例求解如实报告。

## 实施审核与预算
继承上一feasibility2030全部相关digest，扩全部旧结果/补充/失败和人类输入，不覆盖hash，不全34k/旧GPUreplay。原数值映射copied代码逐字一致；actual128S1bfeature scalar/vector EXACT，source-IDsample排除与groupfold核验。固定前4potential双solver吞吐+decode独立原2e-12/完整sciencegate；实际真实feature的固定syntheticweightedlabel fixture用于两armrepeat/objective/Hessian/FD，明确不是真实P监督模型结果。真实OOF每fit重新审核权重/收敛stationarity原1e-6。
planning=(2*max_fixed4_paired_anchor_time*511+2*actual_sourceSHA_time+240)/60，含CPUfit/fullsaved审核缓冲，>30min不启动不减预算。process<=6GiB/physical+commit各reserve6；无OS/pagefile/userkill，内部guard非对话poll。
新目录唯一零triggerOS byte-lockworker，ONEstartup/ONE50%estimatedelapsed→completionCOUNT1，结果完整worker+exit+source/outputSHA+独立fullsaved审核才ANALYZED。完成seed/analysisSHA跳过须绑定源和所有输出，非神经RNGresume；事故code/log/partial/provenance全留。旧E3和旧automationPAUSED不恢复查询kill，baseline/historycycle11/formalcount不重置。

## ARS反方1（同助手非外审）
求解器发现偏差、selective commontraining/validation分布不同及屏蔽U不直接教P>U是关键限制；三折不是独立model seeds且F16seen/context相关/三scene/适应性设计。P增加不是真实repaircoverage。未知不改负、全部potential预算和新track排除阻断最明显乐观偏差，尚不证明迁移。后两反方在完整结果阶段进行。

## 保存基础设施恢复
父18条metadata不能直接恢复raw/path。按原SLSQP固定零初始化+同源数值重构18条，除elapsed外全部记录digest必须EXACT；明确18次数值重构，不假称无计算回放或optimizerRNGresume。每case在前进前保存raw/path独立checkpoint，完整SHA/审核。仅当前child write采用唯一tmp+fsync+6次有界PermissionError替换重试（总等待1.55s），不更改ACL/OS/权限，仍拒绝STOP。32次实际替换smoke通过后才能PASS。父全部源/partial/log/failure保留继承，科学定义/标准/预算不变。

最终record_recovery仅上述canonical JSON对NumPy类型的保存适配；源继承中间2090全部digest不覆盖。完整真实预检检查首8metadata和父记录非耗时逐项EXACT。中间未启动worker，不将预检当生产。
