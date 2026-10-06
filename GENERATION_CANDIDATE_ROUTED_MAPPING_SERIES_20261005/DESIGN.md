# 候选状态固定路由 vs 均匀共享：下一项单因素预注册
2026-10-05，设计快照。DESIGN_IMPLEMENTATION_PENDING；无代码预检、生产worker或正式启动。20:45HK仅实施预检节点，非训练完成ETA。

## 假设与依据
上一物理保护单因素已完整成功执行但major失败。加P显著降低ALLK加速度代价；finalFIT change同完整good仍2603→3237/3254/3265，但CAL243→230/242/232；DEVconstrained733→727/732/737。已有good通常为quality winner、credit不是首要缺口；不能把下一项自动写成loss根因。
检验：在相同physical保护、同双专家nominal预算下，候选状态决定的分工路由，是否比所有候选共用专家均值更容易形成可泛化完整路径且不伤运动学？
不是“共享干扰已证明”，也不是用新的名称扫原normalization/coeff/epoch；唯一注册因素为专家残差混合权重。不给未来GT、GTevent/group或logits作为路由/模型输入。

## 路由标量：只用past与原生预测
past为prepared940的前16（8×2），与之前相同缓存路径；vobs=(past[-1]-past[-2])/.4。
每个immutable native candidate b_k（12×2、起点0），v_k,t=(b_k,t-b_k,t-1)/.4。
q_k=sqrt(mean_12 ||v_k,t-vobs||²)，是原模型预测与末观测速度的偏离，不是实际未来变化概率/语义意图。
只FIT25912×K20=518240个scalar，生产float32逐256窗口计算，CPUfloat64保存固定median m。约1.98MiB标量，而非全候选对无界数组。m保存float64buffer两arm同值；route q以float64比较：q<=m→expert0，否则expert1；tie全部expert0，并报告tie数量/两route样本和slot计数。
禁止从CAL/DEV效果选择阈值、改窗口/heading/OLS、挑分组或扫描分位数。FIT路由退化（任一分支<10%候选或实际FIT8step无法验证两专家连接）为方法阻碍STOP，不改m以求PASS。该10%只是预注册可执行性检查，不是质量成功标准。原candidate state是原预测确定函数，不增加现实观测信息。

## 双专家与唯一因素
Both：两个原45798 trainable persistent raw-state专家，合计预期91596（必须实测计数，不手算作为证据）。每专家保持原state32→32→32 GELU、原nativecontext64+mode64与observedtokenattention、192→128→64→6、learnedsigmoid D2+A2+clock2/center.05-.95/span.05-.5及原rawFITscalers。
旧原decoder/context/modeMLP/F16/scorer全固定；两专家可共享同一固定originaldecoder引用/缓存，不复制科学checkpoint或改变旧buffer。原rawFITscalers及物理scales均从已锁定FIT定义加载，两arm同buffer，不CAL重fit。
每专家产生原bounded correction increments u_e（与原单专家同函数、clip和profile），zero：u=(u0+u1)/2；mean_std：u=u_route(k)。混合后原commonclip L=1.5*(speed+.03)，out=native+cumsum(u)。均值是凸组合；初始各expert最后层zero，所以两arm所有candidate nativeidentity。
两个expert分别从同seed固定初始化顺序生成（expert0/1不要求彼此同参数），两arm全部weights/bufferspaired。双方执行两专家计算和同q，control只有混合恒.5/.5，治疗只有0/1路由；不增加trainable gate参数，不route quality winner/scorer，不GT分配。
原softmax key.bias抵消；zero-last有分阶段梯度，按至少step3验证state/QKV/front连接，不把roundoff或初始zero当模型失败。
同nominal91596不等equal有效容量、每专家训练样本或优化几何；control均值与hard route函数不同，这是路由策略总效果。不能把改善纯归因共享参数干扰，更不能与之前单45798直接比较宣称容量无confound。fresh controls无旧head/warmstart/teacher。

## 固定科学协议和权限
两个arm都使用上一原total+.1 P_acc；P权重/1.01阈值/FITnative scales/group-n/B/空组规则/dt.4不动。这不是再试一个loss权重或追加项。
原hardwinner ADE+.5FDE及FIT balancing/diversity .05/residual .01/residualsmoothness .01/protectederror .1、CALguard.01、LR/batch/order/80epochs/3pairedseed/原FIT25912CAL2879不变。原immutableK20/full20/slotorder/no earlyprune/nativebounds；只有past16/frozenobserved8tokens/nativebase/speedforward。GT只原loss和事后指标；groups仅既有loss，不用于专家路由或predictionforward；logits不forward。
人类持续合法newhead研究与2026-10-05两阶段证据驱动许可作为依据；不扩大原F16/其他decoder/scorer训练、质量candidate assignment/newloss/event监督/heldtest/fold/externaldata/paidcloud/messages/delete权限。若实施发现双newhead不在当前精确模块许可或结构科学可辨识不足，STOP人类，不绕名。
前response3/3、candidate-state3/3已reviewed。本physical-feedback研究上一formal1/3，此项真实生产启动后才登记第二，completed保持1/3；非formal诊断不凑数，历史cycle11不重置。第三新formal完成必须fullreview全部假设/control/CI/coverage/有效goods/motion/FITCAL过拟合/容量与旧失败反例后转方向。

## 全部先验预检通过才启动
1. 全newcode/cache/log/models/results/failures仅本folder；继承所有旧manifest digests，扩入上一完成报告/补充/失败，不能覆盖旧hash。隔离AST只读旧source，不legacy顶层写入；source锁/IDs/旧modelsresultsSHA。
2. FIT-only q fixture/independentfloat64 scalar、constant-velocity q0、nativepred扰动、最后observed source、阈值ties、1033row31/256chunk排序与median、两routecounts和buffer加载验证。Route指标仅FIT、GT无关。已锁定老scalers/物理scales定义不能改。
3. 当前GPU8实际FITsteps，两专家前后层/state/QKV/clock正有限grad，两route有实际支持；commonclip/bounds、finite128K20、oldallnograd/paramsboolscalers逐torch.equal、paired初始化全部weightsbuffers/nativeidentity/neighbor数据不混raw940；原frozen cached-online alignment、原loss+.1P scalar与gradient不变。初始P不活跃额外仅synthetic数值核验并如实报告，不改P规则。
4. BOTHreal FIT/CAL-subset2epoch continuous/interrupted params/allboolscalerbuffers/optimizer/torchCUDAloadernumpyPythonRNG/history/calibration/objective diagnostics EXACT及完成seedSHA-skip/sourcebadoutputreject。不是完整25912两轮或分析恢复替代。
5. 36throwawayFIT128（4warm32measure）+CAL2879实测吞吐/RAM；不savedbenchmarkmodel或选科学outcome。one model/batch128、route stats256，process6GiB/systemphysical+commit每项留6GiB、长期~17GBavailable，不OS/pagefile/userkill。91596参数float32~.349MiB，Adamstates+grads额外少于2MiB，激活/缓存/输出/hash同时驻留实际peak决定可行性，不能只按参数数估内存。无界全18outputs缓存禁止。
6. source/current-realGPU/realresume/资源全部PASS后才报实际ETA。historical22–269min变速非保证，20:45不是生产ETA。独立unique Windows按需0trigger start_independent.ps1 byte-lock，确认无同活跃task，ONLY一次实际PythonPID/serviceparent/currentstage/stderr；未看到TRAIN不能声称epoch。
7. 真实启动后更新runtime/memory/ledger、同automation COUNT1唯一50%elapsed检查/完成计划，再EXIT静默。不轮询对话；唯一half实际剩余合理修正COUNT1 completion、无每日重复旧节点。

## 完成与成功标准
双worker/exitSUCCESS及全部audits。六new80训练18policies/18explicit1516uniqueIDfinite20×12×2/allsample/nativeALLgroups/perseed/bootstrapCI/同完整candidateADE<=.2 AND FDE<=.4覆盖/rescue-lost优先/GT-freecomplete-link.10meanpath/.20endpoint有效good非maximize，最后scorer旧argmax只是继承slot。
originaltotal/P、物理population FITCAL36记录、80epochtrace、路由counts及保存权重/混合复核、全部sourceoldSHA/frozen原decoderbuffers exact。保存算术不是所有model/grad再次forward。quality Jclosest与temporal Qjoint单slot不同，不拼早中晚oracle；clockcenter/span不是语义事件。
major全部原样：changeADE/FDE各>=5%gain、overallstabletransition harm<=1%、3seed一致、native/freshcontrolADE-FDE-jointCIexclude0、samepathcontrol1%、coverage重复hardmotion AND 各seedgroup pooledspeedp95 accelmean,p95 jerkmean,p95<=native1%harmdt.4。.3/.5只固定敏感性，不删难例/换阈值；firstallgates另审scene/sample相关及3seed充分性。
明确失败也有信息：若route没有覆盖净改善/只有FIT提升CALlost/运动学FAIL，不靠扩大专家数、调median、调P系数继续无限变体。完成报告反方2/3（同助手非外审）；科学/ID/源/资源/许可实质阻碍STOP保留证据等人类。

## 反方1
Major：q是native预测与末观测速度差，不是可预判GT事件；路由可强化旧candidate错误，且静止速度0会改变分布。FIT数值和各scene计数预检不可少。
Major：对照平均 vs selector改变函数和分支梯度分布，不能宣称等容量或唯一识别“共享干扰”。只检验实际路由干预是否有用。
Major：此前explicitstate/discrepancy/conditioninggeometry/mode/social、Gaussian/persistent/freeincrements/timequery/velocitybasis、softset-energy-mixture失败仍为反证；新名字不是排除历史失败。
Major：P仍可能压制真实响应、mean hinge非globalgate；所有五运动学与完整coverage硬门槛保持。
Major：CAL已选checkpoint/单university/3scene/window相关/F16seen/repeatedDEV/3seed，是探索非SOTA/意图/独立泛化。
Verdict PASS_EXPLORATORY_DESIGN_PENDING_ACTUAL_PREFLIGHT；任何critical失败STOP，不使用外部审稿或模拟审核通过标签。

