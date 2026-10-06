# 阶段二：直接物理加速度反馈的单因素对照

## 依据与待检验假设
人类2026-10-05明确指定loss→coverage、credit、quality-motion诊断，并授权依据其分支进行两阶段研究。阶段一已完整ANALYZED，见../F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md。
FIT九final覆盖40.61→51.50–53.87%，但加速度均值+15.09–17.89%；CAL覆盖34.71→30.86–34.57%、加速度+14.06–16.27%。Constrained九head FIT/CAL quality下降方向均与物理accelmean冲突，原residual-smoothness尺度很弱且对象不是绝对轨迹运动学。已有good winner通常正确，不优先credit重分配。Final CAL局部mean方向有反例，不能把全部失败归于一个cos。
测试：对固定raw candidate-state/persistent head，加入native-relative物理加速度均值保护，能否保留完整coverage学习并减少副作用？这是反馈干预，不证明当前参数化没有结构性矛盾，也不承诺泛化改善。

## 唯一因素与明确数学定义
Both: actual immutable-native raw candidate-state32/FIT rawscalers、state_encoder32→32→32 GELU、原context64+mode64和observedtoken attention、192→128→64→6、45798相同参数/全paired初始化与buffers。双方state与clock均active。原learned sigmoid persistent D2+A2+clock2、center.05–.95/span.05–.5、u=.5L(tanhD+tanhA*profile)、L=1.5(speed+.03)、native+cumsum/commonclip保持。
zero=原total；mean_std=原total + .1 P_acc。Generic名称只标签；不是masked-state/discrepancy/另一个profile。RAW控制为此前actual架构fresh执行重复；不warmstart旧head/failedteacher，不提前假定输出bitwise重复。

在原训练minibatch B，所有20候选和12预测点x：
v_t=(x_t-x_(t-1))/.4，x_0=0；
a_t=(v_t-v_(t-1))/.4，t=2..12，共11段；
m_i(x)=mean_(20×11) ||a||_2。
对于现有group g∈stable,transition,change，I_g为当前batch原组标签索引：
A_g(x)=mean_(i∈I_g) m_i(x)；A_g(base)同immutable native K20。
s_g=max(mean_(FIT该原组所有窗口)m_i(base),.001 m/s²)。仅FIT25912 native，float32 production路径/float64统计256chunk一次锁定，两armbuffer相同且checkpoint加载；不CAL/DEV选择/重fit。
P_acc=Σ_(g:n_g>0) (n_g/B) ReLU[A_g(x)-1.01 A_g(base)]/s_g。
空组贡献0，native参考detach。阈值1%沿原motion gate；权重.1沿原protection coefficient，不做coeff/floor/LR/epoch扫参。前向不读group，group只沿既有loss训练标签；这不是新事件/onset/方向监督，不需要未来GT提供native物理reference。GT依旧仅原quality/protection和posthoc标签。

保留原primary hardwinner ADE+.5FDE及FITclassweights、原diversity .05/residual .01/residual-smoothness .01/protected-error .1、原CALguard.01/LR/order/batch80epochs/3pairedseeds。不替换原smoothness、不重新分配winner，不同时增加jerk/speed/p95 penalties。Jerk和全部五物理项仍必须接受原1%硬审核。
P是batch组均值hinge，**不等完整FIT/CAL全局均值gate**；非窗口逐条禁止转弯。batch抽样可能让population可接受仍被惩罚，数值尺度/有限优化几何随新增项改变正是干预，不能宣称等优化或无副作用。权重数不变不等“已经证明只loss是根因”。

## 权限与不变范围
本项是人类两阶段分支中quality-motion反馈研究的具体有限干预；不是此前固定loss许可的偷偷改名。除该物理保护项外全部旧协议保持。任何额外loss/assignment、事件监督、更广F16/原decoder/scorer训练/解冻、heldtest/其他fold/externaldata/paidcloud/messages/delete仍需具体新许可。
WholeoldF16/wholeoriginaldecoder/scorer权重/布尔buffers exactfixed；nativeK20 immutable/full20/originalslotorder，不earlyprune。forward只有past0:16/frozenobserved8tokens16:784/restzero/nativebase/speed，不GTfuture/groups/logitsforward，不AR/新未来观测/意图标签。科学FIT25912/CAL2879原IDs和所有原组不删难例；DEV1516只有原固定完成评估，不用来调loss尺度/参数/checkpoint。
所有新code/cache/log/models/results/failure只本全新folder，父scientificsources/models/results及所有旧manifestdigests继承保留、扩入上轮报告/失败，不能重新计算覆盖旧digests。

## 预检要求：PASS后才允许独立正式启动
1. 原control AST total标量/gradient与原实现实际对齐，原loss不重排；zero与此前raw actual同权重非零last输出features/path/residual一致，两arm全paired/nativezeroidentity；旧wholedecoder逐权重boolbuffer exact且no oldgradient。
2. FIT-only s_g/IDs统计锁定，1033跨区段31/256chunk fixture；native输入P=0，knownpositive/negative/threshold/emptygroup/two-group/batchweights fixtures，numpy物理dt.4标量与autograd/数值差分；故意bad-ID/scaler/source/output拒绝。
3. 真实GPU8FIT多步：双方head/state/QKV/clock有限非零梯度按zero-last分阶段，treatment active P梯度有限、control addedterm为0；如果8步未激活hinge，额外固定synthetic active-output只数值核验，不能谎称自然训练已active或擅改权重。旧参数完全nograd、finite128K20、原clip bounds、online原cachealignment/source/split/全部旧SHA前后PASS。
4. 两arm真实FIT/CAL-subset2epoch连续/中断 params/allbool/scalerbuffers/optimizer/torchCUDAloadernumpyPythonRNG/history/calibration EXACT及completedseedSHA-skip/source坏输出拒绝；如P激活导致不可重现，实质阻碍STOP不是放宽容差。
5. 独立36throwawayFIT128步4warm32measure +CAL2879实测，只吞吐/no savedbenchmarkmodel/no scientificoutcome选择。memory实测，约17GBavailable持续；process6GiB、systemphysical+commit每项至少6GiBreserve，batch128、有限groups/scalars，无全18缓存或OS/pagefile/userprogramkill。
6. Current-source/实际preflight/realresume/RAM全PASS后报生产ETA；unique Windows按需0trigger、start_independent.ps1 OS byte-lock确认无同活跃task，一次真实PythonPID/serviceparent/stage/stderr确认。未观察TRAIN不能宣称；真实启动时才新formal study0/3，不提前计数/重置历史cycle11。
7. 启动后唯一50%elapsedTIME检查，原single COUNT=1 nativeheartbeat；按真实剩余修正一次性completionnode然后EXIT静默，不对话轮询/旧PID查询/每天重复/慢就重启。

## 完成与判定
6fresh80epoch/3pairedseeds×2arms，原3policies=18outputs不是18独立训练。Doubleworker+exitSUCCESS，18policySHA/18显式1516ID finite20×12×2/6complete80history/source/allprioroldSHA/frozen全decoderbufferexact。全部saved sample/nativeALLgroups/perseed/bootstrapCI/quality全K20GT-freecomplete-link .10meanpath/.20endpoint/temporalpulse/allpooledmotion独立算术重算。新增original-total/P_acc/add-total和各组样本数/尺度/激活率、gradnorm记录，说明trainbatch vs全population重算不同；单独验证新增loss项而非改旧loss。
主要：同一完整candidate ADE≤.2 AND FDE≤.4至少1覆盖/rescue-lost；有效good簇其次（数量非maximize/copies），scorer最后。固定.3/.5敏感性，不切标准。quality closest J单slot与temporal Q单slot不可拼段；旧nativeargmax继承slot非新路径概率。
所有原major原样：changeADE/FDE各≥5%gain、protectedoverallstabletransition≤1%harm、三seed一致、nativefreshcontrolADE/FDE/jointCIexclude0/samepathcontrol1%、coverage/duplicate/hardmotion AND 每seed每组 pooledspeedp95 accelmean,p95 jerkmean,p95≤native1%harmdt.4。Firstallgates还需scene/sample相关和3seed充分性审核；重复ETHDEV探索非独立泛化/SOTA/意图。
即使accelmean减小，若coverage被压回native/降低或jerk/p95/保护/CAL失败，仍不是成功。不靠新增CAL选checkpoint规则来制造PASS。报告全部arm政策和constrained/final FITCAL迁移。
如果physical反馈足够改善原量却coverage/CAL仍失败，保留mapping/输出参数化路线；如果P没有效梯度/源锁ID资源科学不可辨识，STOP等人类。不要顺手扫.01/.1/1/10或加第二motion项。每3新批准formal全审计后总review再转大方向；本单项完成先报告，不由诊断数量凑formal。

## 反方1（同助手内联，非外部审稿）
Major：native相对保护可能保守，真正转向需要更多加速度；应观察rescue/lost与结构抑制，不称保护越强越好。
Major：只accelmean不是所有motion项，jerk与p95仍可能坏；不放宽原硬门槛。
Major：loss新增项改变有限优化，不证明唯一loss根因；batchhinge与populationgate不同。
Major：仅原数据/3seed/CAL单university+CALSELECTION/windows相关，不能保证泛化/SOTA。
Decision PASS_EXPLORATORY_SINGLEFACTOR_PENDING_ACTUAL_GATES；实施节点18:20HK，不是生产ETA。科学或具体权限实质阻碍立即STOP人类，不越权。

## 当前状态
DESIGN_IMPLEMENTATION_PENDING；尚无实现/preflight/worker/正式训练，不冒称running。下一节点2026-10-05T18:20:00+08:00开展实现与预检，实际独立solelaunch后再报生产ETA和half/completion。
