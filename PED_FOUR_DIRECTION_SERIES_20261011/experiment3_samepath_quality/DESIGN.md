# 方向3预注册：相同位置修正下的同路径瓶颈质量目标

## 动机与唯一因素

方向1position3全36population五motion PASS，严格outer change仅110→111且CI跨0，未达major；acceleration3有加速度均值代价。方向2支持保留原生，但新颖且有冗余槽的31382次尝试均被物理门拒绝，最终policy=原生。不能通过松门/强删/换slot/cluster阈值扫参制造成功。

本项固定position3输出映射，只改变四条完整修正路径的质量functional：
control sum：S_j=ADE_j+.5FDE_j，winner argmin_j S_j，quality=min_j S_j。
treatment joint：T_j=max(ADE_j,.5FDE_j)，winner argmin_j T_j，quality=min_j T_j=.20*min_j max(ADE_j/.20,FDE_j/.40)。
两项单位都是米。每window独立求winner；hardwinner assignment和winner loss由同一functional决定。不能拼minADE与minFDE来自不同路径。T的ADE/FDE精确相等时，torch.stack([ADE,.5FDE],-1).max(-1)第一分量先（ADE），candidate ties第一ascending physical slot。记录ties与活动面；梯度审核不得把ties/量化噪声当有效连接。

原total=quality batch mean+.05*六pair平均平方hinge(.3)；diversity所有路径相同定义、分母与权重。不新增protection项、classbalance、motion项、softwinner、assignment温度或任何扫参。此项在获准质量/保护方向中先隔离quality目标；并未测试显式保护目标，更不能宣称保护已解决。

这同时改变winner责任和选中路径的误差聚合，因此不是纯assignment因果；两function不同数值和优化几何也需保留。无需声称sum是唯一根因，joint不保证严格coverage改善或finite AdamW收益。

## 结构、输入和全部固定量

固定方向1position3原24134参数120→128→64→6 GELU，P2已观察8tokens平均96D+不可变selected native12×2(24D)。共同训练18909×4统计：float32features、float64mean/std累积，std<1e-6→1，再保存float32，两arm所有mean/std/basis buffer相同。禁止inner/outer/CAL参与拟统计。

3位置节点t=4/8/12，tanh向量归一L2≤1，节点乘.20(t/12)^2，t=0残差0后逐段线性插值，原单GT-free trajectorylambda保证所有t位置界。head float32、anchor/residual输出float64。该位置界不是物理安全证明；固定structure必须actualnonzero输出与原position3 BITWISE EXACT，不能改实现求PASS。

原bottom4 immutable frozenlogits stable原slot ties，ascending physicalslots，剩16native EXACT，K20。原生native20 zeroidentity是双方共同基准。方向2policy不嵌入训练或替代primary：它已输出身份且没有可用梯度；整合政策因素不得与目标因素混在一起。原bottom策略可能丢tail支持，必须保留lost。方向4继续gated。

target grouped原train18909/outer1035 fixedepoch25 primary/inner1856/originalCAL2879 posthoc only。seeds42/43/44，双方freshpairedinit=六25runs、150epochs、18保存scientific jobs。不warmstart，不选择checkpoint。AdamW lr.001 wd.0001 batch256 shuffle25 noclip和原batch loader/order保持；全optimizer参数只newhead。Control fresh重复原position3不是独立泛化，完成逐array核验，不预设重复bitwise。

所有旧F16/native20/originaldecoder/scorer immutablecache/sourceSHA；不实例化/训练/optimizer访问旧模块。GT只合法FIT loss和posthoc，groups/IDs/scene/logits不headforward，logits仅已固定slotmap。No DEV/test/其他fold/externaldata/teacher/event/scorer/newoldunfreeze。

## 必须实现的审核

1. 新shim只读AST/类隔离，所有新增只本folder。继承当前1614全部旧digests扩方向2的91输出、全部完成报告/复核/归因/来源与方向1额外CPU审核，旧hash不替换。验证完整IDs/scalers/source，不将相关manifest说成全历史34k审核。
2. NumPy单独sum/joint标量和samecandidate winner、不同路径分别minADE/FDE反例、ties/zero/copies/empty允许语义fixture，原diversity对齐。不能根据结果调系数。
3. Actual GPU8 FIT256，两arm canonical control loss及全部gradient与旧实现 EXACT；treatment原定义独立NumPy标量、固定非零double参数中心FD eps1e-4/1e-5审核。硬winner、max component或trajectorylambda活动面发生切换就记录不可辨并STOP，不放宽容差或另搜eps求PASS。合法newhead分阶段positive finite梯度（zero-last需分阶段），completepairedweights/bool/scalers/basis、initial native20zeroidentity、finiteK20/fixed16/bounds/nativecacheSHA。
4. 两arm真实513FIT+12inner SUBSET2epoch continuous/interrupted ALL参数与bool/stat/basis buffers、AdamW state、TorchCPU/CUDA/loader/Python/NumPy RNG、history/新增sum-joint诊断exact；completed SHA-skip/坏source与output/resumecontract真实拒绝。不是完整18909两轮，也不是分析恢复。
5. Actual outer128保存读回 full同候选quality/原SciPy GT-free ALLK20complete-link .10meanpath/.20endpoint/有效好簇/five pooled motion/control NumPy/bounds/fixed16，eval全weights bool/stat buffers EXACT。不所有旧GPU重放。
6. 36throwaway/arm4warm32 FIT256实测与inner1856 forward/实际full3splitmetric吞吐、peakRAM/disk输入输出临时规模。无保存benchmark模型/科学outcome选择。规划>30min则不生产，保留测量，不缩epoch或改协议。
7. 全科学/ID/GPU/source/resume/resource PASS后锁源；仅unique0trigger Windows独立OS byte-lock，先确认无同task活跃，ONE actualPython/serviceparent/creation/fullcommandline/task0/stage/stderr startup。只有真实生产才标started并报告ETA，未看到epoch不能冒称。

## 输出与诊断分母

每run checkpoint.pt/run_manifest.json/metrics.json/assignment_stats.json及splitIDs/statistics/history，完整绑定sourceSHA/splitSHA/IDs/statistics/objective/tie rule/init/optimizer ALLRNG。History保存每batch n加权pre-update quality/diversity/total、sum vsjoint assignment switch与counts、max component/ties；不混posthoc population。每split保存native20/refined20/refined4/rawcontrols/residual/explicit IDs/GT/groups/track/scene/slots，18有限输出；全source/output initialfinal SHA。

同时用保存的相同4paths报告两functional switch，不把目标变小当覆盖成功；控制history遵循sum但仍记录joint counterassignment，joint反之。只有component值/固定winner诊断，不GT选择slot或部署。

## 完成门与决策

worker+exit双SUCCESS、六完整25history150rows、六完整manifest checkpoint、18保存输出SHA/ARITHMETIC/PRESERVATION/RESOURCE/全部源初末不变才能完整。新completion_review CPU独立全18samecandidateADE≤.2AND FDE≤.4/J、rescue-lost-retained、originalSciPy ALLKcomplete-link .10/.20有效good-distinct、单Jclosest完整path早中晚、五pooled dt.4speedp95 accelmean/p95 jerkmean/p95、16bounds/controls/assignment/counts与history、36track+3seed bootstrap144scalar CI(2000 rng20261011)、所有source-output SHA；无Torch/model/gradient第二次重放，非外审。保存RESULT_ANALYSIS MD/JSON及同助手inlineDA2/3，runtimeANALYZED再报告。

全部原major：changeADE/FDE each≥5%gain、overall/stable/transition harm≤1%、三seed一致、nativefreshADE/FDE/J联合CI排0、samepath coverage/duplicatehardphysics、每seed每group fiveMotion literal≤native1.01，不放宽。.3/.5仅固定敏感性。joint若覆盖仍稀薄/丢tail/运动学FAIL，就负结果，不自动softwinner/protectionweight/节点/widthLR epoch局部扫参。

方向4只有非零稳定完整coverage和所有motion门触发才固定generator再精确预注册独立评分器；否则报告未触发不假完成。当前四方向授权不等旧scorer覆盖或更广数据授权。源/ID/cache/resource/science/scope实质障碍STOP保留真实failure等人类，数值routine助手负责不泛问容差。旧E3/旧automation保持PAUSED，不查询/恢复/kill；所有历史完成PID/task/log不查不重训。

持续process6GiB/physical+commit eachreserve6GiB，~17GBhost偏好，单model/batch256有限输出/internalguard非对话poll，不OS/pagefile/userkill。ONE50%estimatedelapsed→oncecompletion，实际remaining train/calIO/保存qualitycluster-motionCI/finalSHA修deadline，同automation-2 COUNT1，静默退出，无每日重复/慢重启。机器awake/loginOSworker独立，分析需app。

## ARS反方1（同助手非外审）

PASS_WITH_LIMITATIONS。固定结构排除结构变化，但sum→minmax同时改变credit与误差geometry，不能纯credit归因；同单位不表示相同损失量级或有效梯度尺度。Position3依据已注册primary安全证据选而非CAL收益。Bottom4支撑风险仍在，所以lost和protected组不可省。F16已见原pool、target不等context-disjoint、3scene3seed/相关window/reusedCAL/adaptive系列，不理解运动/SOTA/wholemodel独立泛化。方向2零插入不证明所有政策不可能，禁止强删/sweep绕负结果。

