# Loss→完整候选 / credit assignment / 运动学：冻结诊断
本轮人类2026-10-05明确提出三个诊断主题，撤销上一等待科学路线决定，批准本冻结诊断而非newloss训练。所有artifact只此新folder。NONFORMAL/nooptimizer/newtraining/oldweightsupdates，无DEV/test/fold/externaldata/scorer/unfreeze。

## 第一项：checkpoint间一致性
主对象3family×3seed×constrained/final18已有checkpoint，native为参考。同一FIT25912/CAL2879显式ID原组stable/transition/change，change为主、其余反例。比较native→constrained、constrained→final、native→final；constrained由CAL选定，非随机时点，只有端点不能声称完整训练逐步因果。无需重训，训练history仅背景，不将train_loss batch均值当某样例loss。
逐sample保存原quality/diversity/residual/smoothness/protection及total的全split人口归一化贡献，和ΔL_total/primary、minADE/minFDE、同slotjointADE/FDE、J_joint=minK max(ADE/.2,FDE/.4)、覆盖/rescue/lost/rawgood/effectivegood、GT-free所有簇数、逐window运动学。
J_joint<=1严格同完整candidateADE<=.2 AND FDE<=.4；原ADE+.5FDE hardjoint为Q_joint另名，不混淆。两种slot跟踪及每原slot20候选ΔADE/FDE/J，不能拼不同slot或分段oracle。main阈值不更改，.3/.5仅既有敏感性。本研究不定义“多样性越多越好”。
Δ定义after-before；loss明显下降采用1e-8稳定裕度，J按1e-8描述变化/无变化；边界数值敏感性另报，不改coverage标准。互斥分类：L下降且覆盖救回/覆盖丢失/保持good/保持bad（bad中J改善vs未改善）；辅标签loss下降但J变差、FDE变差、簇数/有效good下降、accel/jerk上升。按组/seed分开，重复IDs不加独立人数。Pearson/Spearman描述关系及固定loss改善分位分桶，不p值或可靠sceneCI。样例明细NPZ供后续可视化，无本轮新绘图。

## 第二项：hardwinner候选质量credit
每checkpoint每slot保存ADE/FDE/J、是否good、primary hardwinner、tie gap、直接输出坐标质量梯度norm/直接非赢家norm、regularizers/protection/total输出gradientnorm。head共享参数间接作用不能由坐标0推为完全无学习。优先检验good_present_winner_bad、normalizedjoint closest不是hardwinner、J最接近good但无primary直接反馈；全candidate槽位不重排。
复用旧GT辅助pulse构造的“某native槽位存在合格路径”的事后witness：取所有feasible槽位中固定q最小slot，保存原源ID/路径/slot，对每当前路径仅替换该slot，19其他slot不动，计算原fullpopulationloss有限变化和当前primary/total梯度朝该替换的方向导数。标为OLD_GAUSSIAN_PULSE_GT_WITNESS，不宣称当前sigmoid共享head可学或同结构可达，不把GT输入预测forward。候选质量方向零与有限替换loss下降的交集为描述证据，而非换assignment已证明有效。旧solver失败未知，绝不拟新GTteacher或训练。

## 第三项：质量与运动学梯度
在现有18checkpoint原权重不变、旧F16/decoder/scorerrequires_gradFalse下，只合法newhead45798参数求导；change组fullFIT6410/fullCAL700主分析。不进行optimizer/参数更新或训练。
固定向量：原primary、原residual/diversity/smoothness/protection/total；物理pooledALLK speedp95、accelmean/p95、jerkmean/p95(dt.4)。物理指标是事后诊断梯度，不加入训练loss。change保护项按原规则恒0，记UNDETERMINED，不把它当冲突或不存在监督。
为原total分项采用完整split人口固定denominator的可加贡献，同snapshot第一项，change均值导数不假称原mixedbatch+dropout+AdamW实际更新。原smoothness罚的是修正增量的差分平方，不等全路径物理加速度；需同时报Q与两者cos/dot/norm、沿-Q下降方向运动学方向导数。positive运动学方向导数=局部质量下降有害，negative有利，nearzero<=1e-10 UNDETERMINED。
globalp95使用整个change组ALLK的固定排序/linearquantile两秩输出，在真实batchmodel图上重放所选元素的加权导数；保存两秩flatindex/value/weights/ties。p95gradient极稀疏、tie非光滑，不宣称大多数样例方向；不拿windowp95mean当globalp95。至少预检quantile值与torch/numpy、actual分块gradient31/64/128、数值副本有限差分（winner切换/tie记录，不能辨识STOP）。

## 预检、资源、恢复、流程
继承上一parameter29983old及445source所有digest，扩入其完成报告及补充，不覆盖旧hash；所有science/models/results不改。实际loss AST标量/坐标gradient与独立components核验1e-10 double；实际model128path对matched保存<=2e-5、K20finite/bounds/IDs/scalers/noGTforward/frozen全weightsbuffers exact。primary坐标非赢家0/tie按torch.min固定first，synthetic好路径非winner/disjointminima/20copies1簇与已知冲突/同向/零norm。actualparam梯度chunk阈值1e-6+2e-4norm；double副本FD eps1e-4/1e-5绝差1e-7+1e-3abs导数，物理零norm/tie只记录无无理PASS。fixture坏source/output拒绝与completedSHA-skip不是训练2epochRNG。
约17GBavailablehostRAM持久/process6GiB/systemphysical+commit各6GiBreserve。一次checkpoint/一split、CPUlosschunk64或128、GPU<=128、一物理向量；不常驻18图/全18路径。输入169MB/oneoutput55MB/每checkpoint少量vectors及candidate诊断列，规模实测与benchmark后才报ETA，非微基准生产保证。不改OS/pagefile或用户程序。
独立Windows0trigger unique任务/byte-lock实际预检全部通过后才solelaunch，一次真实Python/服务parent/task/stage/stderr确认，保存runtime/memoryledger/ONE50%timequery/预计completion同步automation，启动后退出不对话轮询。half一次，完成即审计；否则实测remainingstage/job/算术/SHA/IO修正completion后静默。源/ID/resources/scientific不可辨识等实质阻碍保留failureSTOPdirecthuman。
完成worker+exitSUCCESS/alljobs/outputSHA/全savedmetric与gradient摘要算术独立重算/frozen/source/alloldSHA/资源，RESULT_ANALYSIS及inlineDA2/3。人类最新追加两阶段持续研究授权，按DECISION_PLAN.md的分支证据及反证选择精确可辨识下一对照；不是自动同时换loss/scorer/解冻。科学不可辨识或具体模块权限不足时STOP等人类。诊断NONFORMAL不增正式计数，不改major和全组1%运动学保护/F16baseline。

## 反方检查1（同助手内联、非外部审稿）
Major：population loss非trainbatch total，分母差异和dropout/AdamW不推因果，显式限制为evalsnapshot。
Major：oldGaussian witness不是当前persistent共享映射的可学习证明，严格来源命名与19untouched验证。
Major：p95两秩gradient不能代表全轨迹群，tie与nearzero标注，不挑单个cos结论。
Minor：constrained由CAL选，端点对比不能量化“每训练一步效率”，不回看DEV选新的时点。
判定PASS_WITH_LIMITATIONS，足以检验人类三个描述/局部一阶问题，不授权修改loss。
