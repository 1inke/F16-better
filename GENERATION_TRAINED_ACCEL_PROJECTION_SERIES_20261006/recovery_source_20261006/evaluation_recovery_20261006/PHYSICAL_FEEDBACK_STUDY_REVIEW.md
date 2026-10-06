# Physical-feedback三项完整总复盘
2026-10-06。academic-research-suite证据/反方流程；同一助手inline，不是独立外部审稿。完整数值附 PHYSICAL_FEEDBACK_STUDY_REVIEW.json及三study sidecars；原始证据引用三个实验各自 results/summary.json、quality/summary.json、physical_population/summary.json及completion_review/REVIEW.json。覆盖全部72policy-group aggregate、288探索CI、216 seed-group质量/运动学、18训练history1440epochrows与54policy模型；54policy不是54次独立训练。

## 大结论
三项计算/审计都完整通过，但major均FAIL。固定物理P减少加速度代价；固定candidate路由没有解决迁移；训练内端点保持投影首次让正式constrained三seed各四组五motion全通过，但完整coverage、变化ADE/FDE、protected误差和稳定CAL/DEV收益仍未同时成立。baseline不变。
现在仍是“FIT能形成更多完整候选，但变化泛化/共享映射不稳定”的分支，同时final速度副作用未全部消失。不是“loss必错”、不是“所有反馈分配完全正确/梯度无冲突”的终端结论。不能提前换scorer，因为生成仍没有稳定扩大好候选覆盖；继承旧slot从来不是新score概率。

## 三对照，不能跨研究偷换因果
1. PhysicalAccelProtection：双方45798，solefactor原total vs+.1P。rawcontrol9policy与先前CandidateState实际state bitwise重复，fresh重跑不是新独立泛化。
2. CandidateRoutedMapping：双方91596双expert、都+.1P，solefactor均匀混合 vs fixedFITmedian路由；同名义参数非同effective容量/样本梯度/优化。
3. TrainedAccelProjection：双方91596 uniform、都+.1P、float64输出，solefactor投影参与前向反向 vs不投影。与第二uniform float32不同，不假定bitwise重复；函数可行域与优化几何耦合。
跨三项同时容量/数值结构不同，不将每项相对上一项的均值变化称纯物理/结构因果。每一项只能首先解释自己的paired control。

| 治疗constrained | changeADE/FDE | overall good /1516 | change good /223 | pooled五motion通过seed-group /12 |
|---|---|---|---|---|
| Native | .372174740/.696326435 | 733 | 43 | 参照 |
| +P | .371380866/.690220416 | 727/732/737 | 42/47/52 | 4/12 |
| fixedroute | .373324096/.703389943 | 727/738/729 | 42/47/43 | 9/12 |
| trainprojection | .372979381/.707470318 | 736/735/735 | 47/48/42 | 12/12 |

三seed同IDs不加成独立人；native/fresh主要变化ADE/FDE/jointCI全部未给出稳定全项优越证据（top3、其他组或legacy某CI不替代主门槛）。有效簇、coverage、误差、motion必须并列：更多effectivegood不能抵coverage丢失，近静止聚簇不能自动叫collapse，.3/.5只敏感性不可切标准追认成功。

## 所有policy/组的审核
JSONsidecars逐项包含两arm×constrained/legacy/final×overall/stable/transition/change的全部24均值与native相对ADE/FDE；全部96CI按policy/reference记录正负/跨0，并可回到原summary核对所有区间，不只挑好的seed。全部72 seed-group覆盖/rescue/lost/effectivegood与pooled五motion计入。
-P治疗constrained4/12、legacy0/12、final0/12五motionPASS；固定route9/12、1/12、0/12；trainprojection12/12、12/12、9/12。未用window-p95平均替代globalpooledp95。最后三个失败是change speedp95，accmean已经守住不等allmotion全守住。
所有major的5%变化双误差、protected1%、三seed、nativefreshCI、同路径/coverage/重复/hardmotion和ALLmotion1%原样。三项各自science冻结、resume与输入source审核、最终saved算术/全oldSHA已通过；当前恢复父训练失败属于已修复的读取器基础设施，非科学标准改变。原failedcode/log/results仍原样保留。

## 共同的FIT→CAL问题与反例
同组 final change，native FIT2603/6410、CAL243/700：

| 治疗final | FIT good seed1/2/3 | CAL good seed1/2/3 | change accmean害 FIT / CAL |
|---|---|---|---|
| +P | 3237/3254/3265 | 230/242/232 | 1.72–1.88% /1.44–1.55% |
| fixedroute | 3642/3632/3616 | 227/237/221 | 2.25–2.44% /1.91–2.13% |
| trainprojection | 3621/3534/3560 | 226/232/222 | .818–.835% /.815–.833% |

三项final FIT都有大量完整候选收益，CAL均没稳定超过native；第三甚至已控制accmean仍如此。因此“超额加速度是所有FIT收益必要条件”不支持；“修正路径能形成”不等“在未知未来可预测/可泛化”。
反例必须保留：constrainedCAL +P244/247/246、route245/239/244、projection248/252/248，部分确实小幅提高。不能说所有CAL/所有checkpoint都恶化。但这些由CAL筛选过，重复同CAL不是独立泛化。训练最终继续提高FIT同时损失CAL，既可能共享映射/有限优化/场景分布，也可能仍有objective/per-sample干扰；当前不能一锤定因果根因。

## 中间诊断及历史反证
原loss–质量诊断：FIT loss↓完整覆盖↑但motion差，CALloss并非也下降；已有good时qualitywinner通常good，Jclosest不等最可修。旧Gaussian GTwitness插入降低loss不证明persistent可学，nonwinner coordinateprimary0不排间接共享更新。物理局部梯度cos/稀疏p95支撑/ties不等有限mixedbatch AdamW更新。当前投影没有重测所有原梯度，不能假称它们不再冲突。
冻结posthoc投影保留FIT已有good约99.26–99.66%，但CAL仍差且7speedp95fail；训练内投影是后续明确单因素，不把posthoc结果当训练收益。既往freeincrements/timequery/velocitybasis、state/discrepancy/normalization、conditioning/mode/social、tinyFIT泛化、softset/energy/mixture失败全部保留；不以新名字重复同类局部扫参。

## ARS反方2（inline）
Major：跨研究参数/精度与容量不同，不能把顺序表当因果链；保留各自freshcontrol。
Major：只有3scene、CAL单university、相关windows、F16已见、CAL多次选择与DEV重复、3seed，不报告SOTA/独立泛化/意图或不可预测率。
最强反驳：“你们已经在被反复探索的CAL/DEV上做了很多尝试，小幅覆盖变化可能只是筛选或适配；即使motion通过也不能证明学会新的运动变化。”
结论保留有效工程/约束进展，同时拒绝major升级和泛化根因强声称。

## ARS反方3与下一决策
报告PASS限域；研究目标未完成，停止第四局部投影/routequantile/weight/width/LR/epoch/profile。
当前可靠下一路线是验证newhead迁移，不是预先断言换loss/decoder/encoder有效。现行固定FIT/CAL加反复单场景CAL与窗口相关性使新的架构选择可辨识性不足。按人类stop-on-obstruction暂止自动新训练，等待确认是否允许 **现有FIT内部按行人/场景分组的适配训练验证与匹配随机窗口对照**；原CAL保留，原decoder/scorer及F16文件冻结，不新质量loss/事件监督，不test/fold/外部data。该协议必须先精确设计、组身份/重叠审计和样本预算/原门槛，再source、actualGPU、bothreal2epochallRNGresume、资源实测ETA/soleOS/ONEhalf流程，未启动。
这不是泛泛重新索要持续许可，而是请求改变固定训练/验证协议的一项具体决定。若人类不批准，保持当前结果与baseline，不无限局部变体/诊断或换名绕范围。新编码器/原decoder/scorer解冻、assignment/loss监督及数据扩展另需具体权限，不自动做。
AI辅助披露：本报告的归纳/反方由同一助手完成；所有未重放项明确引用已有生产审核，不冒称外部复现。
