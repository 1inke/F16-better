# 观测运动偏差对照：结果分析（2026-10-05）

结论：**完整正常结束，但新表示没有改善本阶段的高质量候选覆盖，也未达到 major。** 相对最后观测速度的条件中心化，配合共同原始 FIT 标准化，保留下来的修正几乎回到原生 F16。运动学保护通过是正面结果，但不能当作运动变化理解得到提升。

## 对照与可靠性

zero 是 active 原始 native 候选 state32；mean_std 是 state32 前30维减最后观测速度的 reference。最后两项加速度幅值/首末预测 heading 不变。两者使用共同 raw FIT scalers、同45798参数头、同 sigmoid persistent profile。全旧 F16/whole original decoder/scorer 固定，原 K20、loss/guards/FIT25912/CAL2879/LR与80epoch预算不变。没有新未来观测/GT forward/意图监督或重新评分。

独立 OS worker 与 results exit 双 SUCCESS，2026-10-05 10:37:04.366423 HK结束，实际28分18.053秒。6次全新80epoch训练、18policy输出（不是18次独立训练）。生产完整核验18SHA/1516显式ID有限20×12×2/全部sample与native ALLgroup/perseed/bootstrapCI、6history、temporal/pulse、冻结、source382和28514旧文件SHA。补充 completion_review 独立从保存float64路径重构18个质量表和complete-link簇、minADE/FDE最大误差3.4368184e-7≤1e-6，18model/382source/28514old SHA前后PASS。保存算术重构不是所有模型重放，也不是外部独立评审。

新 raw 控制9个policy路径与上一实际 state arm **逐元素 bitwise EXACT**；这是新执行重复，不能累计为独立泛化证据。

## 正式 constrained 结果

change组为223个窗口；clear-event组另为469个，不能混分母。下表ADE/FDE分别取各自最优候选，不能代替“同一个完整候选合格”。

| 方法 | change minADE | change minFDE | 相对F16收益ADE/FDE |
|---|---:|---:|---:|
| F16 native | .372174740 | .696326435 | — |
| fresh active raw control | .367063254 | .686794281 | 1.373% / 1.369% |
| observed-centered treatment | .371579647 | .694747508 | .160% / .227% |

治疗相对fresh raw平均差为ADE +.004516406、FDE +.007953181m。native配对bootstrap ADE CI[-.001368132,-.000146698]、FDE CI[-.003962828,-.000095869]为负，但收益极小；native joint CI[-.002814385,.000016941]跨0。fresh raw ADE CI[-.004935330,.014618345]、FDE CI[-.019251808,.037004167]、joint CI[-.012197316,.032675770]均跨0。不能宣称治疗可靠优于raw，也不能以探索性小CI代替5%和同路径标准。重复开发集/窗口相关未解决，不是确认性泛化CI。

## 同完整候选覆盖：真正阶段优先项

阈值固定为至少一个同候选 ADE≤.2m AND FDE≤.4m。

| 1516个窗口 | F16 | seed1 | seed2 | seed3 |
|---|---:|---:|---:|---:|
| raw控制覆盖 | 733 | 744 | 742 | 749 |
| centered治疗覆盖 | 733 | 725 | 730 | 728 |
| 治疗救回原失败 | — | 2 | 3 | 3 |
| 治疗损失原成功 | — | 10 | 6 | 8 |
| 治疗净变化 | — | -8 | -3 | -5 |

覆盖率F16 48.351%；治疗47.823% /48.153% /48.021%。相对fresh raw，治疗救回16/12/23、损失35/24/44，净减19/12/21。同一1516 ID重复3seed，不能相加当4548个独立行人。

clear-event469原92个有好候选；治疗88/92/86，救0/1/0、损4/1/6。change223原43个；治疗42/43/41，零救回、损1/0/2。因此小幅min指标收益没有转化为困难变化窗口跨过完整质量阈值。

## 有效好候选与选择

GT无关全K20几何 complete-link(.10meanpath/.20endpoint)簇里含至少一个好候选的有效总量：F16 929，治疗925/928/929；raw938/942/828。原始好候选总量F16 1146，治疗1142/1145/1146。seed3治疗有效总量比raw多101，但覆盖少21：**更多有效数量不等更多窗口被救回**，不应最大化此数或一概判collapse。

继承原native argmax槽位的合格窗口F16 281，治疗仍281/281/281；raw285/283/289。评分器未改，这不是新路径概率、部署新选择器或运动意图判断。

## 为什么运动学通过，却没有解决问题

治疗所有seed/group pooled ALLK20 speedp95、accelmean/p95、jerkmean/p95均满足native≤1%harm。平均加速度增加约.289%–.635%，显著小于raw5.23%–10.66%的副作用。原hardmotion/coverage/duplicate/protectedmean守门也通过，但same-path joint guard失败；5%与native/fresh联合CI失败，最终major仍FAIL。

CAL选中的治疗epoch是5/6/7，raw为29/25/35。治疗选中pulse幅度仅.000679–.000873（是保存profile幅度量，不能直接当完整路径距离），center跨候选std .000710–.003193、span std .001440–.002030，非常小；未触发native fallback，但修正近乎native。较低副作用来自什么不能仅凭此因果下结论；“早期小修正被保留”是观察。

继续到epoch80，治疗FIT总trainloss下降12.780%–13.133%，但CAL change ADE/FDE相对首epoch均恶化。DEV final change .397637248/.772587955，相对native恶化6.842%/10.952%；stable final反而改善。说明存在分组泛化/目标权衡缺口，不是“再多训些epoch”即可解决。FIT总loss不是FIT minADE/FDE，不能混称轨迹收益。

## 输入尺度证据与下一步边界

只用FIT518240候选点的数值诊断显示：条件中心化后仍套raw scaler，前30纵向标准化均值约-1.315到-1.424，早期纵向std约.322，raw原按自身FIT归一化则接近0/1。这是**实际分布/优化几何变化**，不是数值非法，也不能证明它导致失败；last2统计接近0/1仍保留。

下一可辨识第三项可固定双方centered state与所有结构，仅比较共同raw scaler versus centered FIT自身mean/std（前30维，floor仍.001，最后2维不变）。这是固定FIT-only仿射预条件检验，不是温度/LR/宽度/epoch盲扫，不增加信息/意图监督。函数跨度在有bias与非零尺度时保留，但初始化、有限优化、weight decay几何改变，不能宣称纯信息或容量因果发现。该设计需要先写完整新folder设计、数值/GPU/两arm真实exactresume/RAM/ETA预检；本分析没有启动第三worker。第三正式完成后必须完整三项review，不无限第四局部标准化变体。

## 反方检查2（同助手inline，非外审）

PASS_WITH_LIMITATIONS。Major：若只报告native marginal CI与运动学PASS，会隐藏coverage退化和fresh对照缺乏收益，已同时披露。常数速度参考虽可逆，并不增加信息；优化几何变化不能证明decoder根因。训练与CAL的分组差异是描述，不以总FITloss充当同指标对照。没有Critical来源/ID/数值/资源阻碍。

## 反方检查3（同助手inline，非外审）

PASS_FOR_EXPLORATORY_REPORT。最强反论是“保护通过主要因为选中早期、几乎未改的模型”，与当前结果相符，不把它包装为有效运动理解。第三标准化只能测试工程预条件，不能挽救完整研究假设的确认性地位；即使改善也需全原major gates和相关性审核。3scene、重叠窗口、仅3seed、原F16已见与CAL单university、重复ETHDEV探索、旧GT-witness/freeincrements/timequery/velocitybasis/conditioning/social与tinyFIT→CAL反例仍适用。没有心理意图/不可预测率/SOTA/新baseline声明。

## 资源与文件

生产peakRSS2.579/privatecommit3.367GiB，系统physical至少15.077/commit至少13.167GiB。补充RSS.117/private1.541GiB，未改OS/pagefile或停止用户程序。所有新增解释、重构审核仅此新folder，旧科学文件保持SHA。分析由AI辅助完成，反方检查不是外部评审。
