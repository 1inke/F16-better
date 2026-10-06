# Candidate-state研究：三项强制总复盘
2026-10-05；3/3正式实验均正常完成、审计PASS、major FAIL。18新80epoch训练54policyentries不是54独立训练。所有54模型SHA、18完整history（1440epoch行）、72aggregate及288探索CI、ALLpolicygroup质量与pooled运动学已核对保存在completion_review/THREE_STUDY_REVIEW_DATA.json。本轮18预测独立重构，前两轮使用既存独立重构审计并重新SHA核验，不冒称全部旧模型再次重放。

## 三个假设与控制
| 实验 | 唯一注册因素 | native严格覆盖733对治疗3seed | change ADE/FDE native收益 | 主要反证 |
|---|---|---|---|---|
| CandidateState | 标准化state32乘0 vs实际native预测state32 |744/742/749|1.373/1.369%|CI跨0、seed3 FDE害、accelmean+5.23–10.66%|
| ObservedDiscrepancy | BOTHactive rawstate vs raw-last观测velocityreference30，共同rawscalers |725/730/728|.160/.227%|零change救回、CAL早期微小修正、final害、无可靠增益|
| DiscrepancyNormalization | BOTH同centered32，rawFITscaler vs自身centeredFITscaler前30 |726/733/731|.089/.209%|CI跨0、无native净覆盖、seed3全组accelmean>1%|

第二freshraw控制9policy EXACT第一治疗；第三fresh控制9policy EXACT第二治疗，独立执行复现可靠，但不能扩充独立泛化样本。第一actual和masked名义参数同，不等有效可用容量；第二条件平移及第三仿射保留给定past信息，但有限优化/init/weightdecay几何改变，非纯因果容量证明。

## 覆盖、数量、选择
Samecompletecandidate ADE≤.2 AND FDE≤.4优先。第一净覆盖+11/+9/+16是探索性信号而非major；第二净-8/-3/-5，第三-7/0/-2，没有修复。
Effectivegood native929，三实验分别938/942/828、925/928/929、926/932/930。第一seed3局部多簇变1但1314窗口仍20distinct，不能一概叫全体collapse；第三有效good略多不抵消覆盖下降。不得复制或最大化计数。
Inheritednativechoice native281，第一285/283/289、第二281/281/281、第三281/282/282。旧scorer未重算新候选概率，不能称新头保守选择。生成coverage未稳定，scorer排后。
Clear469与change223不同分母；原全1516窗口同seed重复不是独立行人。松.3/.5只敏感性，没因失败切标准。

## 所有policygroups与CI、运动学、过拟合
完整每项24ALLpolicygroups/96CI及6history见配套JSON，不挑constrained之外DEV最优policy。三项primary均不满足5%、nativefreshjointCI；原protected≤1%均值通过不等完整成功。
第一平均加速度害5.23–10.66%，seed3speed/jerk也失败；第二全motion1%PASS主要与选中早期小修正一致，不等理解提升；第三seed3accelmean四组1.269–1.675%又失败。不能以windowp95平均或硬大阈值掩盖pooled全候选运动学。
第一FIT总loss下降13.2–13.9%、CALfinalchange恶化、选29/25/35；第二12.8–13.1%、同样恶化、选5/6/7；第三13.0–14.5%、同样恶化、选5/2/13。FITloss和CALchange ADE/FDE指标/群组不同，尚缺同组同指标FIT覆盖证据，不能宣布纯overfit或loss根因。Legacy/final没有可靠全项替代，不能延长epoch或改guard。

## 结合此前失败后的方向限制
GT辅助表达可达/原loss奖励与实际共享past→future学习不同；hardwinner直接输出质量反馈稀疏，但共享参数/正则仍作用，历史softset/energy/mixture也失败。自由增量、timequery/velocitybasis、social/mode/conditioning失败，以及CAL单university、3scene/window相关/nativeF16seen/三seed/reusedDEV都是反证。未确证信息不可预测/decoder唯一根因/心理意图/SOTA。
三项否定了继续局部state-centering/normalization优化作为当前主线，停止第四局部变体。暂定优先级2先执行精确同组同指标FIT/CAL学习-迁移分解：固定已有模型，不新增容量/损失/监督/评分或解冻，确认是训练内完整路径也没学好、还是训练内学得好而CAL不迁移。结果后才设计可辨识生成对照。

## 反方检查与结论
同助手inline反方2：第一小幅覆盖增加值得保留但跨seed/motion不足，后两项没有恢复；全部控制重复与CI局限公开，不能包装成功。
同助手inline反方3：下一诊断不是因果鉴定、更不是修复，只补齐当前最关键同指标缺口；不得按DEV或CAL选择新threshold、checkpoint/architecture。非外部审稿。
3/3完整收束；不重置历史cycle11/假goalcomplete。模型baseline仍F16，未达major。下一新folder冻结诊断仅DESIGN/IMPLEMENTATION_PENDING，当前无生产worker；源/ID/资源/方法或权限实质阻碍STOP等人类。
