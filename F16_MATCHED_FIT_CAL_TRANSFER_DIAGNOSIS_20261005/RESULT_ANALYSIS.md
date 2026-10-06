# 同组同指标 FIT/CAL 迁移诊断：结果与下一步

本轮已完成并复核。原模型、三个输入表示族的18个已有 checkpoint 全部冻结，没有新训练、没有修改损失或评分器，也没有读取 DEV/test。优先级2从“FIT总损失下降、CAL误差增大”的间接比较，推进到同组、同质量定义的直接比较。

## 最重要的结果

高质量仍指同一完整候选同时 ADE≤0.2m、FDE≤0.4m，不能拼接 minADE/minFDE 或早中晚不同槽位。以下三行都是第80轮 final 模型，逐seed展示，FIT变化组分母6410，CAL变化组分母700：

| 表示族 | FIT有完整好候选：native2603→ | CAL有完整好候选：native243→ |
|---|---|---|
| 原始预测状态 raw | 3345 / 3336 / 3398 | 230 / 216 / 223 |
| 观测速度中心化、原raw尺度 | 3301 / 3346 / 3306 | 232 / 240 / 238 |
| 同样中心化、自身FIT尺度 | 3453 / 3335 / 3407 | 242 / 235 / 225 |

九个最终模型都在FIT变化组净增加698–850个覆盖窗口：40.61%→51.50–53.87%。CAL变化组却全部净损失1–27个窗口：34.71%→30.86–34.57%。这些是相同样本在不同seed下的重复评估，不能加成独立人数。

FIT变化组minADE改善12.12–13.84%、minFDE改善20.84–23.78%；CAL变化组minADE反而恶化3.51–5.91%、minFDE恶化2.53–6.61%。所以问题不只是“总loss下降但指标没下降”：训练内确实生成了更多完整好路径，改善没有迁移到CAL变化组。

## 但不能说FIT已全部学对

九个final模型在FIT和CAL的原全组运动学保护均不通过。变化组全K20 pooled平均加速度相对native：FIT恶化15.09–17.89%，CAL恶化14.06–16.27%。这是候选集合聚合，不是每条获救好候选都违反运动学，也不等同碰撞/硬运动失败。它仍然阻止升级baseline，严格1%保护不放宽。

因此精确分类是：**FIT位置覆盖改善，但伴随运动学副作用；CAL变化组覆盖不迁移。** 不是“运动学合格的FIT学习完全成功、只是纯泛化失败”。

早期constrained模型大多更接近native：两种centered族在FIT/CAL均通过运动学保护，但变化组覆盖只在native附近。raw constrained在CAL变化组242/253/246，存在两个seed的小增益，却运动学不通过。不能将final训练收益与constrained的保护混成一个成功模型。

## 与人眼观察的关系

同一个 normalized-adequacy-closest 完整槽位的早/中/晚误差（族内三seed均值，m）：

| 数据/模型 | 早 | 中 | 晚 |
|---|---:|---:|---:|
| FIT native变化组 | .11755 | .25786 | .36380 |
| FIT raw final | .11489 | .23736 | .28732 |
| FIT centered/raw final | .11464 | .23828 | .29217 |
| FIT centered/own final | .11405 | .23707 | .28771 |
| CAL native变化组 | .12463 | .26948 | .37393 |
| CAL raw final | .12658 | .28100 | .39455 |
| CAL centered/raw final | .12701 | .28119 | .39090 |
| CAL centered/own final | .12578 | .28089 | .39717 |

训练内收益主要体现在中后段，而CAL后段反而更差。这支持继续研究响应映射如何迁移，不证明终点解码是唯一原因，也不是对真实心理意图的证明。旧native argmax仍是继承槽位，不是新head重新判断的概率或“保守选择”。

原始好候选数与GT无关有效好簇只是次级指标。例如centered/own final CAL变化组seed1有效好簇303→316，同时覆盖243→242；局部数量增加不能抵消更少窗口拥有至少一条好路径。全K20 complete-link阈值仍为meanpath0.10m/endpoint0.20m。

## 反例与解释边界

不是所有CAL组都恶化。centered/raw final整体覆盖1490→1501/1506/1506，稳定组994→999/999/999、过渡组253→270/267/269；但变化组243→232/240/238。整体平均/计数可能掩盖本阶段目标组的失败。

本结果支持“学到的修正在FIT更有效、CAL变化组缺少稳定迁移”的定位假设；不能独立证明纯过拟合、损失错误、容量不足或decoder唯一根因。CAL只有university且参与checkpoint选择；FIT/CAL窗口与轨迹相关，原F16已见过训练资料，三seed不等三个独立数据集。没有可靠跨场景group CI或SOTA声明。

## 反方检查2：分析后（同一助手内联，非外部独立审稿）

- Major：若只报FIT位置改善，就会漏掉FIT自身的运动学退化。已改成“有副作用的位置学习+迁移缺口”，不宣布完全学对。
- Major：整体CAL存在小覆盖收益，反驳“所有CAL都坏”。已分组列出上述反例，不将overall替代change。
- Minor：有效好簇可能随覆盖下降而增加。坚持覆盖优先、数量其次、scorer最后，禁止复制候选凑数量。
- 判定：PASS_WITH_LIMITATIONS；只允许上述描述性定位，不允许因果根因结论。

## 下一步：参数层质量反馈的跨split一致性

停止第四个标准化/epoch/宽度变体。不直接换loss或训练scorer。下一冻结诊断检查18个当前checkpoint中：在newhead参数空间，FIT原primary质量项的下降方向，对CAL同组primary质量项是一致改善、方向冲突，还是梯度近零而无法判定。

这不是再次只看输出坐标梯度：会经过真实newhead Jacobian，涉及state encoder、Q/K/V和响应head。原F16/wholedecoder/scorer保持冻结；所有权重/buffer值不更新，无optimizer。用原ADE+.5FDE hard winner和原全FIT class weights，不引入新的监督或新损失训练。CAL只事后诊断，不选超参/checkpoint。

下一实施/预检规划节点2026-10-05 15:25HK，不是生产ETA，也没有宣称新worker已启动。详见 [下一诊断设计](../F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/DESIGN.md)。实际预检通过后才按吞吐、源/旧SHA与IO估ETA，唯一OSworker启动、预计时间一半一次检查，再更新完成节点并退出静默。实质科学/源/ID/资源或权限障碍即停止等人类。

## 审计与资源

- worker和exit双SUCCESS，真实13:14:22.396883→13:20:49.155371HK，6m26.76s。
- 唯一半程检查13:21:27.3409532发现完整完成；不再查询完成PID31124/任务。
- 38job、76输出SHA、28791显式unique ID、18既有模型权重buffer冻结exact，433source与29871old SHA前后不变。
- CPU补充完整重算所有保存质量列、GT-free完整聚簇、运动学列和全部组aggregate，最大误差0；18模型/76输出/433source/29871old再次前后SHA通过。这是保存坐标算术复核，不是再次重放全部模型或外部审稿。
- 同epoch CAL历史min指标最大差6.24758e-8，低于原2e-5容差。
- 生产peakRSS1.984/private2.660GiB，系统physical/commit最低14.030/13.152GiB；CPU补充private1.909GiB。持续约17GB可用hostRAM限制，process6GiB、系统两项各至少6GiB余量。
- 非formal诊断，candidate-state仍3/3已总review，历史cycle11不重置、goal未完成、F16 baseline不变。
- 科研设计技能用于区分证据/推断/建议并做反方检查；未进行外部文献调查或独立多agent审稿。

## 反方检查3：最终结论与下一方案（同助手内联）

- Major：九个final重复同一数据与相关架构，不能把一致方向当九个独立泛化实验；不合并样本人数、不报告新的显著性或跨scene因果结果。
- Major：下一参数诊断只能检验原primary的局部方向，不是原total加AdamW的真实训练更新；即使得到梯度冲突，也不能自动宣布结构/损失根因。已在新DESIGN中限制结论与后续动作。
- Minor：当前没有新worker；15:25是实现预检规划节点，生产ETA和半程节点必须实际预检后另定。报告不宣称已经启动或预检通过。
- 最强反论点：FIT增益可能是数据分布与局部优化共同造成，并不需要唯一decoder缺陷解释。当前证据不足以排除，故保留为定位假设。
- 判定：PASS_WITH_LIMITATIONS。完成旧诊断与制定下一可回答的问题，不把结果升级为baseline/major或扩大权限。

依据：[RESULT](RESULT.json)、[全组保存算术复核](completion_review/ALL_GROUPS_VERIFIED.json)、[补充审计](completion_review/REVIEW.json)、[冻结审计](FROZEN_REPLAY_AUDIT.json)、[资源](RESOURCE_USAGE.json)。
