# 训练内端点保持投影：完整结果
日期：2026-10-06（香港时间）。使用 academic-research-suite 进行证据、推断与反方分离；同一助手内联审核，不是外部独立审稿。证据均来自本目录 results/summary.json、results/quality/summary.json、results/physical_population/summary.json、results/independent_audit.json及 completion_review/REVIEW.json。

## 结论
实验审核完整通过，但方法 major 未达标，F16 baseline 不变。投影进入训练后，正式 constrained 的三seed、四组、五项 pooled 运动学全部守住 native+1%预算（12/12 seed-group）。但完整覆盖只是小幅变化，变化组 minADE/FDE没有优于native，FIT收益也仍未稳定迁移到CAL。不能进入“loss/credit/motion已经全部对齐，只剩结构”的终端分支；目前更符合“物理副作用部分控制后，生成端学习/泛化仍不稳定”的分支，scorer仍排后。

## 执行与可靠性
原 worker31532完成六fresh80epoch训练后，在02:56:56评估读回阶段失败：dataset-only读取器被误用于自己的18预测文件。原代码、stderr、worker_failure、模型、history/resume及原预测/统计均保留，未重训。
新隔离 evaluation-only worker11344/service2728，03:22:48.132944→03:25:46.554410双SUCCESS，实际178.421466秒。唯一50%时间检查03:29:33.2796765（计划03:28:48，迟到约45秒）发现已完整完成；没有等待旧03:34截止或重复startup/半程/PID查询。
18模型/18显式1516uniqueID finite20×12×2、六80history、全部sample/nativeALLgroup/perseed/bootstrapCI、完整GT-free聚簇与pooled运动学、temporal/projection严格物理算术、36FITCAL原total/P及80epoch trace、18权重bool/scalerbuffer精确固定、541原source+11恢复source/31968old/69父已完成artifact初末SHA通过。恢复不增加formal，第三此时才完整完成，physical-feedback研究3/3。
补充CPU NumPy review：18min指标误差0、36population原total/P误差0、全部quality组计数与六80trace、552sources/31968old/18models/69parentartifact前后SHA PASS，无torch/model forward/optimizer；全部聚簇与CI由完成worker重算，补充没有第二次全聚簇、所有梯度或模型重放。
评估worker RSS1.8604/private2.5764GiB，最低physical17.5375/commit16.7513GiB；CPU补充private.0688GiB。原训练生命周期峰值未在失败时持久保存，不能用预检/评估峰值代替。6GiB进程+BOTH系统6GiB余量和长期约17GB可用限制不变。

## 单因素与解释边界
BOTH91596 uniform双raw-state persistent专家，原FIT25912/CAL2879、rawFITscalers、原total+.1P、hard winner ADE+.5FDE、原所有正则与CALguard/LR/batch/order/80epoch/3paired seed/K20/nativeclip固定。只改是否将已批准端点保持加速度投影放进训练前向/反向；双方路径累加float64，head/base float32。全旧F16/wholedecoder/scorer固定，未来GT/groups/logits不预测forward。
相同codes端点保持不等不同最终模型终点相同；可行域与优化几何耦合，不是纯结构因果。第三control数值累加不同于此前float32uniform，不预设bitwise重复。当前checkpoint小量收敛变化不证明心理意图理解。

## 正式 constrained，固定DEV描述
同一条完整candidate ADE<=.2 AND FDE<=.4，不能拼不同slot或时间段。

| 指标 | F16 | 无投影 seed1/2/3 | 训练内投影 seed1/2/3 |
|---|---|---|---|
| overall good /1516 | 733 | 735/724/731 | 736/735/735 |
| change good /223 | 43 | 46/47/47 | 47/48/42 |
| overall有效good簇总数 | 929 | 947/816/930 | 939/827/829 |
| 旧argmax继承slot good | 281 | 285/281/280 | 281/281/288 |

投影overall rescue/lost为18/15、27/25、24/22；change为10/6、12/7、6/7。同一批IDs跨seed重复，不能加成独立人。有效good不是越多越好，下降也不自动等整体mode collapse；近静止聚簇需具体检查。旧argmax只是继承slot，不是新路径概率/scorer学习或意图判断。

变化组平均minADE/FDE：native .372174740/.696326435；无投影 .376833301/.721487771；投影 .372979381/.707470318。投影比native恶化.2162%/1.6004%，相对较差control均值改善并不等成功。native与freshcontrol的变化ADE/FDE/joint探索CI全部跨0。只有top3对control的区间为负，不能替代完整候选门槛。
protected transition meanADE+1.0913%违反1%保护；选定CALfeasible不能保证DEV保护。5%、各seed一致性、CI、protected等major失败。
投影constrained12/12 seed-group五项motion PASS：最大speedp95害.9825%、accmean.7991%、accp95.04385%、jerkmean.05308%、jerkp95.02601%。这是实质约束进展，不等泛化/完整覆盖达标。

## 同组 FIT→CAL，final不是部署结果
change FIT6410 native2603/6410=40.61%，CAL700 native243/700=34.71%。

| 时点 | 无投影FIT | 投影FIT | 无投影CAL | 投影CAL |
|---|---|---|---|---|
| constrained seed1/2/3 | 2890/2766/2651 | 2730/2738/2727 | 234/246/243 | 248/252/248 |
| final seed1/2/3 | 3547/3530/3541 | 3621/3534/3560 | 225/232/236 | 226/232/222 |

投影final FIT55.13–56.49%高于native40.61%；CAL31.71–33.14%低于native34.71%。不能说“约束使FIT完全学不到好路径”，也不能说“CAL所有时点都差”：constrained CAL248/252/248明显是反例，且这些checkpoint已由CAL选择，不能当未见泛化。
投影final change accmean FIT+.818–.835%、CAL+.815–.833%全部在1%内，population P为0。P=0表示该约束不再超额，不是没有训练作用。36输出原total/P及三组定义不变。
投影final DEV五motion只有9/12组PASS，三个change的speedp95仍害1.439/2.842/1.645%。端点保持accmean约束不控制所有速度分布，不得把accmean通过等同allmotion通过。
final投影全FIT人口原total .297547/.301405/.301136，CAL .427077/.426787/.427690；这些是全split分母，不是change-only loss/实际mixedbatch AdamW方向，不能混用或跨实验容量直接排行。

## 反方2：分析审查（同助手）
- 最强反驳：大部分改善可能只是受约束的温和修正，正式DEV只净增2–3个完整覆盖窗口，并未展示稳健未来响应。CALconstrained收益保留，不能为维持负面结论删去。
- loss已有推动FIT完整覆盖的证据、已有good通常获winner credit，不等当前所有参数梯度/有限优化/泛化都对齐。旧Gaussian GTwitness不是persistent可学性，0直接坐标credit不是共享head无反馈。
- constrained运动学通过与final速度失败必须分开；旧constrained局部梯度冲突不能照搬成新投影head已经测过的梯度结果。
- 三scene/窗口与轨迹相关、F16已见、CAL单university且反复选checkpoint、重复DEV/仅3seed，CI探索而非可靠独立泛化证据。
判定：PASS报告的有限结论，FAIL major与baseline升级。

## 反方3：路线审查与下一步
本小阶段3/3已完成，完整三项总review见 PHYSICAL_FEEDBACK_STUDY_REVIEW.md及JSON/三sidecars。停止第四projection系数、routing分位数、width/LR/epoch/profile局部变体，也不直接进入scorer或softmin。
下一大方向应继续验证共享映射的迁移，而不是凭本次结果断言encoder、loss或decoder唯一根因。现有反复单场景CAL+相关窗口无法清楚区分轨迹记忆、场景分布差异与结构瓶颈；按人的“遇到实质研究阻碍中断计划”要求，下一训练暂不启动。
具体建议等待批准：只使用既有FIT样本池，对newhead建立按行人/场景分组的内部验证与匹配随机窗口对照，原CAL保留、原F16文件及decoder/scorer冻结、不新loss/事件监督、不读heldtest/其他fold/外部数据。此项改变当前固定训练/验证协议，需要人类明确确认后精确预注册规模、分组身份、样本预算、scalers/权重和唯一因素，不能自动称已启动或仅给它换名字。
这是下一协议/可辨识性决策gate，不是任务完成、人工暂停或训练故障；长期goal尚未达成。原研究成功门槛全部不降低。
