# 正确概率接口冻结重放

唯一恢复任务14892：13:53:54.782041→13:54:28.021742HK，双SUCCESS9jobs。13:56:11唯一迟到half检查发现完整完成，不冒称13:54:55按时检查。全相关source/input/checkpoint及父artifact SHA不变，所有frozen weights/buffers EXACT；独立saved-only NumPy/SciPy完整聚簇/运动学/标量重算max8.881784e-16，无optimizer、训练或第二次模型前向。NONFORMAL不增正式计数。

恢复B重现原M1：utility NLL .92842465571 vs原 .92842464977；Brier .23047900127 vs .23047900365；ECE .24394976791 vs .24394976272（误差均<1e-8）。原M2同K指标也重现。恢复依据是原源码CUDAsoftmax，不调整科学阈值。

## 结论

- 同K3 native3整体42.062%、变化20.671%；推荐C1 34.953%/19.697%。全24相对native20变化41.531→44.438%有补充，但是K不同。不能将“多来源”自动当更高质量组合。
- B overall utility NLL .9284 < uniform log3 1.0986、utility平方误差.2305<真实uniform基线.34345，说明存在质量排名信息；但变化组NLL1.22277 > uniform1.09861，平方误差.35604 > uniform.29333，是关键反例。
- B selected-good ECE整体24.395pp、变化35.693pp。Softmax(-2J+3good)效用与真实未来概率/二元合格事件不同，不能称真实发生概率已校准。
- B概率加权ADE .47769m vsnative1 .41921m；不是消除ADE风险。A也.48263m；C仅输入压力测试，未训练适配，不是修复的新校准模型。
- B retained78.09%，保留成功39.753%，同保留native1 37.193%、native3 48.176%；all-window成功31.311%，期望输出1.832。A retained89.52%/success36.334%、同保留native3 44.537%；outputs2.148。比较native3涉及更多输出预算，需下一项固定K，不能当完全等成本风险比较。拒绝者代价未定义，不支持机器人安全或可靠部署。
- 7组合×9jobs×4groups全运动学核验有21条未通过native20五项1%门槛，包括重复seed的相同native3组合；不是21个独立失败，也不证明新增路径变得不物理。

## 分支和下一步

当前更直接的问题是固定预算的组合取舍及概率接口/目标解释，而非已证明生成完全无用或past-only不可预测。先固定输出K3，保留native前两条，只以Learned-3替换第三条；primary与native3同ID配对。副对照其它既有slot仅描述，不根据outer结果选择“最佳配置”。全组救回/丢失、J和5motion保持标准，cluster bootstrap仅探索。此后有稳定净覆盖才考虑新评分研究；需改训练目标/模块时精确核验授权，不盲改loss。

反方2（同助手、非外审）：ECE混合分布与soft-target错配不能直接证明模型毫无信息；overall utility优于uniform，learned追加确有救回。先剥离解释错误与组合错误。

反方3（同助手、非外审）：训练统计恢复不能自动使CAL/场景迁移成功；仅3seed、重复外层、F16已见原pool、可用样例高度相关。需要独立验证才可称可靠泛化；不宣称SOTA、意图识别或不可预测率。
