# 同预算单候选替换诊断

2026-10-07 evidence-driven adaptive exploratory follow-up. NONFORMAL冻结保存数据分析，无模型forward/optimizer/新训练/assignment/loss/scorer/新数据，不增加正式计数。现有正确冻结重放9组（三内部grouped协议×42/43/44），所有sample IDs和模型来源不变。不是独立confirmatory测试。

唯一primary因素：K3 native原logit前三条 vs 保留原生前两条、以已有Learned-3（full24 slot21）替换第三条。副对照用同位置换static(slot20)、Learned-0(slot22)、Learned-1(slot23)，仅预注册描述，不用outer选择最佳部署策略。不同seed的native输出重复不是独立native模型。

所有窗口和stable/transition/change报告同一完整candidate ADE≤.2 AND FDE≤.4覆盖、rescue/lost相对native3、minADE/minFDE/J（J同candidate max(ADE/.2,FDE/.4)）、GTfree complete-link .10meanpath/.20endpoint有效好簇和distinct、五pooled运动学 dt.4，对native20全部1%原保护不放宽。Clustering不是maximize。另记录learned是否救回native20缺失，不复制候选凑数量。

每protocol primary配对差分做target-track+model-seed cluster bootstrap2000，固定seed20261007；track=verified(sequence,pedestrian)，按抽中track multiplicity计算窗口均值，再3seed抽样。包含同窗口seed关联，不把9job当独立数据集。CI探索，scene3recordings不构造可靠独立sceneCI；不同协议不强行合并配对。非major升级，不改baseline。

全部输入/模型原源来自上轮校正savedNPZ及原.metadata，顺序SHA锁定。Math fixture:同path合格、rescue-lost恒等式、20duplicates1cluster、bootstrap同向/零向。实际128rows原质量/聚簇/运动学和IDs审核、资源guard PASS后唯一零trigger OS任务、真实启动确认、ETA、ONE50%TIME→completion；完成独立保存重算与相关SHA初末核验。~17GBhostavailable/process6GiB/physical+commit各reserve6GiB，不OS/pagefile/userkill。新output仅本目录，所有旧source/model/result不改。

如果primary稳定增覆盖、motion及原收益保护均满足，才将组合映射列入下一项合法干预；若lost抵消rescue，停止外层固定替换扫参，回到新head补充质量与泛化诊断，不盲scorer或第四slot/系数变体。科学/source/ID/resource/权限实质阻碍STOP保留证据等human。
