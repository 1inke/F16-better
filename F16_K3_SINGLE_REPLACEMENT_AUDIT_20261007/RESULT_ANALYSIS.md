# 固定K3单候选替换结果

唯一后台42536于14:01:26.394452→14:01:36.148948HK双SUCCESS，实际9.75秒。预估1–3分钟保守；唯一half实际14:03:16.616689迟到发现完整完成，未重查startup/worker。9saved jobs/180group records/72探索primary CIs，全部相关input/source SHA初末不变；独立NumPy/SciPy全质量/聚簇/运动学/rescue-lost/clusterCI maxerror3.552714e-15。No modelforward/optimizer/newtraining，正式计数和baseline不变。

Primary=保留native Top2，以L3换第3条，vs native Top3同ID、同K3：

| 协议 | 整体覆盖差 | 变化覆盖差 | 变化覆盖探索95%CI |
|---|---:|---:|---:|
| scene_family | -4.378pp | +2.054pp | [-1.105,+5.185]pp |
| target_track | -5.250pp | -0.264pp | [-4.202,+3.371]pp |
| recording_time | -1.885pp | 0.000pp | [-5.634,+5.517]pp |

等权protocol×seed macro整体42.062→38.225%、变化20.671→21.268%；macro不是9个独立数据集。每协议变化覆盖CI跨0，scene变化连续ADE/FDE/J改善、整体及stable明显受损，不能选它当普遍改进。停止固定替换slot扫参，不将描述性secondary outer最优升为部署配置。

Scene overall救7/15/9，丢78/78/80；change救6/8/7，丢2/2/4。Track overall救15/14/20，丢72/70/70；change救9/8/11，丢12/9/9。Time overall救10/13/11，丢21/23/22；change救5/7/5，丢5/6/6。这是明确组间取舍，不是新候选全无用或无法预测。

反方2（同助手、非外审）：scene变化连续指标确有改善，native20缺失的窗口仍有新候选救回，说明补充价值不是零；但无稳定净覆盖或protected整体收益，不能优先升级固定替换。

反方3（同助手、非外审）：重复outer和3seed限制仍在；native3对native20 speedp95比较改变候选池组成，不说明某条native路径被修改。后续训练源码审查还发现跨协议objective/diversity/clip不一致，因此不同协议性能差异不能唯一归因于迁移。

下一步优先源协议统一对照，而非新增head/scorer或slot搜索。详见../EXPERIMENT_PROTOCOL_RECONCILIATION_20261007/BLOCKER_AND_RECOMMENDATION.md。源/科学可辨识gate等待具体人类决断，保留所有旧实验，不新训练。
