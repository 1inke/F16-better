# E3生产实现v2：审计器非叶张量复制修复

# E3生产实现v1：冻结原科学协议，补齐生命周期证据

人类2026-10-09“现在就开始继续实施”。承接父DESIGN.md五fold×三seed×full/last2/noNbr=45 fresh H3_30→V2_20 pipeline。该子目录不改父已审核源码、预检、模型或结果；父426个原始来源digest原样继承。新runner仍调用原科学run AST，损失/optimizer/scheduler/EMA/augmentation/validation选择不改。

唯一研究因素raw输入信息可用性；父input transform定义原样。训练ID、scene-tail5%验证、初始化seed/各模型预算共同固定；V2只加载自己branch/fold/seed新H3，不旧warmstart。完整输入通路等价/隔离、GPU8FIT、六真实513FIT+12val两epoch全部状态恢复须对当前实现复核；既有父PASS不是当前源码自动PASS。

实现补充：资源记录改有界累积；生产gate绑定实际source/ID和本任务；在原验证选定模型后只做posthoc保存预测及完整质量/聚簇/运动学，保存数学重算不更新模型/优化器；模型全部weights/bool/statbuffers必须exactfixed。selected输出继承原评价，不以新test指标选checkpoint。

保存explicit raw index IDs/track与原E1固定mask、原GT/CV、world-relative K20 paths/logits、逐window ADE/FDE/J/coverage/effectivegood、所有组五pooled motion(dt.4)、top1/meanK残差skill和配对输出。原native F16既有模型/cache只SHA保护，不作为新模型参数来源；新full也从fresh H3开始。五fold所有test窗口保留；疑难unknown start保留并报数量。train/val窗口/target曝光报告，不冒称target/context-disjoint。

执行成功不是运动理解、majorPASS或基线升级。coverage同candidateADE<=.2且FDE<=.4优先，有效good不是maximize/copies；全部既有major及五motion1%门槛保持。三seed/旧test反复使用、场景少/窗口关联探索局限保留。ARS反方1同助手非外审：full/last2从freshH3差异含有限优化；noNbr含padding/图变化，不能把差异称唯一必要性或心理意图。替代解释在报告保留。

只有当前所有预检、独立保存算术、实际吞吐/RAM/disk、最终SHA全部PASS后唯一OS0trigger/byte-lock启动；ONEstartup/ONE预计50%→ONE完成定点。无epoch比例冒全部完成，无慢就重启/改阈值。过程6GiB/物理+commit各reserve6GiB，单model/batch64，资源有界记录，不OS/pagefile/用户kill。正式ETA按实测分阶段吞吐+保存审核/SHA计算，节点不是生产ETA。



v1真实预检未完成，因before_batch直接deepcopy含上一batch计算图缓存的模型失败；全部v1源码/日志/partialoutputs保留。v2只改变审计副本创建：初始化干净模板，然后逐batch加载原model state_dict并复制训练模式；原trainforward/backward/loss/optimizer语句完全不改。必须重新跑当前六组实际恢复/完整grad等价，不能继承旧PASS。

