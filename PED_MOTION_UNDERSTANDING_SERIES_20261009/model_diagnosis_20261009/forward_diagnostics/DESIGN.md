# D1b / D5b 冻结GPU前向诊断

五fold×既有3seed，沿用旧stage2固定hash selected IDs，ETH364其余最多1024，不按GT挑例。只有既有eval/no_grad，所有参数/布尔/statbuffers逐模型beforeafter torch.equal。暂停E3和automation不恢复。基线真实GPU128batch与原GPU保存outputs paths/logits gate2e-5，先基线核验，再当前模型所有干预；失败立即STOP保留来源，不容差放宽或CPU绕过E8STOP。

D1b：原smoothed3step norm>.03heading vs阈值.01/.06，整7step位移方向>.03，回溯最近单step>.03；无可靠方向仍fixedworldxaxis。原fallback窗口固定0/90/180/270度四orientation，非fallback保持原heading。不是四次随机抽样，明确是固定角度敏感性/GT-free top1 path均值TTA。每种朝向变体分别全scene旋转60度/镜像worldx后inverse-world坐标，报告top1/meanK skill/同ID误差与same-slot/Hungarianbank变换差、mean/P95/尾部，不能仅中位数宣布等变。训练朝向没变，所有替代可OOD，无修复收益因果断言。

D5b：邻居所有观测点镜像于经过target末点沿target最后一步速度方向的轴；target完全不动。axis norm<.02低速无法定义，单独保留并标undefined不纳主要A。相对CV速度模/当前距离/未来12step最近距离保持2e-5；这利用轴与target最后速度共线，若用三step平滑朝向则转弯窗口未必保持威胁，不能照搬附件“总是更干净”的断言。same-ID noNbr基线前向重算。l/lprime是with/without endpoint在同原normal方向的投影；报告A=sum(l+lprime)/sum(abs l+abs lprime)、meanabsolute响应、windowabsolute antisymmetry failure比，避免A=0来自零响应或窗口抵消；A<0不是自动实现错误。all/P1/C3分层，track2000CI。

只实现短诊断，不进行任何训练/新数据/scorer更新。每次model构建只读既有H3/V2checkpoint、sourceSHA与缓存SHA；newoutputs全本目录。ALL五motion1%及主5%收益标准保留，任何干预 point/AUC改善不能升级baseline。保存每batch全部candidate outputs、原raw几何fixture实际closest、frozen模型审计与资源。完成CPU独立保存算术（不是所有模型二次前向/外审）。process6GiB与physical/commit每项reserve6GiB，360秒上限。

反方1：低速orientation依赖存在不等于UNIV整体根因；D1a已显示贡献需按人口分解。侧向反对称是一项行为诊断，不是GT因果避让；TTA平均不在原K20中，需另审运动学方可谈部署。后续T3长训练保持跳过。
