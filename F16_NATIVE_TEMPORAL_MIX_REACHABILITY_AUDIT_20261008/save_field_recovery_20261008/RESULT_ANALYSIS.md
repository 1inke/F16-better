# 原生时间形状混合可达性审计：安全，但有效覆盖见证极少

## 结论
固定的原生轨迹混合确实能产生非零、超出全路径旋转/缩放的时间形状扰动，同时保住本轮原有覆盖和运动学。但主评估 outer 的变化组 **253 个窗口中，新增完整好候选为 0**。这不是可支持新一轮训练的强证据，也不是所有时间形状生成方案不可行的证明。

本轮是冻结保存输出的 NONFORMAL 诊断，无 torch、模型前向/反向、optimizer 或重新训练；正式研究仍 1/3，baseline、历史 cycle11、未完成目标均不变。

## 执行与证据核验
唯一生产 worker 与 exit 均 SUCCESS：10:47:18.504911 → 10:49:02.168278 HK，实际 **103.663367 秒**。唯一迟到半程检查在 10:52:36.7410594 完成（计划 10:49:49），当时已全部结束；没有另做 completion/PID 检查或重新启动。

60 科学 jobs = 20 预注册固定物理 partner × 3 split，120 输出；240 策略×组行。生产 ARITHMETIC_AUDIT/PRESERVATION PASS。补充 CPU 完整重读全部 60 保存 jobs，原 SciPy GT-free 全 K20 complete-link 聚簇、same-candidate 质量、五 pooled 运动学、救回/丢失 ID、几何/shape/alpha 全复算，最大误差 **0**；1480 相关源/输入与 120 输出 SHA 初末不变。不是全部历史 34k 文件核验、所有模型/梯度第二次回放或外部独立评审。[审核](completion_review/REVIEW.json)

补充脚本在所有审核及报告写完后，最后 console JSON 序列化 NumPy 数组失败，terminal exit1；原源码和故障说明保留，单独读回已完成报告及 SHA 的 reader 做了报告出口恢复，没有重跑科学计算或改标准。生产 worker 始终双 SUCCESS。[来源](completion_review/failures/stdout_numpy_serialization/README.md)

## 完整覆盖：固定 K20 与大 bank 必须分开
合格是同一个完整候选 ADE≤.2 且 FDE≤.4，即 J=min_k max(ADE_k/.2,FDE_k/.4)≤1；不拼接不同 slot 或不同时间段。

| Split/组 | 原生好窗口/分母 | 每个固定 K20 策略的好窗口范围 | 大 bank 并集好窗口 | 并集新救回 |
|---|---:|---:|---:|---:|
| outer overall | 625/1035 | 625–626 | 626 | 1，stable |
| outer stable | 434/604 | 434–435 | 435 | 1 |
| outer transition | 81/178 | 81 | 81 | 0 |
| outer change | 110/253 | 110 | 110 | **0** |
| inner overall | 932/1856 | 932–933 | 933 | 1，change |
| inner stable | 601/978 | 601 | 601 | 0 |
| inner transition | 165/381 | 165 | 165 | 0 |
| inner change | 166/497 | 166–167 | 167 | 1 |
| CAL overall | 1490/2879 | 1490–1492 | 1493 | 3 |
| CAL stable | 994/1535 | 994 | 994 | 0 |
| CAL transition | 253/644 | 253–255 | 255 | 2 |
| CAL change | 243/700 | 243–244 | 244 | 1 |

每个固定策略/组 lost 都是 0，这不是仅靠 bank 保留 native 得到的同义反复；但 bank 的不丢失本身确实由保留 native 保证。三 split 共 5770 保存样例 ID，仅 5 个不同窗口得到新成功，其中变化组仅 inner1、CAL1，outer0。不能把重复 partner 的 44 条成功候选记录当 44 个独立救回或 20 seeds。

大 bank 包含 native20 以及最多 76 条非恒等混合路径，不是等 K20 部署策略；没有按 GT 构造/选择 partner，没有把 posthoc 最佳伙伴做成预测器。CAL bank +3 与单固定策略最大 +2 不可混报。[全策略/组](completion_review/ALL_POLICY_GROUPS.json) [全部见证](completion_review/WITNESS_DETAILS.json)

## 连续误差没有实质突破
outer change 的原生 minADE=.249146619、minFDE=.429378298、J=1.390568401。20 个固定策略相对 native 的 ΔminADE 范围 [-.000186046,+.000081796] m，ΔminFDE [-.000230495,+.000407756] m，ΔJ [-.000671750,+.000825827]，有改善也有恶化；最好的单项改善幅度仍远小于 5%，且不能把不同策略单项最优组合成同一算法。

inner change 的 ΔminADE **全部为正** [.000012665,.000138633] m，虽然有一个阈值救回；CAL change 的 ΔminADE 有正有负，ΔminFDE 全部小幅负 [-.000393292,-.000036085] m。阈值偶发跨越不等于广泛误差改善，也不能将未独立抽样的固定策略范围当置信区间。没有新的三 seed 证据。

## 非零与时间形状证据
outer change 中，每固定策略的 4 个生成候选，max residual≥1mm 比例 **52.77–70.16%**，超出 span{anchor,Janchor} 的 shape RMS≥1mm 比例 **24.31–45.95%**。因此“投影把一切压成零”不成立；但 nonzero/shape novelty 也没有转成变化组覆盖。alpha=0 比例 **24.60–36.07%**（包含 self-partner 恒等项），中位数 .3222–1；不能只用一个零比例解释全部失败。

所有策略各组的 GT-free effective distinct 均与 native 相同；outer change effective-good 也全部相同 .505928854。空间/时间微扰并未带来该组新增有效好簇。1mm shape 阈值是预注册描述，不是 stop/turn 语义事件。

5 个救回的 native J 均接近门槛：1.01814、1.06024、1.01505、1.07169、1.08168；每个 ID 都至少存在一个 shape RMS≥1mm 的成功路径。它们支持有限可行性，不支持大幅修复恒速外推。CAL change ID29133 的生成 ADE .169597、FDE .395996、J .989989，相对 native J1.081678；仅 partner5 的该条记录成功，不可据此挑 partner5 部署。

## 运动学和资源
**240/240 策略×组** 的 speed p95、acceleration mean/p95、jerk mean/p95 均按原 literal ≤native×1.01 通过；所有 fixed16 EXACT/position bounds/独立几何重构通过。全行最大相对变化：speed p95 +.0178891%，accel mean −.0709050%，accel p95 0%，jerk mean −.0709053%，jerk p95 0%。这是全路径 pooled 科学 gate；pointwise 浮点误差证书不是评价 epsilon。

生产 peak private .163723 GiB/RSS .181915 GiB，最低可用 physical17.06554/commit26.56127 GiB；补充审核 private .136559/RSS .153610、最低15.31252/24.14624 GiB。process6 GiB、physical/commit 各 reserve6 GiB 未降低，没有 OS/pagefile/关闭用户程序操作。[生产资源](RESOURCE_USAGE.json)

## 研究判断与下一步边界
当前进入的是 **“安全可行，但有效生成空间不足”** 分支，而不是“已找到大量好候选、只差 scorer”，也不是已经证明 objective mismatch / head 唯一根因。

本轮 finite bank 已回答狭窄问题：在固定 .5 原生 partner、原 .20(t/12)^2 位置界与逐点 native-relative v/a/j ray 条件下，存在少量可行时间形状补洞，**但主变化组没有有效救回**。不应马上把这个固定 native-mixture 空间训练成新 head，更不应扫 partner、混合系数、bound 或放宽运动学标准。

若后续专门检验可学习性，精确候选设计只能是：利用 observed P2 与不可变 anchor/partner 几何，GT-free 产生或选择相容残差，再保持 native 支撑及全部原 gate；不能用本轮 GT 见证选 partner、生成 teacher 或改 scorer。本轮收益量级不足以优先批准/启动该训练。对更大的时间形状空间是否可行尚未回答；需先有不同、可辨识的结构假设与具体权限，不无限延续 reachability 变体。本轮按授权到报告结束，不自动新训练或续开旧六小时窗口。

## Inline ARS 反方检查 2（同助手，非外审）
PASS WITH LIMITATIONS。最强反驳：“安全时间形状成功救回了变化窗口，为什么不训练？”回应：承认 inner/CAL 两个变化见证；primary outer 的 143 个原生失败变化窗口救回0，固定策略误差变动小且 inner change ADE全恶化。只有有限支持见证，尚无可靠学习收益依据。
Major：bank 膨胀/GT posthoc 极易被错报为 K20 部署收益；本报告把每固定策略、bank、独立窗口数分开。Minor：alpha 零计数包含 self-partner；不把所有压零归因为约束。没有把无主集见证外推为所有生成器不可能。

## Inline ARS 反方检查 3（同助手，非外审）
PASS WITH LIMITATIONS。最强反驳：“把逐点相对 native 的限制作为结构，会让本来很小的原生曲率约束锁死真正变化，而 pooled 科学标准并不要求逐点全部同形。”回应：这是该参数化的合法限制和替代解释，不能据此证明行人不可预测、decoder 只会恒速或所有满足 pooled gate 的轨迹不可达；也不能事后放宽标准来求 PASS。有效收益不足，停止自动训练推进，不给 loss/scorer 唯一根因结论。
局限：只一个确定的 native pool、三个复用 split，不是三 seed 训练；F16 已见旧数据，target disjoint 非 context-disjoint、3 scene/相关窗口/重复 CAL/自适应研究历史均保留。执行成功不是 major PASS、理解运动、SOTA 或独立 whole-model 泛化。

AI-assisted analysis and the two inline critical reviews were performed by the same assistant, not independent external reviewers. Local evidence only; no invented statistical CI or external references.
