# 持续响应实验结果（2026-10-05）

## 结论

完整生产成功，正式 major 仍失败，不升级 baseline。持续 sigmoid 响应让动态副作用明显小于配对的瞬态 Gaussian，但没有形成稳健的高质量候选覆盖增量，也没有可靠胜过新鲜对照。主政策仍是原 CAL constrained，不能用 DEV 改选 final。

formal change 组 minADE/minFDE 为 **.367384642 / .686950684 m**，native 为 .372174740 / .696326435 m，改善 **1.2871% / 1.3465%**；新鲜瞬态对照为 .364915103 / .686080754 m，两个平均值都更好。minADE 与 minFDE 可来自不同候选，不能拼为同路径成功。[results/summary.json: aggregate、bootstrap、gates]

| primary constrained | overall ADE/FDE | stable ADE/FDE | transition ADE/FDE | change ADE/FDE |
|---|---|---|---|---|
| native | .219154/.399725 | .175177/.313804 | .275791/.512714 | .372175/.696326 |
| 瞬态 Gaussian | .217983/.398973 | .174925/.314402 | .276290/.514956 | .364915/.686081 |
| 持续 sigmoid | .218646/.399322 | .175531/.315542 | .275439/.511058 | .367385/.686951 |

## 同路径覆盖优先，而不是只看误差均值

固定同一 candidate ADE≤.2m AND FDE≤.4m；仅位置覆盖，不表示全部运动质量通过。formal change223 和 clear_event469 是不同分组。

| 集合 | native | 瞬态 seed1/2/3 | 持续 seed1/2/3 |
|---|---:|---|---|
| 全体1516好候选覆盖 | 733 | 723/728/727 | **733/730/735** |
| clear_event469覆盖 | 92 | 88/88/88 | **93/90/93** |
| formal change223覆盖 | 43 | 45/45/46 | **45/44/46** |
| 全体GT无关有效好簇总数 | 929 | 909/911/817 | **934/930/935** |
| 全体 inherited native slot 合格 | 281 | 282/281/283 | **282/282/280** |

持续组 clear_event 救回5/6/5，损失4/8/4；全体救回11/17/13，损失11/20/11。同一批ID重复三seed不能加成人数；第二seed退化，不能只显示两seed净增。有效簇是全K20 GT无关 complete-link .10m meanpath/.20m endpoint 的簇中含≥1合格候选，不是越多越好。小幅数量变化不等新增独立运动意图，接近聚簇边界也可能改变离散数。[results/quality/summary.json；completion_review/REVIEW.json: rows]

旧原生 logits argmax 仅继承到新路径同slot，没有新路径重新评分；这不是新head判断的概率或新部署选择。三组 inherited 合格数量仍接近原281，不能据此声称新模型选择保守或意图判断改善。

quality 早/中/晚使用**同一 normalized-adequacy closest slot** max(ADE/.2,FDE/.4)；原 temporal 使用**同一 ADE+.5FDE joint-oracle slot**。均固定一条候选到末帧，不分段拼oracle；两种参考不得混淆。[prelaunch_review/FIELD_INTERPRETATION.md]

## 有意思的副作用变化，但严格门槛仍不过

以下是 ALLK、组内 pooled 实际运动数组，不是各窗口p95的平均；dt=.4。

| 平均加速度相对native | 瞬态seed1/2/3（%） | 持续seed1/2/3（%） |
|---|---|---|
| overall | 15.94/14.04/13.82 | **2.13/3.39/1.53** |
| stable | 13.21/11.50/11.45 | **1.77/2.94/1.17** |
| transition | 19.66/17.90/18.69 | **2.70/3.90/2.32** |
| change | 26.53/23.50/21.35 | **3.41/5.27/2.65** |

持续profile降低这一副作用是受控配对比较下可描述的结果；但函数类、幅度几何、clock-终点耦合和优化同时受profile影响，不能单独归因于“持续性”或等有效容量。所有组每seed accmean仍超1%保护线，因此不能称物理质量全部通过。其他规定speedp95/accelp95/jerkmean/p95较小，硬motion与重复/coverage通过也不能抵消这项失败。[results/quality/summary.json: global_pooled_motion]

## CI、保护和训练泛化

delta=持续−参照，负为改善；下列只是原注册三seed/相关窗口 bootstrap，非可靠独立scene总体CI。

| change参照 | metric | delta m | 原保存CI |
|---|---|---:|---|
| native | ADE | −.00479014 | [−.01321710,−.00058424] |
| native | FDE | −.00937570 | [−.02956562,+.00223986] |
| native | joint | −.00891196 | [−.02773497,+.00121576] |
| fresh瞬态 | ADE | +.00246954 | [−.00105424,+.00563834] |
| fresh瞬态 | FDE | +.00086991 | [−.00851385,+.00882320] |
| fresh瞬态 | joint | +.00422009 | [−.00319484,+.01074508] |

native ADE有探索性负向CI，但nativeFDE/joint和全部freshcontrol CI跨0。每seed相对native双误差微增益、原误差保护/可行/覆盖/重复/硬motion/samepath joint通过；双5%、native+freshCI、window分布和pooled全组motion失败。原major标准完全保持。

持续组原train loss epoch1→80约下降6.36/6.14/6.35%；CAL constrained选中epoch24/24/19，80epochs有17/36/36次不可行。末轮CAL change ADE .250355/.251677/.247985，FDE .439783/.440283/.434082；首epoch已训练，不能当精确native。FIT优化变好不保证CAL迁移；不可行仅checkpoint资格，不是worker异常。全部六histories、三policy所有组与96CI已汇集，不按DEV选新模型。[completion_review/ALL_THREE_POLICY_GROUPS_AND_CIS.json]

## 完成、资源和范围

03:23:23.027542HK启动，**03:45:31.566613HK完成，22分8.539秒**。04:56:02唯一计划半程核验已发现完成；不等待旧06:23截止，不重复已完成PID查询。实际远快于150–210min估计和129.978min微基准compute，原因未确认，不声称性能优化或更改工作量。

worker与results双SUCCESS；6全新80epoch训练，18policy SHA与18显式1516 ID有限K20×12×2，完整保存sample/nativegroup/ALLgroup/perseed/bootstrapCI/temporal/pulse分解审计PASS max0。whole旧F16/原decoder/scorer固定，347科学源码与27164旧文件不变。18policy不是18独立训练，保存算术不是所有模型再次重放。[results/independent_audit.json；results/saved_metrics_recompute.json]

本轮补充NumPy/JSON独立minADE/FDE、同路径覆盖/选择/closest槽复核及18checkpoint SHA；再次原旧27164全量before/after核验，首次before来自补充第一次格式断言前已完整通过的SHA循环。原 complete.json 状态是 COMPLETE，补充误期待SUCCESS被拒绝；只修正新审阅脚本，原脚本/trace保留completion_review/failures，不是生产失败或阈值更改。全聚簇复算依赖生产worker，补充不冒称再全量聚簇/模型重放。sandbox默认TEMP不可写的额外import失败也保留原因，未进入训练。[completion_review/REVIEW.json；failures/FORMAT_ASSERTION_FAILURE.json]

补充float64坐标参考与保存float32 min指标的最大差异5.00146e−7m，低于既定1e−6m校验界；不将它写为补充max0。生产自己的原算法全重算max0与此精度参考是不同审核。

生产peakRSS2.600GiB/private3.389GiB，minphysical15.820/commit14.434GiB；6GiB进程、system双6GiB余量通过。以后仍按约17GB可用RAM预算，不能全部占用或调整OS/pagefile/停止用户程序。[RESOURCE_USAGE.json]

## 总复盘与后续

第三正式完成，response study记3/3并先完成 RESPONSE_STUDY_REVIEW.md/JSON；历史cycle11只作未审计参考不重置。停止第四个局部pulse/profile/clock系数变体。下一方向是检查共享修正头是否直接利用**原生候选自身运动状态**，而非仅依赖mode embedding隐式描述；预注册可辨识newhead输入对照，所有旧F16/decoder/scorer/原损失数据和K20不变。这是未验证假设，非新的成功模型。任何实质资源/源锁/ID/权限或方法阻碍仍STOP等待人类。

本技能分析采用结论意图清单、逐来源衔接与inline反方检查，非外部审稿。局限：重复ETHDEV、3scene/重叠窗口、F16已见数据、CAL单university、三seed、选择性人工富集反馈；不能宣称SOTA、心理意图、不可预测比例或decoder唯一根因。
