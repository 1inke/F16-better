# 候选条件化路由：完成分析
实验实际21:22:23→22:25:25HK，63分2.359秒。22:40:40唯一完成核验确认worker与results exit双SUCCESS。六次全新80epoch训练、18policy、1516显式ID与有限K20、保存metrics/所有组与bootstrap CI、完整GT-free聚簇、双专家route/mix、36FITCAL原total/P、全原F16/decoder/scorer冻结、510源与31037旧文件SHA全部PASS。计算正常不等方法达标。阶段正式完成2/3，baseline不变。
证据：results/summary.json、results/quality/summary.json、results/physical_population/summary.json、results/independent_audit.json、results/saved_metrics_recompute.json及completion_review/REVIEW.json。

## 直接结论
固定候选状态路由没有解决“完整好候选跨seed/CAL不稳定”的问题。不再调median、专家数量、权重或epoch。
正式constrained治疗change ADE/FDE为0.373324096/0.703389943，native为0.372174740/0.696326435，即恶化0.3088%/1.0144%。均匀双专家control为0.374880612/0.715112627。相对control均值较好不等优于F16；native与freshcontrol的ADE/FDE/joint全部CI跨0，5%收益与seed一致性失败。

## 同一完整候选的覆盖
标准始终ADE<=.2m AND FDE<=.4m落在同一个候选。不拼minima或不同时间段。GT-free complete-link meanpath.10/endpoint.20只是去重，不是越多越好。
| 原生/对照/治疗 | 全体1516窗口，seed1/2/3 | change223窗口，seed1/2/3 | 全体有效good簇总数 |
|---|---|---|---|
| Native | 733 | 43 | 929 |
| Uniform | 736 / 726 / 731 | 45 / 47 / 47 | 944 / 817 / 930 |
| Routed | 727 / 738 / 729 | 42 / 47 / 43 | 926 / 826 / 927 |

Routed原生rescue/lost分别0/6、21/16、0/4。seed2净增加5个，全体另外两seed没有救回，仅丢失。不能把3seed同一批窗口相加成独立人；有效簇减少不能直接宣称整体collapse。继承nativeargmax选中good均281，是旧slot而非新路径概率/scorer或运动意图判断。

## FIT→CAL迁移与物理代价
下列final为固定80epoch终态，非部署或DEV选checkpoint。
| change组 | Native | Uniform final1/2/3 | Routed final1/2/3 |
|---|---|---|---|
| FIT6410完整good | 2603 (40.61%) | 3585 / 3505 / 3519 | 3642 / 3632 / 3616 (56.41–56.82%) |
| CAL700完整good | 243 (34.71%) | 222 / 231 / 232 | 227 / 237 / 221 (31.57–33.86%) |

反例保留：constrained routed CAL245/239/244，其中seed1/3小幅胜native243；constrained FIT2600/2779/2605，seed1几乎原生、seed3也只多2。全体FIT/CAL人口归一化loss参见JSON，不将final训练损失或batch loss与同组人口loss混用。
Routed final ALLK change accelmean较native FIT+2.25–2.44%，CAL+1.91–2.13%；Uniform final FIT+2.01–2.26%，CAL+1.78–1.95%。学习到更多FIT候选仍付出运动学成本，不是完全成功FIT/纯过拟合的证明。
正式constrained routed change accelmean害seed1/2/3为.1569%/1.4681%/.3371%；seed1/3全部group五motion通过，seed2overall/transition/change accelmean超1%。Uniform三个seed的change accelmean均超1%。旧window-p95平均不是global pooled p95，后者是最终门槛。
原P_acc将代价从旧约15%压到约2%但没稳定完成覆盖/迁移；不能据此断言只需调权重，也不能据固定route失败断言所有gating/共享mapping无用。

## 完整性与资源复核
生产peakRSS2.6936GiB/private3.5131GiB，系统physical/commit最少15.6858/15.2653GiB，资源健康。补充CPU NumPy36population原total/P误差0、18savedquality组计数与六80epochobjective trace、18models/510sources/31037old全部前后SHA PASS；没有模型前向/optimizer，也不冒称再次全聚簇/CI/gradient审核。
补充脚本初次假设原selection cache有gt字段而KeyError；failedsource/provenance保留completion_review/failures。该cache没有gt，已移除仅可选的第二次minmetric重放，生产saved_metrics_recompute原本已全metrics误差0。未修改数据、科学阈值或生成端。

## 反方检查2（同助手inline，非外审）
- 不能只选seed2/相对较差uniformcontrol来称成功；另外seed失败，native均值恶化，CI跨0。
- 两专家参数相同并不等有效容量与优化相同；路由也可能强化旧模式错误。
- finalFIT提高且motion不合格，不证明损失完全正确或纯泛化原因。CALconstrained小幅改善保留。
结论PASS_WITH_LIMITATIONS，所有major失败，禁止baseline/心理意图/SOTA/独立泛化声称。

## 反方检查3与下一步
学到完整覆盖和守住物理预算必须同时成立。下一步不盲做第三routing变体，先在全新F16_ENDPOINT_PRESERVING_ACCEL_PROJECTION_DIAGNOSIS_20261005做FIT/CAL冻结可行性检验：把修正增量分成不影响加速度的恒定时间均值与零均值时序扰动，只压后者，理论上保持同codes终点修正。
这只是新输出约束候选的筛查，不是新loss/scorer或第三正式训练。数学正确、保存端点/原clip一致性、资源、实际覆盖rescue/lost与所有motion都必须核验；不能因数学约束满足就宣称质量或泛化成功。GT只后验指标，不用于alpha或forward，不DEV/heldtest/其他fold，不旧头warmstart。
实施预检节点23:10HK，非生产ETA。真实吞吐、源锁和资源通过后才启动独立诊断，安排唯一50%与完成节点。若约束数学/数值或方法使可靠对照不可识别，停止等人类决断，不调阈值求PASS。诊断成功也不自动推定可训练性；第三正式需据FIT主证据精确设计。
AI辅助分析：同一助手做整理与反方核查，非外部独立审稿。
