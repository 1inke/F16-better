# 同组同指标FIT/CAL学习—迁移诊断（优先级2）
状态DESIGN_IMPLEMENTATION_PENDING，无worker/生产完成；非新正式训练，candidate-state三项已3/3reviewed。实施规划节点2026-10-05 13:00HK，不是运行ETA。现行合法scope：已有checkpoint冻结只读重放，原FIT25912/CAL2879，不DEV/test/fold/externaldata，不新增loss/scorer/监督/解冻。

## 为什么做
总FITloss下降而CALchange误差恶化比较不同群组/指标；目前不能准确区分训练内候选就未学好与学好了但不迁移。GT可达和loss奖励不填补该缺口。第三标准化也未改善可靠coverage，禁止第四局部norm变体。

## 固定模型与样本
主要只读三个已完成治疗族：第一active rawstate、第二observed-centered/rawscaler、第三observed-centered/ownFITscaler，各三seed，constrained和final政策共18锁定checkpoint。不得用结果重选checkpoint或混policy；其他重复control另列已有复现，非新独立证据。Native固定参照。
原FIT25912/CAL2879显式ID与原nativecache motion-group masks保持；主组overall/stable/transition/change。若无法按原规则复现ID/分组或纯只读加载，不造新分类，STOP人类。未来GT仅事后质量/分组标签，不进forward；onlypast16/frozen8tokens/nativeK20/base/speed。冻结全部原F16/decoder/scorer与各newhead所有weights/buffers，eval/no_grad，无optimizer/autograd训练。

## 主度量同两split
每窗口同完整candidate ADE≤.2 AND FDE≤.4覆盖、native救回/损失、raw好数与GT-free全K20complete-link(.10meanpath/.20endpoint)有效good簇；.3/.5固定敏感性。各组minADE/FDE、same normalizedadequacyclosest完整slot early/mid/late、selected inheritednativechoice仅描述。全组pooledALLK speedp95 accelmean/p95 jerkmean/p95 dt.4全部原保护。同指标计算FIT和CAL绝对值/相对native差，不拿总loss比min错误或只挑困难组收益。
不把不同split窗口配对、不报告可靠3scene groupCI或意图/不可预测率，CAL单university且用于checkpoint选取，不是独立heldout泛化证明。Constrained与final分开报告，FINAL冻结也不是根据本诊断挑选新部署policy。

## 结果分类（描述假设非因果判定）
1. FIT同组覆盖也低/没有native净救回，CAL也低：训练内映射/目标权衡/表达/信息充分性仍待区分，不能单独宣布容量不足。
2. FIT同组覆盖净增且路径运动学可接受，CAL无改善或恶化：迁移/分布或共享映射问题更值得后续检验，不能证明全部overfit根因。
3. FIT总loss下降但FITchange完整覆盖未增：指标/群组收益分配缺口，不等loss会排斥所有好路径。
4. 候选已好但inherited选择错：独立选择差距，仅排后，不训练scorer。
区分增益来自nearstatic/稳定组还是变化组，不用最大比例自动挑架构、不删难例/改阈值。

## 实施门与资源
所有newcode/cache/log/predictions/audits/failure只本folder。继承最新全old SHA，包含当前第三完整结果与review，不能覆盖旧digest；源/current_checkpoint SHA初末核验。不得导入父脚本导致旧目录写入；使用只读构造/隔离loader，所有参数buffersexact前后。预检：纯数学synthetic samecandidate/独立min非同path/去重20copies1簇/原metric与nativeID基准、实际FIT128 GPU frozen finite/bounds/checkpoint输出一致/原FITscalers不重fit/分组ID检查和CPU独立metric/聚簇重算。不需要训练梯度/optimizer2epoch恢复，不能伪称这些分析步骤为训练resume。
每次只load一个head，batch≤128，统计/聚簇每256窗口，保存压缩float32原路径。单模型两split路径約55.3MB，18模型約995MB未压缩，不全量常驻；64位/主进程6GiB、systemphysical+commit each6GiB reserve，人类约17GB可用RAM持续。含tokens/inputs/temp/聚簇与模型的实际峰值须实测；不OS/pagefile/kill用户程序。生产ETA必须用8实测批+CAL/FIT样本规模+全部hash/聚簇IO估计，不先宣称运行时间保证。
冻结分析job保存ID/checkpoint/source/输出SHA；仅验证完成job后skip，保留failure，不假称训练optimizer/RNG恢复。全部PASS才独立unique0trigger/start_independent.ps1/byte锁，确认唯一realPythonPID/serviceparent/stage/stderr，一次预计50%time跟进→实际completion节点并退出静默。不频繁query/查询已完成旧PID。

## 完成与范围边界
worker+exitSUCCESS/RESULT PASS_DIAGNOSTIC_ONLY/所有18checkpoint两split原ID有限K20/保存metrics+group/聚簇独立算术重算/原全部weightsbuffersexact/source与old全部SHA前后/RESOURCE记录后报告。诊断不增加formal实验计数、不改变baseline/major或证明可学性根因。
它是优先级2的低风险冻结调查，不是第四norm训练；新loss/eventsupervision/scorer训练/更多解冻/heldtestfoldexternaldata/paidcloud/messages/delete旧结果须具体许可。实质源锁/ID/数值/内存/科学或权限阻碍STOP留源等人类，不反复改方法使PASS。分析后精确选生成对照；不盲增宽head或新loss。

