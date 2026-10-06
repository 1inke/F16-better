# Cycle6 experiment2 — attention-only failed

SUCCESS2026-10-02 18:37:11HK，六个80轮模型实际107分54秒，而非预计210–270分钟。最终状态、report/audit、独立18SHA/全部源锁/每arm冻结白名单/18显式ID1516×20×12×2有限预测/全部存储指标重算通过，9172旧结果SHA不变。不能把两个嵌套reporter的旧标签当额外实验。

| constrained三seed平均 | 原native | fresh full时间层 | attention-only |
|---|---:|---:|---:|
| change minADE20 | .372175 | .369789 | .371849 |
| change minFDE20 | .696326 | .692880 | .702968 |
| overall minADE20 | .219154 | .218562 | .219144 |
| overall minFDE20 | .399725 | .399348 | .401268 |
| stable minADE20 | .175177 | .174843 | .175236 |
| stable minFDE20 | .313804 | .314030 | .314704 |
| transition minADE20 | .275791 | .275750 | .275768 |
| transition minFDE20 | .512714 | .512518 | .512264 |

attention-only changeADE相对native仅改善约.0875%，FDE退化约.954%；相对full ADE退化约.557%，FDE退化约1.456%，CI跨零不能据此宣称确定伤害。Native ADEdelta-.000326 CI[-.006056,.007498]，FDE+.006642[-.009349,.031942]；full ADE+.002060[-.004593,.011347]，FDE+.010088[-.008083,.037670]，joint+.006170[-.008230,.027188]。

5%/每seed改善/native AND freshfullcontrolCI/运动分布失败；保护组/cal可行/覆盖/重复/硬运动学/同路径门槛通过。motion transition jerk p95 +4.37%，change jerk p95 +3.74%、acceleration p95 +3.54%，均未达1%保护。选定epoch69/18/14，预期“仅注意力更稳且更好”未获支持。不放宽运动分布或选择新baseline，不证明意图不可预测或F16主因。

full组数值与之前UPSTREAM_TEMPORAL treatment完全相同，符合相同架构/初始化/数据顺序/种子，fresh执行不是独立泛化重复。可训练99942 versus62406容量耦合，非等有效容量纯架构证明。

第三项已预注册在新GENERATION_TEMPORAL_COMPONENT_SERIES/DESIGN.md：同固定LayerNorm、同新head，仅训练attention vs仅训练FFN，以更接近的参数预算检验更新位置。attention37248、FFN37152，合计62406vs62310，差96；实际实例预检再确认，非等有效容量因果保证。FFN大漂移不是伤害证明，仅attention限制未改善不排除FFN所需adaptation；该测试可能仍失败。保留所有旧结果，需GPU梯度、冻结/输入/初始化/有限K20/源锁和双方精确恢复通过才运行。目前设计阶段，无新正式训练。第三项完成必须cycle6全三实验总review，不能盲第四子模块/系数/epoch变体。
