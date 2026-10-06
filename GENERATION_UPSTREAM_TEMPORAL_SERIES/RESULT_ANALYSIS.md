# Cycle6 experiment1：最后时间层联合训练

六次80轮训练完成；最终评估曾因 raw dev 路径映射错误失败。原始失败、源码、检查点不改，evaluate_repair.py 使用 native.extract 已确定的 train 文件及原 dev IDs，仅补评估；所有输出保存在 evaluation_repair/results。补评估16:14:03成功，18模型SHA、18显式ID有限K20预测、全部存储指标独立重算、冻结白名单审计通过，8626旧结果SHA不变。原生产失败记录不等于补评估失败。

| constrained，三seed平均 | 原native | 只训练新head | 最后时间层+新head |
|---|---:|---:|---:|
| change minADE20 | .372175 | .371394 | .369789 |
| change minFDE20 | .696326 | .691890 | .692880 |
| overall minADE20 | .219154 | .219110 | .218562 |
| overall minFDE20 | .399725 | .400658 | .399348 |
| stable minADE20 | .175177 | .175326 | .174843 |
| stable minFDE20 | .313804 | .316274 | .314030 |
| transition minADE20 | .275791 | .275562 | .275750 |
| transition minFDE20 | .512714 | .511687 | .512518 |

change相对native改善ADE约0.64%、FDE约0.49%，远低于各5%。native对照ADE delta -.002385 CI[-.005627,-.000453]，FDE -.003446 CI[-.011963,.000065]；fresh head对照ADE -.001605 CI[-.005913,.001973]，FDE +.000990 CI[-.010706,.011294]，joint -.001521 CI[-.011360,.006664]。未证明解冻的可靠增量，不升级baseline。

保护组、每seed变化改善、cal可行、覆盖、重复、硬运动学、同路径joint门槛通过；5%收益、native AND freshcontrol CI、运动分布门槛失败。transition jerk p95相对native增加2.26%，change jerk p95增加1.97%、acceleration p95增加1.87%，overall jerk p95增加1.55%。运动审计必须检查所有组所有指标，不以overall均值代替。注意oracle候选指标不是实际筛选top1/意图理解证据；三seed及重复ETH dev不是独立泛化。

治疗组选定受保护epoch为14/4/7；seed3后期cal_feasible=False，final策略transition minADE20 .286014、minFDE20 .534472，比早期受保护策略更差。此现象支持“后期训练与校准保护冲突”的待检假设，不能证明过拟合唯一原因，更不能证明F16是瓶颈。训练25158 versus99942参数，容量耦合，非等容量架构因果证明。

下一步：在新文件夹先读训练/校准history及原参数漂移，不访问dev选择子模块/超参。预注册候选方向为last-layer self-attention-only versus fresh完整last-layer，两组同新head/全初始权重/80轮/三seed/原loss-cal-budget；FFN和LayerNorm冻结是唯一训练范围因素。先验证子模块范围、实际GPU梯度、原冻结张量、观测输入、有限K20/native初始化、精确optimizer/RNG恢复；诊断不支持或预检失败不得开跑。此对照容量不同且重复full-arm不是独立泛化。所有结果留存，不放宽任何重大标准。
