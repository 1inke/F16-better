# Cycle6 总复盘：有限解冻没有可靠突破

18次fresh80轮训练、54个policy条目，不是54次独立训练。全部正常完成独立审计；历史失败仅首次评估rawdev路径，补评估证据完整保留。所有旧结果不变。

| 实验 | treatment changeADE/FDE | 相对native收益 | 失败门槛 |
|---|---|---|---|
| full temporal vs frozen head-only | 0.369789/0.692880 | 0.64%/0.49% | change_ade_fde_gain_5pct, native_and_zero_ci, motion_distribution_no_harm_1pct |
| attention vs full temporal | 0.371849/0.702968 | 0.09%/-0.95% | change_ade_fde_gain_5pct, each_seed_change_improves, native_and_zero_ci, motion_distribution_no_harm_1pct |
| FFN vs attention | 0.366534/0.687625 | 1.52%/1.25% | change_ade_fde_gain_5pct, protected_groups_guard, each_seed_change_improves, native_and_zero_ci, same_path_joint_guard, motion_distribution_no_harm_1pct |

## 结论与对照限制

完整时间层、小容量注意力、同预算FFN均未满足change双5%/保护组1%/三seed一致/native及freshcontrolCI/覆盖同路径运动学要求。FFN有名义变化改善但CI跨零、保护组和同路径失败；不能挑最好组或policy宣布突破。所有原始分组、policy、CI、校准history及source SHA完整嵌入JSON，不只报告change。

完整时间层与下一轮fullcontrol、attention-only与第三轮attentioncontrol有相同有效架构/种子/初始化/顺序；重复执行不是独立泛化。只有3seed且反复同ETHdev评估，有样本/场景相关性与适应性探索风险。本轮容量耦合、参数范数分母不同，不能把梯度可达性或大漂移当有用意图/主因证明。

训练FITloss下降但后期CAL保护常失败，支持优化-保护冲突/泛化受限的现象，不证明唯一过拟合原因。所有方法还保留nativeK20基底、原冻结decoder上下文读取、3anchor纠偏和common velocity clip；改编码器并没有测试完全独立候选生成。scorer固定但输入可变，本轮只oracle候选指标，不能主张实际筛选提升。

## 大方向调整

停止再拆分子模块、gate、平滑、系数或epoch局部试探。下一步先在FIT/CAL做固定可达性审计，区分候选纠偏函数类/速度clip的表达限制和实际学习失败：计算逐步box内GT精确可达率、端点必要下界、3anchor GT诊断构造的误差与clip饱和，不以构造当学得结果或最优上界。GT仅离线训练诊断不进入部署forward、不读dev选参数、不放宽guard。该诊断决定是否重新设计新生成头参数化；目前不启动第4局部训练。若需更广原decoder/F16解冻或独立泛化测试，应具体向人类请求新权限，不擅自扩展。
