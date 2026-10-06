# Cycle2 experiment2：原生几何输出头范围对照

假设：完整解冻早期checkpoint最优、后期变差，可能来自上下文/模式表示被扰动。冻结原生context、mode_embedding和mode_mlp，训练原生anchor_templates、proposal_scale、proposal_delta和refiner；共27154可训练参数，对照refineronly26514，仅多640。F16及score_head继续冻结。假设尚未证实，不宣称意图理解。

三次新训练，各80epochs，seed1/2/3；验证并复用上一实验refineronly三个80epoch对照，非独立重复验证。相同原生ETHcheckpoint/fit25912cal2879/数据顺序/损失/学习率/保护项.1/commonvelocitydelta限制1.5*(speed+.03)/校准选择规则；不增加test、跨fold、GT推理输入、分数训练、loss或coef扫参。generic zero为旧refinercontrol，mean_std为新geometryonly，并非集合均值方差特征。

预检原生identity、source/control18sha、逐seed配置、几何梯度/上下文模式冻结通过；2epoch真实GPU中断与连续参数及校准EXACT。初次预检seed对照校验错误及未训练smoke证据保留，不计正式实验失败。主要重大标准与native实验一致，包含samepath/覆盖/运动学分布allgroups<=1%伤害。首次重大信号额外场景相关性/仅3seed不确定性充分性审查，重复ETHdev非独立泛化。

独立后台入口start_independent.ps1，Windows按需任务Trajectory-GeometryHead-Cycle2无周期触发/低权限/无执行时间上限/拒绝多实例；日志追加保留，worker锁。应用可关但电脑须登录且醒着，自动分析仍需应用打开。18模型锁/18显式ID有限K20/重算指标/非白名单冻结审计正常退出后才判完成。复用9个对照产物，不是18次新训练。当前3/3完成后才做周期总review。
