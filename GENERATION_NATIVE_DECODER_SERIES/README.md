# 原生候选生成训练范围对照：cycle2 / experiment1

人类明确授权后扩大到原生候选生成解码器训练；F16主干和score_head仍冻结。唯一对照因素为训练参数范围：原生V2 refiner-only (26,514参数) versus context/mode embedding/mode MLP/trainable anchor templates/scale/delta/refiner (50,080参数)。两组同一ETH原生checkpoint、loss、80epochs、fit/cal划分、数据顺序种子1/2/3、学习率和优化器。容量不等是这次干预本身，不声称同容量架构胜出。zero/mean_std仅为复用报告器的内部标签，分别对应两种范围；没有集合均值方差输入。

不使用历史C00错误版本：使用当前缓存来源的fold-local ETH原生V2；全部train/dev候选缓存对原生推理精确对齐。冻结编码器输入只缓存观测，不传future到forward。训练目标GT/分组仅用于loss、fit权重和cal选择；开发集不选checkpoint。三种policy×两arm×三seed共18模型产物，并非18次独立训练。六次新训练，每次80epochs；旧epoch和失败证据不删。每epoch保留optimizer/各RNG及loader RNG以便精确恢复。

原生输出相对原native路径的逐步速度增量由共同trust-region clip限制到±1.5*(observed speed+.03)，然后累积。此为部署修改幅度限制，非GT性能/意图/物理保证；饱和时有梯度截断。原生refiner自身anchor bounds保留。两arm同样处理，不把生成器解冻等同无限自由生成。score_head参数不变但输入可变，因此分数可能失配，本轮只评估oracle生成质量，不能宣称筛选进步。

沿用>=5%changeADE/FDE收益、保护组<=1%伤害、三seed一致、native及匹配controlCI不跨零、samepathjoint及覆盖重复率硬运动学guard。额外预先规定速度p95、加速度/jerk均值和p95在所有组对native<=1%伤害，单位dt=.4s；不因硬阈值过关忽略分布退化。重复ETH dev探索性结论，首次重大信号需进一步场景相关性与3seed不确定性审计，不是独立泛化证明。不得访问test/跨fold/付费云/解冻F16。

运行入口：`run_native.py --action preflight`、`test_resume.py`、`run_native.py --action run`；独立后台入口`start_independent.ps1`。旧start_experiment.ps1已禁用，避免再次创建Codex子进程。新入口通过Windows Task Scheduler低权限按需任务（无周期触发）执行independent_worker.py，检查源码/完成模型SHA、拒绝重复进程、保留并追加原日志、自动恢复optimizer/RNG。Windows任务名Trajectory-NativeDecoder-Resume-Cycle2，可在任务计划程序中查看/停止；完成后任务定义留存但不会自动再跑。探针证明启动命令已退出后子任务仍运行20秒并正常结束，不等同亲自关闭Codex测试。计算机仍须保持开机、登录且不睡眠；关闭Codex不应影响Windows托管的训练，但Codex自动分析/优化需重新打开应用才能继续。报告位于results/summary.json、REPORT.md、audit.json、frozen_audit.json。
