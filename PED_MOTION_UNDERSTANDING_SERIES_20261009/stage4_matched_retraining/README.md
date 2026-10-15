# E3 fresh匹配重训：当前实现与预检状态

2026-10-09香港时间。当前 **没有生产worker，没有生产ETA，整套未完成**。DESIGN.md是实施前预注册快照，最新状态看runtime_state.json。

已实现 raw full/last2/noNbr 输入消融，插入在训练/验证/测试的所有heading/center/图/decoder预处理之前。隔离原trainer AST，仅保留imports/classes/functions，不运行旧顶层训练入口；原损失、优化器、EMA和调度科学语句不改。epoch恢复保存全部模型/布尔buffer/EMA/optimizer/scheduler/CPU-CUDA-loader-NumPy-Python RNG/history/已选checkpoint；仅承诺已提交epoch边界恢复，冲突的partial selected文件停止保留。

## 实际通过的门槛

- 原30份H3/V2保存训练记录的架构、损失、预算与实际raw数据SHA核对；425相关来源/输入绑定。旧记录并非完整历史训练锁源，因此不是保证旧训练能bitwise重现。
- 三分支×H3/V2共六项 **真实GPU ETH513FIT+12验证、两epoch**：连续 vs epoch1中断恢复，全部状态、历史、验证指标和所选模型tensor EXACT。H3分支初始权重buffer EXACT；V2使用本分支fresh H3，而非旧full warmstart。耗时289.938s，仅预检。
- 兼容completed SHA实际skip；坏源码、坏保存输出、改变epoch预算实际拒绝。没有改原checkpoint或输出。

详见SOURCE_READINESS.json、RESUME_PREFLIGHT.json、NEGATIVE_RECOVERY_FIXTURES.json及各preflight_runs。子集两轮不是完整fold两轮，恢复PASS不是整个生产预检PASS。

## 尚须完成，不能省略

实际full原输入/梯度等价及last2/noNbr完整预测通路信息隔离；全部官方ID/划分/单位审核；选定模型保存fullK20质量/原SciPy完整聚簇/五运动学及CPU独立审核；throwaway H3/V2/验证吞吐、资源和磁盘规模实测；最终source binding/prelaunch复核。

随后才启动45个fresh完整pipeline（五fold×三seed×三分支），每个H3_30→自己分支验证选定H3→V2_20，共90阶段训练。始终只新目录写入。生产代码还禁止没有实际PASSgate的启动。

一次13:15HK继续实施节点已安排，它不是预计生产结束时间。真实启动后，按实测ETA设置一次50%耗时核验和一次完成核验。过程6GiB、物理/commit各预留6GiB，不能放宽或动用户程序/OS/pagefile。
