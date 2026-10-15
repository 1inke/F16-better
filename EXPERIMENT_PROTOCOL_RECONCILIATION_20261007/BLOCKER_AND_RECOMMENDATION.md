# 实质训练协议不一致：下一步建议

## 源证据

现存源文件：

- `AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/run_best_of_4_3seed.py` 约275–297：`l_traj=ADE+.5FDE`，按该值选hardmin winner，6对candidate平均`relu(.3-distance)^2` diversity，乘.05；该训练段无gradient clip。
- `AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007/target_track/run_target_track.py` 约236–259：先按`max(ADE/.2,FDE/.4)`选winner，再优化同winner `ADE+FDE`，diversity=`exp(-closest_pair_distance)`乘.05，clip_grad_norm1。
- `AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007/recording_time/run_recording_time.py`约238–258与target同类。

三个程序均25epoch、AdamW lr.001/wd.0001/batch256，但质量分配、FDE权重、多样性形态、梯度裁剪不相同。模型checkpoint只有参数时也未将完整运行协议/训练输入标准化与权重绑定。以上是现存源码证据，不冒称已经证明每个checkpoint实际运行时源码版本；实际运行原source lock/log可提供进一步核验。

## 影响

各协议已有输出的同ID/同K指标仍可作描述；不能把三种划分结果直接解释成同一训练算法的跨协议迁移，也不能把macro8/9正、小CI或CAL差直接归因于scene分布或共享head。概率/输入契约问题是另一条已证据支持的缺陷，不能用本问题否定所有已保存轨迹。

## 下一优先任务（建议，未启动）

先统一可复现训练协议，冻结F16/native20/原decoder/scorer，原内部三种grouped划分、3seed重新跑同一newhead协议；每个checkpoint绑定sourceSHA、splitIDs、全部统计和objective参数。主比较为相同训练协议的三划分描述；各outer内部native20与相同预算K24对照，各group/同candidate完整覆盖/救回丢失/五motion原1%标准保持。不同outer不强行配对，重复外层仍探索。

建议统一采用已有scene-family实现（ADE+.5FDE hardmin、.05 hinge diversity、原无clip、25epoch/batch256/AdamW），因为这是既有baseline实现和config明确描述；不是在本次新加loss项。但是相对现存track/time会改变assignment/FDE/diversity/clip，需要人类具体确认。另可先提供实际运行锁定source/config/log证明现存源不是当时运行版本，再决定是否重跑。

不建议先换新结构或继续调slot/概率/权重；没有授权时不擅改候选分配、loss协议或训练scorer。现行source/可辨识gate为STOP_SCIENTIFIC_PROTOCOL_RECONCILIATION_WAIT_HUMAN。当前无实验running；旧42536/14892/37976/34284及所有历史完成PID不查询/重训。新证据/直接human决断前安静等待，不资源轮询、不每日重跑节点、不伪造ETA。确认后才精确预注册、实际GPU/frozen/paired/数值/real2epochoptimizerallRNGresume/源锁/resource benchmark、唯一OSworker、实际ETA、ONEhalf→completion。

此分析使用academic-research-suite证据/反方审查，全部同助手，不是外部独立审稿。Baseline与历史cycle11不重置或假complete，新增冻结审计不计formal。
