# 时序响应实验完成分析（2026-10-04）

## 结论

实验运行和完整性审计成功，但注册的重大改善标准未通过，不升级 baseline。以原 CAL 约束选出的 checkpoint 为主分析：12 步修正在 formal change 组 minADE/minFDE 为 **0.371457 / 0.694880 m**，相对 native 的改善仅 **0.193% / 0.208%**。新鲜分段对照为 **0.370760 / 0.691922 m**，反而数值更好；治疗相对对照误差增加 **0.188% / 0.427%**。native 和 fresh control 的 ADE/FDE/joint 置信区间全部跨零。出处：[生产 summary](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/results/summary.json>)、[全部保存输出补充核查](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>)。

本次结果不支持“在这一冻结特征、原监督和校准约束下，单靠放开段内 12 步修正就能明显救回困难候选”这一具体做法已有稳定收益；**没有证明所有时序解码无用、解码器不是问题、未来不可预测，也没有证明自回归更好**。协议和反证见 [DESIGN](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/DESIGN.md>)。

## 完成与审计

- 唯一独立生产任务于 02:46:51.293833 HK 开始，07:15:50.559104 完成；全流程实际 **268 分 59 秒**，包括启动哈希、训练、校准、保存、18 policy 评估与完整性审计。07:42:46 的单次截止核验确认 task Ready、实际进程已退出、新 stderr 0。唯一半程检查已于 04:18:23 完成，未重复启动或轮询完成 PID。速度差异原因未确认，不归因为训练故障。
- 六个 fresh 80-epoch trainings 全部完成；三个 paired seeds × 两 arm × constrained/legacy/final = **18 checkpoint policies，非 18 独立训练**。所有末轮和非可行 checkpoint 均保留。中途 CAL feasible=False 是选择资格，不是运行错误。
- worker 与 results exit 均 SUCCESS；18 模型 SHA、18 保存预测的 1516 显式 ID、有限 K20×12×2、原 F16/整个原 decoder/scorer 冻结、229 注册源码不变、18872 旧文件 SHA 不变，完成 worker 审计 PASS。出处：[worker](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/independent_worker_state.json>)、[完整性审计](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/results/independent_audit.json>)、[冻结审计](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/results/frozen_audit.json>)。
- 完成 worker 独立重算全部 sample metrics、原 native group、72 per-seed-group 行、24 aggregates、96 注册 bootstrap CI、六份连续 1–80 history、18 temporal arrays，最大 sample 差异 0。是**保存坐标算术审计，不是再重放所有模型**。[saved_metrics_recompute](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/results/saved_metrics_recompute.json>)。
- 本次新增 CPU 补充审阅，无模型 forward/optimizer/GPU 训练：再次验证 18 checkpoint SHA、229 source SHA；独立重算 18 输出 minADE/minFDE（最大差异 0）、native/治疗/对照 same-path adequacy、同候选早中晚、事件描述并对照原 secondary 一致；793 个注册来源/生产结果输入前后哈希不变。**18872 旧文件全量比对引用已完成 worker 审计，不冒称补充程序再次跑了该全量比对**。[补充程序](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review.py>)、[补充结果](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>)、[输入哈希](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/input_hashes.json>)。

## 比较究竟是什么

双方保留原 context64+native mode64，使用共同 observed-token time-query/head、固定 timePE；同 43426 nominal train parameters、全初始化和 buffers 配对、lastzero/native output EXACT identity。原 F16、原 decoder/scorer 全固定；原 K20 baseline 不漂移、无 TopM 剪枝、无新未来观测、不是 AR。

唯一差别是相同 clip 之后的时间投影：对照四步增量取均值并重复，治疗保留 12 步增量。**相同权重**时 steps4/8/12 修正总和保存；独立训练后权重不同，不能称 final anchors 匹配。函数容量、投影和优化几何耦合，相同 nominal parameters 不代表相同有效容量。共同 key bias 在 softmax 解析抵消，极小梯度是 roundoff，不是额外学习响应。原 loss/protection.1/CAL guard/clip/LR/数据预算不变。[run_head](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/run_head.py>)、[协议](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/protocol.json>)。

## 预注册主要结果

以下为三个 seed 平均；minADE/minFDE 可来自不同候选，joint 是同一候选的联合误差。constrained 是事前指定主政策；legacy/final 仅完整报告，不能因 DEV 数字更好替换主政策。

| 模型 | policy | 原 motion group | minADE m | minFDE m | joint oracle |
|---|---|---|---:|---:|---:|
| native | fixed | overall | 0.219154 | 0.399725 | 0.432746 |
| native | fixed | stable | 0.175177 | 0.313804 | 0.342700 |
| native | fixed | transition | 0.275791 | 0.512714 | 0.548157 |
| native | fixed | change | 0.372175 | 0.696326 | 0.746631 |
| 分段对照 | constrained | overall | 0.219315 | 0.399612 | 0.432668 |
| 分段对照 | constrained | stable | 0.175310 | 0.314172 | 0.342818 |
| 分段对照 | constrained | transition | 0.277644 | 0.514567 | 0.550730 |
| 分段对照 | constrained | change | 0.370760 | 0.691922 | 0.742923 |
| 分段对照 | legacy | overall | 0.219258 | 0.400034 | 0.432939 |
| 分段对照 | legacy | stable | 0.175126 | 0.314309 | 0.342903 |
| 分段对照 | legacy | transition | 0.277266 | 0.514335 | 0.550054 |
| 分段对照 | legacy | change | 0.371628 | 0.694368 | 0.745047 |
| 分段对照 | final | overall | 0.220148 | 0.402499 | 0.434683 |
| 分段对照 | final | stable | 0.176423 | 0.317413 | 0.345773 |
| 分段对照 | final | transition | 0.279886 | 0.523997 | 0.556848 |
| 分段对照 | final | change | 0.368820 | 0.686482 | 0.736290 |
| 12步修正 | constrained | overall | 0.219471 | 0.400256 | 0.433297 |
| 12步修正 | constrained | stable | 0.175295 | 0.314180 | 0.342809 |
| 12步修正 | constrained | transition | 0.278071 | 0.515929 | 0.552682 |
| 12步修正 | constrained | change | 0.371457 | 0.694880 | 0.745267 |
| 12步修正 | legacy | overall | 0.219572 | 0.400566 | 0.433532 |
| 12步修正 | legacy | stable | 0.175353 | 0.314492 | 0.343080 |
| 12步修正 | legacy | transition | 0.278224 | 0.516807 | 0.553039 |
| 12步修正 | legacy | change | 0.371708 | 0.694602 | 0.745204 |
| 12步修正 | final | overall | 0.219288 | 0.400721 | 0.432803 |
| 12步修正 | final | stable | 0.175546 | 0.314016 | 0.343002 |
| 12步修正 | final | transition | 0.280801 | 0.526906 | 0.560133 |
| 12步修正 | final | change | 0.366241 | 0.687700 | 0.733437 |

[全部指标和 per-seed 数值](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>)。constrained 治疗 overall ADE/FDE 伤害约 0.145%/0.133%，stable 0.067%/0.120%，transition 0.827%/0.627%，均在 1% 保护范围内，但这不抵消收益和运动学 gate 失败。三个治疗 seed change ADE 均小幅好于 native，seed3 FDE **0.697696 > 0.696326**，不满足 each-seed 双指标改善。

### change 的注册 CI

delta=12步修正减参照，负值表示改善。以下为原注册 seed/sample bootstrap CI，只有 3 seeds，且样本存在场景相关性；不是独立跨场景泛化证据。

| 参照 | metric | delta | CI |
|---|---|---:|---|
| native | minade20 | -0.0007175 | [-0.0029463, 0.0016981] |
| native | minfde20 | -0.0014464 | [-0.0076014, 0.0048181] |
| native | top3_ade | 0.0000710 | [-0.0017359, 0.0023969] |
| native | joint_oracle | -0.0013645 | [-0.0062570, 0.0038629] |
| 分段对照 | minade20 | 0.0006976 | [-0.0004647, 0.0022064] |
| 分段对照 | minfde20 | 0.0029579 | [-0.0000328, 0.0067211] |
| 分段对照 | top3_ade | 0.0006451 | [-0.0003631, 0.0020863] |
| 分段对照 | joint_oracle | 0.0023439 | [-0.0002814, 0.0057693] |

全 96 个 CI 保留在 [REVIEW](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>) 的 ALL_registered_CIs；未挑显著项、未添加事后检验或可靠 scene CI 声明。

### 全部重大标准

| gate | outcome |
|---|---|
| change_ade_fde_gain_5pct | FAIL |
| protected_groups_guard | PASS |
| each_seed_change_improves | FAIL |
| native_and_zero_ci | FAIL |
| all_seed_feasible | PASS |
| coverage_guard | PASS |
| duplicate_guard | PASS |
| kinematic_guard | PASS |
| same_path_joint_guard | PASS |
| motion_distribution_no_harm_1pct | FAIL |

因此不是 major，也不是新 baseline。首次全门槛信号才需追加场景/样本相关性和 3-seed 充分性审查，本次没有这样的信号。

### 运动学

下表是 constrained 治疗相对 native 的变化，正数=增加；dt=.4s 原标准不放宽。allgroups mean acceleration 和 mean jerk 多项超过 +1%，即使 hard extreme fraction gate 仍通过。

| group | speed p95 | accel mean | accel p95 | jerk mean | jerk p95 |
|---|---:|---:|---:|---:|---:|
| overall | -0.092% | 3.904% | -0.010% | 1.155% | 0.168% |
| stable | -0.034% | 3.427% | -0.037% | 1.048% | 0.153% |
| transition | -0.085% | 5.000% | 0.019% | 1.420% | 0.197% |
| change | -0.420% | 5.310% | 0.095% | 1.450% | 0.228% |

分段对照的这些约束均在 1% 内。治疗整体平均加速度 +3.90%，change +5.31%；mean jerk change +1.45%。这说明增量的自由变化实际发生，但不是 GT 对齐改善的证据，也不能把所有加速度增加直接称为不物理或某个根因。[summary](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/results/summary.json>)。

## 是否救回“不能生成高质量候选”的样本

使用固定、事前报告的 **同一候选 ADE≤.20 m 且 FDE≤.40 m** 作为严格描述标准；它不是 SOTA 判断或新 major gate。另完整报告 .30/.50 敏感性，不按效果换标准。下表均是 any-of-K 同路径达标，不是 minADE 轨迹和 minFDE 轨迹拼接，也不是部署评分器选中轨迹。

| strict 标准样本 | native 达标 | 对照 seed1/2/3 达标 | 治疗 seed1/2/3 达标 | 治疗对 native rescue | 治疗对 native lost |
|---|---:|---|---|---|---|
| 全部 1516 | 733 | 736 / 733 / 735 | 734 / 735 / 734 | 9 / 10 / 11 | 8 / 8 / 10 |
| clear event 469 | 92 | 91 / 92 / 92 | 90 / 91 / 93 | 1 / 3 / 5 | 3 / 4 / 4 |

原全部 783 个无达标候选，治疗每 seed 仅 rescue 1.15%–1.40%，net +1/+2/+1。原 clear-event 377 个无达标候选，治疗 rescue **0.27%/0.80%/1.33%**，net **−2/−1/+1**。没有接近人类先前讨论的“救回失败样本 30%”设想，且该 30% 并未被擅自改为当前注册重大门槛。

治疗 vs fresh control，strict overall rescue/lost 为 3/5、6/4、10/11；clear-event 为 0/1、0/1、4/3。不能宣称治疗在困难例子有稳定增量。seed 的 ID 相同，不累加 rescue 为独立人数；也不从这些小计推 population significance。[paired_adequacy 与 temporal_rows](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>)。

.30/.50 标准下 native overall 1063、clear-event184；治疗 constrained 分别1068/1064/1062、188/186/184。有局部移动，但仍小且非一致，不能借放松阈值把本次变成成功。

## 时序反馈到底变了多少

clear_event 是旧固定 rule 标注的 **469 个明确事件**，不等同主要结果的原 formal change 分组。以下 GT-assisted **同一 joint-oracle 候选**的三个 seed 平均，只是后验描述；换方法后 chosen index 可改变，不能称同一个 mode 一对一因果对照。

| 模型 | early m | middle m | late m | kind match % | event detected % | onset MAE steps（已检出） | 真转弯方向正确 % | speed curve MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.15797 | 0.38998 | 0.57512 | 56.50320 | 67.37740 | 1.82911 | 53.47432 | 0.14670 |
| 分段对照 | 0.15794 | 0.39177 | 0.57457 | 57.49822 | 67.66169 | 1.83403 | 54.88419 | 0.14607 |
| 12步修正 | 0.15756 | 0.39193 | 0.57695 | 57.21393 | 67.66169 | 1.79731 | 54.88419 | 0.14584 |

治疗 early 略好（0.15756 vs native0.15797），middle/late 反而略高。事件类型正确率由 native56.50% 到57.21%，但分段对照57.50%；event detection 两新组均约67.66%，几乎不变。真转弯子集331的方向正确率，两新组平均均54.88%，native53.47%。这些小幅变化没有转化成 robust same-path coverage。

原 secondary 的 onset MAE 约30steps含 **99=no event** 哨兵，不能解释成真实预测时间错了12秒。补充程序分开报告漏事件和已检出的 onset：治疗约1.797steps，native1.829，对照1.834，即约0.72s条件误差，但遗漏/错误类型仍存在，且不同组有条件选择差异，不能据此称已解决 timing。类型匹配子集的条件 onset、真实转弯检测率/条件方向准确率也全部保留，必须与 coverage 一起读，不能挑约97%条件方向数字当总体准确率。

治疗 clear-event 段内修正增量偏离 block mean 约 **0.000547 m/step**，分段对照仅浮点误差约2.76e−8；final治疗约0.002943。这是真正启用更细增量的数值证据，但小幅动作/更多变化不等于正确的变化。speed-curve MAE 的微小改善（.14670→.14584）也没有伴随轨迹 rescue。[补充结果](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_review/REVIEW.json>)。

## 学习、校准与末轮

| arm | seed | constrained epoch | legacy epoch | FIT train loss change（epoch1→80） | CAL change ADE/FDE change（epoch1→80） |
|---|---:|---:|---:|---:|---|
| 分段对照 | 1 | 20 | 19 | -4.71% | 1.62% / 1.91% |
| 分段对照 | 2 | 21 | 21 | -4.76% | 2.46% / 3.15% |
| 分段对照 | 3 | 22 | 26 | -5.21% | 2.58% / 2.81% |
| 12步修正 | 1 | 20 | 20 | -5.03% | 2.11% / 2.69% |
| 12步修正 | 2 | 21 | 21 | -4.55% | 2.12% / 2.54% |
| 12步修正 | 3 | 23 | 26 | -5.56% | 2.14% / 2.77% |

这不是每步梯度/所有 FIT 轨迹误差审计，且 epoch1 已训练，**不是精确 native CAL 基准**。方向一致是 FIT objective 下降约4.5%–5.6%，CAL change ADE/FDE 恶化，支持“这套训练的改善没有稳健迁移”描述，不能独立证明过拟合唯一根因。

final治疗 DEV formal change 比native好1.59%/1.24%，仍不足5%、CI跨零；transition ADE/FDE 伤害约1.82%/2.77%，mean accel overall+22.37%、change+28.25%。strict clear-event达标90/85/88低于native92，虽然rescue14/11/13，lost16/18/17更多。**不能用 final 的局部均值好看替代已注册 constrained选择**，更不能用延长 epochs/放松 guard 的盲扫当下一步。

## 解释与后续讨论边界

1. 可编码/readable 风险、可细化段内轨迹、可学出正确未来，是三个不同条件。本次给了第二个自由度，但没有稳定达到第三个。
2. “锚点间只能线性，所以单放开12步就能修复困难例子”的充分性假说被这一实验削弱。没有排除更结构化的事件类型/时刻/方向/强度参数化，也没有排除监督信号/特征有效利用/泛化/不确定性问题；这些仍是待检验解释。
3. 新 scorer 没有训练，本次不能判断新的排序反馈对齐效果。GT-assisted oracle不等实际选中质量。
4. 原 GT辅助三锚点可达性、先前 timequery/velocitybasis 失败、FIT可记忆而CAL变差，是必须保留的反证。不能因“理论上更灵活”而默认扩大 decoder/encoder。
5. 下一次讨论宜先区分“事件漏报、类型/时刻/强度不对、局部动作有了但不泛化”。事件监督或状态化动力学若再试，必须单独设计对照并获得人类对新实验/损失范围的明确决定；不是本轮自动启动事项。**不自动第二实验、解冻、换 loss/scorer、加未来观测或新数据**。

旧结果/生产源码/模型未改，新解释仅在此文件夹。新 response study completed **1/3**，满3个新批准完成才做完整方向 review，不篡改历史未审计 cycle11 状态。

## 证据审查与局限

Academic Research Suite 的 source→对照→反证流程用于主代理本地自查，不声称外部或独立审稿通过。范围检查：注册问题只测时间投影，容量/优化耦合保留；分析检查：constrained/legacy/final、救回和丢失、各组运动学、全部CI共同报告；交付检查：不得把 oracle/条件子集/同seed复用/存储算术等同部署/泛化/独立重复/模型重放。伦理：现成轨迹数据，未新招募人工、未伪造盲判断、未声称机构审批。AI 辅助计算与解释已明确；未引入未核验外部论文支持结论。

三paired seeds、重复探索ETH DEV、相关场景/轨迹、native曾使用原训练数据，结论仅适用于本地固定协议，不是heldtest/跨fold泛化或SOTA；未产生全gate major信号，不作场景CI可靠性或不可预测性结论。全 claim预提交范围见 [claim manifest](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/completion_claim_manifest.json>)。
