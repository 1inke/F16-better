# 共同协议：数据、训练、质量、统计与停止规则

版本 1.0；DESIGN ONLY。各阶段文件须与本文一起使用。任何改变须在新结果出现前形成显式版本差异，不修改旧科学协议。

## 1. 固定数据和输入边界

| split | 数量 | 用途 |
|---|---:|---|
| FIT | 18,909 | 拟合网络、标准化、经验中心；全量训练后评价 |
| outer | 1,035 | 主评价，含 253 个 change 窗口 |
| inner | 1,856 | 只作次级报告，不选 checkpoint |
| original_CAL | 2,879 | 只作迁移反例报告，不选方案或超参 |

继承原 target-grouped 精确 ID 文件，不重新抽样。原始观测过去 8 步、未来 12 步、dt=0.4s；使用既有经审核 focal 坐标，观测末点为原点。raw 末速度必须与 G1 的来源、单位和坐标一致。

P2 为已冻结的 96 维 mean-observed-token 表示，可能包含历史和邻居信息，不能称为纯 SELF。所有 normalization、K-means、尺度只由当前 FIT 得到。模型 forward 不接受未来 GT、GT group、scene/track/ID、native 路径或原生 logits；GT 只参与原训练 loss 和评价。

所有阶段固定 K20，不拼 native 成 K40，不事后按 GT 筛选候选。native20 为外部冻结强基线。CV1 明确只有 K1；既有 prototype20/retrieval20 可在 ID/SHA 兼容时复用为次级参照，不重训旧任务。

现有原始数据仅作二次计算研究，不收集新受试者、发布可识别轨迹身份或推断个人敏感属性。既有许可/来源必须保留；本文不代替机构伦理或数据许可认定。

## 2. 固定训练预算

- seeds：42、43、44；每臂固定 25 epoch，固定最后一轮为主结果。
- batch256、shuffle=True、drop_last=False；AdamW lr=0.001、weight_decay=0.0001、betas=(0.9,0.999)、eps=1e-8、foreach=False；不使用 grad clip、scheduler、早停。
- 相同 seed 的两臂使用共同参数逐 tensor 配对初始化、相同样本顺序和独立但匹配的 loader RNG。不同输入导致之后参数不同是预期，不是假定最终相同。
- 18909 人口下每 epoch 74 steps，每 run 1850 steps；每个双臂阶段共 150 epoch、11100 steps。续接初始化成本和上一阶段训练成本另报。
- 神经网络 float32；几何积分/目标叠加/保存路径 float64。不用混合精度，避免额外变因。
- loss 按窗口数量加权汇总；训练 batch 日志和全 population 指标分母分别标注。

## 3. 共同 objective

对同一个候选 k 定义 A_k=mean_t ||p_kt−y_t||、F_k=||p_k12−y_12||。

\[
L_q=\operatorname{mean}_i\min_{k=1}^{20}(A_{ik}+0.5F_{ik})
\]

\[
L_d=\operatorname{mean}_i\frac1{190}\sum_{k<l}
\left[\max\left(0,0.3-\operatorname{mean}_t\|p_{ikt}-p_{ilt}\|\right)\right]^2,
\quad L=L_q+0.05L_d.
\]

winner 和回归项用同一 Q；并列取最低候选索引。不能使用 K4 的除数 6。M3 同目标的不同过程也进入相同 190 对，不另改 diversity。没有新增 endpoint loss、物理惩罚、事件标签、soft assignment 或 class balance。

## 4. 指标定义

### 4.1 完整候选质量

\[
J_i=\min_k\max(A_{ik}/0.2,F_{ik}/0.4),\qquad C_i=\mathbb1[J_i\le1].
\]

主指标为 change 的 mean(C)。另外报告 minADE、minFDE、mean(J)、same-slot winner 的 ADE/FDE，不能把分别最小的 ADE 和 FDE 拼成一条好路径。终点指标 E_i=1[min_k F_ik≤0.4]，只表示必要上界。

报告 native→new 与 control→treatment 的 rescued/lost/retained，原 stable/transition/change/overall 四组，全 3 seed×4 split。J-closest 的同一完整路径早/中/晚误差为步 1–4、5–8、9–12 均值，不分别选择三段的最优候选。Q-winner 与 J-closest 是两个不同 slot，单独保存。

严格继承 GT-free complete-link：D=max(meanpath_distance/0.10,endpoint_distance/0.20)，SciPy 原实现 cut1。有效 good 簇数为至少包含一条 good candidate 的簇数，不是越多越好；重复率、簇数和使用率全部报告。0.3/0.5 仅固定敏感性，不作为换标准入口。

### 4.2 运动学

复用经审计的 native 口径：预测点共 12；速度由末观测原点加预测点共 12 段计算，加速度为这些速度的 11 个差分，jerk 为加速度的 10 个差分；均按 dt=0.4。**初始观测速度 v0 到第一未来段速度的跳变另报，不偷加进既有加速度分母。** 若实际旧科学函数口径与此描述不一致，先定位源并修正文档冲突，不带冲突实施。

每个 seed/split/group 对全部窗口、全部 K20、全部有效时间位置汇总：speed_p95、accel_mean、accel_p95、jerk_mean、jerk_p95。不是先求每窗口 p95 再平均。五项均须 literal new≤1.01×native，不添加评价 epsilon/floor。native 为零时直接比较绝对值，百分比未定义，不除零。

几何等式容差与科学运动学门分开：容差用于核验同一计算的舍入误差，不能放宽科学成绩。原生相对门不是“人类真实行为”的定义；可保存 GT 同口径作解释，但不据此修改门。

### 4.3 共同最终 major 门

每 seed 的 outer change minADE 和 minFDE 相对 native 各≥5%收益；overall/stable/transition 的 ADE/FDE 各≤1% harm；三个 seed 方向一致；native 对比 ADE/FDE/J 的校正区间支持改善；完整覆盖净增加且不存在靠复制候选得分；每 seed、每 split、每 group 五项运动学全部通过。再审查窗口/scene 相关性及只有 3 seed 的限制，才讨论升级。

新阶段的研究门不代替此 major 门；没有通过全部门就不改变 baseline/historycycle11/未完成 goal。

## 5. 统计与事前 family

所有 primary 在同一 outer ID 上做配对。先保存每 seed 每窗口指标，按 sequence+target_track 作为 cluster；每次对 track 有放回抽样保留该 track 全部窗口，同时独立有放回抽取 3 model seeds。两臂和 native 共享抽样；按抽中窗口权重计算均值，不把 seed×window 当独立观测。此 crossed bootstrap 仍仅是 3 seeds 的探索性不确定性。

固定 20,000 draws，rng seed=20261014，记录 RNG 算法、版本、初态和每次抽样索引 SHA。M1/M2/M3 预留一个 26 项主标量 family：

- 每阶段 B−A 与 B−native 的 change coverage/minADE/minFDE/J，共 8 项×3阶段＝24。
- M1 另加 B−A、B−native 的 change endpoint coverage，共 2 项。

Bonferroni simultaneous percentile CI：两端分位数分别为 0.05/(2×26) 和 1−0.05/(2×26)。未运行阶段仍不缩减除数。保存普通 95% CI 供描述，正式机制判定用上述校正区间。该 family 不是对过去全部适应性实验、多次看 outer 或全部 secondary 的校正，不能据此声称确认性检验。

inner/CAL/FIT、其他组和行为分层同法给出 95% 探索区间，明确不是额外 primary。保护和物理门对所有预定组逐项检查；不凭单一正 CI 忽略其他失败。V 的新 transfer family 另在其文档预定，不能混入本 26 项后缩放。

## 6. 共同预检：未来实施时必须真实执行

1. raw/cache/ID/坐标单位及排除关系；FIT-only 统计；旧相关 sources/models/results 原 digest 继承不覆盖。
2. 实际 GPU batch256 前向有限、实际合法梯度分阶段非零、paired 参数/buffers、固定冻结模块无梯度；零初始化导致首步前层梯度为零须如实记录。
3. canonical loss 独立 NumPy 标量及梯度对照；固定无并列 double fixture 两 eps=1e-4/1e-5 参数方向有限差分，原活动面切换单列，不搜索 eps 使结果好看。数值审核并非 float32 量化导数证明。
4. 两臂真实 FIT513+inner12 子集两轮连续/中断：参数、全部 buffers、optimizer、Torch CPU/CUDA、loader、NumPy/Python RNG、history 全 EXACT。子集通过不写成全量训练恢复证明。
5. 两臂真实128窗口保存/读取/full SciPy quality+motion 独立重算；完成 SHA-skip、坏 source/输出/恢复合同真实拒绝。
6. 每臂4 warmup+32测量 batch256，不保存 benchmark 模型作结果；加完整 FIT/outer/inner/CAL 输出、聚簇、CI、终 SHA 和审核估成本。精确参数量由实例化核验，设计中的手算值不是实际 PASS。

几何复算建议 abs≤2e-5m；float64 解析 fixture 建议≤1e-10m。若不适用，必须说明数值原因并在科学结果前锁定修订，不能以扩大科学门来解决。

## 7. 资源与运行纪律

每个双臂 production 对照规划≤30min；预检/实现单列，无实测就不报生产 ETA。process private≤6GiB，系统 available physical 和 commit 各 reserve≥6GiB，遵守长期约17GB可用预算；单模型/batch256、有界输出；不改 OS/pagefile、不杀用户程序。

未来获准实施后，才建立唯一按需0-trigger OS worker；确认无同活跃任务；唯一 startup、一次预计50% elapsed 检查、一次完成检查。完成时间修正按真实剩余，非对话高频轮询。worker 和 exit 双 SUCCESS、所有审核完成才 ANALYZED。旧 completed workers 不查询/重训，旧 PAUSED 不恢复。

## 8. 保存与最终报告合同

每阶段6 runs×4 splits＝24科学 saved jobs；初始化对照另存并单列，不混计24。每 run 至少：checkpoint.pt、run_manifest.json、history.json、metrics.json、assignment_stats.json、statistics、split IDs、explicit-ID K20预测；M2/M3 另存 goals、v0、raw/projected acceleration、全部积分状态和投影修正量。

manifest 绑定 source/split/input/native/cache/statistics/protocol/init/checkpoint SHA、精确 optimizer/RNG、所有训练参数、parent 初始化来源、冻结模块和总/可训练参数量。完整保存 FIT，不能用 minibatch loss 冒充 FIT coverage。

completion_review 从全量 saved 数据重算质量/聚簇/motion/分解/CI/几何，核对6×25history和checkpoint合同，以及全部 source/output SHA。CPU加载checkpoint核验不是再次神经前向或梯度回放。报告由同一助手完成的反方检查不能冒称外审。

结果表同时列 raw 数值、分母、每 seed、校正/未校正区间、所有失败项、工程与生产耗时、资源峰值、为何继续或停止。无论结果如何保留 failed code/logs/partialoutputs，不以新名称重跑同一阴性方案。
