# R2：固定安全监督下的非线性历史—候选匹配

日期：2026-10-12。状态：**设计与核心代码交付；没有实验运行、选择器拟合、求解器重跑或后台任务。**

## 1. 本轮只回答一个问题

在 R1 已保存的安全见证标签、训练窗口、轨迹隔离划分、样本权重、Logistic 损失及凸优化器完全固定后，加入一个固定的非线性历史—候选匹配表示，能否提高 top-4 命中安全可修复锚点的能力？

建议先采用固定 RBF 特征扩展，而不是直接切换 MLP。理由不是 RBF 已被证明更好，而是它仍可沿用 R1 的凸 Logistic 拟合，减少新优化器、随机初始化、训练轮数和早停带来的额外因素。该选择是待检验假设，不是研究结果。

R1 的事实基础：263/384 个预筛选窗口找到安全见证；Witness Hit4=66/263，Envelope=69/263，固定随机=74/263。实际共同训练正例只差49个，两臂85.16%的窗口选择集合相同。change 的最佳 P 排名中位数12，P>N 配对比例46.09%。这些结果支持调查匹配表示，但没有证明它是唯一根因。

## 2. 两个实验臂：2 × 3 OOF 折 = 6 次 CPU 拟合

| 项目 | L136 对照 | L136+RBF64 处理 |
|---|---|---|
| 输入 | 原 R1 的136维 X | 完全相同的136维 X |
| 原特征标准化 | 本折共同训练窗口全部20候选 | 与对照完全相同 |
| 非线性扩展 | 无 | 固定64个训练内 landmark 的 RBF |
| 可拟合系数（含截距） | 137 | 201 |
| 标签 | P正、N负、U屏蔽 | 相同 |
| 损失/权重/优化器 | R1 原样 | R1 原样 |
| 推理候选数/输出 | 原20候选中选4 | 相同 |

唯一实验因素称为“固定非线性表示扩展”。**不能称为等有效容量对照或纯历史因果对照**：新增64系数、特征几何和有效正则化都会变化。两臂相同 C 不等于相同有效复杂度。暂不同时换排序损失、增加原始观测特征、训练生成器或改 scorer。

## 3. 固定数据与防泄漏

直接只读 R1 `record_recovery_20261012/sample_design.npz` 和 `labels.npz`，不重新选384窗口、不重新求1022个解、不重新定义标签。P/N/U 应分别为329/7027/324；有 P 的窗口263。旧三折 target-track 归属不变。

每折训练窗口：不在本折验证集，且至少1P和1N。应为177/180/169。验证使用该折全部窗口；主指标再限制到有已发现 P 的窗口。每折 P 验证窗口86/83/94、P轨迹40/37/38。拒绝任何训练/验证轨迹相交、ID错位或标签数量不符。

原136维顺序严格是：P2观测96、native路径24、v_last与dv共4、原12个速度交互。X是已经保存的float32特征，计算统一float64。不要导入旧 `common.py`、`run.py` 或 `selector.py`，它们有模块顶层副作用；本交付独立读取NPZ，不读GT构造新输入。

以下全部只能从本折共同训练窗口获得：原 mean/std、RBF landmarks、RBF mean/std。包括这些窗口的全部20候选（含U），但不包括任何验证窗口。训练用P/N监督，U权重始终0。ID只用于固定landmark排序，不作为预测特征；fold/track只用于隔离与权重；group/scene只用于事后分析。

本轮继续使用已经多次查看的 A384，是**适应性探索筛查**，不是新的独立确认性验证。B、outer、inner、CAL保持封闭，不导入本代码。不得用这一轮结果回选landmarks、带宽、特征或C。

## 4. 非线性映射的精确定义

原始特征标准化：z=(X−mu)/sigma，sigma<1e−6时替换成1，原算法不变。

历史块 H：维度0:96及120:124，共100维。候选/交互块 Q：96:120及124:136，共36维。对本折所有训练 `(sample_id, physical_slot)` 用固定字符串 `r2-fixed-rbf64-v1|ID|slot` 的 SHA256 排序，取最前64个不同记录为landmarks。哈希碰撞用ID和slot破同值。不看标签、GT、group、scene、logit或效果；特征值重复也不重新抽取。

对每个中心 c_m：

    d_H = mean((z_H - c_m,H)^2)
    d_Q = mean((z_Q - c_m,Q)^2)
    phi_m(z) = exp[-0.5 * (d_H + d_Q)]

两个块按维度取均值，避免96维观测仅因维数大压过候选块。带宽固定1，中心固定64；这都是事先固定的建模选择，不代表已验证最优。

phi再次按本折训练全部候选做mean/std标准化，同样std<1e−6→1。

    score_linear = b + w^T z
    score_rbf = b + w^T z + a^T standardized_phi(z)

RBF项对候选不再是仿射函数，并且含历史与候选的乘性匹配。旧12个交互也在Q中，所以这不是隔离“只让P2参与”的纯实验。原生候选本身也携带历史信息。

评分接口只接收X和保存模型。top4按score稳定降序，分数并列取原physical slot先；保存选中slot时升序。不得按P、U、group或安全证书在推理时排除候选。

## 5. 标签、损失与数值要求

两臂都沿用 Witness：P=1、N=0、U=−1；U不当负例，不估计其“真实概率”。共同训练窗口内正负各占一半权重；轨迹总权重相等；权重总和=20×训练窗口数。`class_weight=None`。

    L = sum_i weight_i * BCEWithLogits(score_i, y_i) / S
        + ||coef||^2 / (2*C*S),  S=sum(weight), C=1

截距不正则化。LogisticRegression：L2、newton-cholesky、tol1e−6、max_iter500、单线程。不改C，不扫带宽/中心数，不加pairwise/listwise loss，不早停选checkpoint。数值收敛warning或独立梯度∞范数>1e−6则STOP；不得把未收敛当科学负结果。

必须先证实新L136与保存R1 Witness的mean/std/weights完全一致，分数max差≤1e−10且top4/Hit4完全一致；达不到先查实现/版本，禁止继续比较处理臂。这个容差只适用于数值重现，不是科学质量/运动学epsilon。

## 6. 主指标、统计与推进门

主指标仍是同一263个已发现P窗口上的 Hit4，不是最终生成轨迹coverage。一次性完成所有OOF后再评价，不边看各折边改代码。

RBF需要同时满足：

1. 三折原支持门：验证P窗口≥20，P轨迹≥10。
2. 对新L136、旧Envelope、固定随机4、bottom4四个参照，Hit4分别至少提高5个百分点，配对target-track bootstrap校正CI下界均>0。
3. 每个change验证折的命中数均不低于上述四个参照；原change门不放宽。

增加一个主比较，因此Bonferroni置信水平从R1的98.333%变为**98.75%**：2000次，四项固定seed20261012..20261015，按原窗口加权estimand重采样整条target-track，区间quantile=[.00625,.99375]。这是同一OOF预测的探索性CI，不包含整套训练重采样不确定性；115条机会轨迹、3折不是3个独立模型seed。不能把不显著解释为等价。

旧S1b只作描述性参照，不纳入因果对照。固定随机结果不能代表所有随机实现，另报告均匀随机选4解析期望。三个运动组/各scene/各折都要列原始分子分母；hotel样本太少不做强归因。

全部384窗口另报已发现P命中下界、若所选U可修复的上界、U选择比例、P>N配对比例和最佳P排名；U不用于真实二分类AUC。训练内与OOF均报同口径指标，训练内仅诊断拟合差距，不用作调参依据。P>N与Hit4不一致可提示目标不匹配，但本轮不换目标。

## 7. 结果分支与停止规则

| 结果 | 可以说什么 | 下一步 |
|---|---|---|
| 训练内、OOF都弱 | 此固定表示/目标组合不够；尚不能区分可表达性与目标错配 | 停止本RBF探针，不自动扫参或S2 |
| 训练内好、OOF弱 | 出现泛化差距，可能有限样本/迁移/发现标签选择 | 停止，不以加训练时间补救 |
| overall升但change门失败 | 稳定组收益不能替代变化理解 | 不B、不S2 |
| 全部筛查门通过 | 有必要做独立迁移确认，不等于理解/部署成功 | 另行锁定B的抽样与安全标签确认方案后再评估 |

“训练内好/弱”为描述，不另设事后科学成功阈值；只有第6节决定推进。**当前B没有本轮同口径安全P/N/U，不得拿包络标签直接替代B安全标签宣称通过。** 本交付不实现或自动触发B标签求解/生成器训练。后续生成器仍必须满足原同路径coverage、5%ADEFDE、保护组1%、各seed各group五运动学1%等所有门。

## 8. 必需预检和保存内容（尚未执行）

执行授权之后，先独立复核R1的完整2119相关source/1040输出SHA及标签证书，再继承旧digest、扩入新文件；不重算覆盖旧digest。`protocol.json`中的6个入口SHA只做入口绑定，**不是完整来源审核**。

真实预检至少包括：

- 384ID/136列顺序/labels/folds/track-disjoint/56条发现轨迹排除；全部数值finite。
- L136原Witness重现；共同训练统计和权重EXACT；验证数据/标签变化不改变训练map。
- RBF向量计算对独立逐项计算≤1e−12；行重排不改变landmark身份集合；rank与physical ties。
- 正负权重对半、track等权、U权重0；原损失和数值梯度/ Hessian检查；真实训练内固定小子集重复拟合，无需神经优化器RNG恢复。
- 每臂每折收敛与独立stationarity；保存模型读回scores和选槽重构；坏source/IDs/输出拒绝。
- 记录RBF分布、常量列数量、中心ID/slot，不因近常量或结果差自动换带宽/中心。
- process≤6GiB，physical与commit各留6GiB；拟合前后和分块计算内guard。实际预检计时后估算6拟合+复核；planning>30min不启动。不调整OS/pagefile，不关闭用户程序。

保存每臂每折：`model.npz`（coef/intercept、原与RBF标准化、64中心及其ID/slot）、`run_manifest.json`（source/split/labels/代码/协议/版本/特征顺序/超参/训练IDs/权重SHA/收敛信息）、训练内scores和OOF scores/slots/Hit4；保存全部CIs抽样结果及分组分母。评分保存后以独立NumPy重构，不再次fit。

本代码核心不包含完整生产launcher、resourceguard实现、source继承器或独立saved auditor，**不能当成生产预检已完成或直接双击可运行的实验**。未来生产封装需按已有独立worker、一次startup、一次50%预计耗时检查→一次完成检查流程。今天不创建任务或自动跟进。

## 9. 文件与使用

- `core_matching.py`：固定映射、原样权重、凸拟合与独立目标梯度、top4；import不拟合/不读数据/不写文件。
- `evaluation.py`：同路径标签Hit4、未知区间、排名、轨迹bootstrap、完整筛查门。
- `input_contract.py`：只读NPZ、入口SHA、schema/ID/fold及计数核验；不import旧模块。
- `test_core.py`：合成数组单元测试；不读项目科学数据、不fit、不求解、不训练。
- `protocol.json`：固定参数与入口SHA。

未来调用顺序（示意，不是生产启动命令）：

    data = load_r1(r1_root, protocol_path)
    # 外部生产封装先完成source/resource/preflight门
    for fold in (0, 1, 2):
        train, valid = split_masks(data, fold)
        for arm in ('linear', 'rbf64'):
            model, diagnostics = fit_selector(
                data['X'][train], data['labels'][train],
                data['track'][train], data['ids'][train], arm,
                guard=actual_resource_guard)
            scores = predict(data['X'][valid], model, guard=actual_resource_guard)
            # 保存训练/验证分数，检查linear重现，禁止OOF期间调参
    # assemble OOF；evaluate_screen；独立saved审核；报告后按门决定

## 10. 反方检查与边界

本方案由同助手使用academic-research-suite设计和inline反方检查，非外审。

- DA1：随机/哈希中心可能不适合可修复边界；因此失败只能否定此固定RBF表示，不能否定全部非线性模型。
- DA2：仍仅用有P共同训练窗口；安全见证不完整、U非随机缺失，R1的选择偏差仍保留。为了隔离表示因素，本轮不同时修这个问题。
- DA3：容量、有效正则化及核几何同时随表示改变；即使成功也不能独证“真正理解历史”。通过A筛查仍须独立B确认，且不能绕过生成/物理门。

没有生成新轨迹，因此本轮不产生新的ADE/FDE或运动学改善结果；只能改变哪些锚点被选中。不要把263个见证或Hit4当生成器实际覆盖。

## 来源

- R1 `record_recovery_20261012/RESULT_ANALYSIS.md`、`protocol.json`、`selector.py`、`labels.py`、`run.py`。
- R1 `completion_review/SUPERVISION_CONTRAST.json`。
- S1b `stage1b_history_interaction_20261012/features.py`，用于核对136列定义。

以上旧文件只读；本轮代码尚无真实数据拟合结果，不能虚构PASS或ETA。
