# T1：完整运动模式支撑下，固定责任是否比动态winner更可迁移？

版本1.0；DESIGN ONLY。须同时遵守[共同协议](COMMON_PROTOCOL.md)。这是首项、也是当前唯一建议优先实施的训练对照；不是M2补跑。

## 1. 问题与可证伪假设

M2显示FIT学习真实存在，而outer相对初始化没有稳定净收益。H_T1：当候选具有共同的完整路径模式初始化时，将GT训练责任固定到FIT运动模式，比每步随模型预测变化的Q-winner更有利于学习可迁移修正。

这不是在比较是否使用future GT训练：两臂都用GT计算回归loss。唯一训练因素是**回归梯度交给哪个模式**。模型输出始终包含全部20个模式，不在推理时输入模式标签，不额外训练分类器。

## 2. 先建立共同的FIT运动模式，不训练神经网络

观测末点x0=0，v0=(x0−x_{−1})/.4。真实未来y_t在同一坐标，t=1..12：

\[
r_i(t)=y_i(t)-t\Delta t\,v_{0i},\quad r_i\in\mathbb R^{24}.
\]

这是完整未来相对末速度CV的**位置残差序列**，单位m；不是GT终点、加速度标签或语义stop/turn类别。不额外按未来速度旋转/缩放，不用测试数据决定方向。

只在FIT18909个r上构造20个L2中心。所有24坐标等权，不按坐标标准化、不做类别重采样。固定一次k-means++初始化，PCG64 seed20261015；Lloyd最多100轮，label完全不变时结束。不做n_init多次择优、不同K/距离/seed扫描。

精确定义：首点用RNG均匀抽FIT行；其后按到已选中心最小平方距离的概率抽未选行。概率和为零则STOP；距离并列取原FIT行序。Lloyd空簇时按当前最大重构平方误差、FIT行序tie，依次取不同样本重新置中心；记录事件，不另开随机试验。最终20中心按展开24维float64字典序排序，完全同值tie保留构造索引，保存中心顺序、每轮目标、标签和RNG。最后重新分配标签，不将旧轮标签误配新中心。

中心是样本平均，**不是20条真实观测轨迹，也不保证生理合理**。均值可抹掉罕见弯曲；这正是本方案的局限，不能把平均中心当动力学真理。

每模式至少50个FIT窗口且至少5个不同(sequence,target_track)，作为监督支持规模门。总FIT及全部模式/场景/track数量完整报告；不满足则STOP这一固定模式设计，不删稀有窗口、减K或换seed凑门。此数值是预定工程可辨识要求，不是统计功效证明。

只做身份/来源/数值和规模预检，不要求P0先赢native，不按outer表现挑中心。所有原型与计数在首次测试结果前锁定。

## 3. 共同生成器：完整模式＋观测条件修正

对窗口i、模式k：

\[
P^0_{ik}(t)=t\Delta t\,v_{0i}+\mu_k(t),\quad
P_{ik}(t)=P^0_{ik}(t)+f_\theta(\tilde h_i,\tilde v_{0i},\tilde\mu_k)_t.
\]

- h：冻结P2 96维；v0：2维；中心mu：24维，共122输入。
- 共享MLP122→128→64→24，两个隐藏层GELU，最后层weight/bias全零。没有learned query、goalhead、积分投影、CV/native选择器。
- 每模式都调用同一个MLP；最后24输出按12×2位置修正解释，单位m。无tanh bound、平滑核、事后clipping、物理penalty。
- NN输出float32转float64，再与float64 P0叠加。所有20路径输出，不删不安全候选凑K。
- 手算可训练参数25,560；实际实例化必须核验。两臂完全相同总/可训练参数，不包含冻结F16全模型参数冒充head规模。

h/v0统计每FIT窗口一次；mu输入统计按每FIT窗口所属固定中心mu_z构成的18909行，逐维float64 mean/std，人口方差ddof0，std<1e-6→1。输入原始特征按float32生产定义生成，统计float64累加，最后标准化输入float32。双方同一套统计，训练期间不重拟。

P0不是native identity，也不是M2-P0：它由当前FIT完整运动模式和当前观测v0确定。新P0输出在模型seed之间相同，不当3个独立随机基线。必须完整保存四split。

## 4. 唯一对照：责任规则

令q_ik=ADE(P_ik,y_i)+.5FDE(P_ik,y_i)，D为原190对平均平方hinge(.3)多样性项。

| 项目 | A：dynamic_winner | B：fixed_motion_responsibility |
|---|---|---|
| 回归责任 | j_i=argmin_k q_ik，每次前向重新确定 | z_i=argmin_k ||r_i−mu_k||²，FIT预先锁定 |
| 回归loss | mean_i q_i,j_i | mean_i q_i,z_i |
| 总loss | 回归+.05D | 回归+.05D |
| 推理 | 全20路径 | 同样全20路径；不用z或GT |

两种并列均取最低最终模式索引。B的标签只在FIT loss侧加载，不能进入forward或测试构造。验证若计算固定模式回归loss，必须明确其GT仅用于离线评价；候选输出先保存再评价。

**B不是对20条候选同时给予正确未来监督**：每窗口仍只有一个分支获得直接回归梯度，共享参数和diversity还会影响其他分支。固定z也不保证训练后模式含义不漂移，更不保证保留任何P0 good。只改变责任，不能称已经实现全候选合理性监督或support-preservation。

不改变Q为J，不新增分类CE、class balance、endpoint项或motion项。固定L2模式assignment与Q误差的几何不一致是明确局限；因此结果只能检验本固定责任方案，不是所有固定assignment与所有WTA的总判决。

## 5. 训练、记录和核心接口设计

6个fresh runs，25epochs×seeds42/43/44，其他完全按共同协议。禁止用M2/G1权重warmstart；只复用已冻结且SHA兼容的P2/原数据。

建议实现模块（本次不创建可执行代码）：

```text
fit_motion_dictionary(fit_past, fit_future, fit_ids) -> centers, labels, provenance
build_fit_statistics(fit_p2, fit_v0, centers, labels) -> immutable_stats
forward(p2, v0, centers, stats) -> paths[B,20,12,2], correction
loss_A(paths, gt) -> scalar, q_winner, diagnostics
loss_B(paths, gt, fit_fixed_label) -> scalar, diagnostics
evaluate_saved(paths, gt, ids, native, p0) -> complete_quality_motion_events
```

不得让forward签名接受future/label/group/scene/ID。为了绘图，ID只能由外层保存器附加。

专项记录：每batch Q-winner/z的switch率；每epoch各模式回归样本数、更新范数；固定z对应Q与实际minQ差；中心→最终候选路径漂移；P0→final rescue/lost以及新增motion代价。训练中不查询outer，不用中间checkpoint选结果。

## 6. 判定规则

S_T1需要同时满足：

1. outer change B−A覆盖三seed均>0，均值≥5个百分点，48-family校正区间下界>0。
2. B−P0覆盖三seed均>0，校正区间下界>0；不能只比一个更差的训练控制好。
3. B change ADE/FDE均值不劣于A；outer保护三组ADE/FDE每seed≤A1.01。

只有S_T1和共同Q门全部通过才进入T2。若A好、B不好，报告动态winner正证据，但不偷换成A开启以B为前提的后续序列。需要新的明确设计决定，不自动测试第三assignment。

| 结果 | 可得结论 | 动作 |
|---|---|---|
| B比A好，但没有超过P0 | 避免了部分训练损失，不是可靠学习增量 | 停止后续训练 |
| B FIT更好，outer不增 | 固定责任未解决迁移 | 停止，不扫模式数或loss权重 |
| B覆盖有增量但motion失败 | 质量—运动学问题仍在 | 停止，不补临时安全项 |
| A/B都优于P0但无可靠差异 | 新生成结构可能有价值，责任假设未证实 | 报告，不声称B机制成立 |
| S_T1与Q均通过 | 有资格测试信息价值 | 才考虑T2 |

## 7. 必须保留的反方解释

原型、初始化和网络整体与M2不同；跨M2差异不能归因于责任规则，只有本轮A/B能。训练GT模式标签不构成推理泄漏，但如果它读取outer拟中心、选K、定类，则是泄漏。三seed不能消除固定原型初始化偶然性；本轮不搜索多个词典，结果只对这一事前固定词典成立。
