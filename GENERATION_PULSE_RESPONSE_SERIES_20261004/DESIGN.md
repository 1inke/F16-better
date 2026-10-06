# 本地预先锁定方案：终点一致的候选特异响应曲线

## 问题与证据

比较固定响应时间曲线与由观测学习其时间位置/跨度，是否能在原生K20候选上提升完整路径覆盖及正式误差，而不损害稳定样例和运动学。依据是已保存响应缺口诊断的CAL731/1007粗响应覆盖但整路径失败（72.6%），FIT74.5%；CAL变化全部university，不能代表独立跨场景泛化。近乎全候选事件覆盖也出现在GT无变化样例，不能证明准确识别线索。[../F16_RESPONSE_GAP_DIAGNOSIS_20261004/INTERPRETATION.md]

此前自由12步修正相对新鲜分块对照没有稳健收益；三锚点GT辅助可达、旧时间查询/速度多项式分支失败及FIT改善CAL恶化均为反证。不能假定解码器唯一根因，也不把本结构当自回归、新未来观测或首次时间建模。[../GENERATION_TIME_RESPONSE_SERIES_20261004/RESULT_ANALYSIS.md]

## 唯一对照因素

两组均由固定原context64+原mode64构成候选特异128维查询；新query128->32，key/value96->32，对8个已观测token读出32维摘要（key保留固定观测PE）。新head160->128->64->6，GELU，最后层全零。两组均39590个名义可训练参数，所有初始权重和buffer配对，原生输出精确恒等。全旧F16/整个原decoder/scorer冻结；原缓存只读。

6个输出分别是二维终点漂移编码D、二维中途响应编码A、中心c与跨度s编码。固定增量中点时间t=(i+.5)/12。中心=.05+.90*sigmoid(c)，跨度=.05+.45*sigmoid(s)。Gaussian=exp(-.5*((t-center)/span)^2)，减去离散均值再除以最大绝对值，得到零和、单位范围的profile。

原limit=1.5*(观测speed+.03)。每步修正：
u=limit/2 * [tanh(D)+tanh(A)*profile]，
再调用原共同逐坐标clamp(-limit,limit)，累计后加到immutable nativeK20。

零和响应使12步终点修正=6*limit*tanh(D)，与profile中心/跨度独立（浮点容差），不是与GT终点一致，也不是两组最终训练终点匹配。按结构单位范围，共同clamp理论上不触发；实际必须检验。固定1/2分配收缩常量漂移表达范围，可能欠拟合，属于两组共有的新结构限制，不放宽原物理边界、不扫描系数。

- zero：仅把最后两个编码乘0，固定中心.5/跨度.275。
- mean_std：使用最后两个编码，按候选及观测学习中心/跨度。
- 唯一arm条件分支是使用/置零这两个编码。固定组对应两行数据梯度0，实验组激活后梯度正。名义参数相同不等有效容量或优化几何，不能作为纯因果容量证明。
- 高斯单响应不是真实语义事件类型/精确onset/duration，也不是compact support；不能保留多次复杂变化。原生路径仍保留全部20个模式，这仅是响应修正，不是独立生成器。
- 保留DT=.4、8观测/12预测、原模式slot顺序。没有GT/未来/group/logit输入，没有事件辅助损失、评分器更新、额外解冻或数据。

## 协议保持与判据

原FIT25912/CAL2879、原balanced hard joint winner及diversity/residual/smoothness/protection.1、原constrained CAL选择、LR/order/batch、80epochs、seed1/2/3均不变。六个全新配对训练；18 policy条目不是18独立训练。不加载失败teacher/旧修正head。仅原native checkpoint作固定基线初始化。

正式major仍要求change ADE/FDE各>=5%收益、overall/stable/transition<=1%伤害、三seed一致、native及freshcontrol双CI不跨0、同路径control1%、覆盖重复硬运动学，以及每组speedp95/acceleration均值与p95/jerk均值与p95<=native1%harm。首次全部通过还需scene/样本关联与仅三seed充分性审查。重复ETH DEV探索不称独立泛化/SOTA。

描述性输出：同candidate ADE/FDE及早中晚误差、固定.2/.4与.3/.5 samepath rescue/lost、响应中心/跨度/幅度实际使用和终点分离误差。原旧事件字段仅作描述，99缺事件不能当真实时差；完成报告重新解释，不用DEV结果选参数/checkpoint。若时间位置/跨度未实际使用、未优于固定组或改善伴随motion/保护伤害，否定此具体充分性假设，不声称所有响应结构无效。

## 必须通过的启动门槛

全部旧结果与父scientific sources/model SHA前后锁；实际GPU8FIT steps检查同配对初始化/native identity/有限K20/commonbound/梯度（key bias解析抵消不误称有效梯度）/整个旧decoder固定/在线原F16-cache alignment；实测响应零和、中心跨度可变、相同D时终点分离；双方真实2epoch连续-中断 parameters/allbuffers/optimizer/全部RNG/history/calibration EXACT、完成seed SHA-skip恢复。未通过不生产启动。

吞吐benchmark只用throwaway FIT步骤和CAL前向，不选科学结果/保存模型；GPU可能与用户图形程序竞争，不能声称隔离吞吐或停止用户进程。先给ETA，再唯一Windows按需无周期任务+字节独占锁真实启动确认。按人类规定单次50%预计时间查询，修正完成deadline并静默；完成立即审计分析。本次记response study experiment2，启动后completed仍1/3，第三新完成需总review。

## 内联反证和完整性检查

Academic Research Suite阶段内联执行，不是外部独立审稿：
1. 设计检查：缺口层级非因果，固定/学习曲线对照仅回答本结构；容量/函数/优化耦合披露。未发现允许launch的Critical逻辑缺陷，欠拟合/单pulse能力是Major局限。
2. 证据检查：保留上述失败、GT辅助反证、无事件候选覆盖、CAL单scene局限，不把已有数据当不可预测/意图证明。
3. 实现交付检查在GPU与恢复审核后保存preflight/resume，结果未出，不伪造有效性审查通过。

AI协助设计、代码和记录，人类授权并监督。仅既有公开轨迹位置子集，无新参与者、身份识别、外部上传或虚构人工盲评；不声称伦理机构批准或豁免。研究输出不用于个人监控/身份判断。公开注册未进行：本地方案在本次结果前锁定，探索性而非外部确认性注册。

全部新artifact仅此新文件夹，旧数据/模型/结果严格保留；不读heldtest、其他fold或外部数据。控制文件.memory/ledger/runtime允许更新。进一步实质扩范围必须具体人类新授权。

