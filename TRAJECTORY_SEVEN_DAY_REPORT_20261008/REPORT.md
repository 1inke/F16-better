# 行人轨迹候选生成研究：最近七天实验报告

> 报告区间：2026-10-02—2026-10-08（香港时间，七个日历日）。截至本次整理时最新已分析结果。  
> 内容：实验设计、定量证据、结论修正、失败与可靠性问题、下一步研究建议。  
> 本报告是已有实验记录的证据整合，不是新增训练或独立测试。原始模型、科学源码与结果均未修改。

## 1. 先给结论

这七天的研究并非没有进展，但**尚未获得可以升级 F16 baseline 的重大突破**。最重要的进展是把问题从“为什么模型像恒速外推”收窄到几个可检验的环节：

1. **新生成路径存在小幅、可迁移的互补价值。** 统一训练协议后，保留 native20 再添加 learned4，在三个 grouped 外层集上的变化组覆盖分别增加 **2.84、2.77、1.64 个百分点**；原 CAL 变化组也增加约 **2.10—2.33 个百分点**。这反驳了“新生成器完全没有用、CAL 完全不能迁移”的笼统判断。
2. **这种价值还没有变成更好的等预算候选集合。** 在 K20 固定、删除低 logit 原生四条并放入 learned4 后，整体覆盖增加，但变化组 minADE/minFDE 反而恶化 **8.67%—15.52% / 9.57%—18.82%**；108 个分组运动学检查中 **79 个失败**。
3. **低 logit 不是冗余指标。** 被删路径对少数窗口是唯一的完整合格支撑。新增路径平均更好，并不能替代这些原生尾部路径。
4. **训练内会学出更多完整路径，但迁移与物理代价没有同时解决。** 早期 persistent/state 系列的 final FIT 变化组覆盖由 **40.61% 提升至 51.50%—53.87%**，CAL 却由 **34.71% 降至 30.86%—34.57%**，且加速度均值明显恶化。后续保护项和投影减少了副作用，但没有形成稳定全面收益。
5. **“更平滑”或“更安全”仍不等于更有用。** Bernstein 显著降低残差 jerk，却未改变完整覆盖集合；强物理 ray 投影几乎把输出压回原生。全路径相似变换做到安全且非零，但主外层变化组仍没有救回。

因此，当前最准确的研究判断是：

> 已有小幅新增候选支撑，但尚未证明能够在相同候选预算下，保护原生稀有支撑、补足变化难例，并同时满足跨 seed 稳定性与全组运动学要求。当前优先问题是**原生支撑友好的互补生成与集合集成**，不是继续盲扫 loss、scorer 或训练轮数。

这一判断依据统一协议、等 K20、尾部 utility 和最新可行域实验，而不是单个最优 seed。[S23][S24][S25][S26][S29][S30]

## 2. 如何读数据：先统一口径

### 2.1 “候选里有好轨迹”必须是同一条完整路径

对窗口 \(i\) 的候选 \(j\)，完整合格要求：

\[
ADE_{ij}\le 0.2\,m\quad\text{且}\quad FDE_{ij}\le 0.4\,m.
\]

相应连续指标：

\[
J_i=\min_j\max(ADE_{ij}/0.2,FDE_{ij}/0.4),\qquad J_i\le1\iff\text{存在完整好候选}.
\]

独立 minADE 与 minFDE 可能来自不同 slot，不能拼成一条合格轨迹。训练质量 winner 的 \(Q=\min_j(ADE_{ij}+0.5FDE_{ij})\) 与 J 最近路径也可能不同；旧报告的 `joint_oracle` 若采用 Q 定义，不直接当成这里的 J。[S11][S23][S24]

### 2.2 三类评价不能相互替代

| 优先级 | 评价对象 | 本报告解释 |
|---|---|---|
| 第一 | 同一完整候选覆盖、rescue / lost、ADE / FDE / J | 是否真正补足正确路径；既有成功是否被破坏 |
| 第二 | GT-free 全候选聚簇后的有效 good 簇与 distinct | 避免复制凑数；簇数不是越多越好 |
| 第三 | 实际选中路径 / scorer | 有候选不等于选得对；没有候选不能靠 scorer 补出来 |

GT-free 聚簇采用 complete-link，mean-path 0.10m / endpoint 0.20m。聚簇构造不用 GT；之后评价簇是否含合格路径才用 GT。五项运动学为全候选池 pooled **speed p95、acceleration mean/p95、jerk mean/p95**，\(dt=0.4s\)。不能用每窗口 p95 的平均替代全局 pooled p95。[S24]

### 2.3 不混分母、预算和重复次数

- 原 DEV：1516 窗口；其中原 formal change 为 223。后来的 clear-event 469 是另一种定义，不能与 223 混比。
- 原 FIT/CAL：25912 / 2879；同组变化窗口为 6410 / 700。
- grouped outer：target 1035（change 253）、time 566（change 142）、scene 1561（change 211）。不同外层 ID 不强行配对，也不按覆盖百分比直接给 grouping 难度排序。
- native20 + learned4 是 K24；保留 16 + learned4 是 K20。K24 的非减覆盖部分来自集合包含，不是等成本优越证明。
- 一项“两臂 × 三 seed”实验通常是 **6 次训练**。三个 checkpoint policy 产生的 18 份输出不是 18 次独立训练；相同架构、seed、数据顺序的 fresh 重复控制，也不是独立泛化验证。
- `SUCCESS` 是执行/审计完成；`major PASS` 是研究门槛通过。两者不能互换。

主研究重大标准仍包括 change ADE/FDE **各至少 5% 收益**、overall/stable/transition **不超过 1% 伤害**、三 seed 一致、相应配对 CI、同路径与覆盖保护，以及**每 seed、每 group 五项运动学不超过 native 1% 伤害**。Oct7 部分人工推进阶段另用 2% 或绝对物理阈值，其“PASS”仅属于该阶段，不能继承为这套严格 major PASS。[S05][S18][S20][S23]

## 3. 七天研究路线概览

| 时间 | 主要研究问题 | 实验/诊断路线 | 最终认识 |
|---|---|---|---|
| 10/02 | 输入读出、社会信息、有限上游更新能否改善候选？ | anchor/slot、token、social、最后时序块；小 FIT 记忆与 GT 可达性诊断 | 小误差改善不稳定；部分训练内可学，但 CAL 与运动学受损 |
| 10/03 | 集合监督、conditioning、邻居消息是否解决失败？ | soft/energy/mixture、dense/rank/tangent、邻居读出与 candidate exchange | 没有重大结果；不能由这些负结果推出不可预测 |
| 10/03—05 | 有线索却没有好路径，卡在哪里？ | cue、GT 曲线见证、head 学习、TIME/PULSE/PERSISTENT、state/centering/normalization | 风险可读、表达可达、loss 奖励、共享映射可学，是不同层次 |
| 10/05—06 | loss 是否推动完整覆盖？是否牺牲物理质量？ | matched FIT/CAL、参数梯度、credit/motion；保护项、路由、训练内投影 | FIT 对齐部分成立；CAL 与安全收益未同时成立 |
| 10/06 | 成功是否依赖相关窗口？ | 匹配 random vs target/time；scene-family 隔离 | 有相关 exposure 信号，但覆盖 CI 不支持“全面崩溃”；实际源码出现协议混杂 |
| 10/07 | 显式运动程序、组合与概率接口是否有效？ | oracle、确定性/MDN、Bo4、portfolio、scorer、校准；随后冻结复核 | 有候选互补；强“意图/概率校准/可靠部署”结论不成立 |
| 10/07 | 消除训练协议混杂后是否能迁移？ | canonical 三 grouping × 三 seed，共 9 run | 小幅 K24 补充可迁移；仍非等预算或 pure grouping 因果 |
| 10/07—08 | 如何保留原生尾部，同时安全加入新轨迹？ | 等 K20、tail/LOO、保护插入、native anchor、Bernstein、ray、similarity、temporal mix | 原生支撑不能按低 logit 简单删除；安全可行域与有效补充尚未同时解决 |

按结果进入报告；目录名中的日期不等于完成日期。例如 removed-tail 目录标记 10/07，实际完成分析为 10/08。周期3中的 DirectAnchor 实际在 10/01 完成，仅作为相邻周期背景，不计七天内新完成实验。

## 4. 10/02—03：大范围局部改法没有形成可靠增量

### 4.1 设计与公共参照

早期系列大多在固定 native K20 上加有界修正，FIT25912/CAL2879/DEV1516、三 paired seed、80 epoch；首先比较各项自己的 fresh control。这里 change native minADE/FDE 为 **0.372175 / 0.696326m**。

多数项冻结旧 F16/decoder/scorer，但不能概括为“七天所有实验都全冻结”：有限上游 temporal 系列曾在明确授权下更新最后时序块；conditioning 系列曾更新指定 context/mode 层。范围扩大已记录，不应与后来的纯 newhead/cache 实验混成一个协议。[S02]—[S10]

### 4.2 主要完成项与定量结果

下表为各项 constrained 治疗三 seed 均值，单位 m；收益相对本表 native。表内“小幅改善”不代表胜过 fresh control。

| 周期/因素 | 实验设计（治疗相对对照） | change ADE / FDE | 主要否定证据 |
|---|---|---:|---|
| C3 anchor feedback | 增加原生候选 anchor 坐标 vs zero 输入 | .374389 / .710006 | 相对 fresh zero 的 FDE 增量 +.018416，探索 CI [.003348,.035736] |
| C3 slot head | 20 slot 输出层解耦 vs shared | .375808 / .705654 | 参数增加 29.46%，未获可靠变化收益；过渡/运动学失败 |
| C4 token attention | candidate 条件注意力 vs uniform 读取 8 token | .368992 / .691457 | uniform 更好；同路径 Q 增量伤害 CI 为正 |
| C4 token contrast | last4−first4 token 差分 vs zero | .367102 / .689724 | 对 fresh zero 的 ADE/FDE CI 跨零 |
| C4 token sequence | 有序中心化 token 读出 vs zero | .367472 / .690307 | 对 fresh zero 的 ADE/FDE CI 跨零 |
| C5 social direct | 已有 social96 直读 vs 常量投影 | .367382 / .691784 | 变化收益 1.29%/.65%；fresh CI 跨零、运动学失败 |
| C5 social gate | 候选条件社会门控 vs 静态门控 | .368086 / .691406 | 变化收益 1.10%/.71%；未获可靠增量 |
| C5 social residual | FIT 去冗余社会表示 vs 完整表示 | .369678 / .693479 | 没有覆盖增加；可能删掉有用信号 |
| C6 temporal full | 最后完整时序块可训练 vs frozen | .369789 / .692880 | 收益 .64%/.49%，CI/运动学失败 |
| C6 attention only | 只更新 attention vs full temporal | .371849 / .702968 | FDE 恶化约 .95% |
| C6 FFN only | FFN vs attention | .366534 / .687625 | 收益 1.52%/1.25%，protected/seed/CI/motion 失败 |
| C7 local basis | Gaussian vs Fourier | .371775 / .695387 | 双误差收益约 .1%；fresh CI 跨零 |
| C7 basis conditioning | FIT 去相关 vs raw Gaussian | .371151 / .697156 | FDE 不改善，fresh FDE 伤害 CI 为正 |
| C7 basis composition | SiLU vs 同参数因子化线性 | .371685 / .697665 | FDE 恶化；非等有效函数类/容量 |
| C8 soft set | 固定 FIT 温度 softmin vs hard winner | .372741 / .695524 | 无可靠 fresh 增量，motion 失败 |
| C8 energy set | whole-set energy-like vs hard winner | .372047 / .694851 | 过渡 FDE +1.38%，保护失败 |
| C8 path mixture | 固定 FIT 尺度 mixture vs hard winner | .371924 / .696006 | native tiny CI 有利但 fresh CI 跨零；final 更差 |
| C9 dense conditioning | context/mode 适应 vs frozen | .372827 / .701321 | final FIT 改善 15.99%/24.77%，CAL 恶化 3.43%/5.04% |
| C9 rank8 | rank8 vs fresh dense | .376174 / .706894 | 容量减小仍 CAL 受损，protected/motion 失败 |
| C9 tangent | 固定原点切线更新 vs fresh dense | .371334 / .690406 | 收益 .23%/.85%；stable FDE +1.05%、motion 失败 |
| C10 pairwise history | 原始邻居历史 vs zero | .370583 / .694233 | branch 被使用但无可靠增量；change accel/jerk mean +7.78% |
| C10 neighbor readout | candidate-query attention vs uniform | .368425 / .694065 | 过渡误差受损、change accel mean +11.72% |
| C10 neighbor message | 非线性后聚合 vs 先均值聚合 | .374874 / .718431 | change FDE 恶化 3.17%，运动学失败 |
| C11 candidate exchange | 20 candidate attention vs own-value diagonal | .370789 / .690903 | 5%/native-fresh CI/motion gate 失败 |

来源：C3—C10 强制复盘和 C11 保存 summary。[S02][S03][S04][S05][S06][S07][S08][S09][S10]

**结论：** 当前具体局部改法没有可靠突破，不能继续把 width/LR/epoch、token/social、rank 或 soft winner 的相邻变体当主路线。但也不能推广为“所有社会信息、所有候选条件化、所有上游训练都无效”。名义参数相同常伴随 zero 分支梯度不同、容量或优化几何不同。

**记录问题：** C11 的保存 summary 已 COMPLETE/GATE_FAIL，但旧 `runtime_state.json` 仍写 RUNNING。报告采用已保存结果，不将该旧控制记录当成有 worker 在运行，也未查询进程；历史 cycle11 不据此虚构 3/3 总复盘。

### 4.3 小 FIT 学习与 GT 可达性：为何不是简单“模型完全学不动”

固定 FIT256/CAL256 的 memorization 诊断中，FIT ADE/FDE 从 **.176169/.300121** 到 **.149176/.236403**，改善 **15.32%/21.23%**；CAL 从 **.196514/.327677** 到 **.200516/.331918**，恶化 **2.04%/1.29%**。这支持训练内存在学习能力，但不是接近完美记忆，更不是完整泛化。[S31]

GT 辅助 3-anchor 或后来的 pulse/运动程序拟合证明受限表示中有可行曲线，**不能当过去输入预测出来的结果**。使用 GT 的诊断和正常部署的 past-only forward 必须分列。[S12][S19]

## 5. 10/03—05：把“有线索、可拟合、能学习”拆开

### 5.1 原问题同时包含生成缺口与选择缺口

原 DEV1516：native20 minADE/minFDE **.219154/.399725m**，实际旧选中路径 **.542432/1.197316m**。严格完整覆盖 **733/1516（48.35%）**，其中旧选中合格 **281**，已有好候选却未选中 **452**，根本没有完整好候选 **783**。另有 **56** 个窗口独立 minADE/minFDE 各合格，却不是同一候选。[S01]

这说明 scorer 不是零问题，但单改 scorer 无法补出 783 个窗口中不存在的完整路径。当前不能把较低 minADE 等同“已经理解行人运动”。

### 5.2 风险线索和未来路径可读不是同一件事

cue probe 在 FIT leave-scene-out 的变化检测上，temporal/social/context 线性 AUPRC 约 **.618/.670/.682**。这些是事件风险的可读性，不是方向、时刻、幅度及完整路径预测性能。

clear-event469 中只有 **21** 个属于所测 proxy 的 all-low cue 桶（4.48%，对应 11 个 target track），不能将 4.48% 命名为“不可预测率”。rule-positive 274 中，首个 GT 事件类型与方向兼容仅 **64** 个，另有 4 个对应较后事件；可见粗线索常不对应首个精确变化。GT pulse witness 可拟合 FIT **99.08%**、CAL **98.75%**，仍不证明可从 past 学得。[S12]

### 5.3 TIME、PULSE、PERSISTENT：表达自由度增加没有兑现覆盖

三项各两臂 × 三 seed × 80 epoch，保留 native K20、旧损失/保护与 frozen backbone。分别改变自由时间增量、可学习 Gaussian clock、sigmoid 持续响应。

| 治疗 | change minADE/FDE | 双误差收益 | 全体严格覆盖（native733） | clear469（native92） |
|---|---:|---:|---|---|
| TIME | .371457/.694880 | .193%/.208% | 734/735/734 | 90/91/93 |
| PULSE | .364915/.686081 | 1.951%/1.471% | 723/728/727 | 88/88/88 |
| PERSISTENT | .367385/.686951 | 1.287%/1.346% | 733/730/735 | 93/90/93 |

PULSE mean error 更小，却丢掉完整覆盖；PERSISTENT 降低动态代价，但仍未过全组 1%。三项相对 fresh control 没有可靠全面增量。Gaussian 与前项相同输出的 fresh 重复不算独立证据。[S11]

### 5.4 state / centering / normalization：小信号不够支持局部扫参

| 因素 | 同一完整路径全体覆盖 | change 双误差收益 | 主要问题 |
|---|---|---:|---|
| masked state32 → actual native state32 | 744/742/749 | 1.373%/1.369% | 有小幅覆盖信号，但 accel mean +5.23%—10.66%、CI/seed 不可靠 |
| raw state → observed-discrepancy centered state | 725/730/728 | .160%/.227% | 没恢复覆盖 |
| centered raw scaler → own FIT centered scaler | 726/733/731 | .089%/.209% | CI 跨零，seed3 全组 accel mean 超 1% |

中心化/自身标准化没有解决失败。可逆输入变换不创造新现实信息，但会改变有限初始化、优化与 weight decay 几何，不能说“已证明 normalization 不是任何问题”或“decoder 是唯一根因”。[S13]

## 6. 10/05：按人类提出的 loss—覆盖—credit—motion 分支做诊断

### 6.1 同组 FIT/CAL：真正的迁移缺口

冻结 18 个已有 checkpoint：raw / centered-raw / centered-own 三族 × 三 seed × constrained/final；完整 FIT25912、CAL2879，只做保存输出或合法 newhead 数值梯度诊断，不做 optimizer 更新。

| final 变化组指标 | Native | 已有 head 结果 |
|---|---:|---:|
| FIT 完整 good /6410 | 2603（40.61%） | 3301—3453（51.50%—53.87%） |
| CAL 完整 good /700 | 243（34.71%） | 216—242（30.86%—34.57%） |
| FIT population total loss | 参照 | 下降 19.01%—21.56% |
| CAL population total loss | 参照 | **上升 1.71%—4.73%** |
| change acceleration mean | 参照 | FIT +15.09%—17.89%；CAL +14.06%—16.27% |

不能将这写成“CAL loss 也下降而覆盖下降”：CAL loss 实际上升。也不能称“FIT 完全正确，只是纯过拟合”，因为 FIT 物理质量未合格。centered-raw 的 CAL overall/stable/transition 小幅正面例外应保留。[S14][S16]

### 6.2 参数反馈并非普遍无法到达 head

原 primary 的 change→change FIT/CAL 参数梯度 **18/18 同向**，cos **.230—.935**；范数不近零。overall→overall 为16/18同向；FIT overall→CAL change 仍有4/18反向。因此“变化质量信号一概相反/反馈被阻断”不成立，但混组干扰、clock 分支冲突、其他损失竞争仍可能存在。

这些是 checkpoint 局部一阶方向，不是实际 mixed-batch AdamW 更新、长期优化轨迹或覆盖因果。[S15]

### 6.3 hard winner 与完整路径 credit

已有完整 good 时，final quality winner 不合格比例：FIT **不足 .8%**、CAL **最多 1.67%**，说明“已经有完整好路径，却普遍给坏路径主要质量梯度”不是该总体的主要现象。

另一方面，GT Gaussian witness 插入可降低 total loss，且 CAL77 见证中64条在当前状态下不是 quality winner、直接 primary 坐标梯度为0；二者可以同时成立。它说明 hard assignment 的局部稀疏性，却不说明 witness 在当前 persistent head 中可学，也不能排除共享参数的间接反馈。J 最近不等于最容易修好。[S12][S16]

### 6.4 质量与运动学梯度确实有竞争，但不是单一根因

constrained 的9个 checkpoint 在每 split 上，quality 下降方向与 acceleration-mean 方向均有冲突；final CAL 有反例，p95 还受稀疏支撑、ties 与非光滑性影响。应同时看范数、dot、方向导数与 finite 副作用，不能只看一个 cos 就决定改 loss/head。[S16]

**对应人类分支：** 当前更接近“训练内完整覆盖提高，但物理代价和 CAL 迁移不足”的混合分支；尚未达到“loss 对齐、assignment 正确、无 motion 冲突，只剩结构泛化”的干净终点，也不足以优先宣布 objective mismatch。

## 7. 10/05—06：物理保护、candidate 路由、训练内投影三项闭环

三项各有自己的 paired control，不能跨项把容量与数值精度变化当单因果链。原质量 hard winner、其他 regularizer、FIT/CAL 与三 seed/80 epoch 预算不变；每项6次训练，三项共18次，54份 policy 输出不是54次训练。

| 实验 | 唯一因素与结构 | constrained change ADE/FDE | DEV整体 good /1516 | change good /223 | 五 motion 合格 seed-group /12 |
|---|---|---:|---|---|---:|
| Native | 原生参照 | .372175/.696326 | 733 | 43 | 参照 |
| Physical protection | 同45798 head，原 total vs total+.1P_acc | .371381/.690220 | 727/732/737 | 42/47/52 | 4/12 |
| Candidate routing | 同91596双专家；uniform .5/.5 vs 固定FIT median route | .373324/.703390 | 727/738/729 | 42/47/43 | 9/12 |
| Train projection | 同91596 uniform；端点保持加速度投影参与 forward/backward vs 无投影；双方float64输出 | .372979/.707470 | 736/735/735 | 47/48/42 | 12/12 |

P_acc 按原 stable/transition/change minibatch 的全20×11加速度均值，与同 batch native 的1.01倍作 hinge，再按固定FIT scale与n_g/B归一化。route 为 native 预测速度相对最后观测速度的 RMS 偏离，FIT median **.3674289**；路由不读 GT，也不是语义意图概率。[S17][S18]

### 7.1 保护项有效，但不是研究成功

final change acceleration mean 的代价从原 head 约15%降到 +P 的 **FIT1.72%—1.88% / CAL1.44%—1.55%**，仍高于1%。final FIT good 为3237/3254/3265，CAL230/242/232；constrained CAL244/247/246则小幅高于native243。不能只挑final负例，也不能用constrained小正例宣布解决迁移。[S17]

### 7.2 固定路由没有解除迁移失败

routed final FIT change good 达3642/3632/3616（56.41%—56.82%），CAL却227/237/221；正式误差逊于native，seed2运动学超标。相对较差uniform control变好，不等于优于F16；同名义参数不等同有效容量或样本梯度。[S18]

### 7.3 投影证明“守住加速度”与“获得覆盖”可以分离

训练内投影 constrained 所有12个seed-group五motion通过，但没有major。final FIT3621/3534/3560、CAL226/232/222；acceleration mean约+.82%已守住，但final仍有3个change speed-p95失败。constrained CAL248/252/248为正面例外。故“FIT所有收益都依赖超额加速度”不成立；“加速度守住后就能稳定迁移”也不成立。[S18a]

评估阶段曾因 dataset-only NPZ reader 被用于读取预测输出而失败；六组80 epoch与18模型已经保存。后续只恢复评估和算术审核，没有重训；失败源码/日志保留。最终三项3/3强制复盘完成，major均FAIL。[S18a]

## 8. 10/06：random vs grouped，能否判定相关窗口依赖？

### 8.1 身份审计先纠正一个前提

原 FIT/CAL 是 ordered 前25912、后2879，**不是随机窗口划分**；CAL 全为 university，111个track中32个与FIT共有。不能直接将原CAL失败叫成“随机划分泄漏被揭穿”。F16自身已见原数据；这些分组实验主要测试 newhead 适应迁移，不是整个模型未见数据的独立泛化。[S21]

### 8.2 Target-track：有连续误差信号，不是覆盖崩溃

两臂各18909训练窗口，sequence×原三loss-group预算匹配；共同交集15572仅供应双方统计。共同 outer1035、共享且整体track排除的inner1856。

random outer中1028个有同target训练曝光，1024个有已知20索引窗口重叠；grouped这两项为0。但场景时间proxy仍1031/1029，因此 target-disjoint **不等于 neighbor/context-disjoint**。

| target outer | Native | Random，三 seed | Grouped，三 seed |
|---|---:|---|---|
| overall /1035 | 625 | 642/649/632 | 642/622/632 |
| change /253 | 110 | 113/115/113 | 107/108/109 |

grouped−random覆盖 overall **−.87pp**、change **−2.24pp**，探索CI均跨0；连续ADE/FDE/J差异支持random有利的局部信号。grouped inner change166→165/172/179、hotel小组native10→均值12.33是反例。原CAL final change两臂仍基本不超过native243。grouped五motion全PASS，random有5个speed-p95严格失败。[S21]

### 8.3 Recording-time：剥离近时间暴露后平均误差变差

固定50个 **SOURCE indices** 时间块（不是秒或已证实视频frame）；按完整20索引区间purge。两臂train16104、common12544、outer566/inner2134。random有566个同时间暴露、564个同target overlap；grouped同时间0，但远时间同target仍227，所以 time-disjoint也不是target-disjoint。

| time outer | Native | Random，三 seed | Grouped，三 seed |
|---|---:|---|---|
| overall /566 | 303 | 323/320/316 | 286/304/312 |
| change /142 | 60 | 59/58/62 | 41/53/61 |

grouped−random整体覆盖 **−3.357pp，主time-block CI[−7.843,+.885]pp**；变化覆盖 **−5.634pp，CI[−13.732,+5.556]pp**，均跨0。整体连续ADE **+.009405m**、FDE **+.030328m**、J **+.062911**的主CI为正；change主CI均跨0，不能拿track敏感性CI替换primary。

inner change237/240/244高于native236、CAL某seed总体1502高于1490，是保留反例。216组motion中214通过，两项grouped seed1 change speed-p95超1%。[S22]

### 8.4 Scene-family与实际训练协议不一致：归因 blocker

zara held-family设计固定train16337/common12183/outer1561（change211）、inner1773；random含4154个zara训练窗口，grouped为0。这里只能匹配global三loss-group预算，不能宣称scene/sequence组成匹配。

后续实际源码审核发现，scene实现采用 **ADE+.5FDE hard winner、hinge diversity、无grad clip**，target/time源码却采用 **J winner、ADE+FDE、exp diversity、clip1**。此前文档“其余相同”的描述不足以绑定实际运行源码。

因此跨grouping横比实际是 \(Y=f(G,P_G)\)，不是只改变G的 \(Y=f(G\mid P=P_{canonical})\)。这是实质归因问题，而非简单统计波动；项目按此停止，保留旧实验并进入人类具体批准的protocol reconciliation。[S20a]

**对人类原分支的回答：** 有相关exposure依赖信号，但尚不能说grouped-FIT完全不能泛化，更不能唯一归因共享head干扰。旧matched random/grouped结果与后续canonical算法不同，不可拼成一个纯因果证据链。

## 9. 10/07人工推进路线：显式程序、portfolio、scorer和校准

这部分保留实际实验数据，但必须与后来的校正结论分开。部分旧报告采用更宽门槛、不同分组编码或强机制措辞；以下“不支持”是对推论强度的修正，不是删除旧结果。

### 9.1 Motion-program oracle：表达潜力很大，学习收益却小

在zara outer1561上，Static K24 good1061；GT优化同类别连续参数的S1为1455，自由GT程序S3为1490。J gap分解中parameter share约80.11%。这证明**当前程序有较大的GT拟合潜力**，不证明连续参数可由过去可靠预测，也不排除类别选择仍会制约部署。[S19]

旧oracle报告的change分母452与随后主研究211不同；本报告不把该452表并入211主结果。旧“分类学完备、分类不是瓶颈”的强判断不能由有限oracle分解单独确证。

确定性参数识别P0/P1/P2/P3中，P2在target/scene分别净增3/17窗口，但time净减4。正式三seed LearnedParam K24总体 **67.97%→68.97%（+1.00pp）**，change相对static下降 **2.37pp**；MDN总体 **+0.49pp**，change **−2.53pp**。两个具体方案都未通过阶段门槛。[S19a][S19b][S19d]

16-restart参数诊断在所选change/stable各211中，近同轨迹但参数分散的简并比例各 **8/211（3.79%）**；P2近邻检索中77.73%窗口的20邻居含至少3种机动类别。正确说法是“所测表示近邻下未来存在明显分散，所测oracle简并率低”，不能写成**彻底否定所有参数不可辨识、本质不可预测、单点方法必然失败**。近邻表示并不等同全部历史/场景信息。[S19c]

### 9.2 Best-of-4、静态/学习互补与固定portfolio

Bo4直接在轨迹空间训练4条path，三seed change覆盖相对Static K24分别 **+.47、0、−.47pp**，均值0；总体+1.39pp，但加速度p95有两seed超阶段2%。无简单slot collapse：有效slot约3.90/4，entropy约1.93bit。失败不能只归因“候选都复制了”。[S20]

K28的native20+static4+learned4 union，相对Static K24总体 **+2.48pp**、change **+2.05pp**；lost0由集合保留保证。GT挑4-of8能保留所测救回，不证明GT-free选择已解决，也不能推广为任何任务“四条预算绝非瓶颈”。[S20b][S20i]

slot审计中learned3偏减速/停留，change winner份额约61%；这是样本内物理语义倾向，不是心理意图识别。固定S1L3/S2L2有小幅组合收益；跨三协议确认报告给S1L3宏平均change **+1.49pp，8/9个run为正**。但它的阶段2%门槛、static基线重现问题、不同协议实际训练P必须同时注明，不能把局部`TRANSFER_PASS`升为项目严格major。[S20j][S20k][S20c]

P4 adaptive组合change平均+0.32pp、未过+1pp门槛；其阶段对accel-p95使用**双侧±2%**，下降约9%也被拒。该规则与主研究“一侧伤害≤1%”不同，不能把这类FAIL解读为轨迹过激或全局物理标准等价。[S20l]

### 9.3 Learned correction和scorer尝试：有净收益也有严重误伤

04A proposal-FiLM相对control平均coverage增量约0；相对各自Static K24虽有净增，却lost82—109窗口。common-support规范化均值69.49%，没有胜过04A69.64%；scene-balanced采样均值66.05%，低于其Static67.58%，lost139—153。它们支持当前具体修正/采样失败，而不是所有head/data balance都无效。[S20m][S20n][S20d]

候选GRU/set probe Top1 good **38.93%**、cross-attention **38.20%**，均低于native prior **40.23%**；selected ADE **.3661/.3701m**，native **.3513m**。这是两个新增scorer方案没有形成增量，不是“过去无法辨认意图”的定理。[S20o][S20e]

冻结Static K24内部验证报告列：28791窗口native good16358→static17012，增加654；原CAL1490→1544、lost0。但冻结proposal的分组复评、绝对物理阈值合规，**不自动等于独立whole-model泛化或主研究五项native-relative1%全通过**。旧“最高成熟度、可靠生成端”措辞应降为该有限配置的内部探索证据。[S20f]

### 9.4 概率接口冻结复核，纠正“校准成功”解释

原M1的CUDA softmax接口恢复后，NLL/Brier/ECE与原记录误差均<1e−8，说明先前重放错配已定位，而不是实验数字凭空无法重现。

但监督目标是 **softmax(−2J+3good) utility**，不是未来真实发生概率。overall selected-good ECE **24.395pp**、change **35.693pp**，不能称“低ECE、真实未来概率已校准”。overall utility优于uniform，change utility NLL **1.22277** 反而高于uniform **1.09861**。

同K3 native3 overall/change覆盖 **42.062%/20.671%**，推荐混合C1为 **34.953%/19.697%**。进一步固定native top2、只换第3条为L3：macro overall **42.062%→38.225%**，change **20.671%→21.268%**；三个协议change CI均跨0。scene有局部变化收益，却以大量overall/stable丢失换取，不能升为稳定配置。拒绝者成本未定义，也不能据selective interface宣称可靠部署安全。[S20g][S20h]

## 10. 10/07：Protocol Reconciliation，九个run消除算法混杂

### 10.1 精确设计与可追溯性

三原grouped splits × seeds42/43/44，共 **9次训练、225 epoch行、27份train以外保存评估输出**。只训练新的64032参数BestOf4Generator，P2 frozen observed8token mean96D、quadrant init/GeneralMotionProgram；原F16/native20/decoder/scorer保持不变。

固定canonical：winner assignment和quality均 **hardmin(ADE+.5FDE)**；六pair均值平方hinge margin.3 / weight.05；AdamW lr.001/wd.0001；batch256 shuffle；25 epoch；无gradient clip。各G仅用自己的train拟合统计，float32 mean/std，std<1e−6设为1。相同算法不等于相同统计数值。

每checkpoint绑定sourceSHA、splitIDs/SHA、statistics、objective/init、optimizer与全部RNG、history；最低产物包括checkpoint、run_manifest、metrics、assignment_stats。旧winner-switch在同四条路径上train1.50%—2.00%、outer2.43%—5.12%、CAL2.78%—3.47%；并没有20%—40%责任大交换证据，但小switch也不说明FDEweight/diversity/clip等价。[S23][S23a]

### 10.2 Primary outer固定epoch25：小幅互补可以迁移

| grouping | outer/变化N | overall native20→full24 | 变化good三seed | 变化覆盖增量 | change ADE/FDE收益 |
|---|---:|---:|---|---:|---:|
| scene | 1561/211 | 66.62%→69.36% | 82→89/88/87 | +2.84pp | 3.25%/5.25% |
| target | 1035/253 | 60.39%→63.57% | 110→115/118/118 | +2.77pp | 1.90%/3.87% |
| time | 566/142 | 53.53%→56.30% | 60→62/61/64 | +1.64pp | 1.49%/2.20% |

变化coverage探索CI分别 **[+.85,+5.25]、[+1.10,+4.74]、[0,+4.46]pp**；time触0。原CAL700 native243→scene254/261/258、target257/260/261、time256/257/262，平均增量2.10/2.33/2.19pp。[S23]

**结论修正：** canonical Bo4确有小幅grouped/CAL互补，不能继续沿用旧persistent“final CAL每次都失败”的概括。但不同算法改变的各项贡献没有被拆开；不能称它纯粹解决了泛化。

### 10.3 为什么仍不通过major？

- K24比K20多4条，lost0部分为集合性质，不是学会保护。
- 三项change双误差没有同时各≥5%；scene仅FDE超过5%。
- full24的108个组运动学检查中 **93PASS、15FAIL**，均为accel mean；outer **34/36PASS**，target seed42/43 transition为 **+1.5982%/+1.9723%**。
- 没有canonical random arm；grouping之间train量、outer IDs与组成不同。统一P解决算法混杂，不是composition-controlled pureG因果，也未排除相关窗口依赖。

训练九run已经完整；父worker在最后保存审核遇资源余量STOP，后续只恢复保存算术，不重训。恢复与完整CPU补充误差0。补充脚本曾把outer motion汇总错误写成全PASS，错误来源保留并校正为34PASS/2FAIL；本报告采用校正数。[S23]

## 11. 10/07—08：等K20与removed-tail归因

### 11.1 固定预算对照：平均更好的新四条没有形成全面更好的集合

冻结上述9个checkpoint的27份保存输出，native20 vs **common16 + learned4**。原生bottom4按冻结logit stable rank移除，不用GT挑slot、不训练任何模型。

| Primary outer | native good→替换三seed | overall覆盖增量 | change good→替换三seed | change覆盖增量 | change ADE/FDE伤害 |
|---|---|---:|---|---:|---:|
| scene |1040→1078/1081/1072|+2.370pp|82→86/85/85|+1.580pp|15.515%/18.819%|
| target |625→655/658/655|+2.995pp|110→113/116/116|+1.976pp|9.560%/17.747%|
| time |303→314/315/318|+2.238pp|60→59/58/61|−.469pp|8.666%/9.568%|

overall coverage CI均正；change coverage CI均跨0，变化ADE/FDE/J伤害CI却为正。108组五motion联合检查 **29PASS/79FAIL**，全部失败项为speed p95，最大+3.209%；其他四项通过不能抵消速度尾部失败。[S24]

learned4相对removed4、同K4的outer change覆盖增量 **+22.27/+18.71/+14.32pp**，因此新路径不是完全无用。问题是平均质量与rare support不同：新增路径能多救一些门槛附近窗口，但少数丢失窗口可能损失很大，导致coverage升、mean oracle error也升。

### 11.2 LOO utility：低rank路径承担什么？

全部lost共 **258个run-window行事件、51个唯一sample ID**，其中change32个ID；CAL九head丢失同7个ID，不是63个人。所有实际lost只有一个native完整合格slot，Q/J winner都在removed4，coverage leave-one-out utility=1。

\[
U_j^{cov}=Coverage(C)-Coverage(C\setminus\{j\}),\quad
U_j^{ADE}=minADE(C\setminus\{j\})-minADE(C).
\]

lost winner都被移除首先是common16不变的代数必然；真正新增证据是它们的**独占性和几何不冗余**。到common16的预注册归一化最近距离D，全部lost最小 **2.4234**，D≤1的案例0；CAL7例最近common平均path separation **.710m**、endpoint **1.460m**。[S25]

CAL7例原生J路径ADE/FDE约 **.150/.229m**；learned4单条J最近路径却为 **1.096—1.215 / 2.253—2.562m**。这里不是轻微跨门槛，而是失去不可替代近似路径。

### 11.3 “压掉低速/停留、救高速”没有普遍成立

51个unique lost按GT全程mean speed≤.1m/s计低速为0，但末3步停留11个。CAL7例GT路径p95平均 **.881m/s**，rescue约 **.575—.708m/s**，与“只救高速丢低速”相反；scene/target类似，time有不同方向。

较稳定证据来自集合组成：例scene seed42 outer speed p95，native20 **1.4245**、common16 **1.4620**、removed4 **1.1219**、learned4 **1.4427**、replacement20 **1.4581m/s**。common16本身甚至更快；不能把全部尾部伤害说成learned4是最快。删除较慢分量、加入相对较快分量会改变pooled分位点。[S25]

### 11.4 正确结论与边界

**低logit不等于低marginal utility，更不等于set redundancy。** 但高rank总体仍有一定utility关联；不能说logit毫无信息。单候选LOO也不保证联合删除安全：多个好slot各自U=0，可能一起删后丢失。

GT utility只用于事后诊断，不可用outer/CAL GT选择部署删哪四条。当前支持“组合策略破坏原生尾部”，不支持“所有新增生成无效”或“已证明安全集成”。

## 12. 10/08：原生支撑保护与可行参数化的三项研究

### 12.1 Protected insertion：安全，但一条也没有接纳

固定GT-free保护规则在27份输出、210744个按序proposal上，**accepted=0**；输出与native20逐数组相同，rescue0/lost0。首个拒绝原因：近已有路径22.38%、无可删原生cluster成员61.39%、交换违反更强逐窗口envelope16.23%。这些是有先后顺序的判定，不是独立因果份额。

该envelope比原科学pooled五motion更强，所以拒绝不证明所有proposal科学不合格。结果是“安全恒等退路”，不是成功集成，更不证明K20安全补充不可能。[S26]

### 12.2 Native anchor vs CV anchor：局部修正保留了一点覆盖，但动态代价仍在

共同25304参数MLP：observed P2 96D + immutable selected native path24D，共120输入；120→128→64→24 GELU。选择原bottom4 logit slots，固定其余16、K20；输出小残差：

\[
r_t=.20(t/12)^2\frac{\tanh(z_t)}{\max(1,\|\tanh(z_t)\|_2)}.
\]

endpoint残差≤.20m、mean-path≤.07523148m。原canonical四path loss/AdamW/25epoch/target grouped18909不变，两臂三seed共6次fresh训练。唯一因素是输出anchor为native4或CV4；输入仍一致。CV own-zero与native own-zero不同，不能只看final绝对值就作公平收益比较。

native arm outer change **110→111/111/111**，每seedrescue2/lost1，净1个；但36个seed-split-group全都因accel mean **+2.61%—3.05%**失败。小幅position residual不是物理安全保证。[S27]

### 12.3 Bernstein时间耦合：平滑残差，不等于改善整条轨迹

两臂相同12个bounded controls，一臂free12 identity，另一臂degree11 Bernstein线性组合。basis rank12、condition约7899；相同名义参数不等有限有界可达几何/优化相同。

- 两臂全部36个paired seed/split/group的**二元覆盖集合与rescue/lost IDs完全相同**。
- outer change仍111/111/111；inner change166→165；CAL243→243/244/245。
- residual jerk mean下降 **86.6%—89.9%**，但全path accel mean仍 **+2.66%—3.07%**，各36组失败。
- 训练四path batch loss下降 **2.38%—2.54%**，不是未保存的fullFIT覆盖提升。

这排除了“只要去掉残差高频就解决整条轨迹运动学”的简单解释；不能据此宣称所有smooth表示无用。[S28]

### 12.4 Native-relative physical ray：全motion通过，但输出退化至近恒等

第三项只在同Bernstein/native-anchor输出上加入31个native-relative speed/accel/jerk点约束，共同ray alpha；训练loss/预算不变。

physical arm36组五motion全部通过，outer/inner/CAL change分别 **110/166/243**，所有seed等于native；全部rescue0/lost0。alpha median0，零值 **53.76%—56.82%**，最大.003085；最大实际残差只有 **4.184×10⁻⁵m（.04184mm）**。它是几乎恒等的可行解，不是安全新mode。

强31点约束比原pooled标准更紧；native近零accel/jerk预算会压缩当前ray。这个负结果不能推成“所有满足原科学门槛的生成器都不可能”。舍入证书只处理约10⁻¹⁷级算术误差，未放宽科学1%门槛；有效数值缺陷仍拒绝。

三项native-anchor系列3/3完成强制总复盘：18次fresh执行、450epoch行、54保存jobs；重复控制不算独立泛化，major均FAIL。[S29]

## 13. 10/08最新方向：安全非零，不再恒等，但变化覆盖仍无突破

### 13.1 全路径similarity对照

在相同native anchor/Bernstein proposal上，比较旧ray与GT-free全路径rotation/scale。相似映射为\((aI+bJ)p\)，系数由native与proposal最小二乘求得，scale cap1.01，再以共同position lambda守bound；不读GT。它保留时间形状，不能新造stop/turn时序；stationary anchor保持stationary。

六次25epoch、18jobs完整完成；similarity9输出所有candidate都有≥1mm残差，median **7.16—8.18mm**、max **73.28—105.29mm**。两臂全部72组五motion通过。

但outer overall **625→625**、outer change **110→110**、inner change **166→166**、CAL change **243→243**，全部seed无变化救回；CAL整体1490→1491只是**同一个transition ID28854**在三个seed重复救回。outer change ADE约 **.249146619→.249037083m**，仅.044%收益；FDE **.429378298→.429420801m**，略有伤害，主要CI均跨0。

**结论：** 安全且有真实非零修正的几何已做到，但这个全局相似函数类没有显著补足变化时间形状，不是恒速偏差已解决。[S30]

### 13.2 Frozen temporal mix可达性审计：稀疏见证，不支持立即训练

只用保存路径、CPU离线，无torch/model/optimizer。每个native physical partner slot0—19定义一个固定K20策略：bottom4 anchors朝该partner作.5凸proposal，再用原position bound和31点ray；16原生保持。20policy × outer/inner/CAL = **60jobs**。

| 评价人口 | Native good | 固定policy / 有限bank结果 |
|---|---:|---|
| outer overall /1035 | 625 | bank626，仅新增1个stable |
| outer change /253 | 110 | **全部20policy及bank仍110**，143个hole无rescue |
| inner change /497 | 166 | bank167 |
| CAL change /700 | 243 | bank244 |
| CAL overall /2879 | 1490 | fixedpolicy1490—1492；bank1493 |

全部240个policy-group五motion literal PASS、lost0；outer change非零≥1mm占52.77%—70.16%，超global-similarity的shape RMS≥1mm占24.31%—45.95%。因此不是完全没有改变时间形状，但**主外层变化组没有一例救回**；distinct clusters与native完全相同。

bank最多含native20加76条新路径，不能把bank+3 CAL窗口当单个K20部署收益；所有救回仅5个唯一ID，44条success记录是重复partner见证，不是44个独立窗口。它支持少量门槛附近可行见证，不支持这个窄mix方案值得直接训练，更不允许GT挑最佳partner。[S32]

## 14. 最近七天整体判断：我们现在走到哪个分支？

| 人类提出的分支 | 当前证据 | 判定 |
|---|---|---|
| loss↓、coverage↑、motion稳定，则优先泛化/结构 | 原persistent FIT coverage↑但motionFAIL；CAL loss↑；canonical小互补但预算增加/transition motionFAIL | **未满足干净前提**，不能直接宣称loss完全正确 |
| loss/ADE/FDE↓却完整coverage不增，则怀疑objective mismatch | PULSE、部分平滑/相似映射表现出误差与覆盖分离；但没有逐epoch全量覆盖因果曲线 | 仍是候选假设，**不是已定根因** |
| rescue—lost交换，判断保护缺口或shared干扰 | 修正head误伤；等K20删除尾部导致unique support丢失，离线固定组合无训练 | 组合/保护问题得到直接证据；共享训练干扰尚未独立确证 |
| hard winner不合理，优先改assignment | 已有good时Qwinner通常good；switch总体较小；旧soft/energy/mixture都未突破 | **不优先盲换soft winner**；局部稀疏反馈仍保留 |
| quality与motion冲突，区分weight和参数化 | .1P减少副作用；平滑残差没守全path；强ray趋于identity | 需物理可行且非退化参数化，不能只加大惩罚 |
| grouped成功CAL失败，优先distribution shift | 旧匹配实验有连续误差信号；canonical三G/CAL有小补充；无canonical random控制 | shift/相关性都未被唯一分离，不做单根因声明 |
| 生成稳定安全提高coverage，再scorer | 当前同K20没有全面稳定安全收益；旧新增scorer也负向 | **scorer不是当前优先** |

**当前最靠近的分支：** 新候选有互补性，但候选支撑与安全集成失配；最新安全参数化又没有有效补足变化hole。研究重点应从“生成更大/更多变化”转为“在可行几何中生成原生集合欠缺、且不会损害尾部支撑的补充”。

这是一项更精确的问题定义，不是重大结果已经达成。候选是现有past/native表示的函数；“贡献新的候选支撑”不等于获得了新的现实信息，更不等于心理意图理解。

### 关于恒速外推现象的直接回答

目前不能说已经大幅改善，更不能说模型开始真正理解行人运动。理由：

- 原native20本来就不是单一CV模型；change label也不是独立核验的CV-failure子集。
- K24的新增补充幅度小，等K20变化oracle误差反而受损。
- 最新similarity和temporal mix对主outer change均没有rescue。
- 运动方向、时刻、持续性和速度匹配，以及泛化/安全仍未同时验证。

若专门回答“超越恒速”，需要固定同ID CV参考，事先定义CV-failure subset，分别报告native和新候选在该subset上的同路径覆盖、早中晚同slot误差、rescue/lost与物理质量；当前不能用“change minADE小幅下降”替代这项证据。

## 15. 实验是否存在问题：已发现的问题与可信边界

### 15.1 实质问题

1. **协议归因混杂。** grouping同时改变loss/assignment/diversity/clip，旧比较不能归因pureG。九runcanonical修复算法一致性，但没有补齐matched random，也未控制三G总体组成。
2. **checkpoint协议绑定曾不足。** 仅参数无法还原训练；canonical之后绑定source、IDs、stats、objective、init/optimizer/RNG/history，解决可追溯性缺口。
3. **概率语义错配。** utility soft target不是未来概率；ECE .2439也不支持“可靠校准”的强表达。
4. **预算与支撑混淆。** append K24、union K28、oracle bank不可当等K20收益；low-logit删除不保护marginal support。
5. **旧报告分组/基线口径有漂移。** 例如scene Static change曾出现87/88/89，冻结classifier初始化边界rank使结果不同；oracle某表change452，其他主表211；stable/transition也出现不同分母。只在同来源、同ID/group契约内比较，不拼接为统一统计表。[S19][S20c]
6. **过强机制语言。** “彻底否定简并”“本质不可预测”“已达最高成熟度”“理解意图”等超出有限诊断。本报告保留数据但收紧推论。
7. **强可行性envelope超过研究标准。** zero insertion或ray identity可能来自过强逐点约束，不能据其失败推出pooled-gate下普遍无解。

### 15.2 数值/基础设施问题不应与科学失败混为一谈

已记录并保留来源的类别包括：dataset-only reader错误、保存字段冲突、递归列表读取器、schema维度、metadata计数字段、CPU/GPU统计末位差、报告汇总常量错误、资源STOP与最后stdout JSON序列化失败。

已完成实验的可信性来自各自源锁、ID契约、冻结权重/buffer、预测有限性、保存算术重构、聚簇/pooled运动学审核与明确恢复范围。**本次报告未重新运行全部模型/梯度、未全历史SHA回放**；引用的是既有审计记录。Oct7/8相关锁数量几百至1480，不冒称重新核验全部历史约34k文件。

资源门槛保持process≤6GiB、system physical/commit各reserve≥6GiB；预检峰值不等生产峰值，训练生命周期未保存的峰值不虚构。资源STOP不是自动授权降门槛、修改pagefile或关闭用户程序。

### 15.3 为什么这些仍是探索而不是论文级确认？

只有三scene族/少量recording、三个seed、同轨迹重叠窗口；F16已见原pool；CAL单university且历史多次选checkpoint/复用；研究路线根据结果不断适应性更新。探索cluster CI不控制整个自适应研究的多重比较，也不能把相关窗口/同ID多seed当独立人。

全组/反例已明确报告；但没有真正独立whole-model held-test确认。因此不声称SOTA、独立泛化、不可预测率、decoder唯一根因或可靠部署。

## 16. 下一步建议与当前状态

本次只是报告，**没有启动下一实验、恢复旧worker或设置新的长期窗口**。所有最新完成项已闭环；baseline仍F16，研究目标未宣布完成。

建议分成两条明确目的、不能同时改多因素的路线：

### A. 若优先回答“究竟能否迁移”

在同canonical下做**同outer、同预算、组成可匹配的random-vs-grouped**，共同train统计或明确唯一统计因素，固定epoch primary；分别量化target/time曝光。它补的是旧协议的归因缺口，不是再比三个不同outer百分比。报告CV-failure条件结果可作为预注册分层，不能结果后删难例。

### B. 若优先继续候选生成

先证明一个**非退化、GT-free、原生支撑友好**的可行参数化，在固定K20下具有足够可达的change hole，再决定训练。不得用当前评估GT utility挑partner/delete配置；不得把强31点约束identity、global similarity无时间形状、有限mix主change无rescue改名重复训练。

新实验必须同时报告own-zero、paired control、native、whole-set与新增/保留子池；coverage/rescue-lost、ADE/FDE/J、有效good与五项物理门槛同列。先source/ID/实际GPU/frozen/真实subset两epoch全optimizer-RNG resume与资源预检，再给实测ETA；部署或独立验证需对应具体权限。

**不建议现在做：** scorer优先、盲换softmin/top-k、slot/partner最佳搜索、继续调loss权重/核degree/bound/width/LR/epochs。已有负证据足够停止这些局部循环，尚不足唯一选出“换loss”或“换head”作为根因答案。

## 17. 反方审核与披露

本报告使用 **academic-research-suite** 的证据整合、来源核验与反方检查；由同一助手内联执行，不是外部独立审稿。

- **设计前检查：** 七天窗口固定；分开训练/冻结诊断、append/replace、primary/secondary、原始报告/恢复校正；不以实验数量冒充独立验证。
- **分析后反方：** 最强正面反证是canonical三G/CAL补充、learned4对removed4同K4更好；最强负面反证是等K20变化误差与speed-tail伤害、最新安全路径无主change rescue。两侧都保留，拒绝“完全无用”和“已经安全成功”两种夸大。
- **交付前检查：** 不拼不同slot/outer、校正CAL loss方向与motion失败数、不把utility当概率、不给未知fullFIT覆盖或未保存峰值填数；所有推荐仅设计层，不虚构新启动。

报告用途：组会/项目复盘和下一项设计的证据底稿，不是直接投稿的独立验证结论。每项原科学协议、完整ALLgroup数据和恢复来源以以下文件为准。

## 附录：主要证据与设计入口

引用点击后打开本地原文。S12为早期诊断综合入口，下列列出对应一手报告；旧综合报告中的pending状态已由后续完成记录取代。

| 编号 | 实验/来源 | 本报告用途 |
|---|---|---|
| [S01] | 原生候选选择视觉审计 | 1516窗口生成与选择缺口 |
| [S02]—[S09] | Cycle3—10强制复盘 | 早期输入/结构/有限上游/监督负证据 |
| [S10] | CandidateExchange保存summary | C11完成数值与gate；runtime旧状态不采信 |
| [S11] | 三项响应研究总复盘 | TIME/PULSE/PERSISTENT全部控制与反例 |
| [S12] | 10/05进度综合报告 | cue—GT可达—目标—head学习证据链 |
| [S12a] / [S12b] / [S12c] | observable / low-cue / concordance | 风险与完整未来路径不同 |
| [S12d] / [S12e] / [S12f] | curve / objective / head-learning | GT witness、credit、实际学习边界 |
| [S13] | Candidate-state三项复盘 | state/centering/normalization |
| [S14] / [S15] / [S16] | matched transfer / parameter concordance / loss-credit-motion | 同组FIT/CAL、参数梯度、目标对齐 |
| [S17] / [S18] / [S18a] | physical P / route / physical三项复盘 | 保护、分工、训练投影 |
| [S19]—[S19c] | oracle / 参数识别 / 确定性与MDN / 多解近邻 | 显式程序与不确定性诊断；旧强结论受限 |
| [S20] / [S20b] / [S20c] | Bo4 / P0—P4 / transfer confirmation | 组合互补与阶段门槛 |
| [S20d] / [S20e] / [S20f] | correction / scorer / internal validation | 人工推进的负证据与有限内部验证 |
| [S20g] / [S20h] | 校准接口恢复 / K3替换 | utility语义校正与等K负结果 |
| [S20a] | protocol blocker | 实际源码不一致 |
| [S21] / [S22] | target / time匹配实验 | related exposure信号及CI |
| [S23] / [S23a] | canonical恢复分析 / 原winner switch | 九run统一P与source绑定 |
| [S24] / [S25] / [S26] | equalK20 / tail utility / protected insertion | 支撑损失与安全退路 |
| [S27] / [S28] / [S29] | native residual / Bernstein / 三项总复盘 | 非平滑、平滑、ray可行性 |
| [S30] / [S32] | similarity / temporal mix | 最新安全非零与稀疏见证 |
| [S31] | learning-signal / FIT memorization综合 | 早期训练内可学与CAL损害 |

上述阶段还保留各目录README/DESIGN/protocol、RESULT/summary、checkpoint/run_manifest、ARITHMETIC/FROZEN/PRESERVATION、completion_review等。一项已有恢复目录时，以恢复完成报告及明确引用的父产物为准，不删除父失败。

便于复查的设计入口：[canonical九run设计](<C:/Users/linke/Documents/New project/PROTOCOL_RECONCILIATION_RUN_20261007/DESIGN.md>)、[等K20固定替换设计](<C:/Users/linke/Documents/New project/F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/DESIGN.md>)、[native-anchor设计](<C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_RESIDUAL_SERIES_20261008/DESIGN.md>)、[similarity设计](<C:/Users/linke/Documents/New project/GENERATION_NATIVE_SIMILARITY_FEASIBLE_SERIES_20261008/DESIGN.md>)、[temporal-mix设计](<C:/Users/linke/Documents/New project/F16_NATIVE_TEMPORAL_MIX_REACHABILITY_AUDIT_20261008/save_field_recovery_20261008/DESIGN.md>)。

[S01]: <C:/Users/linke/Documents/New project/CANDIDATE_SELECTION_VISUAL_AUDIT_20261003/REPORT.md>
[S02]: <C:/Users/linke/Documents/New project/GENERATION_CYCLE_3_REVIEW.md>
[S03]: <C:/Users/linke/Documents/New project/GENERATION_CYCLE_4_REVIEW.md>
[S04]: <C:/Users/linke/Documents/New project/GENERATION_CYCLE_5_REVIEW.md>
[S05]: <C:/Users/linke/Documents/New project/GENERATION_TEMPORAL_COMPONENT_SERIES/CYCLE_6_REVIEW.md>
[S06]: <C:/Users/linke/Documents/New project/GENERATION_BASIS_COMPOSITION_SERIES/CYCLE_7_REVIEW.md>
[S07]: <C:/Users/linke/Documents/New project/GENERATION_PATH_MIXTURE_SERIES/CYCLE_8_REVIEW.md>
[S08]: <C:/Users/linke/Documents/New project/GENERATION_CYCLE_9_REVIEW/CYCLE_9_REVIEW.md>
[S09]: <C:/Users/linke/Documents/New project/GENERATION_CYCLE_10_REVIEW/CYCLE_10_REVIEW.md>
[S10]: <C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_EXCHANGE_SERIES/results/summary.json>
[S11]: <C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>
[S12]: <C:/Users/linke/Documents/New project/TRAJECTORY_RESEARCH_PROGRESS_REPORT_20261005/REPORT.md>
[S12a]: <C:/Users/linke/Documents/New project/F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/RESULT_ANALYSIS.md>
[S12b]: <C:/Users/linke/Documents/New project/F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/RESULT_ANALYSIS.md>
[S12c]: <C:/Users/linke/Documents/New project/F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/RESULT_ANALYSIS.md>
[S12d]: <C:/Users/linke/Documents/New project/F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004/RESULT_ANALYSIS.md>
[S12e]: <C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT_ANALYSIS.md>
[S12f]: <C:/Users/linke/Documents/New project/F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>
[S13]: <C:/Users/linke/Documents/New project/GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/CANDIDATE_STATE_STUDY_REVIEW.md>
[S14]: <C:/Users/linke/Documents/New project/F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>
[S15]: <C:/Users/linke/Documents/New project/F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/RESULT_ANALYSIS.md>
[S16]: <C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>
[S17]: <C:/Users/linke/Documents/New project/GENERATION_PHYSICAL_ACCEL_PROTECTION_SERIES_20261005/RESULT_ANALYSIS.md>
[S18]: <C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005/RESULT_ANALYSIS.md>
[S18a]: <C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md>
[S19]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_01A_ORACLE_DECOMPOSITION_20261007/RESULT_ANALYSIS.md>
[S19a]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_02P_PARAMETER_IDENTIFIABILITY_20261007/RESULT_ANALYSIS.md>
[S19b]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_03_LEARNED_PARAM_K24_20261007/RESULT_ANALYSIS.md>
[S19c]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_MOTION_PROGRAM_TARGET_IDENTIFIABILITY_20261007/RESULT_ANALYSIS.md>
[S19d]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_03_MDN_LEARNED_PARAM_K24_20261007/RESULT_ANALYSIS.md>
[S20]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/RESULT_ANALYSIS.md>
[S20a]: <C:/Users/linke/Documents/New project/EXPERIMENT_PROTOCOL_RECONCILIATION_20261007/BLOCKER_AND_RECOMMENDATION.md>
[S20b]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_P0_COMPLEMENTARITY_20261007/RESULT_ANALYSIS.md>
[S20c]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007/RESULT_ANALYSIS.md>
[S20d]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_04B2_SCENE_BALANCED_20261007/RESULT_ANALYSIS.md>
[S20e]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_05B_REPRESENTATION_20261007/RESULT_ANALYSIS.md>
[S20f]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_06_INTERNAL_VALIDATION_20261007/RESULT_ANALYSIS.md>
[S20g]: <C:/Users/linke/Documents/New project/F16_PORTFOLIO_CALIBRATION_REPLAY_AUDIT_20261007/calibration_contract_recovery/RESULT_ANALYSIS.md>
[S20h]: <C:/Users/linke/Documents/New project/F16_K3_SINGLE_REPLACEMENT_AUDIT_20261007/RESULT_ANALYSIS.md>
[S20i]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_P1_ORACLE_8TO4_20261007/RESULT_ANALYSIS.md>
[S20j]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_P2_SLOT_SEMANTICS_20261007/RESULT_ANALYSIS.md>
[S20k]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_P3_FIXED_PORTFOLIO_20261007/RESULT_ANALYSIS.md>
[S20l]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_PORTFOLIO_P4_ADAPTIVE_SELECTOR_20261007/RESULT_ANALYSIS.md>
[S20m]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_04A_PRIMARY_20261007/RESULT_ANALYSIS.md>
[S20n]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_04B1_COMMON_SUPPORT_20261007/RESULT_ANALYSIS.md>
[S20o]: <C:/Users/linke/Documents/New project/AUDIT_STAGE_05P_RANKABILITY_20261007/RESULT_ANALYSIS.md>
[S21]: <C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment1_target_track/RESULT_ANALYSIS.md>
[S22]: <C:/Users/linke/Documents/New project/GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/RESULT_ANALYSIS.md>
[S23]: <C:/Users/linke/Documents/New project/PROTOCOL_RECONCILIATION_RUN_20261007/audit_recovery_20261007/RESULT_ANALYSIS.md>
[S23a]: <C:/Users/linke/Documents/New project/PROTOCOL_RECONCILIATION_RUN_20261007/OLD_WINNER_SWITCH_ANALYSIS.md>
[S24]: <C:/Users/linke/Documents/New project/F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/RESULT_ANALYSIS.md>
[S25]: <C:/Users/linke/Documents/New project/F16_REMOVED_TAIL_UTILITY_AUDIT_20261007/RESULT_ANALYSIS.md>
[S26]: <C:/Users/linke/Documents/New project/F16_NATIVE_SUPPORT_PROTECTED_INSERTION_DIAGNOSIS_20261008/RESULT_ANALYSIS.md>
[S27]: <C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_RESIDUAL_SERIES_20261008/RESULT_ANALYSIS.md>
[S28]: <C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_BERNSTEIN_SERIES_20261008/RESULT_ANALYSIS.md>
[S29]: <C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/NATIVE_ANCHOR_STUDY_REVIEW.md>
[S30]: <C:/Users/linke/Documents/New project/GENERATION_NATIVE_SIMILARITY_FEASIBLE_SERIES_20261008/RESULT_ANALYSIS.md>
[S31]: <C:/Users/linke/Documents/New project/GENERATION_LEARNING_SIGNAL_DIAGNOSIS/ANALYSIS.md>
[S32]: <C:/Users/linke/Documents/New project/F16_NATIVE_TEMPORAL_MIX_REACHABILITY_AUDIT_20261008/save_field_recovery_20261008/RESULT_ANALYSIS.md>
