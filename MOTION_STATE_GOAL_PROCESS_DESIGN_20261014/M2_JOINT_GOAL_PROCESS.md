# M2：初始运动状态一致的目标—过程联合学习

版本1.0；DESIGN ONLY。依赖[共同协议](COMMON_PROTOCOL.md)和[M1来源选择规则](M1_DYNAMIC_HISTORY.md)。6个新run；不会续开旧G2三次桥。

## 1. 问题与主要对照

H2：在同一动力学解码器和同一初始预测目标下，允许完整路径损失更新目标输出头，能否比固定目标更好地同时满足路径质量与运动学要求？

| arm | history/context/query | goal输出head | process head |
|---|---|---|---|
| M2-A `frozen_goal` | 冻结 | 冻结 | 可训练 |
| M2-B `joint_goal` | 与A共同冻结 | 可训练 | 与A同构、paired初始化、可训练 |

来源由S1固定：通过用M1-B，未通过用原G1。按seed一一对应复制；两臂起始目标、h、q、全部buffers、process参数必须完全相同。

解冻范围只含原共享goal MLP80→64→2（5314参数），不含context、GRU、query、centres和scalers，更不包含旧F16。B不是整个模型端到端解冻。可训练容量A/B不同，这是目标联合优化方案的一部分，不能称纯粹消除了容量因果。

## 2. 新过程解码器：精确离散定义

坐标原点p0=0；v0=(x7−x6)/dt；H=12，dt=.4。goal g仍是**模型预测值**，训练和推理都不用GT终点。

过程head输入 `[h64,q16,normalized_goal2,normalized_v0_2]`，共84维；MLP84→64→24，中间GELU，最后层zero-init，输出raw acceleration b∈R^(12×2)。神经head为float32，b/g/v0转float64后几何计算。

输入goal和v0的统计都在FIT上固定：goal来自该seed初始目标网络的18909×20预测，v0每窗口仅一次。B训练过程中goal数值可变化，但统计不重拟；不使用每epoch或CAL动态归一化。head输出数值按m/s²解释，无额外tanh、幅值搜索或native小修复界。

积分约定：

\[
v_t=v_{t-1}+\Delta t\,a_t,\qquad p_t=p_{t-1}+\Delta t\,v_t.
\]

令w_t=H-t+1，S=Σw_t²=650，则

\[
p_H=H\Delta t\,v_0+\Delta t^2\sum_{t=1}^{H}w_t a_t.
\]

定义目标所需加速度矩量d=(g−HΔt v0)/Δt²；作最小L2仿射修正：

\[
e=d-\sum_t w_t b_t,\qquad
a_t=b_t+\frac{w_t}{650}e.
\]

由此Σw_t a_t=d，数学上p_H=g。该固定线性投影不读取GT，不是逐样本GT拟合，也不是部署时求解未来真值。所有candidate一次性计算，不筛掉不安全路径来凑K。

等价分解：a_t=(w_t/650)d + [b_t−(w_t/650)Σw_s b_s]。第一项为满足终点的最小平方加速度解，第二项为零终点位移的过程自由度；保存两项以便归因。

## 3. 保证与不保证

保证的是离散状态递推、起始状态v0、预测终点约束；**不保证a1=0、第一未来段速度等于v0、速度非负、无回环、加速度/jerk安全或准确覆盖**。初始速度跳变仍是dt*a1，必须另报。

原有12点路径的11个加速度差分并不包含v0→v1这个新增边界项；科学指标不能为新decoder改口径。额外报告包含边界项的量，但不替换旧门。

b有两个被投影消除的加权方向；过程输出的有效线性维数为22而非24。不能要求投影Jacobian满秩，不能把解析零方向的舍入梯度当有效信号。分母650为固定正常数，无小分母、无epsilon或floor。

不能在投影后clip a/v/p，否则将破坏终点方程；若本版本产生不安全过程，如实失败，不临时添加二次投影或物理loss。它与旧三次桥的历史比较只能描述整体差异，不能独立归因于某一个方程。

## 4. 初始化、训练和参数合同

process head手算参数：(84×64+64)+(64×24+24)=7000。A可训练7000，B可训练12314；实际实例化核验后写入manifest。总参数依共同父表示不同而变化，不能只报新增数冒充全部模型规模。

b=0时两臂初始输出相同，均为由预测g和v0决定的最小平方加速度路径，**不是G1直线、native20或零加速度identity**。初始化P0必须全量保存四split，每seed一份（共12份初始化job，另于24份最终job），不能事后换一个有利的零对照。

A的goal用no_grad，B的冻结encoder/query可no_grad但goal head必须保留计算图，g在head输入标准化和终点投影两条路径均参与梯度。两臂原goal/scalers/centres相同，不设置第二种目标惩罚。

共同canonical完整K20 loss、25epochs、AdamW、batchorder和seed。B允许goal、winner责任及过程随训练共同变化；不是纯credit根因实验。上游目标训练成本计入全pipeline，不宣称和单阶段G1等总算力。

## 5. 主要与机制指标

主要：B−A、B−native的outer change C/minADE/minFDE/J，使用共同26-family校正。所有组/seed/split和五motion完整报告。

专项保存：

- 初始/最终目标g、完整p/v/b/a、d、e、最小平方项及零终点项。
- 目标相对初始的位移与g−4.8v0；按所有candidate报告，不能只看GT winner。
- 端点error、v0边界跳变、raw与投影后加速度/jerk；修正能量只是诊断，不是新loss。
- B−A及各自P0−final的rescue/lost；终点good集合的新增/丢失。
- 同时在E_A∩E_B的共同窗口上报告C差异；再在所有窗口报告主结果。交集分析是post-treatment描述，不能单独因果归因或替代主分母。
- 若目标更靠近CV且motion下降，但endpoint/change coverage下降，明确标记保守化，不称兼容性成功。

## 6. 结果规则与下一项触发

**联合优化机制信号 S2**：B−A的outer change覆盖三seed均>0，平均≥5pp且校正CI下界>0；change minADE/minFDE均值均不劣于A，outer保护三组ADE/FDE每seed≤A的1.01。不要求B−A端点恒等，冻结目标A对初始化的端点必须保持。

**M3的可用候选触发 Q2**：S2通过，且B在outer change覆盖每seed均不低于native；outer overall/stable/transition ADE/FDE每seed≤native1.01；B全部48个seed×split×group五motion通过。

Q2只是允许研究预算分配，不是最终major；最终5%ADE/FDE和native校正区间等仍不放宽。若Q2未通过，不自动跑M3或改acceleration范围。M2-A较好而B未达到S2，应保留A的正证据，但不能据此启动以联合目标为前提的M3。

| 结果 | 结论 |
|---|---|
| 覆盖、目标精度、motion共同改善 | 支持目标—过程联合方案，按Q2判是否进入M3 |
| 覆盖改善，motion失败 | 质量—物理权衡；停止本decoder，不加penalty扫描 |
| 目标更准，完整路径无收益 | 目标调整并未解决过程问题 |
| motion改善但覆盖缩回CV附近 | 保守化，不是突破 |
| A与B都差 | 该结构/当前输入下未获支持，不宣称所有动力学生成不可能 |

## 7. M2专项预检

独立NumPy验证Σw²=650、pH=g、积分一致、零raw初始化、g=4.8v0且b=0时CV、零终点投影方向被消除；固定随机fixture测试非零过程自由度。double fixture核验对b/g的解析梯度和有限差分。实际GPU128窗口验证端点容差、float64保存重算和全部tensor有限。

初始A/B路径完全配对；训练期间A原goal全params/buffers EXACT、B仅声明的goal模块允许变；所有其他父模块EXACT。检查B没有误用detach导致假联合。两臂真实子集两轮全optimizer/RNG恢复与共同预检都必须执行。

最强反驳：联合臂更多可训练参数，并可通过改变目标而非真正更好地生成过程获益。因此必须同时报告终点与过程分解，不能只用一个coverage数字下结论。
