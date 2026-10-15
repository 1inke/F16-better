# M3：固定K20下的目标—过程预算分配

版本1.0；**CONDITIONAL DESIGN ONLY**。前置：[共同协议](COMMON_PROTOCOL.md)和[M2的Q2触发条件](M2_JOINT_GOAL_PROCESS.md)。Q2不通过时本项不实施，不能为了凑满系列自动启动。

## 1. 假设与解释边界

H3：在可用的目标/过程生成底座上，将部分候选预算用于同一目标的不同到达过程，可能比把全部预算用于不同目标更能覆盖完整运动。

这是一项**预算分配的结构对照**。同时减少目标数量、增加每目标过程数量，其作用不可被称为“纯时间多样性”或“等有效容量”。不赋予query停止/左转/意图等真实语义，不新增事件监督或新assignment。

## 2. 两个实验臂

| arm | 目标query数量 | 每目标过程 | K | candidate索引 |
|---|---:|---:|---:|---|
| M3-A `goal20_process1` | 20 | 1 | 20 | k=0…19 |
| M3-B `goal5_process4` | 5 | 4 | 20 | k=4j+m，j=0…4，m=0…3 |

B的同组4个candidate必须具有完全同一个预测目标g_j，不能暗中为每过程增加endpoint offset恢复成20目标。A/B都预测全部20候选，不按GT挑目标/过程，不扩K80后筛20。

## 3. 模型与初始化

采用M2所选共同history encoder，逐seed冻结同一份h64及统计。因为M1/G1的encoder是在同一个FIT上训练，允许作为共同初始化；成本和来源必须累计。不得将M2不同arm各自的最终goal/head拿来分别初始化M3两侧。

两臂均fresh训练以下模块：

- goal queries qg：16维；A20个、B5个，Normal(0,.02)。公共前5个paired，其余按组件独立RNG生成。
- 共享goal head80→64→2 GELU，最后层全零，两臂完整paired。
- process queries qp：每个candidate各16维，两臂均20个、同初值Normal(0,.02)。不是B四个共享query反复复制，也不是A无query。
- process head输入[h64,qg16,qp16,normalized_g2,normalized_v0_2]共100维，100→64→24 GELU，两臂完整paired。
- process末层weight使用固定Normal(0,.001)、bias0。此设定用于打破同目标过程的初始完全对称；不通过GT或效果挑初始化，不扫描幅度。相对M2改变了process query接口，因此不能把M3-A直接当成M2-B的精确复现。
- 过程投影和积分逐项采用M2的w/S=650方程，保证数学上的pH=g；没有额外运动学惩罚或事后clip。

目标为g_j=4.8v0+c_j+Δg(h,qg_j)。A的K20中心复用FIT既有G1中心；B的K5中心只对同一FIT endpoint−CV残差，用Kmeans++、seed20261014、n_init1、max_iter100、tol1e-4固定一次。不同目标数导致不同初始化几何，是预算分配方案的一部分，不能假称初始coverage相同。

为避免输入统计另引变因，两臂process goal标准化均使用**共同FIT的既有K20 prototype目标统计**，不分别用5和20中心重拟；v0和h统计也共同冻结。B的不同目标分布不得触发重新拟合或删除异常候选。

手算可训练参数：goal head5314＋process head8024＋process queries320＋goal queries（A320/B80），A13978、B13738，差240。冻结encoder额外报告。**不得添加无效dummy参数制造等参声明。** 所有精确数量待真实实例化核验。

## 4. 训练与保存

每臂3seed、25epoch，共6runs，使用共同完整K20 canonical loss，包括全部190候选对。h冻结，goal/process heads及两类query同时训练；centres/统计固定。没有先给B额外训练goal5的隐藏预算。

最后25epoch为主；initial goals/paths/有效簇也保存，报告初始不同，而非只比较相对一个人为较弱初态的增益。

最终24科学jobs，加逐armseed初始化24份split数据单列。保存明确的goal-index/process-index映射、每goal占用率、每candidate winner、goal和path成对距离、同goal内路径差异、所有共同指标及几何重构字段。

## 5. 必须回答的四个问题

1. **完整路径是否更好？** outer change的B−A覆盖/minADE/minFDE/J，26-family校正。
2. **是否牺牲了去向支撑？** endpoint coverage E、endpoint rescue/lost，5目标与20目标的遗漏；不是只看路径平均误差。
3. **四种过程是否真的不同且有用？** 同目标四路径的GT-free距离/簇、有效good簇、winner占用；微小扰动或复制不算新增模式。
4. **是否用更大速度/加速度换覆盖？** 全组五motion和各query贡献；不只展示一条平滑winner。

同目标四条路径应有相同FDE（数学约束，允许规定几何舍入核验），ADE可以不同。额外报告给定至少一个good endpoint的窗口中完整成功率，同时给出原始分子/分母和E_A∩E_B共同窗口；条件分母变化不能代替主指标。

## 6. 继续与停止规则

研究信号S3：B−A outer change覆盖三seed均>0、均值≥5pp、校正CI下界>0；change minADE/minFDE均值均不劣于A；outer保护三组ADE/FDE每seed≤native1.01；B每seed的outer change覆盖不低于native；全部48组五motion通过。

- S3通过：支持这一个5×4预算方案，可进入V，不证明所有factorization更好。
- 终点覆盖明显减少且完整覆盖没有补回：保留20×1方向，不搜索10×2、4×5等新组合。
- 多过程塌缩：报告当前objective/参数化未建立有效分工，不自动添加语义标签或soft assignment。
- 覆盖增长但motion失败：停止，不增加过滤器使K缩水，不改变阈值求PASS。

最终major仍按共同协议全部门，不以S3替代。M1/M2/M3结束后先总review，没有自动第四个局部版本。

## 7. 专项预检与最强反驳

验证A20/B5个实际目标、B每目标4paths且没有额外endpoint自由度；projection/units/indices和所有190pairs正确；两臂所有声明可训练模块实际连接；zero/非zero fixtures不读GT；实际保存metric/SHA/resume/resource共同预检真实通过后才可能启动。

最强反驳：差异可能来自更少目标、初始化中心变化和不同分工几何，而非模型理解了“先停后走”。因此最终结论限定为预算配置效果，行为语义只作事后描述。
