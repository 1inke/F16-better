# 固定原模型的表达可行证据分类

## 问题和批准范围

现有脉冲曲线在不变native20与观测速度bound下能否构造同一完整好候选？若能，实际学习候选是否达到？这是二级数据、数值构造和只读重放，不是可部署预测或新训练。未来GT只oracle目标；真实forward无GT。所有FIT/CAL窗口保留，DEV不构造、不调整预算，不扩数据/解冻/scorer权限。参考上一recovery的completion_review/NEXT_DESIGN.md与原pulse/run_head.py。

## 真实函数与数值预算

原函数逐坐标box limit L=1.5*(原speed+.03)，不是欧氏范数clip。原tanh漂移与tanh向量幅度分别*.5L；Gaussian在12个midtime上减均值再Linf归一。raw在原box内，累计修正，endpoint=6L*d，d=tanh(D)。c=.05+.90sigmoid(clock0),s=.05+.45sigmoid(clock1)。本构造参数(d2,a2,c,s)保持可逆到有限原codes的内点margin1e-6，生产torch float32重算并留2e-5m质量margin后才认witness。

全部slot先零修正+16固定profile、多种endpoint/线性回归初始化，保留按同slot max(ADE/.2,FDE/.4)最好构造；不能拼两个slot的指标。所有尚未严格好且必要下界未排除slot，确定性最好两初始化各32轮vectorized damped Gauss-Newton，解析d/a梯度和c/s中心差分；endpoint平方权重3固定，line-search1/.5/.25。预算不因CAL表现调节，收敛/局部失败仅未知。已有好slot的zero或成功构造可停止，因为问题是找证据，不声明全局最优。全20slot保留并存最佳有限codes/profile/path。

同bound逐步box是superset：逐坐标残差increment<=L，贪心跟随GT及64轮投影梯度凸平方误差作为可行构造，不称精确最优。必要下界每时刻distance_to_box(GT-base,[-tL,tL])的norm，平均为ADE下界、末时为FDE下界；仅全部slot任一必要指标超过阈值+2e-5才certified wider-bound infeasible。还给pulse端点更窄[-6L,6L]的必要FDE下界，但不把未过其他拟合视为pulse不可达。

## 分类和统计

保留原native coverage；若pulse构造成功而某seed当前无好候选，叫“已证明表达可行、实际学习未达到”。pulse未找到而superset成功叫“pulse未证实、较宽函数有构造证据”，不是pulse必定受限。必要box下界排除叫“当前bound不能达到严格标准”，不是未来不可预测。剩余未知。模型three seed重复同窗口不可加成独立样本；无groupCI/p值。报告全体/clear/noevent/boundary/specific-first-compatible/coarse-path-failure/原source scene/type、native/每seed/pulse/superset覆盖及rescue/lost，GT无关complete-link去重有效好数量；仅描述GT拟合，不改变benchmark。

## 审核与资源

固定全FIT/CAL输入对应与split；只读三模型逐参数buffer前后exact，原decoder对baseline exact；源/旧全部SHA前后不变。生产torch路径与独立numpy公式/质量/clip/endpoint/profile/几何下界合成反例、Jacobian差分、实际模型components对齐；all saved blocks逐项重算，实际块数由64rows决定，done SHA/source不符拒绝，部分块失败完整来源保留后原预算重算，非optimizer/RNG训练resume。

约17GB可用hostRAM不全占。输入预估<.5GiB numpy+旧torch环境常量；full input+one64chunk+3replay outputs峰值实测才launch。预算6GiB，physical/commit各余量6GiB；不改pagefile/停止其他程序。FIT256固定跨段吞吐计时只ETA，丢弃preflight构造结果；ETA含两全旧hash/载入/三seed只读重放/全部拟合块/算术与dedup。独立worker字节锁/按需0trigger/一次真实启动；单次50%TIME及修正完成节点后退出，不轮询。

## 反方检查1（同助手inline，非外审）

PASS_WITH_LIMITATIONS：最大风险是把oracle局部失败当函数不可能、把oracle收益当模型收益、把“表达可行”当学习失败唯一机制。以上都禁止。旧3anchorGT可达、freeincrements/timequery负结果、tinyFIT好CAL差、pulse严格coverage下降反证保留。单脉冲拟合成功不证明已用线索/语义意图，GT高度知情选择所有20slot不具部署意义。只有3uneven scenes、重叠window/track、F16已见训练数据、CAL单university、future-completeness邻居资格偏差；无独立泛化或可靠groupCI。secondary已有公开坐标的内部再分析、无新参与者/身份发布，无新增IRB批准声称。方法是本地预保存探索协议，不是公开preregistration。AI辅助实现/审查披露，完成时补inline反方2/3。

遇到实质源锁/资源/映射/不可辨识方法/新权限阻碍即STOP等人类；不无限coeff/rank/epochs变体。完成分类后不自动第三训练。
