# R3：固定非线性表示的窗口内排序目标对照

2026-10-13。状态：**设计与核心代码交付，未启动真实实验**。本轮不是轨迹生成实验，不进入 B 或 S2。

## 1. 要回答的问题与依据

在完全固定 R2 的 200 维表示、R1 的 P/N/U 标签和原三折 target-track OOF 划分后，将逐候选二分类改为同窗口 P>N 成对排序，能否提高 top4 选中至少一个安全可修复锚点的概率？

依据：R2 原报告及其 `TOP4_FAILURE_ANALYSIS_20261013.md`。同 263 个已发现 P 的窗口，RBF Hit4=77/263=29.28%，linear=66/263=25.10%，固定 random=74/263=28.14%。RBF 相对 linear 是新增38、丢失27、净增11，并非无损提升。186个漏选中169个top4全为已证N；change为53/58。直接由U占据top4不能解释大多数漏选，但不能排除未知标签机制的间接偏差。训练Hit4约41.44%、OOF29.28%，既有拟合不足也有迁移差距，不能预先断定loss是唯一原因。

来源目录：`../R2_NONLINEAR_MATCHING_DESIGN_20261012/production_20261013/report_recovery_20261013/`。

## 2. 唯一实验因素

| 项目 | pointwise 对照 | pairwise 实验 |
|---|---|---|
| 每折表示 | 原R2冻结136+64RBF=200维 | 完全相同的数组与变换 |
| 分数 | w·z+b | w·z，公共b固定0 |
| 监督 | 每窗口P/N各半的BCE | 每窗口全部P–N组合的logistic排序损失 |
| track权重 | 每track等权，其内窗口等权 | 相同 |
| 正则/求解 | C=1，L2；Newton-Cholesky，tol=1e-6，max_iter=500，CPU单线程 | 相同，截距不可识别所以固定0 |
| 输出 | 对全部20个候选打分，稳定排序取4 | 相同，不用GT/PNU过滤候选 |

两臂共 **2目标×3原OOF折=6次小型凸拟合**；这不是3模型种子、不是6个独立数据集。旧R2模型只用于复现校验，不warm-start。没有随机重试、额外seed、C/温度/中心/带宽/特征数扫描。

## 3. 数据与冻结约定

- 原A384窗口，stable/transition/change各128；原329P/7027N/324U、263个known-P窗口、115条known-P轨迹。保留所有384个验证窗口，主指标只在固定263窗口计算。
- 原三折不变；共同训练窗口必须含P和N，训练数177/180/169；验证known-P数86/83/94、tracks40/37/38。保持原56条discovery轨迹排除。训练和验证track不交叉。
- P为原界内安全修复见证，N为原界内保守不可行证书，U未知。**不重新求解标签，不把U当N，不重筛窗口**。
- 直接读取每折R2已保存的mean/std、64中心及RBF标准化数组。只使用该折训练窗口得到的统计；预检须重构一次并逐数组核验，不能用验证集重拟统计。中心ID必须属于对应训练ID。
- 所有旧源、标签、模型、结果、失败记录不可修改；入口SHA不是完整继承来源审核。未来执行时继承R2现有3244相关来源及结果锁，不覆盖原digest，也不称全历史34k核验。
- 不加载F16/原decoder/scorer、不新生成轨迹、不读取B/新DEV/test/fold/外部数据。GT标签只用于拟合/评估，预测接口仅接受固定特征和模型参数。

## 4. 精确损失与权重

设该折训练有n窗口、T条track，窗口i所属track有n_t个训练窗口：a_i=1/(T n_t)，S=20n，lambda=1/S。P_i、N_i为该窗口已知标签集合。

逐候选对照：

    L_BCE = sum_i a_i [mean_p softplus(-s_ip)/2
                       + mean_n softplus(s_in)/2] + ||w||²/(2S)

窗口内排序：

    L_pair = sum_i a_i mean_(p,n) softplus(-(s_ip-s_in)) + ||w||²/(2S)

使用全部P×N组合，不做hard-negative mining、topk重加权、U伪标签或margin调参。每个窗口的总权重相同于对照；pair数量不改变track/window权重。实现把d=z_p-z_n和-d分别标1/0，各占一半权重，权重总和S，fit_intercept=False。这与上式严格等价，不是生成两份独立样本。

pointwise保留原R2权重计算顺序，优先逐bit复现。两臂都使用相同lambda，但损失曲率、梯度和有效正则作用仍会改变；结论只能归于这一目标替换，不能称纯粹credit根因证明。

公共截距在差值中抵消；同窗口共享的100维历史加性特征也抵消，在pair目标下只受L2作用。**历史与候选的非线性交互仍由64RBF等候选相关项表达**。200个系数不等于相同有效自由度，必须报告历史共享系数与RBF系数范数，不能把差值中的零梯度误报模型断连。

## 5. 评价与不放宽的筛选门

主指标：同263known-P窗口的OOF Hit4，候选并列按原physical slot稳定排序，输出4个唯一slot。预测时U也参与排序。

固定5个比较对象：本次pointwise RBF、原linear、原envelope、原fixed-random、原bottom。主要科学问题是pairwise−pointwise；继续保留原四参考，不能仅打败bottom便宣布成功。

预注册探索性进入下一阶段的门（全部同时满足）：

1. 原每折支持门：至少20个P窗口、10条P轨迹，标签/划分/来源审核通过。
2. 相对上述每个参考，Hit4绝对提升至少5个百分点，配对track bootstrap下界>0。
3. 五比较Bonferroni，总alpha=.05，各**99%**区间；2000次整track重采样；五比较共享同一抽样表，rng=20261013；主估计仍窗口加权，不改为track均值。
4. 每个原折change命中数不低于全部五个参考。change分母保持21/25/31。
5. pointwise控制复现、数值/源码/ID/输出审计完整通过。

99%代替R2的98.75%是新增第五比较后的预先校正，不改变旧R2结论。2000重采样尾部精度有限，不做p值或正式确认性声明；不随结果增加抽样、换CI或删比较。Bootstrap不重训模型、不包含训练不确定性；三折训练集重叠、历史多轮适应性探索均须披露。

必报辅指标：每group/fold/scene命中及分母；新增命中/丢失；best-P rank；P>N与ties；top4漏选全N/含U；所有384的已证下界/未知上界/所选U比例；训练与OOF差距。Hit1/2/8可作预先标记的描述性曲线，**Hit4始终primary，不按曲线挑K**。空分组用null+N=0，禁止NaN假JSON或删除整个验证窗口。

这些是锚点选择指标，不是实际repair后coverage/minADE/FDE、运动安全或理解意图。原生成器same-candidate ADE<=.2且FDE<=.4、change ADE/FDE各5%、protected1%、五运动学native1.01等门全部保留，但本实验不产生新轨迹，不能据本轮Hit4判它们PASS。

## 6. 实施顺序与真实预检要求（本次未执行）

1. 锁新目录方案/源码/环境版本；继承旧来源与输出SHA；入口、384ID顺序、原fold/排除轨迹、PNU数、共同训练/验证ID精确匹配。
2. 重建训练专属统计/64中心，和原R2模型逐数组一致；两臂输入特征逐bit一致。验证shared100维确实在窗口内一致，差值为0。
3. 合成fixture检查loss/gradient/Hessian、权重、差值/对称表示等价、U屏蔽及预测不接收标签；真实小subset只用于数值预检，不选择超参。
4. 三折pointwise新拟合必须重现原RBF的系数/截距/完整scores与top4；严格失败则STOP，保存差异，不擅换容差。固定环境优先bitwise；新环境需事前单独数值说明，不能事后迁就结果。
5. pairwise独立NumPy目标梯度inf<=1e-6、全部有限、迭代<500且无任何warning；否则STOP，不自动换求解器/增加迭代/C。共享历史零方向是数学性质，不要求正梯度。
6. 实测单次求解+全部保存/CPU审核资源与耗时。规划>30分钟不启动；process<=6GiB，系统physical及commit各reserve6GiB。有限pairs最多n×100，对称行最多n×200；不改变OS/pagefile/用户进程。
7. 通过后方可单一OS任务运行；新模型/manifest/契约保存逐折doneSHA，完整兼容才能skip；小型凸fit中断不冒称optimizer/RNG精确resume。实际启动、一次half、一次completion的流程仅在以后执行授权与全预检通过时建立。
8. 独立saved CPU核验：六模型featuremap/系数/分数、标签使用与权重、pairs归属、全部OOF排名/指标、5×2000draws与99%CI、所有source/output前后SHA。该审核不重新拟合，不是外部评审。全通过才写RESULT_ANALYSIS，不默认扩B/S2。

## 7. 每折最少产物

`model.npz`（完整固定map、coef、intercept）、`run_manifest.json`、`fit_contract.npz`（训练/验证IDs、labels、track、原weights或pair_indices/weights）、`predictions.npz`（显式ID、全部20scores、top4）、`metrics.json`、`assignment_stats.json`、`done.json`（全部输出SHA）。全局另存OOF、bootstrap draws、source manifest、resource、arithmeticaudit及报告。

manifest至少绑定：自身源码SHA、继承来源、splitIDs/标签/统计SHA、R2模型SHA、200维feature顺序、objective及pair公式、S/C/track权重、intercept gauge、solver/版本/线程、收敛stationarity、训练/验证ID、无GT预测输入。每fold不得只保存系数而丢失协议。

## 8. 结果分支

- **全门通过**：支持固定表示下该排序目标有用。下一步先精确设计独立B迁移验证；不能直接声称S2可学或自动训练生成器。
- **训练与OOF都改善，但未过门**：有排序信号，证据不足；报告效果与区间，不扫描C/温度/损失变体补过门。
- **训练改善、OOF不改善**：优先考虑迁移/标签发现机制，不能继续靠更长拟合或更宽head掩盖问题。
- **训练本身也不能提高Hit4**：当前表示/可分性或all-pairs与any-positive-top4目标不匹配仍可能存在。只否定这一个pairwise测试，不证明所有排序方法无效。
- **U率上升**：检查已证命中与未知上界，不将U视作新成功。

## 9. 内联反方审查（同助手，非外审）

检查1，Major：all-pairs优化所有P>N，不直接优化“前4至少一个P”；已保留Hit4主门而非用pair准确率替代。

检查2，Major：90.86%全N漏选并不证明loss根因；可分性、轨迹间迁移、P发现偏差都可解释。用固定表示/标签对照限制结论，保留R2 transition退步与random反例。

检查3，Major：同一A样本反复探索、条件knownP及相关轨迹的CI不能变成独立确认。通过只允许下一步验证设计，不宣布模型理解运动。截距和shared-history抵消已显式解释，不冒称等有效容量。

本文件和代码由同一助手整理；本次仅合成代码测试，不是真实数据预检、运行结果或外部同行评审。
