# T2：收益真的来自观测条件吗？

版本1.0；DESIGN ONLY。仅[T1](T1_MOTION_PATTERN_RESPONSIBILITY.md)的S_T1与Q通过后实施。不是默认追加GRU或给历史再一次无条件机会。

## 1. 问题

T1成功仍可能只说明FIT模式与末速度先验很好。H_T2：在相同模式、末速度、模型规模和固定责任下，P2观测表示能带来可迁移的额外完整路径覆盖。

此实验测试P2的**增量预测价值**。P2混合历史和邻居，结果不能拆成某一个信息源的贡献，也不能证明模型理解心理意图。

## 2. 唯一因素与控制

两臂都用T1-B固定FIT责任；同25,560参数MLP、同当前FIT模式/标签/统计、同seed配对、同P0、同25epochs。

| 臂 | 标准化P2 | v0与20模式 | 其余 |
|---|---|---|---|
| reduced_context | 恒置零，即训练统计均值基准 | 正常 | 相同 |
| full_context | 真实冻结P2 | 正常 | 相同 |

置零从训练第一步到评估始终执行，不是只在测试时破坏输入。它是一个缺少额外上下文的模型控制，不是假装真实原始观测干预。v0保留，避免把信息去除与最基本的速度坐标一起破坏。

两臂均fresh训练；full臂不直接复用T1末轮冒充新配对训练。其重复结果须与T1逐savedarray核验并如实标注，不能计为新增独立泛化证据。

zero输入使reduced首层P2对应权重梯度解析为零，这是设计的一部分，不以roundoff通过“所有参数有效梯度”检查。名义参数相同不等于有效容量或优化几何相同；只能归因于“提供P2条件”的整个方案。

## 3. 运行与保存

6runs/150epochs/24finaljobs+共同P0引用。保存head全部参数、模式/scaler buffers、精确zero mask、每mode责任、质量/运动学/救回丢失。预检明确reduced输出不随P2输入变化、full非零权重时可受其影响；这只是连通性，不是科学收益。

loss、数据、optimizer、生产预算与审计完全按[共同协议](COMMON_PROTOCOL.md)。没有新encoder训练、历史长度标签、邻居合成、新分类器或参数搜索。

## 4. 主比较与触发

主family已预留full−reduced/full−P0/full−native的C/ADE/FDE/J共12项。

S_T2：outer change full−reduced覆盖三seed均正，平均≥2个百分点、校正区间下界>0；full−P0三seed覆盖净增加且校正区间下界>0；full change ADE/FDE均值不劣于reduced，outer保护三组每seed≤reduced1.01。

2pp是本新信息价值问题的事前最低研究信号，不是替换T1的5pp，也不是降低最终major。S_T2加共同Q门通过才进入T3。

| 结果 | 解释 |
|---|---|
| full/reduced相近，CI跨0 | 尚无可靠增量信息证据，不证明P2完全无用 |
| reduced本身很好，full无增量 | 原型与v0足以解释当前成绩；不能称理解历史/交互 |
| full FIT优但outer不优 | 条件拟合仍未稳定迁移 |
| full可靠优且安全 | 支持P2条件有增量价值；随后检验跨分布稳健性 |

本阶段不追加测试时shuffle来代替训练控制；若将来需要解释某类历史/邻居的作用，必须另作原始观测一致干预设计，不能编辑cached token后声称真实社会反事实。
