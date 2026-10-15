# M1：显式运动历史的增量价值

版本1.0；DESIGN ONLY。前置：[共同协议](COMMON_PROTOCOL.md)。6个新run，150epoch，不启动旧G1/G2。

## 1. 问题、假设与可识别范围

H1：在相同P2、相同新增网络容量及相同目标生成器下，输入真实观测速度/加速度序列，比只输入末速度带来更好的预测终点与完整候选。

P2已可能包含时间信息。本项只识别“给新增支路提供更多动态信息”的效果，不能证明P2必然丢失信息、不能单独区分速度水平/加速度/顺序三者，也不能证明整个模型需要完整历史。

## 2. 两个实验臂

| 名称 | motion支路 | P2与目标网络 |
|---|---|---|
| M1-A `last_velocity` | 7个token均为最后速度；加速度为物理零；相同mask布局 | 与B相同 |
| M1-B `dynamic_history` | 7个真实差分速度与6个有效加速度，按时间顺序 | 与A相同 |

旧G1只作次级历史参考，不能用旧G1替代A：新增支路和参数量不同。两臂必须fresh、paired，不从一个已完成训练的head暖启动。

## 3. 输入张量和统计

令raw过去坐标为x0…x7：v_t=(x_t−x_(t−1))/.4，t=1…7；a_t=(v_t−v_(t−1))/.4，t=2…7。

每token五维 `[vx,vy,ax,ay,acc_valid]`，共[B,7,5]。第一个token的a无定义，归一化后填0、mask0；其余mask1。A和B的mask完全相同，不泄露arm。

两臂共享由FIT真实序列计算的速度2维、加速度2维mean/std（无效a不计）；float32物理特征，float64统计，std<1e-6替换为1。mask不标准化。

**A的有效a=物理0，必须经过同一mean/std变换；不能直接将标准化后的a置零冒充物理零。** A的每个v为该窗口最后速度，经同一速度统计归一化。P2沿用同一FIT统计。

全部统计保存精确ID、有效元素数、dtype、算法及SHA；outer/inner/CAL不参与。低速窗口不丢弃，不计算不稳定的单位朝向，也不从未来补首个加速度。

## 4. 精确模型定义

- motion encoder：单层GRU(input5,hidden32,bias=True,batch_first=True,bidirectional=False,dropout=0)，取第7步hidden32。
- 与P2归一化96拼接成128；context MLP128→128→64，两层后均GELU。
- query：[20,16]，Normal(0,.02)，paired。
- goal head：MLP80→64→2，中间GELU；末层weight/bias零初始化。
- 目标：g_k=4.8*v_last+c_k+head([h64,q_k16])。
- 路径：p_kt=(t/12)*g_k，t=1…12。
- c_k严格复用当前FIT的既有G1 Kmeans20常量，读取原source/ID/statistics兼容后使用；不重新挑中心，不从outer重拟。

手算总参数34,146（GRU3744＋context24768＋query320＋head5314），两臂相同；以实现时实际named_parameters计数为准，不把手算当GPU预检结果。全部新网络可训练，旧F16不可训练。两臂初始路径应逐数组相同，均为prototype20；初始GRU/前层因zero-last而零梯度是正常的分阶段连接现象。

完整loss、优化器和25epoch均见共同协议；不用专门终点loss，也不同时加过程模块。

## 5. 必须保存的诊断

除共同24个最终saved jobs外：

- 每窗口20goals、20paths、末速度、endpoint_good与samepath_good masks。
- FIT全量与三个评价split的终点覆盖、minFDE、完整覆盖、J、五motion和有效good簇。
- mode winner直方图、20终点距离/簇统计，检查目标塌缩。
- B−A和B−native rescue/lost；原组别完整分母，失败窗口不删。
- 首批固定hash128窗口的raw特征、统计前后数值、输出，用于复现，不按效果选例。

不新增“停走/转弯”标签阈值作为主标准；可在既有且源可验证的行为标签上作附录。无可靠标签则只保留原三组，不事后看结果划组。

## 6. 决策规则

新增的**目标表示研究门 S1**（不是major、安全通过或理解证明）：

1. outer change endpoint coverage的B−A三seed均>0，均值≥5pp。
2. 同一差值的26-family校正CI下界>0。
3. B的change minFDE每seed不劣于A；overall/stable/transition的ADE/FDE每seed均不比A差超过1%。

完整覆盖、native终点覆盖和所有motion仍必须完整报告，即使S1通过。若S1通过但B完整路径不安全，结论只能是“有可转用的目标表示信号”，不得将B部署/升级；M2研究的是新解码器，而不是继续旧不安全直线版本。

M2初始化来源在这里固定：S1全通过→三个seed各自M1-B；否则→对应已完成G1。不选最佳seed，不选择M1-A或按CAL结果切换，不把当前outer当全新确认集。即使S1阴性，M2仍可独立研究结构问题。

| 观察 | 解释与行动 |
|---|---|
| 终点和完整覆盖共同改善 | 保留显式动态表示的候选资格，是否安全另判 |
| 终点改善但完整覆盖不改善 | 过程转换仍是瓶颈之一，按固定来源进入M2 |
| FIT好而outer无稳定改善 | 不能加大GRU或延长训练；保留迁移失败结论 |
| 无增量信号 | 停止本表示版本，M2回退G1，不扫输入字段/层数 |

S1为新研究的资源选择规则，不修改旧G2 continuation FAIL；它不能支持“目标已经优于native”。最终major仍是共同协议的全部门。

## 7. M1专项预检

- 原raw末速度与G1逐ID一致；A有效加速度确为物理0；无效a不参与统计。
- P2、中心、stats、两臂参数/buffer完全配对；zero-init路径相同。
- GRU从第一个有效训练更新之后的梯度真实有限连接；冻结旧模块无更新。
- 对改变GT/group/ID但保持合法forward输入的fixture，输出不变。
- 真正FIT513+inner12子集两轮exact-resume；不是仅保存/读取网络权重。
- 不要求B胜A的预检样本结果；预检不得成为效果筛选。

最强反驳：改善仍可能是“额外信息更容易被读出”，不是原模型没有这些信息。最终报告必须保留这个限定。
