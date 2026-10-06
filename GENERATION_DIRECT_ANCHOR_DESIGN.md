# 第三周期首项：直接锚点解码参数化（设计中，未开训）

## 先验诊断结果

来源GENERATION_CONDITIONAL_MODE_SERIES/trainability_diagnosis.json：三个seed的constrained/final checkpoint，在fit/cal每个运动组随机固定128例，共36个checkpoint-batch分析，无optimizer step、无dev/test。这是抽样描述，不是全数据或独立复现。

constrained变化组fit minADE .235977→.227313，约3.67%改善；cal .247113→.245148，约.80%。final fit .214325而cal .243064，约9.17%与1.64%改善。训练集内收益较强而校准收益弱，泛化/参数化值得优先检验，不可据此证明过拟合的唯一原因。

constrained变化组winner clip fit .52%、cal .17%，全部候选clip .56%/.63%；final winnerclip1.00%/.78%。winner梯度norm fit1.095/cal.869，regularizer约.0000566/.0000602，常规正则不是这批样本的梯度主导项。保护项在变化组按原协议为0，但stable/transition约.50–.62，与其winner梯度相近，保护与预测确实可耦合。保留保护，不能据此事后削弱guard。GT相对任一native候选的逐步速度差完全可在commonbound内的比例fit77.34%、cal62.50%：低饱和梯度不排除函数可达性限制；37.5%cal抽样变化病例无法在当前界内完全复现GT，不代表无法降低误差，也不授权放宽界。

## 新参数化单因素

比较相同的新直接3锚点MLP（输入128、hidden128/64、输出6），预测4/8/12未来点，原线性插值12步。两组均不再以原proposal_delta/refiner的小通道校准作为可训练主体，而从观察特征直接预测raw anchors。原生整个decoder/F16/scorer权重全部冻结，只训练新网络。

- native-mode-feature控制：concat原观测context64与原context条件化modeMLP输出64。
- query-context处理：concat相同观测context64与原固定mode_query64，绕过原modeMLP表示。

仅输入参数化不同，结构、参数数、预算和数据一致。不同feature分布有效表达能力无法严格相同，不宣称等有效容量。新MLP参数成对初始化，但拟合初始化的最后层会因输入不同而不同，不宣称全状态完全相同。原固定queries仍继承native模式，不宣称全新学习了意图。

原生轨迹不进入raw anchor MLP，但最终仍与native参照做统一速度残差clip±1.5*(speed+.03)并积分。因此最终不是完全独立生成，bound非GT/物理保证。无GT/groups/logits前向，不改变scorer、不作排名结论。

## 初始化工程优先事项

从零头可能大幅伤害基线，需先检查共同的无GT、FIT ONLY初始化：冻结随机前层、最后层对native教师三锚点做固定ridge闭式拟合；用fit全25912例×20模式累计Gram矩阵，不用cal/dev/test/未来GT。初始化目标只是原模型已产生的观察依赖候选，不是成功标准；不根据dev选择teacher或ridge。建议固定ridge=1e-3*trace(Gram)/dimension，留bias处理并检查条件数/有限数。教师归一化采用既有observed_speed+.03。拟合不会保证nativeidentity，要记录两组fit/cal起点偏差并明确不等初始化；任何不稳定必须在正式开训前解决，不能让epoch0隐式fallback伪装新decoder表现。

若闭式初始化可行，把每arm/seed初始状态+scalers+源/样本哈希锁定，再进行6fresh80epoch原监督训练。第一轮实验控制无GT初始化协议与训练预算一致；不混入新loss、模式扩展、时间点数、更大bound、保护系数或主干训练。可行性检查还包括所有输入native观测、原fitcal disjoint、冻结非白名单、输出有限K20、梯度、新head保存/恢复及optimizer/RNG连续EXACT；测试未通过不得启动。

预计正式六训练约25–35分钟仅作设计参考，最终启动前按实测吞吐报告，不把当前未启动的检查算训练时间。使用独立Windows按需task/锁/SHAverifiedskip/exactresume，启动一次确认，计划T的60%单次进度查询；实际剩余cal/IO/报告修正分析时间。新cycle只在正式启动后计1/3，第三项完先总review。

## 成功与限制

所有原major 5%changeADE/FDE、1%保护、3seed一致/native及freshcontrol置信、samepath、coverage/dup/hardmotion和全组速度p95/加速度mean,p95/jerkmean,p95分布1%保护不变。无可行checkpoint必须明示nativefallback，不算新模型成功。final/legacy诊断不可dev选最佳。观察特征冻结，改善失败都不能直接推出encoder足够或不足。三seed和重复ETHdev探索性，首次重大信号再审查场景相关性和不确定性，heldtest/跨fold/解冻F16需要另授权。
