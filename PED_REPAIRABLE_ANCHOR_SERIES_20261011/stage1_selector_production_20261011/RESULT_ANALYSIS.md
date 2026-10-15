# 可修复锚点选择：数值修复与S1完整结果

日期：2026-10-11。最新人类指示：“自己处理这个问题”。

## 一句话结论

收敛问题已经自行修复，完整S1三个OOF折和独立保存算术审核已完成。选择器明显胜过旧bottom4，但没有胜过固定随机4，因此没有通过预注册S1门，不进入S2生成器训练。

这不是再次卡在数值实现，也不是证明锚点方向无效；当前失败是固定线性选择器的验证性能不足。全程没有神经模型前向/反向、Torch optimizer或旧模型重训。

## 1. 修复了什么，什么没变

原513窗口/10260候选行的lbfgs达到500迭代仍未收敛。新目录改用已安装sklearn1.4.1.post1的Newton-Cholesky；仍是同一个balanced L2逻辑回归目标，C=1、tol=1e-6、max_iter=500、单线程。120维特征、float64训练统计/标准化、身份划分、标签、性能/资源门全部不变，未增加迭代上限、放宽容差、改C、扫描solver或改变模型容量。

目标F=sum_i a_i[logaddexp(0,z_i)-y_i z_i]/S + ||w||²/(2CS)，a_i=n/(2n_class)、S=sum a_i，intercept不正则。安装源码对两solver均使用1/(C S)正则系数；软件SHA已绑定。[官方1.4说明](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html)支持该solver用于样本远多于特征的L2问题。

实际513预检4次Newton迭代收敛、无warning；两次拟合参数/统计/概率BITWISE一致。独立trust-exact参考仅为数值审核，未用于选择锚点或参数：
- 两解目标差2.499267459654675e-11。
- Newton梯度∞范数6.101553828166023e-7，满足原1e-6；参考1.4952657690618534e-11。
- 概率最大差3.2036510637212245e-5；独立梯度/Hessian方向有限差分误差2.3966e-11/3.1045e-11。
- Hessian条件数125220.206，说明存在较不均匀曲率，但不能独断这是原慢收敛的唯一原因。

随后修复两个审核/估时接口：原motion函数须由caller传float64，因此只转换审核副本，不改float32缓存；原估时公式将整个预处理加拟合时间按500步骤放大，得到91分钟，改用固定1033训练子集实际全拟合时间和3倍安全系数估时。失败源、smoke与错误公式全部保留在前序新目录；没有重写旧hash，没有用OOF收益选择这些修复。

1033窗口固定benchmark实测4次迭代，solver0.4791s、含统计/拟合0.6731s，未保存benchmark模型或质量结果。生产规划5min/ETA3–9min，不是保证；实际运行更短，以下给真实时间。

## 2. 固定实验设计

现有target-grouped训练18909窗口，按sequence+pedestrian固定trackhash拆A7334/281轨迹与B11575/408轨迹，未重抽样。A三个OOF验证折2456/2543/2335窗口，含潜力正例轨迹85/95/89；训练折4878/4791/4999。各fold只以自身训练窗口计算均值/标准差并拟合，验证行人完全排除。

输入为观测P2 mean8tokens96维+immutable native完整12×2路径24维，共120维。GT只用于离线标签/验证，group/logit/ID/scene不是selector forward输入。bottom4使用旧冻结native logits，固定random4按预注册hash选，不换随机种子。

标签：某条原生候选本身不满足同路径ADE≤.2且FDE≤.4，但在r_t=.20(t/12)^2理想位置包络的三角不等式下界中可能满足。它是乐观“可修复必要条件”，不是物理可行性、真实覆盖或未来概率。

H4分母为各组**原生20候选全部失败的窗口**；分子为选择4条中至少1条命中该乐观潜力标签的窗口。同一候选算ADE/FDE，不拼不同候选。

原门：OOF overall相对bottom4和固定random4均至少+5个百分点，且两个paired-track97.5%CI下界>0；每fold change不得劣于bottom。只有全部通过才拟合完整A并检验B迁移，B门通过再冻结后续策略/进入S2。

## 3. 全部OOF结果

| 组 | 全窗口 | 原生失败分母 | selected：命中 / H4 | bottom4：命中 / H4 | random4：命中 / H4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 7334 | 3205 | 432 / 13.479% | 51 / 1.591% | 451 / 14.072% |
| stable | 4089 | 1326 | 223 / 16.817% | 12 / 0.905% | 227 / 17.119% |
| transition | 1353 | 775 | 57 / 7.355% | 14 / 1.806% | 100 / 12.903% |
| change | 1892 | 1104 | 152 / 13.768% | 25 / 2.264% | 124 / 11.232% |
| scene_hotel | 218 | 52 | 10 / 19.231% | 0 / 0.000% | 7 / 13.462% |
| scene_university | 5433 | 2546 | 292 / 11.469% | 45 / 1.767% | 346 / 13.590% |
| scene_zara | 1683 | 607 | 130 / 21.417% | 6 / 0.988% | 98 / 16.145% |

主总体：selected432/3205=13.479%，bottom51/3205=1.591%，random451/3205=14.072%。全20锚点有1650个乐观机会窗口，selected只命中其中26.182%，随机命中27.333%；这些仍不是实际轨迹rescue。

| 三折 | overall selected/bottom/random命中数 | change selected/bottom/random命中数 |
| --- | --- | --- |
| 0 | 143 / 12 / 146 | 39 / 4 / 42 |
| 1 | 167 / 20 / 161 | 75 / 12 / 48 |
| 2 | 122 / 19 / 144 | 38 / 9 / 34 |

不能只看第二折。每折change优于bottom门通过；总体相对随机至少+5pp且CI正的主门失败。

## 4. 预注册paired-track置信区间

每比较2000次track bootstrap，OOF fold不是三次模型seed重复。下表单位均为百分点，97.5%区间：

| 比较 | 点估计 | CI97.5% |
| --- | ---: | --- |
| overall: selected − bottom | 11.888 | [9.834, 13.995] |
| overall: selected − random | -0.593 | [-2.605, 1.598] |
| change: selected − bottom | 11.504 | [8.225, 14.788] |
| change: selected − random | 2.536 | [-0.364, 5.566] |

selected较bottom总体+11.888pp，区间完全正；较随机总体−.593pp，区间跨0。不能说统计证明selected更差，但没有证据达到预设正收益。change较随机+2.536pp，区间仍跨0，不满足整体推进标准。

zara/hotel有正的点估计是反例，需保留；university与transition较随机差，不能挑有利场景作为继续训练的理由。不同scene分母高度不均，不能把它们当三场景可靠独立统计。

## 5. 一个结构性限制：历史输入没有参与候选相对排序

线性拼接模型的logit为z_ij = w_obs·obs_i + w_native·native_ij + b。同一窗口内obs项是共同偏置，排序时抵消；sigmoid单调且本次无饱和tie问题。

已用保存三fold模型做纯算术验证：去掉整个观测共同偏置后，**全部7334窗口所选4条路径EXACT不变，0个mismatch**。这不是新拟合或模型重放。证据见completion_review/LINEAR_RANK_INTERPRETATION.json。

因此本S1事实上测试的是“训练得到的原生路径全局线性优先级”，不是历史与候选交互的可修复性判断。即使它成功，也不能声称理解了人的运动；此次结果更不能否定所有history-conditioned addressing。这个限制在FIELD_INTERPRETATION.md已于生产结果之前记录。

## 6. 执行、审核与保留

唯一Windows任务Trajectory-RepairableAnchorSelector-Production-20261011，零trigger；worker记录PID44588/parent2688，23:36:56.000788→23:37:12.689236，双SUCCESS，16.688448秒。

外层启动调用误把PowerShell未设置的LASTEXITCODE当失败，但真正任务已提交；未重开。唯一实际检查23:37:37.4757979时已经taskReady/processnone/worker+exitSUCCESS/stderr0。没有观察到live commandline/parentname，不能伪造“启动时Running已确认”；已保存该迟到检查。没有额外half/完成PID查询，没有新automation，因为首次检查已完整完成。

1953项相关source/input/旧artifact SHA继承前序摘要不覆盖；14个科学输出SHA与初末核验PASS，不是全部历史34k文件，也不是旧F16 GPU重放。完整CPU保存复核23.3573s：所有28791标签、OOF训练统计/权重证书/分数/slot、全部组摘要与门、4bootstrap记录、SHA重算EXACT/MAX0，无新拟合/Torch/optimizer/外审。

生产private峰值.488796GiB、RSS.452015GiB、最小可用physical14.215939/commit20.092480GiB；过程6GiB及physical/commit各余6GiB门保持。预检峰值不替代生产峰值。未改OS/pagefile或用户程序，旧E3/旧automations保持暂停。

## 7. 当前分支与下一步建议

数值问题已解决；现在是**固定线性选择器性能门失败**。完整A拟合、B迁移、outer/inner/CAL冻结策略、S2生成头训练、S3/S4/S5均未执行，原因是前置门未触发，不是技术阻塞或假完成。baseline/historycycle11/正式计数不变。

下一科学步骤宜先单独预注册**一个短预算、能表达历史×候选交互的潜力probe**，仍沿用同A/B/OOF、标签与原门，并保留本线性与固定random负对照。例如候选速度与观测速度的内积/平方差等GT-free交互能改变同窗口排序；简单线性“相减”仍会抵消，不能声称已经解决。此处只是建议，未实施/训练/换模型，不盲扫宽度、C、分位数或直接重训生成器。

### 同助手反方检查2/3，非独立外审

- 强反驳：bottom本来很弱，胜过它不能证明选择器有可用信号；随机4已解释大部分增益。
- change/zara正点估计和整体随机CI跨0保留，不把“未过推进门”误写成“证明没有任何信息”。
- 三角包络标签是假设乐观自由位置修正，未包含五motion硬安全；命中不能换算成候选覆盖、minADE/FDE改善或可学性。
- 原生cache/F16已见旧pool，A/B仅target-disjoint而非time/context-disjoint，只有有限scene/重复窗口，探索而非SOTA/独立wholemodel泛化。
- 本报告由同一AI助手实施、重算与反方检查，非外部审计。

来源：本目录RESULT.json/OOF_RESULT.json、PREFLIGHT.json/NUMERICAL_REPAIR_AUDIT.json/BENCHMARK.json、completion_review/REVIEW.json/LINEAR_RANK_INTERPRETATION.json、source_manifest.json与各fold_manifest.json。
