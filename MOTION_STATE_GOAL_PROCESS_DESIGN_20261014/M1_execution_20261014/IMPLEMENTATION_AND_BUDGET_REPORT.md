# M1 实施与预算门报告

日期：2026-10-14（香港）  
结论：**新实现与实际数值预检已完成，生产预算不合格；未启动正式训练。**

## 1. 本次已实施什么

按父 [M1](../M1_DYNAMIC_HISTORY.md) 与 [共同协议](../COMMON_PROTOCOL.md) 实施第一项，不续开旧G2桥。

- A：末速度重复7token、有效加速度为物理0；B：7真实速度、6有效加速度。两臂mask和共同FIT统计相同，物理0正确标准化。
- 两臂都是GRU(5,32)＋P2＋相同目标网络，实际参数数均为**34,146**，全参数/buffer初始配对。
- 复用经ID/SHA核验的G1 FIT Kmeans20中心；不重选中心，不以未来GT/native/logit进入预测forward。
- 已实现canonical K20/190对loss、25epoch/seed42–44、完整FIT保存、explicit-ID输出、checkpoint全optimizer/RNG合同、独立CPU保存审核脚本及20,000次crossed bootstrap。
- 已实现独立OS启动与单次检查脚本，但**没有调用启动脚本、没有新Windows任务、worker或生产PID，也没有建立自动跟进**。

## 2. 实际已通过的预检（不是科学结果）

依据 [PREFLIGHT.json](PREFLIGHT.json)：

- 既有raw末速度与G1逐ID EXACT；统计31/256分块最大误差1.11855e-14，小于既定1e-11。
- 双方实际8批FIT256、全部合法GRU/context/query/head梯度分阶段正且有限；zero-last首步前层零梯度如实保存。
- paired初始参数/buffers、prototype20初始路径精确一致；canonical AST输出梯度EXACT和独立NumPy标量通过。
- 固定double副本两eps=1e-4/1e-5，对goal head、context和GRU参数有限差分通过，无winner/hinge活动面切换；不是float32量化导数证明。
- **两臂真实FIT513＋inner12子集两轮**连续/中断全部参数、buffers、optimizer、Torch CPU/CUDA、loader、NumPy/Python RNG、history EXACT；不是完整18,909人口两轮或分析恢复证明。
- 双方实际outer128保存读回、samecandidate质量、原SciPy完整聚簇及五motion独立复算通过；坏source/output/resume合同均真实拒绝。
- 向量化独立距离计算与原190对Python循环在双方实际128窗口的聚簇结果EXACT；SciPy complete-link、D=.10/.20、cut1和全部科学阈值不变。
- 固定hash128的raw、实际GPU标准化tokens与初始输出已另存；跨seed/track bootstrap索引合成fixture误差1.66533e-16。
- 3,738个相关来源初末SHA核验不变。完整来源见budget_review/SOURCE_SNAPSHOT.json，这是证据快照，**不是生产启动许可或全历史34k文件/GPU重放证明**。

预检峰值private2.75441GiB、RSS1.85535GiB，系统physical最低13.60054GiB、commit最低15.15580GiB。process6GiB与系统两项reserve6GiB均保留；无OS/pagefile/用户程序处理。最终完整数值预检实际78.2074秒，阅读/实现和第一次预检另计。

## 3. 为什么现在没有启动

父共同协议第7节要求：每项完整生产规划不得超过30分钟。当前完整实测推算是：

| 成本 | 依据 | 名义耗时 |
|---|---|---:|
| 六个完整训练run | 两臂FIT256每步0.047165/0.051636秒，合计11,100 steps | 9.14分钟 |
| 24个完整科学保存job＋双算术及独立复核＋前向 | 六run×(18909+1035+1856+2879)窗口，含完整FIT | 16.14分钟 |
| 20,000 crossed draws、逐draw SHA及复核估计 | 26项主family不因未执行M2/M3而缩小；次级全组另报 | 4.65分钟 |
| 名义合计 | 不含额外不确定性余量 | 29.92分钟 |
| 加25%吞吐/IO余量及150秒固定余量 | 与预检事前公式一致 | **39.91分钟** |

当前估计区间约**28–60分钟**，不是保证，也不是已经运行40分钟；因为未启动，没有预计结束钟点。初末source/output SHA与保存IO的不确定性由规划余量覆盖，不能把名义29.92分钟冒充完整规划小于30。

第一次逐对循环方案规划117.24分钟，触发预算STOP。一次同算术向量化将规划降至39.91分钟，仍超上限。不会继续无界优化/切人口/减epoch/缩CI/省略完整FIT或独立审核来凑PASS。

来源、旧代码、两次预算数据和归档路径错误的NO_OVERWRITE拒绝均保留于 [REPAIR_PROVENANCE.md](REPAIR_PROVENANCE.md) 及failures/。没有训练科学阴性结果，不把预算停止描述为模型无效。

## 4. 当前状态和后续边界

- **M1：NUMERIC_PASS / STOP_BUDGET_WAIT_HUMAN；正式6 runs为0/6。**
- **M2：未启动。** M1没有得到S1科学结果，不能把“未执行”当“阴性”，也不在本轮擅自选初始化或跳步。M1阴性不取消M2是结果分支规则，不等于预算失败后无说明跳过。
- M3与V仍为条件性、未执行；整个系列未完成。
- 旧模型、旧结果、父设计文件、旧E3及暂停自动任务不改变；baseline/historycycle11/未完成goal不重置。

需要的唯一人类决定是：是否为**这一项**明确放宽30分钟单项生产预算，接受约40分钟规划、可能28–60分钟的耗时。若不放宽，保持本预算STOP，不自动重试或启动。无需用户判断数值容差或低层实现；那些已由实施者完成。

本报告使用academic-research-suite组织事前问题/反方与结果边界；同助手inline审核，不是外部审稿。OpenAI Docs仅用于核对后续定点跟进的产品机制；本次因预算门未通过，未创建唤醒任务。
