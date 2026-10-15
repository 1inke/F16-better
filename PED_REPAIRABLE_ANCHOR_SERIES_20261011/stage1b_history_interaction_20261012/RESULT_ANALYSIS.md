# S1b：历史×候选交互选择器对照——结果分析
日期：2026-10-12（香港时间）  
状态：**ANALYZED_S1B_GATE_FAIL_NO_S2**。生产与独立保存审核通过，科学推进门未通过。

## 结论

本次真正加入了历史条件候选排序，但没有得到可靠优于固定随机4的潜力命中。按原规则结束这个短预算探针，不进入S2，不扫C、交互项、阈值或数据划分。

交互处理overall为473/3205=14.7582%，固定随机451/3205=14.0718%；差仅+0.6864个百分点，97.5% paired-track CI为[-1.4152,+3.0002]个百分点，远未达到事前要求的至少+5个百分点且CI下界>0。这不是统计上证明交互必然无效，而是本方案没有达到推进标准。

**所有H4均为“原生全20失败时，选中4条至少一条有乐观幅度修正潜力”的命中率，不是真实修正路径、coverage或rescue。**

## 1. 实验设计：只增加历史与候选的交互

沿用原A7334窗口/281行人轨迹、B11575/408轨迹和固定三个track-OOF折，不重划、不改标签。数据仍是原target-grouped训练18909；没有DEV/test/其他fold或外部数据。

| 项目 | 加性控制 | 交互处理 |
|---|---|---|
| 输入 | P2观测96 + native路径24 + 真实观测运动4，共124维 | 相同124维 + 12交互，共136维 |
| 观测运动 | 最后速度v=(p[-1]-p[-2])/.4；速度变化dv=v-(p[-2]-p[-3])/.4 | 相同 |
| 新机制 | 运动量是同窗口共同加性项 | v和dv分别与native前/中/后各4步均速的dot/cross |
| 模型和数值 | balanced L2 LogisticRegression，C=1，newton-cholesky，tol=1e-6，max_iter500，1thread | 相同 |
| 标准化 | 各OOF训练侧float64 mean/std，std<1e-6置1 | 同算法、自己的训练侧数值 |
| 监督 | 原envelope潜力标签 | 完全相同 |

新增运动和乘积用源float32位置转float64计算，最终特征float32；候选第一步速度从已核验的零观测原点计算，dt=.4。dv为速度差，不冒充已经再除dt的加速度。

真实观测来源与全部28791 ID对齐：encoder past8×2和原observation首16及160:176逐元素EXACT，末观测点严格0。旧extract源码/encoder_audit证明同focal-heading/native原点。没有从GT或预测未来反推观测运动。GT/groups/logits/scene/ID不进特征；ID只用于分折、审计与固定负对照。

加性124和交互136参数数目不同，增维也改变正则化几何，不能称equal effectivecapacity或纯交互因果。旧120结果只读保留，没有重训旧模型。

## 2. 主结果：未达到随机优势门

分母只含native20全部不满足同一完整candidate ADE<=.2 AND FDE<=.4的窗口。

| 人口（native失败分母） | bottom4 | 固定随机4 | 旧120线性（保存参照） | 新124加性 | 新136交互 |
|---|---:|---:|---:|---:|---:|
| overall 3205 | 51 / 1.5913% | 451 / 14.0718% | 432 / 13.4789% | 432 / 13.4789% | 473 / 14.7582% |
| change 1104 | 25 / 2.2645% | 124 / 11.2319% | 152 / 13.7681% | 159 / 14.4022% | 130 / 11.7754% |

交互命中全部1650个有机会的失败窗口中的473个，机会召回28.6667%；随机451/1650=27.3333%。候选标签是乐观上界，实际可实现修复未验证。

主门保留原设定：OOF overall对bottom和random均至少+5pp且97.5%CI下界>0；每fold change不得低于bottom；通过后才完整A和B迁移。

| 比较 | 点差（百分点） | 97.5% paired-track CI（百分点） |
|---|---:|---:|
| 交互−bottom，overall | +13.1669 | [+10.7919,+15.8526] |
| 交互−random，overall | +0.6864 | [-1.4152,+3.0002] |
| 交互−加性，overall（探索） | +1.2793 | [-0.2414,+2.9831] |
| 交互−random，change | +0.5435 | [-2.2286,+3.3658] |
| 交互−加性，change（探索） | -2.6268 | [-5.6097,+0.2972] |

bootstrap2000次，按同sequence+track成对重采样，沿用原rng。三个OOF折不是三个独立模型seed；多个窗口/20候选也不是独立重复。探索的加性/置换比较不能替换primary门或用来挑下一组参数。

三个fold change交互/bottom/random分别47/4/42、60/12/48、23/9/34。原“每fold不低于bottom”门通过，但第3折交互低于random，不能称变化组稳定提升。总体随机优势门失败，故OOF=false/S2=false；B未执行不是B迁移已失败。

## 3. 历史置换：结构缺口已修复，但不能偷换成理解

每OOF验证折仅按身份做固定GT-free置换：track SHA分块、块内ID SHA排序，固定循环移位最大track窗口数，要求最大块<=N/2且所有donor不同track。p2和past共同换给另一个行人，native/GT/labels不动；不重拟合。

| 模型 | 正确历史overall H4 | 置换历史overall H4 | 全7334窗口选槽集合变化 |
|---|---:|---:|---:|
| 加性124 | 432 / 13.4789% | 432 / 13.4789% | 0 |
| 交互136 | 473 / 14.7582% | 340 / 10.6084% | 6019 / 7334（82.07%） |

交互正确−置换overall +4.1498pp，97.5%CI[+2.1942,+6.2823]pp；加性差与CI全0。这支持处理模型确实对历史—候选配对敏感，已不再只有共同偏置。

但change正确130、置换142，正确−置换-1.0870pp，CI[-4.4138,+2.3370]pp。置换改变输入联合分布，是OOD诊断，不是心理意图、因果运动理解或可迁移生成能力的证明，也不能代替真实随机寻址优势。

加性124的共同历史项排序时仍抵消；训练时这些项可以改变拟合得到的路径系数，所以不能预设与旧120全部选槽一致。两者overall均432，但change159与152不同，保存参考均已独立复核。

## 4. 反例与组成：不是只报overall增加

| 组 / 失败分母 | 新加性命中 | 交互命中 | random命中 | 交互置换命中 |
|---|---:|---:|---:|---:|
| stable /1326 | 225 | 300 | 227 | 154 |
| transition /775 | 48 | 43 | 100 | 44 |
| change /1104 | 159 | 130 | 124 | 142 |
| hotel /52 | 11 | 11 | 7 | 8 |
| university /2546 | 295 | 297 | 346 | 259 |
| zara /607 | 126 | 165 | 98 | 73 |

交互−加性overall净+41，由stable+75、transition-5、change-29组成；不能描述为“更会找运动变化”。zara是正向反例，university仍低于random；hotel分母52，不能泛化。没有按场景/运动组择优改变主协议或训练人口。

## 5. 执行、修复与可信度

- 唯一任务Trajectory-RepairableAnchorInteraction-20261012，真实PID1720/serviceparent2688，00:14:43.150101启动，00:15:02.773806 worker+exit双SUCCESS，实际19.623705s。ONEstartup00:14:53.901485确认Running/0trigger/完整commandline/无stderr；additive三fold完成、interaction fold0正在fit。
- 唯一半程scheduled00:16:36、actual00:16:56.7298885（迟20.73秒）发现已全部完成，Ready/processnone/0trigger/stderr0；没有额外completion/PID检查，没有重复启动或重训。原规划3.7525min/ETA3–10min只是吞吐保守估计，完整CPU复核76.506s，无需新appautomation。
- 6OOF模型均收敛且无警告：加性各3iteration，交互各5iteration。独立stationarity满足原1e-6；交互最大梯度约9.565e-11。实际513重复fit BITWISE及独立NumPy objective/Hessian reference/FD通过；不伪称神经两epoch/optimizer-RNG恢复。
- 1993相关科学来源继承1953全部旧digest并扩展，22scientific输出SHA初末PASS；不是全历史34k或旧F16 GPU重放。旧source/models/results/失败均不改。
- [独立保存审核](completion_review/REVIEW.json) PASS/max0：全28791标签/ID/观测、六模型训练IDs/float64统计/权重stationarity、全OOF概率/slots/固定置换及组结果、主门、bootstrap和旧120保存参照全部独立复算。无refit、Torch或神经forward/backward；独立算术不等外部独立评审。
- 第一预检失败仅ID-hash循环置换没有全异track解，原code/design/protocol/failure留在[修复来源](REPAIR_PROVENANCE.md)和failures/preflight1_history_derangement。在任何OOF H4/production前，仅将诊断donor构造改为可证明的trackblock固定移位，完整重预检；模型/标签/标准不变。分析读取时两次大stdout JSON被工具截断，改为小结构提取；没有科学文件修改。
- production private峰值0.510620GiB/RSS0.474251；system可用physical最低18.435914/commit26.796223GiB。审核private0.459660GiB。process6GiB与physical/commit各reserve6保持，不OS/pagefile/userkill。

本次没有生成任何新轨迹，因此没有新增部署运动学收益或失败结论；native路径SHA不变。不重跑原motion指标来伪造新科学结果。原5%change ADE/FDE、protected1%、三seed、同路径coverage与五pooled motion1%门均不变，S1b通过执行不等major成功，不升baseline/cycle11或假goalcomplete。

## 6. 原计划现在走到哪里

已完成S1和本次一次S1b。**完整A fit、B迁移、outer/inner/CAL选择器导出及S2–S5均未执行，因为主OOF门未通过。** 不将未执行写成结果差或不可学。

本次闭环停止该预注册交互探针，不自动扩大交互项/MLP/MoE/C/阈值/split，也不绕过门训练生成器。它否定的是这套低维dot/cross探针已满足实用随机优势，不能否定所有历史条件寻址或generation可能性。

ARS反方2（同助手）：保留stable与zara正向证据，也保留change/transition/university反证，不挑场景或置换CI掩盖primary FAIL。  
ARS反方3（同助手）：最强反驳是乐观标签与真实可修复性不同，增维正则和OOD配对也能解释变化；历史敏感不等理解，且主随机比较仍不可靠。无外审、无SOTA/独立全模型泛化/decoder唯一根因主张。F16已见原pool、targetdisjoint非contextdisjoint、仅3scene/重复CAL、适应性研究的限制保留。

依据：[协议](protocol.json)、[生产RESULT](RESULT.json)、[preflight](PREFLIGHT.json)、[启动检查](startup_check.json)、[唯一半程](halfway_check.json)、[保存审核](completion_review/REVIEW.json)。AI助手协助实施、算术复核与写作，三次反方检查由同一助手inline完成。
