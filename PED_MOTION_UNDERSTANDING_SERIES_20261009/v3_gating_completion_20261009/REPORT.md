# v3 门控、方向幅度及冻结前向补完报告

2026-10-09。依据人类指定的 `ped_next_steps_v3_gating_and_completion.md`。本轮没有神经网络重训、optimizer/backward、scorer 改动、旧 E3 恢复或自动化重启。小规模门控与混合线性拟合已执行。结论为探索性，不升级 baseline、不宣称模型理解意图。

## 1. 结论先行

1. **UNIV 的过度修正确实可以部分缓解，但不能把问题概括成“方向正确，只差幅度”。** top-1 在 P1 上，即使使用测试标签计算最优全局收缩，其技能上限只有 0.019–0.027；mean-K 为 0.142–0.153，线性探针为 0.123–0.139。top-1 与探针的差异不只是尺度，mean-K 则保留了较合理的平均方向。
2. **跨场景门控有整体收益，但未通过研究所需的完整性与保护条件。** UNIV top-1 全人口技能从约 −0.149/−0.157/−0.133 提升为 −0.018/−0.024/−0.043，三 seed 配对区间均为正；P1 完整合格率却下降约 0.92–1.09 个百分点。mean-K 门控在 P1 上也降低技能和覆盖，且有运动学副作用。不能部署，也不自动转 T3-lite。
3. **mean-K 与探针的跨场景混合，没有稳定胜过二者。** UNIV 的拟合并没有得到“探针负责补充大量新信息”的结果；失败也可能来自跨场景系数迁移，不能反推两种表示完全等价。
4. **UNIV 对邻居不是完全没有响应，而是响应较小、方向关系较弱。** 威胁加权后的 mean-K 剂量斜率仍跨零，没有恢复出 ZARA01 那样的方向关系。反对称逐窗口失败比在 UNIV 很高，不能用全局正负抵消解释为通过。
5. **评分前后几何错位只得到局部、阈值相关证据。** 严格标准下 UNIV P1 的 paired AUROC 差仅 seed2 区间为正，另外两个跨零；宽阈值下三 seed 均有小幅差异。因此不能认定这是首要根因，更不能据此立即训练 scorer。
6. **原 UNIV 数值阻碍已在公开、一次固定的几何预算下解决，15 个模型的缺失前向与冻结审核已补齐。** 朝向阈值 0.01 能大幅降低 UNIV 旋转尾部，但几乎不改善实际预测；更长朝向窗口甚至恶化质量。几何一致性与运动预测能力必须分开。

主要数据：[G1](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G1.json>)、[G2](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G2.json>)、[G3](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G3.json>)、[G4](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G4.json>)、[G6](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G6_POPULATIONS.json>)。

## 2. 执行范围与设计

| 项目 | 本轮执行 | 说明 |
|---|---|---|
| G2 方向/幅度 | 五 fold × 三已有 seed，top1/meanK/自身探针/邻居探针 | 全局带符号余弦、cos²、逐窗口分布、训练固定速度箱，配对 track CI |
| G3 混合 | 每目标 fold、每 seed，其他四 fold 拟合两个系数 | 自身和邻居探针分别与 mean-K 混合；目标 GT 不拟合 |
| G7 spread 反例 | 五 fold 对照、速度/静止/CV 成败分组 | 未据结果新增门控特征或调整分箱 |
| G1 门控 | top1/meanK，恒等/标量/spread/full/转导式/TEST 上限 | 只改点预测，不收缩 K20 候选集合 |
| G6 邻居定位 | 既有 with/noNbr 保存输出，五 fold 三 seed | 幅度、方向、距离/数量分箱、固定威胁加权；另补 all/P1/CV 成功人口 |
| G4 评分几何 | 15 个冻结 V2 模型，同一次前向抓取修正前后路径及相同 logits | 严格与宽阈值，mixed-window eligibility 单独保存 |
| G5 前向补完 | ETH/HOTEL 六模型完整旧输出只读复用；UNIV/ZARA 九模型缺失干预新执行 | 原阈值、0.01/0.06、长窗口、回溯、旋转/镜像、四固定角度；全15冻结审计 |
| 随机朝向补充 | 独立的哈希随机朝向小实验 | 四固定角度不是随机；另用每窗口固定一次哈希角度，仅改变原 fallback，非挑最优角 |
| T3-lite | 不启动 | 门控的 P1 完整性和运动学/保护条件不满足 |
| E8 跨设备效率 | 未恢复 | 坐标舍入预算不能作为神经网络后端误差或 logits 的证明 |

完整测试保存数据为 34,161 个窗口，三模型 seed 重复使用同一批窗口，不能按 102,483 个独立样本推断。冻结干预沿用固定 hash 选出的 4,460 个窗口（ETH364，其他各1024），不能把该子人口与全人口直接强配对。P1 及恒速成败定义继承原方案；P1 含未来 GT 的事后人口选择，不是部署时可用标签。

严格完整性始终是**同一条路径 ADE≤0.2 且 FDE≤0.4**；没有拼接不同候选或分段 oracle。敏感性阈值不替换主标准。五项 pooled motion 为 speed P95、acceleration mean/P95、jerk mean/P95，dt=0.4。

[预结果设计](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/DESIGN.md>)、[冻结协议](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/protocol.json>)。

### 门控的两项方法学限定

门控标签来自另外四个 fold，但那些 donor 模型的上游训练可能见过当前目标场景。因此，这个方案实现的是**门控层跨 fold meta 泛化**，不是整个模型流水线完全嵌套的未见场景验证。旧测试标签在本轮已明确授权作为其他 fold 的 meta 拟合标签，结果仍属于适应性探索。

按照指导文件代码骨架，先拟合未截断线性 ridge，再在推理时 clip 到 [0,1]；不能称为直接最小化 clipped 非线性目标。岭系数1固定，按窗口加总，所有系数包括截距惩罚，donor统计拟合 z-score。转导式变体只使用目标输入特征的统计，不使用目标标签。各 fold 窗口量不等，UNIV 在非 UNIV donor 拟合中占很大比例；本轮未事后改成场景等权。

## 3. G2：top-1 不只是过度修正

技能定义是 `1−模型SSE/CV的SSE`；CV 技能为0。全局自由标量 λ* 和 cos² 仅为 TEST 诊断上限，不是部署参数。必须保留余弦符号；零残差的逐窗口余弦记为未定义。

UNIV P1，8,793 窗口 / 653 target tracks：

| 读法 | 当前技能，seed1/2/3 | 最优全局缩放技能，seed1/2/3 | 全局带符号余弦 |
|---|---|---|---|
| top-1 | −0.0622 / −0.0580 / −0.0390 | 0.0188 / 0.0239 / 0.0268 | 0.1369 / 0.1547 / 0.1636 |
| mean-K | 0.1395 / 0.1310 / 0.1264 | 0.1526 / 0.1495 / 0.1422 | 0.3907 / 0.3866 / 0.3771 |
| 自身探针 | 0.1161 | 0.1233 | 0.3511 |
| +邻居探针 | 0.1343 | 0.1388 | 0.3725 |

探针每 fold 只拟合一个，不是三次独立探针重复。top1 的最优缩放技能比邻居探针低 0.112–0.120，三 seed 配对 track CI 全为负；mean-K 与邻居探针之差只有 0.0035–0.0139，三 CI 都跨零。seed1 top1 对邻居探针差 CI 为 [−0.1350,−0.1038]；mean-K 对邻居探针为 [−0.0065,+0.0352]。

这是对实际选中路径的方向/时间残差缺陷的证据，不是“全部候选都没有运动信息”。多模态 top1 与平方误差 mean-K/探针的目标差异也是替代解释，不能直接认定 decoder 或 scorer 的唯一根因。

五 fold 的 P1 基线必须同时保留：

| fold | top1 技能范围 | mean-K 技能范围 | 自身探针 | +邻居探针 |
|---|---:|---:|---:|---:|
| ETH | 0.417–0.484 | 0.302–0.343 | 0.072 | 0.035 |
| HOTEL | 0.271–0.363 | 0.205–0.228 | −0.097 | −0.086 |
| UNIV | −0.062–−0.039 | 0.126–0.140 | 0.116 | 0.134 |
| ZARA01 | 0.113–0.133 | 0.178–0.184 | 0.143 | 0.175 |
| ZARA02 | 0.131–0.149 | 0.203–0.221 | 0.167 | 0.192 |

ETH/HOTEL 的模型强于这组固定线性探针，是重要反例。但“只有 ETH/HOTEL 学到非线性结构”仍过强：有限探针不是所有线性模型上界，性能差异不等于结构可识别证明。

[完整基线记分卡](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/BASELINE_SCORECARD.json>)、[方向配对区间](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/DIRECTION_PAIRED_CI.json>)。G2 原表另保存各速度箱的训练/测试 GT 残差幅度，不事后改箱。

## 4. G1：整体收益不等于难例或物理安全收益

UNIV 全人口 24,334 窗口：

| top1，seed | 原技能 | full gate 技能 | 技能增益95% track CI | 少破坏的CV成功窗口 | 减少的rescue |
|---|---:|---:|---|---:|---:|
| 1 | −0.1492 | −0.0182 | [0.1181,0.1450] | 418 | 67 |
| 2 | −0.1572 | −0.0236 | [0.1204,0.1487] | 328 | 26 |
| 3 | −0.1326 | −0.0429 | [0.0806,0.0991] | 296 | 29 |

损失成功样例的减少大于 rescue 的减少，全人口点预测合格率因此增加1.10–1.44个百分点。但三个 gate 仍低于 CV 技能0。

**必须保留的负结果：**

- UNIV P1 top1 技能变为 +0.0070/+0.0116/+0.0034，但完整合格率分别下降1.092/1.092/0.921个百分点。降低 SSE 不等于完整路径改善。
- UNIV P1 mean-K 技能从0.1395/0.1310/0.1264下降至0.1024/0.1030/0.1096，三配对差 CI 都为负，合格率也下降0.74–0.90个百分点。
- ETH/HOTEL/ZARA02 的 top1 full-gate 技能变化多数点估计为负而 CI 跨零；这不是证明非劣。没有注册非劣 margin，不能把“未检出显著下降”写成“通过非劣”。HOTEL top1 的 rescue 减少71–78个，少破坏只有13–19个。
- 转导式输入 z-score 没有形成稳定优于普通 gate 的 UNIV 结果，不能宣布“只要做目标场景标准化就解决域偏移”。
- 门控只作用于点预测，完整 K20 bank 仍未改动；不能把 point coverage 的改善称为候选生成覆盖改善。
- point motion 也非自动安全：UNIV P1 top1 seed1/2 的 speed P95 相对原点预测约+1.112%/+1.002%，超出原1%保护线。mean-K 门控各 fold 都出现至少一项物理指标超过1%。gate 常数小也不保证所有速度分位数下降，因为 CV 与原模型的速度几何不同。

[全部物理比值](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/GATE_PHYSICAL_PROTECTION.json>)。原 strict major 和五 motion 规则未放宽，本轮不是 major PASS。

## 5. G3：没有获得跨场景稳定互补收益

UNIV +邻居探针混合：`CV + a*R_meanK + b*R_probe`，a约0.357–0.370，b约0.082–0.092；自身探针混合 a约0.430–0.444、b为负（−0.083至−0.095）。这是其他场景拟合出的系数，不是目标优化系数。

P1 的邻居探针混合技能为0.1222/0.1236/0.1184。相对 mean-K 的增益 CI：seed1为[−0.0332,−0.0019]，seed2/3跨零；相对邻居探针：seed1/2跨零，seed3为[−0.0281,−0.0026]。因此不能说混合优于两者。

它不证明模型没有新信息，也不证明简单线性混合永远无效；跨场景分布、系数迁移和相近残差方向都能解释这个负结果。本轮不继续扫描混合系数。

## 6. G6/G7：邻居响应与 spread 反例

G6 在固定 selected C3 人口中，UNIV mean-K 的全路径残差响应 L2 中位数为0.134/0.208/0.269 m（RMS应除以√24），不是零；ZARA01为0.481–0.617 m。UNIV nearest-away 正号比例约50.8–51.6%，signed response接近0。

固定威胁权重为 `exp(-d_min/1m)/(1+t_min/12)`，只由观测 CV 构造。UNIV mean-K 加权剂量斜率为−0.0077/+0.0054/+0.0074，CI分别[−0.0321,+0.0178]、[−0.0337,+0.0449]、[−0.0319,+0.0478]；没有恢复稳定剂量关系。ZARA01同读法为−0.169/−0.147/−0.174，三个区间均为负。

因此更谨慎的表述是：**UNIV 邻居响应较小，且这套方向定义下未呈现稳定剂量关系**，而不是“模型完全忽略邻居”“attention塌缩已证实”或“读取邻居后随机乱用”。无量级等价 margin 和没有所有交互表示的因果识别，不能做更强结论。G6原表含最近当前距离/未来CV closest距离/邻居数分组；补充表同时列all/P1/strict-CV-success人口。

G7显示 ZARA01 all-population spread AUROC仅0.504/0.519/0.543，对照 ZARA02为0.696/0.782/0.883。ZARA01 spread中位数12.25–12.66、P95为20.22–20.80；最高5%窗口仅占总spread的9.0–9.6%。没有“少数巨大离群值主导全部spread”的证据。更像分散度与成败/人口组成的关系不同，但不能凭此确定唯一原因。冻结速度与静止分组已完整保存，未改成事后门控特征。

[G7原表](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G7.json>)、[G6分箱原表](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G6.json>)。

## 7. G4：不能把不同 eligibility 的 AUROC 均值直接作因果差

同一个 V2 模型、同一窗口、相同 logits，比较修正前后候选。只有同时含好/坏候选的窗口才有窗口内AUROC。修正能改变这个资格，因此分别列pre/post的AUROC均值，再在**共同 mixed 窗口**做 paired 差，不能只相减两个不同比较人口的平均数。

UNIV P1，严格0.2/0.4：

| seed | 共同mixed窗口 | paired pre−post AUROC | 95% track CI |
|---|---:|---:|---|
| 1 | 31 | 0.0000 | [−0.00269,+0.00269] |
| 2 | 36 | 0.02063 | [+0.00554,+0.03964] |
| 3 | 35 | 0.00835 | [−0.00301,+0.02711] |

宽阈值0.4/0.8，共同mixed为197/201/198窗口；paired差0.02272/0.01651/0.01497，三个区间均为正。这是局部、阈值相关的错位信号，不是严格标准下三seed稳定根因。ETH严格P1共同mixed每seed只有1个窗口，无法给可靠track区间，应记为不可估，不凑显著。

当前15个模型都具有refiner，完成的是同模型pre/post检查；本轮未额外构造独立H3架构比较，因为那会改变模型协议和来源，而非当前H9的同模型干预。没有训练新scorer，也没有把logits称为校准未来概率。

## 8. G5：数值修订、朝向与逐窗口镜像

### 一次公开数值修订

保留原 `forward_diagnostics/FAILURE.json`。新几何预算在结果前固定为 `max(2e-5,16*eps32*M)`，k不调整。全程float64计算最后速度、单位轴和反射，送入模型前cast为float32；仅这是数值实现修订，不改镜像科学构造或motion评价标准。

UNIV：M=15.3218603 m，float64 closest-distance误差2.4758e−14 m；float64→float32误差2.1119e−5 m，约11.5626倍eps32*M，新固定预算2.9224e−5 m。构造float32与float64的坐标最大差6.0619e−6 m，单纯cast坐标最大差9.5367e−7 m。几何差会在速度/未来closest计算中传播，不能只盯单个原始坐标误差。

这证明本次反射数值问题可以归入固定浮点几何预算，**不证明**E8的模型CPU/GPU路径误差2.77e−4或dimensionless logits误差3.32e−4也可由同一个米制预算免责。E8旧STOP仍保留，不能以本次修订冒称E8效率试验完成。

[逐fold数值记录](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/NUMERIC_REVISION.json>)。

### 几何一致性不等于质量修复

UNIV原始旋转bank Hungarian匹配P95为0.479/0.586/0.392 m；阈值0.01后降至约1.31e−5/1.34e−5/1.53e−5 m。long7也降至约6.79e−6–1.12e−5 m。

但在同一1024窗口子人口，原始top1技能−0.213/−0.227/−0.160；threshold01仅为−0.211/−0.226/−0.160，基本不变；long7反而为−0.396/−0.384/−0.331。backtrack有质量改善但仍为负，并没有消除旋转P95尾部。不能挑一个测试效果好的朝向变体部署，也不能说几何缺陷已经解释UNIV的整体失败。

### 邻居镜像的逐窗口反对称失败

主逐窗口指标是 `abs(l+l')/(abs(l)+abs(l'))`，零响应记为未定义，并另报数量；不是全局正负平均。selected C3 top1逐窗口失败比均值：

| fold | seed1 | seed2 | seed3 |
|---|---:|---:|---:|
| ETH | 0.639 | 0.690 | 0.653 |
| HOTEL | 0.650 | 0.691 | 0.680 |
| UNIV | 0.830 | 0.888 | 0.945 |
| ZARA01 | 0.581 | 0.546 | 0.585 |
| ZARA02 | 0.624 | 0.720 | 0.604 |

UNIV mean-K也为0.839/0.871/0.945。全局有符号A可能接近零而逐窗口仍失败，不能互相替代。比率在很小响应上也可能大，必须结合绝对response magnitude看，不认定每个失败都具巨大实际影响。

ETH/HOTEL完整干预旧输出仅只读复用，使用原数值轴；新九模型使用修订的float64轴后cast。二者实际路径均有保存来源，不假称重新计算过旧六模型全部干预；全15模型的fresh base+参数/bool/statbuffer exact冻结检查已完成。

[全15冻结审计](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/FROZEN_AUDIT.json>)、[完整朝向质量](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G5_HEADING.json>)、[完整几何](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G5_GEOMETRY.json>)、[完整镜像](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/G5_SOCIAL.json>)。

## 9. 审核与成本

主保存分析60.07秒、冻结补完146.31秒、主CPU保存审核46.52秒；补充表/算术各约8秒。随机朝向前向12.63秒，保存审核约2秒。主分析private峰值0.478GiB；冻结前向2.844GiB，最小可用physical13.760/commit16.218GiB。始终保留process6GiB与systemphysical/commit各6GiB余量；没有改OS/pagefile或关闭用户程序。

CPU主审核36,388,140项数组/标量比较，包括所有门控预测、拟合正规方程、G1/G3结果和track CI、全部15模型G4/朝向/镜像结果、全部150组GT-free Hungarian cost重构、TTA及旧六模型social算术。这个计数主要是重复数组元素，**不是独立实验或独立样本数**。补充重算完整G2/G7数值和G6威胁权重/响应。不是所有model/gradient第二次重放；并非所有G2/G6/G7 CI都进行独立第二次bootstrap，更不是外部审稿。

所有实际输入、相关科学source、checkpoint和旧保存结果SHA前后保持；范围是实际相关文件，不声称核验全部历史3万余文件。新报告仅本目录；旧失败源与输出未覆盖。

[主保存审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/REVIEW.json>)、[补充审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/SUPPLEMENT_REVIEW.json>)。随机朝向另有专门结果与保存数据审核，不混入原150组geometry重构计数。

随机朝向采用每(fold,ID)预固定一次hash均匀角度，三model seed共享同角度，只修改原fallback；其余窗口路径和logits bitwise不变，全15参数/bool/scaler前后exact。UNIV selected全人口top1技能仍为−0.2123/−0.2266/−0.1605，对原始朝向的配对增益区间全部跨零，P1也全部跨零，未得到稳定改善。它不是对随机朝向分布的充分重复估计，更不能从一次实现中挑部署角度。[随机朝向结果](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/random_heading_supplement/SUMMARY.json>)、[随机补充审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/v3_gating_completion_20261009/random_heading_supplement/REVIEW.json>)。

## 10. 反方检查与下一步

ARS反方1：已在结果前修正“完整嵌套未见场景”、clip目标、非劣及数值单位问题。反方2：保留P1覆盖下降、mean-K物理副作用、ETH/HOTEL性能反例和G4 eligibility偏差。反方3：最强反驳是门控优化了SSE却牺牲难例完整性，且donor上游可能见过目标场景；这些事实阻止部署或因果根因宣称。三阶段均由同一助手执行，不是外部独立评审。

本轮推进到的分支是：**平均运动信息存在，但实际选中的模式方向和上下文响应不够稳定；单纯幅度收缩只能局部缓解，不能解决完整性目标。** 不能直接缩写成“generation无效”，也不能缩写成“scorer唯一失效”。

当前不进入T3-lite、不训练scorer、不恢复长训练，不扫描门控/朝向参数。下一次若继续，应先采用已有输出做一个严格的**决策与完整性定位**：固定人口比较top1/mean-K/强简单基线的SSE改善、同路径覆盖和五项motion的联合可行窗口，区分“降低平均误差”与“保留已有成功路径”的不可兼得部分，并保留G4共同mixed窗口的小样本限制。不是用GT选部署路径或训练oracle scorer。更大模型结构/损失变更仍需要精确设计及相应预算。

本轮G1–G7的低成本主体与前向补完有实际结果；原E0–E9整套、E8后端效率与被人类跳过的长训练仍不是完成状态。

AI参与：实验代码、算术检查和报告由同一AI助手编写；数据和既有模型来自项目保存材料，未编造外部文献或独立评审。
