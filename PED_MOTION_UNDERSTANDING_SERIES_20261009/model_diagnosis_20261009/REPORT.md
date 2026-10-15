# 模型诊断与运动理解：执行报告

日期：2026-10-09。依据 [用户提供的新方案](C:/Users/linke/Downloads/ped_model_diagnosis_and_understanding_plan.md)。本次遵循“优先现有模型与短实验、跳过长时间训练”的最新预算要求，不恢复暂停的 E3。全部新分析均为探索性。

## 结论先行

**目前最强的分支不是“scorer 无效”或“数据本质不可预测”，而是：UNIV 上存在较明显的过度修正，且简单的训练场景动力学探针能利用模型没有充分转化为有效预测的信号。**

朝向兜底确实关联几何一致性问题，但在全 UNIV 人口中仅占净 SSE 劣势约 **0.52%–0.74%**，不能解释整体失败。候选分散度也并非完全没有自适应：它区分 UNIV 恒速失败的 AUROC 为 **0.826–0.851**。scorer 在严格完整轨迹标准下明显胜过平方距离 medoid，但在宽阈值的非恒速难例中出现反例。下一步应优先验证**训练/验证场景确定的保守残差收缩或门控**，而不是盲目更换选择头、添加复杂交互网络或恢复大批量重训。

本轮已完成零训练诊断和固定轻量探针。冻结前向干预仅完成 ETH/HOTEL 的六个模型；到 UNIV 时，邻居镜像构造的几何一致性误差 **2.1119129587×10⁻⁵ m** 超过预注册 **2×10⁻⁵ m**，立即停止，未放宽门槛。五场景 T2 **未完成**，不能用部分结果宣布朝向修复或普遍避让能力。

## 1. 执行清单

| 项目 | 状态 | 实际实施及未完成原因 |
|---|---|---|
| 模型事实核对 | 完成只读代码/可见元数据核对 | 见第2节；不把当前源码读取当作所有旧训练生命周期的重新认证 |
| D1a 朝向兜底贡献 | 完成 | 全五fold×三seed，all/P1/fallback/no-fallback/起步分解 |
| D2 do-no-harm 与收缩 | 完成 | top1/meanK、严格同路径混淆表、16个固定λ点及解析λstar/track CI |
| D3 分散度自适应 | 完成 | Jensen逐窗口量、固定物理箱、CV失败AUROC及track CI；不在测试上调分箱 |
| D6 强选择基线 | 完成 | top1、平方距离medoid、最近CV、随机期望；五τ、窗口内AUROC |
| D7 kNN与域偏移 | 完成 | 新位置-only K32，复用已审计速度-only/位置速度；训练与测试速度/密度/停留KS距离 |
| D4(a)(b) 动力学探针 | 完成固定线性探针 | 训练场景PCA2+ridge；自身48维 vs 加邻居62维；不是神经模型训练 |
| D4(c) 同场景位置探针 | 未做 | 当前LOSO训练bank不含被留场景；需另立同场景recording训练/测试协议，不能把测试窗口搬进训练来补此项 |
| D5a 邻居数据探针 | 完成固定线性探针 | 原C3+away-defined训练子集拟合g，测试同定义；不称信息上限 |
| D5c 真实剂量响应 | 完成保存输出版本 | 原真实noNbr输出；训练距离四分位分箱、窗口/track区间、连续斜率；投影是原closest-away而非纯侧向，限制保留 |
| D1b 朝向替换/敏感性 | 部分6/15模型 | ETH/HOTEL完整变体及保存审核；UNIV/ZARA未执行，不是五场景完成 |
| D5b 邻居侧向镜像 | 部分6/15模型后STOP | UNIV几何构造未通过既定容差；未继续、未自动换阈值 |
| T3 针对性神经训练 | 未启动 | 本轮仍受低时间预算限制；先报告与核实诊断，不恢复旧E3，不自动新loss/scorer训练 |

## 2. 核对到的模型事实

1. **输入坐标**：`normalize_to_focal_heading` 以目标末观测位置为原点，将整个scene（包括邻居）旋转到目标三步位移朝向；随后每agent观测均值中心化。绝对世界平移信息由构造去除，平移近似一致不能单独归功于模型学会了几何。它仍可看到相对运动/相对邻居，而不是“没有任何位置信息”。[朝向实现](<C:/Users/linke/Documents/New project/FRONT_ENCODER_SERIES/train_front_variant.py:343>)、[实际推理接口](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/stage1.py>)。
2. **回退**：三步位移模不超过0.03m时使用固定世界x轴；同一变换也用于邻居。这不是旋转协变的低速定义。[实现](<C:/Users/linke/Documents/New project/FRONT_ENCODER_SERIES/train_front_variant.py:343>)。
3. **候选与logits**：H3确定性20个mode、三个锚点插值，V2增加有界锚点修正。实际forward中score_head使用`mode + source_anchors`，在V2修正前计算；因此旧排序与修正后候选不是完全同一个几何输入。这是结构事实，**不是已经证实的失败根因**。[decoder](<C:/Users/linke/Documents/New project/CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py:79>)。eval没有每次采样20条未来，初始化随机性不等于推理随机性。
4. **训练监督**：可见ModeQueryLoss根据ADE+加权FDE挑hard winner，winner回归为时间加权SmoothL1，分类包含基于GT quality gap的soft target/可选hard CE与diversity等。当前锁定配置winner FDE权重0.25、分类权重0.5，不能混入此前canonical新head的ADE+0.5FDE协议。UNIV seed1可见H3/V2 contract另有oracle coverage项0.2、reflection增强、无rotation增强；V2 contract记录仅26,514 trainable/321,988 total，不能简单称为所有部件联合重训。[loss源码](<C:/Users/linke/Documents/New project/FRONT_ENCODER_SERIES/locked_mode_query_loss.py>)、[配置](<C:/Users/linke/Documents/New project/OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml>)、[H3 contract](<C:/Users/linke/Documents/New project/OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/univ/h3/f16_temporal_social_cross/seed_1/val_scene_-1/locked_backend_contract.json>)。这段仅核对可见源码/元数据，不重新证明所有历史run源绑定。
5. **地图**：当前加载接口明确关闭pretrained/cached UNet，forward输入为观测位置、valid数量和graph_loc，没有实际地图/出入口标签。不能从经验边界proxy声称模型读到了可行走区域。[加载接口](<C:/Users/linke/Documents/New project/REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py:111>)。

## 3. D2：UNIV过度修正是直接观察，而非只有间接疑点

严格成功始终定义为 **同一完整path ADE≤0.2m AND FDE≤0.4m**。这里的strict CV pass不同于旧按训练FDE分位数定义的CV_success，未混分母。

| fold | top1全人口λstar（三seed） | P1 λstar（三seed） | CV严格成功被破坏率（三seed） |
|---|---|---|---|
| ETH | .868 / .891 / .845 | 1.069 / 1.056 / 1.074 | 12.90 / 12.90 / 19.35% |
| HOTEL | .639 / .552 / .534 | 1.321 / 1.063 / .996 | 9.04 / 10.26 / 10.61% |
| UNIV | **.226 / .237 / .261** | **.325 / .351 / .390** | **24.34 / 26.16 / 27.48%** |
| ZARA01 | .569 / .587 / .611 | 1.005 / .995 / 1.208 | 43.76 / 38.73 / 44.20% |
| ZARA02 | .768 / .820 / .775 | 1.117 / 1.222 / 1.218 | 7.32 / 6.17 / 6.91% |

UNIV共24,334窗口/722tracks，其中CV严格成功5,069窗口：top1破坏 **1,234/1,326/1,393**，救回原CV失败 **860/871/906**。全人口top1相对CV残差skill为 **−.1492/−.1572/−.1326**，三个track CI均为负。seed1全人口λstar=.226，CI[.194,.260]；P1=.325，CI[.272,.378]。CV严格成功组seed1 λstar仅.0134，CI[−.0052,.0318]。

但CV成功组只解释UNIV净SSE劣势约 **10.0%–12.3%**，ADE净劣势约 **28.6%–32.1%**。因此“全是把简单窗口修坏”也不成立：CV失败/非恒速人口中仍有无效或过大的修正。

λstar来自测试GT，只是诊断：不部署、不用来挑模型、不把曲线最高点当新方法成绩。ETH/ZARA的P1 λstar约1或更大是反例；不能把全fold统一缩成UNIV的.25。完整表、16点曲线与meanK结果见 [D2.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D2.json>)。

## 4. D1a：兜底解释几何问题，但不是UNIV整体失败主因

UNIV fallback人口 **2,076窗口/50tracks**，非fallback **22,258窗口**。fallback净ΔSSE仅占全UNIV负收益约 **0.52%–0.74%**；ADE净劣势份额约 **3.70%–5.15%**。去掉fallback后skill仍为 **−.1498/−.1582/−.1332**，三seed CI全部为负。

固定起步定义=原last-step位移<.02m且GT终点距离超过训练CV-FDE q75；UNIV338窗口/60tracks，fallback129/21tracks，非fallback209/42tracks。起步fallback skill **−.0453/−.0418/−.0426**；起步非fallback **+.0006/−.0507/−.0384**。这不是随机分配，组间不同不能当作替换朝向的因果收益。尤其seed2/3非fallback起步也差，不能把起步问题全归给兜底。[D1a.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D1a.json>)

## 5. D3：分散度有信号，不能笼统称为未自适应

逐窗口验证 `random_SSE = meanK_SSE + spread`。候选不是按后验采样训练，spread/error比不是概率校准证据。

| fold | spread区分CV严格失败AUROC（三seed，全人口） |
|---|---|
| ETH | .886 / .920 / .891 |
| HOTEL | .825 / .777 / .865 |
| UNIV | **.850 / .851 / .826** |
| ZARA01 | .504 / .519 / .543 |
| ZARA02 | .696 / .782 / .883 |

UNIV seed1 AUROC=.850，track CI[.827,.869]；all spread/error=1.896，CI[1.792,2.010]，P1=1.245，CI[1.175,1.320]。P1内CV严格失败是单一类别，所以该子集AUROC **未定义**，不是0.5或0。ZARA01接近.5是更值得追的反例，而不是把UNIV全人口说成没有自适应。

固定物理SSE箱保留空箱，不用测试分位数调整结果；各箱Σerr/Σspread和各情境均值见 [D3.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D3.json>)。区分能力可以部分来自速度/误差尺度，不能据此宣称概率可信或“理解不确定性”。

## 6. D6：scorer不是只有居中性，但难例/阈值依赖仍明显

同一窗口里有good/bad两类时算logit AUROC；有好候选窗口内算命中率。medoid为**全12步平方L2**意义上最靠近meanK的原候选，仍是一条完整slot，不是meanK轨迹。

UNIV严格τ=.2（FDE=.4）：

| 人口 | scorer条件命中率（三seed） | medoid条件命中率（三seed） | 窗口内AUROC（三seed） |
|---|---|---|---|
| all | 53.6 / 52.5 / 53.3% | 10.7 / 9.8 / 10.1% | .899 / .900 / .896 |
| P1 | **10.9 / 11.2 / 11.7%** | **4.3 / 6.4 / 6.1%** | .680 / .685 / .685 |

seed1 all中scorer−medoid为+.4297，CI[.3647,.4843]；P1为+.0654，CI[.0360,.0933]。最近CV规则在all命中54.2%（seed1）接近scorer，但在P1命中0%，不能把“最靠近恒速”当作全情境方案。

**保留反例**：宽敏感性τ=.4/FDE=.8的UNIV P1，seed1 scorer命中6.83%、medoid10.30%、随机期望7.00%；scorer−medoid −3.48pp，CI[−5.15,−1.99]pp。严格条件下的优势不意味着选择头在所有quality definition下都最优。原严格标准不切换，五τ完整表见 [D6.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D6.json>)。

## 7. D7与D4：UNIV不是只能依赖位置，也不是已证实不可预测

UNIV P1（8,793窗口/653tracks）的单一路径skill：速度-only kNN **+.0546**，位置+速度 **+.0644**，位置-only **−.1389**；模型top1 **−.0622/−.0580/−.0390**。seed1速度-only对top1 skill增益+.1168，track CI[.0902,.1423]。位置-only在跨场景世界坐标上更差，不能把原kNN优势说成“地图记忆”。[D7.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D7.json>)

UNIV训练/测试速度KS距离=.2917、密度=.1358、停留=.1211；这些只是组成/分布差异，无窗口iid的KS显著性断言。其他fold也存在较大分布距离，所以shift存在不等于UNIV失败唯一原因。[D7_SHIFT.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D7_SHIFT.json>)

固定PCA2+ridge在其他训练场景拟合；PCA2保留训练残差方差 **96.73%–96.81%**。自身48维/加邻居62维，标准化与basis全训练限定、固定alpha=.01，无测试选超参。

| fold | P1自身探针skill | P1自身+邻居探针skill |
|---|---:|---:|
| ETH | +.0722 | +.0348 |
| HOTEL | −.0968 | −.0857 |
| UNIV | **+.1161** | **+.1343** |
| ZARA01 | +.1432 | +.1747 |
| ZARA02 | +.1670 | +.1924 |

UNIV seed1模型对照：自身探针skill增益+.1783，CI[.1588,.1979]；加邻居+.1965，CI[.1760,.2165]。UNIV非恒速C1自身probe +.0804、C2 +.1241，均高于对应模型负skill。其余seed比较全部保存；探针本身每fold只拟合一次，三seed不是三次独立probe重训。ETH/HOTEL模型明显优于这两个探针，是重要反例。

这是“已有观测有可利用信号”的有限正证据，**不是线性模型信息上限**；也没有要求所有严格coverage/五motion通过，所以不升级baseline。[D4.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/probes/D4.json>)、[训练统计](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/probes/FIT_INFO.json>)

## 8. D5a/D5c：不支持立即宣布交互不存在或换数据

D5a固定训练C3+away-defined、预测GT endpoint偏离g。UNIV test20,723窗口/696tracks：自身probe相对训练均值skill −.00959，加邻居 −.00515；差值+.00444，探索track CI[.000023,.00897]。微小改善不等于可靠避让；其他四fold差值CI均跨0，也不能据此宣称它们本质没有交互。没有多重检验确认性结论。[D5a.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/probes/D5a.json>)

D5c原真实noNbr输出、训练距离分箱：UNIV seed1 top1响应随CV最近距离的连续斜率 **−.00250**，CI[−.1140,.1089]；GT投影斜率 **−.4194**，CI[−.7258,−.1063]。ZARA01/02 seed1模型斜率分别−.1969/−.1862，探索CI负；不能只保留UNIV负结果，也不能把closest-away投影叫纯侧向真实反事实。所有seed、meanK与四箱人口保留，连续斜率是T0后的补充探索，不retroactive primary。[D5c.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D5c.json>)、[斜率](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/D5c_SLOPES.json>)

## 9. T2部分结果及停止原因

ETH/HOTEL各三模型完成原始/threshold.01/.06/long7/backtrack，加固定四orientation敏感性、60度全scene旋转、镜像与邻居镜像，原GPU128 baseline cache gate通过。逐模型代码中的before/after参数/布尔/statbuffer `torch.equal` 断言在打印六个完成标记前执行；**全15模型FROZEN_AUDIT尚未生成**，不冒称具有完整生产审计。

HOTEL原始rotation bank-matched P95三seed **.2675/.2613/.2494m**，threshold.01 **.1953/.2320/.2291m**，backtrack **.1909/.2329/.2295m**；threshold.06反而 **.3328/.3239/.2853m**。这只是test-time OOD替换，不能挑最好阈值部署或称为已经修复。

D5b在target最后速度轴有定义的C3窗口：ETH top1 A=−.251/−.135/+.018，HOTEL=−.227/−.306/+.019。第三seed A近0，但逐窗口绝对反对称失败比仍约 **.605/.632**，说明全局A≈0可能来自抵消，不等于真正侧向响应一致；负A也不是自动代码错误。[部分几何](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/partial_forward_review/D1b_GEOMETRY_PARTIAL.json>)、[部分侧向结果](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/partial_forward_review/D5b_PARTIAL.json>)

UNIV镜像rawobs转float32后实际CV最近距离差 **.0000211191295868m**，gate **.000020m**，未通过。原因可能涉及世界坐标/float32变换舍入，但尚未独立定因。没有把浮点误差当评价epsilon、没有擅自加宽gate；该科学几何对照停止，代码/失败/全部partial输出保留。[FAILURE.json](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/forward_diagnostics/FAILURE.json>)。之后只有CPU对六模型保存结果的算术审核，无GPU重试。

## 10. 完成与审计口径

- T0+saved D5c：约 **60.20s**，实际119相关输入SHA前后不变、85个保存NPZ；D1a210/D2 420/D3 30/D6 150/D7 90/D5c120行，另15分布记录。
- 轻量探针：约 **15.19s**，57相关输入SHA、五fold训练统计/参数/预测保存；不是F16训练或原E3恢复。
- 保存审核：**1,170,007**数值检查，最大误差8.88×10⁻¹⁶；补充验证所有probe训练normal equations/预测重构、全部D7指标/KS、D5c slope：**1,642,643**检查，最大7.77×10⁻¹⁶；部分前向保存审核187,380检查，误差0。逐数组计数不是几百万独立科学实验。[主审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/REVIEW.json>)、[补充审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/SUPPLEMENT_REVIEW.json>)、[部分审核](<C:/Users/linke/Documents/New project/PED_MOTION_UNDERSTANDING_SERIES_20261009/model_diagnosis_20261009/partial_forward_review/REVIEW.json>)
- 审核是CPU保存算术，不是外部独立评审，不是所有模型第二次前向、所有nearest search或所有bootstrap再次计算。D3 AUC CI未完整二次bootstrap，point已独立rank复核。
- T0/private peak .627GiB，probe .465GiB，前向失败前生命周期2.918GiB；各阶段physical与commit均保留6GiB以上。未调OS/pagefile、未关闭用户程序、未恢复暂停worker或app automation。

## 11. 更新后的理解记分卡

| 能力 | 当前更精确的判断 |
|---|---|
| U1 不多余修正 | UNIV/ZARA01有明显破坏；但UNIV失败不只来自CV成功组。UNIV λstar低，其他fold P1不能照搬收缩 |
| U2 线索预判 | UNIV固定自身历史probe和速度kNN优于模型，否定“已有信息一定不够”的强断言；ETH/HOTEL模型优势保留 |
| U3 几何一致 | fallback是具体实现问题、不是UNIV整体主因；T2只部分，未确认哪一种替换可稳定修复 |
| U4 分散度适应 | UNIV全人口有较强区分能力；ZARA01较弱。不把best-of-K spread/error当概率校准 |
| U5 邻居特异性 | probe增量弱/不稳健；部分侧向结果存在偏置和抵消，五fold检查未完成，不宣布数据没有交互 |
| 选择 | strict all/P1 scorer胜medoid但远未完美；宽难例反例。不能优先盲换scorer |

## 12. 下一步建议与权限边界

优先窄地验证**保守残差校准**：只在训练/合法验证场景确定收缩或门控，再在固定五fold all/P1/CV成功人口评估破坏/救回、完整coverage与五motion；不使用本报告测试λstar作为参数。不重新跑旧90fits、不盲改loss/decoder/scorer，不因诊断正结果自动扩大训练。

朝向与邻居镜像需要先把原容差下的真实几何构造核实，保留旧失败并独立记录数值修复；本轮没有继续T2或训练pilot。T3的长投入仍按用户要求跳过。若做短pilot，应另立单因素、固定训练/验证选择与实际资源ETA，不等于恢复原长任务。

### 反方审查2/3（同一助手不同阶段，不是外审）

最强反驳：探索性测试λstar和固定线性probe并不证明某种新loss/结构能有限更新改善；good定义/人群组成/域偏移/选择过程相互耦合。报告保留ETH/HOTEL反例、阈值敏感性和T2失败。CI按track聚类，但五场景、重复窗口/seed、模型历史已经评测与自适应多轮分析限制仍在；不作SOTA、心理意图、不可预测率或decoder唯一根因结论。

编辑/伦理审查：所有未完成项明确列出；没有把partial6当full15、没有从测试挑阈值部署、没有删失败。AI助手完成实施和算术核对；无外部审稿人。剩余关键缺口为T2数值一致性与训练/验证限定的校准验证，不能凭本轮执行成功升级baseline/完成整个项目。
