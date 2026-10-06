# 参数层原primary反馈跨FIT/CAL：结果与阶段决策

## 结论

本轮完整完成。最重要的反证是：**18/18 checkpoint的变化组原primary参数梯度在FIT与CAL同向，而不是普遍反向。** 能通过真实newhead Jacobian读到质量反馈，不等于训练更新能稳定生成完整好路径。本诊断没有证明损失正确或错误，也没有证明某个decoder模块是根因。

上一matched诊断已确认九个final在FIT变化组完整好候选覆盖40.61%→51.50–53.87%，CAL却34.71%→30.86–34.57%，两split都有运动学退化。本轮使“因为变化组梯度一概冲突，所以无法迁移”的解释不再适合作为主解释。

## 设计、计算与审计

18个现有checkpoint，3表示族×3seed×constrained/final；FIT25912、CAL2879，28791显式unique ID。原weighted hardmin(ADE+.5FDE) primary，原FIT class weights；仅现行newhead45798参数求导，无optimizer/参数更新。原F16/wholedecoder/scorer全冻结。未来GT与分组只用于事后目标，forward仍只past/tokens/nativebase/speed；无DEV/test/新数据。

worker35368/service2728于2026-10-05 15:37:59.808617HK启动，15:40:39.184208HK完整SUCCESS，实际159.375591秒。唯一半程状态核验实际15:46:33.5915458HK（计划15:40:59.808617；控制写入及审批使检查延迟），发现taskReady/processnull/18done/COMPLETE/worker和exit双SUCCESS/stderr0。不重复半程或查询已完成PID。

生产18job/36输出SHA、原primary保存值和全部16组配对×9参数分支摘要重算、全weight/bool/scalerbuffer exact、445source及29983old完整初末SHA均PASS。CPU补充另实现保存vector算术重算，最大误差3.552713678800501e-15，36输出/18模型（模型SHA包含于source）/445source/29983old前后SHA PASS。此为保存梯度的独立算术实现，不是全部18模型再次反向，也不是外部独立评审。

生产peakRSS1.760/private2.568GiB，系统physical/commit最低17.214/16.796GiB；CPU补充private约.762GiB。约17GBhostRAM长期约束、process6GiB、系统physical/commit各6GiB余量保持，没有OS/pagefile/用户程序更改。两个只读JSON显示超长被工具截断、一个只读PowerShell投影语法错误已用窄字段投影纠正，未修改科学文件，不是生产失败。

## 全组结果与反例

cos>0仅表示原primary的局部Euclidean FIT下降方向对CAL一阶有利；不是概率、收益百分比、有限AdamW更新或coverage改善。

| 同组参数方向 | 同向checkpoint | 反向checkpoint | cos范围 |
|---|---:|---:|---:|
| overall→overall | 16/18 | 2/18 | -.235～.722 |
| stable→stable | 16/18 | 2/18 | -.248～.702 |
| transition→transition | 14/18 | 4/18 | -.025～.414 |
| change→change | 18/18 | 0/18 | .230～.935 |
| FIT overall→CAL change | 14/18 | 4/18 | -.225～.915 |

变化组constrained的cos .472～.935；final .230～.612。主变化组梯度norm不近零：FIT .176～.531、CAL .165～.651，不能解释为反馈完全到不了参数。跨表示族norm不比较能力或排名，单位受坐标参数化影响。

混组冲突仍存在：raw三个constrained的FIT overall→CAL change都反向，raw final seed3也反向。不能由18/18 change同向说所有组、所有训练方向一致，原训练使用混组及其他loss，不是只变化组primary。

### 全18明细

下表均为cos，时间列为checkpoint epoch。C-change为FIT overall→CAL change；clock为仅最后两clock输出行的change→change。全部16pair、所有分支和原norm/dot/方向导数详见completion_review/REVIEW.json，不筛选有利结果。

| checkpoint | epoch | overall | stable | transition | change | C-change | clock-change |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_seed1_constrained | 29 | 0.215 | 0.635 | 0.210 | 0.519 | -0.095 | -0.246 |
| raw_seed1_final | 80 | 0.165 | 0.226 | 0.097 | 0.498 | 0.291 | -0.727 |
| raw_seed2_constrained | 25 | 0.144 | 0.693 | 0.414 | 0.504 | -0.093 | -0.890 |
| raw_seed2_final | 80 | 0.143 | 0.266 | 0.125 | 0.272 | 0.027 | 0.568 |
| raw_seed3_constrained | 35 | -0.060 | 0.702 | 0.116 | 0.472 | -0.018 | 0.720 |
| raw_seed3_final | 80 | -0.235 | 0.117 | -0.022 | 0.336 | -0.225 | -0.024 |
| centered_raw_seed1_constrained | 5 | 0.669 | 0.502 | 0.054 | 0.919 | 0.842 | 0.985 |
| centered_raw_seed1_final | 80 | 0.278 | -0.242 | 0.398 | 0.524 | 0.363 | -0.139 |
| centered_raw_seed2_constrained | 6 | 0.627 | 0.439 | 0.284 | 0.920 | 0.858 | 0.826 |
| centered_raw_seed2_final | 80 | 0.257 | 0.199 | 0.375 | 0.463 | 0.297 | -0.350 |
| centered_raw_seed3_constrained | 7 | 0.619 | 0.625 | 0.202 | 0.928 | 0.828 | 0.617 |
| centered_raw_seed3_final | 80 | 0.283 | 0.581 | -0.025 | 0.537 | 0.251 | -0.126 |
| centered_own_seed1_constrained | 5 | 0.708 | 0.474 | 0.300 | 0.923 | 0.838 | 0.942 |
| centered_own_seed1_final | 80 | 0.285 | -0.248 | -0.021 | 0.612 | 0.391 | 0.325 |
| centered_own_seed2_constrained | 2 | 0.722 | 0.533 | 0.233 | 0.935 | 0.915 | 0.984 |
| centered_own_seed2_final | 80 | 0.149 | 0.252 | 0.091 | 0.230 | 0.094 | 0.173 |
| centered_own_seed3_constrained | 13 | 0.291 | 0.608 | -0.018 | 0.913 | 0.691 | 0.104 |
| centered_own_seed3_final | 80 | 0.287 | 0.427 | 0.240 | 0.433 | 0.318 | -0.483 |

## 模块定位能说到哪一步

变化组head、最后D2漂移和A2幅度输出行都18/18同向。因此“主质量反馈不能穿过head”作为普遍解释不成立，不能自动据此加大head或改共享结构。

clock输出行在constrained中2/9反向、final中6/9反向；overall配对final clock有7/9反向。值得保留为时序映射的局部线索，但clock范数/尺度与分支数量不同；clock是曲线中心/跨度，不是语义事件时刻，且sigmoid clock与终点耦合。三个final clock-change仍同向而CAL完整覆盖仍下降，反驳“仅修clock即可解决”。不能按CAL分支结果直接冻结一支、选超参或宣称根因。

state_encoder change方向constrained4/9反向，final1/9反向；Q/K各final4/9反向，value final2/9反向。方向并非全模块一致，也不是一个所有族/seed通用的坏模块。18checkpoint是相关架构/同数据重复，非18独立泛化实验。

## 对人类问题地图的更新

1. 输入表示/标准化局部闭环已经3/3reviewed；不再第四normalize/width/LR/epoch/profile变体。
2. 共享映射/迁移缺口是真实现象，但本轮未发现一致的主变化组梯度反向或参数反馈阻断；不能因此断言某种架构必需替换。
3. **下一讨论重点上调为“目标反馈与完整候选覆盖、运动学的对齐”，并横向检查有限优化/迁移。** 平均primary质量下降、至少一条同candidate ADE≤.2 AND FDE≤.4、全K20运动学保护是不同目标；hardwinner质量分配和其他正则仍需检验。当前结果支持研究这个缺口，不证明loss有错，也不证明换loss必有效。
4. scorer仍最后。继承F16 argmax是旧slot，不是newhead运动概率；生成稳定覆盖改善前不抢先训练scorer。

原major及全部组运动学1%保护不变，.3/.5只敏感性，不能删困难样本、拼分段oracle或复制候选凑数。本诊断不增formal计数，candidate-state3/3及response3/3reviewed，F16baseline不变，goal未达成。

## 下一步与停止条件

现有证据足以否定一个主解释，但**不足以指定一个可辨识、预计改善覆盖且无运动学副作用的下一newhead单因素训练**。直接根据cos选择clock或head冻结只是CAL驱动猜测；直接换multi-candidate loss又超过当前授权。遵循人类“实质阻碍先停等决断”，本轮在科学选型/权限门停止自动新训练，不制造第四局部变体，也不擅自暂停或删除automation。

建议人类决定以下二选一的路线，均需先精确预注册，不是已经启动：

- 较小范围：原primary、原smoothness/protection等原total分项的参数方向对照，外加预注册有限扰动后的同路径coverage/rescue-lost/motion检查。仍仅冻结已有head数值副本、无训练/newloss；它可区分primary代理不足与原total方向竞争，但不保证新架构解法，不无限追加诊断。
- 若愿意进入优化目标研究：具体批准新loss/assignment/多候选primary质量反馈单因素对照，保留原data/K20/frozen旧模块/major保护。需针对历史soft-set/energy/path-mixture失败设计，不重复“均分反馈”口号。当前没有这一权限，不能自动启动。

**状态：ANALYZED，等待人类明确选择下一科学路线/范围。** Automation保留ACTIVE但等待期间不查询已完成worker，不重复请求同一授权，也不创建新训练。

## 反方检查2（同助手内联，非外部审稿）

Major：cos同向不能否定泛化问题；它仅checkpoint处局部原primary的一阶关系，与finite训练历史、strict coverage及motion不同。保留上一matched证据。
Major：18/18同向不能说“loss正确”。仅反驳普遍主变化梯度反向/nearzero，保留混组及clock例外。
Minor：clock最终6/9冲突而其余3同向都coverage下降，不能只挑6个。已列全部明细，未选新模块/threshold。
结论PASS_WITH_LIMITATIONS，无科学阈值变更。

## 反方检查3（最终解释）

最强反论点：原total/AdamW、局部曲率/hardwinner切换、CAL构成及选checkpoint都可能使有限训练失败，即使主变化梯度同向。当前不能将差距唯一归因于损失或结构。
Major：CAL单university、三scene/windows相关、原F16见过训练资料、相关seed和表示族，结果非独立泛化/SOTA/心理意图/不可预测率。
Major：下一训练单因素缺少可辨识选择且目标更改未授权，应停止自动新训等人类，而不是以新名称绕范围。
结论PASS_WITH_LIMITATIONS；academic-research-suite用于证据/推断区分与反例审阅，未外部文献搜索或独立多agent审稿。

依据：RESULT.json、ARITHMETIC_AUDIT.json、FROZEN_AUDIT.json、RESOURCE_USAGE.json、completion_review/REVIEW.json，以及上一matched RESULT_ANALYSIS.md。

