# v2 A1–A6：零训练证据闭环

依据人类2026-10-09批准的Downloads/ped_claim_evidence_v2_and_next_experiments.md。优先现有保存结果，不恢复暂停E3，不新训练scorer，不改旧源、输出或评价门。新的分析均探索性，配置在本轮计算前保存，不声称确认性注册。原同candidate ADE≤.2 AND FDE≤.4主标准与五运动学门不变；新阈值仅敏感性。

按A1→A2→A3→A4→A5→A6执行。CPU/NumPy/SciPy、无torch/model forward/backward/optimizer；输入显式ID/track与既有done SHA匹配，初末输入完整性审核。fold间不同样本不当配对；三seed同人口分别报告，不把它们当三份独立数据。track bootstrap 2000、seed20261009，CI保存.025/.5/.975；不新造确认性多重比较p。

## 固定定义和必要校正

1. A1保存all/P1四读法人口sum SSE技能与同track配对top1-random差。随机读法是均匀随机完整候选的期望SSE，不是meanK轨迹。严格rank原代码含自身，连续无平局随机期望19/40=.475；新主rank排除自身、等误差计半，随机期望严格.5。保存两者。Jensen分解每window independently验证。
2. A2全部五阈值、所有fold/seed all与P1保存无好/有好没选/已选好三项及trackCI，三者和1；τ=.20是原主标准，不因曲线结果改变主标准。
3. A3沿用stage1原C1末初速度比/C2角度/C3最近12步CV小于1m。分别报告nonCV中各重叠线索及完整8个互斥bit patterns，不能相加重叠分母作归因。在无近邻P1内按原训练速度和CV FDE边界分层。kNN优势仍按原ADE比较，报告类别组成，不能把相关当唯一动力学或交互原因。
4. A4UNIV/ZARA02轮转dT>P95，保存所有selected窗口feature。仅有轨迹坐标，没有真实walkable地图边界；用各recording输入位置经验axis-box归一化边缘距离作描述性代理，不声称实际场景边界。镜像按原残差幅度三层，同时报告mean/median/P95及零分母。tail选择本身是事后诊断，不把tail-local差当独立确认性。
5. A5 TRAIN合法C3且GT-away g>0的50/75分位都报。测试GT只事件标记，三control noNbr/farNbr/replaceNbr和两读法分别算base-control响应。报告事件/track分母、window Wilson描述性、trackCI、approx MDE、undefinedaway/zeroresponse；旧事件75分位对齐。不因远移/替换更漂亮选对照。m/g仅eventg>0，away不一定纯横向、noNbr与替换都可能OOD。
6. A6“3m内邻居数”仍离散，本轮主变量用last-frame所有有效邻居Gaussian kernel之和，R和均值/std只TRAIN确定，无邻居density=0。原两箱N2不替换。原始ΔSSE为因变量，density每训练SD为自变量，控制原speed bins/recording；pooled进一步fold×speed及recording fixed effects。track充分统计bootstrap、无窗口skill比值。关联不等于因果，真实场景组成仍混杂；设计欠秩/无density变异标不可辨。

## 资源/范围

process≤6GiB、physical与commit各≥6GiB余量、内部guard、最多600秒CPU计算；不用OS/pagefile/kill。每次onefold/onemodel候选，不无界缓存。短CPU任务同步完成，无新后台worker/重复half/旧PID查询。B1/B2fresh和E9按当前时间预算跳过，E8单硬件短评估另作可行性边界，不虚构第二硬件或N>9。whole E0–E9不完成，baseline/cycle11/旧goal不变。

ARS反方1（同助手，非外审）: A1 meanK优于top1不足以证明旧排序反信息；rank自身计数会偏移随机基准，已校正。A2宽阈值改善不能消除严格目标未达标。A3标签重叠不可单根因归因。A4没有真地图，不能从经验位置边缘直接推出绝对坐标根因。A5两quantile共同结果不能升级为因果避让。A6Gaussian密度与速度/记录因素仍 observational。PASS限定探索性保存审计。
