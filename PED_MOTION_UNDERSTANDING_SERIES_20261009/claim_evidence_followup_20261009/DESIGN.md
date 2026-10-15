# N0/N1/N2/N3/N6：低成本建议落实

依据2026-10-09新建议文件与人类继续指令，维持“不投入长时间、现有模型优先”决定。只读原保存输出/原训练观测，CPU计算；无模型前向、optimizer、新训练、外部数据，不恢复暂停E3。N7/E9重训跳过；E8需要独立真实硬件/MAC口径，暂不冒称完成或虚构N>9输入。

本次都是**结果后提出的新探索分析**。旧结果已见，不把现在冻结配置追溯称为确认性。当前文件在计算新N1/N2/N3/N6结果前保存；后续不根据结果改阈值/挑seed。原P1分层、coverage ADE≤.2 AND FDE≤.4、全部五运动学gate不变。

- N0：从已审核原summary提取四简单基线、top1/meanK全部原子集绝对ADE/FDE、skill与相对训练选择strongest_simple的增益；填E7同K20模型/经验CV残差RNG三对照的原阈值扫描、coverage、spread—error关联和既有CI。RNG控制不是模型replicate，未校准的logit不是概率。
- N1：同fold固定stage2窗口，三seed对(1,2),(1,3),(2,3)以K20完整12步mean-Euclidean Hungarian配对。d_seed是这些配对路径/同配对endpoint距离的窗口池中位数，d_T来自先前完整配对补充。报告median(d_T)/median(d_seed)，1.25仅冻结为**描述性参考**，不是统计等效界、置信区间或“已证明等变”的阈值。再按原seed候选内部190对完整meanpath平均距离、各候选相对CV-last的meanpath残差幅度归一化；严格零分母标未定义，不加floor。保存各seed/变换/窗口，尺度另列不要求刚体等变。
- N2：每fold仅原train IDs，R为观测末帧“最近有效邻居距离”的中位数（无邻居不拟R，重合距离允许0）；密度是末帧≤R的有效邻居个数。原训练密度1/3,2/3分位作为边界；离散计数重复边界去重，按右侧归箱，如实报告不足三箱，不搜索R/分位数使结果显著。速度使用原训练校准四分位边界/单位步位移，换m/s仅展示。P1原NonCV-cue-positive为主，all为辅，逐密度箱和速度箱×密度箱报告top1 residual skill及2000目标track bootstrap；不同fold不合并坐标或窗口为独立。技能—密度斜率以各箱的skill对箱mean-density、按人口加权OLS计算并同track bootstrap。原数据valid≤9所以密度截断，不能外推真实全场人群密度或因果互动不足。
- N3：原CV预测12个未来步，选距离最近的有效邻居/时刻（先slot后time稳定tie），原C3=该距离<1m。away为该时刻邻居→自身单位向量；恰好零距离未定义、明确计数。此方向未必与目标速度正交，故名为away投影，**不冒称纯横向**。训练C3且g=(GTend−CVend)·away>0的合法窗口g之75分位固定δ_g；训练无合法正g则该fold阻碍不改阈值。测试g≥δ_g且C3定义事件，其他C3、非C3合法方向为对照，保留无邻居/零方向排除分母。复用stage2真实noNbr，with与without各自原argmax，另meanK；m为end差away投影。报告正向比例、m均值、事件m/g均值和2000track CI（控制组g可能零/负，不强算m/g）。事件定义使用GT仅posthoc，不能证明实际因果避让；不同组可有速度/构成混杂。
- 合成E4重分析：所有15model×2side×5distance×top1/meanK=300项，报告正比例/Wilson区间、响应均值和track CI、零响应/tie分母。附n_eff=track数、双侧alpha.05/power.8正态近似MDE=2.8*sd(track均值)/sqrt(track数)，不是实际功效证明、不是把不显著改成有效。Wilson按window仅描述，重复窗口相关性以track CI为主。
- N6：全部原P1人口同seed top1/meanK/full-path平方误差最小单一candidate oracle skill。Oracle必须选**一条完整轨迹**，不拼ADE/FDE/时间段；另列原Q=ADE+.5FDE排序与samecandidate严格覆盖。报告top1的Q-quality rank/完整SSE oracle的旧logit rank、好candidate存在却未选比例、完全无好candidate比例；oracle减top1的skill gap只是选择上限，并不排除生成缺口。按position_velocity kNN好于top1的窗口与其余P1对比观测速度/密度/world位置均值，逐fold/seed探索；世界坐标跨recording不对齐，不当因果记忆证据。

bootstrap固定rng20261009，2000次，目标track聚簇，每fold/seed分别，不伪造seed独立窗口。所有原父模型/源码/结果不改。新source+实际用到原输入的原SHA绑定，初末核验；过程≤6GiB，physical与commit各预留≥6GiB，CPU单线程/有界chunk，计算协作预算600秒，超出保存partial而不是加范围。无daily/poll/旧PID查询。

ARS反方1（同助手非外审）：原建议C9说等K20的增益“来自多给名额”不成立，等K已经排除直接K数量差，不能把等K20与K24混为一谈；N6oracle正不足以排除生成缺口；N1seed底不能消除固定模型自身的不等变。以上均在交付表校正后实施。
