# E6：位置与速度记忆检验（输出前登记）

范围为现有五折 ETH/UCY train cache IDs 和完整固定 test IDs；不新增数据、不改旧模型、阈值或候选。只有训练未来可成为残差库，测试未来仅评价。确定性 kNN 不是三个模型 seed 的新复现。

14 个策略：五折 × position_velocity / velocity_only（10）；ZARA01/ZARA02 再各两种特征剔除对方 recording（4）。固定 k=32，不搜索 k、距离权重或测试结果。Euclidean 距离在每个实际训练库自己的 float64 mean/std 标准化空间，std<1e-6→1。近邻按距离再原 sample ID 升序，边界同距离必须 radius 全收集并稳定选32；统一权重平均训练未来相对 CV-last 的 12×2 世界坐标残差，再加当前测试 CV-last。位置用原世界坐标米，速度为末一步米/step；非跨 recording 地图对齐，此限制必须报告。velocity_only 是位置增量价值的反例对照，不是相同有效信息或容量。

严格身份：每 train-cache ID 的原 target20世界坐标（float32四位小数）匹配该 fold 正式 train/val 标签连续20行。recording+ped 必须唯一，开始时刻多匹配保留并记录。ZARA01 剔 crowds_zara02、ZARA02 剔 crowds_zara01；不自动扩大到zara03，剩余其他同族/同地点依赖需报告。原测试身份来自已锁 stage1，训练/测试完整20路径身份重叠与 recording 组成全部保存。未知身份或多 recording 无法确定即 STOP，不靠窗口索引猜来源。

预检先 all old related SHA、所有train/test IDs/完整身份/剔除结果、统计 chunk32/256一致≤1e-11；known exact-zero/tie fixtures，实际固定首32 queries KDtree 与全训练库暴力 stable-nearest32完全相同、预测数值≤1e-10。全train统计只各策略train；测试不参与fit。No GPU/backward/2epochRNG，因为没有模型训练。实际8×128查询和全指标时间/峰值/磁盘估算后定生产ETA。只一个独立OS零trigger worker，ONE startup / estimated50% / completion。

每策略全test保存 explicit IDs/track/masks/GT/pred/CV/selected32 trainIDs/distances/trainstats/trainmembership SHA。以 stage1相同 masks/calibration 的 overall/nonCV/nonCV_cue_positive/stationary 等分母报告 ADE/FDE、技能比值、同一单路径 .2/.4覆盖、纵横方向，不把 kNN1 与 oracleK20 等预算混淆。配对 strongest_simple、三个原模型 top1 和meanK，2000 target-track bootstrap，探索性不虚构多重校正显著。没有重训 “F16排除兄弟场景”——此部分仍待后续，不能用 kNN 排除代替模型排除。

资源 process6GiB/物理和commit各reserve6GiB，单fold/128查询有界、内部guard非对话poll、不OS/pagefile/userkill。全部新增只本folder，继承旧scientific digests不得覆盖，锁源后不编辑。完成需双SUCCESS/14done-outputSHA/独立保存算术、identity/source initialfinalSHA/RESOURCE，再阶段分析，整套仍unfinished。

反方1（同助手非外审）：位置标准化跨不对应地图坐标，不是共同地图空间；位置kNN好不能独证神经模型记忆，坏不能排记忆；训练库剔除改变大小与分布，不纯兄弟因果。保留速度-only/全分母、train recording组成/未知时间、所有seed与反例，禁止从test挑k或场景。
