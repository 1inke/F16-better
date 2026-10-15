# 实施设计（结果前登记）

## 问题与范围

判断 F16/native20 的收益是有效、情境相关的运动修正，还是主要来自 CV 已成功窗口、无方向的候选散布或场景相关性。采用用户提供的操作化方案，不把行为证据当心理意图。第一阶段使用现有五折三种子模型；不复用一个 seed 的输出冒充三 seed。所有新输出只在本目录，旧模型源码和结果不可改。

官方五折已有 checkpoint 过去已被评测：本次是预注册操作化诊断、不是第一次未见测试或独立确认性研究。原验证集选出的 best_selected 固定使用，不能在本次测试结果中更换 checkpoint。保留 ETH/UCY 相邻窗口相关、三种子及场景少的限制。

## 第一阶段 E0/E1/E2/E7

五折 eth/hotel/univ/zara01/zara02，seeds 1/2/3，原 8/12、dt=.4 米。训练统计使用既有正式 cache 的训练 sample_id（不含原验证）。每折阈值只从这批训练 GT/观测生成并写入 calibration.json，**全部五折校准完成后才读取任何测试输出**。

E0：CV-last、CV-avg3；CA 采用末两速度的差，训练加速度模长 P95 截断，未来增量 a×.8^t 的离散累加；Kalman CV 二维独立模型 Q=q[[.25,.5],[.5,1]]、R=r、dt 按一采样步，固定 q∈{.001,.01,.1}, r∈{.01,.1,1} 网格，仅训练 ADE+.5FDE 最小选择，ties 按网格顺序。报告最强训练选定简单基线与每种基线，不能测试挑最好再不披露。

E1：主 NonCV 为训练 CV-last FDE P75，强 NonCV 为 P90；CV-success≤P50。C1 末/首三速度均值比 outside [.75,1.33]；首均值为0且末>0为启动线索，两者0为无线索。C2 首三/末三速度向量夹角>15°，各模长≥.02m/step，否则方向未知。C3 所有有效邻居 CV 与目标 CV 在未来12步离散最近距离<1m。cue-positive 是代理，不是真正可预见性证明。驻止定义末步<.02m，方向指标不填假零。

训练速度 quartile × 邻居数 {0,1,2–3,4–8} 分箱冻结。CV-success 对照按 NonCV cue-positive 的箱计数、ID哈希确定抽样、无放回；不足箱不扩大阈值，记录未匹配数量。非CV/正线索/负线索/强非CV/驻止/匹配成功及全体都给窗口、track 分母。

E2：top1 用**原 frozen scorer logits argmax**，meanK 是20路径平均点预测，oracle不作理解证据。skill 为 ratio-of-sums；零分母 UNDETERMINED。余弦零模长剔除并报数，横/纵符号在 GT 非零且方向定义子集报告，另报标签多数类基率。1000 次速度×密度箱内置换 Rm，Monte Carlo p=(1+count(null≥actual))/1001；箱内窗口仍相关，置换结果只作为该条件零模型，track bootstrap 2000 另列。固定随机种子20261009。

P1 主终点是 NonCV cue-positive 的 skill（top1 与 meanK共同报告，用 Holm 校正两种读法×五折×三种子置换检验）；ratio-of-sums track 重采样95%区间，与同ID最强训练选定基线的 ADE/FDE 差配对。逐种子一致性不以窗口数量代替模型重复。

E7：等 K20 的 CV+训练经验残差对照，按训练速度 quartile 从训练 GT−CV 的局部速度朝向坐标残差整条抽样，再转入测试观测朝向；stationary 残差只从 stationary 池抽样，至少1条，不取测试 GT；三对照 RNG seeds 42/43/44，对照种子不是 F16 模型种子。固定完整覆盖阈值 (.1,.2),(.15,.3),(.2,.4),(.3,.6),(.4,.8)，.2/.4主阈值，不选择最有利阈值。非CV候选 δ 是训练 GT−CV 的 meanpath P75，固定。报告 minADE/minFDE/同path J/coverage、非CV candidate 命中、端点散布与 CV/模型误差 Spearman（相关不是概率校准）。

## 身份、统计与审计

从原 label 全20点 world-XY float32 rounded4 fingerprint 恢复 recording+pedestrian身份，原 frame 保留；未知/多 start_time 不当独立身份。若同窗口匹配同一 track 的多时点，track仍可用、时点记 ambiguous。如果身份不可唯一确定则停止需要身份的主推断，不把窗口ID当track。所有有限输出保存 explicit fold/sample_id/track/sourceframe、20路径和原logits，逐模型权重与 bool/统计 buffer 前后 torch.equal，源码/数据/模型 SHA 初末一致。

完成前再从保存路径重算 ADE/FDE/残差技能/分组/置换与 CI摘要（独立 NumPy读取而非第二次模型前向），所有 job 文件 SHA、manifest checkpoint绑定核验。明确独立算术不是外部评审。

## 后续阶段登记原则

E3–E6 的原始观测/变换与 OOD 范围在下一实现前固定；不从第一阶段的最有利窗口选例，使用 hash 固定全体或规模子集并保留分母。合成最近距离公式必须实际核验 closest approach，附录的 n*d 偏置若与相对速度不正交不能宣称恰好最近距离。所有负结果继续执行其余预定诊断，不因简单基线失败跳过。

训练消融只在新目录新实验中完成，不修改/继续旧 checkpoint。E9 外部阶段需要真实来源、单位和划分审核；不可用时列为限制，不无限下载或以伪造数据替代。当前用户批准文件所列系列，不批准无关新 loss、scorer 创新或以更大模型寻突破。

## 反方检查1（同助手，不是外审）

PASS_WITH_LIMITATIONS。Major：既有 checkpoint 与测试已历史评测，不能称全新确认性；NonCV 用GT定义有选择效应；原 logits不是校准概率；样本级置换忽略track相关；五折训练集有重叠。已用同子集配对、track区间、训练冻结、多重比较和明确探索限制处理。Minor：matching 箱缺少成功对照时必须显示缺口。最强反驳：即使技能正且干预有响应，也可能只是低阶运动规律，不能证明行人意图；E3 重训和 E4 方向/剂量证据仍必须完成。
