# 候选运动相对于已观测运动的偏差：预注册第二对照

状态：DESIGN_ONLY，尚无生产worker、GPU预检或正式计数。下一实施规划节点为2026-10-05 09:45 HK；不是实验完成ETA。必须先完整核验candidate-state第一项，且实际全部预检通过后才启动。本研究仍candidate_state_20261005，每3新正式完成总review；本设计不是第四profile/系数变体。

## 问题与证据边界

第一项只比较state32可用性。严格同路径高质量覆盖有小幅增长，但change均值收益约1.37%且native/freshcontrolCI跨0，seed3恶化、全组平均加速度保护失败。名义同参数并没有消除masked输入第一层数据梯度为0的有效容量差异。候选状态可以帮助少数窗口，不等可靠运动理解；source/results/全部历史仍保留。

下一问题不是加宽head，而是：双方都实际读取候选状态时，将候选预测运动表达为“相对于目标最后已观测速度的偏差”，是否比原始预测状态更利于共享修正映射？原observed context/mode/tokens已有运动信息，这不是增加新现实信息，不证明原模型没有利用该信息。此设计由历史探索提出，不能当独立确认研究。

## 唯一注册因素：条件运动中心化

zero=实际原始native-state32；mean_std=相对于最后观测速度中心化的同32维state。双方state分支均真实非零可训练，保持原45798名义参数、32→32→32 GELU stateencoder、原context64+mode64/tokenattention以及192→128→64→6 head、learned sigmoid persistent profile。不改时间自由度、profile、loss、clip或scorer。

原state32：12预测速度24、均速2、末速2、终点位移2、平均加速度幅值1、首末预测速度signed heading1，origin0/dt.4。只从immutable nativeK20计算。

从原prepared observation[:16]恢复8×2过去位置（原focal-heading坐标），v_obs=(past[-1]-past[-2])/.4。固定reference32=[重复12次v_obs的24值, v_obs2, v_obs2, 4.8*v_obs2, 0, 0]。治疗state32=原state32-reference32，故仅前30个速度/终点数值条件中心化；最后两个加速度/首末预测heading统计完全不变。不得临时改OLS速度、平均窗口、heading基准、floor或多个reference选择。

双方使用相同原始state的FIT25912×20均值/std，floor.001，不再为治疗单独拟合新标准化。z_raw=(raw-mu)/std；z_delta=(raw-reference-mu)/std。这保留共同尺度，包含输入均值/优化几何变化；给定past可逆，不宣称新增信息或纯因果证明。双方全部buffers相同并保存在新checkpoint，不从CAL/DEV选尺度、类型/方向或候选slot。原raw控制同第一项actual arm架构/seed的fresh重复，并非独立泛化。

## 反例和适用风险

最后一次观测速度可能有噪声或变化；中心化可能过度强调与短时惯性的差异，也可能强化错误native预测。单pulse/既有modeconditioning、geometry/state/社会/时序head失败、GT辅助witness不等可学以及FIT→CAL缺口均保留，不能归结decoder唯一根因。更集中不必然坏，effectivegood减少必须与coverage/错误/运动学一起看，不一概称collapse。

## 完全不变范围

全旧F16/whole original decoder/scorer所有权重buffers固定，immutable K20/fullK/原slotorder。仅过去16数值/frozen8observedtokens和既有native预测输入，GT/future/groups/logits不进forward。原hard ADE+.5FDE winner、balancing/diversity/residual/smoothness/protection.1/calguard.01/FIT25912CAL2879/原LR顺序batch/80epochs3pairedseeds/clip1.5(speed+.03)保持。D2+A2+clock2/center.05–.95/span.05–.5保持，不增加监督/概率/AR/teacher或旧失败headwarmstart。

## 实施与拒绝门

全部新code/cache/log/models/results/failure只本folder，继承所有旧SHA digests（包括第一项completed报告和失败暂存provenance），不改任何父科学文件/既有结果。现有持续newhead授权覆盖本对照；任何源锁/ID/资源/不可辨识/权限实质阻碍STOP留证据等human，不绕新名扩范围。

1. FIT-only实际数值：32顺序/scalar fixture/reference30维/coordinate-origin/固定共同scalers/518240点/finite非退化；sameweights下raw arm与原actual-state forward实际对齐，online/cached native identity锁IDs，原base不漂移。只按数值合法性检查，不按CAL/DEV结果调整representation。
2. GPU8真实FIT steps：双方state first-layer/head/front/output实际梯度正且有限（last-zero分阶段梯度要在有效更新后查）；fullpaired全部权重buffer、nativezeroidentity、frozen、finite128K20/bounds/splits/source/全部priorSHA实际通过。
3. 双方real2epochs连续/中断参数、全部bool/scaler buffers、optimizer/torchCUDAloadernumpyPythonRNG/history/CAL EXACT，以及完成seedSHA-skip；不得用分析恢复代称训练恢复。
4. 独立36throwawayFIT steps/4warm32measure/CAL2879，实际内存和ETA。约17GB可用hostRAM，process6GiB/system physical+commit各reserve6GiB；reference32按batch计算约.31MiB/128，不全量复制多份native状态。内部batch资源guard，不轮询对话/OS/pagefile/kill用户程序。参考生产22–269min变速原因未知，微基准不是保证；生产ETA待实测。
5. 全通过后unique0trigger Windows/start_independent.ps1/独立byte-lock，确认无同任务活跃，一次actual PythonPID/serviceparent/stage/TRAIN(未看到不声称)/stderr。此时才正式登记第二项，写memory/ledger/runtime/start/单次50%elapsedTIME和预计完成节点，同步automation后退出。半程只一次真实剩余修正并静默到完成。

## 结果和完整审计

先至少一个同candidate ADE≤.2 AND FDE≤.4的覆盖/rescue/lost，再全K20 GT-free complete-link meanpath.10/endpoint.20簇包含好候选的有效数量（不是越多越好），最后旧native argmax继承slot不是新概率/scorer。.3/.5只固定敏感性，不删困难样例。

6fresh80epoch/18policy outputs并非18独立trainings；全部18SHA/1516unique IDs finite20×12×2、80history、sample/native ALLpolicygroups/perseed/bootstrapCI独立算术、qualityclusters/globalmotion/temporal/pulse分解/frozen/source/oldSHA/双SUCCESS。quality normalizedadequacy slot与temporal jointoracle slot不得混淆或拼分段oracle。

重大标准全部不降：changeADE/FDE各≥5%、overall/stable/transition≤1%harm、3seed一致、native与freshraw ADE/FDE/joint CIexclude0、samepathcontrol1%/coverage重复硬运动，以及每seed每group pooled speedp95/accelmean,p95/jerkmean,p95≤native1%harm(dt.4)。首次全部满足还审scene/sample相关和3seed充分性；重复ETHDEV/3scene/重叠窗口/F16已见/CAL university是探索，非独立泛化/SOTA/心理意图。完成第二项计2/3，第三完成先总review不盲第四。

## 反方1（同助手inline，非外部独立评审）

PASS_TO_IMPLEMENT_WITH_GATES。Major：条件中心化同时改变优化几何，结论仅限这项工程对照，不可声称增加信息或因果发现。Minor：raw控制重复历史训练不是独立样本；完整记录数值重复而非合并种子数量。若共同scalers/原坐标无法实际对齐，Critical阻碍必须STOP而不是改变reference使通过。未进行或声称任何生产预检PASS。
