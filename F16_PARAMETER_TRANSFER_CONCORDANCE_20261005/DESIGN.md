# 参数层FIT/CAL质量反馈一致性诊断
状态DESIGN_IMPLEMENTATION_PENDING；未预检、未启动worker；NONFORMAL不增正式计数。2026-10-05 15:25HK是实施预检节点，不是生产ETA。接续人类持续合法newhead授权及优先级2共享映射/迁移调查。

## 研究问题与范围
matched诊断九个final模型均有FIT变化组位置覆盖增益、CAL变化组覆盖损失，且两split运动学退化。检验“降低FIT原primary质量项的真实newhead参数方向，对CAL同组质量项是否同向有利”，不直接据此换结构/loss。
只读raw/centered_raw/centered_own各3seed constrained+final共18已有锁定checkpoint，FIT25912/CAL2879全部原ID/overall-stable-transition-change组。无DEV/test/otherfold/externaldata、新模型训练/optimizer/新loss/event监督/scorer。
前向仅past16/frozenobserved8tokens/nativebase/speed。未来GT/group只作事后质量/导数标签，不进forward。原F16/wholedecoder/scorer requires_grad=False；仅合法newhead求导，所有weights/buffers值初末exact不更新。启用newhead导数不是训练或解冻旧模块。全K20/nativeorder/clip/scalers原样不重fit。AST只读隔离，不legacy顶层写入。

## 固定分析定义
q=mean12 Euclidean+.5 endpoint Euclidean，hardminK，乘原全FIT固定class weights；不变assignment/winner/weight或监督。
J(S,g)=原加权primary组内均值。<=128批以组内sum/全split该组N计算真实参数梯度，float32前向反向、CPU float64累积。stable/transition/change三组保存，overall由N加权合成，并抽查直接总体一致。全部参数名/顺序预锁定45798 whitelist，不能启用旧decoder。
同checkpoint FIT/CAL梯度报告norm、dot、cos及d=-gFIT/||gFIT||的CAL方向导数dot(gCAL,d)：负值局部一阶有利，正值冲突。梯度近零/精度不稳则UNDETERMINED，不把null当0或不可学。
报告overall与3x3组间关系，不只挑change；固定state_encoder/query/key/value/head分支，head最后D2/A2/clock2输出行单列。key.bias解析softmax抵消，roundoff不当有效学习。
这只检验原primary，不重构原total/AdamW更新：regularizers、Adam预条件、weight decay/历史状态不在方向中。不同表示族分别同参数空间比较FIT/CAL；跨族norm单位/坐标依赖，不排行模型能力。hardmin tie/switch与局部曲率限度明确。正向导数不保证有限一步coverage或运动学改善，strict coverage和global p95非本项光滑代理。

## 预检与审计
全18元数据/scalers/ID/原decoder exact；原全FIT class weights核验，不CAL重估。
实际GPU FIT128前向finiteK20/bounds、同checkpoint匹配matched保存输出<=2e-5；旧全部params无grad，newhead whitelist正确。所有params/bool-scalersbuffers初末torch.equal。
原primary标量与独立numpy q/weights重算；真实梯度finite、state/QKV/head连接，bias不要求全正；softmax取消/零head分阶段不误判。
实际128样本分31/64/128chunk累积梯度，固定vector L2差≤1e-6+2e-4×参考vector L2 norm；不因结果放宽。整体梯度与三组N加权和按相同精度实际核验。norm≤1e-10记UNDETERMINED；chunk重构或数值精度未通过时不能对方向作结论。
独立synthetic线性共享head已知同向/反向/正交/零梯度fixtures，零norm不定义cos；保存完整vectors后重新算norm/dot/cos。
隔离实际checkpoint float64数值副本方向finite-difference核验，预固定epsilon1e-4与1e-5，绝对差≤1e-7+1e-3×导数绝对值；winner切换必须记录，不把非光滑点硬判导数。两epsilon均无法辨识则STOP。它验证数值副本推导，不伪称float32生产逐项bitwise；生产float32标量/finite/向量chunk审核另做。无扰动模型保存/optimizer，不伪称真实训练一步受益。
完成jobSHA-skip/badsource-badoutput拒绝，非2epochoptimizerRNG训练恢复。初末source/allpriorSHA/全部weightsbuffers exact/资源检查。

## 资源、实施与完成节点
所有code/cache/log/vector/results/failure仅新folder。继承matched29871全部old digest再扩其完成报告/复核，不替换旧digest，不改旧source/model/results；控制memory/ledger/runtime可更新。
18model×2split×3组45798维float64 gradient约37.74MiB（39.57MB）；另少量overall/汇总。一次一head、一批、有限vectors，不常驻18图/全候选输出。inputs实际bytes与峰值预检；~17GBavailable/process6GiB/systemphysical+commit eachreserve6GiB，内部guard非对话poll，不OS/pagefile/kill用户程序。
实测8批forward/backward/CPU累积、初终SHA/保存IO才估生产ETA，不将15:25规划节点当ETA。所有预检PASS后唯一0triggerWindows/start_independent.ps1独立OSbyte-lock，先确认无同任务活跃；ONE实际Python/parent/stage/stderr startup，之后runtime/memoryledger/PID/start/唯一50%elapsed检查与completiondeadline同步automation，退出静默。
完整完成必须worker+exitSUCCESS/18done输出SHA/原primary值与保存vectors摘要独立算术重算/全部weightsbuffer exact/source/alloldSHA/资源；不是外部独立审稿或全部gradient第二次重放。诊断不增正式counter，candidate-state3/3reviewed/F16baseline不变。
科学/source/ID/资源/权限实质阻碍STOP等待直接人类，保留failure，不能改定义反复使PASS。

## 反方检查1（同助手内联，非独立外审）
Major：方向冲突也可能来自FIT局部收敛、样本构成、winner非光滑，不能证明共享结构错误；保留全部组/norm与constrained/final，允许UNDETERMINED。
Major：原primary Euclidean方向不是total+AdamW实际更新，已限制结论；即使同向也不保证有限更新/coverage/motion提升。
Minor：CAL用于选checkpoint且只有university、window相关/原F16已见，不可靠跨scene CI或SOTA。
PASS_WITH_LIMITATIONS：能检验局部参数层方向一致性，不能单独解决全部迁移原因。

## 后续限定
若冲突一致，讨论共享映射/分布与结构约束但仍需可辨识架构对照；若同向却coverage不迁移，检查有限优化/目标-覆盖差异等，不自动换loss/epoch。若近零/科学不可识别，STOP人类，不无限诊断链。
原major/motion1%不放宽，coverage第一/GT-free有效good第二非max/scorer最后。更广F16decoder/scorer/newloss监督/heldtestfoldexternaldata paidcloud/messages/delete需具体新许可。
