# M2 精确实施设计 v1.0

## 研究问题与父来源

在同一离散过程解码器和同一预测目标初始化下，让完整路径目标更新新goal输出head是否优于冻结goal？M1 S1已FAIL，因此按seed42/43/44使用原G1，不重训G1。不是旧G2桥的重试。

## 唯一对照和参数边界

A frozen_goal：G1全部参数/buffers固定，只训练过程head。B joint_goal：同一父的context/query/centres/scalers固定，仅新goal MLP80→64→2和同构过程head可训练。过程输入h64+query16+normalized_goal2+normalized_v0_2，共84维；84→64→24 GELU/zero-last。手算process7000、goal5314，因此A7000/B12314可训练，总33306均须实际实例化确认。两臂paired过程初始化与相同样本顺序；不同可训练容量不作纯因果排除。

## 离散方程与初始对照

H12、dt.4、p0=0、v0=(past7-past6)/.4。raw b为12×2，w=(12,...,1)，S=650。

- d=(g−12*.4*v0)/.4²；e=d−sum(w*b)。
- a=b+(w/650)*e；v=v0+.4*cumsum(a)；p=.4*cumsum(v)。
- 最小平方项(w/650)*d；零终点项b−(w/650)*sum(w*b)。

g为预测目标，不GT teacher；b/g/v0几何float64，NNfloat32。过程有效线性维数22，不要求24满秩；解析消除方向不把舍入梯度当有效连接。无额外tanh、clip、native修复域、物理penalty或投影后裁切。只保证递推和数学终点，绝不保证覆盖、a1=0、v1=v0或五motion安全。原12速度/11加速度/10jerk pooled口径不改；v0到首未来速度跳变另报。

b=0时A/B初始P0相同，是预测g和v0确定的最小平方加速度轨迹，不是原G1直线/nativeidentity。必须每seed完整保存FIT/outer/inner/CAL，共12个P0，与24最终job分开。A最终目标保持P0，B可变化；B的g同时通过输入标准化和终点投影反传，不detach。

## 数据与固定训练

FIT18909/outer1035(change253)/inner1856/CAL2879精确既有targetIDs；原始past8、future12、dt.4/既有focal坐标。P2可能已含历史邻居，非纯SELF。初始每seed FIT378180个目标拟统计，v0每窗口仅一次；float32特征/float64meanstd/std<1e-6→1，不在训练或CAL重拟。GT/groups/ID/scene/native/logits不forward。

seed42/43/44×双臂固定25epoch，batch256/74step每epoch/1850step每run/11100总step。AdamW .001/wd.0001/betas.9,.999/eps1e-8/foreachFalse，shuffle/drop_lastFalse，无clip/scheduler/早停；固定25主结果，inner/CAL仅posthoc。

canonical loss=mean hardmin20(ADE+.5FDE)+.05×190pair平均平方hinge(.3−meanpath距离)。winner并列取最低slot。没有新loss、event、softassignment、保护项或scorer；不同时变输入/表示或积分系数。

## 审核、资源、启动条件

完整清单见protocol.json：继承3746旧相关原digest，逐项扩M1的107outputs及报告/审计/补充/失败来源，不覆hash；准确绑定原G1checkpoint和实际split/cache/statistics。只读AST/类shim隔离旧顶层写入。NumPy独立积分/投影/650/CV/零方向，固定double eps1e-4/1e-5，实际GPU8 FIT256梯度、冻结、A/B初始、B双链反传、canonical输出grad、actual128保存quality全SciPyclusters-fiveMotion。两臂真实FIT513+inner12子集两轮连续中断ALLparams/buffers/optimizer/TorchCPUCUDA/loader/NumPy/PythonRNG/history EXACT，拒坏source/output/resume合同；不冒充全FIT两轮。

实际perarm4warm32batch256吞吐，加36个全量保存job生产/重读算术、独立CPU审核、20000draw及SHA/IO/disk/RAM成本。规划≤30min才启动；不砍FIT/epoch/审核使PASS，M1预算例外不继承。process6GiB、physical/commit各reserve6GiB；不OS/pagefile/用户kill。

启动仅唯一按需0trigger独立OS byte-lock worker，先核验无同活跃；ONE真实startup，然后一次预计50%elapsed和一次真实完成，更新现有m1自动跟进后退出，不对话poll。若已有runtime真实启动不得重开。锁后科学文件不编辑。异常保留来源/失败/partial，不盲重试或改阈值。

## 评价、统计和决策

完整candidate同时ADE≤.2且FDE≤.4/J≤1；minADE/minFDE单项不拼轨迹。原GTfree ALLK SciPy complete-link D=max(meanpath/.10,endpoint/.20)cut1；有效good不是maximize。全四组×四split×三seed，same-slot Qwinner/Jclosest早中晚、endpoint和P0-final/A-B rescue/lost、目标相对CV/初始变化、raw/projected/边界motion全报。共同endpointgood交集只post-treatment描述，主分母不变。

20000draw numpyPCG64 seed20261014，whole(sequence,target)track与3modelseed独立cross，paired共享抽样，每drawindicesSHA；事前26family固定不缩小，Bonferroni尾.05/(2*26)。M2主8项为B-A/B-native outerchange C/ADE/FDE/J；95CI只描述。重复outer/CAL、仅3seed/3scene、窗口/context相关/F16seen，非独立泛化/SOTA/理解。

S2：B-A outerchange覆盖三seed均正、均值≥5pp且26family lowerCI>0，change ADE/FDE均值不差A，outeroverall/stable/transition各seedADE/FDE≤A1.01。Q2=S2且B changecoverage各seed≥native，outer保护ADE/FDE各seed≤native1.01，全部48组B五motion literal≤native1.01。Q2未过不M3，不扫系数/loss/degree/width/LR/epoch。原major每seed changeADE AND FDE各≥5%、保护1%、native联合校正CI、完整coverage/duplicate与全五motion门不放宽。

完成必须worker+exit双SUCCESS、6×25history、6完整绑定checkpoint、24final+12P0显式IDfiniteK20×12×2及geometry/source-outputSHA/ARITHMETIC/PRESERVATION/RESOURCE。独立CPU完整保存审核仅checkpoint合同载入，不NNforward/optimizer/全gradient重放；RESULT_ANALYSIS MDJSON含同助手DA2/3非外审，runtimeANALYZED后再按Q2决定。执行成功不baseline/cycle11/goalcomplete。
