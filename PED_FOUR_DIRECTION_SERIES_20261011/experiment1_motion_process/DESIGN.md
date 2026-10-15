# 预注册：三锚点位置修正 vs 三节点加速度运动过程

## 问题与唯一因素

在相同过去编码P2(96D)+不可变原生候选12×2(24D)、24134参数MLP120→128→64→6/GELU、完整paired初始化下，只改变六输出到12步残差的映射。不是宣称新架构已有效；原生支撑、时间形状、可学性和泛化需分开。

position3：三节点t=4,8,12位置残差，节点tanh向量L2≤1，按.20*(t/12)^2缩放，t=0残差0后逐段线性插值。
acceleration3：三节点t=1,6,12加速度残差，节点tanh向量L2≤1，乘固定.05 m/s²，逐段线性插值到12步；δv_t=.4 cumulative δa_t；δp_t=.4 cumulative δv_t。静止anchor可非零启动，不是仅全路径相似变换，也不是degree11位置Bernstein平滑。
双方最后统一单个GT-free lambda=min(1,min_t bound_t/||δp_t||)，bound_t=.20*(t/12)^2；每候选整条使用同lambda，保留时间结构。该positionbound不保证运动学，通过与否完全用实际路径评价。加速度映射的有效可达空间/优化几何不等于位置映射；相同参数量不是有效容量相同，也不是纯因果根因。

## 固定训练协议

原target grouped FIT18909，outer1035(primary fixedepoch25)、inner1856和originalCAL2879只posthoc。原28791 IDs和原native20/split/scalers算法保持。三pairedseed42/43/44，6 fresh runs=150epochs，18科学保存jobs。不warmstart旧模型。
只修原bottom4固定logits选槽；其余16 EXACT。这是继承的受限架构对照，不是推荐部署的整合策略，可能丢失原生尾支撑，必须报告lost。Native20 zero是双方共同身份对照。方向2随后独立处理整合，不能混入本项。
canonical hardmin4(ADE+.5FDE)+.05六pair平均平方hinge(.3)，AdamW lr.001 wd.0001，batch256 shuffle25epoch，无gradclip。GT/group/logits不headforward；logits只预注册slotmap。旧F16/decoder/scorer不实例化或训练，immutable缓存与SHA保护。不读DEV/test/其他fold/外部数据。不改变loss/评分/解冻旧模块。

## 验证和资源

先CPU known acceleration积分、zeroidentity、basis、单lambda、常量fixture与纯NumPy重构；实际GPU8FIT256双方合法参数分阶段positivefinite梯度、paired全部权重/bool/scalers、canonical AST scalar/allgrad EXACT、NumPy标量、非零参数doublecopy中心FD eps1e-4/1e-5，若硬winner/lambda活动面切换则记录不可辨并停止而非放宽；真实513FIT+12inner子集2epoch连续/中断全部params/buffers/optimizer/TorchCPUCUDAloader/PythonNumPy RNG/history EXACT与坏source/output恢复拒绝。不是完整FIT两轮。
实际outer128输出保存读回与两套同路径质量/完整原SciPy GT-free K20 complete-link.10/.20/五pooledmotion审核；同时保存六boundedcontrols独立重构。36throwaway/arm4warm32measure和inner实际评估后给ETA，benchmark不保存模型或按收益选择。
process6GiB/physical+commit各reserve6GiB，不调OS/pagefile/kill用户程序。新系列启动前短预算gate：规划>30min不启动，保存实测原因；不是更改epoch求结果。新单一0triggerOS worker/byte-lock/一次startup/一次50%预计elapsed→完成节点，健康静默。

## 完成与科学判断

双SUCCESS与6×25history、6checkpoint(manifest绑定source/splitIDs/stats/objective/optimizer全部RNG)、18输出+quality+controls、全部旧相关SHA不变、RESOURCE、保存算术PASS后才完成。saved算术非模型/梯度再次回放。
samecandidate ADE≤.2 AND FDE≤.4/J≤1覆盖第一、GT-free有效good第二(非最大化)、scorer最后。报告ALL groups/seed/split/native绝对值、rescue/lost、minADE/FDE/J、单Jclosest早中晚、同IDpaired探索track+3seed2000bootstrap(20261011)。全部原major(changeADE/FDE各5%、protectedoverall/stable/transition1%、3seed/联合CI)和每seed每group五motion≤native1.01不放宽；.3/.5仅固定敏感性。F16seen/target非context-disjoint/3scene/相关窗口/重复CAL/固定bottom4/adaptive历史限制保留。失败不独证loss/head根因或运动不可预测。

## ARS反方1

PASS WITH LIMITATIONS：Major，相同六输出及参数量不消除输出basis/conditioning/可达集差异；本对照只测试这个结构包。Major，bottom4可能重要，zero完整native和lost必报，不能宣称支撑已保护。Minor，短预算18909target单split不能支持独立场景泛化；不将CAL用于选择。先结构后整合/目标/评分，旧长训练与暂停自动化保持原状。非外部评审。
