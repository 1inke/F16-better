# Position3 真实可修复性：固定小样本可行见证审计
## 问题与范围
用户“继续执行任务”批准这一窄诊断。S1b 的 potential 是逐时刻球包络必要条件，不是现有 position3 结构、完整K20运动学或可学习性保证。
只使用既有 A7334 训练侧，冻结所有旧模型、路径与标签。无 Torch、模型前向、反向、神经优化器、S2、数据下载。数值求解仅每案例6维控制，不训练共享参数。

## 样本预注册
A 中 native20 同路径 .2/.4 全失败且至少一个 potential slot 的窗口，分 change32 / transition16 / stable16。
各组按 SHA256("position3-feasibility-v1|sample_id") 排序，取首个未重复 target track，直到配额；跨组轨迹可能重复，报告实际数量。规模不足 STOP，不换 tag。
每窗口所有 potential anchors 按 physical slot 升序，均跑 shape 和 pointwise 各一次；不按结果挑slot。
64个机会窗口是选择性诊断，不外推全A成功率；不访问B/outer/inner/CAL作求解。

## 原结构与求解
单次只编辑一个 native anchor，其余19原样；不声称排除4锚联合修复。
position3 节点4/8/12，零起点线性插值，节点界 .20(node/12)^2，实际输出沿用 tanh/L2截断/全路径lambda/.20(t/12)^2。
以单位节点控制u和epigraph z最小化 J=max(ADE/.2,FDE/.4)。shape约束节点L2<=1、全部时刻位移界；pointwise另约束所编辑路径的逐时刻v/a/j范数<=1.01原路径对应范数。这是原完整K20五motion门的更强充分内域，不替换科学标准。
SciPy SLSQP：零初始化，maxiter150, ftol1e-9，无重启或超参搜索，两模式一律执行。
固定 raw=atanh(clip(u,-1+1e-7,1-1e-7)).float32 后按原结构float64解码并独立重构。clip只是构造有限raw，不是评价epsilon。
只有实际路径 SAME candidate ADE<=.2且FDE<=.4、每步位移界及完整K20五motion literal<=native1.01都通过，且两种重构均通过，才是 SAFE_WITNESS。
solver.success 不是可行证明；即使返回失败码，只要保存的实际路径严格通过独立审核，仍是存在性见证，不声称最优。无见证一律 UNKNOWN，不报告 infeasible。
quality-only可行见证单列；边界两种算术分类分歧保守不算safe。重构max>2e-12为基础设施阻碍，不放宽科学门。

## 评价和保存
dt=.4，速度包含原点到首点，后续差分a/j；五项 speed_p95/accel_mean,p95/jerk_mean,p95（NumPy linear quantile）。
每案例保存ID/track/group/slot/mode、raw32/控制、路径、solver退出码/迭代/约束余量、质量及所有运动学，固定19验证。
诊断oracle组合：每窗口按slot升序、shape优先采用第一个safe见证；否则保留native。单窗口PASS不保证总体分位数PASS，另报样本总体/三组pooled五motion、samepath质量、GTfree原SciPy complete-link .10meanpath/.20endpoint有效good与distinct。
比较S1b/random/bottom4是否包含已找到见证的slot，只是已发现见证下界，不是真实feasibility标签。
所有案例保存并二次CPU独立复算，SHA锁源继承1993相关digests+22旧输出+当前设计代码；非全部历史34k或模型二次回放。

## 预算、门与执行
资源process6GiB，systemphysical与commit各reserve6GiB。先资源，再源/ID/fixture、固定前4锚双模式实测；保守planning=2*最慢实测锚耗时*总锚数+两轮源审核时间+120秒；>30分钟不启动，不改预算求PASS。
唯一无trigger OS任务/byte lock，ONEstartup、ONE预计50%检查后完成定点，不查旧PID或恢复旧暂停任务。
全保存审核与源SHA通过后分析。No formal increment/baseline/historycycle11/goalcomplete。

## ARS 反方检查1（本助手inline，非外审）
主要风险是将求解失败当结构不可达、将GT逐例控制当可学习生成器、将选择性小样本当总体。
本协议以UNKNOWN、oracle限定、抽样分母及单锚限制处理。未实现全局对偶不可行证书；零见证也只能停止本次有限求解，不能证明无解。没有新人群采集或身份重识别，沿用已授权去标识轨迹。
下一步由证据决定：发现安全见证只支持研究学习/寻址障碍；仅quality见证提示安全耦合待查；多数UNKNOWN只能说明此有限审计不充分。不自动S2或新训练。
