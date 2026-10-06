# 终点保持的加速度投影：冻结可行性预注册
DESIGN_IMPLEMENTATION_PENDING；当前没有代码、预检、worker或新正式训练。23:10HK只是实施预检节点，预计生产耗时必须实测。当前formal study2/3，NONFORMAL不增加计数。

## 问题与依据
前两项：P_acc降低了运动学成本却未解决稳定CAL覆盖；固定FITmedian双专家route同样未给出稳定覆盖，finalFIT增加但CAL减少。固定route的失败不等共享mapping无干扰，更不等更强gating必有益。
先排查一个更窄、可直接审查的输出约束问题：能否保持现有候选终点修正与原clip，把时序扰动压回原生1%加速度预算，同时保住完整路径/rescues？若压制后主要救回丢失，不能盲训练这种约束。
不扫描路由分位数/专家数/权重/宽度/LR/epoch；不改loss或candidate assignment。

## 冻结范围
仅当前已锁定uniform/routed各3seed final六checkpoint。FIT25912/CAL2879同IDs及nativegroups。原F16/decoder/scorer/六head全部eval/no_grad，参数bool/scalerbuffers逐torch.equal。只past16/frozen8tokens/nativebase/speed forward，GT/groups/logits不参与投影或预测；groups只事后分类。无DEV/test/fold/externaldata、optimizer、新model/loss/监督/scorer/解冻。
只读隔离AST类/function，不执行会写旧目录的legacy顶层。所有code/cache/log/results/failures在此newfolder，继承全部旧SHA，不覆盖原digest。不能把冻结诊断resume声称训练2epochoptimizerRNG。

## 唯一固定投影：不从CAL选择alpha
现有head给出已按原commonclip的12增量u，原native路径b。
mu=mean_time(u)，z=u-mu；同一窗口所有20candidate共同alpha。
原生a=diff(diff(b,prepend0)/dt)/dt，dt=.4，M_b=mean(K20,T11)||a||。
修正delta_a=diff(u,time)/dt²，D=mean(K20,T11)||delta_a||。
若D=0令alpha=1；否则alpha=clamp(.01*M_b/D,0,1)。系数.01是既有1%保护，不新扫参/floor/更换事件标准。
u_proj=mu+alpha*z，p_proj=b+cumsum(u_proj)。
实数算术：sum_time z=0，故终点修正保持；常数mu不改变12step速度差，triangle bound mean||a+alpha*delta_a||<=M_b+alpha*D<=1.01M_b。u_proj是u与其time均值的凸组合，保留原坐标clip。只保证accelmean充分界，不保证speedp95/accelp95/jerkmean,p95或完整路径。保守界可能过分压制响应；输出几何改变不是根因证明。
必须独立float64参考与实际float32保存几何核验。原2e-5m路径/端点一致性容差只用于数值审核，不可拿来放宽1%科学gate。M_b=0/D>0与D=0、静止/恒速/时间均值、badinput、finite等fixture必须覆盖；实际roundoff若破坏严格guard，STOP记录来源，不偷偷改alpha系数/保护门槛。不能宣称最终训练终点matched（只有同输出codes投影保持）。

## 审查与输出
1. 真实FIT128原head冻结重放对齐原缓存、rawparams数91596/所有weightbuffer固定、finiteK20/originalbounds/ID/source SHA。无梯度/训练恢复宣称。
2. 数学scalar/时间均值/端点不变/convexclip/必要加速度界，与真实保存float32/float64误差。初alpha/nativeidentity不宣称表达所有GT。
3. 只FIT/CAL逐batch<=128一个head。保存native/original/projection同完整candidate ADE<=.2 AND FDE<=.4/固定.3/.5敏感性、rescue/lost/原→投影success retained、GT-free ALLKcomplete-link .10meanpath/.20endpoint有效good非maximize、早中晚同一Jclosestslot（不拼oracle）。
4. 所有组pooled speedp95 accelmean,p95 jerkmean,p95 dt.4，原1%保护；同时报alpha分布/endpoint误差/clip误差/收缩强度与覆盖交换。不能把加速度变小当成功、不能删困难例或CAL选系数。人口与window分母明确。
5. 保存输出独立算术、done输出SHA-skip/source或坏输出拒绝、完整旧源/模型/SHA初末；不要声称外部外审或全部模型第二次前向。
6. FIT作主要可行性证据，CAL作探索反例/迁移描述而非新超参选择。所有六seed结果都报告，不只选救回多的模型。投影后的good不证明重新训练可学/泛化。

## 资源与流程
长期约17GBavailablehostRAM，不全部占用。process6GiB/systemphysical+commit每项reserve6GiB；onehead/batch128/metricscluster256，几何按窗口/chunk，不18outputs常驻。单FIT25912*20*12*2 float32路径47.45MiB，original与project输出约94.90MiB，加native/临时量必须实测。内部guard不是对话轮询；不OS/pagefile/userkill。
全部预检/source/frozen/ID/资源PASS才报告实测吞吐/初终SHA+IO/聚簇ETA，再独立unique0trigger Windows byte-lock worker。真实启动一次PythonPID/serviceparent/stage/stderr确认，之后ONE50%elapsed与completion COUNT1，节点前静默，不对话poll。未启动不得伪称running。
源锁/数值/ID/cache/资源/科学可辨识或权限实质阻碍立即STOP等直接人类，留failure，不能反复盲重试/放宽规则。
诊断完整后先保存结果与inline反方2/3，不计第三formal。不因诊断好就宣布major或自动新loss/更广解冻。第三正式输出投影head设计还须实际可微连接/两armGPU与real2epochexactresume等现行流程证据。

## 反方1
Major：投影保守，可能保endpoint但毁中间转向与rescue，是筛查目标而非已知可用方案。
Major：只限accelmean，其他物理项仍可坏；native预算不是现实行人运动真值或安全上限。
Major：冻结后验投影效果不是原head重新训练/共享mapping根因，不能把alpha收缩称意图理解。
Major：同一数据、三scene/window相关/F16已见/CALuniversity选checkpoint；无独立泛化/可靠groupCI/SOTA。
Verdict PASS_EXPLORATORY_DESIGN_PENDING_ACTUAL_PREFLIGHT。所有科学结果阈值及major保持，critical发现STOP。
