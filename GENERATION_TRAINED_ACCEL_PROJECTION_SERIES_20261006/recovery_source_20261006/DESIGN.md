# 训练内固定端点保持加速度投影：预注册

## 问题与证据
冻结投影保留 FIT 变化组 99.26–99.66% 已有完整好候选，却没有恢复 CAL；48 组加速度均值达标，7 组速度 p95 仍失败。它证明大部分位置收益不必依赖过量加速度，不证明训练可学、泛化或唯一结构根因。现在比较模型是否能在投影参与前向和反向的条件下适应约束。

## 唯一干预
两组均为两个原 raw-state persistent 专家，均匀 .5/.5 混合，各 91596 合法可训练参数。zero：原有 bounded increments 直接双精度累加；mean_std：同 increments 经已批准固定投影后双精度累加。原 .1 P_acc 与全部原 loss、hard winner、FIT classweights、CAL guard、数据顺序、LR、80 epoch、3 paired seed不变。原 F16、整个原 decoder 和 scorer冻结，原 K20/slotorder不变。无学习路由、无未来观测/事件监督/新质量 assignment、无失败 head warmstart。

数值：两组 base/increment 网络计算 float32，累加/输出 float64；故 control 不声称与上一 float32 control bitwise重复。projection逐项沿用已批准 double reservation/zero-budget mean float32-grid算法，无 motion epsilon/floor；实际路径每窗口 M<=1.01Mb 严格复算，endpoint/reference审核仍2e-5m。投影梯度通过 mean、D、safe alpha与clip；Mb=0 时浮点cast的反传不证明量化函数局部有限差分。D0/零预算/截断/clip分支须实际审核，不以自然 hinge不激活虚构有效性。相同 codes 端点保持，不是两个最终训练模型端点匹配。same nominal非same函数/优化几何，不能称纯因果 head优越。

## 数据、统计及成功定义
固定 FIT25912/CAL2879，DEV1516只重复探索评估；不heldtest/其他fold/外部data。GT、groups、logits不forward。GT仅原训练loss和posthoc；groups仅既有loss。FIT-only immutable state/scales/IDs原样锁定，无CALDEV重fit。主要至少一同完整candidate ADE<=.2 AND FDE<=.4覆盖/rescue/lost；其次GT-free全K20 complete-link .10meanpath/.20endpoint有效good簇，非最大数量；最后继承原argmax，仅旧slot非新概率。Jclosest与Qjoint各自同slot，不拼分段oracle；.3/.5仅敏感性。原全部major及每seed/组 speedp95、accmean/p95、jerkmean/p95 1%保护不放宽。

## 实施与审核
只新folder所有code/cache/log/model/results/failure。AST读取原类/函数，不执行旧module顶层；canonical旧decoder只读加载；newhead新随机初始化。继承所有prior digest，再扩新增旧artifact，前后全SHA。实际GPU8FIT双方梯度、finiteK20/bounds/nativeidentity/frozen/paired/splits；原loss scalar/gradient及独立NumPy物理算术；两组真实FIT/CAL-subset2epoch连续/中断参数/buffers/optimizer/全部RNG/history/CAL EXACT；完成seed SHA-skip/source-output坏值拒绝。36throwaway4warm32measureFIT128+CAL2879实测吞吐与内存，不保存benchmark模型/选科学outcome。全部PASS才独立OS唯一0trigger byte-lock启动/一次startup/一次50%时间检查→一次完成定点，退出不对话轮询。
process6GiB，physical+commit每项余6GiB；长期约17GBavailable，不OS/pagefile/userkill。资源/源锁/ID/科学可辨识/权限实质阻碍STOP等人类，不放宽标准或盲重试。

## 完成与阶段位置
第三formal仅实际唯一启动后登记；完整双SUCCESS+18policy SHA/显式1516IDs finiteK20/六80history/全部stored sample-group-perseed-bootstrapCI/GT-freequality/pooledmotion/原total-P+投影算术/frozen/source-alloldSHA后ANALYZED、physical-feedback study3/3并先mandatory全三项总review。停止无限第四projection系数/widthLR/epoch/profile变体。加速度达标而CAL覆盖压回native/速度或jerk坏不成功；评估firstallgates还需场景/轨迹相关性与3seed充分性。

## 反方检查1（同助手 inline，不是外审）
- 原CAL单university、曾选checkpoint，旧F16已见、窗口相关；没有独立泛化/SOTA/意图证据。
- 固定输出投影不能补端点预测；此前freeincrements/timequery/velocitybasis/GTwitness/tinyFITCAL/conditioning/social/softenergy失败均保留。
- P可能因投影不激活；原smoothness作用于投影后residual，正是前向参数化因素，不能另改loss权重。
- 本对照检验适应现有约束，不单独分解梯度重分配/函数可行域/有限优化。速度p95不受数学保证，必须实测，不声称所有motion已解决。
