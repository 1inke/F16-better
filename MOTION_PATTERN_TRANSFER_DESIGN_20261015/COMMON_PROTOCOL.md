# 共同协议：真正的学习增量，而非更多不同的轨迹

版本1.0；DESIGN ONLY。本文给出新的系列协议，不覆盖旧协议。任何实施前变更须保存版本差异，不能看结果后修订门槛。

## 1. 数据与研究对象

T1/T2沿用target-grouped精确ID：FIT18909、outer1035（change253）、inner1856、original_CAL2879。过去8步、未来12步、dt=0.4s，继承已验证focal坐标，末观测点为原点。所有样本保留，不按预测效果筛选。

只有当前FIT可用于原型、输入统计和训练。outer固定末轮主评价；inner/CAL只报告，不选checkpoint、K、loss或模型。F16曾接触原数据，故不是整个模型的首次未见泛化。

允许的forward输入：冻结P2 mean-observed-token 96维、真实观测末速度v0、固定FIT模式中心。P2可能包含历史/邻居，不能称纯自身历史。禁止future GT、GT组、scene/track/ID、native路径、原生logits输入新head。native20仅评价基线。

GT可用于FIT原型与训练标签；这与测试时GT构造、选slot或拟合路径完全不同。原型从**当前训练身份**获得，不能把旧全FIT原型带入新的隔离划分。

## 2. 固定预算与公平性

- seeds42/43/44；固定25epochs，末轮主结果，无早停/最佳轮选择。
- AdamW lr.001、weight_decay.0001、betas(.9,.999)、eps1e-8、foreach=False；batch256、shuffle、drop_last=False，无clip/scheduler。
- 同seed双臂逐tensor初始化、buffers、输入统计、模式中心、batch顺序配对。不同训练分支有独立而匹配的全部RNG。
- NNfloat32；坐标叠加及保存float64；不使用混合精度。
- 固定K20；没有native+new的K40，没有GT删选，没有用分类概率TopK改变候选预算。
- 训练日志按窗口数加权；完整FIT coverage必须另算，不能用minibatch loss代替。

本系列不是M2继续训练。所有head fresh初始化；已有神经模型冻结，不访问旧optimizer。T2的fresh full-context重复是新的匹配控制，不当作独立泛化证据。

## 3. 共同指标

对候选k：A_k为12步平均欧氏误差，F_k为终点误差；Q_k=A_k+0.5F_k。

J=min_k max(A_k/.2,F_k/.4)，C=1[J≤1]。C必须来自同一完整candidate；minADE、minFDE分开报告，不能拼接成一条good路径。

全四组overall/stable/transition/change、四split、三seed均报告：

1. C、minADE、minFDE、J、终点good；native/P0/control→treatment的rescued/lost/retained。
2. Q-winner与J-closest分别保存；同slot早/中/晚误差，不按段换slot。
3. 原SciPy complete-link，D=max(meanpath/.10,endpoint/.20)，cut1；有效good簇、总簇、重复率。不是越多越好。
4. ALL K20、所有窗口时间pooled的speed_p95、accel_mean、accel_p95、jerk_mean、jerk_p95。
5. 训练责任占用、责任变化、各模式FIT样本/track/scene计数，以及初始→最终模式漂移。类别含义不能事后按效果命名。

速度12段（包含末观测原点→首预测点），加速度11差分，jerk10差分，dt=.4。v0→首未来速度的边界跳变另外报告。不平均每窗口p95来替代全population p95。

全部五motion逐seed/split/group要求literal new≤1.01native；不加epsilon/floor；native=0时直接比较绝对量。此门是项目预定比较标准，不是人类运动可行性的充分定义。GT同口径可描述，但不能据此换标准。

## 4. 学习信号、继续门和最终major分开

T1的机制门S_T1、T2的信息门S_T2详见各文件。任何机制信号都不是部署成功。进入下一训练阶段还须共同Q门：

- 当前treatment在outer change每seed完整覆盖≥native，且相对自己的P0每seed净增加。
- outer overall/stable/transition ADE/FDE每seed≤native1.01。
- treatment在四split×四组×三seed全部五motion通过。

最终major仍要求：每seed outer change ADE与FDE相对native各≥5%改善；上述保护条件；三seed一致；native对比ADE/FDE/J校正区间支持改善；完整覆盖增加而非复制得分；全部五motion及相关性复核通过。研究门不替代或降低原major。

T1的P0本身不保证安全，也不要求它先赢native才允许训练。P0来源、身份和算术必须通过；若训练后仍无安全可迁移增量，停止，不用unsafe prototype当新baseline。

## 5. 统计：事前固定48项主family

每比较四标量C/minADE/minFDE/J，全部以当前split的outer change为primary：

- T1：B−A、B−P0、B−native，12项。
- T2：full−reduced、full−P0、full−native，12项。
- T3：time和scene各B−A、B−P0、B−native，24项。

总48，未实施项不缩小除数。20,000次配对crossed cluster×三modelseed重采样，PCG64 seed20261015；两臂与native/P0共享draw。target/scene按(sequence,target_track)整簇，time按既有(sequence,50-source-index-block)整簇；按抽到窗口加权，保存索引与SHA。两尾α=.05/(2×48)，普通95%仅描述。没有将所有历史探索校正为确认性，也不声称仅3seed足够可靠。

T3 time的track敏感性区间另作描述，不替换timeblock主区间；三个场景/少量recordings不伪造可靠独立scene CI。P0/native确定性重复不能当3个独立模型。

保护/运动学为预定逐项约束，未纳入48个primary标量也必须完整报告，不择好项。其余组/FIT/inner/CAL与模式分层为描述性结果。

## 6. 预检与资源：只在未来获准实施时执行

1. split IDs、raw/cache坐标/时间/单位、FIT-only统计、原型来源与旧digest继承；任何源冲突先STOP。
2. 实际GPU256有限值、合法梯度分阶段正且有限、全paired初始化、旧模块冻结、P0精确一致；zero-last第一步前层无梯度如实说明。
3. 独立NumPy loss与assignment算术；固定无tie double fixture参数方向FD，eps1e-4/1e-5。不搜索eps掩盖活动面切换，不冒称float32量化导数证明。
4. 每臂真实FIT513+inner12子集两轮连续/中断：参数、bool/scaler/prototype buffers、optimizer、TorchCPU/CUDA/loader/NumPy/Python RNG与完整history EXACT；完成SHA-skip、坏源/输出/恢复合同拒绝。不是全量两轮或分析恢复。
5. 实际128窗口保存后独立重算同路径质量、全SciPy聚簇、五motion、几何、标签与初始化。
6. 每臂4warm+32实测batch，加入全FIT/outer/inner/CAL保存、全聚簇、CI、最终SHA、CPU复核成本，才给生产ETA。

每个双臂对照planning≤30min才按短预算方案启动；超过则报告实测成本并跳过/等待该项明确预算决定，不能用上次M2约15分钟替代估算，也不削减epoch/人口/审计使预算通过。实现、预检、生产分别计时。

process private≤6GiB，physical/commit各reserve≥6GiB，长期约17GB可用条件；bounded chunks、单模型，不改OS/pagefile、不kill用户程序。每个获准任务唯一按需0trigger OS worker、一次startup、一次预计50%elapsed检查→实际完成定点；不对话轮询，不重复旧任务。

## 7. 保存与报告合同

每双臂阶段6runs×4split=24finaljobs，另保存P0四split（原型共用时4份，不伪造12份独立初始化），另存epoch5/10/15/20的模型checkpoint/history但不在过程中查询outer或用中间轮替换末轮。中间checkpoint仅保留，不自动展开新的曲线诊断。

每run绑定checkpoint、run_manifest、history、assignment_stats、metrics、explicit-ID预测、P0引用、源/输入/split/prototype/stats/objective/init/optimizer全部RNG及SHA。保存中心/每FIT模式标签、各模式计数、候选修正、Q/Jslots与rescuelost。

worker与exit双SUCCESS、全部保存复算/冻结与源保护/资源审核完成才ANALYZED。新completion_review只在新目录独立CPU重算完整saved质量/聚簇/motion/责任/CI/来源，不声称所有模型梯度再次回放或外审。源码锁后不可改，失败/partial保留。

本研究使用既有授权数据，不增加受试者或对外发布可识别轨迹；保留数据许可与去标识身份。内部技术报告不套用临床试验标准，也不虚构伦理审批。
