# 后续实验设计与核心代码方案
日期：2026-10-11  
状态：DESIGN_ONLY；未启动新实验、未修改旧科学源码、未恢复暂停任务。  
核心原则：先检验能否找到值得修正的锚点，再检验能否修好；不同时改变寻址、结构、loss和整合。

## 1. 为什么改这条路线

已经完成的方向1–3没有达到major标准。安全的position3与joint目标，在outer change上都是110/253→111/253，置信区间跨0；严格原生支撑插入策略接受0/138480次尝试。这不支持继续盲改同一bottom4上的loss或节点。

保存数据的必要可达性分析更直接：

|人口|native失败窗口|当前bottom4可能救回上限|任意20个原生anchor的乐观上限|
|---|---:|---:|---:|
|outer overall|410|6|209|
|outer change|143|3|56|
|inner overall|924|4|471|
|CAL overall|1389|21|659|

后三列不是训练收益。上界忽略时序basis、五运动学、有限优化和推理选槽难度。它证明当前选槽范围过窄，而不是证明另选4条一定有效。

证据：
- [当前综合分析](../CURRENT_MODEL_ANALYSIS_AND_NEXT_STEPS.md)
- [方向3结果](../experiment3_samepath_quality/RESULT_ANALYSIS.md)
- [必要可达性保存诊断](../experiment3_samepath_quality/completion_review/REACHABILITY_ENVELOPE.json)
- [方向2零接受结果](../experiment2_native_support/RESULT_ANALYSIS.md)

## 2. 执行顺序：有条件的路线，不是五项全部必跑

|步骤|唯一问题|工作类型|进入下一步的条件|
|---|---|---|---|
|S1 可修正潜力寻址|仅凭观测和native路径，能否选到更有潜力的4个anchor？|训练集内轻量线性探针，不训练F16或轨迹head|内部按行人隔离的潜力命中优于bottom4及固定随机4|
|S2 寻址配对训练|相同position3修正器，换寻址是否变成实际覆盖？|2 arms×3 seeds，6个短训练|有真实rescue则审整合；潜力增大但未兑现才考虑时序|
|S3 有条件接受局部编辑|已生成的修正能否通过安全回退留下净收益？|冻结保存输出，9 jobs，零神经网络训练|覆盖与五motion全部满足才保留；全回退则停止该整合规则|
|S4 保留时间轴|已选到有潜力anchor但修不好，是否读出时序信息不足？|新mean控制vs固定DCT时间读出，6个短训练|覆盖、安全及统计门通过，否则不继续扩结构|
|S5 最终路径评分|已有稳定、安全good候选时，能否更可靠地选中？|独立小评分器，生成器全冻结|只有前置候选质量门通过才执行|

这不是把旧“第四方向scorer”强行补齐。S5未触发应写NOT_EXECUTED_GATE_NOT_TRIGGERED。S3与S4不是必然串行：S2没有rescue而只有乐观潜力时，不能靠安全接受策略凭空创造路径；应走S4或停止。

每个生产阶段规划超过30分钟则不启动，不能临时减epoch使其合格。具体分钟数须在该阶段真实数据预检和吞吐测量后给出；这里不把旧5分钟运行当新ETA。整套总时间依分支而定，不承诺全部都跑。

## 3. 共同数据协议和评价

### 3.1 先把选择器监督与head训练隔开

仅使用现有target-grouped训练18909、outer1035、inner1856、originalCAL2879。outer固定最终epoch是primary；inner/CAL只报告，不选择checkpoint、不调阈值。旧F16已见历史数据，本研究仍是新模块适应性探索，不是全模型未见泛化。

在18909内部固定身份键为“recording/sequence + pedestrian track”。令：
- bucket = SHA256("repair-address-v1|len(sequence):sequence|len(track):track")前8字节的大端整数 mod 5（UTF-8；length为字符串字符数，防连接歧义）。
- bucket 0、1为A：只拟合selector。
- bucket 2、3、4为B：只训练后续refiner/scorer。
- 同一身份的全部窗口必须同侧；未知身份不能编造独立性。
- A>=5000、B>=8000且原stable/transition/change三组均存在。未满足则报告规模阻碍，不换hash直到通过。
- A内另用独立固定tag "repair-oof-v1" 按同一身份划3折；各折训练与验证身份完全隔离。
- 保存A/B/折号、原18909完整registry、各scene/recording/group组成、剩余同时间邻居暴露，不宣称context-disjoint。

这是为避免“用同一窗口的GT训练selector，又让它替该窗口挑head训练anchor”带来的乐观训练寻址。代价是head只在B训练，不再是18909全量；因此S2两arm都用相同B，不能把新结果与旧全量head直接当单因素配对。样本量变化不得隐瞒。

所有head arms使用B内旧bottom4生成的同一120维统计；两个arm加载完全相同mean/std。新选槽feature落到不同分布时如实报告，不重新给治疗组拟统计。统计float32特征、float64累积、std<1e-6置1与既有协议一致。

### 3.2 共同不变量

以下refiner参数是S2基线；S4只按其专节替换输入读出并明确新增参数，S1/S5的轻量模型按各自专节。任何例外都不得暗中混进S2。

- native20、原F16/decoder/scorer权重和cache固定；不训练或覆盖旧模块。
- 12步、dt=.4；K20；每次只修改4个选中native，其他16逐数组EXACT。
- head：120→128→64→6，GELU，24134参数；节点4/8/12，末层零初始化。
- 每步残差界r_t=.20(t/12)^2，原trajectory-lambda不变。
- canonical min4(ADE+.5FDE)，六pair mean squared hinge margin .3、weight .05。
- AdamW lr .001、wd .0001、batch256、shuffle、25epoch、无grad clip；seeds42/43/44。
- 每次配对两arm初始全weights/bool/scaler/basis相同，数据批次顺序相同；不旧head warmstart。
- GT仅训练监督/事后评价；group仅分析，logits只旧bottom4控制排序；ID仅分组和审计，不进模型。
- 训练head的quality仍只对4条修正候选计算；完整质量评价必须对最终K20计算，分母不得混用。

### 3.3 成功、推进和失败是三回事

主质量：同一完整candidate满足ADE<=.2且FDE<=.4；J=min_k max(ADE_k/.2,FDE_k/.4)。独立minADE与minFDE另报，不能拼成一条good。

所有旧major门不降低：change ADE和FDE各>=5%改善；overall/stable/transition伤害<=1%；3 seed一致；配对native比较ADE/FDE/J证据与相关性复核；同路径覆盖、重复候选和全部五motion同时过门。五motion是ALLK、ALLgroup pooled speed p95、accel mean/p95、jerk mean/p95，每seed逐项<=1.01 native。native为0则按绝对零门判，不靠分母epsilon放宽。

覆盖/effective-good/GT-free complete-link(.10 meanpath,.20 endpoint)、rescued/lost/retained、单Jclosest早中晚全部保存；.3/.5只固定敏感性。有效good不是越多越好。

CI：同ID配对，以sequence+track为cluster，2000次bootstrap，固定rng20261011；训练对照同时成对重采样3个model seeds。S1 selector折不是独立模型seed；候选20条也不是20个独立窗口。以overall与change coverage两项为预注册主报告，其余探索。报告逐seed和95%区间；若作显著性主张，主两项用Bonferroni 97.5%区间，不能挑95%显著的说确认性。3 scene/3 seed与重复评测仍不足以建立独立泛化。

“值得推进的信号”不等major成功，不升baseline/historycycle11，不假complete。

## 4. S1：训练集内可修正潜力选择器

### 问题与标签

对native候选j和训练GT，逐步误差e_jt：
- lower_jt=max(e_jt-r_t,0)
- J_j=max(mean(e_j)/.2,e_j12/.4)
- J_lower_j=max(mean(lower_j)/.2,lower_j12/.4)
- 标签y_j=1[J_j>1 AND J_lower_j<=1]。

只标“原本不是good、但没有被幅度界排除”的候选；已经good不当新增覆盖正例。这个标签是乐观潜力，不含物理/position3可实现性，也不是未来路径概率。GT只存在labels_offline，predict/select接口不接收GT。

### 最小探针和对照

先用最小可解释模型，不上MoE：
- 输入120D：现有observed P2 mean8tokens96D + 原生候选12×2=24D。
- 不用logit/slot onehot/scene/track/GT/group。每个window的20候选共用同一selector。
- LogisticRegression(C=1, penalty=l2, class_weight=balanced, solver=lbfgs, tol=1e-6, max_iter=500)，固定软件版本；不搜C、宽度、类别权重或阈值。
- 每个OOF fit的normalizer只拟合其A训练身份，std<1e-6→1；不使用该折验证统计。
- 排score降序、精确并列physical slot升序，取4条，再按physical slot升序供head。
- 对照1：原logit bottom4。
- 对照2：固定hash(sample ID, physical slot, "repair-random-v1")选择4条；只是随机寻址负对照，ID不进学习selector。
- GT envelope best-of20仅作上界，不是部署策略。

A内3个OOF模型给每个A窗口一次真正out-of-fold预测；通过预注册门后再拟合一次完整A selector。B、outer、inner、CAL均用同一个完整A selector，不在B上重新拟合。这避免OOF训练路由与“全train重拟selector”混杂到B head训练过程。

### 评价与进入S2门

在native失败窗口中定义H4=1[选中4条至少一条y=1]；报告H4/A20机会召回、选中正例数、slot分布、各运动组/场景结果；不能用全候选accuracy掩盖稀少正例。

预注册工程推进门（不是旧major门）：
1. A各OOF验证折至少20个含正标签窗口的独立target tracks，且训练折正负类均存在；否则证据不足停止，不重抽split。
2. OOF overall native-failed H4相对bottom4及固定随机4均至少+5个百分点，paired-track 97.5% CI下界均>0（两对照）。
3. change的H4点估计在三个折均不低于bottom4；不以该组CI跨0伪称可靠收益。
4. 完整A selector在B的H4两对照差均>0，且change不比bottom4低；B只是训练侧transfer检查，不用来改模型/挑checkpoint。
5. selector不收敛、组失衡或资源规划>30分钟则停止，保留失败，不换模型直到通过。

5个百分点是为短预算设置的事前实用门，不是既有结果或公认统计阈值。失败只否定这一个线性潜力寻址方案，不证明潜力完全不可预测。

输出：split_registry、4个selector模型/统计、OOF predictions、完整A模型、B/三评估split frozen slots与score、labels单独文件、S1_ANALYSIS。先冻结策略与SHA，再进行S2；不能根据outer效果返回调整selector。

## 5. S2：只换锚点的配对轨迹训练

控制arm：B训练样本的原bottom4。治疗arm：完整A selector选择4个anchor。两边都只用B、同共享统计、24134 position3、sum loss、幅度、batch顺序、25epoch、seeds42/43/44。

共6模型/150epoch；每模型在outer、inner、CAL保存一次，共18科学jobs。两边初始native20 identity；虽然每arm修改的4个槽不同，其余16必须按该arm自己的slots核验。不能要求两arm的保留16集合相同。

核心分析：
- 每seed最终K20相对native及两arm配对的coverage/ADE/FDE/J。
- 每个实际rescue必须落在对应selected4的envelope可能集；超界说明代码/ID/统计错误而非新发现。
- 保存selected4潜力上界、真实rescue、lost，以及“可能但没救回”的gap；不能用乐观上界替代训练结果。
- 报告native唯一good是否被选中编辑及因此丢失；GT仅事后。
- 保留候选间相关性、选择top4可能集中于同一模式的反例。

分支：
- 两arm相近且selector在B都无潜力收益：S1结论失稳，停止，不调loss。
- 潜力覆盖上界增加，实际rescue没有增加：只允许进入一次S4时序读出对照；不先scorer。
- 出现更多rescue但lost/motion阻止净收益：进入S3冻结接受实验。
- 全部major含相关性检查通过：停止扩结构，报告探索局限；S5可作为独立部署排序问题，不把它计为生成突破必需项。

## 6. S3：冻结候选、可回退的局部编辑整合

仅使用S2治疗组3 seeds×3 splits的9个保存jobs，NO模型前向/optimizer。比较native20、S2原始编辑K20、以下固定policy K20。K24如保存，只作不等预算附录。

与旧零接受策略区别：不另删冗余native去插入新轨迹，也不改变聚簇阈值；只决定是否接受“anchor j→它自己的局部修正”，拒绝则保留原j。

对每个window，固定枚举4个编辑的16个mask，含空mask：
1. 按mask把对应修正放回其原physical slot。
2. 独立计算完整K20五个逐window motion量。
3. 全部literal<=1.01该window immutable native才可行。
4. 可行mask中最大化选中编辑的selector概率之和；并列优先更少编辑，再优先较小整数mask。
5. 空mask总可回退。selector score只是冻结潜力排序，不是真效用；不用GT/rescue标签或LOO oracle选mask。

这不是GT最优子集搜索，也不是16个训练超参；是事前确定的GT-free有限部署算法。不过它比无回退多了一层部署策略，只能作为独立因素解释。按128窗口有界chunk计算，不一次存所有window×16集合。

逐窗口p95门不推出population pooled p95门，必须重新做全部group/seed pooled评价。无native good无损保证；不同候选编辑仍可能破坏唯一good。

若所有mask均只接受空集，报告IDENTITY_POLICY，停止；不放松1%、不换mask打分、阈值或selector模型去追PASS。若安全但coverage收益消失，也判不能部署。主结论仍依S2/S3配对全数据，不只展示成功窗口。

## 7. S4：只检验显式时间读出，不再换输出参数化

进入条件：S1选择潜力门通过，S2“有潜力但未兑现”；或者S3回退后仍未形成有效候选。只执行一次该结构对照，不顺手加social分支/加速度输出/新loss。

现有P2为tokens.mean(1)，固定8×96 observed tokens在旧cache中已存在。先审cache字段、时间次序、ID/SHA，禁止把未来token当观测；如果token不可可靠映射，标BLOCKED，不从均值伪造时间轴。

新配对：
- 两arm都读同8×96 tokens，经B训练集共同token mean/std后，共享形式的Linear(96,32)，相同参数预算。
- mean_control：将8步投影均值重复3份，得到96D。
- temporal_dct：分别取固定0/1/2频率的时间投影，各32D，拼96D。
- h_0(t)=1/8；h_k(t)=sqrt(2)/8 cos(pi*k*(t+.5)/8)，k=1,2；每行L2范数相同。
- 拼同24D native path，仍120→128→64→6 position3；不改输出basis、范围、loss或S2冻结slots。
- 两arm总可训练参数27238=24134+3104；完全paired初始参数/注册basis。
- 25epoch×3seed×2arms，6模型18输出。B只拟合token/path统计；两arm一致，无由可训练Linear输出重拟normalizer。

这检验“明确低频时间对比能否帮助修正”，不是证明P2均值丢掉全部时间信息：旧tokens本身可能编码时间。也不是邻居机制的直接实验；没有增加raw社会几何信息。它与过去output Bernstein/acceleration mapping不同，本次只改observed-token读出，输出position3固定。

控制arm新增投影/重复输入，本身不等旧24134 head；主要因果比较仅本次27238两arm，不能拿旧head直接证明时序因果。固定basis的秩/数值尺度和有效表示容量差异必须报告。

预检需恒定tokens的k1/k2为零、mean控制的3块均等于时间均值；两arm零初始化均为native identity，但非零权重下不要求两arm输出一致。另需时间反转后k1符号翻转/k2不变、mask padding语义、实际所有合法新参数梯度连接；head末层零时分阶段验证。若实际质量仍无稳定增益，不升级到Transformer/MoE或扫频率/宽度。

## 8. S5：生成门通过后，再做最终几何评分

进入门：S2或S4（及预注册S3如使用）的最终K20在outer overall/change有3seed一致净覆盖改善、主比较区间支持、全部五motion和protected门PASS。未满足则不运行scorer。这个进入门不冒充完整major已达成。

冻结唯一确定的generator、selector和整合策略，cache最终20条完整path。只新增独立scorer，不修改原scorer、generator或candidate集合；不把未修正的anchor几何冒充final geometry。

最小新scorer：
- 输入每candidate的P2观测96 + 最终path24（120D），shared Linear120→32→1/GELU，共3905参数。
- target q_j=max(ADE_j/.2,FDE_j/.4)，训练soft target softmax(-q)，温度固定1，不扫温度；它是quality utility不是未来概率。
- 全20候选参与，无good窗口也保留；不能只训练有good的“容易人口”。
- AdamW .001/.0001、batch256、25epoch、seeds42/43/44；只在B拟统计和训练。
- 3个已冻结generator各配1个对应seed scorer，三runs，不做9×种子交叉膨胀。
- 原native logit读取对应physical slot作为对照（已改路径上的旧分数只作历史基线）；均匀随机top1期望及oracle只作界限。

报告top1 ADE/FDE/同pathgood、条件命中P(top1 good|集合存在good)、绝对全窗口good选择率、oracle-selection gap；同时严格确认新旧scorer的K20候选SHA和oracle指标完全相同。候选数量/覆盖若变化，说明混入生成因素。

全候选pooled motion不因排名改变，但selected top1 motion可能恶化，必须另外报告，不把集合安全当选中路径安全。若无改善，不加复杂scorer，不把条件命中提高误称生成变强。

## 9. 核心代码设计与工程边界

配套 [core_design.py](core_design.py) 是纯NumPy参考内核：无磁盘读取、无模型导入、无任务启动、无生产训练。包含：
- envelope标签（独立GT-only入口）；
- 训练身份分区、GT-free feature/选槽、gather/scatter；
- 原position3映射；
- 固定16-mask局部接受算法（由调用方注入既有精确motion函数）；
- mean/DCT读出。
[test_core_design.py](test_core_design.py) 只运行合成数学/接口测试，不证明实际数据预检PASS。

建议生产模块边界：

```python
# offline_labels.py -- 只在训练A或事后审核中可调用
labels = envelope_targets(native20, gt_future)

# selector.py -- 不可接收gt/group/id/scene/logit
features20 = selector_features(p2_observed, native20)
scores20 = frozen_selector.predict_proba(features20)
slots4 = select4(scores20)
anchors4 = gather(native20, slots4)

# refiner.py -- 继承原position3；仅anchor寻址变动
features4 = concat(repeat(p2_observed, 4), flatten(anchors4))
proposal4 = head(features4, anchors4)
raw_k20 = scatter_native(native20, slots4, proposal4)

# integration.py -- optional S3；不收GT或labels
final_k20, accepted_mask = accept_edit_subset(
    native20, slots4, proposal4, selected_scores, registered_motion
)

# evaluation.py -- 最后才引入GT、运动groups、tracks
report = evaluate_saved(final_k20, gt_future, groups, tracks)
```

Selector训练伪代码（未执行）：

```python
for fold in range(3):
    fit_ids = A_ids[track_fold != fold]
    val_ids = A_ids[track_fold == fold]
    normalizer.fit(X[fit_ids])        # 不可用val、B、outer、inner、CAL
    selector.fit(normalizer(X[fit_ids]), y[fit_ids])
    oof_scores[val_ids] = selector.predict(normalizer(X[val_ids]))
assert predetermined_S1_gate(oof_scores, training_labels)
final_normalizer.fit(X[A_ids].reshape(-1, 120))
final_selector.fit(final_normalizer(X[A_ids]).reshape(-1, 120),
                   y[A_ids].reshape(-1))
freeze(final_selector, final_normalizer, A_ids, protocol, source_sha)
# B/outer/inner/CAL同一个完整A selector；不按结果再挑模型。
# 每个OOF fit同样把window×candidate摊平拟合，但必须按track分折，
# 不能在摊平后的candidate行上随机划分。
```

新时序head核心（接口伪代码）：

```python
z = projection((tokens8x96 - token_mean_B) / token_std_B)  # [...,8,32]
coeff = einsum("kt,btc->bkc", registered_H, z)             # [...,3,32]
context96 = coeff[:, [0,0,0]].flatten(1) if arm=="mean" else coeff.flatten(1)
features120 = concat(context96, normalized_native_path24)
raw6 = same_position3_mlp(features120)
proposal = unchanged_position3_decode(raw6, native_anchor)
```

实际Torch版本必须与参考矩阵、float64位置累计和原lambda对齐；这里不复制旧run.py的有顶层行为导入链。未来实现使用新目录纯模块/只读AST适配并锁源，不改旧run.py。

## 10. 每一步真正启动前必须通过的工程门

- 新目录、原始source digests继承且只扩展，不覆盖历史hash；相关范围如实记载，不冒称全34k。
- source/cache/GT/native/slot/track/coordinate/dt/ID/统计来源逐项EXACT；A/B/outer泄漏审核；不把missing metadata补成伪身份。
- 实际GPU8批次合法参数梯度有限且有效、paired权重/buffer、zero identity、原control scalar/allgrad一致；新公式独立NumPy及固定double finite difference审核，不为PASS搜索epsilon。
- S2/S4/S5每个神经训练arm实际513FIT+12inner子集两epoch：连续vs中断的参数、全部bool/scaler/basis buffers、optimizer、TorchCPU/CUDA/loader/NumPy/Python RNG、history EXACT；这不是完整B两epoch。S1的LBFGS线性探针无epoch/CUDA恢复契约，要求固定输入/版本/线程的重复拟合一致和模型/统计SHA绑定，不冒称完成了神经训练RNG恢复。
- saved128实际pipeline全quality、原SciPy聚簇、五motion、边界、selection与map重构、坏source/output拒绝；CPU独立算术不是所有model/gradient再次回放。
- 每arm4 warm+32 measure真实batch256、实际inner指标保存/重读和SHA IO；输出未压缩/压缩/临时空间规模及磁盘余量。
- process<=6GiB，physical与commit各reserve6GiB；内部resourceguard；不改OS/pagefile、不kill用户程序。
- 预检全过才source-lock和独立OS唯一0trigger worker；先确认无同任务活跃；ONEstartup、ONE50%elapsed检查，再一次completion节点。只更新当前合法automation，不恢复旧PAUSED任务。
- 双SUCCESS+完整结果+独立saved CPU审核后才ANALYZED；事故保存failedcode/log/partial/source，不重复worker/旧重训。

每个checkpoint至少绑定：source_manifest_sha、split及A/B/OOF IDs SHA、selector/model/statistics SHA、slotmap SHA、objective、basis/positionbound、optimizer、init参数SHA、所有RNG、完整history、seed、candidate顺序/坐标/dt。新增候选选择器不再允许“只有参数不知道训练输入和协议”。

## 11. 停止规则与反方检查

本设计是根据已反复见过的结果制定的探索计划，不能称全新confirmatory。各步骤不回看outer调整阈值；跨步骤的适应性必须披露。

反方检查1（设计）：上界潜力不是可修复性。已将label命名为envelope potential，S1通过后仍需S2真实训练，不将209个写成收益承诺。
反方检查2（数据）：selector与head共享训练GT会制造乐观路由。采用A/B身份隔离，并把少训数据的影响放进新配对控制；不是与旧全量模型直接对比。
反方检查3（实现/结论）：安全小编辑仍可能丢唯一good，DCT变化也改变表示容量。保留lost、逐组motion、fresh matched controls与失败分支，不靠scorer掩盖候选缺口。

三次反方检查由同一助手按[Academic Research Suite](C:/Users/linke/.codex/skills/academic-research-suite/SKILL.md)进行，非外部独立评审。该技能用于明确可辨识性、证据与推断边界、以及推进/停止门，没有触发训练。

当前最先应实施的是S1，而不是同时启动S1–S5。此文及参考代码交付不代表S1已经拟合或任何新worker已经启动。

## 12. 本次代码检查记录

2026-10-11，使用现有astra2 Python运行纯NumPy unittest：13/13通过，测试器报告0.051秒。覆盖同路径ADE/FDE不拼接、径向乐观下界、good不当潜力正例、身份hash、稳定ties、固定随机四条、feature形状、gather/scatter保留16、position3零恒等和界、潜力命中、零物理预算回退、16-mask规则、DCT基底与时间反转。

测试仅为合成数组；其中S3 motion回调是测试fixture，不是实际科学motion函数。没有拟合selector/读取真实新split/神经前向/optimizer/GPU/生产任务；不将这些测试写成source-ID、真实resume或生产预检PASS。恒定非零tokens在DCT高频通道为零，mean控制却重复均值，因此不错误要求两arm非零权重时输出相同。
