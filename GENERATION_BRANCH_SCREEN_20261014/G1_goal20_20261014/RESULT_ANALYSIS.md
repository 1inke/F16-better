# G1：完整K20目标生成结果与下一分支

结论：执行与完整保存审计通过；**最终 major FAIL，baseline 不升级**。当前P2条件目标网络优于经验中心，但尚未优于native完整候选。按事前门控进入一次固定预测终点的G2过程对照，不增加loss、不重训G1、不扫描参数。

## 1. 设计与复现

来源：[系列预注册](../SERIES_EXPERIMENT_DESIGN.md)、[G1设计](DESIGN.md)、[protocol](protocol.json)、[实际预检](PREFLIGHT.json)。训练18909，outer1035固定25epoch主评价，inner1856/CAL2879仅posthoc；seed42/43/44。P2 mean96和真实最后观测速度预测20个目标，26306参数。train-only Kmeans20残差中心初始化；零末层等于prototype20，非20copies。目标g=4.8v+c+delta；路径p=s*g。190对均值hinge与hardmin20(ADE+.5FDE)，AdamW .001/.0001、batch256、25epoch，无clip。每seed74×25=1850步，共5550。旧F16/decoder/scorer不实例化训练，仅读取不可变cache作为外部native20。

基线CV1明确K1；prototype20不读取评价GT；retrieval20仅train donor、不同20tracks且排除query整个track。新网络不读取native路径/logits/未来GT/groups/ID/scene。所有指标包括完整FIT，未挑窗口/epoch。覆盖须同一个candidate ADE≤.2且FDE≤.4；minADE和minFDE分别取最小值不是覆盖证明。

## 2. 主结果：outer变化组，n=253

|方案|完整达标|终点达标|mean minADE(m)|mean minFDE(m)|J|rescue/lost(native)|
|---|---:|---:|---:|---:|---:|---:|
|native20|110|161|0.2491|0.4294|1.3906|0/0|
|CV1|10|10|0.7993|1.8096|4.6179|0/100|
|prototype20|46|99|0.3343|0.5816|1.8968|11/75|
|retrieval20|70|109|0.3774|0.7289|2.1345|14/54|
|seed42|77|147|0.3129|0.5614|1.7989|22/55|
|seed43|72|139|0.3188|0.5778|1.8203|17/55|
|seed44|82|152|0.3075|0.5424|1.7637|22/50|

G1平均完整覆盖30.4348%，native43.4783%，差−13.0435pp。三个seed全为负，不是只由一个坏seed造成。终点147/139/152比native**完整**110分别多14.6245/11.4625/16.6008pp，故G2上界门PASS；但native终点达标161比G1所有seed都高，不能称“终点预测已超过native”。G1完整达标比自身终点达标少70/67/70个窗口，是同终点过程实验的空间，不保证可学、可安全救回。

相对prototype20完整coverage平均+12.2530pp，校正CI为[+2.7835,+21.9826]pp。相对retrieval20完整coverage平均+2.7668pp，CI[−8.3695,+12.0055]pp跨0；ADE/FDE/J的均值差CI均小于0，但不能以此替代完整覆盖或物理门。

## 3. 不能忽略的泛化与反例

|split|native overall/change|G1 seed42 overall/change|seed43|seed44|
|---|---:|---:|---:|---:|
|FIT|10858/1937|10821/1563|11021/1579|10869/1561|
|outer|625/110|582/77|581/72|592/82|
|inner|932/166|878/122|875/116|905/133|
|original_CAL|1490/243|1321/133|1302/126|1275/122|

FIT变化组1937→1563/1579/1561，不是只有CAL迁移才失败；FIT稳定组7414→7791/7939/7878与outer稳定434→425/435/431是正/混合反例。训练total首末分别42: 0.421207→0.341316；43: 0.421346→0.339325；44: 0.421356→0.339891。loss下降和FIT终点覆盖改善不等于完整变化路径拟合成功。不能独断“全是泛化”或“唯一decoder/loss根因”。CAL变化243→133/126/122，原CAL不选checkpoint。

## 4. 统计与物理安全

主family48scalar：3参照×4groups×4metrics，2000次whole-track和3模型seed配对bootstrap，rng20261014，分位点.05/96和1−.05/96（99.8958%探索区间）。3模型seed不等于3独立数据集；区间未覆盖所有训练/历史自适应不确定性。以下coverage差用pp，其余为米或J，方向G1−参照；正error更差。

|group/参照|coverage差[CI] pp|ADE差[CI] m|FDE差[CI] m|J差[CI]|
|---|---|---|---|---|
|overall/native20|-3.86473 [-8.95603,0.60019]|0.02101 [0.01156,0.03520]|0.02140 [-0.00873,0.05583]|0.12750 [0.06870,0.21501]|
|overall/prototype20|17.48792 [12.11210,24.84507]|-0.02894 [-0.04441,-0.01720]|-0.05582 [-0.09027,-0.02489]|-0.15378 [-0.23172,-0.08415]|
|overall/retrieval20|7.53623 [2.82938,13.95729]|-0.03935 [-0.06045,-0.02335]|-0.10546 [-0.15664,-0.06401]|-0.22071 [-0.32914,-0.13151]|
|stable/native20|-0.60706 [-7.80984,4.57150]|0.00211 [-0.00643,0.01126]|-0.02372 [-0.04840,0.00198]|0.01034 [-0.03874,0.06192]|
|stable/prototype20|18.43267 [10.22678,28.83718]|-0.03049 [-0.04535,-0.01754]|-0.06120 [-0.09741,-0.03098]|-0.16069 [-0.24019,-0.09071]|
|stable/retrieval20|9.82340 [3.29901,18.94339]|-0.02841 [-0.05065,-0.01262]|-0.07634 [-0.12598,-0.03745]|-0.16976 [-0.29444,-0.07840]|
|transition/native20|-1.87266 [-11.10632,8.85342]|0.02418 [0.00374,0.04426]|0.01850 [-0.02449,0.07017]|0.13245 [0.01982,0.25120]|
|transition/prototype20|21.72285 [10.79277,32.27490]|-0.03471 [-0.06027,-0.00824]|-0.08695 [-0.14330,-0.01536]|-0.20324 [-0.32900,-0.05064]|
|transition/retrieval20|6.55431 [-3.32953,17.63068]|-0.04101 [-0.08531,-0.01236]|-0.11481 [-0.20280,-0.04148]|-0.22367 [-0.44304,-0.07509]|
|change/native20|-13.04348 [-23.59780,-2.35533]|0.06392 [0.03342,0.10329]|0.13115 [0.04488,0.25738]|0.40372 [0.19939,0.67017]|
|change/prototype20|12.25296 [2.78346,21.98257]|-0.02120 [-0.04696,0.00933]|-0.02110 [-0.10729,0.09494]|-0.10249 [-0.25781,0.08811]|
|change/retrieval20|2.76680 [-8.36946,12.00545]|-0.06430 [-0.12697,-0.01930]|-0.16840 [-0.31073,-0.05952]|-0.34025 [-0.62706,-0.09890]|

相对native，outer变化coverage/ADE/FDE/J四项校正区间均指向更差；overall和transition ADE/J也显著更差，coverage及部分FDE区间跨0。原change5%ADE AND FDE、保护组≤1%harm、三seed一致/联合CI、同路径覆盖、重复候选与五motion所有门仍有效，故major FAIL。

G1的48个split×group×seed物理检查46PASS、2FAIL：seed44 outer overall speed_p95相对native +1.25228%；seed44 inner transition +1.22654%。1%门不放宽。变化组所有seed五motionPASS；直线的加速度/jerk约1e−15只表明恒速结构，不能证明真实运动理解。prototype20全部16组speed_p95 FAIL；retrieval20全部16组speed_p95/accel_mean/accel_p95/jerk_mean FAIL；CV1同样保留速度反例。简单基线无安全豁免。

## 5. 完成证据与资源

[RESULT](RESULT.json)：双SUCCESS、3训练、75epoch、28保存jobs、72科学输出SHA。后台01:04:03.226036→01:10:18.973175，约375.747秒（RESULT计算阶段373.162585秒）。唯一half实际01:11:04.872039，比计划01:10:19.448迟45.424秒，已发现完成；没重复startup或补查旧completion。

[完整保存审核](completion_review/REVIEW.json)：一次CPU186.796422秒，全部28保存samepath质量、全SciPy聚簇、五运动学、goal/path重构、独立检索20track键、3 checkpoint/manifest/statistics/75history/48CI192scalar、3538相关来源与72输出SHA初末PASS。仅CPUcheckpoint读取，不是网络前向、optimizer更新、全部梯度第二次重放或外审。来源继承R5全部3450原digest+49outputs，不是所有历史34k文件。

生产private峰值3.363934GiB/RSS2.437450，系统最小physical15.726131/commit24.327068GiB；审计private1.670815/RSS1.274471/min17.178024/26.339195。过程6GiB、physical/commit各reserve6不变，无OS/pagefile/user程序kill。初预检CPUstep设备读取和审核step计数schema错误来源保留[REPAIR_PROVENANCE](REPAIR_PROVENANCE.md)，全预检重验后才启动；生产无stderr/科学源码修改。

## 6. ARS反方检查与下一步

DA1在[prelaunch review](prelaunch_review/REVIEW.json)完成；DA2：最强反驳是学到对经验中心的条件再定位，而不是学会非恒速运动。变化完整覆盖显著低于native与CAL/FIT负例必须保留；near20distinct不等于好20mode。DA3：工程终点门不是major，直线motion低不等于正确，重复outer与P2已含social/history使泛化和因果解释受限。结论只限探索性候选预测能力，不理解心理意图/SOTA。检查由同助手inline完成，非外部审稿。

下一步G2严格固定每seed G1的所有参数/统计/query/h/goals。A复用直线不训练，B只训练82→64→4过程头，p=s*g+s(1−s)a+s(1−s)(2s−1)b，25epoch×3seed。末层零初始化A=B；全部candidate endpoint/FDE EXACT，故B−A不要求FDE改善。预测goal输入只来自冻结G1，非GT teacher；无旧.20修复界、新loss、保护系数、degree/width/LR扫描。B−A是否改善完整coverage/ADE/J，是否保留保护组并通过全部native五motion，是预注册继续门。相对native最终FDE5%门不免除。全新预检/实际预算≤30min后才启动，G1不重训。

## 附录：全部人口与组指标（量化显示四/七位，原值见RESULT）

|split/port/group|n|complete/end|ADE|FDE|J|effective-good|distinct|motionFAIL|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|FIT/native20/overall|18909|10858/14080|0.1730|0.2992|0.9777|0.7916|19.7221|none|
|FIT/native20/stable|10767|7414/8986|0.1332|0.2292|0.7525|0.9891|19.6464|none|
|FIT/native20/transition|3417|1507/2206|0.2099|0.3648|1.1900|0.5701|19.8911|none|
|FIT/native20/change|4725|1937/2888|0.2368|0.4114|1.3375|0.5016|19.7723|none|
|FIT/CV1/overall|18909|4996/5467|0.4723|1.0509|2.7253|0.2642|1.0000|speed_p95|
|FIT/CV1/stable|10767|4434/4867|0.2961|0.6312|1.6791|0.4118|1.0000|speed_p95|
|FIT/CV1/transition|3417|399/424|0.5898|1.3245|3.4070|0.1168|1.0000|speed_p95|
|FIT/CV1/change|4725|163/176|0.7887|1.8095|4.6164|0.0345|1.0000|none|
|FIT/prototype20/overall|18909|7631/11786|0.2210|0.3618|1.2422|0.4057|20.0000|speed_p95|
|FIT/prototype20/stable|10767|5908/8001|0.1618|0.2658|0.9034|0.5499|20.0000|speed_p95|
|FIT/prototype20/transition|3417|842/1729|0.2714|0.4387|1.5290|0.2499|20.0000|speed_p95|
|FIT/prototype20/change|4725|881/2056|0.3196|0.5251|1.8067|0.1896|20.0000|speed_p95|
|FIT/retrieval20/overall|18909|8965/12050|0.2351|0.4227|1.3415|1.0114|16.4227|speed_p95,accel_mean,accel_p95,jerk_mean|
|FIT/retrieval20/stable|10767|6501/8199|0.1614|0.2716|0.9148|1.3622|15.4882|speed_p95,accel_mean,accel_p95,jerk_mean|
|FIT/retrieval20/transition|3417|1047/1764|0.2905|0.5202|1.6559|0.5762|18.5320|speed_p95,accel_mean,accel_p95,jerk_mean|
|FIT/retrieval20/change|4725|1417/2087|0.3630|0.6966|2.0865|0.5268|17.0269|speed_p95,accel_mean,accel_p95,jerk_mean|
|FIT/seed42/overall|18909|10821/15341|0.1836|0.2720|1.0229|0.8297|19.9895|none|
|FIT/seed42/stable|10767|7791/9661|0.1295|0.1860|0.7168|1.1176|19.9842|none|
|FIT/seed42/transition|3417|1467/2588|0.2265|0.3352|1.2687|0.5446|19.9950|none|
|FIT/seed42/change|4725|1563/3092|0.2758|0.4224|1.5428|0.3799|19.9977|none|
|FIT/seed43/overall|18909|11021/15354|0.1822|0.2691|1.0139|0.9560|19.9719|none|
|FIT/seed43/stable|10767|7939/9710|0.1282|0.1836|0.7094|1.3086|19.9606|none|
|FIT/seed43/transition|3417|1503/2526|0.2248|0.3347|1.2579|0.6096|19.9824|none|
|FIT/seed43/change|4725|1579/3118|0.2743|0.4166|1.5313|0.4030|19.9901|none|
|FIT/seed44/overall|18909|10869/15379|0.1827|0.2691|1.0178|0.8811|19.9901|none|
|FIT/seed44/stable|10767|7878/9732|0.1290|0.1842|0.7137|1.2030|19.9946|none|
|FIT/seed44/transition|3417|1430/2519|0.2259|0.3317|1.2669|0.5499|19.9854|none|
|FIT/seed44/change|4725|1561/3128|0.2738|0.4173|1.5307|0.3869|19.9831|none|
|outer/native20/overall|1035|625/794|0.1755|0.2966|0.9808|0.8145|19.6986|none|
|outer/native20/stable|604|434/511|0.1349|0.2277|0.7526|1.0066|19.6258|none|
|outer/native20/transition|178|81/122|0.2085|0.3419|1.1724|0.6011|19.9101|none|
|outer/native20/change|253|110/161|0.2491|0.4294|1.3906|0.5059|19.7233|none|
|outer/CV1/overall|1035|277/301|0.4679|1.0170|2.6690|0.2676|1.0000|speed_p95|
|outer/CV1/stable|604|250/274|0.3030|0.6258|1.6989|0.4139|1.0000|speed_p95|
|outer/CV1/transition|178|17/17|0.5564|1.2177|3.1907|0.0955|1.0000|speed_p95|
|outer/CV1/change|253|10/10|0.7993|1.8096|4.6179|0.0395|1.0000|speed_p95|
|outer/prototype20/overall|1035|404/632|0.2254|0.3739|1.2620|0.3923|20.0000|speed_p95|
|outer/prototype20/stable|604|319/450|0.1675|0.2652|0.9237|0.5298|20.0000|speed_p95|
|outer/prototype20/transition|178|39/83|0.2674|0.4473|1.5081|0.2191|20.0000|speed_p95|
|outer/prototype20/change|253|46/99|0.3343|0.5816|1.8968|0.1858|20.0000|speed_p95|
|outer/retrieval20/overall|1035|507/649|0.2358|0.4235|1.3290|0.9884|16.2647|speed_p95,accel_mean,accel_p95,jerk_mean|
|outer/retrieval20/stable|604|371/444|0.1654|0.2803|0.9327|1.2864|15.2070|speed_p95,accel_mean,accel_p95,jerk_mean|
|outer/retrieval20/transition|178|66/96|0.2737|0.4752|1.5285|0.6798|18.8483|speed_p95,accel_mean,accel_p95,jerk_mean|
|outer/retrieval20/change|253|70/109|0.3774|0.7289|2.1345|0.4941|16.9723|speed_p95,accel_mean,accel_p95,jerk_mean|
|outer/seed42/overall|1035|582/801|0.1962|0.3166|1.1067|0.8367|19.9836|none|
|outer/seed42/stable|604|425/529|0.1366|0.2045|0.7641|1.1159|19.9735|none|
|outer/seed42/transition|178|80/125|0.2328|0.3490|1.2857|0.5506|19.9944|none|
|outer/seed42/change|253|77/147|0.3129|0.5614|1.7989|0.3715|20.0000|none|
|outer/seed43/overall|1035|581/788|0.1982|0.3229|1.1175|0.9362|19.9729|none|
|outer/seed43/stable|604|435/533|0.1371|0.2032|0.7629|1.2914|19.9702|none|
|outer/seed43/transition|178|74/116|0.2343|0.3665|1.3218|0.5730|20.0000|none|
|outer/seed43/change|253|72/139|0.3188|0.5778|1.8203|0.3439|19.9605|none|
|outer/seed44/overall|1035|592/811|0.1950|0.3146|1.1006|0.8783|19.9845|speed_p95|
|outer/seed44/stable|604|431/538|0.1373|0.2042|0.7620|1.1838|19.9917|none|
|outer/seed44/transition|178|79/121|0.2309|0.3656|1.3071|0.5730|19.9719|none|
|outer/seed44/change|253|82/152|0.3075|0.5424|1.7637|0.3636|19.9763|none|
|inner/native20/overall|1856|932/1272|0.1934|0.3382|1.0877|0.6649|19.8427|none|
|inner/native20/stable|978|601/763|0.1537|0.2660|0.8652|0.8333|19.8057|none|
|inner/native20/transition|381|165/244|0.2088|0.3657|1.1751|0.5827|19.9134|none|
|inner/native20/change|497|166/265|0.2596|0.4592|1.4586|0.3964|19.8612|none|
|inner/CV1/overall|1856|330/374|0.5429|1.2251|3.1620|0.1778|1.0000|speed_p95|
|inner/CV1/stable|978|286/320|0.3537|0.7661|2.0245|0.2924|1.0000|speed_p95|
|inner/CV1/transition|381|36/45|0.5846|1.3317|3.4226|0.0945|1.0000|speed_p95|
|inner/CV1/change|497|8/9|0.8830|2.0466|5.2007|0.0161|1.0000|speed_p95|
|inner/prototype20/overall|1856|658/1084|0.2475|0.4062|1.3984|0.3561|20.0000|speed_p95|
|inner/prototype20/stable|978|475/695|0.1858|0.2988|1.0318|0.4857|20.0000|speed_p95|
|inner/prototype20/transition|381|103/208|0.2711|0.4400|1.5430|0.2756|20.0000|speed_p95|
|inner/prototype20/change|497|80/181|0.3508|0.5916|2.0090|0.1630|20.0000|speed_p95|
|inner/retrieval20/overall|1856|760/1086|0.2686|0.4953|1.5398|0.7996|17.3508|speed_p95,accel_mean,accel_p95,jerk_mean|
|inner/retrieval20/stable|978|518/693|0.1855|0.3252|1.0589|1.0757|16.8650|speed_p95,accel_mean,accel_p95,jerk_mean|
|inner/retrieval20/transition|381|131/212|0.2988|0.5234|1.6963|0.6299|18.6352|speed_p95,accel_mean,accel_p95,jerk_mean|
|inner/retrieval20/change|497|111/181|0.4090|0.8087|2.3662|0.3863|17.3219|speed_p95,accel_mean,accel_p95,jerk_mean|
|inner/seed42/overall|1856|878/1298|0.2267|0.3694|1.2766|0.6950|19.9925|none|
|inner/seed42/stable|978|607/793|0.1645|0.2620|0.9147|0.9652|19.9867|none|
|inner/seed42/transition|381|149/257|0.2422|0.3854|1.3794|0.5354|19.9974|none|
|inner/seed42/change|497|122/248|0.3373|0.5683|1.9100|0.2857|20.0000|none|
|inner/seed43/overall|1856|875/1322|0.2252|0.3670|1.2716|0.7462|19.9714|none|
|inner/seed43/stable|978|611/806|0.1632|0.2532|0.9064|1.0583|19.9591|none|
|inner/seed43/transition|381|148/268|0.2379|0.3753|1.3636|0.5512|19.9816|none|
|inner/seed43/change|497|116/248|0.3375|0.5847|1.9197|0.2817|19.9879|none|
|inner/seed44/overall|1856|905/1330|0.2238|0.3613|1.2568|0.7365|19.9925|none|
|inner/seed44/stable|978|622/806|0.1612|0.2516|0.8943|1.0092|19.9939|none|
|inner/seed44/transition|381|150/266|0.2416|0.3811|1.3737|0.5669|19.9948|speed_p95|
|inner/seed44/change|497|133/258|0.3332|0.5620|1.8806|0.3300|19.9879|none|
|original_CAL/native20/overall|2879|1490/2026|0.1806|0.3033|1.0126|0.7312|19.6558|none|
|original_CAL/native20/stable|1535|994/1254|0.1314|0.2137|0.7312|0.9648|19.5277|none|
|original_CAL/native20/transition|644|253/387|0.2267|0.3768|1.2639|0.4984|19.8898|none|
|original_CAL/native20/change|700|243/385|0.2459|0.4320|1.3985|0.4329|19.7214|none|
|original_CAL/CV1/overall|2879|818/874|0.4980|1.1070|2.8769|0.2841|1.0000|speed_p95|
|original_CAL/CV1/stable|1535|720/765|0.2883|0.6099|1.6328|0.4691|1.0000|speed_p95|
|original_CAL/CV1/transition|644|80/87|0.6505|1.4397|3.7318|0.1242|1.0000|speed_p95|
|original_CAL/CV1/change|700|18/22|0.8176|1.8910|4.8183|0.0257|1.0000|none|
|original_CAL/prototype20/overall|2879|1140/1729|0.2267|0.3605|1.2731|0.3987|20.0000|speed_p95|
|original_CAL/prototype20/stable|1535|887/1158|0.1507|0.2361|0.8415|0.5792|20.0000|speed_p95|
|original_CAL/prototype20/transition|644|157/332|0.2867|0.4382|1.5958|0.2469|20.0000|speed_p95|
|original_CAL/prototype20/change|700|96/239|0.3380|0.5619|1.9226|0.1429|20.0000|speed_p95|
|original_CAL/retrieval20/overall|2879|1254/1770|0.2437|0.4366|1.3874|0.9173|15.3873|speed_p95,accel_mean,accel_p95,jerk_mean|
|original_CAL/retrieval20/stable|1535|906/1179|0.1564|0.2506|0.8750|1.2964|13.7811|speed_p95,accel_mean,accel_p95,jerk_mean|
|original_CAL/retrieval20/transition|644|190/327|0.3097|0.5501|1.7364|0.5978|18.3292|speed_p95,accel_mean,accel_p95,jerk_mean|
|original_CAL/retrieval20/change|700|158/264|0.3744|0.7400|2.1897|0.3800|16.2029|speed_p95,accel_mean,accel_p95,jerk_mean|
|original_CAL/seed42/overall|2879|1321/2055|0.2080|0.3271|1.1715|0.6342|19.9979|none|
|original_CAL/seed42/stable|1535|974/1292|0.1377|0.2059|0.7650|0.9160|19.9961|none|
|original_CAL/seed42/transition|644|214/414|0.2632|0.4122|1.4802|0.4193|20.0000|none|
|original_CAL/seed42/change|700|133/349|0.3112|0.5148|1.7790|0.2143|20.0000|none|
|original_CAL/seed43/overall|2879|1302/2038|0.2090|0.3268|1.1715|0.7711|19.9927|none|
|original_CAL/seed43/stable|1535|967/1294|0.1372|0.2031|0.7602|1.1557|19.9948|none|
|original_CAL/seed43/transition|644|209/406|0.2625|0.4082|1.4626|0.4581|19.9938|none|
|original_CAL/seed43/change|700|126/338|0.3171|0.5231|1.8054|0.2157|19.9871|none|
|original_CAL/seed44/overall|2879|1275/2069|0.2084|0.3250|1.1703|0.7141|19.9896|none|
|original_CAL/seed44/stable|1535|952/1285|0.1394|0.2063|0.7740|1.0664|19.9922|none|
|original_CAL/seed44/transition|644|201/426|0.2589|0.3994|1.4434|0.4270|19.9876|none|
|original_CAL/seed44/change|700|122/358|0.3133|0.5171|1.7883|0.2057|19.9857|none|

