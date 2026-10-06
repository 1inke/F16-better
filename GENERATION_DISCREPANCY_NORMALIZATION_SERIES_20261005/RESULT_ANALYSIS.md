# Centered FIT标准化：第三项完整分析
2026-10-05。结论：正常完整完成，但未恢复原始state的覆盖收益，major FAIL。停止继续局部标准化变体，转向同组、同指标的学习/迁移分解。

## 完成与可靠性
worker与results exit双SUCCESS，12:25:52.851193HK完成。11:24:49启动至完成墙钟61分03.576秒，其中人工挂起约12分21秒；扣除挂起约48分43秒，仅运行经过时间，不是GPU计算时间。6次全新80epoch训练/18policy，不是18独立训练。源402/全部29191旧文件SHA、冻结、1516显式ID有限20×12×2、生产全部sample/nativegroups/72perseed组/24aggregate/96探索CI、temporal与pulse保存算术核验通过。
新completion_review独立重构18组minADE/FDE（最大差4.1402037620486e-7≤1e-6）、完整同路径质量与GT无关complete-link簇、模型SHA18、源402和旧29191前后核验PASS。无模型前向/optimizer/newdata，使用相同注册聚簇定义，非外部独立审稿。首个读取尝试将目录/字段误写成root quality和paired_ci，修正为results/quality及bootstrap；只读字段错误，不是科学或训练失败。
控制9policy全部逐元素bitwise EXACT上一ObservedDiscrepancy治疗路径；新执行重复不是独立泛化。

## 同一路径严格覆盖（ADE≤.2m AND FDE≤.4m）
| 1516窗口 | native | raw scaler控制seed1/2/3 | centered FIT scaler治疗seed1/2/3 |
|---|---:|---|---|
| 至少一个完整好候选 |733|725 /730 /728|726 /733 /731|
| 相对native救回 |—|2 /3 /3|2 /4 /9|
| 相对native损失 |—|10 /6 /8|9 /4 /11|
| clear-event469覆盖（不是change223） |92|88 /92 /86|88 /91 /92|
| GT-free有效good簇总数 |929|925 /928 /929|926 /932 /930|
| 继承native argmax好窗口 |281|281 /281 /281|281 /282 /282|

治疗比fresh控制净+1/+3/+3窗口，但相对native净-7/0/-2，不能称可靠覆盖改善。Change223 native43，治疗42/43/44；seed3救2损1，其余零救回。三seed同1516IDs，不能相加为独立人数。原始state第一项744/742/749显著更高，但第一项仍CI/seed3/运动学失败，不升级baseline。

## 正式constrained均值与CI
Change native minADE/FDE .372174740/.696326435；fresh控制 .371579647/.694747508；治疗 .371845335/.694867909，相对native仅.0885%/.2095%，相对控制均值略差。ADE/FDE为分别oracle，不代表同candidate达标。
治疗-native ADE CI[-.001849450,.001341438]、FDE[-.007379350,.003868407]、joint[-.004499042,.003264319]，fresh对应[-.000928397,.002222092]/[-.004599738,.006180976]/[-.002658496,.004943205]，全部跨0。5%与native/fresh联合CI失败。Protected均值、各seednative改善、覆盖/重复/硬运动/同路径legacy guard通过不等全部major通过。

## 学习与运动学
治疗CAL选epoch5/2/13，control5/6/7；选中pulse幅度.000639/.000446/.002525，非完整路径距离。治疗FIT总loss改善14.515/13.036/13.949%，但CAL change ADE/FDE每seed从首epoch到80均恶化；这是不同指标与群组的学习-评估缺口，不单独证明loss错或纯过拟合。
治疗final DEV change .388179123/.761124015，对native害4.300%/9.306%；legacy也无正式胜利，不按DEV改选policy。更充分学习没有转化稳定完整高质量候选。
全组pooled ALLK motion严审：seed1/2平均加速度增加≤.457%，seed3 overall+1.567%、stable+1.675%、transition+1.296%、change+1.269%，违反原1%保护。不能用window-p95平均、较小jerk或大速度硬阈值通过替代此门。
Quality normalizedadequacyclosest单完整slot不同temporal ADE+.5FDEjointoracle单slot；没有分段拼oracle。center/span非语义事件，onset99无事件、direction需turn-only条件分母。继承native槽位不是新头概率或运动意图。

## 可得结论与局限
这一次fixed FIT affine预条件未解决退化；不支持继续第四归一化/floor/LR/epoch变体。不能据此断言所有坐标表示都无效、信息缺失或decoder唯一根因。正面限定是数值、来源、恢复及工程可靠性通过；科学效果仍失败。
repeatDEV/3scene/窗口重叠/nativeF16已见/CAL单university/三seed不足确认独立泛化，CI是探索性。旧GTwitness、freeincrements/timequery/velocitybasis/conditioning/social、tinyFIT→CAL、softset/energy/mixture失败保留。没有intent/SOTA/新baseline或major。
生产peakRSS2.580/private3.366GiB，最低physical14.454/commit11.273GiB；补充RSS.119/private1.542/min15.564/14.305GiB。6GiB预算/各6GiB余量通过，不调整OS或停止用户程序。
同助手反方2：否定证据主要是无native净覆盖、CI跨0和seed3加速度害，而不是只看泛化措辞；已披露。
同助手反方3：GT可达和原loss奖励不是past-only共享可学性证明；下步须同组指标后再定位，不盲改head/loss。两检查非外部独立评审。

第三记完整3/3，mandatory总review见CANDIDATE_STATE_STUDY_REVIEW.md与completion_review/THREE_STUDY_REVIEW_DATA.json。
