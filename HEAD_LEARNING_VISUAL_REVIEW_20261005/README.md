# 本轮响应头学习：改善与恶化样例浏览器

打开index.html即可离线浏览。全部新代码、数据、图片、分类与核验报告仅在此新文件夹；旧实验、模型与此前工具不改。

本轮不是此前80epoch脉冲响应正式实验，而是F16_HEAD_LEARNING_DIAGNOSIS_20261005的six fresh diagnostic runs，seed7/8/9、final120epoch，没有CAL选checkpoint。FIT256是本轮训练集，CAL393未用于本轮optimizer训练；CAL仍来自曾探索的数据，并非全新外部泛化测试。

内容：原F16+固定时间响应+学习时间响应，七套路径、649个唯一ID、全部K20和历史/GT；三seed独立显示。只读取保存输出及已存在FIT/CAL观察/评分缓存，没有torch、模型前向、训练、DEV/test/原始数据读取，也没有重新评分。

## 怎么看

1. 默认CAL、seed7、恶化列表。点“改善”可切换到全部救回案例；不是只展示挑出的代表。
2. 代表入口按双阈值距离变化幅度，分别提供中位与最大变化，明确区分FIT与CAL。ID输入自动切到该集合；全部649 ID均可查。
3. 三个空间图共用范围和米制等比例；同一样例三方对照。默认保留全20条，可一键聚焦，聚焦明确提示裁剪，能恢复完整视图。
4. ADE–FDE散点每点是一条candidate。绿色框是同一条ADE≤.2且FDE≤.4；独立minADE/minFDE可能属于不同slot，不能拼接成成功。
5. 点击轨迹、散点或slot表，三方联动同slot。下方逐帧误差/速度/加速度可比较同slot，或各自双阈值最近路径；每条曲线不逐帧切换slot。
6. 原F16橙线是真实logit最高slot；两个新方法橙线只继承原slot，没有重新评分。深青路径是GT事后最近，不是部署选择。
7. 有效好候选簇显示原诊断保存的GT无关complete-link簇含好candidate数量，仅主口径.2/.4，不是越多越好。这里不再次改聚簇算法。
8. 备注与原实验隔离。点击保存，再导出JSON交给助手。自动QA没有保存或伪造任何人工标注。

## 分类和限制

改善＝原无同路径合格candidate，新有；恶化＝原有，新无；另外保留仍有/仍无两类。位置变化过滤还能检查没跨界却双阈值距离下降/上升或独立minADE/FDE一好一坏的案例。这是位置指标分类，不保证符合运动学、更自然或有心理意图改善。

主阈值.2/.4不变；.3/.5是独立敏感性选项，不能挑更有利标准称成功。GT用于事后材料，未来事件标注也事后计算；onset99表示未检出事件，不是真实延迟。没有新邻居观察材料或未保存的pulse clock参数，不重跑模型补数据。

复现：Python运行build_review.py读取保存输出；为保护已经完成的可视化，脚本拒绝覆盖已存在audit.json。Node运行test_browser.js作隔离headless Edge功能验证，未操作用户浏览器或程序。数据audit.json与UI_QA.json均PASS。观看说明START_HERE.md、分类REPORT.md；CSV保留全精度，图坐标舍入1μm，绘图误差<4e-6m，不用舍入坐标重新分类。

资源估算：仅逐NPZ加载和逐样例输出，无K×K全人群数组；内存守卫process6GiB、systemphysical和commit各reserve6GiB。实际build峰值RSS约0.141GiB/private约0.128GiB；约17GB可用内存限制仍保留，不调整OS或停止他人程序。
