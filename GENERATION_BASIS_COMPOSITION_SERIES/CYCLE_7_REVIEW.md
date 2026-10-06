# 第七周期三项总复盘
     
结论：三项均正常完成且完整审计通过，均未满足重大突破标准。停止继续扫基底、去相关强度、读出激活或训练轮数。原 baseline 不变；没有证明意图被理解、筛选被改善，也不能由负结果推断未来不可预测。

| 预注册干预（constrained） | change ADE / FDE | 对原生变化组收益 | 对照与判定 |
|---|---|---|---|
| Gaussian vs Fourier | .371775 / .695387 | +.107% / +.135% | 对 Fourier 双CI跨零，未达到5% |
| FIT去相关 vs原Gaussian | .371151 / .697156 | +.275% / -.119% | 相对raw FDE差+.001769 CI[.000716,.002866]，运动分布失败 |
| SiLU vs同参数两层线性 | .371685 / .697665 | +.131% / -.192% | 相对linear双CI跨零，三seed/5%/运动分布失败 |

Native change ADE/FDE .372175/.696326；第三项fresh linear .371146/.696782。SiLU-native ADECI[-.002152,.001563], FDE[-.002124,.005391]；SiLU-linear ADECI[-.000714,.002206],FDE[-.002154,.004516]。第三项同路径joint保护、overall/stable/transition1%质量保护、覆盖/重复/硬运动学通过，不抵消重大指标与运动分布失败。各组、各policy、全部CI及运动分布比值保存于JSON，不只报告最优项。

三项18 fresh80epoch训练、54policyentries；不是54独立训练。第二项rawGaussian对照有效架构、初始化及seed与第一项Gaussian相同，fresh执行数值重复不是独立验证。第三项两层线性虽然可表达原128->6仿射函数，但优化是因子化的；SiLU增加函数非线性，同参数并非同有效容量。FIT相似度基底不是意图标签，不创造新观测。去相关可放大弱分量，同时改变有限预算优化与权重衰减先验；有偏、有界candidate修正仍依附原生K20。

完整审计：每项18模型SHA、18显式1516ID有限K20预测、全部存储指标独立重算、原decoder/scorer/F16冻结与源锁、worker和exitSUCCESS、旧文件哈希不变。另行cycle复核重新计算全部模型/源/旧文件SHA，读取18份epoch80恢复历史与选定calibration，不重训/不修改原结果。最终JSON保留全部summary、CI、全部policy group指标、motion比率、FIT/CAL history及原来源哈希。

校准与过拟合：三项仍按原CAL受约束选模型，最终epoch不可替代constrained来挑收益。JSON包含每seed选中epoch、完整FIT/CAL轨迹以核查fit改善与cal保护/变化误差之间的关系；不能将局部FIT梯度冲突或CAL波动当作因果主因。此前tiny-FIT可学习且CAL stable退化，说明简单增加读出能力未自动解决泛化，但不证明容量/损失是唯一根因。既往MLP、history、token、social和部分上游解冻均失败的证据仍有效。

大方向决策：在保持原winner监督、校准和native修正的连续约束下，七周期21项尚无重大结果；不再把局部表示/优化变体当主路线。下一候选方向是“训练监督从单winner变为候选集合分布监督”的单因素对照：不改推理输入、K20、不读未来GT、不改scorer、不放宽保护、其余损失和cal预算相同。此方向可能改善候选学习分配，也可能候选塌缩或抹平多模态，尚未证明原winner是主因。由于现行锁定协议要求original winner loss不变，正式修改监督前需用户明确允许；本复盘不擅自开跑新loss实验。不要再解冻更广F16/native decoder或使用heldtest/crossfold调参。

探索性局限：反复使用同一ETHdev、只三seed、轨迹/场景相关及多次自适应尝试，不构成独立泛化证据。所有重大门槛5%change ADE/FDE、保护1%、三seed一致、native和freshcontrolCI、同路径/覆盖/重复/硬运动学及全部组运动分布1%保留，不因21次负结果降低阈值。
