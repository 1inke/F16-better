# 数值校验修复契约（不改变科学方法）

原任务的profile零和残差6.079673767e-6略超固定6e-6；路径差1.116892575e-6m通过2e-5m。此修复只处理该不随profile归一化条件变化的断言，不改变trajectory、拟合、统计、质量或bound标准。

独立高精度参照从生产torch产生的float32 Gaussian g开始，把每个值准确转换为float64，使用math.fsum求均值。高精度reference=(g-mean64)/max(maxabs(g-mean64),1e-6)。生产mean32、center32和denom32全部保留，必须profile与原生产公式逐项bitwise相同。

u=eps32/2，eta=最小float32次正规数/2。每项center误差上界E=abs(mean32-mean64)+u/(1-u)*abs(g-mean32)+eta。normalized误差上界为E/denom32 + abs(center64)*abs(denom32-denom64)/(denom32*denom64) + u/(1-u)*abs(center32/denom32)+eta，并纳入float64参照运算误差。实际profile逐项差与math.fsum(profile)必须分别落在此推导上界；高精度参照总和必须近零。不是按CAL或最大报错数值任选新常数。

原2e-5m路径/终点、float64质量及必要几何下界/保存生产路径bitwise/有限性全部保留。新增实际旧模型components验证构造路径，并检查PRIMARY(.2,.4)同候选质量和严格margin witness window在原production操作次序下不变。若不变性失败不能启动或换标准。测试故意损坏profile必须拒绝。

171完成块只读SHA链接，不复制修改旧结果；新revalidated audit和done在本目录。全部原171block输入ID、原295sourceidentity、输出SHA、算术分类与有效好数量核验。只重算未完成279块的原预算，不增加starts/iterations。完整450块最终saved audit重新运行新可靠性规则，完整旧文件和源SHA、三seed权重buffer exact通过才报结果。

此恢复是工程审核修复，不是新formal实验/新科研假设，不能把GT辅助成功当部署性能，也不能把局部优化失败当数学不可达。父级DESIGN全部反方限制继续有效。完成再做FIT/CAL分类解释，不自动第三训练。
