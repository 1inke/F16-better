# 预算预检1与一次等价算术加速

首次真实预检的GPU、所有合法梯度、canonical AST/NumPy、固定两eps FD、双方513FIT+12inner两轮全optimizer/RNG恢复、saved128完整质量/原SciPy聚簇/运动学均通过，但规划117.241503分钟超过30分钟。没有启动生产/worker/OS任务。原BENCHMARK/PREFLIGHT/全部源和smoke保存于 failures/preflight1_budget/；原STOP traceback仍保留 failures/preflight1/。

瓶颈是独立审核中190候选对的Python内层循环。只做一次固定等价加速：每128窗口向量化相同190对的12步距离减法/平方/sum/sqrt/mean；仍每窗口调用原SciPy linkage(method=complete)/fcluster cut1，D和阈值、顺序均不变。实际两arm128保存人口逐项对比旧逐对循环clusters EXACT，及原全部quality/motion复算，通过后才可能启动。production代码未锁源，尚无生产结果；不覆盖旧hash/科学函数，不以近似聚簇或删FIT来凑预算。

同时补齐纯报告字段：固定hash128的实际GPU raw→normalized tokens/初始输出单独保存；初始观测速度→第一未来速度跳变另报，不混入原11加速度门。不改变forward、loss、优化器、人口、阈值、epoch或输入。第二次完整预检执行后以其实际预算决定。若仍>30min则停止，不反复试加速或减规模，不启动长任务。

所有修复由同助手实施，非外审，不把数值预检通过写成科学收益。
