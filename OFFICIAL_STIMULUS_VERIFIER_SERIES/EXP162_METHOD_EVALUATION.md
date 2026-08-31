# EXP-162 方法评估与执行说明

## 总体评估

EXP-162值得执行，并且适合作为最后一次评分监督范式实验。它只用真实训练未来拟合条件轨迹似然，不依赖candidate winner、mode ID或best-ADE标签，因此与EXP-161的候选排序监督具有实质区别。

该方法仍存在明确风险：生成式似然不必然等于判别式轨迹质量。对角GMM可能偏好平滑、接近高密度中心的轨迹，而不一定选中与单个GT最接近的多模态分支。因此必须同时报告verifier-only和与原始V2 logits的融合结果，不能只看winner hit。

## 协议修正

部分`processed_data_diverse/*_data_train.npy`仍含少量与outer fold相同的scene ID。脚本不会直接使用整个outer train文件，而是：

1. 按官方映射`ETH=0, Hotel=1, Univ=2, Zara1=3, Zara2=4`显式删除outer scene；
2. 要求剩余scene恰好等于另外四个官方场景；
3. 在四个source scene上执行inner leave-one-scene-out；
4. inner阶段只访问outer-train V2缓存；
5. 锁定min_scale、alpha和final epoch后，才首次打开outer-test数据及候选缓存。

## 预先锁定的结构与搜索空间

- verifier：8-component trajectory-level diagonal GMM；
- self/social/decoder hidden：96/64/192；
- dropout：0.10；
- min_scale：0.03、0.05、0.08；
- fusion alpha：0.0、0.2、0.4、0.6、0.8、1.0；
- V2候选及原始logits：复用EXP-161冻结缓存；
- final epoch：chosen min_scale四个inner fold最佳epoch的中位数。

alpha=0被保留为安全回退。inner选择优先要求ADE、FDE和p95 regret不劣于原始分数；若没有任何非零alpha满足安全条件，将自动选择原始V2分数，而不会利用outer test强行挑选融合权重。

## 通过标准

相对Original V2必须同时满足：

- 宏平均Top-1 ADE、FDE改善均不低于3%；
- mean regret改善不低于10%；
- ADE和FDE均至少4/5折不退化；
- 任一折ADE/FDE退化不超过5%；
- 任一折p95 regret退化不超过10%；
- 宏平均Spearman不低于原始分数；
- 不允许出现winner hit提高但mean regret恶化。

## 结果解释

- verifier-only通过：说明GT条件似然可以独立承担候选排序，进入V2/M2 verifier公平比较；
- verifier-only失败但fusion通过：说明生成器先验和轨迹合理性互补，可保留nested-CV确定的融合；
- 两者均失败：停止Top-1 scorer主线，接受set predictor定位，不再扩大reranker。

## 工程状态

脚本已完成语法检查、全五折输入检查，以及真实的小样本inner训练、final retrain和outer evaluation链路测试。试运行结果已删除，不会与正式结果混淆。
