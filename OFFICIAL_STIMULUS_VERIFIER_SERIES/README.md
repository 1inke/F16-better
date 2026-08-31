# EXP-162 Official-Fold Ground-Truth-Only Stimulus Verifier

该系列在冻结F16、H3、V2及K=20候选轨迹的条件下，只使用真实训练未来拟合条件轨迹似然。训练过程不使用candidate winner、mode label或测试折GT。

`run_exp162_nested_verifier.py` 对每个outer fold执行四场inner scene-held-out CV，选择`min_scale`、fusion alpha和最终训练epoch，然后在outer test上只评估一次。脚本同时输出original、verifier-only和fusion结果。

注意：脚本会按scene ID显式排除outer scene，并要求剩余scene恰好等于另外四个官方场景，防止部分processed train文件残留outer scene样本造成协议泄漏。
