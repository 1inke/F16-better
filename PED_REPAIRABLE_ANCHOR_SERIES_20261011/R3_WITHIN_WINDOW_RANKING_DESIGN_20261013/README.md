# R3 窗口内排序目标：设计与核心代码

**状态：DESIGN_CORE_ONLY_NOT_LAUNCHED。未进行真实数据拟合，没有worker或自动跟进任务。**

推荐阅读顺序：

1. [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)：问题、损失公式、固定因素、六次拟合计划、统计门、后续分支与反方检查。
2. [protocol.json](protocol.json)：机器可读协议、入口与三个R2模型/契约/manifest的SHA。
3. [core_ranking.py](core_ranking.py)：权重、全部同窗口P–N配对、对称logistic实现、独立loss/gradient/Hessian与显式未来fit接口。
4. [frozen_inputs.py](frozen_inputs.py)：显式只读入口，复用SHA锁定的R2纯设计模块与每折固定表示；不导入旧production入口。
5. [evaluation.py](evaluation.py)：Hit4、已证下界/未知上界、全N漏选、best-P排名、五比较共享track重采样及严格screen。
6. [test_core.py](test_core.py) 与 [VALIDATION.md](VALIDATION.md)：实际运行的合成测试及未验证范围。

## 核心调用设计（示意，不是运行入口）

```python
# 只有完成真实来源、输入、资源预检并获执行授权后，生产代码才能调用。
# actual_guard 必须检查真实process/physical/commit限制，禁止用lambda: None替代。
from frozen_inputs import load_inputs, fold_features
from core_ranking import fit_objective, score, rank4

# d, maps, r2core, r2contract = load_inputs(project_root, protocol_path, guard=actual_guard)
# for fold in (0, 1, 2):
#     z_train, z_valid, train, valid = fold_features(
#         d, maps, r2core, r2contract, fold, guard=actual_guard)
#     # 此前需独立重建map与原R2逐数组一致；两目标复用完全相同z数组。
#     for arm in ('pointwise', 'pairwise'):
#         model, fit_info = fit_objective(
#             z_train, d['labels'][train], d['track'][train], arm,
#             guard=actual_guard)
#         val_scores = score(z_valid, model)
#         chosen4 = rank4(val_scores)   # 全20候选；不传PNU/GT做筛选
#         # 保存完整map/coef/intercept、ID/label/weight或pair契约、manifest和doneSHA。
#         # pointwise必须先对原R2同折系数、截距、训练/验证scores和top4严格复现。
```

代码模块没有`run.py`、训练CLI、launcher或定时任务。`fit_objective`只是可显式调用的核心函数，并不代替生产来源审核、真实resource guard、持久化/恢复/完整独立saved审核；这些必须在未来执行阶段补齐。`frozen_inputs.load_inputs`要求fresh interpreter，一次明确加载，拒绝模块同名碰撞。

输入适配器不读取原未来GT轨迹、不执行标签solver，但会在未来明确调用时读取既有P/N/U监督；预测函数本身不接收标签、ID或场景。统计评价是对已知修复见证的选槽，不是新生成轨迹收益。

## 本次可运行的唯一验证命令

```powershell
cd 'C:/Users/linke/Documents/New project/PED_REPAIRABLE_ANCHOR_SERIES_20261011/R3_WITHIN_WINDOW_RANKING_DESIGN_20261013'
& 'C:/coding_toobox/python.exe' -B -m unittest -v test_core
```

该命令仅合成NumPy数据；不读真实NPZ、不调用拟合/solver、无Torch/worker/模型写入。

## 预算与决策

预计规模只有两目标×三折的CPU凸拟合，不是长时间神经网络重训。**尚无本轮实测生产ETA**；以后需测得预检、六fit、保存与独立审核总预算，planning超过30分钟则不启动。不能拿这次0.280秒的合成测试时长当生产ETA。

仅在全部五个参考均Hit4提升≥5pp、99%区间下界>0、每折change不退步及所有审核通过时，才建议设计下一步独立迁移验证。若pointwise精确复现77/263，pairwise至少需91/263才达到相对它的5pp点估计门；91只是必要条件，不保证CI/其他门通过。

本次没有启动B/S2，不修改generator/旧scorer，不把U当N，也不扫描损失权重或kernel参数。ARS流程使设计明确记录了目标与Hit4不完全一致、公共偏置/历史项抵消、重复探索统计局限；内联检查来自同一助手，不是外审。
