# R4 冻结跨轨迹邻域一致性诊断：设计与核心代码

状态：**仅设计/核心实现，真实实验未启动。** 无模型拟合、神经前向、新标签求解、worker、自动检查或真实数据诊断；不进入 B/S2。

完整方案见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)，固定机器可读配置见 [protocol.json](protocol.json)。

## 为什么先做这个

R3 从77/263提高到82/263，未过科学筛选门；181个漏选窗口中，75个best-P在第5–8名，106个在第9–20名。因此既不能断言“只差top4目标”，也不能断言“输入没有信息”。本轮只判断固定200D度量下，遗漏P是否有跨行人训练邻域支持。

## 文件与职责

| 文件 | 核心职责 |
|---|---|
| `core_neighbors.py` | 无标签K=5不同track检索、训练LOTO半径、窗口P/N基率校正、best-P事后索引、类条件窗口宏平均 |
| `fold_diagnostic.py` | 单折纯几何标定/验证检索；全部三折完成后附PNU；OOF组装后取每窗口best-P |
| `summary_stats.py` | 六项预注册统计，整query-track共享2000次bootstrap，支持不足null，严格JSON转换 |
| `input_adapter.py` | 显式调用才读取SHA固定的旧纯模块、R2冻结map和R3 OOF；不fit/不执行旧production入口 |
| `test_core.py` | 合成输入测试，不调用真实输入适配器 |
| `VALIDATION.md` | 本次实际验证范围与尚未验证项 |

## 核心方程

```
d(z,z') = sqrt(mean_200((z-z')**2))
neighbors = nearest representative of each distinct train track, then first 5
radius = weighted_q95(train leave-whole-track-out d5)
supported = (validation_d5 <= radius) AND (known_neighbors >= 3)
expected_P = mean(P_count_w / (P_count_w + N_count_w)) over selected known neighbors
enrichment = observed_neighbor_P_fraction - expected_P
```

同track排除；U保留不补位；distance/sampleID/physicalslot解并列；训练标签不进入检索。多数邻居为N不自动等于冲突。`boundary/deep`特指排名5–8/9–20，不是几何半径内/外。

## 未来执行器的调用顺序（接口示意，不是本次执行）

```python
# 0. 实施完整源锁、ID/分折审计、CPU单线程和资源guard，真实几何benchmark。
#    不得直接把此接口示意当作已完成的生产预检/独立审计器。
data, maps, core2, contract, transform_fold = load_frozen(
    project_root, r4_protocol_path, guard=production_guard)
geometry = []
for fold in range(3):
    ztr, zva, tr, va = transform_fold(data, maps, core2, contract, fold,
                                     guard=production_guard)
    g = geometry_fold(ztr, data['ids'][tr], data['track'][tr],
                      zva, data['ids'][va], data['track'][va],
                      guard=production_guard)
    # 保存tr/va ordered IDs、neighbor keys、距离分量、半径/权重/源SHA。
    geometry.append((g, tr, va))

# 三折全部7680验证candidate几何完成后，才附标签；U不能参与补邻居。
for g, tr, va in geometry:
    posthoc = posthoc_fold(g, data['labels'][tr])
    # 将每项reshape为[sum(va),20,...]，按va放回原384 ID顺序。
    # 同时保留全部query包括U、far、known不足，不只保存supported。

primary = primary_rows(data['pairwise_scores'], data['labels'],
                       all_candidate_fields_in_original_ID_order)
i = primary['window_index']
statistics = six_statistics(primary['enrichment'], primary['supported'],
    primary['rank'], data['groups'][i], data['track'][i], guard=production_guard)
# 全部P/N窗口宏平均、分fold/group/scene描述、R2→R3迁移均另行完整报告。
# 保存全数组、source/outputSHA，独立逐track算法复核全部检索与统计。
```

这里没有生产持久化、资源检测实现、继承完整源锁、独立全量审核器或launcher。以后执行时必须补齐，不能把合成PASS当真实数据PASS。`input_adapter`需fresh interpreter；既有R2模块名不允许静默复用。读取旧map是加载冻结参数，不调用其fit。

## 本地合成测试

在本目录执行（只生成内存中的合成数据）：

```powershell
& 'C:/coding_toobox/python.exe' -B -m unittest -v test_core
```

代码无Torch、sklearn训练调用或自动入口；`test_core.py`是唯一测试执行入口。生产预计时长必须来自真实几何benchmark；规划超过30分钟停止，不靠删样本、减fold或改K适配时间。

## 结果如何决定下一步

- 局部支持充分且enrichment可靠为正：才有理由设计一个固定top4集合目标对照，不能声称它一定成功。
- rank9–20大量远邻或U多：先定位训练覆盖、表示或标签发现问题，不盲换loss。
- 支持充分但enrichment无正证据：只说明此度量/条件人口下没有超基率支持，不证明所有表示不可学。
- 区间跨0/支持不足/折间相反：结论不确定，不扫邻居数、距离或阈值寻找正结果。

采用 academic-research-suite 的设计/内联反方检查，重点防止N基率误判、同track伪重复和从局部负结果推导不可学习；同助手检查不是外部独立评审。
