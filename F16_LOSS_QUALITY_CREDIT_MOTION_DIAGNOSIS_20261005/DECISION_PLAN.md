# Human-authorized two-stage decision plan (2026-10-05)

阶段一实施本冻结诊断；阶段二依据全部FIT/CAL、三seed、全部表示族、constrained/final结果选择一个可辨识对照。人类明确授权持续按这些分支研究，但不因单一描述现象就宣称根因、同时更换多模块或忽略旧失败。

| 证据组合 | 暂定优先路线 | 必须保留的反证/判断边界 |
| --- | --- | --- |
| loss下降且严格完整覆盖提高、运动学保护通过，但CAL/seed不稳定 | 共享head、candidate conditioning与迁移 | 只能说明当前对照部分对齐，不能给loss全盘免责；CAL选择、群组构成同时检查 |
| loss/ADE/FDE下降但覆盖未增 | objective与完整路径目标一致性 | J连续改善但未跨门槛与J不改善必须分开；起点距门槛、原已覆盖饱和也能解释 |
| loss下降伴随lost | 成功保护或共享参数干扰 | 当前已有protection.1且change组不受此保护；类型集中/普遍不足以单独区分loss与head，需后续单因素干预 |
| winner不接近J合格路径，witness可改善loss但无直接primary梯度 | credit assignment | 最近J不等最容易由共享head修成；旧Gaussian witness不是当前persistent可学性；先复核旧softset/energy/mixture失败，不自动重复softmin扫参 |
| winner合理、质量信号到位但修不好 | newhead/candidate-conditioned mapping | 坐标直接credit不等共享参数有效更新；有限优化与泛化仍候选解释 |
| 质量与physical梯度冲突 | 输出参数化与motion权衡 | cosine只局部一阶；同时报告norm、方向导数、primary vs total、原smoothness vs physical、p95稀疏与tie；不能仅凭大/小冲突直接决定架构/权重 |
| 不冲突但最终motion差 | weighting/normalization/保护机制候选 | 参数轨迹、有限步长/非线性、mixedbatch和AdamW也能解释，不能直接排除结构 |
| FIT覆盖增加/CAL丢失，即使credit/局部梯度合理 | 共享映射与泛化 | FIT同时motionFAIL时不能说FIT已完全学好；CAL单scene且选checkpoint限制保留 |
| 生成严格覆盖与motion稳定改善，但选择仍差 | refined-candidate scorer对齐 | 继承native slot不是新概率；当前不训练scorer。先达到生成稳定门槛，再精确设计及确认模块范围 |

总体顺序仍coveragefirst、GT-free有效goodsecond（非越多越好）、scorerlast。所有原major/运动学1%门槛、原数据冻结范围和~17GBRAM约束不变。第二阶段先保存明确单因素设计、资源/实际预检、ETA与独立OS节点，再执行；若证据无法辨识路线或超出现行具体模块权限，STOP解释并等待人类，不无限追加诊断或局部参数变体。

本次诊断不引入训练loss/事件监督/optimizer/newtraining，不解冻旧F16/decoder/scorer，不读DEV/test/其他fold/外部数据。只有18已有checkpoint回放/保存数据和合法newhead数值求导。连续阶段授权覆盖按该分支推进研究，不推定新增外部资源、收费、消息或删除权限。
