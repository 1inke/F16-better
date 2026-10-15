# E3关键fresh同架构重训：实施前协议

2026-10-09。本文件是待实现预注册设计，不是已预检或已启动声明。

## 首个生产组

五官方fold×seeds1/2/3×full/last2/noNbr=45个fresh完整pipeline。每pipeline H3 30epoch→同branch验证选定H3→V2 20epoch；共有90个阶段训练，不把45pipeline说成45单阶段。训练预算、batch64、loss、scheduler、优化器、K20、参数结构和原pipeline一致；H3全原合法参数，V2保持原冻结策略只原修正参数训练。不存在旧full H3向last2/noNbr泄露warmstart。

原来源：OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/run_exp156_fold_local_pipeline.py、OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES/run_final_v2_seeds23.py、FRONT_ENCODER_SERIES/train_front_variant.py及实际各fold H3/V2 manifest。必须核验实际旧参数/预算与该描述一致，再锁新source；不能只按默认CLI猜协议。

共同原scene-tail内层5%、seed20260830；选择validation oracle_combo，最终test仅选择后评价，沿用已冻结E1分层，测试不改epoch/超参。针对验证track重叠另完整登记窗口/track区间与曝光；不是声称target/context-disjoint。

## 唯一变动：输入可用信息

三个branch architecture保持完全相同。变换发生在dataset输出原始世界观测后、增强/heading/center/图/F16状态及decoder前，对train/validation/test一致，不只mask编码token。

- full：原8观测/原有效邻居。
- last2：所有有效agent仅保留第7/8点；前6点复制第7点。不会偷读第三点计算heading；原future只loss与评价，原模型维度仍8。重复前缀是固定缺失历史编码，不把它说成真实静止历史。
- noNbr：只目标有效，num_valid=1，其余slots以目标观测填充但mask无效；旧未来/label同原，邻居future不作为模型特征。图与社交同时使用变换后的有效数。

paired同fold/seed模型和参数初值，加载各branch自有H3可导致V2初值不同，这正是训练信息差异，不是最终参数同值。originalfull原始路径数值/梯度需审核；不更改旧模型结构/原评分损失或新teacher。

## 必须实施的预检

锁新输入变换/完整原source-ID-state绑定；full路径同源数值/梯度EXACT，last2早6点任意改动不影响transform后的任何预测/图输入，noNbr原邻居任意改动不影响输出；只有past和valid通路，无future/scene/ID作为预测特征。

真实GPU8FIT步骤，有限全合法梯度/paired初值/参数与stage冻结边界；三branch全部实际train/validation子集两epoch连续与中断恢复，params/all bool buffers/EMA/optimizer/scheduler/Torch CPU-CUDA/loader/NumPy/Python RNG/history及checkpoint选择EXACT。旧trainer没有完整resume：不得直接运行旧脚本并冒称满足恢复，需新隔离实现或可审计源转换。

训练ID不从test取得，需严格所有fold官方train/test raw/cache一致、训练内层无test身份、原future单位米/dt.4、batch资源限制。benchmark只throwaway、不保存科学模型或依效果选协议；实际H3/V2/验证耗时与磁盘占用后才给ETA。原source/model/results不改。

尚无实现/preflight/worker/预计生产时间。完成这些gate后唯一按需零trigger OS worker；一次startup、一次50%预计elapsed、一次完成核验，失败保留来源，不重启旧worker或把旧实验计新完成。

## 后续组（先不自动并行启动）

E5同五fold三个seed fresh no-rotation/rotation，共用本full作为no-rotation控制的前提是其他协议完全相同；维持原reflection/speed增强；rotation唯一新增训练输入因素，test只posthoc。E6 ZARA两fold三个seed共享剔兄弟validation下all-train/no-sibling另设fresh共同控制，不能把本full不同validation选择当匹配。仅速度/位置需要完整输入通路定义，不临时改架构冒称同参数因果。

反方1（同助手非外审）：原F16+H3+V2不是单个fresh head，缺少H3阶段会让last2仍携带长历史学得参数；新控制必须走完整相同生命周期。本设计维持原损失/预算，不以难实现为由缩短epoch。依然重复旧测试、三seed/有限场景探索，不证明心理意图。
