# 设计包验证记录

日期：2026-10-13（香港）。状态：**DESIGN_CORE_ONLY_NOT_LAUNCHED**。

## 实际完成

- `C:/coding_toobox/python.exe -B`，threadpoolctl限制数值库CPU1线程：**18项合成unittest全部PASS，测试体0.210秒**，不是生产耗时。
- 5个Python文件AST语法解析PASS；protocol.json解析及严格JSON序列化PASS。
- R3旧`core_ranking.py`、`frozen_inputs.py`、`protocol.json`三项实际SHA与本设计entry pins一致；只读，无旧文件修改。
- 初版17测试PASS后增加“不得按邻域好坏挑P”的测试；最终18测试均通过。

检验覆盖：整query-track排除；不同track去重；U不补位；ID/slot并列；原始独立逐track/逐slot标量距离核对；31/256chunk一致；距离两块加和；不足5轨迹拒绝；NaN/重复ID拒绝；track等权半径；零半径；同ID/track分折拒绝；基率和至少1P解析期望；全U输出null；best-P固定旧分数；窗口而非候选加权；整track共享bootstrap确定性及独立单draw重构；支持不足/空人口不虚构CI。

## 验证工具的一次路径错误

首次外层语法/SHA核验命令使用`Path('.').parent`，未resolve绝对工作目录，寻找相邻R3目录失败，抛出FileNotFoundError。改为`Path('.').resolve().parent`后，完整同项检查PASS。错误在只读验证命令，不在实验核心/adapter，未修改任何旧数据、源码、SHA或科学标准。没有生产worker或真实实验失败。

## 明确尚未验证 / 不可声称

- 没有调用`input_adapter.load_frozen`、没有读取R1数据/标签、R2模型map或R3真实OOF NPZ；其保存数据SHA pins和全3313继承来源+47输出尚待实际执行预检。
- 没有真实7680查询、训练LOTO半径、真实PNU邻域统计、六项真实CI或新科学结果。
- 没有完整源锁/生产资源guard/真实输入preflight/benchmark/全量独立saved审核器/持久化/launcher/自动跟进。
- 没有fit、神经前向、标签solver、新模型、B/S2、旧PID/task/log查询、旧实验重训或旧文件删除。
- 测试里的no-op guard仅适用于小合成fixture，禁止作为生产资源guard。

本记录验证的是**核心计算契约**，不是项目数据已通过审核或实验已完成；没有生产ETA。
