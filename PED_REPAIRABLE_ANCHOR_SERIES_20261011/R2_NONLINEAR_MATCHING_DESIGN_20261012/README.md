# R2 非线性匹配：设计与核心代码交付

交付完成日期：2026-10-13（香港时间）；方案最初保存于2026-10-12。

**仅设计与参考代码，不是已启动或已通过真实预检的实验。**

主文档：[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。固定协议：[protocol.json](protocol.json)。

核心对照是 R1 的安全见证监督和同三轨迹OOF折完全固定下，比较136维线性选择器与136维原表示加64个固定RBF特征。共计划6次CPU拟合，不重新做1022个逐案例求解，不训练F16或生成器，不进入B/S2。

## 文件

| 文件 | 职责 |
|---|---|
| core_matching.py | train-only统计、固定landmarks/RBF、原权重、凸拟合接口、独立目标梯度、评分与top4 |
| input_contract.py | 只读入口SHA、384ID与P/N/U、原fold和排除轨迹、保存参照一致性 |
| evaluation.py | Hit4、U上下界、P排名、配对track bootstrap与四比较筛查门 |
| test_core.py | 14项合成单元测试，无真实数据、fit、solver或Torch |
| VALIDATION.md | 本次实际验证范围与尚缺的生产工作 |

所有模块import都不会启动实验；拟合函数必须被显式调用。没有launcher、自动跟进或生产命令。

只运行合成测试的命令（在本目录）：

```powershell
python -B -m unittest -v test_core.py
```

依赖：合成测试只需NumPy及Python标准库；未来拟合接口需要与R1兼容并锁定版本的scikit-learn/threadpoolctl。当前没有安装依赖、没有实际调用拟合接口。请先阅读设计第8节的真实来源/数值/资源预检要求，不将本测试PASS当成生产许可或科学结论。
