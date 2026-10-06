# F16 cue-response concordance

新文件夹保存全部新增资料，旧科学结果不可改。

- 目标：先高质量完整候选覆盖，次有效好候选数，最后选择。
- 当前：冻结保存输出诊断；非新生成训练、非正式第三项。
- 方法及决策边界见DESIGN.md和protocol.json。
- 运行：diagnosis.py --preflight；PASS后start_independent.ps1。
- 跟进：一次startup、planning50%一次check_once.ps1，随后实际完成节点；无轮询。
- 完成：worker与exit双SUCCESS、RESULT与ARITHMETIC_AUDIT PASS、全部旧文件与源锁SHA不变。
- 无新数据/test/其他fold/GT-forward/optimizer；整个原模型保持不变。
