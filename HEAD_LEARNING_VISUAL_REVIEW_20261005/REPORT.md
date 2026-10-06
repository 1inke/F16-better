# 本轮改善 / 恶化分类

同一完整candidate ADE≤0.2m且FDE≤0.4m；六个final120epoch诊断输出。每行同一集合重复seed，不能合计为独立人。CAL富集393例不是总体比例。

| 集合 | seed | 原有覆盖 | 学习后覆盖 | 救回 | 损失 |
|---|---:|---:|---:|---:|---:|
| FIT256 | 7 | 77 | 154 | 83 | 6 |
| FIT256 | 8 | 77 | 148 | 74 | 3 |
| FIT256 | 9 | 77 | 155 | 81 | 3 |
| CAL393 | 7 | 169 | 167 | 28 | 30 |
| CAL393 | 8 | 169 | 161 | 32 | 40 |
| CAL393 | 9 | 169 | 165 | 29 | 33 |

跨seed去重ID：FIT曾救回109、曾损失7；三seed均救回50、均损失1。CAL曾救回50、曾损失53；三seed均救回9、均损失20。某ID可随seed变类，unique救回/损失两个集合不保证互斥。完整IDs在summary.json，所有病例在case_classification.csv，不删不利案例。

空间图及ADE–FDE散点、逐帧位置误差/速度/加速度，让人工检查转向、速度、时机、终点与抖动。“更接近阈值”不是“理解运动意图”，边界跨越可能很小；提供中位与最大变化是为了避免只看极端样例。

三方控制可检查共同新head效果和学习clock额外效果，名义参数相同不代表有效容量相同。本轮小FIT拟合成功不是泛化；新输出继承原评分slot而未重算scorer，不把橙线当新选择算法性能。

## 数据核验

- 649唯一ID：FIT256+CAL393，七套路径，原native与保存eval cache逐项一致；past与原缓存对齐，全K20有限。
- 所有final预测的minADE/minFDE、逐候选完整覆盖及所有保存group计数与完成报告一致。
- 查看器输入来自six registered final120epoch，不读取80epoch正式结果替代本轮，也不按CAL结果重新选checkpoint。
- 本轮诊断文件夹、原脉冲查看器和直接读取缓存/metadata共1760文件SHA前后不变。不是重新运行全部历史25004 SHA；该全历史证据来自已完成实验审计。
- build实际峰值RSS0.141GiB/private0.128GiB，无训练、forward、optimizer、DEV/test读取。audit.json为PASS_FIT_CAL_SAVED_OUTPUT_VISUAL_REVIEW。
- 隔离headless Edge UI_QA PASS：默认计数、FIT/CAL切换、子组、跨seed稳定性、三方控制、敏感性、连续误差过滤、同slot联动、空列表安全、导出及390px布局；无页面报错，无保存人工标注。仅确认文本导出，不声称所有嵌入式浏览器都支持下载。

完整实验解释仍见 ../F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md；此处供人眼检查、提出后续假设，不增加实验次数，不开启下一训练。
