# 保存读取器修复来源

首轮N0/N1/N2/N6完成，N3首seed后synthetic循环将n改为128，第二seed真实ETH364索引与128不匹配。属于脚本局部索引变量错误，不是数据/模型/科学结果异常。父FAILURE.json、analyze_saved.py、DESIGN/protocol及全部partial保留。child only两项差异：P指向原seriesroot（隔离目录）；每seed真实noNbr读取前重置n为原selected IDs数。定义、阈值、分母、GT使用、模型/输出均不改。全短分析重算，有界600秒；不是重训/optimizer恢复，不覆盖旧证据。

