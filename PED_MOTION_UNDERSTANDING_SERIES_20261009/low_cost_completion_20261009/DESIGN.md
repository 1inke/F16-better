# 低成本完善：只读保存结果，不训练

依据2026-10-09人类最新时间预算：现有模型/保存输出优先，长时间重训跳过；原E3进程继续暂停，不恢复、不终止、不重复启动。

本补充唯一计算问题是E5候选集合的几何差异是否主要为slot重排。读取已审核的五fold×三seed×五几何干预，共75份保存输出；每fold原固定hash测试人口不变，原数据已经逆变换到同一坐标系。不读取新模型、不前向、不使用GT选择配对。

令D(i,j)是原slot i与变换后slot j的12步平均Euclidean路径距离（米）。报告原slot平均距离、Hungarian最小一一配对平均距离、双向nearest集合平均距离；endpoint距离沿同一个ADE最优配对计算，不另外选endpoint oracle。Hungarian比同slot更低是优化定义的必然结果，不把这个差值本身当运动理解证据。近似等变无预定科学界，不新选阈值声明PASS；尺度改变物理分布，不要求精确等变。

保留all/nonCV/nonCV_cue_positive/stationary/C3/noC3等原mask、每fold/seed/变换分母和幅度。不同seed同样窗口不是独立重复。只作已有结果之后的描述性诊断，不新增p值/多重比较主张，不替代fresh旋转增强训练。

资源：CPU单线程、每次一份输出、计算chunk128；process上限6GiB，physical/commit各至少6GiB余量。协作式计算上限120秒（I/O单操作可能略超）；超出保存partial并停止，不改数据或扩大预算。输入done/source/output SHA初末验证；新文件仅本目录。

反方检查1（同助手非外审）：历史输出已知，禁止借匹配指标追溯改主判据。只降低slot差异不能证明整个候选集合等变或scorer正确。通过这些限制继续。
