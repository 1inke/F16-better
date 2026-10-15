# E8现有模型短测：单硬件、固定padding9

不训练。使用实际已有ETH seed1 F16+原H3/V2完整decoder/scorer，CPU一线程和同机器CUDA0各测valid agent N=1/3/5/7/9，实际张量始终9agents。这是mask有效人数曲线，不是动态缩短token或理想N²缩放实验。输入选择原train首个valid9 ID，不按GT效果；checkpoint/输入/相关来源初末SHA，模型全参数/boolbuffer beforeafter torch.equal。先实际cache-ID前向对齐≤2e-5，再测量，不用benchmark选择scientificcheckpoint。

实际总参数按deduplicatedparameters计数，逐顶层组件拆分，不把eval requiresgrad0误报为模型0参数。实际model含哪些地图/scene组件以named参数和cfg为准；配置USE_PRETRAINED_UNET/CACHED_FEATURES false如实报告。

MAC用torch profiler对它支持的matmul/conv FLOPs÷2，命名为partial supported arithmetic MAC；softmax、layernorm、elementwise、控制流等不计，不能伪称完整FLOPs。理论attentionQK+AV每次B×H×Nq×Nk×D(两乘积)只说明定义，9agent×8tokenspadding和模型分支使有效N不等于实际Nq/Nk。不外推N>9。

两device每N50warm+500measure，报告mean/P95和所有样本；计时涵盖obs从NumPy、设备传输、rawheadingnormalize、完整模型K20、逆world和CPU候选/logits返回；不含磁盘/checkpoint加载。CPU和GPU是同一主机的两个计算设备，不是第二部署硬件。基线CV/kNN input→CPU output均相同warm/measure，kNN bank已在RAM，含原训练bank标准化特征、原严格K32检索/ID平局排序和平均残差轨迹重建；不把baseline预处理剔除。精度轴引用原已审核ETH整体与P1，同模型同seed人口，不用timing虚构所有模型/边缘设备Pareto。

最多600秒、process6GiB/physical+commit eachreserve6GiB；GPU和CPU model顺序单实例，不改OS/pagefile/旧暂停进程。不修改旧源/权重/训练，新增仅本目录。

反方1同助手：本短测不能满足跨硬件效率、完整MAC、所有fold速度结论；固定padding恰好可以暴露声称按邻居数节约计算的不成立。测量和精度跨人口不当作严格Pareto优越性。PASS执行不等于C10全部支持。
