# M1 实施预注册

直接实施父 M1_DYNAMIC_HISTORY.md / COMMON_PROTOCOL.md，无科学因素更改。A末速度重复7token、物理零加速度；B真实7速度+6有效加速度。共同FIT统计、mask、P2、34146参数GRU32目标网络/20query/既有G1中心，全fresh paired，目标直线路径。

物理差分精度固定：存储past32转64差分/.4后cast32速度；速度32转64差分/.4后cast32加速度。共同float64均值标准差；标准化token转回NNfloat32。锚定末速度沿既有G1float64来源，逐ID精确核验。不以特征cast差异创造真实新观测。所有raw统计来源仅FIT，无效首加速度不计。

每seed两臂完全相同初始state字节；同loader顺序。只合法预测输入P2/past/末速度，不接收GT/group/ID/native/logit/scene。旧F16不实例化/训练，缓存SHA保护，不冒称全旧GPU重放。

反方DA1（academic-research-suite，同助手inline非外审）：P2原本可能有历史；比较只能识别新增动态输入的可读出增量，不独证顺序/加速度唯一因果或理解。equal nominal不等有效容量。预检不筛选效果。M1-S1不是最终major或安全门，原五motion literal1%与5%ADEFDE全保留。

所有24finaljobs含完整FIT/outer/inner/CAL；exact samecandidate质量/190pair/SciPy GTfree complete-link/五pooledmotion原12速度11加速度10jerk口径。26family/20000crossed bootstrap，不缩family，不把3seed×窗口当独立。完整保存rawhash128、goals/path/assignment统计。source/stat/IDs/checkpoint/optimizerALLRNG绑定，失败来源保留。预算>30min不启动，不偷偷削人口或epoch。
