# EXP-175A-R：SDD Soft Semantic Repair

本实验针对 EXP-175A 暴露出的语义资产失效，不训练轨迹模型。

修复内容：

- 不再对 ADE20K 类别名称做容易误匹配的子字符串搜索；
- 使用精确类别集合；
- 保留 SegFormer softmax 概率，不先做 argmax 硬分割；
- 将可行走与障碍概率归一化为互补软置信度；
- 输出 320×320 六通道缓存及 validation/test 视频 QA 叠加图；
- 用修复资产重新执行完整 EXP-175A 门槛。

SegFormer-B0 仍只参与离线预处理，不计入在线轨迹推理参数和 FLOPs。

