# EXP-128 零门控 TCN 残差

保留完整 F16 temporal Transformer，在其输出后增加 dilation 1/2 的两层 depthwise temporal convolution；残差门控从严格0开始。
