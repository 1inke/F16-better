# EXP-040 F16 跨源场景验证

使用固定 K20 解码器，对 F4 与 F16 在 val_scene=0/2/3/4、seed=1、30 epochs 下进行完全相同的 leave-one-scene-out 对照。与既有 scene=1 结果合并，检验 F16 的提升是否跨场景成立。
