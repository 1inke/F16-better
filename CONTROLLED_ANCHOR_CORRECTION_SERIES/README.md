# EXP-142：冻结源模型的受控锚点校正

本系列针对“接近10%的min指标提升”测试两个高杠杆机制。加载每个split/seed对应的EXP-139C三锚点最佳检查点，冻结编码器、交互、候选生成器和评分器，只训练轻量锚点校正头。

校正头同时修正第4、8、12步锚点，最大位移分别限制为0.20、0.25、0.35。与此前细化器不同，终点不再锁死，因此minADE和minFDE都可能改善。20条候选在训练和推理时全部校正，不存在Top-8漏选或训练—推理不一致。

- EXP-142A：保持原损失权重，单独验证受控锚点校正结构。
- EXP-142B：强化winner FDE、终点监督、端点多样性和ADE+FDE覆盖对齐。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\CONTROLLED_ANCHOR_CORRECTION_SERIES\run_controlled_correction_batch.py" --device cuda:0 --epochs 20 --batch-size 64 --seeds 1 2 3
```

默认运行2种机制 × General/Challenge × 3 seeds，共12次新训练。支持断点续跑，不要添加`--force`。

