# Attention Compute Series

本系列直接修改注意力计算机制，固定 K20 解码器、运动学稀疏图与训练协议。每个正式变体写入 `AUTONOMOUS_EXPERIMENTS/EXP_043` 至 `EXP_049` 的独立目录。

运行：

```powershell
python run_attention_compute_batch.py --device cuda:0 --seed 1 --val-scene 1 --epochs 30
```
