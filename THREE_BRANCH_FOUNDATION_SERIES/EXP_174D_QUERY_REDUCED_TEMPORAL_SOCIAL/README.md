# EXP-174D：Three-Representative Temporal-Social Cross

Profiler显示F16编码器约占总设备时间32%。代码审计进一步确认，当前F16已经是
每个智能体一个聚合Q，并非8个历史token全部作为Q。因此本实验实际压缩的是
temporal-social cross中的K/V：

```text
8 history K/V
  -> long summary (t1-t4 mean)
  -> short summary (t5-t7 mean)
  -> exact latest state (t8)
```

单个agent Q、稀疏邻接、关系偏置、关系值消息、H3、Joint-V2和K20保持不变。
该设计与F35的两个半段平均不同：它明确保留最后状态，并将近期历史单独汇总。

正式协议为ETH/UCY五折、seed 1、H3 30 epochs + Joint-V2 20 epochs。通过条件：

- batch1 latency改善 >=15%，或FLOPs改善 >=25%；
- macro J退化 <=0.5%；
- macro minFDE退化 <=1%。

