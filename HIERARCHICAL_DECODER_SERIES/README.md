# EXP-139：无CV层次化提议—选择—细化解码系列

固定 F16、Top-K=4 稀疏注意力、robust state、损失、EMA和原数据协议，只替换解码器。所有参考点均由模型相对最后观测位置直接生成；最后速度只作为特征，不进行 `last_velocity × horizon` 的CV外推。

四组单变量递进实验：

1. `endpoint_proposal`：20个学习终点，线性粗轨迹；
2. `endpoint_select_refine`：终点模式评分，Top-8用12个时间查询细化，终点严格保持；
3. `anchor_proposal`：每个模式学习第4/8/12步三个锚点，分段线性粗轨迹；
4. `anchor_select_refine`：三锚点模式评分，Top-8经锚点交叉注意力和时间查询细化。

细化版本最终仍输出20条轨迹：8条被选轨迹进行细化，另外12条保留粗轨迹，因而可与当前minADE20/minFDE20严格比较。训练时保证粗候选Oracle模式进入Top-8；推理完全由预测分数选择。

默认运行 General/Challenge、seed 1/2/3，共24次新训练。当前K20 Baseline直接复用，不访问官方测试折。

## 运行命令

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\HIERARCHICAL_DECODER_SERIES\run_hierarchical_decoder_batch.py" --device cuda:0 --epochs 30 --batch-size 64
```

支持断点续跑。正常情况下不要加入 `--force`。

## 前置验证

- 四个解码器均通过语法、前向、联合损失反向和有限梯度检查；
- 代码中不存在 `last_step × pred_len`、CV endpoint或恒速参考轨迹；
- Top-8训练选择与推理选择张量形状正确；
- 细化器最终点与粗提议严格一致；
- 最复杂的三锚点选择细化版本完成1轮端到端训练，EMA、checkpoint和诊断指标正常写出；
- 连通性短训练中，细化使粗minADE从0.4509降至0.4353，同时minFDE保持0.8150不变。该数值不作为正式性能结果。

参数量分别为：

- Endpoint Proposal：294,870；
- Endpoint Select–Refine：329,816；
- Three-anchor Proposal：295,474；
- Three-anchor Select–Refine：330,548。

最大模型仍比TUTR复现的441,113参数少约25.1%。

## 时间估计

共24次新训练。提议版本预计接近原模型，细化版本会增加约30%–50%的训练时间。预计总时间约6.5–7.5小时，建议预留8小时并夜间运行。
