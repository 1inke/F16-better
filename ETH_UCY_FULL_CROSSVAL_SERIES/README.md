# 当前模型：完整 ETH/UCY 五折训练与推理

## 实验目标

用当前筛选模型执行标准五数据集 leave-one-out：

1. ETH 为测试集，其余四个数据集训练；
2. Hotel 为测试集，其余四个数据集训练；
3. Univ 为测试集，其余四个数据集训练；
4. Zara1 为测试集，其余四个数据集训练；
5. Zara2 为测试集，其余四个数据集训练。

这与此前只在 `univ_data_train.npy` 内部做场景验证不同。每折使用已有的完整 `*_data_train.npy`，从每个来源场景的时间尾部保留5%作为 checkpoint 验证集，最终对完整 `*_data_test.npy` 推理一次。

## 固定模型

- F16 temporal-social cross；
- Kinematic K20；
- Coverage 0.20；
- EMA 0.995；
- 目标航向规范化；
- 50%概率镜像与0.9–1.1速度增强；
- 9个智能体槽位、稀疏 Top-K=4；
- 不使用图像场景分支。

## 推荐命令

```powershell
python "C:\Users\linke\Documents\New project\ETH_UCY_FULL_CROSSVAL_SERIES\run_eth_ucy_full_crossval.py" --device cuda:0 --epochs 30 --batch-size 64
```

脚本会依次执行五折，实时显示训练输出；若中途中断，再次运行会自动跳过已有 `metrics.json` 的折并继续。

只运行某一折：

```powershell
python "C:\Users\linke\Documents\New project\ETH_UCY_FULL_CROSSVAL_SERIES\run_eth_ucy_full_crossval.py" --device cuda:0 --epochs 30 --only eth
```

若显存不足，改为 `--batch-size 32`。不要使用 `--force` 恢复中断任务，因为它会重跑已经完成的折。

## 输出

- 每折 checkpoint、日志和指标：`EXP_133_CURRENT_MODEL_FULL_ETH_UCY/results/<fold>/...`
- 五折等权与样本加权汇总：`eth_ucy_full_summary.json`
- 便于表格处理的逐折结果：`eth_ucy_full_summary.csv`

ETH/UCY论文通常报告五折等权平均，不能用样本加权平均替代。样本加权值只用于检查大规模 Univ 测试集对总体轨迹误差的影响。
