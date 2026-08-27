# Stanford Drone Dataset 低成本试跑

## 目的

只检验当前无场景F16+K20模型能否迁移到更密集的SDD轨迹，以及最多9人/Top-K4与最多16人/Top-K6的初步差异。该实验不是官方SDD benchmark结果。

## 数据口径

- 数据源：`D:\Datasets\Stanford_Drone\stanford_campus_dataset`；
- 仅使用Pedestrian类别与`lost=0`标注；
- bookstore video0、video1训练，video2验证；
- 每12帧采样一次，8帧观测、12帧预测；
- 每个视频最多600个窗口；
- bbox中心作为轨迹位置；
- 20像素归一化为1模型单位；最终汇总同时给归一化误差与像素误差；
- 不使用图像或场景语义分支。

## 一键运行

```powershell
& "C:\coding_toobox\envs\astra2\python.exe" "C:\Users\linke\Documents\New project\SDD_LOW_COST_SERIES\run_sdd_lowcost.py" --dataset-root "D:\Datasets\Stanford_Drone\stanford_campus_dataset" --device cuda:0 --epochs 10
```

脚本会自动预处理数据，然后依次运行N9/K4和N16/K6。中断后重复相同命令可续跑；已有完整metrics时自动跳过。只有显式添加`--force`才覆盖训练，`--rebuild-data`用于重新生成预处理数据。

运行完成后统一结果位于`sdd_lowcost_summary.json`。

正式低成本试跑已经完成，分析见`FINAL_REPORT.md`。当前N9/K4优于N16/K6，但两者Top-1都接近恒速参照。

## 限制

这是同地点、跨视频的低成本可行性试验。由于原始下载内容没有单应性/米制标定，结果不能与ETH/UCY米制指标或采用官方SDD划分的论文直接比较。
