# EXP-183A-R2-V：Spatial DINO Mode-Conditioned Scene Audit

## 结论

最终判定：**STOP_SCENE_VISUAL_BRANCH**。

实验完整使用 SDD 的 31/12/17 视频划分，包含 8418 个训练样本、3763 个验证样本和 2679 个测试样本。EXP-175S0 seed1 的 F16、H3、Joint-V2、20条候选轨迹及原始评分器全部冻结；本实验没有改变 minADE/minFDE 候选集。

验证集选择的是 Q3 video-centered，但它相对同容量 Q0 model-only 在测试集上的结果为：

| 指标 | Q0 | Q3 | 变化 |
|---|---:|---:|---:|
| Video-macro Spearman | 0.7981 | 0.7948 | -0.0032 |
| Top1 winner hit | 18.85% | 25.35% | +6.50 pp |
| Top5 winner recall | 71.00% | 71.44% | +0.45 pp |
| Mean regret | 15.989 px | 16.164 px | 恶化 1.09% |
| P95 regret | 58.556 px | 58.689 px | 恶化 0.23% |

只有 11/17 个测试视频的 Spearman 提高，且提高幅度通常很小；较少数视频的明显退化抵消了这些收益。因此 Spearman、Top5、跨视频一致性和 mean-regret 四项 PASS gate 全部失败。

## Camera-style suppression

训练集内部 video-ID 诊断为：pooled DINO 22.68%、spatial raw 32.13%、crop-centered 19.39%、video-centered 61.81%。Crop centering 确实降低了视频身份信息，但没有增加候选质量判别能力。Video-prototype removal 反而形成了更强的视频指纹，并在测试视频上失去验证集中的微小优势。

## 解释

Q3 提高了精确 Top1 winner hit，却让平均 regret 变差，说明它偶尔能找到最佳候选，但错误时会选择代价更高的候选。当前原始 Native score 的 Top1 hit 为 27.10%、mean regret 为 14.891 px，仍优于 Q3 的 25.35% 和 16.164 px。

固定 224 source-pixel crop 下，测试集 t4/t8/t12 锚点的有效覆盖率分别为 98.37%、72.85% 和 54.98%；远期查询覆盖不足是限制之一，但依据本轮预先定义的 stop rule，不再进行 crop size、视觉骨干、DINO层或去域系数搜索。

结合 EXP-183A 的全局场景意图结果，本轮表明：即使保留 DINO 空间结构、抑制部分 camera style，并对每条 H3 候选分别查询场景，静态视觉特征仍未提供足够稳定的候选质量残差信息。后续研究应回到候选生成、覆盖与原始评分机制本身。
