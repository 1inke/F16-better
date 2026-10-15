# 原生锚定 Bernstein 时间耦合对照：结果分析

结论：执行与保存结果审核全部通过，但科研成功条件仍全部未满足。Bernstein 确实平滑了**残差本身**，却没有增加同一条完整好候选的覆盖，也没有解决完整轨迹的加速度均值退化。baseline 不变；当前 native-anchor 系列完整审核完成 2/3。

## 1. 本次真正比较了什么

两组都以相同 bottom4 原生候选为锚，保留其余16条不动，最终K20。共同25304参数、120维输入、同一组训练统计与原hardmin4 ADE+0.5FDE/平方hinge diversity/AdamW/25epoch/3seed；唯一改变是12个有界输出控制量直接逐时刻使用，或经degree11 Bernstein矩阵混合。残差上界仍为0.20(t/12)^2。旧F16/native/decoder/scorer未训练，输入只观察token与不可变原生路径；GT仅loss和事后指标。

训练18909、primary outer1035、inner1856、原CAL2879；后两者事后分析，不用来选择结果。Bernstein满秩且条件数7899.26447；同名义参数并不等于同有限可达集合、有效容量或优化几何。相同raw控制下端点相同，不意味着训练后两组端点相同。

## 2. 完整好候选覆盖没有新增

标准始终是至少一条**同一候选**同时ADE≤0.2m且FDE≤0.4m；不能用不同slot的minADE/minFDE拼成功。

| 变化组 | 分母 | 原生成功 | free12（42/43/44） | Bernstein（42/43/44） |
|---|---:|---:|---|---|
| outer，主结果 | 253 | 110 | 111 / 111 / 111 | 111 / 111 / 111 |
| inner，次结果 | 497 | 166 | 165 / 165 / 165 | 165 / 165 / 165 |
| 原CAL，次结果 | 700 | 243 | 243 / 244 / 245 | 243 / 244 / 245 |

不仅计数相同：全部36个seed×split×group配对记录的逐窗口成功标记、rescue IDs、lost IDs都完全相同。outer每seed rescue2/lost1；inner rescue0/lost1；CAL rescue4/lost4、3、2。路径数值不同，所以“相同覆盖”不等于模型或所有误差完全等价。

保护组也没有新增覆盖：outer overall625→628、stable434→435、transition81→82（两组每seed相同）；inner overall932→931、stable601和transition165不变；CAL overall1490→1493/1494/1495，stable994→993、transition253→257。完整所有72组记录见RESULT_ANALYSIS.json及completion_review/SUMMARY.json。有效good簇与distinctness仅次级描述，不能抵消变化覆盖或运动学失败。

## 3. 连续误差仍是微小变化，不是5%收益

outer变化组三seed均值：

| 指标 | native20 | free12 | Bernstein |
|---|---:|---:|---:|
| minADE，m | 0.249146619 | 0.249167579 | 0.249161776 |
| minFDE，m | 0.429378298 | 0.430941977 | 0.430913426 |
| J=minK max(ADE/0.2,FDE/0.4) | 1.390568401 | 1.394441346 | 1.394397882 |

Bernstein相对native反而ADE+0.00001516m、FDE+0.00153513m、J+0.00382948。相对free12的ADE/FDE改善仅约0.00000580/0.00002855m。

主outer变化覆盖对native净增+0.3953个百分点，探索target-track+3seed bootstrap95%CI为[-0.8890,+1.8315]个百分点；ADE/FDE/J差异CI也都跨0。Bernstein-free覆盖差为0且CI[0,0]，来源于保存窗口的二元成功集合完全相同，不是总体等价证明。其连续差异CI也跨0。全部36条bootstrap记录/144个scalar CI独立确定性复算误差0；多重探索、3seed和窗口相关性限制不消失。

## 4. 平滑有效，但完整路径物理保护仍失败

全部72个final组记录均违反加速度均值≤native1.01，只有这一项失败；其余speed p95、acceleration p95、jerk mean/p95均通过1%线。

- free12的36组加速度均值恶化+2.61265%～+3.05441%。
- Bernstein的36组恶化+2.66488%～+3.07252%。
- primary outer变化组Bernstein三个seed分别约+3.07252%、+3.05166%、+3.06910%。

因此不应把“jerk下降”当作全部运动学通过。这里的gate是完整ALLK路径、每seed每group pooled统计，而不是残差的运动学。

事后保存残差分解显示，outer变化组residual jerk mean分别从0.0157606/0.0149237/0.0151339降到0.0019519/0.0015347/0.0020263，即约87.6%/89.7%/86.6%下降；残差加速度只下降约4%～4.6%。这是平滑确实生效的证据，而不是新成功标准。完整路径仍FAIL。

残差幅度比上界的中位数free12为1，Bernstein约0.999943～0.999962，仍接近包络。保留二次增长包络可能使持久、幅度接近上限的修正仍带来加速度代价，这是机制假设，不是已证明的唯一因果解释。不能据此扫degree/bound/loss权重。

## 5. Loss下降的含义有限

两组epoch1→25训练total下降约2.38%～2.54%，diversity绝对值一直0，百分比未定义。这里是训练四条修正路径的batch人口归一化指标，不是整个K20的完整覆盖，更不是保存的全FIT集合指标。没有全18909 FIT coverage，不能虚构本次FIT覆盖成功而CAL不迁移。

free12全部9个科学输出，与上一native-anchor组11个对应保存字段bitwise EXACT；这是fresh重复执行的复现证据，不增加独立泛化证据。

## 6. 完成和审核的真实来源

生产03:03:03.109977启动，worker与exit双SUCCESS于03:07:06.139696，实际243.029719秒。唯一半程核验实际03:10:58.4496679才执行，比计划晚235.3406679秒，发现已经完成；没有按时半程或另一次completion检查的声称。

六个25epoch共150记录、六checkpoint-manifest绑定、18科学输出、103输出SHA、1003相关来源SHA、原生cache和固定16/bounds/B控制量保存算术全部PASS。CPUcompletion_review完整18保存质量、原SciPy GT-free complete-link聚簇、全部pooled motion、assignment、36 bootstrap/144 CI独立重算max0，初末SHA不变。它不是所有模型/梯度第二次回放，也不是外部评审；1003相关来源不是全部历史34k文件。

生产private峰值2.70721GiB，最小可用physical14.48670/commit24.14457GiB；CPU审核private峰值0.29998GiB。process6GiB及两项各reserve6GiB均保留，不更改OS/pagefile或关闭用户程序。

新增PAIRED_DETAILS辅助报告的布尔字段all72_seed_split_group_records命名不准确：实际36配对行，对应72组arm cells。原文件和源码保留；PAIRED_DETAILS_FIELD_CORRECTION.json单独校正元数据，不改任何算术或生产协议。

## 7. 反方检查（同一助手，非外部独立评审）

### DA2：是不是平滑根本没生效？

实际残差jerk显著降低，路径数值也改变，所以不能解释为B未执行。可是fullrank B的有限有界几何与优化不同，残差jerk不是完整路径gate；不能由这一个对照证明高频不是任何情况下的问题，或结构/loss为唯一根因。当前可否定的是：**这个固定Bernstein处理足以改善覆盖并满足完整物理约束**。

### DA3：是否应据此立刻改loss或scorer？

不能。训练objective只管四条修正路径，完整K20同时受固定16支持影响；覆盖阈值、锚定可达集合、饱和幅度、共享映射都仍可能有关。已有候选覆盖没有稳定改善且运动学失败，scorer不应提前。全FIT覆盖没保存，不能把本次直接归为“泛化且其他因素已排除”。相同二元结果也不证明生成器普遍无效。

## 8. 下一步边界

停止把本次Bernstein当作解决方案；不继续degree/kernel/包络系数/LR/epoch局部扫描。更值得设计的是“保留原生支持的修正能否在**完整路径物理可行性**下产生真正互补覆盖”的精确单因素问题，并明确它与四路径质量目标的关系。但本次不自动启动第三训练，不增新loss/assignment/scorer、旧模块解冻或数据权限。

原major各项、运动学1%、baseline、historycycle11和未完成总目标均不变。系列完成2/3，第三完成后才mandatory总review；这次执行成功不等于研究目标完成。F16已见原pool、target-disjoint非context-disjoint、3scene/3seed、重复CAL与adaptive bottom4的探索限制保留，不声称理解行人运动、独立wholemodel泛化或SOTA。

