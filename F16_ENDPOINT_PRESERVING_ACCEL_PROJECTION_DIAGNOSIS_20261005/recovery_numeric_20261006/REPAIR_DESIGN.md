# 数值修复预注册

人类已批准“先修复数值实现，保持科学标准不变”。不修改父失败科学代码或报告，不加入物理绝对floor、浮点宽限的科学gate或经验调alpha系数。

1. 六已有head仍float32完全冻结eval/no_grad；原生b和head增量u在投影前转换float64。预测及原输出仍原float32，投影坐标/增量以float64累计和保存，输出dtype差异明确记录。
2. alpha实数目标仍min(1,.01*M_b/D)，D0仍alpha1；不使用GT/groups/logits/CAL或质量表现调整alpha。只为有限精度保护预留由IEEE754 double unitroundoff导出的数值余量，而非放宽1%规则。
3. R=max_abs(b)+12*max_abs(u)（逐窗口，覆盖候选坐标累计范围）；unitroundoff=eps64/2，gamma512=512*u64/(1-512*u64)，E=8*gamma512*R/dt²，dt=.4。512是保守浮点运算计数上界（12步均值/累计、有限差分、二维范数及220项均值），8来自二阶差分绝对系数和二维L1范数上界；不是拟合系数/物理floor/场景超参。实际alpha_safe=min(1,max(0,(.01*M_b-E)/D))，并报告nominal-safe差和E。此保守数值预约与直接重算保护必须同时报告；不能用E作为评价容差。若资源或实测保护失败STOP，不改变E公式以求通过。
4. 严格M_b=0且D>0时alpha0。常数time-mean在float32可表示格上舍入一次后转float64，避免非二进制有限均值的12次累计产生非零二阶差分。原u是float32，因此该舍入只是数值表示，端点sum(u)与12*mean的误差仍须原2e-5m内；绝不声称bitwise终点相同。D0直接采用原u转换float64（恒定增量）。其余情况用float64 time-mean与原零均值时序偏离。
5. commonclip用输入原bound转换float64（生产有speed时）；保持坐标wise原clip。D0/alpha0/均值格处理都有实际endpoint与clip审核，不添加事件回退、删静止或候选复制。
6. 每批保存float64路径按与原报告相同的坐标→速度→加速度定义，独立NumPy重算native和projected，严格判断M_project<=1.01*M_native，不加任何epsilon或floor。修复后同一候选endpoint误差<=2e-5m，独立float64原投影一致性<=2e-5m。所有组另外查五项motion，不把accelmean通过说成全motion成功。

重要限制：E是保守实现预约，不凭它代替实测或宣称所有输入形式验证；下溢/溢出/不有限/超端点误差/独立科学gate失败均停止。若预约显著改变alpha/轨迹，须报告并STOP方法判断，不靠CAL选取。只合成测试通过不是六head或全旧SHA通过。修复不证明投影可学、CAL迁移或结构根因。

预检先原五fixture×CPU/CUDA，另同向紧边界、反向扰动、确定RNG20261006的64窗口mixed静止/恒速/弯曲。NaN拒绝、保存算术、源锁/旧失败source不变。之后实际FIT128冻结六模型重放/91596参数结构/全weight bool/scalerbufferexact/ID/cache/split/旧SHA/资源及8batch吞吐。

数据仅FIT25912/CAL2879；原major和motion1%/质量.2.4/GT-free去重/scorerlast及6GiB过程/系统两项各6GiB余量不变。NONFORMAL保留2/3。无optimizer/newloss/event监督/旧decoder或scorer训练/DEV/外部data。

反方1（同助手inline）：数值格均值与alpha余量会使数值函数与实数表达略有差异，必须量化，不能叫bitwise同endpoint；float64输出也必须标识。纯数学保证不替代严格保存坐标重算。若出现超过舍入量级的改变或不能满足原审核即STOP。
