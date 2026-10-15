# 读取器修复与生产边界

- 首次预检在完成全部18source ID/native/group/slot审核后，actual128读取时因旧NPZ含零维metadata，统一slice触发IndexError。未运行生产worker。
- 保留失败JSON与failures/preflight1_scalar_reader/preflight.py；仅改预检小批读取器，对逐sample数组切片、保留零维元数据。未改政策、阈值、旧科学来源或模型。
- 完整重跑预检PASS：1490继承来源扩为1614；全18 ID/cache/slots+两128保存读回完整quality/SciPy聚簇/motion/两政策loop误差0；源初终SHA PASS；分析skip与坏source/output拒绝PASS。不是训练RNG恢复或旧模型重放。
- 8实际128双政策+四port完整算术共17.71555s，34620跨arm-seed重复row名义598.93783s。生产12–22min/planning15不是保证，completion CPU复核另计。
- 预检private1.66938GiB/minphysical16.09297/commit25.79577，仅预检非生产峰值。process6与两系统余量6维持。
- 新源已锁；README/DESIGN pending为预注册历史快照，实际runtime和唯一worker优先，不重复实施。
