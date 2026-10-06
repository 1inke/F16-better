# F16可观测运动变化线索诊断

人类已明确授权执行前一份设计。此次只训练独立分析探针，原F16/decoder/scorer全冻结，原K20与原筛选不变；不计正式生成实验、不能宣布新baseline或major。

全部新增代码、缓存、探针模型、日志、失败来源、人工材料及报告均在本新文件夹。已有结果/源码/模型不修改，旧文件完整SHA锁定并前后核验。原设计文件夹保持不变。

执行：`preflight.py`通过后，使用唯一`start_independent.ps1`创建无周期触发的按需Windows任务。worker字节独占锁防重复。独立于Codex进程寿命，但机器需保持开机登录、不睡眠。诊断恢复先核验源锁，完成reader job核验SHA后跳过；未完成MLP恢复参数/optimizer/RNG/history。

原始数据的sequence/person/time从同一ETH-training来源标签恢复。采用最粗scene-family留出，以隔离重叠窗口。Zara录制合并，students/uni合并：只有三个场景组，因此不虚报可靠群组CI/Holm显著性。CAL保留原拆分、仅描述；F16已有训练历史不能被新探针分组“洗成”独立泛化。

未来GT仅离线标签/误差/元数据身份恢复，不进入任何探针特征、native forward或邻居选择。变化风险、类型、时刻、方向与候选/筛选分别报告。探针可读不代表原模型有效使用；输入敏感性不代表有益利用；弱信号不证明不可预测。

原预处理`get_valid_id`要求最小未来存在性；本调查保留既有邻居/valid-count，不用未来重新构造输入，也不传未来坐标。这一资格选择偏差必须披露，不能把当前输入下的表现无条件解释为部署时可观测信息。旧结果哈希继承完整历史manifest并加入相关新科学产物，不枚举无关外部数据仓。

计划输出：`binary_probes.json`、`secondary_and_controls.json`、`candidate_audit.json/npz`、`case_evidence.json`、`functional_sensitivity.json`、`prediction_audit.json`、`frozen_native_audit.json`、`RESULT_ANALYSIS.md`及最终`RESULT.json`。保存完整预测ID、读出概率及模型哈希。人工材料只有past-first/reveal数据，人工判断尚待你填写，不制造人类结论。

本轮改为预计时间的 **50%** 做一次进度检查，再按真实进度修正完成时间；启动确认后结束当前轮，不留在前台等实验。完成后自动分析需Codex app运行。
