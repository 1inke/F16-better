# Gaussian transient vs sigmoid persistent response

新response study第三项，仅newhead响应profile变更，双方均learnedclock；zero Gaussian/mean_std sigmoid，不是原fixed-clock对照。所有原F16/decoder/scorer冻结，旧科学代码、模型、结果不改。DESIGN.md是预注册科学与资源边界。

先run_native.py --action preflight --device cuda:0，再test_resume.py、benchmark.py；全部当前source验证PASS且报告实测ETA后才start_independent.ps1独立任务。run_native.py --action run仅用于独立worker；不要重复worker。正式6NEW80epoch/3seed/18policy不是18独立训练。

训练后completion_audit.py独立保存指标重算（非全model重放），新supplement算全同路径coverage/有效好簇/继承slot/早中晚。完成必须全部审计+双SUCCESS+fulloldSHA；第三完成必须完整response研究总review，不能盲开第四profile/LR/系数/epoch变体。

约17GBavailablehostRAM限制，process6GiB，系统physical/commit各reserve6GiB。预检记录实际峰值；任何资源/科学/权限实质阻碍停止并等待人类。仅中途一次50%TIME与完成截止查询，其他时间退出静默。
