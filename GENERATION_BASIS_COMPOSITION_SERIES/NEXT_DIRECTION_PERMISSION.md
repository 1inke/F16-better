# 下一方向授权提案（未实施、未训练）
     
现状：七周期21项正式实验没有达到全部重大标准。第七周期三项和之前FIT记忆诊断提示，增加/重新参数化观测读出或减小数值病态并未稳定改善突变组终点误差；SiLU的FITloss约.3723->.3690但DEV收益不可靠。不能因此证明原winner损失是主因，也不能保证更软的监督有收益。
     
建议请求的新增权限仅为：允许在新的独立实验文件夹修改候选生成训练目标中的单winner项，并与原winner项做全新配对对照；原regularizers/protection.1、calibration/guard、nativeK20、scorer、数据边界和重大门槛保持不变。不请求更多F16解冻、原decoder/scorer训练、外部数据或heldtest。
     
待确认后的首个候选设计：samehead/init/fit25912cal2879/80epoch3seed/budget，hard minimum jointwinner对照 versus软minimum候选集合监督。训练GT只进入误差与loss，推理只读历史观察，绝不按GT分组路由；软权重不作为推理评分器。温度仅按预注册FIT尺度固定，不用CAL/DEV选值、不扫参数。软minimum改变winner梯度分配，可能照顾多个近GT候选，也可能鼓励同质/塌缩或抹去远模态，因此原coverage/duplicates/motion约束必须同时审核。详细公式需先核查原objective实现再锁协议，不将本提案当已运行方案。
     
在明确授权前不实施或启动新loss训练。现有同范围授权仍有效，但不擅自撤销“original winner loss保持不变”的锁定约束。当前无训练运行，旧结果全部保留。用户授权后重新建立自动回访，完成实际GPU/冻结/有限K20/source/IDS/原结果SHA以及双arm实际2epoch连续中断optimizer/RNG/params/buffers/history/calibrationEXACT，报ETA/独立OS唯一启动确认后才开启下一正式周期。
