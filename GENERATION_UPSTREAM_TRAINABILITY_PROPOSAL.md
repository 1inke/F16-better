# 下一大方向：受控F16末端时序块可训练性对照（待授权）
不是已实现、不是已启动。当前原始F16全部冻结范围下15项新实验均无major；这只削弱已测局部修正的价值，不证明F16瓶颈。
请求具体扩大范围：允许仅F16 agent_aware_trans_encoder.temporal_encoder最后TransformerEncoderLayer（当前代码num_layers=1，实际checkpoint/module须再次核验）的参数参与训练，其余F16模块、原native decoder/scorer均冻结。与全新同架构生成支路联合训练，对照同fresh支路/F16全冻结，三pairedseed/同预算/原loss/cal/保护/K20。额外可训练参数数不同须明示，不是等容量架构证明。
动态更新的观测tokens只供新生成支路/固定原decoder特征计算，不改变固定原native20基准或加载失败治疗模型。基准由锁定原F16/native decoder产出作为bound、安全比较参照；评分器权重固定不代表logits固定，仍只做候选oracle而不宣称筛选改进。
实际module路径/参数白名单、raw observation forward与缓存校验、梯度传递（新head零初始化首次encoder梯度可能为零，后续须实测非零）、冻结权重/source/data哈希、FIT/CAL/test隔离/finiteK20、两arm2epoch连续中断全params/optimizer/RNG/history/calibration EXACT须在授权后实施预检。不得继续用固定tokens缓存假称encoder可训练。
训练预算及ETA由真实在线forward预检吞吐决定，不能把最近cached18分钟直接套到onlineencoder。开训前报告ETA、Windows独立任务一次确认、保存PID/60%query/deadline和automation。原5%/protected1%/三seed/CI/samepath/coverage/dup/hardmotion/allgroupmotion1%及首次重大scene/sample/三seed充分性审查均保留。
不授权heldtest/跨fold/外部数据/付费cloud/外部消息/删除旧数据/解冻其他F16或scorer。若人类不批准，需重新决定仍冻结范围内的新大方向，不能擅自解冻或重复局部系数。
