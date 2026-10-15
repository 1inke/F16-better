# R3生产封装（2026-10-13）
人类“继续推进”明确批准当前ONE固定表示的窗口内排序实验，撤销设计-only等待门。只pointwise BCE vs全部同窗口P-N logistic，固定R2每fold200维map/PNU/OOF/权重/正则/求解器。六次CPU生产拟合，预检throwaway另计；无生成器/F16/scorer/Torch/solver/newdata/B/S2。
原设计与核心不改；本子目录补来源继承、真实资源guard、预检、完整保存与独立CPU审核。空分组保存N0及null，保留全部384验证窗口。process6GiB/physical+commit各reserve6；planning>30min STOP不启动。只ONE0trigger OSworker，启动后给实际ETA，一次预估half→completion，不高频轮询。ARS内联DA1非外审：all-pairs非any-positive-top4，shared历史项与截距抵消非断连；有效自由度与损失曲率变动，目标替换非唯一credit原因。
