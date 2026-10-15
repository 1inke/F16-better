# R2生产封装（2026-10-13）
用户明确授权根据父目录设计开始执行。本子目录只实现固定L136 vs L136+RBF64三target-track OOF，六次CPU生产拟合；不重求R1见证，不训练旧模型或生成器，不自动B/S2。
父设计/核心代码保持原样，protocol绑定。先完整2119旧source/1040输出与标签证书复核、实际L136全三折重现、固定小子集两臂重复拟合/独立目标梯度Hessian及saved-map、资源吞吐后生产。预检拟合是数值/吞吐throwaway不用于结果选择。
最终source-lock继承旧digest，不覆盖；新报告/控制不改锁源。持续process6GiB/physical+commit EACH reserve6GiB。唯一按需0trigger任务和独立byte lock；ONEstartup/ONE50%elapsed→completion。所有门失败如实停止，不扫描参数。
ARS DA1（inline非外审）：RBF同时增加容量和有效正则化几何；同A适应性探索、P发现偏差/U未知与共同训练条件化仍在。命中锚点不等于部署覆盖或理解运动。
