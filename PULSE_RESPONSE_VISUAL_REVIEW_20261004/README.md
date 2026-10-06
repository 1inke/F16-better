# 脉冲响应的救回 / 退化案例浏览器

直接打开 `index.html`，完整保留 `data` 子目录；不需联网。沿用昨天的候选、ADE–FDE、所选与GT联动语义，改成三方法并排对照。所有1516个ID和六份constrained候选输出均可查，三个种子独立切换。

默认显示明确事件样本中，学习响应相对原F16“救回”的全部案例；点击“退化”可看全部损失案例。可切参考为同seed固定响应，或切跨seed稳定性。代表入口按双阈值距离变化幅度的中位数和最大值抽取，不按肉眼好看选例。详细口径和精确ID见 `REPORT.md`、`summary.json`、`case_classification.csv`。

深青线是事后最近双阈值候选，不是部署选择；蓝虚线minADE、紫点线minFDE可能来自不同slot。橙线在F16为真实原logit argmax，新方法只是继承原slot（未重新评分）。点击候选或ADE–FDE点，三方法联动同slot；可比较误差、速度、加速度随时间，以及对应脉冲中心/跨度。

只读保存的输出，不训练或模型重放、不读raw/test、不改原结果；新文件仅本目录。输入和原可视化/脉冲实验文件夹前后SHA见 `audit.json`。人工备注独立浏览器保存，关闭前请保存并导出JSON；内嵌下载不可用时复制导出文本。

复现：`C:/coding_toobox/envs/astra2/python.exe -B build_review.py`。完成输出不允许覆盖，需新目录。此工具不是新正式试验，不改变训练计数、成功标准或checkpoint选择。
