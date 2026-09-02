# EXP-175A-R2：Observation-Conditioned Appearance Audit

R1表明ADE20K类别语义不适合SDD航拍视角。R2不再假设“road/grass/building”标签
可靠，而是使用每个样本已经观测到的行人轨迹作为无泄漏可通行外观原型。

离线缓存每个reference image的RGB、HSV、局部纹理和边缘图；在线仅在历史轨迹
和20个候选的少量坐标处查询。候选经过与历史可通行表面外观相似的区域时，获得
更高可行性。该审计不使用未来轨迹构建原型，不训练网络，也不读取测试标签调参。

三个固定候选分数先在12个validation视频上选择，再仅一次报告17个test视频：

- `appearance_similarity`
- `appearance_plus_continuity`
- `endpoint_appearance`

门槛沿用EXP-175A：至少12/17个测试视频中oracle候选违反率更低，且validation
macro Spearman >=0.20或oracle违反率相对下降>=20%。

