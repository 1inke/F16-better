# Stage2：冻结F16原始观测干预（结果前登记）

状态仅由runtime/worker决定；这是预注册快照。前阶段E0/E1/E2/E7已审核，本轮E3测试扰动、E4邻居、E5几何，E6随后单独train-only kNN，不能把未做项称完成。

五折×三个现有模型seed，每折固定SHA256 tag 20261009-stage2:fold:ID选最多1024测试窗口，不用GT选择。模型来源、NonCV/cue阈值沿用stage1已冻结；所有stage1和旧科学文件SHA不可改。测试复用已评测benchmark，OOD敏感性不是必要性/泛化/理解证明。全部32种干预×15模型=480科学jobs，另15 baseline保存不混计。

E3：L2/3/4/6把全部agent前缀填成首保留位置，不制造CV历史；最后两个位置保留。前6点反转/ID固定shuffle，末两点不变，故检验末速之外的历史敏感性而非全序列倒转。完整L8为baseline。关键Mlast2/MnoNbr和feature消融fresh重训仍待后阶段，不把遮蔽替代重训。

E4：noNbr(valid1)、全部邻居沿目标观测法线远移10m、下一个hash窗口邻居重定位到同目标当前原点（没有未来选择）；leave1..8真正压缩有效agent，非法slot是identity记录。全部从rawobs重算focalheading、center、encoder、decoder/logits，不能编辑cachedtokens。通过局部运行包装原torch.softmax采集F16 temporal_social_attention实际B,H,9,72权重，不改值；按时间求和/heads平均，preflight必须普通前向BITWISE相同。保存每slot权重与leave效应；无捕获必须明确不可观测，不能把mask当attention。

合成：每折最多128 hash顺序、末步>=.02m的窗口，不按GT筛；从该折正式train真实有效邻居速度池抽样，以ID/固定tag选，最多32次确保相对速度>=.02m/step及法线侧向投影abs>=.1。构造w=vN-vT，n=perp(w)/||w||且dot(n,target-left)>0，xN=xT-5w+d n。左右d=.3/.6/1/1.5/2.5，保留target并仅一synthetic邻居，baseline用相同target无邻居。连续CV tca实际核验误差<=.002采样步/最近距离误差<=2e-5m。n不强制等于target法线，属于为了真实closest-approach的几何修正，邻居速度仍取原经验向量，未用GT曲线。报告抽样/资格/失败数。P2 signed endpoint away=-n，top1与meanK分开，每d两侧所有种子与trackCI/剂量斜率；多重比较最终预注册family本轮2reads×5fold×3seed×5d×2side，不挑距离，未知机制保留。联合预测minimum距离按两agent各自top1（非联合概率）。合成输入是分布外干预。

E5：每ID固定hash旋转0..2pi、x镜像、平移(10,-7)m、尺度.8/1.2。先raw变换再完整前向，世界相对输出逆变换与原预测配对；translation输出原本相对target，不能再次减offset。同slot差/原score选top1和meanK分开；尺度改变物理速度不是exactinvariance。增强对照后续fresh重训。

保存每job显式ID/track/GT/obs/valid/20原预测/原logits/干预预测/合成构造以及每group同candidateADE<=.2AND FDE<=.4、J、rescue/lost、原SciPyGTfreeALLK complete-link .10meanpath/.20endpoint有效good、distinct和五pooledmotiondt.4、pairedtrackCI。主要subset all/nonCV/nonCVcuepos/C3/noC3/stationary。未来GT/groups/IDs不modelforward，GT只事后评价。old所有parameters/bool/statbuffers eval/no_grad/gradNone和values torch.equal。source/preflight/source-badoutput拒绝完成后才production。没有optimizer/newloss/scorer/unfreeze/newdata，frozen stage不冒称两epoch优化恢复。

资源：process6GiB/system physical和commit各reserve6GiB、长期约17GB；每batch128/单模型/有界输出、不调OS/pagefile/杀用户程序。preflight实际全部15模型干预frozen、保存数学roundtrip/closestfixtures/geometry变换/相同source和数据、真实8batch+保存统计聚簇时间后估ETA。sole0trigger OS task+byte-lock，一次startup，唯一预计50%检查再实际完成定点。所有产物只stage2_raw_interventions。

反方1（同助手非外审）：prefixmask和synthetic均OOD，响应可能来自归一化/排序变化、attention相关非因果、镜像不保证训练分布；保留固定score-slot补充、所有负例和三seed，继续重训而不把本轮当理解证明。
