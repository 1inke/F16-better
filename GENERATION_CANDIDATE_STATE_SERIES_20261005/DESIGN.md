# Candidate native-state conditioning：下一方向预注册（未启动）

## 问题与证据

已完成三response正式试验3/3总复盘。更自由时间增量、学习pulseclock、持续profile都没有稳健覆盖增量，tinyFIT学习后CAL迁移差，单改曲线不继续。新head的native_inputs是原context64+modeMLP64；base原候选仅在输出加法里，不直接参与condition。因此测试实际候选速度/曲率/终点显式状态是否帮助**观测→候选修正**共享映射，而不是增加profile自由度。state是原输出确定函数，不新增现实信息、不证明mode没信息/原decoder根因。之前mode/conditioning/geometry/social失败是反证。

## 唯一因素

zero：FIT标准化后的candidate-state32乘0；mean_std：actual32。双方相同nativecontext64+mode64、observed8token attention和state encoder32→32→32 GELU；同全初始化/buffers/zero last输出。state32经encoder与原160inputs拼接，head192→128→64→6；common learned sigmoid persistent response，D2+A2+clock2、center.05–.95/span.05–.5和原commonclip/integration原样。未改变profile、epochs、loss、scorer或nativebaseline。

状态来自immutable baseK20 displacement path：（1）12步两坐标速度24值，origin0且dt.4；（2）12步meanvelocity2；（3）末步velocity2；（4）endpoint displacement2；（5）11段acceleration magnitude均值1；（6）首末velocity heading signed atan2(cross,dot)1，任一速度<.03m/s则0。共32，顺序锁定。不读GT/group/logits/new未来观测；native预测位置不是新观测的未来GT。candidate slot原20order不任意重排。

FIT25912×20点逐块计算featuremean/std，floor.001，仅FIT一次，无GT选slot/样本/尺度；双方全部相同scalerbuffers，不在CAL/smoke重新fit。32维有确定冗余，非满秩/等有效容量证明；control第一层state weight数据梯度0但bias存在，treatment正。state分支权重数实际GPU核验，不只手算。

仅condition输入可用性不同，非纯信息/优化causalproof。native所有候选运动都可能错，显式state可能把错误强化；所以native/freshcontrolCI和覆盖/重复/motion全保留。

## 不变协议与权限

wholeoldF16/whole原decoder/scorer权重buffer固定；nativeK20不可漂移，fullK，无早剪枝/新rank，原20slotorder。仅past16+frozenobserved8tokens16:784/restzero加已预测base；GT仅原训练loss与posthoc评估。原balancedhardwinner ADE+.5FDE/diversity/residual/smoothness/protection.1、CALguard.01、FIT25912CAL2879、originalLR/order/batch80epochs/3pairedseed、clipL=1.5(speed+.03)保持。6fresh80epoch，3policy18outputs非18独立trainings。不warmstart旧failedhead/teacher。

探索架构依据过往DEV复盘提出，不冒充独立未看DEV确认；新parameters/scales只FIT不按CAL/DEV调宽度/系数/LR/epoch。scorer remains inheritednative slot，不能解释新路径概率。原humannewhead持续授权有效，不解冻原decoder/F16，不新loss/eventlabels监督/heldtest/fold/externaldata/cloud/messages/delete。

## 实施前必须通过

- 只在本新folder复制shim和newlauncher，不改任何父scientificsource/model/results；完整allprioroldmanifest继承旧digests、sourcebyte锁。失败provenance留存。
- FIT-only特征数值有限/variance/sensitivity，精确IDslot/basealignment和缓存nativeonline误差。
- GPU真实8FIT多步两arm headfront/output/stateencoder梯度有限按mask预期，allold权重buffer exact、fullpaired初始化/nativezeroidentity、finite128K20/bounds/splits/current-source/alloldSHA。
- 两arm真实2epoch continuous/interrupted params/allboolscalerbuffers/optimizer/torchCUDAloadernumpyPythonRNG/historycalibration EXACT，completedseedSHA-skip。
- 独立36throwawayFITsteps4warm32measure+CAL2879实测，仅吞吐，不选outcome或保存模型。原profile项目实际22min、PULSE85min、TIME269min，速度差原因未知，不能再把短基准compute当生产保证。
- approx17GB availableRAM，input/output/temp+peak实测，process6GiB/systemphysical+commit均reserve6GiB；按chunk运行，不调整OS/pagefile/kill用户程序。预估原数据及CPU旧模型约1–3GiB，加32states全FIT20约63.3MiB float32（不得误当518240×32×8多个副本无限缓存）。
- 报告实际ETA后unique0trigger Windows byte-lock/start_independent.ps1独立worker，唯一realPythonPID/serviceparent/currentstage/TRAIN/stderr确认；此前不登记正式运行。启动再新study0/3/保存memoryledgerstateautomation，ONE50%elapsedTIME→真实完成节点→退出，不轮询。

## 结果与成功条件

首先至少1同一candidate ADE≤.2m AND FDE≤.4m，完整coverage/rescue/lost;第二GT-free complete-link meanpath.10/endpoint.20有效good簇，不能复制候选凑数；scorer最后。.3/.5敏感性不替主标准。
原formalmajor全部保持：change ADE/FDE≥5%、overallstabletransition≤1%harm、三seed一致、native/freshcontrolADE/FDE/jointCIexclude0、samepathcontrol1%/coverage/duplicate/hardmotion AND pooledALLgroups speedp95/accelmean,p95/jerkmean,p95≤native1%harm dt.4。首次满足还审scene/samplecorrelation及3seed充分性。
保存18policySHA/18explicit1516ID finite20×12×2/6complete80history/allstoredsample/nativegroup/allpolicygroups/perseed/bootstrapCI重算、alloriginalfrozen/source/alloldSHA、双SUCCESS。算术不是所有model重放。全部结果包括失败；每3formal完成fullreview后改方向。

## 当前状态与反方1

IMPLEMENTATION_PENDING，未进行生产训练或宣称预检PASS。下一规划节点2026-10-05T06:00:00+08:00开始实现/preflight，不是实验完成ETA；只有正式real启动后才确定半程与完成节点。
反方1 inline同助手：state与mode可能冗余；显式反馈或增强容量可能仍伤CAL。若输入源/ID/gradient/scaler非可辨识、resource/source锁或权限实质阻碍，STOP保留真实失败并等human，不盲重新扫参数或改变损失/保护。
