# S1b 精确预注册

唯一新增机制为12个双线性交互；124维加性控制和136维处理同C1/balancedL2/newton-cholesky/tol1e-6/max500/one thread。保留旧120维保存结果及bottom/random，不能把parameter count差异称equal capacity。

观测源GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz past8×2，ID与原train_observations.npz全28791严格一致，past与observation首16/160:176逐元素EXACT，最后观测坐标严格0；旧只读extract源码证明native以最后观测为原点且focalheading同坐标。dt=.4沿用原event协议，非视频frame推算。只用已观察last3points，delta_v是速度差非除dt加速度。native前中后各4个速度从[0,native]计算，不把未来预测当新观测。

u=v_last或delta_v，与三段native均速的dot及signedcross共12，按u-band-dot/cross固定次序。浮点定义位置源float32，减法/产品float64，最终所有新增特征float32。控制共享120+观测4，加性历史在同窗口仍会抵消；处理interaction才history-conditioned。改变feature数量也改变正则几何，结果不是purecausal运动理解。

完全复用A7334/B11575/3trackOOF标签和hash，无重划；train-only每arm每foldmean/std，算法同但数值不同。保护旧models/results/hash1953全部原digest继承，不覆盖。源为相关范围非全历史34k，无torch/GPU/model forward。

主门：OOF overall nativefailedH4对bottom4和固定random4均>=5pp且97.5%trackCI下界>0；三个foldchange不低于bottom；正标签track每折>=20。只有处理过OOF才fullA/B，B overall对两个control差均>0且change>=bottom才S2可行。本次不自动新神经训练。失败停止这一个probe，不扫C、宽度或split、不放宽。原major和五motion1%未来仍保持，本次native路径不变不制造部署coverage收益。

固定历史置换：每fold先按tag s1b-history-swap-v1的track SHA顺序分块、块内ID SHA排序，固定循环位移最大track窗口数；要求最大块<=N/2且所有donor不同targettrack；只按identity不GT。p2和past一起置换，native/GT/标签不动，不拟合新模型。保存全donor IDs及分组H4、选槽变化。它为distribution-shift敏感性，不能单独证明理解/因果；加性选槽原则不变但sigmoid精确tie需实际报告。

独立复核：全28791标签/ID/past对齐，实际128 vector/scalar特征EXACT、constantvelocity/delta_v0、dot/cross反射/共同rotation/translation、GT和group无forward接口；both513固定fixture真实fit重复BITWISE/独立objective Hessian reference/FD，badsource-output拒绝；真实1033 throwaway吞吐，不选H4或保存benchmark模型。预检PASS才source-lock，planning<=30min/process<=6GiB/physical+commit各reserve6GiB。

生产6OOFfits+conditionalfullA；保存所有train/validationIDs/normalizer/weight/manifest/prob/slot/swap/summary和CI，全部scientificSHA。CPU saved独立复核无refit/torch。唯一按需零triggerWindows byte-lockworker，ONEstartup/ONEhalf→completion；首次已完整不重复检查。

ARS inline反方1（同助手非外审）：正例为乐观包络，交互不保证真实可修复；增维/正则、F16seen/contextoverlap、adaptive S1b限制如实披露。改变历史诊断为OOD而非心理意图证据。后两反方在完整结果时审核。
