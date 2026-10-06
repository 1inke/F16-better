# 启动前源码锁定失败：停止说明

状态：STOP_NEW_SOURCE_EDIT_AFTER_LOCK_WAIT_HUMAN。无生产任务、PID或正式训练，physical-feedback研究仍2/3；旧baseline/正式计数不变。

## 原因与责任
助手在当前源版本预检、真实中断恢复和吞吐测试完成后，又补改了新 quality_audit.py 的报告 verdict/title 字符串及 check_once.ps1 的进度展示字段。最后源码校验检测 quality_audit.py 与已保存541-source锁不一致，立即停止；check_once.ps1也发生差异。不是训练失败、模型数值失败或旧科学文件更改。不得悄悄更新expectedhash、回退掩盖修改、跳过最后gate或直接启动。此编辑顺序错误由助手造成。

failures/preflight_1791222401792826400完整保存trace/报告/代码。locked_quality_audit.py与locked_check_once.ps1的SHA分别
8dcd32bc25fc2994761b421f49f68db4a5a805334620baee4099e613dfd7b7a1 /
f2f1e32e5c3666de1ee0d8c17394ef6f79c50b403d3ae9178678f385116523da，
实际Get-FileHash与预检锁精确匹配。它们是失败来源保留，不覆盖当前文件。

## 已实测，但不能替代最终 current-source PASS
- 初始源版本实际GPU8 FIT steps，两arm91596 uniform双专家，paired weights/buffers/nativezeroidentity/finiteK20/bounds/旧wholedecoder frozen、canonicalnative cachepath 1.1920929e-6/logit1.2516975e-6m容差内、独立物理算术PASS。31868old继承digest初末在科学预检阶段通过；末次全workflow在source错误处停止，不能冒称最终全SHA完整审核通过。
- 两arm真实FIT/CAL-subset2epoch连续/中断参数及bool/scalers buffers、optimizer、torch/CUDA/loader/numpy/Python全部RNG、history/calibration/objective diagnostics EXACT，完成seedSHA-skip及badsource/output拒绝PASS；不是完整25912两轮，非分析恢复。
- 前8个真实FIT步骤 P均0，投影alpha均1，尚未自然激活；梯度非零证明网络连接，不证明实际激活投影学习。D0/零预算/非零 synthetic output投影梯度fixtures有限，但不证明float32网格量化的有限差分可微。若人类批准重新预检，应补充固定非零actualhead参数的active-alpha分支连接审核，不调科学超参/训练权重。
- 36throwaway FIT128每arm4warm/32measure：.0177641/.0215476s；CAL2879 .0724312/.0947165s；compute32.58974min仅吞吐，不是生产ETA，没有保存benchmark模型或选科学outcome。
- 峰值RSS2.59914/private3.59842GiB，systemphysical最低16.75097/commit15.69151GiB；既定6GiB进程/系统两项各6GiB余量通过。无OS/pagefile/用户程序变更。

## 下一步（等待直接人类确认，不自动重试）
仍按DESIGN唯一因素训练内投影，两组都uniform双专家、float64累加、原loss+.1P、FITCAL/LR80预算/旧冻结范围/K20不变。以最终代码建立可验证的新预检运行来源，保留本轮所有证据、不覆盖resume_smoke，重新执行适用实际gates与current-source/allold SHA。加入上述activebranch明确验证；任何实质科学数值/源/ID/资源/权限障碍停止。全部PASS才实际报ETA并唯一OS独立任务启动，随后ONE50%预计耗时→完成节点退出不对话轮询。无新loss/scorer/解冻/新数据或放宽1%保护授权。原投影冻结诊断已ANALYZED，所有旧完成worker不查询/重训。
