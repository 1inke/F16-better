# Observed temporal contrast experiment (cycle4 experiment2)

Formal start: 2026-10-02T02:55:55.570976+08:00; OS worker PID45988, serviceparent2728, task Trajectory-TokenContrast-Cycle4 (on-demand, no recurring trigger). Six fresh trainings: zero contrast vs temporal contrast x seeds1/2/3,80epochs. Estimate35–50min (planning40); single60%check 2026-10-02T03:19:56+08:00; initialanalysis 2026-10-02T03:35:56+08:00.

Both36454 newparams: value projection96->32, originalcontext64+mode64, mean32, zero32 vs last4-minusfirst4 projectedtokencontrast32, head192->128->64->6(lastzero). Identical fullinit/nativeoutputidentity. Originaldecoder/F16/scorer all frozen; nativebase+3anchorcorrection/commonvelocityclip1.5*(speed+.03). Not independentabsolute generation, intent recognition, or ranking improvement. Control contrast input gradientszero; nominalparams not equal effectivecapacity. Contextual tokens mean already contains temporal information. No attention/query/key/timeposition/geofeedback/rawquery bypass.

Originalfit25912cal2879/loss/protection.1/constrainedcal/order/LR/K20 retained. GT/groups/logits absent inference forward. Locked original native20 ordering; no arbitrarypermutation claim. GPUpreflight and botharm actual2epoch continuous/interrupted optimizer/RNG/parameters/buffers/history/calibration EXACT passed; boolchecker legacy formattingwarning verifiedexact. PriorTOKEN_READOUT checkpoints source-locked only, not initialization.

Run/resume in PowerShell: & 'C:/Users/linke/Documents/New project/GENERATION_TOKEN_CONTRAST_SERIES/start_independent.ps1'
Never start duplicate worker. OS task holds exclusivebyte lock, appendlogs and workerstate. Completedhash-verified seeds skip, interruptedseed optimizer/RNGresume. Computer must remain on/loggedin/awake; automatedanalysis requires app.

Completion must verify18modelSHA,18explicitIDsfiniteK20,independentmetricsrecompute,originaldecoder/scorer frozen,report andSUCCESS. Keepall5%changeADE/FDE/protected1%/3seed/CI/samepath/coverage/duplicate/hardmotion AND allgroupmotiondistribution1% gates. RepeatedETHdev exploratory, not independentgeneralization. Afterthird newcompletion mandatorycycle4totalreview. No test/fold/cloud/externalmessages/unfreeze or criteria loosening.
