# Cycle3 experiment3 proposal: untied native-slot output heads

Status DESIGN ONLY — not implemented, preflighted or launched. Not a new completed experiment.

## Evidence and hypothesis

Direct rawquery bypass and native-anchor feedback both failed. FIT/CAL-only shared-head diagnosis shows some stable/change winner gradients oppose (fit2/3 negative; cal2/3 negative), but single fixed batch/group/checkpoint is descriptive, not proven causality or persistent conflict. Different groups use overlapping winner slots. Hypothesis: a shared 64->6 output map forces incompatible correction mappings across native modes. Test untying only this map; may fail because upstream representation remains shared or mode slots overlap groups. No GT-group-specific branches, family expert loss, new loss or guard relaxation.

## Proposed control/treatment

Both use original frozen context64+native-modeMLP64 inputs. No geometry feedback, rawquery bypass or new history/basis.
Fresh control new128->128->64->6 shared correction head:25158 params.
Treatment same new128->128->64 front, 20 independent64->6 linear output maps indexed by fixed original native mode ID:32568 params. Shared front weights initialized identical across paired seed; all last layers zero => exact native K20 output. One factor is tying versus untying output weights, but it necessarily adds7410 params(+29.46%): improvement would combine mode specialization/capacity, not equal-capacity causal proof. Do not call slots learned intent/future groups. Original ID stable and data-source locked; if candidate permutation supported, carry IDs jointly, never reassign by GT/logit.

Nativebase+3anchor correction+same world-coordinate velocityclip1.5*(observedspeed+.03). Native data already focal-heading normalized; adding heading rotation again is not a justified factor. Still constrained native refinement, not absolute independent generation. All old decoder/F16/scorer exact frozen. Slots preserve original K20; no deletion or logits input. Shared hidden gradients remain coupled, so this does not guarantee elimination of interference.

Six NEW80epoch trainings shared/untied×seed1/2/3. Use original loss/balanced joint winner/regularizers/protection.1, fit25912/cal2879, constrainedcal/guards, LR/order/budget. Do not reuse prior control as independent fresh replication. Finite K20/preflight checks, exact shared-front init/nativeidentity, per-slot gradient reachability using synthetic inputs AND realFIT objective, all frozen tensors/source/cache/split hashes. Fix original ID handling; demonstrate ID-preserving permutation or explicitly restrict to locked native order. Real2epoch both-arm continuous vsinterrupted EXACT parameters/buffers/optimizer/RNG/history/calibration before launch. Nativeepoch0 fallback explicit/not learned.

All5% changeADE/FDE,1%protectedgroups,3seedconsistent,native+freshcontrol CI,samepath/coverage/duplicates/hardmotion/allgroupmotiondistribution1% gates unchanged. Dev only post-lock, not design-parameter tuning. Firstmajor scenario/samplecorrelation and3seed adequacy review; repeatedETHdev exploratory. No heldtest/crossfold/unfreeze/cloud/externaldata/messages/deletion.

## Execution handoff

Implementation must be self-contained shim around DirectAnchor (not loading failedridge/teacher, not accidentally retaining arm rawquery input). Check inheritance module HERE rebinding and frozen audits. Protocol source must state capacitydifference and semantic arm names. Windows independent on-demand unique task/worker/exclusivebyte lock, no periodic OS trigger. Reuse proven start_independent.ps1 pattern; never start before preflight+resume PASS. Basedon45min current6trainings versus32min prior, provisional estimate30–50min planning40; refine from implementation smoke and actual workload before launch. One60%-time check, actualremaining train/calIO/evaluation revise. Persist actualPID/start/deadlines/ledger/memory/automation once launched. Cycle3count2/3 until this new third normalcomplete audit; then fullcycle3review before any fourth direction.
