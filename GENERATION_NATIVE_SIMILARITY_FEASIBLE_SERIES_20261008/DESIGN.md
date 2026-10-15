# Native global-similarity feasible refinement — preregistration
Latest human: 开始下一个方向. ONE new structural contrast, not a reset of 03:30–09:30 six-hour window, not a fourth projection-coefficient sweep.

## Question and evidence
Prior native-anchor fullthree is reviewed, allmajorFAIL (prior approved child RECENT_EXPERIMENT_SYNTHESIS and RESULT_ANALYSIS). Ray all five pooled gates passed but residual maximum .04184mm, alpha median0 and no coverage change. Does a different feasible output geometry retain nonzero movement and improve same-candidate coverage without sacrificing native motion support?

## Fixed protocol
Target-track groupedtrain18909, outer1035 primary, inner1856 and originalCAL2879 posthoc ONLY, seeds42/43/44, fixed25epochs. Six fresh trainings,150rows,18 scientific saved jobs. Whole F16/decoder/scorer never instantiated/trained/optimizer accessed; immutable native cache SHA protected. Same observedP2 mean8tokens96+selected immutable native path24 input120; shared MLP120→128→64→24 GELU25304 finalzero; paired all weights/buffers. GT/groups/IDs/scenes/logits not prediction inputs; logits only fixedbottom4 stable-originaltie slotmap; other16 native EXACT, finalK20.
Common per-training-only feature mean/std, float32 features/float64 accumulation std<1e-6→1. Same degree11 Bernstein bounded controls and envelope .20(t/12)^2, headfloat32/outputfloat64.
Canonical hardmin4 ADE+.5FDE and sixpair mean squaredhinge margin.3 weight.05; AdamW.001 wd.0001 batch256shuffle25 noclip. No loss/assignment/scorer/event/unfreeze/data change.

## Sole factor: feasible output geometry
Ray control uses prior production closed quadratic-root intersection31 v/a/j constraints, native-relative1.01 and numeric interior1−2^-20, bitwise same given nonzero parameters.
Similarity treatment fits proposal to a global transform of native positions, not to GT:
E=sum_t ||p_t||²; a=1+sum p·r/E; b=sum cross(p,r)/E; Jp=(-p_y,p_x). Exact E=0 uses identity; no magnitude floor.
If s=hypot(a,b)>1.01 radial scale cap to1.01(1−2^-20). Fit residual rfit=(aI+bJ)p−p. A single lambda enforces every original position bound: min(1,min_t bound_t/||rfit_t||), zero residual unconstrained; numeric interior only when lambda<1. Final p'=p+lambda*rfit.
Final coefficients (1+lambda(a−1),lambda*b) lie on segment between identity and capped (a,b), within radius1.01 by convexity. With origin p0=0, each linear discrete derivative transforms by same similarity; every norm is scaled by common s_final≤1.01 in exact arithmetic. Thus fixed16 plus selected4 imply all five pooled metrics no worse than1.01 in exact arithmetic. Independently evaluate original literal pooled gates; do not replace by proof or roundoff certificate.
Both same nominal parameters, different effective dimensionality/reachable geometry/optimization. NOT pure shared-head causal test, not new stop/turn temporal shape generation. Exact stationary anchor stays stationary; fullcurve rotates/scales, cannot invent temporal event. Can still lose successes, despite native-relative motion safety.

## Numeric and scientific preflight
Resource guard first process6GiB/physical and commit eachreserve6GiB, longstanding ~17GB available. Preserve failures/no OS/pagefile/userkill.
Independent NumPy versus actual torch geometry, identity/zeroenergy/rotation/scale cap/position cap/knownactive/nearzero derivative roundoff fixtures. Known constant-velocity rotation must retain >1mm correction; actual fixedseed20261008 nonzero lastlayer first128FIT max>=1mm and >=5% candidates residualmax>=1mm, no GT selection. Failure STOP, not search bounds/seeds.
Actual GPU8FIT256 both phase-positive finite gradients, allpairedweight/boolscaler/basis EXACT, nativezeroidentity/bounds/cache IDs, prior ray nonzero output and canonical AST scalar/allgrad EXACT. Pure NumPy objective. Double isolated parameter finite differences decreasing eps1e-4..1e-7, retain all, require lasttwo unchangedwinner/localface and original1e-7+1e-3|derivative| errors; nonsmooth large-step discrepancy not float32 quantization proof.
Both real513FIT+12inner SUBSET2epoch uninterrupted/interrupted allparams/boolscalers/basis/optimizer/TorchCPU-CUDA-loader-Python-NumPyRNG/historyCAL EXACT, source/output bad reject and completedSHA skip. Not fullFIT2epochs, not analysisresume.
Actual savedouter128 full same-path metrics/ALLK originalSciPy GTfree complete-link .10meanpath/.20endpoint/pooledmotion independent max toleranceoriginal1e-12, allnewhead frozenboolbuffers EXACT, similarity coefficient reconstruction and pointwise derivative certificate. Certificate gamma_n absolute binomial scale handles arithmetic error only; no floor/epsilon in original five pooled 1% gates. Synthetic substantive1e-7 defect againstzero derivative rejected.
Actual36throwaway4warm32measure perarm FIT256+innerfull3port metrics/RAM/disk before ETA, no savedbenchmarkmodel or outcome selection. Current source and inherited1193 related manifest digests preserved/extended, not all historical34k or oldF16GPU replay.

## Analysis and branch decisions
Coverage SAME completecandidate ADE<=.2 AND FDE<=.4/J<=1 first; GTfree effectivegood second notmaximize/copies; scorerlast. Report eachseed/allgroups, all sameID rescue/lost/retained and nonzero-residual support; ownzero native20. Fixed .3/.5 sensitivity no standard switching or difficultsample removal.
Primary outerfixed25, inner/CAL secondary. Paired targettrack+modelseed2000rng20261008 exploratory36records144scalarCI; no cross-study differentID pairing.
All previous major gates unchanged: changeADE and FDEeach>=5%, overallstabletransition<=1%harm,3seedconsistent,nativefreshADE/FDE/JjointCI excludes0,samepathcoverage+duplicatehardphysics AND everyseed/group fivepooledmotions<=native1.01 dt.4. First fullPASS additionalcorrelation/3seedadequacy review. Runtime executionSUCCESS not majorPASS/baselineupgrade.
If nondegenerate and safe but no coverage, record limited global geometry, not universal generation impossibility. If benefit inconsistent/CALpoor, do not conclude shared mapping root; current FIT coverage not saved so cannotinvent. No automatic next loss/scorer/oldmodule/data or rotationbound/width/LRepoch sweep.

## Completion and workflow
Only all actual gates and source reviewPASS allow unique0trigger OSbyte-lock task/one startup realPIDparentcommandstage/newstderr. Only then report measuredETA and schedule one half elapsed-time check -> one actualcompletion node, no conversational polling, no oldPIDqueries.
Worker+exit doubleSUCCESS, six25histories,checkpoint/source-ID-statistics-objective-ALLoptimizerRNG bindings,18 finite uniqueID K20outputs/fullsavedmetrics-clusters-motion/controlsgeometry-fixed16/36CI144scalars/source-relatedSHA/RESOURCE/ARITHMETIC_PRESERVATION required.
CPU completion_review all18 saveddata fullSciPy clusters/motion/geometry/manifests150history/SHA no model/gradient secondreplay. Inline ARS DA2/3 by same assistant NOT externalreview. Save RESULT_ANALYSIS MDJSON/runtimeANALYZED. Newstudyfirst completed only after fullaudit; prioranchor3/3 stays reviewed, cycle11/baseline/unfinishedgoal unchanged.
Latest ONE experiment safetydeadline12:00HK, no automatic six-hour reset. Routine numeric diagnosis assistant-owned; genuine source/ID/resource/scientific/scope obstacleSTOP preserve evidence human. New artifacts ONLY thisfolder, legacy evidence immutable.

## Ethics and evidence status
Secondary existing de-identified trajectory caches, no new participant collection/contact/identity inference; no IRB assertion fabricated. Local exploratory computational preregistration, not clinical CONSORT. All rationale anchored to saved local experiment sources above, geometric proof derived here. AI-assisted code/design/review disclosed.

## Inline ARS DA1
PASS_WITH_LIMITATIONS before implementation. Major resolved: nominal25304≠effective output capacity, rotations/scales preserve shape not new event; explicitlimits and pairedsameinputs/loss. Strongest counterargument: conservative2D map may be safe and nonzero yet cannot repair missing turn/stop, and bottom4 still irreplacabletail. Rebuttal is this finite falsifiable test, no guaranteed coverage claim. Minor: source-related1193 notfullhistoricSHA; disclose. DA2/3 mandatory afterresults, not yet performed.
