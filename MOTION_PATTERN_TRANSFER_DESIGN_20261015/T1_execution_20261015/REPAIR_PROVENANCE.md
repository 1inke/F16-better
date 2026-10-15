# Pre-production isolated derivative-fixture repair

The first numeric preflight stopped at the fixed double-parameter FD check:
dynamic winner / head.4.weight / eps1e-4 / relative error1.19623991888e-8,
above the unchanged1e-8 threshold. It did not launch any production training.
All code and generated smoke evidence were preserved under failures/preflight1.
The original FAILURE.json is retained, not rewritten as PASS.

The first isolated synthetic GT was near a residual-norm origin, where the
norm's curvature makes a finite-difference truncation error material. The
corrected fixed fixture has negative-y GT at least20m from the nonnegative-y
synthetic centers. Both eps1e-4/1e-5, threshold1e-8, active-face checks and all
real-FIT tests are unchanged. No eps, neural seed or scientific prototype/loss
search is performed. The synthetic center construction remains isolated; the
scientific centers/labels/data/model/assignment/loss/statistics are unchanged.

This is an infrastructure numerical fixture correction before production
source-lock, not a scientific training outcome or evidence of model benefit.
The complete numeric/resume/resource preflight must rerun and pass. ARS DA1
requires retaining this limitation: double smooth-fixture FD does not prove
derivatives of quantized float32 arithmetic or every actual loss face.
