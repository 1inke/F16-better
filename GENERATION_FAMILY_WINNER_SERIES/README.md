# Single family-winner supervision

Hypothesis: global winner loss can choose a geometrically close but wrong heuristic-maneuver candidate, starving the matching maneuver slot. Previous top3 family concentration failed; this pilot instead adds only one matching-family minimum ADE+0.5FDE, coefficient0.25 fixed before dev, weighted by original fit-only maneuver weights. If no matching original candidate, auxiliary loss is zero. Global winner loss remains. No top3 attraction or slot pruning.

Same large-bound1.5 architecture/settings/35epochs/protection0.1/calibration constraints as preceding capacity experiment; three fresh family trainings paired with verified reused control seeds1/2/3. Same-seed control predictions must reproduce. Both arms have20 original slots, frozen backbone/scorer. Candidate family defined from original geometry, GT family computed only in training loss; not an inference input or true intention recognition. All18 policy models locked before dev. Constrained is primary; legacy/final remain diagnostics. No test/fold expansion or independent-data claims.

Preflight covers source/config/control hashes, raw/cache provenance, fit-only scales, split, single-minimum gradient, missing-family zero gradient, GPU backward, candidate/loss permutation equivalence. Isolated enlarged-loss resume smoke must match continuous2epoch parameters/calibration exactly. Same fixed5%change/1%protected-group/three-seed pairedCI and coverage/kinematic gates.

Runtime estimate based on last actual run: 30–40 minutes, reserve45 minutes silence. Run start_experiment.ps1 once. Results include explicit-ID NPZs, per-seed/group metrics, paired crossed bootstrap, summary, report, model-lock and exit status.
