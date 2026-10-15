# K24 Transfer Gate: Candidate Expansion Generalization & Exposure Audit

## 1. Executive Summary & Scientific Purpose

Following the formal research directive:
- **K20 pruning route is terminated** due to single-candidate fragility (511 / 733 DEV successful windows depend on exactly 1 good candidate, with 100% of pruned lost windows depending on 1 candidate).
- **K24 additive proposal route** (Native20 + 4 observation-driven explicit maneuver proposals) is promoted to **Provisional Generation Protocol (Stage-04 candidate baseline)**.
- **Stage 04 formal GPU training is conditionally gated** on `K24_TRANSFER_GATE`.
- **Scorer remains strictly frozen**.

This audit implements the preregistered **`K24_TRANSFER_GATE`** to evaluate whether the +46 rescue gain (+3.03 pp coverage gain) observed on formal DEV transfers robustly to held-out isolation splits, specifically auditing against **training set exposure leakage** in proposal selection.

---

## 2. Experimental Protocols & Control Relationships

### 2.1 Evaluated Isolation Sets
1. **Zara Scene-Family Outer ($N=1,561$)**:
   - Source: `GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family`
   - Scientific property: Strictly unseen scene family (Zara 1 / Zara 2). Evaluates out-of-scene-family transfer.
2. **Target-Track Outer ($N=1,035$)**:
   - Source: `GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment1_target_track`
   - Scientific property: Strictly unseen pedestrian IDs. Evaluates out-of-agent/trajectory transfer.
3. **Recording-Time Outer ($N=566$)**:
   - Source: `GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time`
   - Scientific property: Strictly unseen continuous time blocks. Evaluates temporal distribution shift.
4. **Original CAL ($N=2,879$)**:
   - Source: `F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005` (samples 25,912 to 28,791 of `eth_train.npz`)
   - Scientific property: Standard held-out calibration split used across earlier phases.

### 2.2 Candidate Protocols Evaluated
- **Native20 ($K=20$) [Baseline Control]**:
  - The frozen F16 model's native 20 candidate trajectories.
- **Duplicate8 ($K=28$) [Null Control 1]**:
  - Native20 + top-8 native slots duplicated (exact same trajectories).
  - Tests whether candidate budget expansion alone induces spurious coverage.
- **Jitter8 ($K=28$) [Null Control 2]**:
  - Native20 + top-8 native slots with Gaussian spatial jitter ($\sigma \in \{1\text{cm}, 2\text{cm}, 5\text{cm}\}$).
  - Tests whether minor stochastic local perturbations explain maneuver gains.
- **Augmented28 ($K=28$) [Upper Bound Protocol]**:
  - Native20 + all 8 deterministic observation-driven maneuver proposals (stop, decel, accel, mild L/R, sharp L/R, reverse).
- **K24-global ($K=24$) [Candidate Protocol, Global Selector]**:
  - Native20 + Top-4 explicit proposals selected using the frozen global `IntentClassifier` (trained on `fit = 0..25912` of `eth_train.npz`).
  - *Exposure note*: The global training set `fit` contains Zara samples (7,802 samples). Thus, for Zara outer, `K24-global` has prior scene exposure during selector training.
- **K24-isolated ($K=24$) [Candidate Protocol, Strictly Isolated Selector]**:
  - Native20 + Top-4 explicit proposals selected using an `IntentClassifier` trained strictly on each experiment's `grouped_train_index` with zero outer data exposure.
  - *For Zara outer*: The selector was trained on 16,337 samples containing strictly **0 Zara samples** (15,578 university, 759 hotel).
  - *For Target-track outer*: The selector was trained on 18,909 samples containing strictly **0 target-track outer pedestrian samples**.
  - *For Recording-time outer*: The selector was trained on 16,104 samples containing strictly **0 recording-time outer samples**.
  - *For CAL*: The global selector was trained on `fit = 0..25912`, which already held out all 2,879 CAL samples.

---

## 3. Core Hypotheses & Falsification Criteria

### Hypotheses
1. **Superset Invariance**:
   - Because $K=24$ strictly includes all 20 native slots, $lost = 0$ must hold identically across all datasets.
2. **Transferability of Explicit Proposals**:
   - In all 4 isolation sets, Augmented28 and K24 achieve substantial positive rescue ($\Delta\text{Coverage} > 0$).
3. **Selector Exposure Invariance**:
   - `K24-isolated` preserves at least $75\%$ of the rescue achieved by `K24-global` on Zara outer, proving that maneuver selection is driven by physical history kinematics rather than memorized scene-specific trajectories.

### Falsification Criteria
- If $lost > 0$ on any dataset $\rightarrow$ Implementation / protocol violation.
- If Duplicate8 produces $rescue > 0$ $\rightarrow$ Bug in metric evaluation.
- If `K24-isolated` rescue on Zara outer collapses ($\le 50\%$ of K28 rescue) $\rightarrow$ Maneuver selection depends critically on scene familiarity, invalidating zero-shot scene transfer for Stage 04.

---

## 4. Evaluation Metrics
In accordance with user rules:
1. **Complete Good Candidate Coverage**: $ADE \le 0.2\text{ m} \land FDE \le 0.4\text{ m}$ on the same candidate.
2. **Rescue & Lost**:
   - $Rescue = \sum \mathbf{1}(\text{cov}_K \land \neg\text{cov}_{20})$
   - $Lost = \sum \mathbf{1}(\neg\text{cov}_K \land \text{cov}_{20})$ (strictly required: $Lost \equiv 0$)
3. **Capture Rate**: $Rescue(K24) / Rescue(K28) \times 100\%$
4. **Continuous Joint Index**: $J = \min_k \max(ADE_k / 0.2, FDE_k / 0.4)$
5. **minADE / minFDE**: Standard oracle trajectory errors.
6. **Temporal Path Errors**: Early (steps 1–4), Middle (steps 5–8), Late (steps 9–12).
7. **Kinematic Metrics**: Speed p95, Acceleration mean/p95, Jerk mean/p95.
8. **GT-Free Effective Good Clusters**: Complete linkage clustering at $d_{thresh} = 1.0$.
9. **Subgroup Disaggregation**: Evaluated across `stable`, `transition`, and `change` subsets.
