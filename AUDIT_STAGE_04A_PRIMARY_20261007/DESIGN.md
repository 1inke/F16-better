# Stage 04A-P: Proposal-Conditioned Correction Head Primary Experiment

## 1. Scientific Objective & Preregistration Context
Following the formal authorization in `ana.md` and the passing of `K24_TRANSFER_GATE`:
- The candidate generation baseline is established as **Static K24** (Native20 + Top-4 observation-driven explicit proposals).
- On the Zara scene-family outer evaluation set ($N=1,561$), Static K24 rescues 15 / 18 windows (83.3% of the theoretical K28 ceiling) with strictly $Lost \equiv 0$ when trained with zero Zara exposure.
- **The Core Scientific Question for Stage 04A**: Does providing explicit maneuver proposal provenance and conditioning to the learned residual correction head improve shared conditional mapping on held-out scene-family distributions, without over-editing or destroying valid static candidates?

---

## 2. Experimental Arms & Control Relationships

To prevent confounding, Stage 04A implements three reference benchmarks evaluated on all partitions:

| Arm | Description | Features Given to Head | Purpose |
| :--- | :--- | :--- | :--- |
| **Static K24** | Unmodified K24 candidates ($\Delta \tau \equiv 0$) | None | Verifies whether learned correction outperforms doing nothing. |
| **Control Head** | Correction head ($K=24$, seeds 1, 2, 3) | `observation` + `state32` + **Zeroed proposal types** | Baseline learned correction without maneuver awareness. |
| **Treatment Head**| Correction head ($K=24$, seeds 1, 2, 3) | `observation` + `state32` + **Real proposal types** | Evaluates the causal effect of maneuver-type conditioning. |

### Strict Invariants Between Control and Treatment:
- **Same Architecture**: `ProposalConditionedCorrectionHead` (obs_dim=193, hidden=128, state_dim=32, type_dim=16, num_types=9, steps=12).
- **Same Parameters**: Both instantiate the exact same embedding, FiLM, and trunk layers.
- **Zero Output Initialization**: Initialized with $\Delta \tau \equiv 0$ identically.
- **Same Training Loss**:
  $$\mathcal{L} = \min_{k=1..24} (ADE_k + 0.5 \cdot FDE_k) + 0.01 \|\Delta \tau\|^2 + 0.01 \|\Delta \dot{\tau}\|^2 + \mathcal{L}_{accel}$$
- **Same Optimizer & Budget**: AdamW, lr=$3 \times 10^{-4}$, weight_decay=$10^{-4}$, batch_size=128, epochs=50.
- **Same 3 Seeds**: Seed 1, Seed 2, Seed 3.
- **Zero Frozen Model Changes**: F16 backbone frozen, decoder frozen, scorer frozen.

---

## 3. Training Set Posterior Leakage Prevention (5-Fold Cross-Fitting)
As mandated by `ana.md` §8:
- If training samples were assigned proposals using a classifier trained on their own future trajectories, the head would receive an artificially clean signal during training that does not exist at test time.
- **Mitigation**: The 16,337 samples in `grouped_train_index` are partitioned into 5 stratified folds.
- For each fold $f \in \{1..5\}$: an `IntentClassifier` is trained strictly on the other 4 folds.
- Predictions and Top-4 proposals for fold $f$ are generated purely out-of-fold.
- Inner calibration ($N=1,773$), Zara outer ($N=1,561$), and CAL ($N=2,879$) receive proposals from the final classifier trained on all 16,337 samples of `grouped_train_index` (strictly zero outer exposure).

---

## 4. Preregistered Decision Thresholds (Automated Decision Engine)
1. **04A_CONFIRM_TRANSFER (Advance to Confirmation)**:
   - $\text{Treatment} - \text{Control}$ outer coverage mean $> 0\text{ pp}$.
   - $\ge 2 / 3$ seeds show positive gain.
   - $\text{Treatment} \ge \text{Static K24}$ (does not underperform the unedited candidate pool).
   - Original CAL degradation $\le 1.0\text{ pp}$.
   - Motion guard passed (acceleration and jerk p95 safe).
2. **04A_ALPHA_CALIBRATION (Residual Scale Calibration)**:
   - $\text{Treatment} > \text{Control}$ and $\ge 2 / 3$ seeds positive, BUT $\text{Treatment} < \text{Static K24}$ or $lost > 0$ or motion violation.
   - Triggers post-hoc grid search $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$ chosen on inner calibration.
3. **04B1_COMMON_SUPPORT (Support Shift / Normalization Intervention)**:
   - Treatment shows gains on FIT/inner but fails to transfer to Zara outer ($\le \text{Control}$), or no signal.
