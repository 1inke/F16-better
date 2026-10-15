# DESIGN: Stage B Best-of-4 Generator 3-Seed Formal Validation (Zara Scene-Family Outer Split)

## 1. Scientific Context & Purpose
- **Context**: Stage A Pilot (Seed 42) demonstrated the first positive signal on Change in the project series:
  - Change Coverage: +0.47 pp (41.71% $\to$ 42.18%)
  - Change $\min\text{ADE}$: +1.67%, Change $\min\text{FDE}$: +4.13%
  - Net Change Rescue: +1 (5 rescues vs 4 lost)
  - Slot Collapse ruled out: Mean pairwise distance = 0.638 m, max dominance = 38.1%, effective slots = 3.93 / 4.0.
  - Overall Coverage: +1.47 pp (67.97% $\to$ 69.44%), Net rescue = +23.
- **Purpose**: Stage B performs preregistered 3-seed paired evaluation (Seeds 42, 43, 44) to test whether Seed 42's positive signal is robustly replicable across random initializations, without any hyperparameter tuning or architectural modifications.

## 2. Experimental Control & Treatment
- **Control**: Static-K24 (Native 20 frozen + 4 static proposals: Decelerate, Stop, Accelerate, Reverse / Mild Turns).
- **Treatment**: Bo4-K24 (Native 20 frozen + 4 learned proposals from trajectory-space $\min_m \mathcal{L}_{\text{traj}}$).
- **Candidate Budget**: Strictly $K=24$ (20 Native + 4 Bo4 slots).
- **Frozen Protocol**:
  - Representation: P2 (96-d F16 encoder tokens mean).
  - Architecture: BestOf4Generator (Linear(96, 128) $\to$ GELU $\to$ Linear(128, 128) $\to$ GELU $\to$ 4 heads $\times$ (Linear(128, 64) $\to$ GELU $\to$ Linear(64, 8))).
  - Initial Biases: Quadrant initialization (Slot 0: cruise/accel, Slot 1: left turn, Slot 2: right turn, Slot 3: decel/stop).
  - Training Budget: 25 epochs, AdamW(lr=1e-3, wd=1e-4), batch size 256.
  - Loss: $\min_m (\text{ADE} + 0.5 \cdot \text{FDE}) + 0.05 \cdot \text{div\_loss}$ ($\delta = 0.30\text{ m}$).
  - Split: Zara scene-family (Train $N=16,337$, Outer $N=1,561$, Change $N=211$).

## 3. Preregistered Decision Rules
1. **Quality Gate**:
   - Change Coverage gain $> 0$ in at least 2/3 seeds;
   - 3-seed mean Change Coverage gain $\ge +1.0$ pp;
   - Overall $\min\text{ADE}$ and $\min\text{FDE}$ harm $\le 1.0\%$;
   - Slot collapse ruled out (pairwise distance $\ge 0.1\text{m}$, dominance $< 75\%$).
2. **Motion Guard**: $|\Delta\text{speed}_{p95}| \le 2.0\%$, $|\Delta\text{accel}_{p95}| \le 2.0\%$, $|\Delta\text{jerk}_{p95}| \le 2.0\%$.
3. **Outcome Actions**:
   - Quality gate pass, but only acceleration guard slightly fails $\implies$ `BO4_POSITIVE_MOTION_REPAIR_REQUIRED` (halt and await single-factor kinematics repair design).
   - Overall/Stable improves but Change shows consistent negative gains $\implies$ `STOP_BO4_CHANGE_NOT_SOLVED`.
   - Only Seed 42 is positive $\implies$ `STOP_SEED_SPECIFIC`.
   - $\ge 2/3$ seeds Change positive, mean $\ge +1.0$ pp, and all major guards pass $\implies$ `BO4_STRONG_PASS`.
