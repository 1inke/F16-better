# Stage R3: Representation / Ambiguity Audit Design

## Objective
Directly test whether past-only history representations ($P2$) suffer from intrinsic future motion ambiguity, explaining why single deterministic Top-1 selection fails to convert newly generated candidate support without incurring severe overall false-alarm harm.

## Scientific Hypothesis
For samples in behavioral transition/change, pedestrians facing similar past trajectories can validly pursue multiple mutually exclusive future maneuvers (straight continuation, decelerating, stopping, or turning). Consequently:
$$P(\text{candidate is correct} \mid \text{past})$$
has high conditional entropy, rendering deterministic single Top-1 selection inherently regret-prone under past-only information.

## Methodology
1. **$k$-NN in Past Feature Space**:
   - Metric: Normalized Euclidean distance on 96-d $P2$ pooled history tokens (normalized strictly on training split).
   - Neighbors: $k \in \{10, 20\}$ from the training split for each outer evaluation sample.
2. **Ambiguity Metrics**:
   - Neighbor future motion-mode entropy: $H(\text{mode}) = -\sum p_m \log_2 p_m$.
   - Neighbor best-candidate provenance entropy: $H(\text{prov}) = -\sum p_{\text{prov}} \log_2 p_{\text{prov}}$.
   - Neighbor endpoint circular variance: $\text{Var}(\theta) = 1 - R$.
   - Neighbor speed-profile variance: $\text{Var}(v) = \frac{1}{k}\sum (v_j - \bar{v})^2$.
3. **Stratification**:
   - Stable vs Transition vs Change vs clear-event.
4. **Uncertainty & Multi-hypothesis Diagnostics**:
   - Failure correlation: correlation between past entropy and Top-1 error.
   - Confidence abstention: Top-1 precision when filtering top 20% ambiguous samples.
   - Multi-hypothesis Top-3 set coverage (Native Top-1 + Static Top-1 + Learned-3).
