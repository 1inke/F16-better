# Latent Hook Map (F16 JointTrainableV2Decoder)

> **Source Target Model**: `FrontEncoderASTRA(f16_temporal_social_cross)` with `JointTrainableV2Decoder`  
> **Source Definitions**:
> - `FRONT_ENCODER_SERIES/front_encoder_models.py`
> - `E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py`
> - `CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py`
> - `HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py`

---

## 1. Internal Forward Graph & Information Flow

```text
past trajectory & graph [B, N, 8, 2]
    │
    ▼ (Temporal + Social Cross Encoder)
encoded_tokens [B, N, 8, 64]
    │
    ├─► recent = encoded_tokens[:, :, -1]              [B, N, 64]
    ├─► pooled = encoded_tokens.mean(dim=2)            [B, N, 64]
    └─► motion_features(past_loc)                      [B, N, 7]
    │
    ▼ torch.cat(recent, pooled, motion_features)
context_input [B, N, 135]
    │
    ▼ model.decoder.context (Linear + GELU + Linear)
context [B, N, 64]  ───────────────────────────────────────────┐ (Used for proposal scale)
    │                                                          │
    ▼ + mode_embedding [20, 64]                                │
context_mode_sum [B, N, 20, 64]                                │
    │                                                          │
    ▼ model.decoder.mode_mlp (LayerNorm + Linear + GELU)        │
L3: mode query hidden [B, N, 20, 64]                           │
    │                                                          │
    ├─► proposal_delta(mode) ──► delta [B, N, 20, 3, 2]         │
    │                                                          │
    └─────────────────┐                                        │
                      ▼                                        ▼
    source_anchors = scale * (templates + delta)               │
    source_anchors.flatten(-2) [B, N, 20, 6]                   │
                      │                                        │
                      ▼ torch.cat(mode, source_anchors.flatten(-2))
    L5: score_head input / pre-logit [B, N, 20, 70]
                      │
                      ▼ model.decoder.score_head (LayerNorm + Linear + GELU + Linear)
    native_logits [B, N, 20]
                      │
    ┌─────────────────┴────────────────────────────────────────┐
    │
    ▼ refiner(mode, source_anchors) (LayerNorm + Linear + GELU + Linear + GELU + Linear)
    BoundedAnchorCorrection features = torch.cat(mode, source_anchors.flatten(-2))
    L4: mode-specific hidden after non-linear refinement [B, N, 20, 128]
    (network[1] / network[4] intermediate activations)
    │
    ▼
    corrected_relative = interpolation @ corrected_anchors
    trajectories [B, N, 20, 12, 2]
```

---

## 2. Hooked Tensor Specifications

| Tensor Name | Source Location | Shape (Focal) | Pre/Post Operation | Sample Conditioned? | Mode Conditioned? | Used by Trajectory Head? | Used by Confidence Head? |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **`L3` (Mode Latent)** | `model.decoder.mode_mlp` output | `(20, 64)` | Post `mode_mlp` | Yes ($R=0.04$, Cos $\approx 0.52$) | Yes | Yes (proposal delta & refiner) | Yes (score_head input) |
| **`L4` (Refiner Latent)** | `model.decoder.refiner.network[3]` output | `(20, 128)` | Post 2nd GELU before final linear | Yes | Yes | Yes (directly drives final trajectory offsets) | Pre-logit upstream |
| **`L5` (Pre-Logit Latent)** | Pre `model.decoder.score_head` input | `(20, 70)` | `torch.cat((mode, source_anchors), -1)` | Yes | Yes | Yes (composed of mode + anchor) | Directly fed to `score_head` |
| **`native_logits`** | `model.decoder.score_head` output | `(20,)` | Post `score_head` | Yes | Yes | No | Direct selector logits |

---

## 3. Hook Verification Protocol
- **Zero numerical interference**: All forward hooks only clone and detach CPU tensors.
- Verification test asserts `max |pred_hook_on - pred_hook_off| < 1e-7`.
