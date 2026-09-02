# Scene-Conditioned Intent Series

EXP-183A freezes the official EXP-175S0 seed-1 `F16 + H3 + Joint-V2` model and
tests whether scene context adds information beyond its post-context motion/social
representation. It trains only small intent probes, uses the official 31/12/17
video split, fits DINOv2 PCA on training videos only, and includes a cross-video
motion-matched scene-swap audit.

The primary comparison is `P3_rich` versus motion-only `P0`. This is a mechanism
audit and does not modify H3 anchors, V2 refinement, candidate scores, or the
official trajectory model.
