# EXP-183A-R2-V Spatial DINO Mode-Conditioned Scene Audit

This experiment freezes the accepted EXP-175S0 seed-1 `F16 + H3 + Joint-V2`
trajectory model. It extracts the 20 source H3 mode anchors and uses each
mode's t4/t8/t12 locations to query a frozen DINOv2-small 16x16 patch grid.

It compares raw spatial DINO, crop-centered DINO, and video-prototype-centered
DINO against an equal-input-capacity model-only quality probe. PCA is fit on
training videos only. The target is centered candidate quality
`ADE + 0.25 FDE`; test-video macro Spearman, Top-k winner recall, and regret
determine the stop/pass decision. Tiny MLP probes run only if the fixed ridge
stage first shows a stable cross-video positive signal.

No trajectory generator, H3, Joint-V2, candidate set, or native score parameter
is changed by this mechanism audit.
