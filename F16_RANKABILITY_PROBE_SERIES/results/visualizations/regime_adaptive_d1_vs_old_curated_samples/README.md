# Old/New curated visualization comparison

- Top panel in every composite: unchanged legacy curated PNG.
- Bottom panel: fold-local ETH V2 candidates scored by the D1 social-gated reranker, best checkpoint, head seed 1.
- The matched sample key is `sample_id`; cache index equality was also verified.
- The formal cache candidates were independently checked against the legacy cache ground truth: recomputed ADE/FDE matched the stored formal metrics within 2e-5 m.
- These samples belong to the ETH development split used for checkpoint selection, not held-out test.

Rendered 25 image pairs (20 unique samples). D1 changed the current fold-local native top-1 in 4 entries; ADE improved/worsened/tied in 3/1/21 entries.
