# Redundancy protected proposal coverage

REDUNDANCY_COVERAGE_GATE_FAIL

ETH reused dev; K20; up to4 original slots replaced only with protected same-family witness within.1m ADE-distance/.2m endpoint; no logits/GT allocation; explicit history proposals plus learned correction, not end-to-end intent understanding; no family auxiliary/test/backbone/scorer training

| Arm | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| control | constrained | overall | 0.217429 | 0.399570 |
| control | constrained | change | 0.366540 | 0.692019 |
| control | legacy | overall | 0.216541 | 0.399641 |
| control | legacy | change | 0.367847 | 0.699003 |
| control | final | overall | 0.216680 | 0.402176 |
| control | final | change | 0.375132 | 0.722238 |
| coverage | constrained | overall | 0.216804 | 0.399565 |
| coverage | constrained | change | 0.368198 | 0.700532 |
| coverage | legacy | overall | 0.216749 | 0.399879 |
| coverage | legacy | change | 0.367632 | 0.702986 |
| coverage | final | overall | 0.216910 | 0.401497 |
| coverage | final | change | 0.373687 | 0.718932 |
| constructed | untrained | overall | 0.219153 | 0.399660 |
| constructed | untrained | change | 0.372186 | 0.696350 |