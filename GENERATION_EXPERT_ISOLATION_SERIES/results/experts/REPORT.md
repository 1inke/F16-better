# experts architecture experiment

ARCHITECTURE_GATE_FAIL

ETH reused dev; K20; same balanced/protected global winner loss, NO family auxiliary; original shared128 controls reused; shared526 vs seven128 experts capacity matched; no test/backbone/scorer training

| Arm | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| control | constrained | overall | 0.217429 | 0.399570 |
| control | constrained | change | 0.366540 | 0.692019 |
| control | legacy | overall | 0.216541 | 0.399641 |
| control | legacy | change | 0.367847 | 0.699003 |
| control | final | overall | 0.216680 | 0.402176 |
| control | final | change | 0.375132 | 0.722238 |
| experts | constrained | overall | 0.218818 | 0.400522 |
| experts | constrained | change | 0.371612 | 0.699962 |
| experts | legacy | overall | 0.218169 | 0.401928 |
| experts | legacy | change | 0.371841 | 0.714207 |
| experts | final | overall | 0.221543 | 0.411550 |
| experts | final | change | 0.388258 | 0.751373 |