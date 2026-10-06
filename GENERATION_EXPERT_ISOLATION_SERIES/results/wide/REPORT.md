# wide architecture experiment

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
| wide | constrained | overall | 0.217444 | 0.399433 |
| wide | constrained | change | 0.367797 | 0.692592 |
| wide | legacy | overall | 0.217095 | 0.400226 |
| wide | legacy | change | 0.368735 | 0.703036 |
| wide | final | overall | 0.218128 | 0.404153 |
| wide | final | change | 0.377024 | 0.724028 |