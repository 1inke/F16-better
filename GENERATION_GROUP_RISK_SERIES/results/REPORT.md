# Smooth worst-group relative risk

GROUP_RISK_GATE_FAIL

ETH reused dev; K20; shared128; fit-native-normalized smooth worst-group oracle loss, NO family auxiliary; GT groups loss-only; matched large-bound1.5 shared128 control; no test/backbone/scorer training

| Arm | Policy | Group | minADE | minFDE |
|---|---|---|---:|---:|
| control | constrained | overall | 0.217429 | 0.399570 |
| control | constrained | change | 0.366540 | 0.692019 |
| control | legacy | overall | 0.216541 | 0.399641 |
| control | legacy | change | 0.367847 | 0.699003 |
| control | final | overall | 0.216680 | 0.402176 |
| control | final | change | 0.375132 | 0.722238 |
| group_risk | constrained | overall | 0.218049 | 0.399676 |
| group_risk | constrained | change | 0.368860 | 0.693801 |
| group_risk | legacy | overall | 0.217042 | 0.400791 |
| group_risk | legacy | change | 0.371340 | 0.708509 |
| group_risk | final | overall | 0.216614 | 0.400538 |
| group_risk | final | change | 0.376255 | 0.723560 |