# F16 response-head feedback / tiny-FIT diagnosis

Human-approved 2026-10-05. See DESIGN.md for fixed protocol and interpretation.
New folder only; preserve all previous science/models/results. GPU fresh small
head learning plus no-optimizer old checkpoint gradient observations. No full
production candidate trial, no DEV/test, formal response count remains2/3.

`diagnosis.py --preflight` validates real gradients/identity/frozen/split/alignment,
memory, both-arm real2epoch exact resume and locks. `start_independent.ps1` launches
one independent OS worker after PASS. `check_once.ps1` is used once halfway, then
only at the revised completion deadline. Runtime/RESULT/audits hold real evidence.
Approximately17GB available hostRAM; process6GiB and each system reserve6GiB.
