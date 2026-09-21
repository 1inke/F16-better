"""
Sync metrics for completed trained models (QA0, QA1_L005, QA1_L010, QA1_L020) with final_model.pth.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PYTHON = sys.executable

dirs = [
    HERE / "results" / "baseline_reproduction" / "QA0_STANDARD_FINETUNE",
    HERE / "results" / "pilot" / "QA1_L005_T010",
    HERE / "results" / "pilot" / "QA1_L010_T010",
    HERE / "results" / "pilot" / "QA1_L020_T010",
]

for d in dirs:
    final_pth = d / "final_model.pth"
    best_pth = d / "best_selected.pth"
    if final_pth.is_file():
        # Copy final_model.pth to best_selected.pth
        shutil.copyfile(final_pth, best_pth)
        # Evaluate
        eval_script = HERE / "evaluate_quality_aware_f16.py"
        import subprocess
        cmd = [
            PYTHON, "-u", str(eval_script),
            "--checkpoint", str(best_pth),
            "--fold", "eth", "--seed", "1",
            "--output-json", str(d / "metrics.json"),
        ]
        res = subprocess.run(cmd, cwd=str(REPO), check=True)
        print(f"Synced {d.name}")
