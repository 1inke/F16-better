import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "EXP_022_DYNAMIC_SPARSE_ROUTING",
    "EXP_023_GROUPED_NEAR_RISK_ATTENTION",
    "EXP_024_NODE_EDGE_NODE_ITERATION",
    "EXP_025_TEMPORAL_SOCIAL_CROSS_ATTENTION",
    "EXP_026_SPARSEMAX_ATTENTION",
    "EXP_027_RELATION_MODULATED_QK",
]
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")

for name in EXPERIMENTS:
    script = ROOT / name / "run_screen.py"
    print(f"[run] {name}", flush=True)
    result = subprocess.run([str(PYTHON), str(script)])
    if result.returncode:
        raise RuntimeError(f"Failed: {name}")
print("[series done]", flush=True)
