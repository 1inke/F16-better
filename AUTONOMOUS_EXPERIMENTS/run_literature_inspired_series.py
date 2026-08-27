import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
EXPERIMENTS = [
    "EXP_029_NODE_STRUCTURE_ENCODING",
    "EXP_030_AGENT_PROXY_ATTENTION",
    "EXP_031_TEMPORAL_CROSS_ENTMAX15",
    "EXP_032_DYNAMIC_TEMPORAL_CROSS",
]

for name in EXPERIMENTS:
    print(f"[run] {name}", flush=True)
    result = subprocess.run([str(PYTHON), str(ROOT / name / "run_screen.py")])
    if result.returncode:
        raise RuntimeError(f"Failed: {name}")
print("[series done]", flush=True)
