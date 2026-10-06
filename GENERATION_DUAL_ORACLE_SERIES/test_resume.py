from pathlib import Path
import importlib.util
import sys
import run_dual as d

path=d.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('dual_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=Path(__file__).resolve().parent;smoke.m=d.m
if __name__=='__main__':
    smoke.main()
    d.m.write_json(d.HERE/'resume_check.json',dict(status='PASS',continuous_interrupted_parameters_exact=True,calibration_exact=True))
