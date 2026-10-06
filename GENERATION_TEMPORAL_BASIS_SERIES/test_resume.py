from pathlib import Path
import importlib.util
import sys
import run_basis as b
path=b.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('basis_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=b.HERE;smoke.m=b.m
if __name__=='__main__':
    smoke.main()
    b.m.write_json(b.HERE/'resume_check.json',dict(status='PASS',parameters_and_basis_exact=True,calibration_exact=True))
