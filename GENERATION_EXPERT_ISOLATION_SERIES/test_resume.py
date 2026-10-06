from pathlib import Path
import importlib.util
import sys
import run_experts as e
path=e.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('expert_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.m=e.m
if __name__=='__main__':
    for variant in e.VARIANTS:
        e.current_variant=variant
        smoke.HERE=e.HERE/('smoke_'+variant)
        smoke.main()
