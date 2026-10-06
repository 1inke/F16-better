from pathlib import Path
import importlib.util
import sys
import run_family as family
path=family.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('family_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=Path(__file__).resolve().parent;smoke.m=family.m
if __name__=='__main__':smoke.main()
