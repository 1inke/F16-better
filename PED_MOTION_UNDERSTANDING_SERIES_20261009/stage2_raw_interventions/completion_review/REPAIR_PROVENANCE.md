# Saved audit path repair

First audit stopped before data arithmetic because it assumed results/exit_status.json. The actual locked check_once.py/worker workflow writes root exit_status.json. Original audit_saved.py and FAILURE.json preserved. audit_saved_v2.py changes only this read path and its own failure output name. No source lock, metric definition, tolerance, prediction, model or production artifact changed.
