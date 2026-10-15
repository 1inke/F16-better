"""Sole byte-locked orchestrator. Trainer subprocess PID reported separately."""
import json,msvcrt,os,subprocess,sys,time,traceback
from datetime import datetime
import psutil
from controls import H,atomic_json,digest,guard,verify_binding
from commands import command

def main():
    lock=(H/'worker.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    start=datetime.now().astimezone().isoformat();pid=os.getpid();parent=psutil.Process().ppid()
    state=dict(status='RUNNING',pid=pid,parent_pid=parent,start=start,stage='INITIAL_SOURCE_INPUT_SHA',pipelines_done=0,stage_fits_done=0)
    atomic_json(state,H/'worker_state.json');guard();verify_binding(H/'source_input_binding.json')
    gate=json.loads((H/'PREFLIGHT.json').read_text());assert gate['status']=='PASS_ALL_ACTUAL_PREFLIGHT_AND_FINAL_SOURCE' and gate['binding_sha']==digest(H/'source_input_binding.json')
    fits=0;pipelines=0
    for fold in ('eth','hotel','univ','zara01','zara02'):
        for seed in (1,2,3):
            for branch in ('full','last2','noNbr'):
                for stage in ('h3','v2'):
                    guard();cmd,env,dest=command(fold,seed,branch,stage);log=H/'logs'/f'{fold}_{seed}_{branch}_{stage}.log';log.parent.mkdir(exist_ok=True)
                    if log.exists():raise RuntimeError('STOP_EXISTING_TRAINING_LOG: no automatic restart')
                    with log.open('x',encoding='utf-8') as handle:
                        child=subprocess.Popen(cmd,cwd=H,env=env,stdout=handle,stderr=subprocess.STDOUT)
                        state.update(stage='TRAINER_LIFECYCLE',fold=fold,seed=seed,branch=branch,decoder=stage,trainer_pid=child.pid,stage_fits_done=fits,pipelines_done=pipelines)
                        atomic_json(state,H/'worker_state.json');code=child.wait()
                    if code:raise RuntimeError(f'STOP_TRAINER_EXIT {code}: {log}')
                    done=json.loads((dest/'done.json').read_text());assert done['status']=='SUCCESS' and not done['preflight_subset_only']
                    for path,sha in done['outputs'].items():assert digest(path)==sha
                    fits+=1
                pipelines+=1
    state.update(stage='INDEPENDENT_ALL_SAVED_ARITHMETIC',stage_fits_done=fits,pipelines_done=pipelines,trainer_pid=None);atomic_json(state,H/'worker_state.json')
    from audit_saved import audit
    audit();verify_binding(H/'source_input_binding.json')
    state.update(status='SUCCESS',stage='SUCCESS',completed_at=datetime.now().astimezone().isoformat());atomic_json(state,H/'worker_state.json')
    atomic_json(dict(status='SUCCESS',pid=pid,completed_at=state['completed_at']),H/'exit_status.json')
if __name__=='__main__':
    # New logs only; do not overwrite scientific/old failures.
    with (H/'worker_stdout.log').open('x',encoding='utf-8') as out,(H/'worker_stderr.log').open('x',encoding='utf-8') as err:
        sys.stdout=out;sys.stderr=err
        try:main()
        except Exception:
            traceback.print_exc();err.flush()
            failure=dict(status='FAILURE',time=datetime.now().astimezone().isoformat(),pid=os.getpid(),traceback=traceback.format_exc(),partial_outputs_preserved=True)
            atomic_json(failure,H/'FAILURE.json');atomic_json(failure,H/'worker_state.json');atomic_json(failure,H/'exit_status.json');raise


