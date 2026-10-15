import os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controls import H,guard
from commands import command
def main():
    for branch in ('full','last2','noNbr'):
        for stage in ('h3','v2'):
            guard();cmd,env,dest=command('eth',1,branch,stage,benchmark=True)
            log=H/'prelaunch_review'/f'benchmark_{branch}_{stage}.log'
            if log.exists():raise RuntimeError('Existing benchmark retained, no automatic rerun')
            with log.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=H,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Benchmark failed: '+str(log))
            print(branch,stage,'THROUGHPUT_PASS',flush=True)
if __name__=='__main__':main()
