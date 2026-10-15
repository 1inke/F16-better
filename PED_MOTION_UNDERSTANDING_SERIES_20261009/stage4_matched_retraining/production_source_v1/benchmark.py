"""Actual 4 warm + 32 measured original FIT steps; no saved benchmark model."""
import argparse,json,os,sys,time
import numpy as np
import torch
from controls import H,P,REPLAY,ResourceAccumulator,guard,load,atomic_json,verify_binding

class BenchmarkDone(Exception):pass
class BenchmarkRecovery:
    def before_batch(self,s):pass
    def observe(self,s):pass
    def commit(self,s):pass
    def skip_compatible(self,s):raise RuntimeError('No benchmark reuse')
    def finish(self,s):raise RuntimeError('Benchmark must stop before science outputs')
    def begin(self,s):
        samples=[];resources=ResourceAccumulator();iterator=iter(s['train_loader']);model=s['model'];model.train()
        policy=getattr(model.decoder,'apply_training_policy',None)
        if callable(policy):policy(model)
        for i in range(36):
            resources.append(guard())
            try:batch=next(iterator)
            except StopIteration:iterator=iter(s['train_loader']);batch=next(iterator)
            t=dict(s,epoch=0,equivariance_losses=[],past=batch[0],future=batch[1],num_valid=batch[2],sample_scene_ids=batch[3])
            torch.cuda.synchronize();start=time.perf_counter();exec(REPLAY,self.module.__dict__,t);s['optimizer'].zero_grad(set_to_none=True);t['loss'].backward();torch.nn.utils.clip_grad_norm_(s['trainable_parameters'],s['args'].grad_clip);s['optimizer'].step()
            if s['ema'] is not None:s['ema'].update(model)
            if s['scheduler'] is not None:s['scheduler'].step()
            torch.cuda.synchronize()
            if i>=4:samples.append(time.perf_counter()-start)
        resources.append(guard());torch.cuda.synchronize();start=time.perf_counter()
        self.module.evaluate(s['ema'].module if s['ema'] is not None else model,s['val_loader'],s['device'],None,s['args'].fusion_alpha,s['args'].focal_heading_normalize)
        torch.cuda.synchronize();valtime=time.perf_counter()-start
        out=dict(status='PASS_ACTUAL_THROWAWAY_THROUGHPUT',train_step_seconds=float(np.mean(samples)),validation_seconds=valtime,train_rows=len(s['train_indices']),validation_rows=len(s['val_indices']),batch64=True,warm=4,measured=32,resource=resources.report(),no_saved_benchmark_model=True,no_outcome_selection=True)
        atomic_json(out,H/'benchmark'/f'{os.environ["E3_BRANCH"]}_{os.environ["E3_STAGE"]}.json');raise BenchmarkDone()

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--input-branch',required=True);args,original=parser.parse_known_args()
    verify_binding(H/'source_input_binding.json');r=BenchmarkRecovery();module,_=load(args.input_branch,r);r.module=module
    sys.argv=[str(P/'isolated_trainer.py'),*original]
    try:module.run()
    except BenchmarkDone:pass
if __name__=='__main__':main()
