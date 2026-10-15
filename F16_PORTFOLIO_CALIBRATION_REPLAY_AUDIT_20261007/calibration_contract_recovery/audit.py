"""One-field frozen replay repair; original parent is immutable."""
import os,sys,importlib.util,json
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('frozen_parent_audit',PARENT/'audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.HERE=HERE
oldinfer=a.infer;oldmanifest=a.manifest
def infer(m,arrays):
 if m.__class__.__name__!='IntentClassifier':return oldinfer(m,arrays)
 pieces=[]
 with a.torch.inference_mode():
  for i in range(0,len(arrays[0]),128):
   a.guard();logits=m(*[a.torch.from_numpy(x[i:i+128]).to(a.DEVICE) for x in arrays])
   p=a.torch.softmax(logits,dim=-1);assert p.shape[1]==7 and a.torch.isfinite(p).all() and p.min()>=0 and p.max()<=1
   assert (p.sum(1)-1).abs().max()<1e-6
   # Independent stable softmax on saved logits is a numeric scalar contract fixture.
   l=logits.cpu().numpy().astype(a.np.float64);q=a.np.exp(l-l.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
   assert abs(q-p.cpu().numpy()).max()<2e-7
   pieces.append(p.cpu().numpy())
 return a.np.concatenate(pieces)
def manifest(ns):
 locks=oldmanifest(ns)
 for p in [PARENT/'audit.py',PARENT/'DESIGN.md',PARENT/'RESULT.json',PARENT/'completion_review/REVIEW.json',PARENT/'completion_review/AGGREGATE.json',*sorted((PARENT/'saved').glob('*.npz'))]:locks[str(p)]=a.sha(p)
 return locks
a.infer=infer;a.manifest=manifest
if __name__=='__main__':
 try:a.main('--smoke' in sys.argv)
 except Exception:
  a.write('FAILURE_'+str(a.time.time_ns())+'.json',dict(status='STOP',traceback=a.traceback.format_exc(),resource=a.RESOURCE));raise
