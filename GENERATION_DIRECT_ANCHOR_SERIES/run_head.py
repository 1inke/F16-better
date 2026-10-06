"""Direct observation-to-anchor head; native-feature vs raw-query inputs.
Native whole decoder/F16/scorer fixed. Final output retains common native clip.
No GT in forward or fit-only native-teacher ridge initialization.
"""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('direct_native_source',HERE.parent/'GENERATION_NATIVE_DECODER_SERIES/run_native.py')
n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n)
c=n.c;m=n.m;r=n.r;p=n.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BUILD_INITIAL=False
BaseNative=n.NativeGenerator

class DirectHead(BaseNative):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        for parameter in self.decoder.parameters():parameter.requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        self.register_buffer('native_fallback',torch.tensor(False))
        # Failed ridge/distillation initialization is preserved, never loaded.
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*n.TOKEN_DIM].reshape(-1,1,8,n.TOKEN_DIM)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        second=d.mode_mlp(context[:,:,None]+query) if self.arm=='zero' else query
        return torch.cat((context[:,:,None].expand_as(second),second),-1)[:,0]
    def forward(self,observation,base,speed):
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        anchors=self.head(self.inputs(observation)).reshape(-1,20,3,2)*(speed[:,0,None,None,None]+.03)
        # Direct branch predicts corrections at3anchors, not absolute trajectories.
        delta=torch.einsum('tp,bkpc->bktc',self.decoder.interpolation,anchors)
        velocity=torch.diff(delta,dim=2,prepend=torch.zeros_like(base[:,:,:1]))
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        residual=velocity.clamp(-limit,limit)
        return base+residual.cumsum(2),residual

def frozen_audit(model):
    for key,value in model.decoder.state_dict().items():assert torch.equal(value.detach().cpu(),n.DECODER_STATE[key]),key
    assert not any(p.requires_grad for p in model.decoder.parameters())
n.NativeGenerator=DirectHead;c.Correction=DirectHead;r.m.Correction=DirectHead;n.frozen_audit=frozen_audit
ORIGINAL_SAVE=m.save_atomic
def save_atomic(value,path):
    # Trainer epoch0 identity is not native for a ridge head. Make fallback explicit.
    if Path(path).name=='constrained.pth' and value.get('epoch')==0:
        value={**value,'state_dict':dict(value['state_dict']),'native_fallback_explicit':True}
        value['state_dict']['native_fallback']=torch.tensor(True)
    ORIGINAL_SAVE(value,path)
m.save_atomic=save_atomic

def initialize(device):
    global BUILD_INITIAL
    _,_,original=n.prepare_backend(device);del original
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(fit)&set(cal))
    values=r.prepared(cache,False,fit=fit);obs,base,speed=values[:3]
    provenance={str(p.TRAIN_CACHE):p.sha256(p.TRAIN_CACHE),str(n.HERE/'train_encoder.npz'):p.sha256(n.HERE/'train_encoder.npz'),
        str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json'),str(Path(__file__).resolve()):p.sha256(Path(__file__))}
    diagnostics=[];BUILD_INITIAL=True
    for seed in (1,2,3,7):
        for arm in c.ARMS:
            path=HERE/f'initial/{arm}/seed_{seed}.pth'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['source_hashes']==provenance and saved['fit_ids']==cache['sample_id'][fit].tolist()
                print(f'[INITIAL VERIFIED] {arm} seed{seed}',flush=True);continue
            p.seed_everything(seed);model=DirectHead(obs.shape[1],c.settings(seed,arm)).to(device);model.eval()
            gram=torch.zeros(65,65,dtype=torch.float64,device=device);cross=torch.zeros(65,6,dtype=torch.float64,device=device)
            with torch.no_grad():
                for start in range(0,len(fit),256):
                    ids=fit[start:start+256];o=torch.from_numpy(obs[ids]).to(device)
                    feature=model.head[:4](model.inputs(o)).reshape(-1,64).double()
                    augmented=torch.cat((feature,torch.ones(len(feature),1,dtype=feature.dtype,device=device)),-1)
                    teacher=(base[ids][:,:,[3,7,11]]/(speed[ids,0,None,None,None]+.03)).reshape(-1,6)
                    target=torch.from_numpy(teacher).to(device,dtype=torch.float64)
                    gram+=augmented.T@augmented;cross+=augmented.T@target
                penalty=cfg['ridge_relative']*gram[:64,:64].trace()/64
                diagonal=torch.ones(65,dtype=gram.dtype,device=device);diagonal[-1]=0
                matrix=gram+torch.diag(diagonal*penalty)
                coefficients=torch.linalg.solve(matrix,cross)
                assert torch.isfinite(coefficients).all()
                model.head[-1].weight.copy_(coefficients[:64].T.float());model.head[-1].bias.copy_(coefficients[64].float())
                condition=float(torch.linalg.cond(matrix));frozen_audit(model)
                initial=dict(seed=seed,arm=arm,fit_only_no_gt=True,fit_ids=cache['sample_id'][fit].tolist(),source_hashes=provenance,
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},condition_number=condition,ridge=float(penalty))
                path.parent.mkdir(parents=True,exist_ok=True);ORIGINAL_SAVE(initial,path)
                # Native teacher fidelity only; no GT metric used to select this initialization.
                ids=cal[:256];o,b,s=[torch.from_numpy(v[ids]).to(device) for v in values[:3]]
                out,res=model(o,b,s)
                record=dict(seed=seed,arm=arm,condition_number=condition,ridge=float(penalty),cal_first256_teacher_ade=float((out-b).norm(dim=-1).mean()),
                    cal_first256_teacher_fde=float((out[:,:,-1]-b[:,:,-1]).norm(dim=-1).mean()))
                diagnostics.append(record);print('[INITIAL] '+json.dumps(record),flush=True)
    BUILD_INITIAL=False
    m.write_json(HERE/'initialization_audit.json',dict(status='PASS',fit_only_no_gt=True,fit_n=len(fit),cal_n=len(cal),diagnostics=diagnostics,
        scope='raw-head initialization not native identity; last layer differs across arms after teacher fit; native finalclip retained; seed7 resume smoke only'))

def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    previous=json.loads((HERE.parent/'GENERATION_CONDITIONAL_MODE_SERIES/preflight.json').read_text())
    for path,digest in previous['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    scale=torch.tensor(previous['fit_protection_scales'],device=device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);counts={};front=[];timings={}
    for arm in c.ARMS:
        p.seed_everything(1);model=DirectHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        front.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        counts[arm]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        start=time.perf_counter();out,res=model(o,b,s)
        assert out.shape==b.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert torch.equal(out,b) and res.abs().sum()==0
        assert (res.abs()<=cfg['velocity_residual_scale']*(s[:,0,None,None,None]+.03)+1e-6).all()
        loss=m.objective(out,b,gt,res,s,group,cfg,'balanced_protected',scale);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
        assert any(p.grad.abs().sum()>0 for p in model.head.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);optimizer.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=time.perf_counter()-start
        model.native_fallback.fill_(True);fallback,zero=model(o,b,s);assert torch.equal(fallback,b) and zero.abs().sum()==0
    assert counts['zero']==counts['mean_std']
    assert all(torch.equal(front[0][k],front[1][k]) for k in front[0])
    paths=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'initialization_audit.json',
        HERE.parent/'GENERATION_DIRECT_ANCHOR_DESIGN.md',HERE.parent/'GENERATION_CONDITIONAL_MODE_SERIES/trainability_diagnosis.json']+list((HERE/'initial').glob('*/*.pth'))
    sources={**previous['source_hashes'],**{str(path):p.sha256(path) for path in paths}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=previous['fit_protection_scales'],source_hashes=sources,
        parameters_trainable=counts,seconds_per_128step=timings,paired_full_init_exact=True,full_init_not_equal=False,native_identity=True,native_fallback_explicit=True,test_access=False))
    print('PREFLIGHT PASS directhead/frozen/finiteK20/gradient/fitcal/bounds/frontinit/fallback/hash',flush=True)

def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch new direct3anchor correction branch trainings; both identical25158parameter MLP, full paired initialization exact and nativeoutputidentity. zero=context+native-mode-feature; mean_std=context+fixedrawquery. Original decoder/F16/scorer wholeweights exactfrozen. No nativepath as branchMLP input, output is nativepluslearnedanchorcorrection undercommon velocityclip; NOTabsolute/independent generator. Differentfeature distributions not equal effectivecapacityproof. Failedridge/teacherfit preserved, unused. Nativeepoch0fallback explicit;notlearnedsuccess. K20/original loss+cal/no test/no intent or ranking claim; repeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'DIRECT_ANCHOR_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Direct anchor decoder\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
c.preflight=preflight;c.evaluate=evaluate

def main():
    if '--initialize' in sys.argv:raise RuntimeError('Ridge initialization failed; preserved only. Current branch uses exact zero correction.')
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=n.prepare_backend(torch.device('cpu'));del original
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
