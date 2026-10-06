from common import *
import native, population, probes
import torch, copy

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b

def main():
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    attempt=1
    while (HERE/f'preflight_resume_attempt{attempt}').exists():attempt+=1
    if (HERE/'preflight.json').exists():
        write(f'preflight_previous/attempt{attempt-1}.json',read('preflight.json'))
    t=time.perf_counter()
    stage('PREFLIGHT_SOURCE_OLD_SHA_SNAPSHOT')
    old=old_snapshot();meta=population.metadata()
    cache,enc,obs=load_population('train');dev,_,_=load_population('dev')
    assert (len(cache['sample_id']),len(dev['sample_id']))==(28791,1516)
    assert not np.intersect1d(cache['sample_id'],dev['sample_id']).size
    # Event-rule fixtures are independent of any actual outcomes.
    past=np.stack((np.arange(-7,1)*.4,np.zeros(8)),-1)[None].astype(np.float32)
    continued=np.stack((np.arange(1,13)*.4,np.zeros(12)),-1)[None].astype(np.float32)
    turned=np.stack((np.zeros(12),np.arange(1,13)*.4),-1)[None].astype(np.float32)
    stopped=np.zeros((1,12,2),np.float32)
    assert events(past,continued)['y'][0]==0
    assert events(past,turned)['event_types'][0,0]==1
    assert events(past,stopped)['event_types'][0,3]==1
    assert events(np.zeros_like(past),continued)['event_types'][0,2]==1
    train_labels=population.labels('train')
    take=np.random.default_rng(SEED).choice(np.arange(FIT_N),128,replace=False)
    model=native.build()
    initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    local=torch.tensor(obs[take,160:304].reshape(-1,9,8,2),device='cuda:0')
    counts=torch.tensor(obs[take,304],device='cuda:0')
    torch.cuda.synchronize();begin=time.perf_counter()
    with torch.no_grad():result=native.forward(model,local,counts)
    torch.cuda.synchronize();native_time=time.perf_counter()-begin
    arrays={k:v.cpu().numpy()for k,v in result.items()}
    path_error=float(np.abs(arrays['paths']-cache['candidate'][take,:,:24].reshape(-1,20,12,2)).max())
    token_error=float(np.abs(arrays['tokens']-enc['tokens'][take]).max())
    logit_error=float(np.abs(arrays['logits']-cache['logits'][take]).max())
    social_error=float(np.abs(arrays['social']-cache['social'][take,:96]).max())
    assert max(path_error,token_error,logit_error,social_error)<2e-5
    assert arrays['E1'].shape==(128,768) and arrays['E3'].shape==(128,64)
    assert np.isfinite(arrays['paths']).all()and arrays['paths'].shape==(128,20,12,2)
    functional=native.functional(model,obs[take],cache['sample_id'][take])
    assert functional['input_jacobian_abs_mean']>0
    assert all(torch.equal(v.cpu(),initial[k])for k,v in model.state_dict().items())
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    del model;torch.cuda.empty_cache()
    x=arrays['E2_motion'];y=train_labels['y'][take]
    pair,z=probes.transform_fit(x,'full')
    assert len(np.unique(y))==2
    begin=time.perf_counter();fit=probes.logistic(z,y,1);logit_seconds=time.perf_counter()-begin
    # True two-epoch continuous versus interrupted optimizer/parameter/RNG test.
    checkpoint=HERE/f'preflight_resume_attempt{attempt}/interrupted.pt'
    assert not checkpoint.exists(), 'Preserve failed evidence; unique smoke checkpoint required'
    continuous,op1,h1=probes.train_mlp(z,y,7,epochs=2)
    state1=copy.deepcopy(continuous.state_dict());optimizer1=copy.deepcopy(op1.state_dict());rng1=probes.rng_state()
    probes.train_mlp(z,y,7,checkpoint,epochs=2,stop=1)
    resumed,op2,h2=probes.train_mlp(z,y,7,checkpoint,epochs=2)
    assert equal(state1,resumed.state_dict())and equal(optimizer1,op2.state_dict())and h1==h2 and equal(rng1,probes.rng_state())
    gradient_check=probes.TinyProbe(z.shape[1]).to('cuda:0')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(gradient_check(torch.tensor(z,device='cuda:0')),torch.tensor(y,dtype=torch.float32,device='cuda:0'))
    loss.backward();grads={k:float(v.grad.abs().mean())for k,v in gradient_check.named_parameters()}
    assert all(np.isfinite(v)and v>0 for v in grads.values())
    begin=time.perf_counter();probes.train_mlp(z,y,9,epochs=32);mlp128_seconds=time.perf_counter()-begin
    # Whole original FIT raw-observation benchmark, one throwaway logistic reader;
    # no CAL/DEV metric or model-selection decision is made from this benchmark.
    raw=raw_features(obs)
    tr=np.arange(FIT_N);mask=~train_labels['uncertain'][tr]
    begin=time.perf_counter();tf,rz=probes.transform_fit(raw['R2'][tr][mask],'full');probes.logistic(rz,train_labels['y'][tr][mask],1.);rawfit_seconds=time.perf_counter()-begin
    verify(old)
    write('preflight.json',dict(status='PASS',all_native_weights_buffers_exact=True,no_future_forward=True,
        metadata_scene_families=read('metadata_audit.json')['scene_family_counts'],native_path_max_error=path_error,
        token_max_error=token_error,logit_max_error=logit_error,social_max_error=social_error,
        native128_seconds=native_time,logistic128_seconds=logit_seconds,MLP128_32epochs_seconds=mlp128_seconds,
        rawR2_fullFIT_logistic_seconds=rawfit_seconds,probe_actual_gradients=grads,
        two_real_epochs_continuous_interrupted_params_optimizer_allRNG_history_exact=True,
        functional=functional,old_files_unchanged=len(old),source_hashes=lock_sources(),
        seconds=time.perf_counter()-t,planning_estimate_basis='Actual frozen native replay, one fullFIT raw reader, probe epoch throughput; wide estimate because nested full-representation fits/PCA and disk hashing differ.'))
    print('[PREFLIGHT PASS] '+json.dumps({k:v for k,v in read('preflight.json').items()if k not in ('source_hashes','functional')}),flush=True)

if __name__=='__main__':main()
