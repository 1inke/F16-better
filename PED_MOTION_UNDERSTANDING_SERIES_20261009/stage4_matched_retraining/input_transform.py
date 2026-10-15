"""Raw-input E3 availability interventions, before every model preprocessing path."""
import numpy as np
def transform(past, valid, branch):
    x=np.array(past,copy=True);n=np.array(valid,copy=True)
    assert x.ndim==4 and x.shape[2:]==(8,2) and n.shape==(len(x),)
    assert np.isfinite(x).all() and (n>=1).all() and (n<=x.shape[1]).all()
    if branch=='full':return x,n
    if branch=='last2':x[:,:,:6]=x[:,:,6:7]
    elif branch=='noNbr':x[:,1:]=x[:,:1];n[:]=1
    else:raise ValueError(branch)
    return x,n
def fixtures():
    rng=np.random.default_rng(20261009);x=rng.normal(size=(7,9,8,2)).astype(np.float32);n=np.arange(1,8)
    y,m=transform(x,n,'full');assert np.array_equal(x,y) and np.array_equal(n,m)
    changed=x.copy();changed[:,:,:6]=rng.normal(size=changed[:,:,:6].shape)
    for branch in ('last2',):
        y,m=transform(x,n,branch);z,k=transform(changed,n,branch)
        assert np.array_equal(y,z) and np.array_equal(y[:,:,6:],x[:,:,6:]) and np.array_equal(m,k)
    changed=x.copy();changed[:,1:]=rng.normal(size=changed[:,1:].shape)
    y,m=transform(x,n,'noNbr');z,k=transform(changed,n,'noNbr')
    assert np.array_equal(y,z) and np.array_equal(m,np.ones_like(n)) and np.array_equal(y[:,0],x[:,0])
    assert np.array_equal(x,transform(x,n,'full')[0])
    return {'status':'PASS_INPUT_TRANSFORM_FIXTURES_ONLY','full_exact':True,'last2_erases_early6':True,'noNbr_erases_neighbors':True,'no_model_forward_gradient_resume_preflight_yet':True}
if __name__=='__main__':
    import json
    print(json.dumps(fixtures()))
