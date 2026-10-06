"""Read-only AST constructor. Never executes legacy module top-level code."""
from pathlib import Path
import ast, math, types
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from resource_guard import guard
ROOT=Path(__file__).resolve().parent.parent
FAMILIES={
 'raw':'GENERATION_CANDIDATE_STATE_SERIES_20261005',
 'centered_raw':'GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005',
 'centered_own':'GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005'}
SOURCES=['HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py',
 'CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py',
 'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py',
 'GENERATION_NATIVE_DECODER_SERIES/run_native.py',
 'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py',
 'GENERATION_TIME_RESPONSE_SERIES_20261004/run_head.py']
NATIVE=ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/eth/v2/f16_temporal_social_cross/seed_1/val_scene_-1/best_selected.pth'

def nodes(path,names,ns):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert len(selected)==len(names),(path,names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)

def build(family,checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['input_dim']==940 and saved['config']['set_context']=='mean_std'
    assert np.array_equal(saved['mean'],np.zeros((1,940),np.float32))
    assert np.array_equal(saved['std'],np.ones((1,940),np.float32))
    native=torch.load(NATIVE,map_location='cpu',weights_only=True)
    expected={k[len('decoder.'):]:v for k,v in native.items() if k.startswith('decoder.')}
    assert expected and all(torch.equal(saved['state_dict']['decoder.'+k],v) for k,v in expected.items())
    ns=dict(torch=torch,nn=nn,Tensor=Tensor,np=np,math=math,F=F,guard=guard,Path=Path)
    nodes(ROOT/SOURCES[0],['endpoint_template_initialization','interpolation_weights','AnchorConditionedRefiner','HierarchicalLearnedAnchorDecoder'],ns)
    ns['SOURCE']=types.SimpleNamespace(HierarchicalLearnedAnchorDecoder=ns['HierarchicalLearnedAnchorDecoder'])
    nodes(ROOT/SOURCES[1],['BoundedAnchorCorrection','FrozenControlledAnchorDecoder'],ns)
    ns['CONTROLLED']=types.SimpleNamespace(FrozenControlledAnchorDecoder=ns['FrozenControlledAnchorDecoder'])
    nodes(ROOT/SOURCES[2],['JointTrainableV2Decoder'],ns)
    ns.update(DECODER_STATE=expected,DECODER_CLASS=ns['JointTrainableV2Decoder'],TOKEN_DIM=96)
    nodes(ROOT/SOURCES[3],['NativeGenerator'],ns)
    ns['BaseNative']=ns['NativeGenerator']
    ns['n']=types.SimpleNamespace(TOKEN_DIM=96)
    nodes(ROOT/SOURCES[4],['DirectHead'],ns)
    ns['a']=types.SimpleNamespace(DirectHead=ns['DirectHead'])
    nodes(ROOT/SOURCES[5],['positions','TimeResponseHead'],ns)
    ns['t']=types.SimpleNamespace(TimeResponseHead=ns['TimeResponseHead'])
    ns['OriginalTimeResponseHead']=ns['TimeResponseHead']
    folder=ROOT/FAMILIES[family];ns['HERE']=folder
    feats=['state32','load_scalers']
    if family!='raw':feats+=['observed_reference']
    nodes(folder/'state_features.py',feats,ns)
    nodes(folder/'run_head.py',['PulseResponseHead'],ns)
    model=ns['PulseResponseHead'](940,saved['config'])
    model.load_state_dict(saved['state_dict'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    return model,dict(epoch=int(saved['epoch']),family=family,selection_policy=saved.get('selection_policy'),fallback=bool(model.native_fallback),config=saved['config'])

def exact(model,state):
    assert model.state_dict().keys()==state.keys()
    assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
    assert not model.training and not any(p.requires_grad for p in model.parameters())

