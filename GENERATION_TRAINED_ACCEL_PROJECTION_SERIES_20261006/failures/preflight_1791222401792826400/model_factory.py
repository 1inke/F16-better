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

def classes():
    native=torch.load(NATIVE,map_location='cpu',weights_only=True)
    expected={k[len('decoder.'):]:v for k,v in native.items() if k.startswith('decoder.')}
    assert expected
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
    folder=ROOT/'GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005';ns['HERE']=Path(__file__).resolve().parent
    nodes(ROOT/FAMILIES['raw']/'state_features.py',['state32','load_scalers'],ns)
    nodes(ROOT/FAMILIES['raw']/'run_head.py',['PulseResponseHead'],ns)
    ns['RawParent']=ns['PulseResponseHead']
    pn=dict(np=np,torch=torch,Path=Path)
    nodes(folder/'physical_accel.py',['load'],pn)
    ns['physical']=types.SimpleNamespace(**pn)
    rn=dict(np=np,torch=torch,Path=Path)
    nodes(folder/'route_features.py',['load','statistic'],rn)
    ns['route']=types.SimpleNamespace(**rn)
    nodes(folder/'run_head.py',['PhysicalExpert','PhysicalHead'],ns)
    return ns['PhysicalHead'],expected
