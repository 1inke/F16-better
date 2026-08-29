"""TECM1: add a time-resolved, receiver-specific edge message to locked F16."""
from __future__ import annotations
import math, types
from typing import Optional
import torch
from torch import Tensor, nn

EXPERIMENTS={"tecm1_time_resolved_edge_message"}

class EdgeTimeMessage(nn.Module):
    """r(i,j,t) -> delta-v(i,j,t), zero at E0 initialisation."""
    def __init__(self, dim:int, hidden:int=16) -> None:
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.enabled=True; self.last_residual:Optional[Tensor]=None
    def forward(self, feature:Tensor)->Tensor:
        value=self.net(feature)
        agents=value.shape[1]
        eye=torch.eye(agents,dtype=torch.bool,device=value.device)[None,:,:,None,None]
        value=value.masked_fill(eye,0.0)
        if not self.enabled: value=torch.zeros_like(value)
        self.last_residual=value.detach()
        return value

def _relation_tokens_tap(self,past:Tensor):
    features,distance,closing=self.relation_features(past); self.last_tecm_features=features
    encoded=self.relation_encoder(features)
    if self.use_multiscale_relation:
        scales=torch.stack((encoded[:,:,:,-2:].mean(3),encoded[:,:,:,-4:].mean(3),encoded.mean(3)),3)
        tokens=(torch.softmax(self.relation_scale_score(scales),3)*scales).sum(3)
    else: tokens=(torch.softmax(self.relation_time_score(encoded),3)*encoded).sum(3)
    return tokens,distance,closing,features[:,:,:,-1,6]*10.0+features[:,:,:,-1,7]

def _temporal_social_attention_tecm(self,temporal:Tensor,agent:Tensor,relation:Tensor,allowed:Tensor,type_features=None,risk_score=None)->Tensor:
    del type_features,risk_score
    b,n,t,_=temporal.shape
    feature=self.last_tecm_features
    if feature is None or feature.shape[3]!=t: raise RuntimeError("TECM time features unavailable or not aligned with tokens")
    q=self.q_proj(agent).reshape(b,n,self.heads,self.qk_head_dim).permute(0,2,1,3)
    k=self.k_proj(temporal).reshape(b,n,t,self.kv_heads,self.qk_head_dim).permute(0,3,1,2,4)
    v=self.v_proj(temporal).reshape(b,n,t,self.kv_heads,self.head_dim).permute(0,3,1,2,4)
    if self.kv_heads!=self.heads:
        repeats=self.heads//self.kv_heads;k=k.repeat_interleave(repeats,1);v=v.repeat_interleave(repeats,1)
    # Routing remains byte-for-byte F16's content score plus pair-level bias.
    scores=torch.einsum("bhid,bhjtd->bhijt",q,k)/math.sqrt(float(self.qk_head_dim))
    scores=scores+self.relation_bias(relation).permute(0,3,1,2)[...,None]
    mask=allowed[:,None,:,:,None].expand_as(scores) if allowed.ndim==3 else allowed[:,:,:,:,None].expand_as(scores)
    weights=torch.softmax(scores.masked_fill(~mask,-1.0e4).flatten(-2),-1).reshape_as(scores)
    base=v[:,:,None].expand(-1,-1,n,-1,-1,-1)
    pair=self.relation_value(relation).reshape(b,n,n,self.heads,self.head_dim).permute(0,3,1,2,4)[...,None,:]
    residual=self.tecm_edge_message(feature).reshape(b,n,n,t,self.heads,self.head_dim).permute(0,4,1,2,3,5)
    social=(weights[...,None]*(base+pair+residual)).sum((3,4)).permute(0,2,1,3).reshape(b,n,self.dim)
    return self.out_proj(social)

def replace_encoder(model:nn.Module,cfg,experiment:str)->None:
    del cfg
    if experiment not in EXPERIMENTS or model.front_variant!="f16_temporal_social_cross": raise ValueError("TECM1 requires locked F16")
    base=model.agent_aware_trans_encoder
    forbidden=("use_pair_deformable_temporal","use_pair_temporal_soft_prior","use_scaled_cosine_attention","use_differential_attention","use_relation_type_bias","use_soft_risk_bias","use_temporal_relative_bias")
    active=[x for x in forbidden if bool(getattr(base,x,False))]
    if active or not base.use_relation_tokens or not base.use_temporal_social_cross: raise ValueError(f"incompatible F16 flags: {active}")
    base.tecm_edge_message=EdgeTimeMessage(int(base.dim));base.last_tecm_features=None
    base.relation_tokens=types.MethodType(_relation_tokens_tap,base)
    base.temporal_social_attention=types.MethodType(_temporal_social_attention_tecm,base)
