"""Fold-local explicit 5 endpoint x 4 path-shape prototype decoder."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor,nn


EXPERIMENTS={"p20_explicit_endpoint_path_prototypes"}


class ExplicitPrototypeDecoder(nn.Module):
    def __init__(self,token_dim:int,pred_len:int,prototypes:Tensor,hidden_dim:int=64,max_endpoint_offset:float=.75):
        super().__init__();self.num_modes=20;self.pred_len=pred_len
        if tuple(prototypes.shape)!=(20,pred_len,2):raise ValueError(f"Prototype shape must be (20,{pred_len},2), got {tuple(prototypes.shape)}")
        self.register_buffer("trajectory_prototypes",prototypes.float())
        context_dim=token_dim*2+7;self.context=nn.Sequential(nn.LayerNorm(context_dim),nn.Linear(context_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.mode_embedding=nn.Parameter(torch.empty(20,hidden_dim));nn.init.normal_(self.mode_embedding,std=.02)
        self.mode_mlp=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.residual_head=nn.Linear(hidden_dim,pred_len*2);nn.init.zeros_(self.residual_head.weight);nn.init.zeros_(self.residual_head.bias)
        envelope=torch.linspace(.20,max_endpoint_offset,pred_len);self.register_buffer("residual_envelope",envelope)
        self.score_head=nn.Sequential(nn.LayerNorm(hidden_dim+4),nn.Linear(hidden_dim+4,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        nn.init.zeros_(self.score_head[-1].weight);nn.init.zeros_(self.score_head[-1].bias)
        self.last_relative_trajectories=None

    @staticmethod
    def motion_features(past:Tensor)->Tensor:
        short=past[:,:,-1]-past[:,:,-2];lag=min(3,past.shape[2]-1);long=(past[:,:,-1]-past[:,:,-1-lag])/float(lag)
        acceleration=short-long;speed=torch.linalg.vector_norm(short,dim=-1,keepdim=True);return torch.cat((short,long,acceleration,speed),dim=-1)

    def forward(self,encoded_tokens:Tensor,past_loc:Tensor):
        recent=encoded_tokens[:,:,-1];pooled=encoded_tokens.mean(2)
        context=self.context(torch.cat((recent,pooled,self.motion_features(past_loc)),dim=-1))
        mode=self.mode_mlp(context[:,:,None]+self.mode_embedding[None,None])
        residual=torch.tanh(self.residual_head(mode).reshape(*mode.shape[:-1],self.pred_len,2))*self.residual_envelope[None,None,None,:,None]
        relative=self.trajectory_prototypes[None,None]+residual;trajectories=past_loc[:,:,None,-1:]+relative
        endpoint=relative[...,-1,:];mean_speed=torch.linalg.vector_norm(torch.cat((relative[...,:1,:],relative[...,1:,:]-relative[...,:-1,:]),dim=-2),dim=-1).mean(-1,keepdim=True)
        logits=self.score_head(torch.cat((mode,endpoint,mean_speed,torch.linalg.vector_norm(endpoint,dim=-1,keepdim=True)),dim=-1)).squeeze(-1)
        self.last_relative_trajectories=relative.detach();return trajectories,logits


def replace_decoder(model,cfg,name:str)->None:
    if name not in EXPERIMENTS:raise ValueError(name)
    prototype_text=os.environ.get("EXP163_PROTOTYPE_PATH","").strip();source_text=os.environ.get("EXP163_SOURCE_CHECKPOINT","").strip()
    if not prototype_text or not source_text:raise RuntimeError("EXP163_PROTOTYPE_PATH and EXP163_SOURCE_CHECKPOINT are required")
    prototype_path=Path(prototype_text);source_path=Path(source_text)
    if not prototype_path.is_file():raise FileNotFoundError(prototype_path)
    if not source_path.is_file():raise FileNotFoundError(source_path)
    with np.load(prototype_path) as raw:prototypes=torch.from_numpy(raw["trajectory_prototypes"])
    model.decoder=ExplicitPrototypeDecoder(model.token_dim,model.pred_len,prototypes)
    source=torch.load(source_path,map_location="cpu",weights_only=True);encoder_state={k:v for k,v in source.items() if not k.startswith("decoder.")}
    incompatible=model.load_state_dict(encoder_state,strict=False);expected={k for k in model.state_dict() if k.startswith("decoder.")}
    if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:raise RuntimeError((incompatible.missing_keys,incompatible.unexpected_keys))
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():parameter.requires_grad_(True)
    model.decoder.prototype_path=str(prototype_path);model.decoder.source_checkpoint=str(source_path)
    cfg.MODEL.DECODER.NUM_MODES=20;cfg.MODEL.DECODER.CLASSIFICATION_WEIGHT=1.0
