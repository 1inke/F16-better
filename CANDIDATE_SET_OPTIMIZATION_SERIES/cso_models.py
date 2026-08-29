"""CSO1: single-winner regression plus winner-protected candidate-set regularization."""

from __future__ import annotations
import importlib.util, os
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
SOURCE_PATH=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
BASE_LOSS_PATH=WORKSPACE/"FRONT_ENCODER_SERIES"/"locked_mode_query_loss.py"
EXPERIMENTS={"cso1_winner_protected_repulsion":False,"cso1_repulsion_prototype":True}

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise ImportError(path)
 module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

SOURCE=load("cso_grouped_source",SOURCE_PATH); BASE=load("cso_base_loss",BASE_LOSS_PATH)

class CSODecoder(SOURCE.GroupedModeTemplateDecoder):
 def __init__(self,token_dim:int,pred_len:int,use_prototype:bool)->None:
  super().__init__(token_dim,pred_len,use_template=True,use_groups=False); self.use_prototype=bool(use_prototype)
  self.register_buffer("mode_prototypes",torch.zeros(20,3,2)); self.register_buffer("prototype_seen",torch.zeros(20,dtype=torch.bool))
  self.register_buffer("prototype_updates",torch.zeros(20,dtype=torch.long)); self.last_origin=None; self.last_set_repulsion=None; self.last_prototype_loss=None
  self.cv_reference=False
 def apply_training_policy(self,owner:nn.Module)->None: owner.train(True); self.eval()
 def forward(self,encoded_tokens:Tensor,past_loc:Tensor):
  self.last_origin=past_loc[:,:,-1].detach(); return super().forward(encoded_tokens,past_loc)

class CandidateSetLoss(nn.Module):
 def __init__(self,cfg,decoder:CSODecoder)->None:
  super().__init__(); self.decoder=decoder; self.base=BASE.ModeQueryLoss(cfg); self.base.diversity_weight=0.0
  self.diversity_weight=float(cfg.MODEL.DECODER.DIVERSITY_WEIGHT); self.margin=float(cfg.MODEL.DECODER.DIVERSITY_MARGIN)
  self.prototype_weight=.01 if decoder.use_prototype else 0.; self.prototype_momentum=.99
 @staticmethod
 def weighted(values,sample_weight):
  per=values.reshape(values.shape[0],-1).mean(-1)
  if sample_weight is None:return per.mean()
  weight=sample_weight.to(values).reshape(values.shape[0]);weight=weight/weight.mean().clamp_min(1e-8);return (per*weight).mean()
 def repulsion(self,predictions,winner,sample_weight):
  endpoints=predictions[...,-1,:];distance=torch.cdist(endpoints,endpoints);modes=distance.shape[-1]
  ids=torch.arange(modes,device=distance.device).view(*([1]*(distance.ndim-2)),modes)
  nonwinner=ids!=winner[...,None];pair=nonwinner[..., :,None]&nonwinner[...,None,:]
  upper=torch.triu(torch.ones_like(distance,dtype=torch.bool),diagonal=1);mask=pair&upper
  penalty=F.relu(self.margin-distance).square()/max(self.margin,1e-8)
  count=mask.reshape(mask.shape[0],-1).sum(-1).clamp_min(1);per=(penalty*mask).reshape(mask.shape[0],-1).sum(-1)/count
  return self.weighted(per,sample_weight)
 def prototype(self,predictions,winner,sample_weight):
  if not self.decoder.use_prototype:return predictions.new_zeros(())
  if self.decoder.last_origin is None:raise RuntimeError("CSO origin state missing")
  anchors=predictions[...,[3,7,11],:]-self.decoder.last_origin[:,:predictions.shape[1],None,None,:]
  index=winner[...,None,None,None].expand(*winner.shape,1,3,2);selected=anchors.gather(-3,index).squeeze(-3)
  proto=self.decoder.mode_prototypes[winner];valid=self.decoder.prototype_seen[winner]
  error=(selected-proto).abs().mean((-1,-2));loss=self.weighted(error*valid.to(error),sample_weight)
  with torch.no_grad():
   flat_mode=winner.reshape(-1);flat_anchor=selected.detach().reshape(-1,3,2)
   for mode in flat_mode.unique():
    member=flat_anchor[flat_mode==mode];mean=member.mean(0);k=int(mode)
    if self.decoder.prototype_seen[k]:self.decoder.mode_prototypes[k].mul_(self.prototype_momentum).add_(mean,alpha=1-self.prototype_momentum)
    else:self.decoder.mode_prototypes[k].copy_(mean);self.decoder.prototype_seen[k]=True
    self.decoder.prototype_updates[k]+=len(member)
  return loss
 def forward(self,predictions,target,logits,cfg=None,sample_weight=None):
  total,parts=self.base(predictions,target,logits,cfg,sample_weight=sample_weight);winner=parts["winner"]
  rep=self.repulsion(predictions,winner,sample_weight);proto=self.prototype(predictions,winner,sample_weight)
  total=total+self.diversity_weight*rep+self.prototype_weight*proto
  self.decoder.last_set_repulsion=rep.detach();self.decoder.last_prototype_loss=proto.detach();parts["diversity"]=rep.detach();parts["prototype"]=proto.detach()
  return total,parts

def build_loss(cfg,model):
 if not isinstance(model.decoder,CSODecoder):raise TypeError(type(model.decoder))
 return CandidateSetLoss(cfg,model.decoder)

def replace_decoder(model,cfg,experiment:str)->None:
 if experiment not in EXPERIMENTS:raise ValueError(experiment)
 checkpoint=Path(os.environ.get("CSO_SOURCE_CHECKPOINT",""))
 if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
 model.decoder=CSODecoder(model.token_dim,model.pred_len,EXPERIMENTS[experiment]);source=torch.load(checkpoint,map_location="cpu",weights_only=True);target=model.state_dict()
 compatible={k:v for k,v in source.items() if k in target and tuple(v.shape)==tuple(target[k].shape)};incompatible=model.load_state_dict(compatible,strict=False)
 expected={"decoder.mode_prototypes","decoder.prototype_seen","decoder.prototype_updates"}
 if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:raise RuntimeError((incompatible.missing_keys,incompatible.unexpected_keys))
 skipped=[k for k in source if k not in compatible]
 if skipped:raise RuntimeError(f"Skipped source keys: {skipped}")
 for p in model.parameters():p.requires_grad_(False)
 for p in model.agent_aware_trans_encoder.parameters():p.requires_grad_(True)
 model.decoder.source_checkpoint=str(checkpoint);model.decoder.cso_experiment=experiment;cfg.MODEL.DECODER.NUM_MODES=20
