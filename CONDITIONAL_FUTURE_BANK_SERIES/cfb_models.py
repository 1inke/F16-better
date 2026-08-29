"""CFB1: train-only conditional pseudo-futures for unique non-winner modes."""
from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
import numpy as np,torch
from torch import Tensor,nn

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent
SOURCE_PATH=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py";BASE_LOSS_PATH=WORKSPACE/"FRONT_ENCODER_SERIES"/"locked_mode_query_loss.py"
EXPERIMENTS={"cfb1_cross_scene_three_future"}
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
SOURCE=load("cfb_grouped_source",SOURCE_PATH);BASE=load("cfb_base_loss",BASE_LOSS_PATH)
class CFBDecoder(SOURCE.GroupedModeTemplateDecoder):
 def __init__(self,token_dim:int,pred_len:int,cache_path:Path)->None:
  super().__init__(token_dim,pred_len,use_template=True,use_groups=False);cache=np.load(cache_path,allow_pickle=False);self.cache_metadata=json.loads(str(cache["metadata_json"].item()))
  self._pseudo=torch.from_numpy(np.asarray(cache["pseudo_futures"],np.float32));self._batch_indices=None;self.last_origin=None
  self.register_buffer("pseudo_assignment_counts",torch.zeros(20,dtype=torch.long));self.register_buffer("pseudo_assignment_total",torch.zeros((),dtype=torch.long));self.last_pseudo_loss=None;self.cv_reference=False
 def set_scene_batch(self,encoded_indices:Tensor,past:Tensor)->None:
  del past;indices=-encoded_indices.long()-2
  if (indices<0).any() or (indices>=len(self._pseudo)).any():raise RuntimeError("CFB sample-index sidecar incompatible")
  self._batch_indices=indices.detach().cpu()
 def apply_training_policy(self,owner:nn.Module)->None:owner.train(True);self.eval()
 def forward(self,encoded_tokens:Tensor,past_loc:Tensor):self.last_origin=past_loc[:,:,-1];return super().forward(encoded_tokens,past_loc)
class CFBLoss(nn.Module):
 def __init__(self,cfg,decoder:CFBDecoder)->None:super().__init__();self.decoder=decoder;self.base=BASE.ModeQueryLoss(cfg);self.pseudo_weight=.10
 def forward(self,predictions,target,logits,cfg=None,sample_weight=None):
  total,parts=self.base(predictions,target,logits,cfg,sample_weight=sample_weight)
  if self.decoder._batch_indices is None or self.decoder.last_origin is None:raise RuntimeError("CFB batch state missing")
  pseudo=self.decoder._pseudo[self.decoder._batch_indices].to(predictions.device)[:,None]+self.decoder.last_origin[:,:predictions.shape[1],None,None,:]
  winner=parts["winner"];used=torch.zeros_like(logits,dtype=torch.bool);used.scatter_(-1,winner[...,None],True);chosen=[]
  for m in range(pseudo.shape[2]):
   error=torch.linalg.vector_norm(predictions-pseudo[:,:,m,None],dim=-1);cost=error.mean(-1)+.25*error[...,-1];choice=cost.detach().masked_fill(used,float("inf")).argmin(-1);chosen.append(choice);used.scatter_(-1,choice[...,None],True)
  assignment=torch.stack(chosen,dim=-1);index=assignment[...,None,None].expand(*assignment.shape,predictions.shape[-2],predictions.shape[-1]);expanded=predictions.unsqueeze(-4).expand(*predictions.shape[:2],pseudo.shape[2],*predictions.shape[2:]);selected=expanded.gather(-3,index.unsqueeze(-3)).squeeze(-3)
  # selected and pseudo are [B,A,M,T,2]; each pseudo target owns exactly one mode.
  pseudo_loss=self.base.regression(selected,pseudo)
  total=total+self.pseudo_weight*pseudo_loss;self.decoder.last_pseudo_loss=pseudo_loss.detach()
  with torch.no_grad():
   counts=torch.bincount(assignment.reshape(-1),minlength=20).to(self.decoder.pseudo_assignment_counts.device);self.decoder.pseudo_assignment_counts.add_(counts);self.decoder.pseudo_assignment_total.add_(assignment.numel())
  parts["pseudo"]=pseudo_loss.detach();parts["pseudo_assignment"]=assignment.detach();return total,parts
def build_loss(cfg,model):
 if not isinstance(model.decoder,CFBDecoder):raise TypeError(type(model.decoder))
 return CFBLoss(cfg,model.decoder)
def replace_decoder(model,cfg,experiment:str)->None:
 if experiment not in EXPERIMENTS:raise ValueError(experiment)
 checkpoint=Path(os.environ.get("CFB_SOURCE_CHECKPOINT",""));cache=Path(os.environ.get("CFB_CACHE",""))
 if not checkpoint.is_file() or not cache.is_file():raise FileNotFoundError((checkpoint,cache))
 model.decoder=CFBDecoder(model.token_dim,model.pred_len,cache);source=torch.load(checkpoint,map_location="cpu",weights_only=True);target=model.state_dict();compatible={k:v for k,v in source.items() if k in target and tuple(v.shape)==tuple(target[k].shape)};incompatible=model.load_state_dict(compatible,strict=False)
 expected={"decoder.pseudo_assignment_counts","decoder.pseudo_assignment_total"}
 if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:raise RuntimeError((incompatible.missing_keys,incompatible.unexpected_keys))
 skipped=[k for k in source if k not in compatible]
 if skipped:raise RuntimeError(f"Skipped source keys: {skipped}")
 for p in model.parameters():p.requires_grad_(False)
 for p in model.agent_aware_trans_encoder.parameters():p.requires_grad_(True)
 model.decoder.source_checkpoint=str(checkpoint);model.decoder.future_bank_cache=str(cache);cfg.MODEL.DECODER.NUM_MODES=20
