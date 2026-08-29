"""FCRA1: online full-consistent residuals, explicit Top-3 responsibility and no-op."""

from __future__ import annotations
import hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor,nn

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent
IR1_PATH=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"irs_models.py"
SOURCE_PATH=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
EXPERIMENTS={"fcra1_full_consistent_top3"}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

IR1=load("fcra_ir1_taps",IR1_PATH);SOURCE=load("fcra_grouped_source",SOURCE_PATH)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def masked_mean(values,mask):
    weight=mask.to(values.dtype);return (values*weight).sum()/weight.sum().clamp_min(1.0)

class ResponsibilityCorrectionHead(nn.Module):
    def __init__(self,social_dim:int,mode_dim:int,hidden:int=16)->None:
        super().__init__();self.social_norm=nn.LayerNorm(social_dim);self.social_down=nn.Linear(social_dim,hidden);self.act=nn.GELU()
        self.need=nn.Linear(hidden,1);joint=mode_dim+hidden
        self.assignment=nn.Sequential(nn.LayerNorm(joint),nn.Linear(joint,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.residual=nn.Sequential(nn.LayerNorm(joint),nn.Linear(joint,32),nn.GELU(),nn.Linear(32,6))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,social,mode):
        hidden=self.act(self.social_down(self.social_norm(social)));joint=torch.cat((mode,hidden[:,:,None].expand(-1,-1,mode.shape[2],-1)),dim=-1)
        return hidden,self.need(hidden).squeeze(-1),self.assignment(joint).squeeze(-1),self.residual(joint).reshape(*mode.shape[:-1],3,2)

class FCRADecoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self,token_dim:int,pred_len:int,social_dim:int,cache_path:Path,top_m:int=3,temperature:float=.10)->None:
        super().__init__(token_dim,pred_len,use_template=True,use_groups=False);self.social_dim=int(social_dim);self.top_m=int(top_m);self.temperature=float(temperature)
        self.head=ResponsibilityCorrectionHead(self.social_dim,int(self.mode_embedding.shape[-1]),16)
        cache=np.load(cache_path,allow_pickle=False);self.cache_metadata=json.loads(str(cache["metadata_json"].item()))
        self._cache_need=torch.from_numpy(np.asarray(cache["need_label"],dtype=np.float32));self._batch_indices=None
        self.last_need_logits=None;self.last_need_probability=None;self.last_assignment_logits=None;self.last_sparse_responsibility=None
        self.last_predicted_residual=None;self.last_applied_anchor_delta=None;self.last_uncorrected_trajectories_graph=None;self.last_uncorrected_anchors_graph=None
        self.last_target_responsibility=None;self.last_target_residual=None;self.last_aux_need=None;self.last_aux_assign=None;self.last_aux_corr=None;self.last_aux_noop=None
    def set_scene_batch(self,encoded_indices:Tensor,past:Tensor)->None:
        del past;indices=-encoded_indices.long()-2
        if (indices<0).any() or (indices>=len(self._cache_need)).any():raise RuntimeError("FCRA sample-index sidecar incompatible")
        self._batch_indices=indices.detach().cpu()
    def apply_training_policy(self,owner:nn.Module)->None:
        owner.train(True);self.eval();self.head.train(True)
    def _sparse_predicted_responsibility(self,logits):
        probability=torch.softmax(logits,dim=-1);indices=probability.topk(self.top_m,dim=-1).indices
        mask=torch.zeros_like(probability,dtype=torch.bool);mask.scatter_(-1,indices,True)
        sparse=probability*mask.to(probability.dtype);return sparse/sparse.sum(dim=-1,keepdim=True).clamp_min(1e-8)
    def auxiliary_training_loss(self,prediction:Tensor,target:Tensor,logits:Tensor)->Tensor:
        del prediction,logits
        if self._batch_indices is None or self.last_uncorrected_trajectories_graph is None:raise RuntimeError("FCRA auxiliary state missing")
        device=target.device;label=self._cache_need[self._batch_indices].to(device);positive=label>0.5;negative=~positive
        base_traj=self.last_uncorrected_trajectories_graph[:,0];base_anchor=self.last_uncorrected_anchors_graph[:,0]
        focal_target=target[:,0];distance=torch.linalg.vector_norm(base_traj-focal_target[:,None],dim=-1)
        cost=distance.mean(-1)+.5*distance[...,-1]
        target_logits=-cost.detach()/self.temperature;top=target_logits.topk(self.top_m,dim=-1).indices
        top_mask=torch.zeros_like(target_logits,dtype=torch.bool);top_mask.scatter_(-1,top,True)
        target_logits=target_logits.masked_fill(~top_mask,-1e4);responsibility=torch.softmax(target_logits,dim=-1)
        target_anchor=focal_target[:,[3,7,11]];residual_target=(target_anchor[:,None]-base_anchor).detach()
        predicted_residual=self.last_predicted_residual[:,0];assignment_logits=self.last_assignment_logits[:,0]
        need_loss=torch.nn.functional.binary_cross_entropy_with_logits(self.last_need_logits[:,0],label)
        assignment_per=-(responsibility*torch.log_softmax(assignment_logits,dim=-1)).sum(-1)
        correction_per=(predicted_residual-residual_target).abs().mean(dim=(-1,-2));correction_per=(responsibility*correction_per).sum(-1)
        applied=self.last_applied_anchor_delta[:,0].abs().mean(dim=(-1,-2,-3))
        assign_loss=masked_mean(assignment_per,positive);corr_loss=masked_mean(correction_per,positive);noop_loss=masked_mean(applied,negative)
        self.last_target_responsibility=responsibility.detach();self.last_target_residual=residual_target.detach()
        self.last_aux_need=need_loss.detach();self.last_aux_assign=assign_loss.detach();self.last_aux_corr=corr_loss.detach();self.last_aux_noop=noop_loss.detach()
        return .05*need_loss+.05*assign_loss+.10*corr_loss+.05*noop_loss
    def forward(self,encoded_tokens:Tensor,past_loc:Tensor):
        social=IR1._SOCIAL_DELTA.get()
        if social is None or social.shape[-1]!=self.social_dim:raise RuntimeError("FCRA did not receive F16 social delta")
        recent=encoded_tokens[:,:,-1];pooled=encoded_tokens.mean(2);context=self.context(torch.cat((recent,pooled,self.motion_features(past_loc)),dim=-1))
        mode=self.mode_mlp(context[:,:,None]+self.mode_embedding[None,None]);scale=torch.nn.functional.softplus(self.proposal_scale(context))+.25
        proposal_delta=.75*torch.tanh(self.proposal_delta(mode)).reshape(*mode.shape[:-1],3,2)
        source_anchors=scale[:,:,None,None]*(self.anchor_templates[None,None]+proposal_delta);base_anchors=source_anchors+self.refiner(mode,source_anchors)
        base_relative=torch.einsum("tp,bakpc->baktc",self.interpolation,base_anchors)
        focal_relative,focal_anchors=self.template_head(context[:,:1],mode[:,:1],base_anchors[:,:1],self.interpolation,None)
        uncorrected_relative=torch.cat((focal_relative,base_relative[:,1:]),dim=1);uncorrected_anchors=torch.cat((focal_anchors,base_anchors[:,1:]),dim=1)
        _,need_logits,assignment_logits,residual=self.head(social,mode);need_probability=torch.sigmoid(need_logits)
        sparse_responsibility=self._sparse_predicted_responsibility(assignment_logits)
        applied=need_probability[:,:,None,None,None]*sparse_responsibility[...,None,None]*residual
        corrected_relative=uncorrected_relative+torch.einsum("tp,bakpc->baktc",self.interpolation,applied);corrected_anchors=uncorrected_anchors+applied
        score_anchors=source_anchors+applied;mode_logits=self.score_head(torch.cat((mode,score_anchors.flatten(-2)),dim=-1)).squeeze(-1)
        last=past_loc[:,:,None,-1:,:];trajectories=last+corrected_relative;batch,agents,modes=mode_logits.shape
        selected=torch.arange(modes,device=mode_logits.device).view(1,1,modes);self.last_selected_indices=selected.expand(batch,agents,modes).detach()
        self.last_coarse_trajectories=(last+uncorrected_relative).detach();self.last_anchor_proposals=(past_loc[:,:,None,None,-1,:]+uncorrected_anchors).detach()
        self.last_corrected_anchors=(past_loc[:,:,None,None,-1,:]+corrected_anchors).detach()
        self.last_uncorrected_trajectories_graph=last+uncorrected_relative;self.last_uncorrected_anchors_graph=past_loc[:,:,None,None,-1,:]+uncorrected_anchors
        self.last_need_logits=need_logits;self.last_need_probability=need_probability.detach();self.last_assignment_logits=assignment_logits
        self.last_sparse_responsibility=sparse_responsibility.detach();self.last_predicted_residual=residual;self.last_applied_anchor_delta=applied
        self._training_target=None;return trajectories,mode_logits

def replace_decoder(model,cfg,experiment:str)->None:
    if experiment not in EXPERIMENTS:raise ValueError(experiment)
    checkpoint=Path(os.environ.get("FCRA_SOURCE_CHECKPOINT",""));cache_path=Path(os.environ.get("FCRA_NEED_CACHE",""))
    if not checkpoint.is_file() or not cache_path.is_file():raise FileNotFoundError((checkpoint,cache_path))
    metadata=json.loads(str(np.load(cache_path,allow_pickle=False)["metadata_json"].item()))
    if metadata.get("source_checkpoint_sha256")!=sha256(checkpoint):raise RuntimeError("Need cache/checkpoint fingerprint mismatch")
    base=model.agent_aware_trans_encoder
    if not bool(getattr(base,"use_temporal_social_cross",False)):raise ValueError("FCRA requires F16")
    base.social_gate=IR1.SocialDeltaTapGate(*list(base.social_gate.children()),dim=int(base.dim));base.register_forward_pre_hook(IR1.EncoderContextTap())
    model.decoder=FCRADecoder(model.token_dim,model.pred_len,int(base.dim),cache_path,3,.10)
    source=torch.load(checkpoint,map_location="cpu",weights_only=True);target=model.state_dict();compatible={k:v for k,v in source.items() if k in target and tuple(v.shape)==tuple(target[k].shape)}
    incompatible=model.load_state_dict(compatible,strict=False);prefixes=("decoder.head.",);expected={k for k in target if k.startswith(prefixes)}
    if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:raise RuntimeError((incompatible.missing_keys,incompatible.unexpected_keys))
    skipped=[k for k in source if k not in compatible]
    if skipped:raise RuntimeError(f"Skipped source keys: {skipped}")
    for p in model.parameters():p.requires_grad_(False)
    for p in model.agent_aware_trans_encoder.parameters():p.requires_grad_(True)
    for name,p in model.named_parameters():
        if name.startswith(prefixes):p.requires_grad_(True)
    model.decoder.source_checkpoint=str(checkpoint);model.decoder.need_cache=str(cache_path);model.decoder.cv_reference=False;cfg.MODEL.DECODER.NUM_MODES=20

