"""Native mode features retained both arms; six native-anchor inputs or zeros.
Native/GT prediction selection never used as a forward feature.
"""
from pathlib import Path
import importlib.util,sys,json
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('feedback_direct_source',HERE.parent/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
n=h.n;c=h.c;m=h.m;r=h.r;p=h.p
h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
Parent=h.DirectHead
class FeedbackHead(Parent):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.head=nn.Sequential(nn.Linear(134,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*n.TOKEN_DIM].reshape(-1,1,8,n.TOKEN_DIM)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
    def forward(self,observation,base,speed):
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        # Coordinate feedback is derived from the exact original candidate, no logits/GT.
        denominator=(speed[:,0,None,None,None]+.03)*base.new_tensor([4.,8.,12.])[None,None,:,None]
        geometry=(base[:,:,[3,7,11]]/denominator).flatten(2)
        if self.arm=='zero':geometry=torch.zeros_like(geometry)
        feature=torch.cat((self.inputs(observation),geometry),-1)
        anchors=self.head(feature).reshape(-1,20,3,2)*(speed[:,0,None,None,None]+.03)
        delta=torch.einsum('tp,bkpc->bktc',self.decoder.interpolation,anchors)
        velocity=torch.diff(delta,dim=2,prepend=torch.zeros_like(base[:,:,:1]))
        residual=velocity.clamp(-self.strength*(speed[:,0,None,None,None]+.03),self.strength*(speed[:,0,None,None,None]+.03))
        return base+residual.cumsum(2),residual
h.DirectHead=FeedbackHead;n.NativeGenerator=FeedbackHead;c.Correction=FeedbackHead;r.m.Correction=FeedbackHead
DIRECT_PREFLIGHT=h.preflight
def preflight(device):
    DIRECT_PREFLIGHT(device)
    audit=json.loads((HERE/'preflight.json').read_text())
    assert audit['parameters_trainable']==dict(zero=25926,mean_std=25926)
    paths=[Path(__file__),HERE.parent/'GENERATION_DIRECT_ANCHOR_SERIES/results/model_lock.json',HERE.parent/'GENERATION_DIRECT_ANCHOR_SERIES/results/summary.json']
    prior=json.loads(paths[1].read_text())['models']
    for path,digest in prior.items():assert p.sha256(Path(path))==digest
    audit['source_hashes'].update({str(path):p.sha256(path) for path in paths});audit['source_hashes'].update(prior)
    audit['scope']='both original context/native mode features; geometryfeedback vszeros6; nativecoordinates not GT/logits; sameparams not equaleffectivecapacity'
    m.write_json(HERE/'preflight.json',audit)
    print('PREFLIGHT PASS anchorfeedback/native-mode-retained/pairedinit/identity/source',flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch new direct3anchor correction branch trainings. BOTH originalcontext64+native-mode-feature64 retained, plus6 zero/control or originalnative3anchorcoordinates/(observedspeed+.03)/stepindex treatment. Identical25926parameterMLP/fullpairedinit/nativeoutputidentity. Alloriginaldecoder/F16/scorer frozen, K20/commonnativevelocityclip/originalloss/cal. Equalparamcount not equaleffectivecapacityproof. Candidategeometryfeedback, not independentgeneration/intent/ranking. GT/group/logits not forward; nativeepoch0fallback explicit; repeatedETHdev exploratory, no test.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'ANCHOR_FEEDBACK_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Anchor geometry feedback\n\n'+json.dumps(summary,indent=2),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
