"""Read-only source/result integrity and actual F16 module inspection."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('upstream_inspect_source',ROOT/'GENERATION_NATIVE_DECODER_SERIES/run_native.py')
n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    destination=HERE/'old_results_manifest.json'
    if not destination.exists():
        files={str(path):digest(path) for folder in ROOT.glob('GENERATION_*_SERIES') if folder!=HERE for path in (folder/'results').rglob('*') if path.is_file()}
        destination.write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')
    manifest=json.loads(destination.read_text())['files']
    for path,sha in manifest.items():assert digest(Path(path))==sha,path
    cfg,trainer,model=n.prepare_backend(torch.device('cuda:0'))
    block=model.agent_aware_trans_encoder.temporal_encoder
    assert isinstance(block,torch.nn.TransformerEncoder) and len(block.layers)==1
    allowed=[name for name,_ in model.named_parameters() if name.startswith('agent_aware_trans_encoder.temporal_encoder.layers.0.')]
    assert allowed
    output=dict(status='PASS_INSPECTION_ONLY',old_result_files=len(manifest),allowed_names=allowed,allowed_parameters=sum(p.numel() for name,p in model.named_parameters() if name in allowed),temporal_layers=len(block.layers),token_dim=model.token_dim,agents=model.num_pedestrians,module_path='agent_aware_trans_encoder.temporal_encoder.layers.0',original_checkpoint=str(n.f.fold_run('eth','v2')/'best_selected.pth'),checkpoint_sha=digest(n.f.fold_run('eth','v2')/'best_selected.pth'),original_all_frozen=not any(p.requires_grad for p in model.parameters()))
    (HERE/'backend_inspection.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
