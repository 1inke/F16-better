"""Read existing FIT/CAL histories and checkpoint deltas only, never dev/test."""
from pathlib import Path
import importlib.util,sys,json,csv,math
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_UPSTREAM_TEMPORAL_SERIES'
def main():
    spec=importlib.util.spec_from_file_location('drift_upstream',SOURCE/'run_head.py')
    a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
    a.prepare_backend(torch.device('cpu'))
    reference=a.ORIGINAL_STATE;prefix='original.'+a.PREFIX
    result=dict(status='PASS_DESCRIPTIVE_ONLY',dev_access=False,test_access=False,optimizer_steps=0,seeds=[],source_hashes={})
    result['source_hashes'][str(SOURCE/'run_head.py')]=a.p.sha256(SOURCE/'run_head.py')
    for seed in (1,2,3):
        root=SOURCE/'results/models/mean_std'/f'seed_{seed}'/'balanced_protected'
        history=list(csv.DictReader((root/'history.csv').open(encoding='utf-8-sig')))
        assert len(history)==80
        completion=json.loads((root/'complete.json').read_text())
        selected=completion['best_feasible_epoch'];early=history[selected-1];late=history[-1]
        entry=dict(seed=seed,selected_epoch=selected,history_source='FIT loss and CAL metrics only',early={k:early[k] for k in ('train_loss','cal_feasible','cal_change_key','transition_minade20','transition_minfde20')},final={k:late[k] for k in ('train_loss','cal_feasible','cal_change_key','transition_minade20','transition_minfde20')},drift={})
        for tag in ('constrained','final'):
            path=root/(tag+'.pth');digest=a.p.sha256(path);assert completion['model_hashes'][str(path)]==digest
            result['source_hashes'][str(path)]=digest
            saved=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];groups={}
            for name,value in saved.items():
                if not name.startswith(prefix):continue
                raw=name[len('original.'):];suffix=name[len(prefix):]
                group='attention' if suffix.startswith('self_attn.') else ('ffn' if suffix.startswith(('linear1.','linear2.')) else 'layernorm')
                groupstats=groups.setdefault(group,dict(parameter_elements=0,reference_squared_norm=0.,update_squared_norm=0.))
                groupstats['parameter_elements']+=value.numel()
                groupstats['reference_squared_norm']+=float(reference[raw].double().square().sum())
                groupstats['update_squared_norm']+=float((value.double()-reference[raw].double()).square().sum())
            for value in groups.values():value['relative_update_norm']=math.sqrt(value['update_squared_norm']/max(value['reference_squared_norm'],1e-30))
            entry['drift'][tag]=groups
        result['seeds'].append(entry)
    result['interpretation']='Weight update norms and FIT/CAL trends describe drift, do not identify causal harmful block, useful new information, or benefit of attention-only training. No dev-driven hyperparameters.'
    (HERE/'DRIFT_DIAGNOSIS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
