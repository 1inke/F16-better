"""CPU saved-output explorer. No optimizer, inference, raw/test read, or old edits."""
from pathlib import Path
import sys, json, csv, hashlib, importlib.util
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'CANDIDATE_SELECTION_VISUAL_AUDIT_20261003'
TRIAL = ROOT / 'GENERATION_PULSE_RESPONSE_SERIES_20261004'
KEYS = ['native'] + [f'{arm}{seed}' for seed in (1,2,3) for arm in ('fixed','learned')]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
def js(path,name,value):
    path.write_text(name+'='+json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+';\n',encoding='utf-8')
def csvout(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def transition(old,new):
    return np.where(~old&new,'rescued',np.where(old&~new,'lost',np.where(old&new,'retained','missing')))

def main():
    assert not (HERE/'audit.json').exists(), 'Preserve completed visual-review output; use another folder.'
    worker=read(TRIAL/'independent_worker_state.json')
    assert worker['status']==read(TRIAL/'results/exit_status.json')['status']=='SUCCESS'
    assert read(TRIAL/'results/saved_metrics_recompute.json')['status']=='PASS'
    # Snapshot actual inputs and both prior visualization/experiment folders. Not an all-history re-audit.
    source=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py'
    cachepath=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_dev.npz'
    encpath=ROOT/'GENERATION_NATIVE_DECODER_SERIES/dev_encoder.npz'
    watched=[source,cachepath,encpath]
    for folder in (OLD,TRIAL):
        watched.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp'))
    before={str(p):sha(p) for p in watched}
    print('INPUT_SHA_BEFORE',len(before),flush=True)
    common=module('visual_review_event_rules',source)
    arrays=load(OLD/'data/arrays.npz');cache=load(cachepath);encoder=load(encpath)
    ids=arrays['sample_id'];past=arrays['past'];gt=arrays['future'];scores=arrays['logits']
    assert len(ids)==len(np.unique(ids))==1516
    assert np.array_equal(cache['sample_id'],ids) and np.array_equal(encoder['sample_id'],ids)
    assert np.array_equal(arrays['native'],cache['candidate'][...,:24].reshape(1516,20,12,2))
    assert np.max(abs(past-encoder['past']))<1e-6
    assert np.array_equal(scores,cache['logits'])
    truth=common.events(encoder['past'],gt+encoder['past'][:,-1,None])
    clear=(truth['y']==1)&~truth['uncertain'];assert int(clear.sum())==469
    paths={'native':arrays['native']};pulse={};max_metric_error=0.
    for seed in (1,2,3):
        for arm,key in [('zero',f'fixed{seed}'),('mean_std',f'learned{seed}')]:
            saved=load(TRIAL/f'results/seed{seed}_{arm}_constrained.npz')
            pulse[key]=load(TRIAL/f'results/pulse/seed{seed}_{arm}_constrained_pulse.npz')
            assert np.array_equal(saved['sample_id'],ids) and np.array_equal(pulse[key]['sample_id'],ids)
            paths[key]=saved['trajectory'];assert paths[key].shape==(1516,20,12,2) and np.isfinite(paths[key]).all()
            error=np.linalg.norm(paths[key]-gt[:,None],axis=-1)
            for name,value in [('minade20',error.mean(-1).min(-1)),('minfde20',error[:,:,-1].min(-1))]:
                diff=float(abs(value-saved[name]).max());max_metric_error=max(max_metric_error,diff);assert diff<2e-6
    ade={};fde={};selected=scores.argmax(-1)
    for key,p in paths.items():
        e=np.linalg.norm(p-gt[:,None],axis=-1);ade[key]=e.mean(-1);fde[key]=e[:,:,-1]
    good={key:((ade[key]<=.2)&(fde[key]<=.4)).any(-1) for key in KEYS}
    assert int(good['native'].sum())==733 and int((good['native']&clear).sum())==92
    reports=read(TRIAL/'completion_review/REVIEW.json')
    counts=[];rows=[];case_flags={}
    for ta,tf in ((.2,.4),(.3,.5)):
        yes={key:((ade[key]<=ta)&(fde[key]<=tf)).any(-1) for key in KEYS}
        for seed in (1,2,3):
            new=yes[f'learned{seed}']
            for base in ('native',f'fixed{seed}'):
                changed=transition(yes[base],new)
                for group,mask in [('overall',np.ones(1516,bool)),('clear_event',clear)]:
                    entry=dict(ade_limit=ta,fde_limit=tf,seed=seed,reference=base,group=group,n=int(mask.sum()),
                        reference_good=int((mask&yes[base]).sum()),learned_good=int((mask&new).sum()),
                        rescued=int((mask&(changed=='rescued')).sum()),lost=int((mask&(changed=='lost')).sum()),
                        retained=int((mask&(changed=='retained')).sum()),missing=int((mask&(changed=='missing')).sum()))
                    counts.append(entry)
                    if base=='native':
                        original=next(x for x in reports['temporal_rows'] if x['file']==f'seed{seed}_mean_std_constrained.npz' and x['group']==group)
                        expected=original[f'adequacy_{ta}_{tf}']
                        assert [entry['learned_good'],entry['rescued'],entry['lost']]==[expected['count'],expected['rescued'],expected['lost']]
                if ta==.2:
                    case_flags[f'{base}_to_learned{seed}']=changed
                for i,id_ in enumerate(ids):
                    oldkey=base;newkey=f'learned{seed}'
                    oldslot=np.maximum(ade[oldkey][i]/ta,fde[oldkey][i]/tf).argmin()
                    newslot=np.maximum(ade[newkey][i]/ta,fde[newkey][i]/tf).argmin()
                    rows.append(dict(sample_id=int(id_),seed=seed,reference=base,clear_event=bool(clear[i]),
                        formal_group=['stable','transition','change'][int(arrays['change_group'][i])],
                        ade_limit=ta,fde_limit=tf,transition=changed[i],native_argmax_slot_1based=int(selected[i])+1,
                        old_closest_slot_1based=int(oldslot)+1,new_closest_slot_1based=int(newslot)+1,
                        old_samepath_ADE=float(ade[oldkey][i,oldslot]),old_samepath_FDE=float(fde[oldkey][i,oldslot]),
                        new_samepath_ADE=float(ade[newkey][i,newslot]),new_samepath_FDE=float(fde[newkey][i,newslot]),
                        old_minADE=float(ade[oldkey][i].min()),old_minFDE=float(fde[oldkey][i].min()),
                        new_minADE=float(ade[newkey][i].min()),new_minFDE=float(fde[newkey][i].min())))
    native_flags=np.stack([case_flags[f'native_to_learned{s}'] for s in (1,2,3)],-1)
    unions={}
    for group,mask in [('overall',np.ones(1516,bool)),('clear_event',clear)]:
        unions[group]={flag:dict(unique_IDs=int((mask&(native_flags==flag).any(-1)).sum()),
            all_three_seeds=int((mask&(native_flags==flag).all(-1)).sum()),
            ids=ids[mask&(native_flags==flag).any(-1)].tolist(),all_three_ids=ids[mask&(native_flags==flag).all(-1)].tolist())
            for flag in ('rescued','lost')}
    reps=[]
    for seed in (1,2,3):
        new=f'learned{seed}';delta=np.maximum(ade[new]/.2,fde[new]/.4).min(-1)-np.maximum(ade['native']/.2,fde['native']/.4).min(-1)
        flags=case_flags[f'native_to_learned{seed}']
        for flag in ('rescued','lost'):
            ix=np.where(clear&(flags==flag))[0]
            order=ix[np.lexsort((ids[ix],np.abs(delta[ix])))]
            if len(order):
                for rule,index in [('median_change',order[(len(order)-1)//2]),('largest_change',order[-1])]:
                    reps.append(dict(seed=seed,transition=flag,rule=rule,id=int(ids[index]),row=int(index)))
    (HERE/'data/cases').mkdir(parents=True,exist_ok=True)
    # Full precision metrics decide classes; rounded coordinates only draw the charts.
    meta=[]
    for i,id_ in enumerate(ids):
        record=dict(id=int(id_),row=i,clear=bool(clear[i]),uncertain=bool(truth['uncertain'][i]),
            group=int(arrays['change_group'][i]),event_kind=int(truth['kind'][i]),
            onset=int(truth['onset_step'][i]),ade={k:ade[k][i].tolist() for k in KEYS},fde={k:fde[k][i].tolist() for k in KEYS},
            selected=int(selected[i]))
        meta.append(record)
        display=dict(id=int(id_),past=np.round(past[i].astype(float),6),gt=np.round(gt[i].astype(float),6),
            scores=scores[i],neighbors=np.round(arrays['neighbor_past'][i,:int(arrays['num_valid'][i])-1].astype(float),6),
            paths={k:np.round(paths[k][i].astype(float),6) for k in KEYS},
            pulse={k:{name:pulse[k][name][i] for name in ('center','span','profile','drift','pulse_amplitude')} for k in pulse})
        js(HERE/f'data/cases/{id_}.js',f'window.PULSE_CASES[{int(id_)}]',display)
    common_data=dict(cases=meta,keys=KEYS,counts=counts,unions=unions,representatives=reps,
        event_labels=['无明确事件','转向','速度变化','启动','停止'],dt=.4,
        source_sha=before[str(cachepath)],default_id=reps[0]['id'],default_seed=reps[0]['seed'],default_filter='rescued')
    js(HERE/'data/common.js','window.PULSE_REVIEW',common_data)
    csvout(HERE/'case_classification.csv',rows);csvout(HERE/'counts.csv',counts)
    write(HERE/'summary.json',dict(counts=counts,unique_IDs=unions,representatives=reps))
    # Exact data for reproducible outside-the-browser inspection, preserving ALL slots.
    np.savez_compressed(HERE/'data/exact_arrays.npz',sample_id=ids,past=past,future=gt,logits=scores,
        clear_event=clear,event_kind=truth['kind'],event_onset=truth['onset_step'],**paths)
    checks=[];max_round_error=0.
    for key in KEYS:
        pp=np.round(paths[key].astype(float),6);gg=np.round(gt.astype(float),6)
        e=np.linalg.norm(pp-gg[:,None],axis=-1)
        err=max(float(abs(e.mean(-1)-ade[key]).max()),float(abs(e[:,:,-1]-fde[key]).max()))
        max_round_error=max(max_round_error,err);assert err<4e-6
        checks.append(dict(model=key,finite_coordinates=int(pp.size),max_web_metric_error=err))
    for name,digest in before.items():assert sha(Path(name))==digest, name
    write(HERE/'input_hashes.json',before)
    write(HERE/'audit.json',dict(status='PASS_SAVED_OUTPUT_VISUAL_REVIEW',IDs=1516,K=20,prediction_sources=7,
        all_6_constrained_outputs=True,exclusions=0,all_classes_match_completed_review=True,
        old_visualizer_native_exact_identity=True,saved_min_metric_max_error=max_metric_error,
        display_coordinate_decimals=6,web_metric_max_error=max_round_error,checks=checks,
        old_input_folder_files_SHA_before_after=len(before),old_SHA_scope='Old visualization + pulse experiment files and three directly-read prior inputs; not all-history manifest rerun',
        training=False,model_forward=False,optimizer=False,test_or_raw_data_read=False,
        selection='Native argmax unchanged. New paths inherit same slot and are NOT rescored.',
        classification='Exists SAME candidate ADE<=.2 AND FDE<=.4; .3/.5 sensitivity separately',
        oracle='Closest threshold path argmin max(ADE/threshold,FDE/threshold) is GT-assisted, not deployment'))
    text=['# 脉冲响应：救回与退化的逐例可视化','',
        '只读已完成的 constrained 输出：原F16 + 固定响应/学习响应各3种子。没有重新训练或推理，旧结果不改。',
        '主口径：同一候选 ADE≤0.2m、FDE≤0.4m。明确事件469例≠正式change组。','',
        '| 种子 | 参考 | 原本覆盖 | 新覆盖 | 救回 | 退化 |','|---|---|---:|---:|---:|---:|']
    for x in counts:
        if x['ade_limit']==.2 and x['group']=='clear_event':
            text.append(f"| {x['seed']} | {x['reference']} | {x['reference_good']} | {x['learned_good']} | {x['rescued']} | {x['lost']} |")
    text+=['',f"相对native，跨种子去重：明确事件救回{unions['clear_event']['rescued']['unique_IDs']}个ID，退化{unions['clear_event']['lost']['unique_IDs']}个ID。三个种子都救回{unions['clear_event']['rescued']['all_three_seeds']}例，都退化{unions['clear_event']['lost']['all_three_seeds']}例；不要把各seed计数相加当独立人数。",'',
        '默认列表包含该种子全部救回/退化案例，不是只选代表。代表入口按双阈值距离变化幅度的中位数/最大值确定；全1516 ID均可定位。',
        '深青线为事后“双阈值最近候选”：argmin max(ADE/.2,FDE/.4)，保证已有合格候选时展示一条合格候选。蓝色minADE与紫色minFDE可能不是同一条，不能拼接。',
        '橙线原F16是实际logit argmax。两个新方法橙线只是继承原slot，未对新轨迹重新评分；不能当新筛选器部署表现。',
        '误差/速度/加速度曲线固定同一slot：默认各方法各自的双阈值最近轨迹，也可切原生选择或联动slot。GT事件标注事后使用；中心/跨度不是GT事件时刻/持续时间。',
        '所有空间图共用范围，等比例米制，默认全20条不裁剪。可切换聚焦查看，但会明确提示远候选可能裁剪。',
        '人工备注按ID+seed保存在新命名空间，不回写实验；关闭前保存并导出JSON，内嵌下载不支持时可复制文本。',
        '阈值边界敏感；.3/.5仅独立敏感性口径。三种子/同人重叠时间窗/repeated ETH dev为探索性材料，不是SOTA或因果证明。',
        f'输入及两旧文件夹{len(before)}文件SHA前后不变；不是重新核验全历史19558文件。完成实验已有全历史审计保留。']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',counts=[x for x in counts if x['ade_limit']==.2 and x['group']=='clear_event'],unique=unions['clear_event'],reps=reps),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
