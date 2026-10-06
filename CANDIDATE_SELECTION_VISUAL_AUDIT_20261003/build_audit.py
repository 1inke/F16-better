"""Read-only CPU data audit and offline case browser build. No trainer imports."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, time
from collections import Counter
import numpy as np
from classification import diagnose, LABELS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse')
MANIFEST = ROOT / 'GENERATION_CANDIDATE_EXCHANGE_SERIES/old_results_manifest.json'
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_dev.npz'
MODELS = [
    ('native', '原生F16/V2：真实最高分选择', None),
    ('pairwise', 'Pairwise history · constrained seed1（继承slot参考）', 'GENERATION_PAIRWISE_HISTORY_SERIES'),
    ('readout', 'Neighbor attention · constrained seed1（继承slot参考）', 'GENERATION_NEIGHBOR_READOUT_SERIES'),
    ('messages', 'Neighbor messages · constrained seed1（继承slot参考）', 'GENERATION_NEIGHBOR_MESSAGE_SERIES'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def verify_prior(files, phase):
    started = time.time()
    for i, (path, expected) in enumerate(files.items()):
        assert sha(path) == expected, f'Old source/result changed: {path}'
        if i and i % 4000 == 0: print(f'{phase}: {i}/{len(files)} old SHA verified', flush=True)
    print(f'{phase}: all {len(files)} unchanged, {time.time()-started:.1f}s', flush=True)


def local_coordinates(coords):
    # Same row-vector rotation as locked focal-heading source; data scale is1.
    focal = coords[:, 0, :8]
    displacement = focal[:, -1] - focal[:, -4]
    speed = np.linalg.norm(displacement, axis=-1)
    heading = np.zeros_like(displacement); heading[:, 0] = 1
    active = speed > .03
    heading[active] = displacement[active] / speed[active, None]
    relative = coords - focal[:, -1, None, None]
    cosine, sine = heading[:, 0, None, None], heading[:, 1, None, None]
    return np.stack((relative[..., 0]*cosine+relative[..., 1]*sine,
                     -relative[..., 0]*sine+relative[..., 1]*cosine), -1).astype(np.float32)


def write_js(path, assignment, value):
    text = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())
    path.write_text(assignment+'='+text+';\n', encoding='utf-8')


def summarize(d, group, key, label):
    rows = []
    for name, mask in [('overall', np.ones(len(group),bool))] + [(g,group==i) for i,g in enumerate(['stable','transition','change'])]:
        count = int(mask.sum())
        for cat, title in enumerate(LABELS):
            cmask = mask & (d['category']==cat)
            rows.append(dict(model=key, model_label=label, group=name, class_id=cat, category=title,
                n=int(cmask.sum()), denominator=count, fraction=float(cmask.sum()/count),
                mean_selected_ade=float(d['ade_selected'][cmask].mean()) if cmask.any() else None,
                mean_selected_fde=float(d['fde_selected'][cmask].mean()) if cmask.any() else None,
                mean_ade_regret=float(d['ade_regret'][cmask].mean()) if cmask.any() else None))
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--rebuild',action='store_true'); args=parser.parse_args()
    if (HERE/'audit.json').exists() and not args.rebuild:
        raise RuntimeError('Completed output exists. Explicit --rebuild required, only derived files here are replaced.')
    started=time.time()
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    prior=json.loads(MANIFEST.read_text())['files']
    verify_prior(prior,'BEFORE')
    with np.load(CACHE,allow_pickle=False) as source: cache={k:source[k].copy() for k in source.files}
    ids=cache['sample_id']; n=len(ids); assert n==1516 and len(np.unique(ids))==n
    raw_data=RAW/'eth_data_train.npy'; raw_num=RAW/'eth_num_train.npy'
    raw=np.load(raw_data,mmap_mode='r'); num=np.load(raw_num,mmap_mode='r')
    assert ids.min()>=0 and ids.max()<len(raw)
    coordinates=np.asarray(raw[ids,:9,:20,-2:],dtype=np.float32)
    local=local_coordinates(coordinates)
    past,future=local[:,0,:8],local[:,0,8:20]
    past_error=float(abs(past.reshape(n,16)-cache['motion'][:,:16]).max())
    assert past_error<5e-5,past_error
    with np.load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/dev_encoder.npz') as enc:
        assert np.array_equal(enc['sample_id'],ids)
        encoder_past_error=float(abs(enc['past']-past).max()); assert encoder_past_error<5e-5
    candidates=cache['candidate'][...,:24].reshape(n,20,12,2)
    scores=cache['logits'].astype(np.float64); selected=scores.argmax(-1)
    err=np.linalg.norm(candidates.astype(np.float64)-future[:,None],axis=-1)
    ade_error=float(abs(err.mean(-1)-cache['ade']).max()); fde_error=float(abs(err[...,-1]-cache['fde']).max())
    assert ade_error<2e-5 and fde_error<2e-5,(ade_error,fde_error)
    valid=np.minimum(np.asarray(num[ids]).reshape(n),9).astype(int)
    assert valid.min()>=1 and valid.max()<=9
    (HERE/'data').mkdir(exist_ok=True)
    sources=[CACHE,raw_data,raw_num,ROOT/'GENERATION_NATIVE_DECODER_SERIES/dev_encoder.npz',MANIFEST,
        ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/formal_config.json',
        ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py',
        ROOT/'FRONT_ENCODER_SERIES/train_front_variant.py',
        ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',
        ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml']
    rows=[]; summaries=[]; sensitivity=[]; representatives=[]; metadata=[]; precise={}
    prediction_checks={}
    for key,label,folder in MODELS:
        print('Building '+key,flush=True)
        if folder:
            path=ROOT/folder/'results/seed1_mean_std_constrained.npz'; sources.append(path)
            with np.load(path,allow_pickle=False) as saved:
                assert np.array_equal(saved['sample_id'],ids)
                paths=saved['trajectory'].copy()
                assert paths.shape==(n,20,12,2) and np.isfinite(paths).all()
                d,ade,fde=diagnose(paths,future,scores)
                prediction_checks[key]=dict(minade_max_error=float(abs(d['minade']-saved['minade20']).max()),
                    minfde_max_error=float(abs(d['minfde']-saved['minfde20']).max()))
                assert max(prediction_checks[key].values())<2e-5
        else:
            paths=candidates.copy(); d,ade,fde=diagnose(paths,future,scores)
        assert np.array_equal(d['selected'],selected)
        precise[key]=paths
        model_rows=[]
        for i in range(n):
            item=dict(model=key,sample_id=int(ids[i]),row_index=i,group=['stable','transition','change'][int(cache['change_group'][i])],
                class_label=LABELS[int(d['category'][i])],selection_semantics='native_logits_argmax' if key=='native' else 'inherited_native_slot_reference_NOT_rescored')
            item.update({field: value[i].item() for field,value in d.items()})
            item.update(ade_limit_m=.3,fde_limit_m=.5)
            rows.append(item); model_rows.append(item)
        sub=summarize(d,cache['change_group'],key,label); summaries.extend(sub)
        metadata.append(dict(key=key,label=label,selection='真实原生选择' if key=='native' else '继承原生slot参考；未重新评分',
            selected_ade=float(d['ade_selected'].mean()),selected_fde=float(d['fde_selected'].mean()),
            minade=float(d['minade'].mean()),minfde=float(d['minfde'].mean()),counts=[int((d['category']==j).sum()) for j in range(6)],
            exact_ade_hits=int(d['selected_exact_ade_slot'].sum()),exact_fde_hits=int(d['selected_exact_fde_slot'].sum()),
            oracle_disagreement=int((~d['oracle_same_slot']).sum())))
        # Median joint-regret exemplar and maximum joint-regret stress case per category.
        for cat,title in enumerate(LABELS):
            indices=np.where(d['category']==cat)[0]
            if not len(indices): continue
            order=indices[np.lexsort((ids[indices],d['joint_regret'][indices]))]
            for rule,index in [('median_joint_regret',order[(len(order)-1)//2]),('max_joint_regret',order[-1])]:
                representatives.append(dict(model=key,category=cat,label=title,rule=rule,sample_id=int(ids[index]),row=int(index)))
        for al,fl in protocol['sensitivity_presets_m']:
            ds,_,_=diagnose(paths,future,scores,al,fl)
            for g,mask in [('overall',np.ones(n,bool))]+[(name,cache['change_group']==i) for i,name in enumerate(['stable','transition','change'])]:
                for cat,title in enumerate(LABELS):
                    sensitivity.append(dict(model=key,ade_limit=al,fde_limit=fl,group=g,category=title,n=int(((ds['category']==cat)&mask).sum()),denominator=int(mask.sum())))
        write_js(HERE/'data'/f'{key}.js',f'window.AUDIT_MODELS["{key}"]',dict(
            p=np.round(paths.astype(np.float64),5),a=ade,f=fde,d=model_rows))
    input_hashes={str(p):sha(p) for p in sources}
    common=dict(ids=ids,past=np.round(past.astype(float),5),gt=np.round(future.astype(float),5),
        neighbors=[np.round(local[i,1:valid[i],:8].astype(float),5).tolist() for i in range(n)],
        groups=cache['change_group'],scores=scores,models=metadata,labels=LABELS,
        data_sha=input_hashes[str(CACHE)],protocol=protocol)
    write_js(HERE/'data/common.js','window.AUDIT_COMMON',common)
    np.savez_compressed(HERE/'data/arrays.npz',sample_id=ids,past=past,future=future,logits=cache['logits'],
        change_group=cache['change_group'],neighbor_past=local[:,1:,:8],num_valid=valid,**precise)
    csv_write(HERE/'per_sample_classification.csv',rows)
    csv_write(HERE/'class_summary.csv',summaries)
    csv_write(HERE/'threshold_sensitivity.csv',sensitivity)
    json_write(HERE/'representatives.json',representatives)
    with gzip.open(HERE/'candidate_metrics.csv.gz','wt',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['model','sample_id','slot_1based','score_rank','native_logit_reference','ADE_m','FDE_m','joint_ADE_plus_halfFDE','is_selected_slot'])
        for key,_,_ in MODELS:
            _,ade,fde=diagnose(precise[key],future,scores)
            order=np.argsort(-scores,axis=1,kind='stable'); ranks=np.empty_like(order)
            np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,21),order.shape),axis=1)
            for i in range(n):
                for k in range(20): w.writerow([key,int(ids[i]),k+1,int(ranks[i,k]),scores[i,k],ade[i,k],fde[i,k],ade[i,k]+.5*fde[i,k],int(k==selected[i])])
    json_write(HERE/'summary.json',dict(models=metadata,groups=Counter(cache['change_group'].tolist()),classes=summaries,sensitivity=sensitivity,protocol=protocol))
    links=[]
    for x in representatives:
        rule='中位regret代表' if x['rule'].startswith('median') else '最大regret压力例'
        links.append(f'<li><a href="index.html?model={x["model"]}&id={x["sample_id"]}">{x["model"]} · {x["label"]} · {rule} · ID{x["sample_id"]}</a></li>')
    (HERE/'gallery.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><link rel="stylesheet" href="style.css"><title>分类案例入口</title><main><h1>分类代表案例</h1><p>全样本都在主浏览器。此页按各类联合regret中位数与最大值确定性抽取，不是人工挑选；空类别不造例。</p><a href="index.html">打开全样本浏览器</a><ul>'+''.join(links)+'</ul></main></html>',encoding='utf-8')
    primary=metadata[0]
    report=['# 候选与选择：全量人工诊断报告','',
        '本报告为事后描述，不训练、不改变任何实验门槛；全1516样本/20候选纳入，官方test未读取。',
        '主视图是当前生成研究的原生F16/V2，而不是旧EXP-173缓存；最近三项seed1生成修正仅作继承slot的参考比较，不能称为校正后重新评分。','',
        f'原生所选ADE/FDE：{primary["selected_ade"]:.6f}/{primary["selected_fde"]:.6f}m；minADE/minFDE：{primary["minade"]:.6f}/{primary["minfde"]:.6f}m。',
        f'精确minADE slot选中{primary["exact_ade_hits"]}/{n}，精确minFDE slot选中{primary["exact_fde_hits"]}/{n}；两个oracle slot不同{primary["oracle_disagreement"]}/{n}。','',
        '## 默认分类：同一条轨迹ADE≤0.30m且FDE≤0.50m','',
        '| 类别 | 数量 | 全体比例 |','|---|---:|---:|']
    report += [f'| {title} | {count} | {count/n:.2%} |' for title,count in zip(LABELS,primary['counts'])]
    support=sum(primary['counts'][:2]);miss=primary['counts'][1]
    report += ['',f'有同路径合格候选：{support}/{n}；其中漏选{miss}/{support}（{miss/support:.2%}）。这是给定候选与阈值下的潜在选择空间，不是可学习性或排序根因的证明。','',
        'ADE regret可以精确分解为selectedADE−minADE；FDE同理，但两个min可能来自不同候选。联合Q=ADE+0.5FDE必须使用同一候选。多指标类别是阈值依赖的描述，不是天然因果机制。','',
        '## 阈值敏感性（原生整体）','', '| ADE/FDE口径(m) | 选中合格 | 合格漏选 | 分离最优 | 仅路径 | 仅终点 | 覆盖不足 |','|---|---:|---:|---:|---:|---:|---:|']
    for al,fl in protocol['sensitivity_presets_m']:
        counts=[x['n'] for x in sensitivity if x['model']=='native' and x['group']=='overall' and x['ade_limit']==al and x['fde_limit']==fl]
        report.append(f'| {al:.2f}/{fl:.2f} | '+ ' | '.join(map(str,counts))+' |')
    report += ['', '## 给人工观察的建议','',
        '- 先看“合格候选被漏选”：是否明显转向、减速、停下的GT已经在集合里，但分数仍偏向惯性延续？只能形成后续可检验假设。',
        '- 看“分离最优”：路径贴合与终点贴合是否发生在不同轨迹；不要用两条oracle拼成不存在的好路径。',
        '- 看“整体覆盖不足”：区分方向、速度/进度、转向时机与几何形状差异；备注可由人工自由填写，不预先猜根因。',
        '- “选中合格”也查看曲线与运动学；小误差不自动代表自然轨迹。',
        '- 用三档口径和全量浏览抵抗阈值/代表案例选择偏差，避免只看最漂亮或最糟糕的案例。','',
        '## 完整性与局限','',
        '坐标为以最后观察点为原点、平滑历史方向为x轴的米制坐标；两空间图等比例且共用范围。邻居仅显示有效观察历史。网页位置显示舍入5位，分类计算使用完整精度；精确数组另存。',
        '运动组是GT事后标签；原生logits的softmax质量未校准为概率。一个已发生GT不否定其他合理未来。重复ETHdev与单生成器seed不能支持独立泛化或显著性结论。本诊断不计入正式实验轮数。',
        '旧文件全部SHA前后验证，源缓存/GT索引/模型比较均在audit.json中记录；该报告和所有新文件仅在新目录。']
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    verify_prior(prior,'AFTER')
    script_hashes={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.html','.css','.js')}
    audit=dict(status='PASS_POSTHOC_VISUAL_DIAGNOSTIC_ONLY',n=n,K=20,models=4,sample_rows=len(rows),candidate_rows=n*20*4,
        exclusions=0,explicit_ids_unique=True,all_ids_aligned=True,prior_SHA_before=len(prior),prior_SHA_after=len(prior),
        input_SHA_before_equal_after={str(p):sha(p)==expected for p,expected in input_hashes.items()},sources=input_hashes,
        past_alignment_max_error=past_error,encoder_past_alignment_max_error=encoder_past_error,
        native_ADE_recompute_max_error=ade_error,native_FDE_recompute_max_error=fde_error,comparison_recompute=prediction_checks,
        selected_is_native_argmax=True,GT_used_for_selection=False,GT_used_for_posthoc_only=True,
        optimizer_steps=0,GPU_used=False,test_access=False,source_edits=False,
        output_scripts_SHA=script_hashes,seconds=time.time()-started)
    assert all(audit['input_SHA_before_equal_after'].values())
    json_write(HERE/'audit.json',audit)
    print(json.dumps(dict(status=audit['status'],seconds=audit['seconds'],native=primary),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
