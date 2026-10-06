"""Bounded saved-output completion interpretation. No model forward or training."""
import collections, hashlib, json, pathlib, datetime, time
import numpy as np
import psutil
HERE=pathlib.Path(__file__).resolve().parent.parent
OUT=pathlib.Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(name,value):
    p=OUT/name
    assert not p.exists(),('Preserve previous report',p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pct(a,b): return None if b==0 else 100*a/b
assert psutil.virtual_memory().available >= 6*1024**3
started=time.time()
r=read(HERE/'RESULT.json')
audit=read(HERE/'ARITHMETIC_AUDIT.json')
worker=read(HERE/'independent_worker_state.json')
exit_state=read(HERE/'exit_status.json')
resource=read(HERE/'RESOURCE_USAGE.json')
manifest=read(OUT/'claim_intent_manifest.json')
assert r['status']=='PASS_FROZEN_SAVED_DATA_CONCORDANCE'
assert audit['status']=='PASS' and worker['status']=='SUCCESS' and exit_state['status']=='SUCCESS'
assert audit['all_old_SHA_unchanged']==21780 and audit['source_SHA_files']==278
assert len(r['source_hashes'])==278 and len(r['all_summaries'])==9
checks={}
rows={}
names=list(r['all_summaries'])
for name in names:
    done=read(HERE/'jobs'/f'{name}.done.json')
    assert done['status']=='SUCCESS' and done['source_hashes']==r['source_hashes']
    for path,digest in done['outputs'].items():
        p=pathlib.Path(path)
        assert p.resolve().is_relative_to(HERE.resolve())
        assert sha(p)==digest,('New job output SHA',path)
        checks[str(p)]=digest
    ss=read(HERE/'jobs'/f'{name}.summary.json')
    assert ss==r['all_summaries'][name]
    # All numeric path counts are independently recomputed from saved per-case arrays.
    with np.load(HERE/'jobs'/f'{name}.npz',allow_pickle=False) as z:
        n=len(z['sample_id'])
        assert len(np.unique(z['sample_id']))==n
        masks={'all':np.ones(n,dtype=bool),
               'clear':z['GT_event'] & ~z['GT_uncertain'],
               'specific_compatible_clear':z['GT_event'] & ~z['GT_uncertain'] & z['cue_first_compatible'],
               'explicit_cue_clear':z['GT_event'] & ~z['GT_uncertain'] & z['cue_any']}
        selected=z['selected_slot'].astype(int)
        good=(z['ade']<=.2)&(z['fde']<=.4)
        assert np.array_equal(good.sum(1),z['raw_good_count'][:,0])
        rows[name]={}
        for g in ss:
            if g['group'] not in masks: continue
            m=masks[g['group']]
            assert g['n']==int(m.sum())
            if not g['n']: continue
            q=g['quality_and_dedup'][0]
            assert g['candidate_failure']==int((m&~good.any(1)).sum())
            assert q['coverage']==int((m&good.any(1)).sum())
            assert q['raw_good_distribution']==np.bincount(good.sum(1)[m],minlength=21).tolist()
            picked=good[np.arange(n),selected]
            assert g['selected_good']==int((m&picked).sum())
            assert g['good_but_selection_miss']==int((m&good.any(1)&~picked).sum())
            assert q['effective_good_distribution'][0]==np.bincount(z['effective_good_count'][m,0,0],minlength=21).tolist()
            coarse=g['response_gap_counts']['coarse_response_covered_path_inadequate']
            assert coarse==int((m&(z['class_0']==7)).sum())
            assert coarse==g['alternate_gap_counts']['coarse_response_covered_path_inadequate']
            rows[name][g['group']]={
                'n':g['n'],'no_good':g['candidate_failure'],'good_miss':g['good_but_selection_miss'],
                'selected_good':g['selected_good'],'coverage':q['coverage'],
                'coverage_pct':pct(q['coverage'],g['n']),
                'coarse_path_wrong':coarse,'coarse_among_generation_failures_pct':pct(coarse,g['candidate_failure']),
                'gap':g['response_gap_counts'],'alternate_gap':g['alternate_gap_counts'],
                'first_compatible':g['first_cue_compatible'],'any_compatible':g['any_cue_compatible'],
                'primary_quality_dedup':q,'sensitivity_quality_dedup':g['quality_and_dedup'][1],
                'verified_track_count':g['verified_track_count'],'ambiguous_track_windows':g['ambiguous_track_windows']}
    if psutil.Process().memory_info().rss>6*1024**3: raise RuntimeError('Process memory budget')
assert len(checks)==27
source_checks={}
for path,digest in r['source_hashes'].items():
    assert sha(pathlib.Path(path))==digest,('Completion source',path)
    source_checks[path]=digest
cases=read(HERE/'DEV_274_CASES.json')
c=cases['cases']
assert len(c)==274 and len({x['sample_id'] for x in c})==274
assert sum(not x['generation_good'] for x in c)==228
assert len(cases['scoring_pool_IDs'])==41
cuecounts=dict(collections.Counter(x['cue_class'] for x in c))
failcue=dict(collections.Counter(x['cue_class'] for x in c if not x['generation_good']))
scenes={}
for j in ['FIT_native','CAL_native']:
    scenes[j]=[{k:g.get(k) for k in ['group','n','candidate_failure','good_but_selection_miss','selected_good','response_gap_counts']}
               for g in r['all_summaries'][j] if g['group'].startswith(('scene_','compatible_scene_','compatible_GTkind'))]
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
out={
 'status':'PASS_SAVED_OUTPUT_INTERPRETATION_AUDIT',
 'at':at,'source_SHA_current_reverified':len(source_checks),
 'new_job_output_SHA_reverified':len(checks),'scientific_done_jobs_verified':9,
 'completion_worker_all_old_SHA_verified':21780,'all_old_SHA_rerun_in_this_supplement':False,
 'all_old_weights_exact_evidence':'No model forward, optimizer or tensor mutation; immutable checkpoint/cache/source SHA. Not a new tensor-by-tensor model replay.',
 'model_forward':False,'optimizer':False,'formal_count_increment':0,
 'all_rows_primary_counts_exact':True,'samepath_threshold':[.2,.4],
 'resource':resource,'native_and_pulse_descriptive':rows,
 'FIT_CAL_scene_kind_counterevidence':scenes,
 'DEV274':{'n':274,'cue_categories':cuecounts,'generation_failure_cue_categories':failcue,
            'first_compatible':sum(x['first_type_direction_compatible'] for x in c),
            'any_compatible':sum(x['any_event_compatible'] for x in c),
            'generation_failure':228,'selection_miss':41,'selected_good':5},
 'claim_manifest_SHA':sha(OUT/'claim_intent_manifest.json'),
 'direct_input_SHA':{str(HERE/x):sha(HERE/x) for x in ['RESULT.json','ARITHMETIC_AUDIT.json','RESOURCE_USAGE.json','DEV_274_CASES.json','independent_worker_state.json','exit_status.json']},
 'verified_output_SHA':checks,'elapsed_seconds':time.time()-started,
 'peak_RSS_bytes':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
 'limits':['Windows and tracks overlap; not independent subjects.','Only three unequal scene families; CAL only university.',
           'Native F16 has previously seen these populations; CAL/DEV repeated exploratory use.',
           'Cue compatibility and candidate event tests are rule proxies, not causal model use or human intent.',
           'Future-completeness neighbor eligibility remains biased.',
           'No measured unpredictability fraction or new p-values/confidence interval.']}
save('REVIEW.json',out)
save('../RESULT_ANALYSIS.json',{k:v for k,v in out.items() if k not in ['verified_output_SHA','direct_input_SHA']})
fit=rows['FIT_native']['specific_compatible_clear']
cal=rows['CAL_native']['specific_compatible_clear']
dev=rows['DEV_native']['specific_compatible_clear']
elapsed=(datetime.datetime.fromisoformat(worker['finished_at'])-datetime.datetime.fromisoformat(worker['started_at'])).total_seconds()/60
peak_private=max(resource['peak_private_bytes'],resource['final_memory']['lifetime_peak_private_bytes'])
peak_rss=max(resource['peak_rss_bytes'],resource['final_memory']['lifetime_peak_rss_bytes'])
text=f"""# 线索—响应一致性诊断：内存恢复完成后的解释

完成时间 {worker['finished_at']}；正式响应研究仍 **2/3**。这是冻结保存数据诊断，不是第三次正式训练。

## 结论

当前证据优先指向“粗运动响应已有覆盖，但完整路径仍不够准确”，不优先支持再增加转向/速度类别或继续扫脉冲参数。下一步应先检查现有受限曲线的可达性，再区分学习与泛化缺口；本报告不启动新训练。

研究分析与统计审查流程要求同一候选通过整条路径、先报告分母和反证，再提出设计，不能把风险可读或某个候选恰巧转对方向当作意图理解。

## 1. 工程恢复与完整审计

- 新 worker 与 child exit 双 SUCCESS，9/9科学 jobs 和27个输出SHA重新核验；完成 worker 核验全部21780旧文件不变，本补充没有再次全扫这些旧文件。278登记源/输入SHA在解释补充中再次核验。
- 全部保存算术审计 PASS，最大误差0。补充逐job从NPZ重算原始好候选、有效好簇、覆盖和选择数量，全部一致；没有重放模型。
- 实际运行 {elapsed:.3f} 分钟。OS进程终身峰值RSS {peak_rss/1024**3:.3f}GiB，私有提交 {peak_private/1024**3:.3f}GiB；采样最小系统可用物理内存 {resource['minimum_system_available_bytes']/1024**3:.2f}GiB。
- 256行分块把单个最大成对数组从901MiB降至8.91MiB。原科学定义/float64/ID/旧失败来源均保持。旧OS分配失败完整原因未被证明。
- 后续持续按约17GB可用RAM估算输入、输出、临时量，并实测RSS/private与系统physical/commit；本方案6GiB进程预算且系统双余量至少6GiB，不占满内存，不调pagefile、不关闭用户程序。

证据：[完整算术审计](ARITHMETIC_AUDIT.json)、[资源记录](RESOURCE_USAGE.json)、[补充核验](completion_review/REVIEW.json)。

## 2. 最关键的收缩：规则线索并不等于正确变化线索

原DEV274个“有规则线索且未来清晰变化”窗口中，只有64个规则目标线索与GT第一变化的类型和方向相容；另4个只与更晚事件相容。94个仅邻居接近的泛风险，111个目标线索不相容，1个方向不清。这不是说后者不可预测，而是不能继续把全部274当作正确预兆。

原274中的228个无好候选、41个已有好候选未选中、5个选中好候选仍原样保留；不删难样例，不改正式评估。相容子组DEV64有55个无好候选、8个有好候选未选中、1个选中。274是重叠窗口而非274独立人。

证据：[逐例274分类](DEV_274_CASES.json)、RESULT.all_summaries.DEV_native；完整分层见 completion_review/REVIEW.json。

## 3. FIT/CAL的主要证据：同一候选粗响应相符，完整路径不达标

主质量定义固定：同一完整候选 ADE<=0.2m AND FDE<=0.4m；不能拼minADE与minFDE不同slot。

| 第一事件类型/方向相容窗口 | FIT | CAL | DEV（仅描述） |
|---|---:|---:|---:|
| 全部相容窗口 | {fit['n']} | {cal['n']} | {dev['n']} |
| 无合格完整候选 | {fit['no_good']} | {cal['no_good']} | {dev['no_good']} |
| 其中：同一候选粗响应通过但路径仍差 | {fit['coarse_path_wrong']} | {cal['coarse_path_wrong']} | {dev['coarse_path_wrong']} |
| 占本组生成失败比例 | {fit['coarse_among_generation_failures_pct']:.2f}% | {cal['coarse_among_generation_failures_pct']:.2f}% | {dev['coarse_among_generation_failures_pct']:.2f}% |
| 有好候选却未选中 | {fit['good_miss']} | {cal['good_miss']} | {dev['good_miss']} |
| 原argmax选中好候选 | {fit['selected_good']} | {cal['selected_good']} | {dev['selected_good']} |

粗响应包含事件、类型、方向、时刻±2步及强度；这一组每个窗口至少有一条同candidate通过它们，但没有一条完整路径合格。它不是所有候选都正确，也不是证明网络有益使用了线索。±2步=±0.8秒且标签粗糙，可能掩盖很重要的形状/时间误差。

生成失败的其余部分：FIT onset143/direction8/strength64，CAL16/1/12；交换时刻和方向审查顺序后FIT89/62/64、CAL12/5/12。单项归属会变化，但“粗响应通过、路径不合格”576/77完全不变，因此大方向较稳，不应过度定位单一clock或direction模块。

全clear变化窗口的生成失败中，FIT5275/7085=74.45%、CAL731/1007=72.59%属于粗响应相符而path差；这与小相容子组一致，但不能跳过线索相容的限定。

证据：RESULT.all_summaries.FIT_native / CAL_native 的specific_compatible_clear、clear；逐例NPZ和双顺序审计。

## 4. 去重后的有效好候选：先补“从0到1”

相容组FIT207/998有好候选，其中199只有一个有效好簇，8个至少两个；CAL31/137有好候选，28个仅一个、3个至少两个；DEV9/64均仅一个。这些组原始好数量和主阈值有效好簇几乎/完全相同，不能说这里的主要失败是重复候选挤占名额。

主去重使用全K20、不看GT的complete-link，平均路径0.10m/终点0.20m；簇含>=1合格路径才算有效好簇。更宽/更窄去重阈值均保留。其他尤其近静止子群确实有重复，不能用相容组否定人眼对全部候选的重复观察；单一GT也不能判定其他模态不合理。

FIT总体覆盖14867/25912=57.38%，CAL1490/2879=51.75%；相容变化组覆盖仅20.74%/22.63%。因此本阶段应优先新增至少一个高质量完整路径，再考虑有意义的第二条，评分放后。

0.3/0.5敏感性相容组FIT466/998、CAL66/137、DEV20/64，不代替0.2/0.4达标标准。

## 5. 脉冲轮反证：粗类别变化不等于质量突破

DEV274原生覆盖46，固定脉冲三seed45/42/41，学习脉冲44/43/43；相容64原生9，固定9/10/9，学习10/10/10。整体1516原生733，学习723/728/727。每seed重复同一批ID，不加成独立样本，也不能挑64子组宣布成功。正式脉冲major与运动学仍失败。

## 6. 反方检查2：证据之后（同一助手inline角色，非外审）

- **Major：多候选事后挑选很容易覆盖粗标签。** event/type几乎总有匹配，不能证明条件化推理或网络有效捕获线索。解决：全部结论限定为候选集合的事后描述；下一诊断必须保留原mode级别/同候选路径，而非换slot拼响应。
- **Major：大比例粗响应正确可能只是阈值宽/参考点不同。**过去cue和未来GT事件参考点不同、±.8秒容差及单一dominant event可能漏多次变化。解决：同时保存逐时误差和双顺序分项、GT辅助可达性不作为模型收益。
- **Major：CAL仅university，相容转向/速度数量114/23，start/stop为0。** 不推广到三场景/所有行为；FIT相容hotel0/university878/zara120，方向需要跨场景反证。不给可靠groupCI/p值。
- **Counterevidence：先前三anchorGT构造已可达；free12increment/time-query失败；tinyFIT好CAL差；pulse覆盖下降。**更多自由度或换decoder未被证明一定有用，不忽略上述证据。

结论审查状态：限定描述与下一诊断建议PASS；任何“解码器唯一根因/已理解意图/不可预测率”FAIL。

## 7. 反方检查3：最终最强反驳（同一助手inline角色，非外审）

最强反驳是：即使GT可由某参数化表达，观察信息不足、损失约束、优化和泛化仍可导致学不到；即使本次多次局部oracle拟合失败，也不能证明函数族不可达。此报告不把两者强行二分为唯一原因。

还可能有：GT位置噪声/状态规则误分、输入邻居资格偏差、原F16已见过数据、轨迹/窗口相关、重复DEV研究、按GT子组挑结果。全部保留原评估，不作SOTA/独立泛化/人类意图结论。对单项onset/direction排序敏感已经显式展示。

## 8. 下一步与规划节点

已保存 [下一诊断设计](completion_review/NEXT_DESIGN.md)：FIT/CAL-only、GT辅助的“受限曲线表达可达性 vs 学习缺口”，固定原native20+原bound，不训练新head/不改损失/不解冻。只检查现有曲线族能不能在这些窗口表达至少一条完整好候选，以及与已学输出的差距。

目前是设计建议，不把GT拟合当部署模型，不自动开第三正式训练。下一推进先20–35分钟完成实现可行性、预算与小规模压力/算术预检，然后报实测运行ETA和唯一50%节点；诊断保守暂估15–40分钟，未经实测不当启动承诺。若求解无法辨识/内存不安全/需新科学权限，按人类要求停止等决断，不降低标准。

来源均为本地锁定原始输出，而非文献综述；manifest先于本次解释生成，源/断言/局限保存在claim_intent_manifest和REVIEW。所有原结果、科学源码和模型保持不变。
"""
dest=HERE/'RESULT_ANALYSIS.md'
assert not dest.exists()
dest.write_text(text,encoding='utf-8')
save('REPORT_OUTPUT_SHA.json',{str(dest):sha(dest),str(OUT/'REVIEW.json'):sha(OUT/'REVIEW.json'),str(HERE/'RESULT_ANALYSIS.json'):sha(HERE/'RESULT_ANALYSIS.json')})
print(json.dumps({'status':out['status'],'jobs':9,'outputs_SHA':27,'source_SHA':278,'elapsed_seconds':out['elapsed_seconds'],'peak_RSS_bytes':out['peak_RSS_bytes'],'FIT':fit['coarse_among_generation_failures_pct'],'CAL':cal['coarse_among_generation_failures_pct']},ensure_ascii=False))
