"""Approved observable-cue diagnosis, independently executable and stage-resumable."""
from common import *
import native, population, probes
import torch

def report():
    binary=read('binary_probes.json');candidate=read('candidate_audit.json');metadata=read('metadata_audit.json')
    rows=[]
    for key,record in binary['outputs'].items():
        row=dict(reader=key,**record['metrics']);rows.append(row)
    contrasts=[]
    for mode in ('full','common64'):
        for a,b in (('R2','R1'),('E2','E1'),('E3','E2_motion')):
            lhs=binary['outputs'][f'OOF_logistic_{mode}_{a}']['metrics'];rhs=binary['outputs'][f'OOF_logistic_{mode}_{b}']['metrics']
            contrasts.append(dict(mode=mode,contrast=f'{a}-{b}',AUPRC_delta=None if lhs['AUPRC'] is None or rhs['AUPRC'] is None else lhs['AUPRC']-rhs['AUPRC'],
                            Brier_delta=lhs['Brier']-rhs['Brier'],log_loss_delta=lhs['log_loss']-rhs['log_loss'],
                            CI=None,Holm_p=None,evidence='Exploratory scene-transfer readout contrast, not causal information-loss test'))
    risk=npz(HERE/'dev_risk_readouts.npz');audit=npz(HERE/'candidate_audit.npz')
    assert np.array_equal(risk['sample_id'],audit['sample_id'])
    tags=[]
    for i,identifier in enumerate(audit['sample_id']):
        labels=[]
        if audit['GT_uncertain'][i]: labels.append('event_boundary_uncertain')
        if not audit['coverage_0.2_0.4'][i]:labels.append('adequate_whole_path_candidate_not_present')
        elif not audit['selected_good_0.2_0.4'][i]:labels.append('adequate_candidate_present_but_native_selection_missed')
        else:labels.append('native_selected_adequate')
        if audit['GT_event'][i] and not audit['event_type_time_coverage'][i]:labels.append('operational_event_type_timing_not_covered')
        if audit['near_static'][i]:labels.append('near_static_GT_path')
        tags.append(dict(sample_id=int(identifier),descriptive_tags=labels,
            observed_risk_readout_R2=float(risk['final_logistic_full_R2'][i]),
            encoded_risk_readout_E2_motion=float(risk['final_logistic_full_E2_motion'][i]),
            decoder_risk_readout_E3=float(risk['final_logistic_full_E3'][i]),
            no_case_level_causal_verdict=True))
    write('case_evidence.json',dict(status='EXPLORATORY_REVIEW_TAGS_NOT_CAUSAL_CLASSIFICATION',cases=tags))
    write('analysis.json',dict(status='COMPLETE_DIAGNOSTIC_ANALYSIS',formal_trial=False,major_claim=False,
            readout_rows=rows,primary_contrasts=contrasts,candidate_rows=candidate['rows'],metadata=metadata,
            human_annotations_status='NOT_COLLECTED',
            recommendations='Review raw-versus-representation readout, risk-versus-type/onset/direction and candidate-versus-ranking separately. No automatic decoder change authorized by diagnostic score differences.'))
    lines=['# F16可观测运动变化线索诊断','',
        '这是诊断，不是新的候选生成实验，也不是新的 baseline。原 F16、解码器和 scorer 全冻结；只训练独立的分析探针。',
        '', '## 分组与证据边界','',
        '按原 ETH-training 数据来源恢复序列/人物/时间，使用最粗 scene-family 整组留出。Zara录制合并一组，students/uni合并一组，hotel一组。',
        '仅三个场景组，不报告可靠群组CI、Holm显著性或形式置换p值。原 F16 已见过原训练集；本结果不是独立原模型泛化证据。CAL保持原拆分，仅描述；歧义元数据和末端不满足持续性标签的样例另列。',
        '', '## FIT场景整组留出的变化风险读出','',
        '| reader | n | prevalence | AUPRC | Brier | log loss |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        ap='NA'if row['AUPRC']is None else f"{row['AUPRC']:.4f}"
        lines.append(f"| {row['reader']} | {row['n']} | {row['prevalence']:.4f} | {ap} | {row['Brier']:.4f} | {row['log_loss']:.4f} |")
    lines+=['','## 三个主要探索性对照','', '| contrast | dimension control | ΔAUPRC | ΔBrier |','|---|---|---:|---:|']
    for row in contrasts:lines.append(f"| {row['contrast']} | {row['mode']} | {row['AUPRC_delta']} | {row['Brier_delta']:.5f} |")
    lines+=['','风险读出较好不代表能准确预测发生时刻、转向方向或完整路径。探针可读出也不等于原解码器正在有效使用；功能响应只说明依赖，不证明有益利用。弱读出不能证明线索不存在或未来不可预测。',
        '', '## 候选与原始筛选','', '| group | n | minADE | minFDE | selectedADE | selectedFDE | SAMEPATH .2/.4 rate |','|---|---:|---:|---:|---:|---:|---:|']
    for row in candidate['rows']:
        lines.append(f"| {row['group']} | {row['n']} | {row['mean_oracle_ADE']:.4f} | {row['mean_oracle_FDE']:.4f} | {row['mean_selected_ADE']:.4f} | {row['mean_selected_FDE']:.4f} | {row['SAMEPATH_20_40_rate']:.4f} |")
    lines+=['','## 人工材料与限制','',
        '`human_blind/past_only_cases.json`只包含观察历史。请先记录变化概率、事件、时间及可见线索，再打开reveal文件；120例是否真正未看过GT需你声明。没有生成任何人工答案。',
        '新数据、检查点、失败记录和报告均在当前文件夹；旧文件SHA在开始与结束核验。未来GT仅标签/离线打分，不进特征或原模型前向。小幅邻居干预不保证每例分布内，禁止对干预轨迹相对原GT解释因果收益。',
        '原预处理的邻居/样本资格要求一定未来存在性。本轮没有用未来重新选择邻居，也没有传入未来坐标；但已有资格选择偏差限制“部署时可利用线索”的解释，需要独立past-only资格审计才能进一步断言。',
        '人工盲评尚未完成，群组数很少、单个GT、原模型训练历史、probe容量与优化差异均限制结论。此次没有重大改进声明；下一次生成训练必须依据诊断与人类复核另行设计。',
        '', 'AI辅助说明：代码与报告由Codex协助实现；不存在虚构外部审稿或人工标注。']
    (HERE/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    started=time.time();pre=read('preflight.json')
    assert pre['status']=='PASS'
    assert not (HERE/'RESULT.json').exists(), 'Diagnosis already completed; do not rerun'
    stage('SOURCE_AND_OLD_RESULT_HASH_VERIFY')
    verify(pre['source_hashes'])
    old=read('old_results_manifest.json')['files'];verify(old)
    torch.use_deterministic_algorithms(True)
    meta=population.metadata()
    stage('OFFLINE_EVENT_LABELS')
    train_labels=population.labels('train');population.labels('dev')
    population.candidate_audit();population.blind_materials()
    model=native.build()
    original={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    features=native.extract_population('train',model);native.extract_population('dev',model)
    _,_,obs=load_population('train')
    take=np.random.default_rng(SEED).choice(np.arange(FIT_N),128,replace=False)
    stage('FUNCTIONAL_SENSITIVITY')
    write('functional_sensitivity.json',native.functional(model,obs[take],features['sample_id'][take]))
    assert all(torch.equal(v.detach().cpu(),original[k])for k,v in model.state_dict().items())
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    write('frozen_native_audit.json',dict(status='PASS_ALL_ORIGINAL_PARAMETERS_AND_BUFFERS_EXACT',parameters=len(list(model.parameters())),
            input_jacobian_not_parameter_training=True))
    del model;torch.cuda.empty_cache()
    base,groups=probes.nested_binary(features,train_labels,meta)
    stage('SECONDARY_TYPE_ONSET_DIRECTION_AND_NEGATIVE_CONTROLS')
    probes.secondary_and_controls(features,train_labels,base,groups)
    stage('INDEPENDENT_STORED_PREDICTION_METRIC_AUDIT')
    n=probes.independent_prediction_audit();report()
    stage('FINAL_SOURCE_AND_ALL_OLD_SHA_VERIFY')
    verify(pre['source_hashes']);verify(old)
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*')if p.is_file()and p.name not in ('progress.json','stdout.log','stderr.log','independent_worker_state.json','runtime_state.json','independent_worker.lock') and not p.name.endswith('.tmp')}
    write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY',formal_trial=False,major=False,seconds=time.time()-started,
        completed_reader_jobs=n,all_original_parameters_frozen=True,source_hashes_unchanged=True,
        old_files_unchanged=len(old),all_1516_native_finite_K20=True,human_annotations_collected=False,
        independent_scene_families=read('binary_probes.json')['scene_families'],formal_group_CI_withheld=True,
        artifact_sha256=files,limitations=['Only three scene families','Original F16 previously trained on these observations',
          'CAL/DEV repeatedly explored; no hyperparameter selection from them','Readout/sensitivity are not causal beneficial utilization','Human blind evaluation pending']))
    stage('SUCCESS_DIAGNOSTIC_COMPLETED',reader_jobs=n,seconds=time.time()-started)

if __name__=='__main__':main()
