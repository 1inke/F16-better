"""Independent read-only production audit, then NEW three-experiment review files."""
from pathlib import Path
import importlib.util,sys,json,csv
import numpy as np,torch
HERE=Path(__file__).resolve().parent
def main():
    spec=importlib.util.spec_from_file_location('component_independent_audit',HERE/'run_head.py')
    a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
    a.a.prepare_backend(torch.device('cpu'))
    for path,digest in json.loads((HERE/'preflight.json').read_text())['source_hashes'].items():assert a.p.sha256(Path(path))==digest,path
    locked=json.loads((HERE/'results/model_lock.json').read_text())['models'];assert len(locked)==18
    for path,digest in locked.items():
        assert a.p.sha256(Path(path))==digest,path
        saved=torch.load(path,map_location='cpu',weights_only=False);model=a.SubspaceHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);a.frozen_audit(model)
    cache=a.p.load_npz(a.p.DEV_CACHE)
    values=a.r.prepared(cache,False,mean=np.zeros((1,940),np.float32),std=np.ones((1,940),np.float32));base,speed,gt=values[1:4]
    for seed in (1,2,3):
        for arm in a.c.ARMS:
            for tag in a.c.POLICIES:
                saved=a.n.RAW_LOAD(HERE/'results'/f'seed{seed}_{arm}_{tag}.npz')
                assert np.array_equal(saved['sample_id'],cache['sample_id'])
                assert saved['trajectory'].shape==(1516,20,12,2) and np.isfinite(saved['trajectory']).all()
                for key,value in a.r.generation_metrics(saved['trajectory'],base,speed,gt).items():np.testing.assert_allclose(saved[key],value,rtol=0,atol=1e-6)
    old_count=a.check_old()
    audit=dict(status='PASS',models_sha_verified=18,source_hashes_verified=True,frozen_whitelist_verified=True,explicit_prediction_ids_verified=18,finite_N_K_T_XY=[1516,20,12,2],all_stored_metrics_independently_recomputed=True,old_results_unchanged=old_count)
    a.m.write_json(HERE/'independent_audit.json',audit)
    experiments=[]
    for folder,relative,label,arms in [('GENERATION_UPSTREAM_TEMPORAL_SERIES','evaluation_repair/results','full temporal vs frozen head-only',{'zero':'frozen F16/head','mean_std':'full temporal/head'}),('GENERATION_TEMPORAL_SUBSPACE_SERIES','results','attention vs full temporal',{'zero':'full temporal/head','mean_std':'attention/head'}),('GENERATION_TEMPORAL_COMPONENT_SERIES','results','FFN vs attention',{'zero':'attention/head','mean_std':'FFN/head'})]:
        root=HERE.parent/folder;path=root/relative/'summary.json';summary=json.loads(path.read_text());histories=[]
        for arm in ('zero','mean_std'):
            for seed in (1,2,3):
                modelroot=root/'results/models'/arm/f'seed_{seed}'/'balanced_protected'
                completion=json.loads((modelroot/'complete.json').read_text());rows=list(csv.DictReader((modelroot/'history.csv').open(encoding='utf-8-sig')))
                assert completion['epochs']==80 and len(rows)==80
                histories.append(dict(arm=arm,seed=seed,selected_epoch=completion['best_feasible_epoch'],first_fit_loss=rows[0]['train_loss'],last_fit_loss=rows[-1]['train_loss'],last_cal_feasible=rows[-1]['cal_feasible'],last_cal_change_key=rows[-1]['cal_change_key']))
        experiments.append(dict(experiment=folder,comparison=label,arm_meanings=arms,summary_source=str(path),summary_source_sha=a.p.sha256(path),all_policy_group_metrics_and_CIs=summary,training_calibration_history=histories))
    review=dict(status='COMPLETE_REVIEWED',cycle=6,experiments=experiments,fresh_trainings=18,epochs_each=80,policy_entries=54,major_results=0,replication_warning='Full-arm repeats and attention-arm repeats share effective architecture/init/order/seeds. Fresh execution and 54 policies are not54 independent trainings or generalization replicates. Repeated ETHdev exploratory.',source_manifest=audit)
    a.m.write_json(HERE/'CYCLE_6_REVIEW.json',review)
    lines=['# Cycle6 总复盘：有限解冻没有可靠突破','','18次fresh80轮训练、54个policy条目，不是54次独立训练。全部正常完成独立审计；历史失败仅首次评估rawdev路径，补评估证据完整保留。所有旧结果不变。','','| 实验 | treatment changeADE/FDE | 相对native收益 | 失败门槛 |','|---|---|---|---|']
    for exp in experiments:
        summary=exp['all_policy_group_metrics_and_CIs'];row=next(x for x in summary['aggregate'] if x['arm']=='mean_std' and x['policy']=='constrained' and x['group']=='change');native=summary['native']['change']
        gains=[100*(1-row[k]/native[k]) for k in ('minade20','minfde20')]
        lines.append('| '+exp['comparison']+' | '+f"{row['minade20']:.6f}/{row['minfde20']:.6f}"+' | '+f'{gains[0]:.2f}%/{gains[1]:.2f}%'+' | '+', '.join(k for k,v in summary['gates'].items() if not v)+' |')
    lines+=['','## 结论与对照限制','','完整时间层、小容量注意力、同预算FFN均未满足change双5%/保护组1%/三seed一致/native及freshcontrolCI/覆盖同路径运动学要求。FFN有名义变化改善但CI跨零、保护组和同路径失败；不能挑最好组或policy宣布突破。所有原始分组、policy、CI、校准history及source SHA完整嵌入JSON，不只报告change。','','完整时间层与下一轮fullcontrol、attention-only与第三轮attentioncontrol有相同有效架构/种子/初始化/顺序；重复执行不是独立泛化。只有3seed且反复同ETHdev评估，有样本/场景相关性与适应性探索风险。本轮容量耦合、参数范数分母不同，不能把梯度可达性或大漂移当有用意图/主因证明。','','训练FITloss下降但后期CAL保护常失败，支持优化-保护冲突/泛化受限的现象，不证明唯一过拟合原因。所有方法还保留nativeK20基底、原冻结decoder上下文读取、3anchor纠偏和common velocity clip；改编码器并没有测试完全独立候选生成。scorer固定但输入可变，本轮只oracle候选指标，不能主张实际筛选提升。','','## 大方向调整','','停止再拆分子模块、gate、平滑、系数或epoch局部试探。下一步先在FIT/CAL做固定可达性审计，区分候选纠偏函数类/速度clip的表达限制和实际学习失败：计算逐步box内GT精确可达率、端点必要下界、3anchor GT诊断构造的误差与clip饱和，不以构造当学得结果或最优上界。GT仅离线训练诊断不进入部署forward、不读dev选参数、不放宽guard。该诊断决定是否重新设计新生成头参数化；目前不启动第4局部训练。若需更广原decoder/F16解冻或独立泛化测试，应具体向人类请求新权限，不擅自扩展。','']
    (HERE/'CYCLE_6_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print('INDEPENDENT AUDIT PASS '+json.dumps(audit),flush=True)
    print('CYCLE6 FULL REVIEW SAVED / 3 experiments /18freshtrainings /54policies / no major',flush=True)
if __name__=='__main__':main()
