"""Audit completed pilot only: no training, tuning, or test-set access."""
from pathlib import Path
import hashlib
import json
import statistics as st
import numpy as np
from selection_core import metrics

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'review'
RESULTS = HERE / 'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = read(RESULTS / 'preflight.json')
    assert sha(HERE / 'config.json') == manifest['config_sha256'], 'Config drift'
    for name, digest in manifest['code_sha256'].items():
        assert sha(HERE / name) == digest, 'Code drift: ' + name
    for name, source in manifest['sources'].items():
        assert sha(source['path']) == source['sha256'], 'Source drift: ' + name
    config = read(HERE / 'config.json')
    with np.load(manifest['sources']['development_cache']['path']) as f:
        ade, fde, ids, native_idx = (f[k] for k in ('ADE_final', 'FDE_final', 'sample_id', 'native_selected_index'))
    native = metrics(ade, fde, native_idx, native_idx)
    records, preds, ablation = [], {}, []
    for variant in config['variants']:
        for seed in config['ranker_seeds']:
            path = RESULTS / f'{variant}_seed{seed}'
            r = read(path / 'metrics.json')
            assert not r['smoke_only']
            assert sha(path / 'selector.pth') == r['checkpoint_sha256']
            with np.load(path / 'predictions.npz') as p:
                assert np.array_equal(ids, p['sample_id']), 'Sample alignment failure'
                idx = p['scores'].argmax(1)
                assert np.array_equal(idx, p['selected_index'])
                assert np.allclose(ade[np.arange(len(ids)), idx], p['selected_ade'])
                preds[(variant, seed)] = idx.copy()
            recomputed = metrics(ade, fde, idx, native_idx)
            for key, value in recomputed.items():
                if value is None:
                    assert r[key] is None
                else:
                    assert np.isclose(value, r[key], atol=1e-7), (variant, seed, key)
            h = read(path / 'history.json')
            assert len(h) == config['epochs']
            chosen = h[r['best_epoch'] - 1]
            assert np.isclose(chosen['calibration_ade'], min(x['calibration_ade'] for x in h))
            assert chosen['alpha'] == r['alpha']
            r['loss_first'], r['loss_last'] = h[0]['loss'], h[-1]['loss']
            r['low_regret_improvement_m'] = float((ade[np.arange(len(ids)), native_idx] - ade[np.arange(len(ids)), idx])[
                ade[np.arange(len(ids)), native_idx] - ade.min(1) <= .2].mean())
            records.append(r)
            if variant == 'real_history':
                a = read(path / 'history_ablation.json')
                ablation.append(dict(seed=seed, cv_replacement_delta_mm=next(x['delta_ade_vs_intact'] for x in a if x['kind']=='cv_history')*1000,
                                     shuffle_mean_delta_mm=st.mean(x['delta_ade_vs_intact'] for x in a if x['kind']=='shuffled_residual')*1000))
    means = {v: {key: st.mean(r[key] for r in records if r['variant']==v) for key in
                  ('selected_ade','selected_fde','regret','harmful_rate','mean_harm_m','mean_benefit_m',
                   'high_regret_improvement_m','low_regret_improvement_m','pure_utility_ade')} for v in config['variants']}
    summary = read(RESULTS / 'summary.json')
    for v, vals in summary['means'].items():
        for key, value in vals.items():
            assert np.isclose(value, means[v][key], atol=1e-7), 'Summary mismatch'
    differences = []
    for seed in config['ranker_seeds']:
        a, b = preds[('cv_history',seed)], preds[('real_history',seed)]
        gain = ade[np.arange(len(ids)), a] - ade[np.arange(len(ids)), b]
        differences.append(dict(seed=seed, real_minus_cv_ade_mm=-float(gain.mean())*1000,
                                changed_selection_n=int((a!=b).sum()), better_n=int((gain>1e-6).sum()),
                                worse_n=int((gain < -1e-6).sum())))
    with np.load(manifest['sources']['train_cache']['path']) as f:
        train_ids, train_ade, train_logits = (f[k] for k in ('sample_id', 'ADE_final', 'native_logits'))
    with np.load(RESULTS / 'split_ids.npz') as split:
        cal_mask = np.isin(train_ids, split['calibration'])
    cal_native = float(train_ade[np.arange(len(train_ade)), train_logits.argmax(1)][cal_mask].mean())
    result = dict(audit='PASS', means=means, native=native, paired_seed_differences=differences,
                  ablations=ablation, calibration_native_ade=cal_native, inference_scope=manifest['scope'],
                  new_training_executed=False, test_accessed=False)
    OUT.mkdir(exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 完成实验复核与后续判断', '',
             '结论：本轮未发现真实历史相对匹配恒速对照的稳定、实际有意义的选模增益。暂停扩展历史分支或直接增加邻居模块；先处理校准和检查点选择的稳定性。', '',
             '核验通过：配置、核心代码、缓存、源划分、生成器及评分器权重哈希一致；全部训练完成；逐样本ID、argmax、ADE/FDE及汇总重算一致。未进行新训练或访问test。', '',
             '| 组别 | 平均ADE | 平均FDE | 相对Native改善 | Regret闭合 |', '|---|---|---|---|---|',
             f"| Native | {native['selected_ade']:.6f} | {native['selected_fde']:.6f} | — | — |"]
    for v, m in means.items():
        gain = native['selected_ade'] - m['selected_ade']
        lines.append(f"| {v} | {m['selected_ade']:.6f} | {m['selected_fde']:.6f} | {gain/native['selected_ade']:.2%} | {gain/native['regret']:.2%} |")
    lines += ['', '## 匹配种子与历史消融', '', '| 种子 | H1−H0 ADE (mm，负数更好) | 换成CV后ΔADE (mm) | 打乱残差平均ΔADE (mm) |', '|---|---|---|---|']
    for d, a in zip(differences, ablation):
        lines.append(f"| {d['seed']} | {d['real_minus_cv_ade_mm']:+.3f} | {a['cv_replacement_delta_mm']:+.3f} | {a['shuffle_mean_delta_mm']:+.3f} |")
    lines += ['', '部分种子在匹配比较与消融上有小幅历史信号，但不足以通过预设门槛。不能将无明显收益写成历史信息无用，也不能剔除不利种子。', '',
              '## 获益被哪些损害抵消', '', '| 组别 | 全体平均获益 (mm) | 全体平均损害 (mm) | 大regret子集净获益 (mm) | 其余样本净获益 (mm) |', '|---|---|---|---|---|']
    for v,m in means.items():
        lines.append(f"| {v} | {m['mean_benefit_m']*1000:.3f} | {m['mean_harm_m']*1000:.3f} | {m['high_regret_improvement_m']*1000:.3f} | {m['low_regret_improvement_m']*1000:.3f} |")
    lines += ['', '上述是按Native regret定义的事后诊断，不是部署时可直接知道的分组。说明存在误干预抵消现象，不证明一个可学习gate必定能识别这些样本。', '',
              '## 校准和优化线索', '',
              f"校准集Native ADE={cal_native:.6f}，development Native ADE={native['selected_ade']:.6f}。分布不同且生成器对两者的训练暴露不同，不能把校准收益直接视为泛化保证。", '',
              '| 组别 | 种子 | 最佳epoch | alpha | 校准ADE | development ADE | 起始MSE | 最终MSE |', '|---|---|---|---|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['variant']} | {r['ranker_seed']} | {r['best_epoch']} | {r['alpha']} | {r['calibration_ade']:.6f} | {r['selected_ade']:.6f} | {r['loss_first']:.6f} | {r['loss_last']:.6f} |")
    lines += ['', '历史两组的融合系数均达到预设网格边界；不存在alpha归零掩盖历史效果的情况，但网格尚未包围校准最优点。此现象不能证明扩大alpha就会改善development。',
              '晚期回归MSE改善而校准选模表现不一定改善；评分学习目标与离散Top-1选择存在不一致的线索。线性MSE模型对种子和早期检查点敏感，也提示应先建立确定性线性参照。', '',
              '## 建议的下一轮（尚未执行）', '',
              '优先进行一次选择协议与稳定性实验，而不是增加特征：',
              '1. 固定当前候选和输入，先用确定性Ridge/闭式线性解对照现有AdamW线性头。正则强度只用校准集选择。这是新基线，不能冒称旧Tiny复现。',
              '2. 对已有检查点做校准集专属的alpha曲线，包含纯utility端点；在查看development前锁定选择规则。已有development被观察过，新结果只能继续作为探索。',
              '3. 下一次训练预先保存校准所需的逐epoch输出；比较固定早期预算与校准早停的稳定性，不能回头用development选最佳epoch或种子。',
              '4. 若需要正式确认，优先补场景/轨迹分组元数据并建立整条生成器隔离的验证流程；当前ID随机划分不能代替场景独立验证。',
              '只有这个阶段显示稳定选模收益，才决定是否回到历史条件或研究以预测干预净收益为目标的残差选择。此前风险gate结果不能当作成功保证。', '',
              '本轮是固定ETH生成器、多个评分器种子的development结果；不进行样本独立bootstrap，不声称统计显著性或五折Macro结论。']
    (OUT / '完成实验复核.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
