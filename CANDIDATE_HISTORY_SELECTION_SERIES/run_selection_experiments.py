"""Prepare / smoke / pilot / report. No test data or generator training path."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from selection_core import (INPUT_KEYS, Normalizer, Selector, calibrate, features,
                            history_for, history_tokens, metrics, set_seed, split_ids, utility_target)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_cache(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def validate(data):
    required = (*INPUT_KEYS, 'sample_id', 'future_relative', 'ADE_final', 'FDE_final',
                'native_selected_index', 'oracle_final_index')
    n = len(data['sample_id'])
    if n == 0 or len(np.unique(data['sample_id'])) != n:
        raise ValueError('Empty cache or duplicate IDs')
    for key in required:
        if len(data[key]) != n or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid field {key}')
    ade, fde = data['ADE_final'], data['FDE_final']
    if ade.shape != data['native_logits'].shape or fde.shape != ade.shape:
        raise ValueError('Candidate/logit/metric shapes disagree')
    errors = np.linalg.norm(data['candidate_final'] - data['future_relative'][:, None], axis=-1)
    if not np.allclose(errors.mean(-1), ade, atol=2e-5, rtol=2e-5):
        raise ValueError('ADE labels fail independent recomputation')
    if not np.allclose(errors[..., -1], fde, atol=2e-5, rtol=2e-5):
        raise ValueError('FDE labels fail independent recomputation')
    if not np.array_equal(ade.argmin(1), data['oracle_final_index']):
        raise ValueError('Oracle alignment failure')
    if not np.array_equal(data['native_logits'].argmax(1), data['native_selected_index']):
        raise ValueError('Native alignment failure')
    if not np.allclose(data['past_relative'][:, -1], 0, atol=2e-5):
        raise ValueError('Expected focal-last relative coordinates')


def prepare(config, config_path, out):
    paths = {k: (ROOT / config[k]).resolve() for k in
             ('train_cache', 'development_cache', 'generator_checkpoint', 'generator_identity', 'generator_split')}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    train, dev = load_cache(paths['train_cache']), load_cache(paths['development_cache'])
    validate(train)
    validate(dev)
    if np.intersect1d(train['sample_id'], dev['sample_id']).size:
        raise ValueError('Train/development sample IDs overlap')
    with np.load(paths['generator_split'], allow_pickle=False) as split:
        if not np.array_equal(np.sort(train['sample_id']), np.sort(split['train'])):
            raise ValueError('Training IDs differ from source generator split')
        if not np.array_equal(np.sort(dev['sample_id']), np.sort(split['validation'])):
            raise ValueError('Development IDs differ from source generator split')
    fit, cal = split_ids(train['sample_id'], config['calibration_fraction'], config['split_seed'])
    base, past, _ = features(train)
    manifest = dict(protocol=config['protocol'], config_sha256=sha(config_path),
                    sources={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
                    code_sha256={p.name: sha(p) for p in (HERE / 'selection_core.py', Path(__file__))},
                    base_dim=int(base.shape[-1]), history_dim=int(history_tokens(past).shape[-1]),
                    n_fit=len(fit), n_calibration=len(cal), n_development=len(dev['sample_id']),
                    development_native=metrics(dev['ADE_final'], dev['FDE_final'], dev['native_selected_index'], dev['native_selected_index']),
                    scope='ETH generator seed 1 development pilot; not five-fold/test confirmation',
                    independence='ID-disjoint only; no scene/track group metadata, so no inferential bootstrap',
                    lineage='Existing cache content validated and source files pinned; live generator equality inherited, not re-executed',
                    test_accessed=False)
    lock_path = out / 'preflight.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Protocol/source/code drift: choose a NEW output directory; no overwrite allowed')
    out.mkdir(parents=True, exist_ok=True)
    write_json(lock_path, manifest)
    np.savez_compressed(out / 'split_ids.npz', fit=train['sample_id'][fit], calibration=train['sample_id'][cal],
                        development=dev['sample_id'])
    print(f"PREPARE PASS: fit={len(fit)}, calibration={len(cal)}, development={len(dev['sample_id'])}", flush=True)
    return train, dev, fit, cal, manifest


def subset(data, indices):
    return {k: v[indices] for k, v in data.items()}


def predict(model, base, hist, device, batch=512):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(base), batch):
            x = torch.from_numpy(base[start:start + batch]).to(device)
            h = torch.from_numpy(hist[start:start + batch]).to(device)
            result.append(model(x, h).cpu().numpy())
    return np.concatenate(result)


def fit_one(config, variant, seed, fit_data, cal_data, dev, manifest, run_dir, device, smoke):
    run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    fit_b, fit_p, fit_cv = features(fit_data)
    cal_b, cal_p, cal_cv = features(cal_data)
    dev_b, dev_p, dev_cv = features(dev)
    # SAME fit-only normalization for real/CV arms: no treatment-dependent rescaling.
    norm = Normalizer(fit_b, history_tokens(fit_p))
    xb, hb = norm.apply(fit_b, history_for(fit_p, fit_cv, variant))
    xc, hc = norm.apply(cal_b, history_for(cal_p, cal_cv, variant))
    xd, hd = norm.apply(dev_b, history_for(dev_p, dev_cv, variant))
    model = Selector(xb.shape[-1], hb.shape[-1], variant, config['hidden'], config['history_hidden']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    labels = torch.from_numpy(fit_data['ADE_final'].astype(np.float32))
    targets = utility_target(labels)
    loader = DataLoader(TensorDataset(torch.from_numpy(xb), torch.from_numpy(hb), targets),
                        batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    native_fit = fit_data['native_logits']
    native_scale = float((native_fit - native_fit.mean(1, keepdims=True)).std())
    history, best, start_time = [], None, time.perf_counter()
    for epoch in range(1, (1 if smoke else config['epochs']) + 1):
        model.train()
        loss_sum = 0.0
        for x, h, target in loader:
            x, h, target = x.to(device), h.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x, h), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(x)
        fit_u = predict(model, xb, hb, device)
        scale = native_scale / max(float(fit_u.std()), 1e-6)
        cal_u = predict(model, xc, hc, device)
        choice = calibrate(cal_data['native_logits'], cal_u, cal_data['ADE_final'], scale, config['alpha_grid'])
        history.append(dict(epoch=epoch, loss=loss_sum / len(xb), calibration_ade=choice['ade'],
                            alpha=choice['alpha'], scale=scale))
        if best is None or choice['ade'] < best['calibration_ade'] - 1e-10:
            best = dict(epoch=epoch, calibration_ade=choice['ade'], alpha=choice['alpha'], scale=scale,
                        weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f'{variant} seed={seed} epoch={epoch}: calibration ADE={choice["ade"]:.5f}, alpha={choice["alpha"]}', flush=True)
    model.load_state_dict(best['weights'])
    checkpoint = dict(weights=best['weights'], normalizer={k: torch.from_numpy(v) for k, v in vars(norm).items()},
                      variant=variant, base_dim=xb.shape[-1], hist_dim=hb.shape[-1], hidden=config['hidden'],
                      history_hidden=config['history_hidden'], alpha=best['alpha'], scale=best['scale'],
                      seed=seed, source_manifest=manifest, config=config)
    torch.save(checkpoint, run_dir / 'selector.pth')
    reloaded = torch.load(run_dir / 'selector.pth', map_location='cpu', weights_only=True)
    check_model = Selector(xb.shape[-1], hb.shape[-1], variant, config['hidden'], config['history_hidden']).to(device)
    check_model.load_state_dict(reloaded['weights'])
    utility = predict(model, xd, hd, device)
    reloaded_u = predict(check_model, xd, hd, device)
    if not np.allclose(utility, reloaded_u, atol=1e-6, rtol=1e-6):
        raise RuntimeError('Checkpoint reload output mismatch')
    scores = dev['native_logits'] + best['alpha'] * best['scale'] * utility
    indices = scores.argmax(1)
    result = metrics(dev['ADE_final'], dev['FDE_final'], indices, dev['native_selected_index'])
    result.update(variant=variant, ranker_seed=seed, parameters=n_params,
                  best_epoch=best['epoch'], alpha=best['alpha'], calibration_ade=best['calibration_ade'],
                  training_seconds=time.perf_counter() - start_time, smoke_only=smoke,
                  checkpoint_sha256=sha(run_dir / 'selector.pth'))
    result['pure_utility_ade'] = metrics(dev['ADE_final'], dev['FDE_final'], utility.argmax(1), dev['native_selected_index'])['selected_ade']
    np.savez_compressed(run_dir / 'predictions.npz', sample_id=dev['sample_id'], scores=scores,
                        selected_index=indices, selected_ade=dev['ADE_final'][np.arange(len(indices)), indices])
    if variant == 'real_history':
        # Fixed diagnostics, never used for epoch/alpha selection or automatic configuration search.
        ablations = []
        for kind in ('cv_history', 'shuffled_residual'):
            permutations = range(3) if kind == 'shuffled_residual' else range(1)
            for permutation_seed in permutations:
                permutation = np.random.default_rng(9000 + permutation_seed).permutation(len(dev_p))
                ax, ah = norm.apply(dev_b, history_for(dev_p, dev_cv, kind, permutation))
                au = predict(model, ax, ah, device)
                asi = (dev['native_logits'] + best['alpha'] * best['scale'] * au).argmax(1)
                am = metrics(dev['ADE_final'], dev['FDE_final'], asi, dev['native_selected_index'])
                am.update(kind=kind, permutation_seed=permutation_seed,
                          delta_ade_vs_intact=am['selected_ade'] - result['selected_ade'])
                ablations.append(am)
        write_json(run_dir / 'history_ablation.json', ablations)
    write_json(run_dir / 'history.json', history)
    write_json(run_dir / 'metrics.json', result)
    print(f"DONE {variant} seed={seed}: development ADE={result['selected_ade']:.5f}; params={n_params}", flush=True)
    return result


def report(out, config, smoke=False):
    manifest = json.loads((out / 'preflight.json').read_text(encoding='utf-8'))
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('*/metrics.json'))]
    expected = {(v, s) for v in config['variants'] for s in ([config['ranker_seeds'][0]] if smoke else config['ranker_seeds'])}
    actual = {(r['variant'], r['ranker_seed']) for r in results}
    if actual != expected:
        raise RuntimeError(f'Incomplete or unexpected runs: expected={expected}, actual={actual}')
    native = json.loads((out / 'native.json').read_text(encoding='utf-8'))
    for r in results:
        if abs(r['minade'] - native['minade']) > 1e-6:
            raise RuntimeError('Frozen candidate oracle drift')
    for seed in {r['ranker_seed'] for r in results}:
        counts = [r['parameters'] for r in results if r['ranker_seed'] == seed and r['variant'] in ('cv_history', 'real_history')]
        if len(set(counts)) != 1:
            raise RuntimeError('Matched architecture parameter counts disagree')
    cols = ['variant', 'ranker_seed', 'selected_ade', 'selected_fde', 'regret', 'near_oracle_05',
            'regret_p90', 'harmful_rate', 'useful_rate', 'parameters', 'alpha', 'best_epoch']
    with (out / 'leaderboard.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    means = {v: {k: statistics.mean(r[k] for r in results if r['variant'] == v)
                 for k in ('selected_ade', 'selected_fde', 'regret', 'harmful_rate')} for v in config['variants']}
    real, control = means['real_history'], means['cv_history']
    relative = (control['selected_ade'] - real['selected_ade']) / control['selected_ade']
    positives = sum(next(r for r in results if r['variant'] == 'real_history' and r['ranker_seed'] == s)['selected_ade'] <
                    next(r for r in results if r['variant'] == 'cv_history' and r['ranker_seed'] == s)['selected_ade']
                    for s in {r['ranker_seed'] for r in results})
    checks = dict(relative_ade_gain_pass=relative >= config['minimum_relative_gain'],
                  fde_guard_pass=real['selected_fde'] <= control['selected_fde'] * (1 + config['maximum_relative_fde_increase']),
                  seed_direction_pass=positives >= config['minimum_positive_seeds'],
                  native_guard_pass=real['selected_ade'] < native['selected_ade'],
                  linear_guard_pass=real['selected_ade'] <= means['linear']['selected_ade'],
                  harmful_rate_guard_pass=real['harmful_rate'] <= control['harmful_rate'])
    verdict = 'SMOKE_ONLY_NO_SCIENTIFIC_VERDICT' if smoke else ('PROMISING_REQUIRES_REVIEW' if all(checks.values()) else 'NO_CLEAR_HISTORY_GAIN_REVIEW')
    write_json(out / 'summary.json', dict(means=means, native=native, checks=checks,
                                        relative_ade_gain_vs_cv=relative, positive_seeds=positives, verdict=verdict))
    lines = ['# 候选轨迹选择试点结果', '', f'状态：{verdict}', '',
             '仅 ETH 生成器种子下的 development 诊断；不是五折 Macro，也不是独立测试。',
             'CV 与真实历史组具有相同架构、参数量、目标、预算、候选及拟合集归一化；只有历史内容不同。',
             '校准集来自原生成器训练缓存的显式 ID 划分；不具有场景独立性。历史验证集已被此前实验使用。',
             '没有独立分组元数据，不计算样本独立假设下的显著性区间。', '',
             '| 方法 | 种子 | ADE | FDE | Regret | 参数 | alpha |', '|---|---|---|---|---|---|---|',
             f"| Native | generator | {native['selected_ade']:.6f} | {native['selected_fde']:.6f} | {native['regret']:.6f} | 0 | — |"]
    for r in results:
        lines.append(f"| {r['variant']} | {r['ranker_seed']} | {r['selected_ade']:.6f} | {r['selected_fde']:.6f} | {r['regret']:.6f} | {r['parameters']} | {r['alpha']} |")
    lines += ['', f'真实历史相对 CV 对照 ADE 改善：{relative:.2%}；正向种子数：{positives}。',
              '这只是预设资源筛选门槛，不是显著性检验；历史打乱若变差仅说明模型对该输入敏感，不能单独证明增量信息或因果机制。',
              '若通过：先人工审阅困难样本、误干预和校准稳定性，再设计邻居观察条件或跨生成器确认；脚本不自动开展下一阶段。',
              '若未通过：区分校准 alpha 归零、预测 utility 失效、历史分支未学会、或真实历史无稳定增益；不能写成不可恢复的理论极限。']
    (out / 'RESULT_ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'state.json', dict(status='STOP_POINT_1_REVIEW', verdict=verdict, test_accessed=False))
    print(f'REPORT: {out / "RESULT_ANALYSIS.md"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'smoke', 'pilot', 'report'], required=True)
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    smoke = args.stage == 'smoke'
    out = (args.output_dir or HERE / ('smoke_results' if smoke else 'results')).resolve()
    if args.stage == 'report':
        locked = json.loads((out / 'preflight.json').read_text(encoding='utf-8'))
        if locked['config_sha256'] != sha(args.config):
            raise RuntimeError('Report config differs from locked run; restore the exact config')
        report(out, config)
        return
    train, dev, fit, cal, manifest = prepare(config, args.config, out)
    if args.stage == 'prepare':
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    if smoke:
        fit, cal = fit[:128], cal[:64]
        order = np.random.default_rng(config['split_seed']).permutation(len(dev['sample_id']))[:64]
        dev = subset(dev, order)
    write_json(out / 'native.json', metrics(dev['ADE_final'], dev['FDE_final'], dev['native_selected_index'], dev['native_selected_index']))
    write_json(out / 'state.json', dict(status='RUNNING', smoke_only=smoke, test_accessed=False))
    for seed in ([config['ranker_seeds'][0]] if smoke else config['ranker_seeds']):
        for variant in config['variants']:
            run_dir = out / f'{variant}_seed{seed}'
            if run_dir.exists():
                raise RuntimeError(f'Existing run, refusing overwrite: {run_dir}. Choose a new output directory.')
            fit_one(config, variant, seed, subset(train, fit), subset(train, cal), dev, manifest, run_dir, device, smoke)
    report(out, config, smoke)


if __name__ == '__main__':
    main()
